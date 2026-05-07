import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput

try:
    from core.conscious_gaze.cds import detect_cognitive_demand, compute_logits_entropy, get_token_collector
    from core.conscious_gaze.fci import get_hii_collector
    CONSCIOUS_GAZE_CDS_AVAILABLE = True
except ImportError:
    CONSCIOUS_GAZE_CDS_AVAILABLE = False


var_k=1.0
def shapley_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id


    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # Initialize optional output caches
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    # auto-regressive generation
    cg_inputs_embeds = getattr(self, "cg_inputs_embeds", None)
    base_inputs_embeds = model_kwargs.get("inputs_embeds")

    if cg_inputs_embeds is not None:
        nomask_embeds = cg_inputs_embeds.get("nomask", base_inputs_embeds)
        maskimage_embeds = cg_inputs_embeds.get("maskimage_shapley", nomask_embeds)
        masktext_embeds = cg_inputs_embeds.get("masktext", nomask_embeds)
        maskall_embeds = cg_inputs_embeds.get("maskall", nomask_embeds)
    else:
        nomask_embeds = base_inputs_embeds
        maskimage_embeds = base_inputs_embeds
        masktext_embeds = base_inputs_embeds
        maskall_embeds = base_inputs_embeds

    def _clone_kwargs_with_inputs(source_kwargs: Dict[str, Any], new_inputs):
        cloned_kwargs = source_kwargs.copy()
        if isinstance(new_inputs, dict):
            cloned_kwargs.update(new_inputs)
        else:
            cloned_kwargs["inputs_embeds"] = new_inputs
        return cloned_kwargs

    nomask_input_model_kwargs = _clone_kwargs_with_inputs(model_kwargs, nomask_embeds)
    maskimage_inputs_model_kwargs = _clone_kwargs_with_inputs(model_kwargs, maskimage_embeds)
    masktext_inputs_model_kwargs = _clone_kwargs_with_inputs(model_kwargs, masktext_embeds)
    maskall_inputs_model_kwargs = _clone_kwargs_with_inputs(model_kwargs, maskall_embeds)

    conscious_gaze_state_tracker = model_kwargs.get('conscious_gaze_state_tracker', None)
    conscious_gaze_config = model_kwargs.get('conscious_gaze_config', None)
    if conscious_gaze_state_tracker is None:
        conscious_gaze_state_tracker = getattr(self, "conscious_gaze_state_tracker", None)
    if conscious_gaze_config is None:
        conscious_gaze_config = getattr(self, "conscious_gaze_config", None)
    use_conscious_gaze_cds = (
        CONSCIOUS_GAZE_CDS_AVAILABLE and
        conscious_gaze_state_tracker is not None and
        conscious_gaze_config is not None and
        conscious_gaze_config.cds_enabled
    )

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **nomask_input_model_kwargs)
        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        print("synced_gpus",synced_gpus,"this_peer_finished",this_peer_finished)
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        
        next_token_logits = outputs.logits[:, -1, :]

        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        



        # Contrastive decoding passes with masked inputs
        print("maskimage success")
        maskimage_inputs = self.prepare_inputs_for_generation(input_ids, **maskimage_inputs_model_kwargs)
        outputs_cd = self(
            **maskimage_inputs,
            return_dict=True,
            output_attentions=output_attentions_wo_img,
            output_hidden_states=output_hidden_states_wo_img,
        )
        next_token_logits_cd = outputs_cd.logits[:, -1, :]

        print("masktext success")
        masktext_inputs = self.prepare_inputs_for_generation(input_ids, **masktext_inputs_model_kwargs)
        outputs_masktext = self(
            **masktext_inputs,
            return_dict=True,
            output_attentions=output_attentions_wo_img,
            output_hidden_states=output_hidden_states_wo_img,
        )
        next_token_logits_masktext = outputs_masktext.logits[:, -1, :]

        print("maskall success")
        maskall_inputs = self.prepare_inputs_for_generation(input_ids, **maskall_inputs_model_kwargs)
        outputs_maskall = self(
            **maskall_inputs,
            return_dict=True,
            output_attentions=output_attentions_wo_img,
            output_hidden_states=output_hidden_states_wo_img,
        )
        next_token_logits_maskall = outputs_maskall.logits[:, -1, :]


        if use_conscious_gaze_cds:
            current_entropy = compute_logits_entropy(next_token_logits)

            beta, variance = detect_cognitive_demand(
                next_token_logits,
                next_token_logits_cd,
                next_token_logits_masktext,
                next_token_logits_maskall,
                threshold=conscious_gaze_config.variance_threshold,
                top_k=conscious_gaze_config.top_k_tokens
            )

            conscious_gaze_state_tracker.update(beta, variance, current_entropy)

            print(f"[Conscious Gaze CDS] β={beta}, variance={variance:.4f}, entropy={current_entropy:.4f}")

            # Collect token-level CDS statistics for analysis
            token_collector = get_token_collector()

            # Decode the current candidates for logging
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # Grab the top-k candidates
                top_k_values, top_k_indices = torch.topk(next_token_logits, k=min(5, next_token_logits.size(-1)), dim=-1)

                for batch_idx in range(next_token_logits.shape[0]):
                    for k in range(top_k_indices.shape[1]):
                        token_id = top_k_indices[batch_idx, k].item()
                        token_text = self.tokenizer.decode([token_id]).strip()

                        # Store one entry per candidate token
                        token_collector.add_token_data(
                            token_text=token_text,
                            variance=variance,
                            token_id=token_id
                        )


        cd_beta = 0.1
        cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values

        shapley_value = next_token_logits-next_token_logits_cd - next_token_logits_masktext+ next_token_logits_maskall
        shapley_variance = torch.var(shapley_value,dim=1)
        mask = shapley_variance <var_k
        shapley_value[mask] = 0

        
        diffs = next_token_logits+ shapley_value 
        cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
        cd_logits = logits_processor(input_ids, cd_logits)
        cd_logits = logits_warper(input_ids, cd_logits)
        cd_probs = nn.functional.softmax(cd_logits, dim=-1)
        next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)





        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        
        nomask_input_model_kwargs = self._update_model_kwargs_for_generation(
            outputs, nomask_input_model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        masktext_inputs_model_kwargs = self._update_model_kwargs_for_generation(
            outputs_masktext, masktext_inputs_model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        maskimage_inputs_model_kwargs = self._update_model_kwargs_for_generation(
            outputs_cd, maskimage_inputs_model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        maskall_inputs_model_kwargs = self._update_model_kwargs_for_generation(
            outputs_maskall, maskall_inputs_model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )        



        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break
        
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids


def evolve_ours_sampling(k=1.0):
    global var_k
    var_k = k
    transformers.generation.utils.GenerationMixin.sample = shapley_sample
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = shapley_sample
