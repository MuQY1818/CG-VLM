"""Lightweight helpers for patching attention layers with Conscious Gaze."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .fci import apply_visual_attention_boost, should_apply_fci


class AttentionWrapper:
    """Inject FCI logic into existing attention layers via monkey patching."""

    def __init__(self, config, state_tracker, tokenizer=None):
        """
        Args:
            config: Instance of :class:`ConsciousGazeConfig`.
            state_tracker: Tracks CDS statistics across decoding steps.
            tokenizer: Optional tokenizer for identifying low-information tokens.
        """
        self.config = config
        self.state_tracker = state_tracker
        self.tokenizer = tokenizer
        self.original_forwards = {}

    def wrap_qformer_attention(self, qformer_model, layer_idx: int):
        """Replace the forward pass of the given Q-Former attention layer."""
        layer = qformer_model.encoder.layer[layer_idx]
        attention_module = layer.attention.attention

        original_forward = attention_module.forward
        self.original_forwards[f"qformer_layer_{layer_idx}"] = original_forward

        def wrapped_forward(
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
        ):
            # Reproduce the original attention logic to expose the scores.
            batch_size, seq_length, _ = hidden_states.size()

            mixed_query_layer = attention_module.query(hidden_states)
            is_cross_attention = encoder_hidden_states is not None

            if is_cross_attention:
                key_layer = attention_module.transpose_for_scores(
                    attention_module.key(encoder_hidden_states)
                )
                value_layer = attention_module.transpose_for_scores(
                    attention_module.value(encoder_hidden_states)
                )
            else:
                key_layer = attention_module.transpose_for_scores(
                    attention_module.key(hidden_states)
                )
                value_layer = attention_module.transpose_for_scores(
                    attention_module.value(hidden_states)
                )

            query_layer = attention_module.transpose_for_scores(mixed_query_layer)

            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / torch.sqrt(
                torch.tensor(attention_module.attention_head_size, dtype=attention_scores.dtype)
            )

            if (
                not is_cross_attention
                and self.config.fci_enabled
                and should_apply_fci(layer_idx, self.config.fci_layers, self.state_tracker.get_beta())
            ):
                visual_range: Tuple[int, int] = (
                    self.config.visual_token_start,
                    self.config.visual_token_end,
                )
                attention_scores = apply_visual_attention_boost(
                    attention_scores,
                    visual_range=visual_range,
                    alpha=self.config.alpha,
                    enabled=True,
                )

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = attention_module.dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_shape = context_layer.size()[:-2] + (attention_module.all_head_size,)
            context_layer = context_layer.view(new_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            return outputs

        attention_module.forward = wrapped_forward

    def wrap_all_qformer_layers(self, qformer_model):
        """Apply the wrapper to every Q-Former attention layer."""
        num_layers = len(qformer_model.encoder.layer)
        for layer_idx in range(num_layers):
            self.wrap_qformer_attention(qformer_model, layer_idx)

    def restore_original_forwards(self):
        """Placeholder for restoring the patched layers (reloading is recommended)."""
        pass


def inject_conscious_gaze(model, config, state_tracker, tokenizer=None):
    """Attach Q-Former attention hooks required by Conscious Gaze."""
    wrapper = AttentionWrapper(config, state_tracker, tokenizer)

    if hasattr(model, "qformer"):
        wrapper.wrap_all_qformer_layers(model.qformer)

    return wrapper
