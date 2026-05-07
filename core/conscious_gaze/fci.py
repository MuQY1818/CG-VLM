"""
Focused Consensus Induction (FCI) implemented as a visual attention boost.

Inspired by the "Middle Layers" observation:
- Only modify the last query token's attention toward visual tokens.
- Use the head-wise mean magnitude as the boost signal.
- Activate the boost only when CDS reports beta = 1.
"""

import torch
import torch.nn as nn
import json
import os
from typing import Optional, Tuple, List, Dict


def apply_visual_attention_boost(
    attention_scores: torch.Tensor,
    visual_range: Tuple[int, int],
    alpha: float = 0.5,
    enabled: bool = True
) -> torch.Tensor:
    """
    Apply the visual attention boost to pre-softmax attention scores.

    Only the last query position (current decoding token) is modified and only
    on the visual token range.
    """
    if not enabled:
        return attention_scores

    batch_size, num_heads, seq_len_q, seq_len_k = attention_scores.shape
    vis_start, vis_end = visual_range

    # Validate the range
    if vis_start < 0 or vis_end > seq_len_k or vis_start >= vis_end:
        return attention_scores

    # Pull the [batch, num_heads, vis_tokens] slice for the last query
    last_query_vis_attn = attention_scores[:, :, -1, vis_start:vis_end]

    # Head-mean absolute magnitude serves as the boost
    mean_abs_attn = last_query_vis_attn.abs().mean(dim=1, keepdim=True)

    # Add the boost to the target span
    attention_scores[:, :, -1, vis_start:vis_end] = (
        attention_scores[:, :, -1, vis_start:vis_end] + alpha * mean_abs_attn
    )

    return attention_scores


def apply_consensus_induction(
    attention_scores: torch.Tensor,
    alpha: float = 0.3,
    enabled: bool = True
) -> torch.Tensor:
    """
    Deprecated pre-boost consensus implementation retained for reference.
    """
    if not enabled:
        return attention_scores

    batch_size, num_heads, seq_len_q, seq_len_k = attention_scores.shape

    # Compute head-wise consensus bias
    abs_scores = torch.abs(attention_scores)
    mean_consensus = abs_scores.mean(dim=1, keepdim=True)
    consensus_bias = alpha * mean_consensus

    # Broadcast to all heads
    modified_scores = attention_scores + consensus_bias

    return modified_scores


def compute_head_inconsistency_index(
    attention_probs: torch.Tensor,
    method: str = 'variance'
) -> torch.Tensor:
    """
    Compute the Head Inconsistency Index (HII) for diagnostic purposes.
    """
    batch_size, num_heads, seq_len_q, seq_len_k = attention_probs.shape

    if method == 'variance':
        # Head-wise variance per query/key location
        variance = torch.var(attention_probs, dim=1)

        # Summation across key positions
        hii = variance.sum(dim=-1)  # [batch, seq_len_q]

    elif method == 'entropy':
        # Mean distribution entropy across heads
        mean_probs = attention_probs.mean(dim=1)  # [batch, seq_len_q, seq_len_k]

        # Avoid numerical issues
        mean_probs = mean_probs.clamp(min=1e-9)

        # H = -Σ p*log(p)
        entropy = -(mean_probs * torch.log(mean_probs)).sum(dim=-1)  # [batch, seq_len_q]
        hii = entropy

    else:
        raise ValueError(f"Unknown HII method: {method}")

    return hii


def should_apply_fci(
    current_layer_idx: int,
    fci_layers: list,
    beta: float
) -> bool:
    """
    Decide whether the visual boost should run on the current layer.
    """
    return beta == 1.0 and current_layer_idx in fci_layers


class FCIHook:
    """
    Registers attention hooks to inject the visual attention boost.
    """

    def __init__(self, config, state_tracker):
        """
        Args:
            config: ConsciousGazeConfig handle
            state_tracker: CognitiveStateTracker reference
        """
        self.config = config
        self.state_tracker = state_tracker
        self.hooks = []

    def register_hook(self, module, layer_idx: int):
        """
        Register the boost hook on a given attention module.
        """
        def fci_forward_hook(module, input, output):
            # Pull the latest beta flag
            beta = self.state_tracker.get_beta()

            # Exit early if FCI is disabled for this layer/step
            if not should_apply_fci(layer_idx, self.config.fci_layers, beta):
                return output

            # Placeholder: a concrete integration needs direct access to attn scores

            return output

        handle = module.register_forward_hook(fci_forward_hook)
        self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


class HIIIDataCollector:
    """
    Collects HII statistics for hallucination analysis.
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Args:
            output_dir: Directory used for serialization.
        """
        self.output_dir = output_dir
        self.fci_data = []  # Stores {token, hii_value, is_hallucinated, layer_id, sample_id}
        self.current_sample_id = 0
        self.current_step = 0

    def reset_sample(self, sample_id: int):
        """Reset counters for a new sample."""
        self.current_sample_id = sample_id
        self.current_step = 0

    def add_hii_data(self,
                    token_text: str,
                    hii_value: float,
                    is_hallucinated: bool,
                    layer_id: int,
                    token_id: int = None,
                    attention_probs: torch.Tensor = None):
        """
        Append a single HII datapoint.
        """
        data_point = {
            "sample_id": self.current_sample_id,
            "step": self.current_step,
            "token_text": token_text,
            "token_id": token_id,
            "hii_value": hii_value,
            "is_hallucinated": is_hallucinated,
            "layer_id": layer_id,
            "timestamp": self.current_step
        }

        # Store summary statistics if the raw attention is available
        if attention_probs is not None:
            data_point.update({
                "attention_mean": attention_probs.mean().item(),
                "attention_std": attention_probs.std().item(),
                "attention_max": attention_probs.max().item(),
                "attention_min": attention_probs.min().item()
            })

        self.fci_data.append(data_point)
        self.current_step += 1

    def add_layer_hii_batch(self,
                           tokens: List[str],
                           hii_values: List[float],
                           layer_id: int,
                           attention_probs: torch.Tensor = None):
        """
        Bulk helper for layer-wise HII dumps.
        """
        for i, (token, hii_val) in enumerate(zip(tokens, hii_values)):
            self.add_hii_data(
                token_text=token,
                hii_value=hii_val,
                is_hallucinated=False,  # Updated later using ground truth
                layer_id=layer_id,
                token_id=i,
                attention_probs=attention_probs[i] if attention_probs is not None else None
            )

    def update_hallucination_labels(self,
                                   generated_tokens: List[str],
                                   ground_truth_tokens: List[str]):
        """
        Update hallucination labels by comparing with ground-truth tokens.
        """
        for i, gen_token in enumerate(generated_tokens):
            if i < len(ground_truth_tokens):
                is_hallucinated = gen_token.lower() != ground_truth_tokens[i].lower()
            else:
                is_hallucinated = True

            for data_point in self.fci_data:
                if (data_point["sample_id"] == self.current_sample_id and
                    data_point["step"] == i and
                    data_point["token_text"] == gen_token):
                    data_point["is_hallucinated"] = is_hallucinated
                    break

    def save_fci_data(self, filename: str = "fci_hii_data.json"):
        """Persist collected HII statistics."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.fci_data, f, ensure_ascii=False, indent=2)

        print(f"FCI HII data saved to: {filepath}")
        print(f"Total data points collected: {len(self.fci_data)}")

        # Quick aggregates for sanity checks
        hallucinated = [d for d in self.fci_data if d['is_hallucinated']]
        real = [d for d in self.fci_data if not d['is_hallucinated']]

        print(f"Hallucinated tokens: {len(hallucinated)}")
        print(f"Real tokens: {len(real)}")

        if hallucinated:
            hallucinated_hii = [d['hii_value'] for d in hallucinated]
            print(f"Hallucinated HII mean: {sum(hallucinated_hii)/len(hallucinated_hii):.3f}")
        if real:
            real_hii = [d['hii_value'] for d in real]
            print(f"Real HII mean: {sum(real_hii)/len(real_hii):.3f}")

    def get_data_summary(self) -> Dict:
        """Return a dictionary summary of collected data."""
        hallucinated = [d for d in self.fci_data if d['is_hallucinated']]
        real = [d for d in self.fci_data if not d['is_hallucinated']]

        summary = {
            "total_tokens": len(self.fci_data),
            "hallucinated_count": len(hallucinated),
            "real_count": len(real),
            "hallucinated_hii_values": [d['hii_value'] for d in hallucinated],
            "real_hii_values": [d['hii_value'] for d in real]
        }

        if hallucinated:
            hallucinated_hii = [d['hii_value'] for d in hallucinated]
            summary["hallucinated_hii_mean"] = sum(hallucinated_hii) / len(hallucinated_hii)
            summary["hallucinated_hii_std"] = torch.var(torch.tensor(hallucinated_hii)).sqrt().item()
        else:
            summary["hallucinated_hii_mean"] = 0.0
            summary["hallucinated_hii_std"] = 0.0

        if real:
            real_hii = [d['hii_value'] for d in real]
            summary["real_hii_mean"] = sum(real_hii) / len(real_hii)
            summary["real_hii_std"] = torch.var(torch.tensor(real_hii)).sqrt().item()
        else:
            summary["real_hii_mean"] = 0.0
            summary["real_hii_std"] = 0.0

        return summary


# Global helper references
_hii_collector = None

def get_hii_collector(output_dir: str = "outputs") -> HIIIDataCollector:
    """Return the shared HII collector."""
    global _hii_collector
    if _hii_collector is None:
        _hii_collector = HIIIDataCollector(output_dir)
    return _hii_collector

def reset_hii_collector():
    """Reset the shared HII collector to a clean state."""
    global _hii_collector
    _hii_collector = None


# Convenience routine used during generation to gather HII data
def collect_hii_from_attention(attention_probs: torch.Tensor,
                             layer_id: int,
                             generated_tokens: List[str],
                             collector: HIIIDataCollector = None):
    """
    Record the HII value from a batch of attention probabilities.
    """
    if collector is None:
        collector = get_hii_collector()

    # Focus on the latest query position
    hii_values = compute_head_inconsistency_index(attention_probs, method='variance')

    last_query_hii = hii_values[:, -1]  # [batch]

    for batch_idx in range(attention_probs.shape[0]):
        if batch_idx < len(generated_tokens):
            hii_val = last_query_hii[batch_idx].item()
            token_text = generated_tokens[batch_idx] if batch_idx < len(generated_tokens) else "<unk>"

            collector.add_hii_data(
                token_text=token_text,
                hii_value=hii_val,
                is_hallucinated=False,  # Updated later once labels are available
                layer_id=layer_id,
                attention_probs=attention_probs[batch_idx]  # Retain the full attention map for inspection
            )
