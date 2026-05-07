"""
Cognitive Demand Sensing (CDS).

Measures task complexity on the fly and decides when to enter focus mode by
computing the variance of Harsanyi interaction values.
"""

import torch
import torch.nn as nn
import json
import os
from typing import Dict, Tuple, Optional, List

# spaCy is optional and only used for POS tagging heuristics
try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    print("Warning: spaCy not available. POS tagging will be disabled.")


def compute_harsanyi_dividend(
    logits_nomask: torch.Tensor,
    logits_maskimage: torch.Tensor,
    logits_masktext: torch.Tensor,
    logits_maskall: torch.Tensor,
    top_k: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Harsanyi interaction dividends using a top-k approximation.

    For coalition {v, p} (visual input v and textual input p):
    I({v, p}) = L(v, p) - L(v) - L(p) + L(∅)

    Args:
        logits_nomask: Logits with both modalities [batch, vocab_size]
        logits_maskimage: Logits with the image masked out
        logits_masktext: Logits with the text masked out
        logits_maskall: Logits with both modalities masked
        top_k: Number of top tokens used for the approximation

    Returns:
        interaction_values: Interaction strengths [batch, top_k]
        top_k_indices: Token indices used for the computation [batch, top_k]
    """
    batch_size, vocab_size = logits_nomask.shape

    # Select the top-k candidates from the full logits
    top_k_values, top_k_indices = torch.topk(logits_nomask, k=min(top_k, vocab_size), dim=-1)

    # Gather corresponding logits under different masking setups
    logits_vi_topk = torch.gather(logits_masktext, dim=-1, index=top_k_indices)  # L(v)
    logits_pi_topk = torch.gather(logits_maskimage, dim=-1, index=top_k_indices)  # L(p)
    logits_empty_topk = torch.gather(logits_maskall, dim=-1, index=top_k_indices)  # L(∅)

    # I({v,p}) = L(v,p) - L(v) - L(p) + L(∅)
    interaction_values = (
        top_k_values
        - logits_vi_topk
        - logits_pi_topk
        + logits_empty_topk
    )

    return interaction_values, top_k_indices


def detect_cognitive_demand(
    logits_nomask: torch.Tensor,
    logits_maskimage: torch.Tensor,
    logits_masktext: torch.Tensor,
    logits_maskall: torch.Tensor,
    threshold: float = 0.5,
    top_k: int = 50
) -> Tuple[float, float]:
    """
    Determine whether the current decoding step requires focus mode.

    The variance of interaction values D_y_t is compared with threshold k:
    - D_y_t > k  →  β = 1 (high cognitive demand, focus mode)
    - D_y_t ≤ k  →  β = 0 (default mode)
    """
    interaction_values, _ = compute_harsanyi_dividend(
        logits_nomask,
        logits_maskimage,
        logits_masktext,
        logits_maskall,
        top_k=top_k
    )

    # Aggregate the interaction variance across the batch
    variance = torch.var(interaction_values, dim=-1).mean().item()

    # Focus flag
    beta = 1.0 if variance > threshold else 0.0

    return beta, variance


class CognitiveStateTracker:
    """
    Tracks beta decisions, entropy, and variance statistics across decoding.
    """

    def __init__(self, entropy_window_size: int = 10):
        """
        Args:
            entropy_window_size: Sliding window length for entropy smoothing.
        """
        self.entropy_window_size = entropy_window_size
        self.reset()

    def reset(self):
        """Clear runtime statistics."""
        self.beta_history = []
        self.variance_history = []
        self.entropy_window = []
        self.entropy_history = []
        self.current_beta = 0.0
        self.current_variance = 0.0

    def update(
        self,
        beta: float,
        variance: float,
        entropy: float
    ):
        """
        Update the tracker with the latest beta/variance/entropy tuple.
        """
        self.current_beta = beta
        self.current_variance = variance

        self.beta_history.append(beta)
        self.variance_history.append(variance)

        # Maintain the entropy sliding window
        self.entropy_window.append(entropy)
        if len(self.entropy_window) > self.entropy_window_size:
            self.entropy_window.pop(0)

        # Keep the full entropy history for offline analysis
        self.entropy_history.append(entropy)

    def get_entropy_mean(self) -> float:
        """Return the sliding-window mean entropy."""
        if len(self.entropy_window) == 0:
            return 1.0
        return sum(self.entropy_window) / len(self.entropy_window)

    def get_beta(self) -> float:
        """Return the latest beta flag."""
        return self.current_beta

    def get_variance(self) -> float:
        """Return the latest interaction variance."""
        return self.current_variance

    def get_focus_rate(self) -> float:
        """
        Fraction of steps where beta equals one.
        """
        if len(self.beta_history) == 0:
            return 0.0
        return sum(self.beta_history) / len(self.beta_history)

    def get_stats(self) -> Dict:
        """
        Summarize core statistics for logging or debugging.
        """
        return {
            'current_beta': self.current_beta,
            'current_variance': self.current_variance,
            'focus_rate': self.get_focus_rate(),
            'total_steps': len(self.beta_history),
            'entropy_mean': self.get_entropy_mean(),
            'variance_mean': sum(self.variance_history) / len(self.variance_history) if self.variance_history else 0.0,
        }


def compute_logits_entropy(logits: torch.Tensor) -> float:
    """
    Compute token-level entropy using softmax probabilities.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    return entropy


def classify_token_type(token_text: str) -> str:
    """
    Classify tokens into keywords versus function words.
    """
    if not SPACY_AVAILABLE or not nlp:
        # Lightweight fallback if spaCy is not available
        function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'and', 'or', 'but', 'so', '.', ',', '?', '!'}
        return "function_word" if token_text.lower().strip() in function_words else "keyword"

    try:
        doc = nlp(token_text)
        if len(doc) == 0:
            return "keyword"

        # Look at the POS tag of the first token
        pos = doc[0].pos_

        if pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
            return "keyword"
        elif pos in ['ADP', 'DET', 'PUNCT', 'CCONJ', 'SCONJ', 'AUX']:
            return "function_word"
        else:
            return "keyword"
    except:
        return "keyword"


class TokenDataCollector:
    """
    Collects per-token CDS statistics for visualization and analysis.
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Args:
            output_dir: Directory used to store serialized statistics.
        """
        self.output_dir = output_dir
        self.cds_data = []  # Stores {token, variance, pos_tag, is_keyword}
        self.current_sample_id = 0
        self.current_step = 0

    def reset_sample(self, sample_id: int):
        """Reset counters at the beginning of each sample."""
        self.current_sample_id = sample_id
        self.current_step = 0

    def add_token_data(self,
                      token_text: str,
                      variance: float,
                      token_id: int = None):
        """
        Add a new token-level variance record.
        """
        # Run token classification heuristics
        token_type = classify_token_type(token_text)

        data_point = {
            "sample_id": self.current_sample_id,
            "step": self.current_step,
            "token_text": token_text,
            "token_id": token_id,
            "variance": variance,
            "token_type": token_type,
            "is_keyword": token_type == "keyword"
        }

        self.cds_data.append(data_point)
        self.current_step += 1

    def save_cds_data(self, filename: str = "cds_token_data.json"):
        """Persist collected CDS statistics as JSON."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.cds_data, f, ensure_ascii=False, indent=2)

        print(f"CDS token data saved to: {filepath}")
        print(f"Total tokens collected: {len(self.cds_data)}")

        # Basic aggregate statistics for debugging convenience
        keywords = [d for d in self.cds_data if d['is_keyword']]
        function_words = [d for d in self.cds_data if not d['is_keyword']]

        print(f"Keywords: {len(keywords)}, Function words: {len(function_words)}")
        if keywords:
            keyword_variances = [d['variance'] for d in keywords]
            print(f"Keyword variance mean: {sum(keyword_variances)/len(keyword_variances):.3f}")
        if function_words:
            function_variances = [d['variance'] for d in function_words]
            print(f"Function word variance mean: {sum(function_variances)/len(function_variances):.3f}")

    def get_data_summary(self) -> Dict:
        """Summarize token statistics for downstream consumers."""
        keywords = [d for d in self.cds_data if d['is_keyword']]
        function_words = [d for d in self.cds_data if not d['is_keyword']]

        summary = {
            "total_tokens": len(self.cds_data),
            "keywords_count": len(keywords),
            "function_words_count": len(function_words),
            "keyword_variances": [d['variance'] for d in keywords],
            "function_word_variances": [d['variance'] for d in function_words]
        }

        if keywords:
            keyword_variances = [d['variance'] for d in keywords]
            summary["keyword_variance_mean"] = sum(keyword_variances) / len(keyword_variances)
            summary["keyword_variance_std"] = torch.var(torch.tensor(keyword_variances)).sqrt().item()
        else:
            summary["keyword_variance_mean"] = 0.0
            summary["keyword_variance_std"] = 0.0

        if function_words:
            function_variances = [d['variance'] for d in function_words]
            summary["function_variance_mean"] = sum(function_variances) / len(function_variances)
            summary["function_variance_std"] = torch.var(torch.tensor(function_variances)).sqrt().item()
        else:
            summary["function_variance_mean"] = 0.0
            summary["function_variance_std"] = 0.0

        return summary


# Global singleton helpers
_token_collector = None

def get_token_collector(output_dir: str = "outputs") -> TokenDataCollector:
    """Return the shared token collector instance."""
    global _token_collector
    if _token_collector is None:
        _token_collector = TokenDataCollector(output_dir)
    return _token_collector

def reset_token_collector():
    """Drop the shared token collector so a new one can be created."""
    global _token_collector
    _token_collector = None
