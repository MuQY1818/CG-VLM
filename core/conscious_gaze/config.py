"""
Conscious Gaze configuration management utilities.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConsciousGazeConfig:
    """Complete configuration for the Conscious Gaze framework."""

    # Visual token range inside the language model input (Q-Former outputs)
    visual_token_start: int = 1   # Starting index of Q-Former outputs in the LM input
    visual_token_end: int = 33    # Q-Former produces 32 tokens, hence [1, 33)

    # CDS (Cognitive Demand Sensing) parameters
    cds_enabled: bool = True
    variance_threshold: float = 0.5  # Interaction variance threshold k
    top_k_tokens: int = 50           # Number of top candidates to approximate the variance
    entropy_window_size: int = 10    # Window size for token-level entropy smoothing

    # FCI (Focused Consensus Induction) parameters using the visual attention boost
    fci_enabled: bool = True
    alpha: float = 0.5               # Strength of the visual attention boost
    fci_layers: Optional[List[int]] = None  # LM layer indices where the boost is applied

  
    # Default layer range
    def __post_init__(self):
        if self.fci_layers is None:
            # Align with middle-layer prior: apply FCI to layers 5 through 18
            self.fci_layers = list(range(5, 19))

    # Low-information token definitions (punctuation, stopwords, etc.)
    low_information_tokens: List[str] = field(default_factory=lambda: [
        '.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}',
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
        'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
    ])

    def get_state_dict(self):
        """Return a lightweight dictionary representation."""
        return {
            'cds_enabled': self.cds_enabled,
            'fci_enabled': self.fci_enabled,
            'visual_token_start': self.visual_token_start,
            'visual_token_end': self.visual_token_end,
            'variance_threshold': self.variance_threshold,
            'top_k_tokens': self.top_k_tokens,
            'alpha': self.alpha,
            'fci_layers': self.fci_layers,
        }
