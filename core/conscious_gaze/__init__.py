"""Public exports for the Conscious Gaze framework."""

from .config import ConsciousGazeConfig
from .cds import compute_harsanyi_dividend, detect_cognitive_demand, CognitiveStateTracker
from .fci import apply_visual_attention_boost, apply_consensus_induction, should_apply_fci

__all__ = [
    "ConsciousGazeConfig",
    "CognitiveStateTracker",
    "compute_harsanyi_dividend",
    "detect_cognitive_demand",
    "apply_visual_attention_boost",
    "apply_consensus_induction",
    "should_apply_fci",
]
