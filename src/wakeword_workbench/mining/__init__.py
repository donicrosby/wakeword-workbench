"""Hard negative mining module for wake word detection.

This module provides tools for mining hard negative samples from long audio
recordings, extracting false positives, and merging them back into the training
dataset for iterative model improvement.
"""

from .extractor import ExtractedClip, extract_false_positives
from .long_audio import WindowPrediction, process_long_audio
from .merge_back import MergeBackError, MergeResult, add_to_training

__all__ = [
    "WindowPrediction",
    "process_long_audio",
    "ExtractedClip",
    "extract_false_positives",
    "MergeResult",
    "MergeBackError",
    "add_to_training",
]
