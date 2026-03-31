"""False Acceptance Rate (FAR) calculator."""

from __future__ import annotations

import numpy as np


def calculate_far(
    predictions: np.ndarray,
    ground_truth: np.ndarray | None,
    audio_duration_hours: float,
    threshold: float = 0.5,
    cooldown_frames: int = 75,
) -> float:
    """Calculate False Accepts per Hour (FAR).

    FAR measures how many false positives occur per hour of audio.
    A false positive occurs when the model incorrectly predicts a wake word
    on non-wake-word audio.

    Args:
        predictions: Array of model confidence scores (0-1) for each frame.
        ground_truth: Optional array of ground truth labels (not used for FAR,
            but included for API compatibility with calculate_frr).
        audio_duration_hours: Total duration of audio in hours.
        threshold: Confidence threshold for detection (default: 0.5).
        cooldown_frames: Number of frames to skip after a detection to prevent
            double-counting (default: 75, which is ~1.5 seconds at 20ms frames).

    Returns:
        False accepts per hour (FAR value).

    Raises:
        ValueError: If audio_duration_hours is not positive.

    Example:
        >>> predictions = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6])
        >>> far = calculate_far(predictions, None, audio_duration_hours=2.0, threshold=0.5)
        >>> # Returns false accepts per hour
    """
    if audio_duration_hours <= 0:
        raise ValueError("audio_duration_hours must be positive")

    if len(predictions) == 0:
        return 0.0

    # Find all frames above threshold
    detections = predictions > threshold

    # Count false positives using cooldown logic
    # Each group of consecutive detections above threshold counts as ONE false positive
    # After counting a detection, skip cooldown_frames to prevent double-counting
    false_positives = 0
    i = 0
    n = len(detections)

    while i < n:
        if detections[i]:
            # Found a detection - count it as one false positive
            false_positives += 1
            # Skip cooldown frames to prevent double-counting same trigger
            i += cooldown_frames
        else:
            i += 1

    # Calculate FAR: false positives per hour
    far = false_positives / audio_duration_hours
    return float(far)


def count_false_positives(
    predictions: np.ndarray,
    threshold: float = 0.5,
    cooldown_frames: int = 75,
) -> int:
    """Count the number of false positive groups in predictions.

    This is a helper function that returns the raw count of false positive
    groups without dividing by audio duration. Useful for testing and
    debugging.

    Args:
        predictions: Array of model confidence scores (0-1) for each frame.
        threshold: Confidence threshold for detection (default: 0.5).
        cooldown_frames: Number of frames to skip after a detection.

    Returns:
        Number of false positive groups detected.

    Example:
        >>> predictions = np.array([0.1, 0.6, 0.8, 0.2, 0.7])
        >>> count_false_positives(predictions, threshold=0.5, cooldown_frames=2)
        2  # Two groups: [0.6, 0.8] and [0.7]
    """
    if len(predictions) == 0:
        return 0

    detections = predictions > threshold
    false_positives = 0
    i = 0
    n = len(detections)

    while i < n:
        if detections[i]:
            false_positives += 1
            i += cooldown_frames
        else:
            i += 1

    return false_positives
