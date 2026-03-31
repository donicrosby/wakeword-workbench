"""False Rejection Rate (FRR) calculator."""

from __future__ import annotations

import numpy as np


def calculate_frr(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Calculate False Rejection Rate (FRR).

    FRR measures the proportion of positive samples that are incorrectly
    classified as negative. A false negative occurs when a positive sample
    (wake word present) has a prediction score below the threshold.

    FRR = false_negatives / total_positives

    Args:
        predictions: Array of model confidence scores (0-1) for each sample.
        ground_truth: Binary array where 1 = positive (wake word present),
            0 = negative (wake word absent).
        threshold: Confidence threshold for detection (default: 0.5).

    Returns:
        False Rejection Rate as a float between 0.0 and 1.0.
        Returns 0.0 if there are no positive samples.

    Raises:
        ValueError: If predictions and ground_truth have different lengths,
            or if threshold is not in the valid range [0, 1].

    Example:
        >>> predictions = np.array([0.9, 0.3, 0.8, 0.2, 0.6])
        >>> ground_truth = np.array([1, 1, 1, 0, 0])
        >>> frr = calculate_frr(predictions, ground_truth, threshold=0.5)
        >>> # Positive samples: indices 0, 1, 2 with predictions 0.9, 0.3, 0.8
        >>> # Only index 1 is below threshold (false negative)
        >>> # FRR = 1 false_neg / 3 total_positives
        >>> round(frr, 3)
        0.333

        >>> # With all positives correctly detected (no false negatives)
        >>> predictions = np.array([0.9, 0.8, 0.7])
        >>> ground_truth = np.array([1, 1, 1])
        >>> calculate_frr(predictions, ground_truth)
        0.0

        >>> # With no positive samples
        >>> predictions = np.array([0.1, 0.2, 0.3])
        >>> ground_truth = np.array([0, 0, 0])
        >>> calculate_frr(predictions, ground_truth)
        0.0
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions and ground_truth must have the same length, "
            f"got {len(predictions)} and {len(ground_truth)}"
        )

    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    if len(predictions) == 0:
        return 0.0

    # Find positive samples (ground truth = 1)
    positive_mask = ground_truth == 1
    total_positives = int(np.sum(positive_mask))

    # No positive samples - return 0.0 (no rejections possible)
    if total_positives == 0:
        return 0.0

    # Count false negatives: positive samples where prediction < threshold
    positive_predictions = predictions[positive_mask]
    false_negatives = np.sum(positive_predictions < threshold)

    # Calculate FRR
    frr = false_negatives / total_positives
    return float(frr)


def count_false_negatives(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> int:
    """Count the number of false negatives in predictions.

    A false negative is a positive sample (ground truth = 1) that has
    a prediction score below the threshold.

    Args:
        predictions: Array of model confidence scores (0-1) for each sample.
        ground_truth: Binary array where 1 = positive, 0 = negative.
        threshold: Confidence threshold for detection (default: 0.5).

    Returns:
        Number of false negatives.

    Raises:
        ValueError: If predictions and ground_truth have different lengths.

    Example:
        >>> predictions = np.array([0.9, 0.3, 0.8, 0.2])
        >>> ground_truth = np.array([1, 1, 1, 1])
        >>> count_false_negatives(predictions, ground_truth, threshold=0.5)
        2
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions and ground_truth must have the same length, "
            f"got {len(predictions)} and {len(ground_truth)}"
        )

    if len(predictions) == 0:
        return 0

    positive_mask = ground_truth == 1
    positive_predictions = predictions[positive_mask]
    return int(np.sum(positive_predictions < threshold))
