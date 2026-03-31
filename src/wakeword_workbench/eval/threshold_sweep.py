"""Threshold sweep analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .far import calculate_far
from .frr import calculate_frr

if TYPE_CHECKING:
    pass


@dataclass
class ThresholdResult:
    """Result of threshold analysis at a single threshold point."""

    threshold: float
    """Confidence threshold value (0.0 to 1.0)."""

    far: float
    """False Acceptance Rate (false accepts per hour)."""

    frr: float
    """False Rejection Rate (fraction of wake words missed)."""


def sweep(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    audio_duration_hours: float,
    thresholds: int = 100,
) -> list[ThresholdResult]:
    """Generate FAR and FRR across a range of confidence thresholds.

    Iterates from 0.0 to 1.0 (inclusive) and calculates FAR and FRR
    at each point, returning a list of ThresholdResult dataclasses.

    Args:
        predictions: Array of model confidence scores (0-1) for each frame.
        ground_truth: Array of ground truth labels (1 = wake word present).
        audio_duration_hours: Total duration of audio in hours.
        thresholds: Number of threshold points to evaluate (default: 100).
            Values are evenly spaced from 0.0 to 1.0 inclusive.

    Returns:
        List of ThresholdResult dataclasses, one per threshold value.

    Raises:
        ValueError: If thresholds < 2 or audio_duration_hours <= 0.
    """
    if thresholds < 2:
        raise ValueError("thresholds must be at least 2")
    if audio_duration_hours <= 0:
        raise ValueError("audio_duration_hours must be positive")

    # Generate threshold values from 0.0 to 1.0 inclusive
    threshold_values = np.linspace(0.0, 1.0, num=thresholds)

    results: list[ThresholdResult] = []
    for t in threshold_values:
        far = calculate_far(predictions, None, audio_duration_hours, threshold=float(t))
        frr = calculate_frr(ground_truth, predictions, threshold=float(t))
        results.append(ThresholdResult(threshold=float(t), far=far, frr=frr))

    return results


def find_optimal_threshold(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    audio_duration_hours: float,
    thresholds: int = 100,
) -> ThresholdResult | None:
    """Find the threshold where FAR and FRR are most closely balanced.

    Returns the threshold point with the minimum absolute difference
    between FAR and FRR. Useful for finding an operating point that
    balances false accepts and false rejections.

    Args:
        predictions: Array of model confidence scores (0-1) for each frame.
        ground_truth: Array of ground truth labels (1 = wake word present).
        audio_duration_hours: Total duration of audio in hours.
        thresholds: Number of threshold points to evaluate (default: 100).

    Returns:
        ThresholdResult with the most balanced FAR/FRR, or None if no results.

    Raises:
        ValueError: If thresholds < 2 or audio_duration_hours <= 0.
    """
    results = sweep(predictions, ground_truth, audio_duration_hours, thresholds)

    if not results:
        return None

    return min(results, key=lambda r: abs(r.far - r.frr))
