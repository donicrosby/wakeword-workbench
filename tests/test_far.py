"""Tests for False Acceptance Rate (FAR) calculator."""

from __future__ import annotations

import numpy as np
import pytest

from wakeword_workbench.eval.far import calculate_far, count_false_positives


class TestCalculateFar:
    """Tests for calculate_far function."""

    def test_zero_false_positives(self) -> None:
        """No detections above threshold should result in FAR of 0."""
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2])
        far = calculate_far(predictions, None, audio_duration_hours=1.0, threshold=0.5)
        assert far == 0.0

    def test_single_false_positive(self) -> None:
        """Single detection above threshold should count as one false positive."""
        predictions = np.array([0.1, 0.2, 0.8, 0.3, 0.2])
        far = calculate_far(predictions, None, audio_duration_hours=1.0, threshold=0.5)
        assert far == 1.0

    def test_far_calculation_with_hours(self) -> None:
        """FAR should be false_positives divided by audio duration in hours."""
        # With small cooldown, 3 separated detections = 3 groups
        predictions = np.array([0.8, 0.3, 0.9, 0.4, 0.7, 0.2])
        far = calculate_far(
            predictions, None, audio_duration_hours=2.0, threshold=0.5, cooldown_frames=2
        )
        assert far == 3.0 / 2.0  # 1.5 false accepts per hour

    def test_consecutive_detections_count_as_one(self) -> None:
        """Consecutive frames above threshold should count as ONE false positive."""
        # All frames above threshold = 1 false positive group
        predictions = np.array([0.8, 0.9, 0.7, 0.6, 0.5])
        far = calculate_far(predictions, None, audio_duration_hours=1.0, threshold=0.5)
        assert far == 1.0

    def test_cooldown_skips_frames(self) -> None:
        """Cooldown should prevent counting frames within cooldown window."""
        # First detection at frame 0, cooldown of 3 should skip frames 0-2
        # Then detection at frame 3 should count as second false positive
        predictions = np.array([0.8, 0.9, 0.7, 0.6, 0.8, 0.9])
        far = calculate_far(
            predictions, None, audio_duration_hours=1.0, threshold=0.5, cooldown_frames=3
        )
        assert far == 2.0

    def test_empty_predictions(self) -> None:
        """Empty predictions array should return FAR of 0."""
        predictions = np.array([])
        far = calculate_far(predictions, None, audio_duration_hours=1.0)
        assert far == 0.0

    def test_all_below_threshold(self) -> None:
        """All predictions below threshold should return FAR of 0."""
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.49])
        far = calculate_far(predictions, None, audio_duration_hours=10.0, threshold=0.5)
        assert far == 0.0

    def test_exactly_at_threshold_not_counted(self) -> None:
        """Predictions exactly at threshold should not be counted."""
        predictions = np.array([0.5, 0.5, 0.5])  # Exactly at threshold
        far = calculate_far(predictions, None, audio_duration_hours=1.0, threshold=0.5)
        assert far == 0.0

    def test_ground_truth_ignored(self) -> None:
        """Ground truth should be ignored for FAR calculation."""
        predictions = np.array([0.8, 0.9, 0.3])
        ground_truth = np.array([1, 1, 0])  # Would cause issues if used
        far = calculate_far(predictions, ground_truth, audio_duration_hours=1.0, threshold=0.5)
        assert far == 1.0

    def test_invalid_audio_duration_raises(self) -> None:
        """Zero or negative audio duration should raise ValueError."""
        predictions = np.array([0.8, 0.9])
        with pytest.raises(ValueError, match="audio_duration_hours must be positive"):
            calculate_far(predictions, None, audio_duration_hours=0.0)
        with pytest.raises(ValueError, match="audio_duration_hours must be positive"):
            calculate_far(predictions, None, audio_duration_hours=-1.0)

    def test_custom_threshold(self) -> None:
        """Custom threshold should affect detection."""
        predictions = np.array([0.3, 0.4, 0.6, 0.7])  # Last two above 0.5 but first two above 0.3
        far_05 = calculate_far(predictions, None, audio_duration_hours=1.0, threshold=0.5)
        far_03 = calculate_far(predictions, None, audio_duration_hours=1.0, threshold=0.3)
        assert far_05 == 1.0
        assert far_03 == 1.0  # Consecutive detections count as one

    def test_large_cooldown(self) -> None:
        """Large cooldown should skip past all remaining frames."""
        predictions = np.array([0.8, 0.9, 0.7, 0.6, 0.5])
        far = calculate_far(
            predictions,
            None,
            audio_duration_hours=1.0,
            threshold=0.5,
            cooldown_frames=100,  # More than length of array
        )
        assert far == 1.0

    def test_example_from_docstring(self) -> None:
        """Test the example from the docstring."""
        predictions = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6])
        # With cooldown_frames=2: groups are [0.8, 0.9] and [0.7, 0.6]
        far = calculate_far(
            predictions, None, audio_duration_hours=2.0, threshold=0.5, cooldown_frames=2
        )
        assert far == 2.0 / 2.0  # 1.0 false accepts per hour


class TestCountFalsePositives:
    """Tests for count_false_positives helper function."""

    def test_no_false_positives(self) -> None:
        """No detections should return 0."""
        predictions = np.array([0.1, 0.2, 0.3, 0.4])
        count = count_false_positives(predictions, threshold=0.5)
        assert count == 0

    def test_single_false_positive_group(self) -> None:
        """Single detection should return 1."""
        predictions = np.array([0.1, 0.6, 0.3, 0.2])
        count = count_false_positives(predictions, threshold=0.5)
        assert count == 1

    def test_multiple_separated_groups(self) -> None:
        """Separated detections should each count as separate false positive."""
        # Using cooldown_frames=2: detections at frames 0, 2, 4
        predictions = np.array([0.6, 0.3, 0.7, 0.2, 0.8])
        count = count_false_positives(predictions, threshold=0.5, cooldown_frames=2)
        assert count == 3

    def test_consecutive_frames_count_as_one(self) -> None:
        """Consecutive frames above threshold count as one group."""
        predictions = np.array([0.6, 0.8, 0.9, 0.7, 0.3])
        count = count_false_positives(predictions, threshold=0.5)
        assert count == 1

    def test_empty_array(self) -> None:
        """Empty array should return 0."""
        predictions = np.array([])
        count = count_false_positives(predictions, threshold=0.5)
        assert count == 0

    def test_cooldown_effect(self) -> None:
        """Cooldown should skip frames within cooldown window."""
        predictions = np.array([0.6, 0.8, 0.3, 0.6, 0.8])
        count_with_cooldown = count_false_positives(predictions, threshold=0.5, cooldown_frames=2)
        # Group 1: [0.6, 0.8] at frames 0-1, Group 2: [0.6, 0.8] at frames 3-4
        assert count_with_cooldown == 2

    def test_docstring_example(self) -> None:
        """Test the example from count_false_positives docstring."""
        predictions = np.array([0.1, 0.6, 0.8, 0.2, 0.7])
        count = count_false_positives(predictions, threshold=0.5, cooldown_frames=2)
        # Group 1: [0.6, 0.8] - counted at frame 1, frames 1-2 skipped
        # Frame 3 is below threshold
        # Group 2: [0.7] - counted at frame 4
        assert count == 2


class TestEdgeCases:
    """Edge case tests for FAR calculation."""

    def test_frame_above_threshold(self) -> None:
        """Frame above threshold should count."""
        predictions = np.array([0.3, 0.51, 0.2])  # Middle frame above threshold
        far = calculate_far(predictions, None, audio_duration_hours=1.0, threshold=0.5)
        assert far == 1.0

    def test_all_frames_above_threshold(self) -> None:
        """All frames above threshold should count as one false positive."""
        predictions = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.9])
        far = calculate_far(predictions, None, audio_duration_hours=1.0, threshold=0.5)
        assert far == 1.0

    def test_alternating_with_cooldown(self) -> None:
        """Alternating above/below threshold with cooldown should count appropriately."""
        # With cooldown=2: frame 0 (0.8) detected, skip 0-1, frame 2 (0.9) detected, skip 2-3
        # frame 4 (0.7) detected, skip 4-5, done
        predictions = np.array([0.8, 0.2, 0.9, 0.1, 0.7, 0.3])
        count = count_false_positives(predictions, threshold=0.5, cooldown_frames=2)
        assert count == 3

    def test_very_small_audio_duration(self) -> None:
        """Very small audio duration should give large FAR."""
        # With cooldown_frames=2: [0.8, 0.9] are consecutive, counts as 1 group
        predictions = np.array([0.8, 0.9])
        far = calculate_far(
            predictions, None, audio_duration_hours=0.01, threshold=0.5, cooldown_frames=2
        )
        assert far == 1.0 / 0.01  # 100 false accepts per hour

    def test_large_audio_duration(self) -> None:
        """Large audio duration should give small FAR."""
        # With cooldown_frames=2: [0.8, 0.9] are consecutive, counts as 1 group
        predictions = np.array([0.8, 0.9])
        far = calculate_far(
            predictions, None, audio_duration_hours=100.0, threshold=0.5, cooldown_frames=2
        )
        assert far == 0.01  # 0.01 false accepts per hour

    def test_single_isolated_detection(self) -> None:
        """Single isolated detection should count as one false positive."""
        predictions = np.array([0.1, 0.1, 0.9, 0.1, 0.1])
        far = calculate_far(predictions, None, audio_duration_hours=1.0, threshold=0.5)
        assert far == 1.0

    def test_multiple_isolated_detections_with_cooldown(self) -> None:
        """Multiple isolated detections with cooldown should count separately."""
        # Detections at frames 0, 3, 6 with 2 frames gap between each
        predictions = np.array([0.8, 0.1, 0.1, 0.9, 0.1, 0.1, 0.7])
        # With cooldown=2: frame 0 detected, skip 0-1, check 2 (below), check 3 (detected)
        # skip 3-4, check 5 (below), check 6 (detected)
        count = count_false_positives(predictions, threshold=0.5, cooldown_frames=2)
        assert count == 3
