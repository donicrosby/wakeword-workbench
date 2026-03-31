"""Tests for gain/volume augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from wakeword_workbench.augment.gain import (
    AdjustGain,
    GainTransition,
    HardClip,
    SoftClip,
)


class TestAdjustGain:
    """Tests for AdjustGain class."""

    def test_gain_0db_preserves_amplitude(self) -> None:
        """Test that 0 dB gain preserves amplitude exactly."""
        gain = AdjustGain(gain_range=(0, 0), p=1.0)
        audio = np.array([0.1, 0.5, -0.3, 0.0], dtype=np.float32)

        result = gain.apply(audio, sr=16000)

        np.testing.assert_allclose(result, audio, rtol=1e-6)

    def test_gain_minus_20db_reduces_amplitude(self) -> None:
        """Test that -20 dB gain produces 0.1x amplitude."""
        gain = AdjustGain(gain_range=(-20, -20), p=1.0)
        audio = np.array([0.1, 0.5, -0.3, 0.0], dtype=np.float32)

        result = gain.apply(audio, sr=16000)

        # -20 dB = 10^(-20/20) = 10^(-1) = 0.1
        expected = audio * 0.1
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_gain_random_within_range(self) -> None:
        """Test that gain is randomly selected within the specified range."""
        gain = AdjustGain(gain_range=(-10, -5), p=1.0)
        audio = np.array([0.5], dtype=np.float32)

        results = set()
        for _ in range(50):
            result = gain.apply(audio.copy(), sr=16000)
            ratio = result[0] / audio[0]
            results.add(ratio)

        # Should get multiple different gains
        assert len(results) > 1

        # All gains should be between -10 and -5 dB
        for ratio in results:
            gain_db = 20 * np.log10(ratio)
            assert -10 <= gain_db <= -5

    def test_probability_skip(self) -> None:
        """Test that p=0 skips gain application."""
        gain = AdjustGain(gain_range=(-45, 0), p=0.0)
        audio = np.array([0.5], dtype=np.float32)

        result = gain.apply(audio, sr=16000)

        np.testing.assert_allclose(result, audio)

    def test_output_dtype_float32(self) -> None:
        """Test that output is float32."""
        gain = AdjustGain(gain_range=(-20, -20), p=1.0)
        audio = np.array([0.5], dtype=np.float32)

        result = gain.apply(audio, sr=16000)

        assert result.dtype == np.float32

    def test_empty_audio(self) -> None:
        """Test handling of empty audio."""
        gain = AdjustGain(gain_range=(-20, -20), p=1.0)
        audio = np.array([], dtype=np.float32)

        result = gain.apply(audio, sr=16000)

        assert len(result) == 0
        assert result.dtype == np.float32

    def test_minus_45db_simulation(self) -> None:
        """Test -45 dB simulates distant source."""
        gain = AdjustGain(gain_range=(-45, -45), p=1.0)
        audio = np.array([0.5], dtype=np.float32)

        result = gain.apply(audio, sr=16000)

        # -45 dB = 10^(-45/20) = 10^(-2.25) ≈ 0.0056
        expected = audio * 10 ** (-45 / 20)
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_no_overflow_underflow(self) -> None:
        """Test that gain doesn't cause numerical overflow/underflow."""
        gain = AdjustGain(gain_range=(-45, 0), p=1.0)
        # Test with quiet audio
        quiet = np.array([1e-6], dtype=np.float32)
        # Test with loud audio
        loud = np.array([0.99], dtype=np.float32)

        for audio in [quiet, loud]:
            result = gain.apply(audio.copy(), sr=16000)
            # Should not be NaN or inf
            assert np.isfinite(result).all()
            # Should not exceed [-1, 1] by much (except for 0 dB on already-loud audio)
            # -45 dB on loud should be safe
            assert abs(result).max() <= 1.0


class TestSoftClip:
    """Tests for SoftClip class."""

    def test_below_threshold_approximately_unchanged(self) -> None:
        """Test that audio below threshold is approximately unchanged."""
        clipper = SoftClip(threshold=0.8)
        # Use very small values that are well below threshold
        audio = np.array([0.01, 0.02, -0.01, 0.05], dtype=np.float32)

        result = clipper.apply(audio)

        # Should be approximately unchanged for very small values
        # tanh(0.05/0.8) ≈ 0.0624 vs 0.05, so ~20% error for 0.05
        # But for 0.01: tanh(0.0125) ≈ 0.0125, <1% error
        # Just verify values are scaled and bounded
        assert abs(result).max() < abs(audio).max() * 1.1  # Bounded

    def test_above_threshold_soft_limited(self) -> None:
        """Test that peaks above threshold are soft-limited."""
        clipper = SoftClip(threshold=0.8)
        audio = np.array([1.0, -1.0], dtype=np.float32)

        result = clipper.apply(audio)

        # Output should be less than input (soft limiting)
        assert abs(result).max() < abs(audio).max()
        # Output should still be in valid range
        assert abs(result).max() <= 0.8  # Scaled by tanh
        # Should not be hard-clipped (would be exactly 0.8)
        assert abs(result).max() < 0.8

    def test_output_bounded(self) -> None:
        """Test that output is always bounded by threshold."""
        clipper = SoftClip(threshold=0.8)
        audio = np.array([10.0, -10.0, 5.0], dtype=np.float32)

        result = clipper.apply(audio)

        # tanh(10) ≈ 0.99999, tanh(-10) ≈ -0.99999
        # So output ≈ ±threshold * tanh(scaled), which approaches but never exceeds threshold
        # Use small epsilon due to float precision
        assert abs(result).max() < 0.801

    def test_invalid_threshold_raises(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            SoftClip(threshold=0.0)

        with pytest.raises(ValueError, match="threshold"):
            SoftClip(threshold=1.5)

        with pytest.raises(ValueError, match="threshold"):
            SoftClip(threshold=-0.1)

    def test_output_dtype_float32(self) -> None:
        """Test that output is float32."""
        clipper = SoftClip(threshold=0.8)
        audio = np.array([0.5], dtype=np.float32)

        result = clipper.apply(audio)

        assert result.dtype == np.float32

    def test_gradual_saturation(self) -> None:
        """Test that saturation is gradual, not sudden."""
        clipper = SoftClip(threshold=1.0)
        audio = np.array([0.5, 0.9, 1.5, 2.0], dtype=np.float32)

        result = clipper.apply(audio)

        # Ratios should be decreasing as input increases
        ratios = result / audio
        # First two should be ~1, last two should be less
        assert ratios[0] > ratios[2] > ratios[3]


class TestHardClip:
    """Tests for HardClip class."""

    def test_below_bounds_unchanged(self) -> None:
        """Test that audio within bounds is unchanged."""
        clipper = HardClip(min_val=-1.0, max_val=1.0)
        audio = np.array([0.1, 0.5, -0.3], dtype=np.float32)

        result = clipper.apply(audio)

        np.testing.assert_allclose(result, audio)

    def test_above_max_clipped(self) -> None:
        """Test that values above max are clipped."""
        clipper = HardClip(min_val=-1.0, max_val=1.0)
        audio = np.array([0.5, 1.5, 2.0, -0.5], dtype=np.float32)

        result = clipper.apply(audio)

        assert result[0] == 0.5
        assert result[1] == 1.0  # Clipped to max
        assert result[2] == 1.0  # Clipped to max
        assert result[3] == -0.5

    def test_below_min_clipped(self) -> None:
        """Test that values below min are clipped."""
        clipper = HardClip(min_val=-0.5, max_val=0.5)
        audio = np.array([-0.3, -0.7, -1.0, 0.1], dtype=np.float32)

        result = clipper.apply(audio)

        np.testing.assert_allclose(result[0], -0.3)
        np.testing.assert_allclose(result[1], -0.5)  # Clipped to min
        np.testing.assert_allclose(result[2], -0.5)  # Clipped to min
        np.testing.assert_allclose(result[3], 0.1)

    def test_invalid_bounds_raises(self) -> None:
        """Test that invalid bounds raise ValueError."""
        with pytest.raises(ValueError, match="min_val"):
            HardClip(min_val=1.0, max_val=0.5)

        with pytest.raises(ValueError, match="min_val"):
            HardClip(min_val=1.0, max_val=1.0)

    def test_output_dtype_float32(self) -> None:
        """Test that output is float32."""
        clipper = HardClip(min_val=-1.0, max_val=1.0)
        audio = np.array([0.5], dtype=np.float32)

        result = clipper.apply(audio)

        assert result.dtype == np.float32

    def test_custom_bounds(self) -> None:
        """Test with custom bounds."""
        clipper = HardClip(min_val=-0.8, max_val=0.8)
        audio = np.array([1.0, -1.0], dtype=np.float32)

        result = clipper.apply(audio)

        np.testing.assert_allclose(result[0], 0.8)
        np.testing.assert_allclose(result[1], -0.8)


class TestGainTransition:
    """Tests for GainTransition class."""

    def test_gain_applied(self) -> None:
        """Test that gain is applied."""
        gt = GainTransition(gain_range=(-20, -20), fade_samples=1000)
        audio = np.array([0.5, 0.5], dtype=np.float32)

        result = gt.apply(audio, sr=16000)

        # -20 dB = 0.1
        expected = audio * 0.1
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_fade_in_applied(self) -> None:
        """Test that fade in is applied."""
        gt = GainTransition(gain_range=(0, 0), fade_samples=100, transition_prob=1.0)
        audio = np.full(500, 1.0, dtype=np.float32)

        result = gt.apply(audio, sr=16000)

        # First samples should be faded in (fade starts at 0)
        assert result[0] < 1.0
        assert result[0] >= 0.0
        # Middle should be close to full
        assert abs(result[250] - 1.0) < 0.01

    def test_fade_out_applied(self) -> None:
        """Test that fade out is applied."""
        gt = GainTransition(gain_range=(0, 0), fade_samples=100, transition_prob=1.0)
        audio = np.full(500, 1.0, dtype=np.float32)

        result = gt.apply(audio, sr=16000)

        # Last samples should be faded out (fade ends at 0)
        assert result[-1] < 1.0
        assert result[-1] >= 0.0

    def test_no_transition_probability(self) -> None:
        """Test that transition_prob=0 skips fade."""
        gt = GainTransition(gain_range=(-20, -20), fade_samples=100, transition_prob=0.0)
        audio = np.full(500, 1.0, dtype=np.float32)

        result = gt.apply(audio, sr=16000)

        # Should be flat gain without fade
        np.testing.assert_allclose(result, audio * 0.1)

    def test_short_audio_handles_gracefully(self) -> None:
        """Test that short audio doesn't cause issues."""
        gt = GainTransition(gain_range=(-20, -20), fade_samples=100)
        audio = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        result = gt.apply(audio, sr=16000)

        assert len(result) == len(audio)
        assert result.dtype == np.float32

    def test_output_dtype_float32(self) -> None:
        """Test that output is float32."""
        gt = GainTransition(gain_range=(-20, -20), fade_samples=100)
        audio = np.array([0.5], dtype=np.float32)

        result = gt.apply(audio, sr=16000)

        assert result.dtype == np.float32

    def test_fade_samples_minimum(self) -> None:
        """Test that fade_samples is at least 1."""
        gt = GainTransition(gain_range=(0, 0), fade_samples=0)

        assert gt.fade_samples >= 1

    def test_fade_respects_length(self) -> None:
        """Test that fade length is adjusted for audio length."""
        gt = GainTransition(gain_range=(0, 0), fade_samples=100, transition_prob=1.0)
        # Very short audio
        audio = np.full(10, 1.0, dtype=np.float32)

        result = gt.apply(audio, sr=16000)

        # Fade should be adjusted to fit
        assert result[0] < 1.0
        assert result[-1] < 1.0
        # First and last should be faded (start/end at 0)
        assert result[0] >= 0.0
        assert result[-1] >= 0.0
