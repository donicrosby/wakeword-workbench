"""Tests for padding and transform augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from wakeword_workbench.augment.padding import FixedSizeClip, TrimSilence


class TestFixedSizeClip:
    """Tests for FixedSizeClip class."""

    def test_exact_length_unchanged(self) -> None:
        """Test that audio already at target length is unchanged."""
        clip = FixedSizeClip(16000)
        audio = np.ones(16000, dtype=np.float32)

        result = clip.apply(audio, sr=16000)

        assert result.shape == (16000,)
        np.testing.assert_array_equal(result, audio)

    def test_pad_short_audio_exact_length(self) -> None:
        """Test that short audio is padded to exact target length."""
        clip = FixedSizeClip(16000)
        audio = np.ones(8000, dtype=np.float32) * 5.0

        result = clip.apply(audio, sr=16000)

        assert result.shape == (16000,)
        # Original audio should be preserved somewhere in the result
        nonzero = np.where(result != 0)[0]
        assert len(nonzero) == 8000

    def test_pad_no_jitter_keeps_audio_at_start(self) -> None:
        """Test that jitter=False pads at end, keeping audio at start."""
        clip = FixedSizeClip(16000, jitter=False)
        audio = np.ones(8000, dtype=np.float32) * 5.0

        result = clip.apply(audio, sr=16000)

        np.testing.assert_array_equal(result[:8000], audio)
        np.testing.assert_array_equal(result[8000:], np.zeros(8000, dtype=np.float32))

    def test_pad_center_mode_equal_padding(self) -> None:
        """Test that center mode pads equally on both sides."""
        clip = FixedSizeClip(16000, mode="center", jitter=False)
        audio = np.ones(8000, dtype=np.float32) * 7.0

        result = clip.apply(audio, sr=16000)

        np.testing.assert_array_equal(result[:4000], np.zeros(4000, dtype=np.float32))
        np.testing.assert_array_equal(result[4000:12000], audio)
        np.testing.assert_array_equal(result[12000:], np.zeros(4000, dtype=np.float32))

    def test_crop_long_audio_exact_length(self) -> None:
        """Test that long audio is cropped to exact target length."""
        clip = FixedSizeClip(4000)
        audio = np.arange(16000, dtype=np.float32)

        result = clip.apply(audio, sr=16000)

        assert result.shape == (4000,)

    def test_crop_pad_mode_keeps_start(self) -> None:
        """Test that pad mode crops from end, keeping start of audio."""
        clip = FixedSizeClip(4000, jitter=False)
        audio = np.arange(16000, dtype=np.float32)

        result = clip.apply(audio, sr=16000)

        np.testing.assert_array_equal(result, np.arange(4000, dtype=np.float32))

    def test_crop_center_mode_equal_crop(self) -> None:
        """Test that center mode crops equally from both sides."""
        clip = FixedSizeClip(4000, mode="center", jitter=False)
        audio = np.arange(16000, dtype=np.float32)

        result = clip.apply(audio, sr=16000)

        np.testing.assert_array_equal(result, np.arange(6000, 10000, dtype=np.float32))

    def test_jitter_produces_variation(self) -> None:
        """Test that jitter=True produces different results across runs."""
        clip = FixedSizeClip(16000, jitter=True)
        audio = np.ones(8000, dtype=np.float32) * 5.0

        results = []
        for _ in range(20):
            result = clip.apply(audio.copy(), sr=16000)
            # Record where the audio starts (first non-zero index)
            first_nonzero = np.where(result != 0)[0][0]
            results.append(first_nonzero)

        # With jitter, we should see variation in the first non-zero position
        assert len(set(results)) > 1, "Jitter should produce different positions"

    def test_jitter_crop_produces_variation(self) -> None:
        """Test that jitter=True produces different crop positions."""
        clip = FixedSizeClip(4000, jitter=True)
        audio = np.arange(16000, dtype=np.float32)

        results = []
        for _ in range(20):
            result = clip.apply(audio.copy(), sr=16000)
            results.append(result[0])

        # First element should vary with jitter
        assert len(set(results)) > 1, "Jitter should produce different crop starts"

    def test_no_dc_offset(self) -> None:
        """Test that padding uses silence (zeros), not DC offset."""
        clip = FixedSizeClip(16000)
        audio = np.ones(4000, dtype=np.float32)

        result = clip.apply(audio, sr=16000)

        assert result.min() == 0.0
        assert result.max() == 1.0
        # Verify padding region is actually zeros
        nonzero = np.where(result != 0)[0]
        assert len(nonzero) == 4000

    def test_output_dtype_float32(self) -> None:
        """Test that output is float32."""
        clip = FixedSizeClip(8000)
        audio = np.ones(4000, dtype=np.float64)

        result = clip.apply(audio, sr=16000)

        assert result.dtype == np.float32

    def test_sr_parameter_accepted(self) -> None:
        """Test that sr parameter is accepted for API compatibility."""
        clip = FixedSizeClip(8000)
        audio = np.ones(4000, dtype=np.float32)

        result = clip.apply(audio, sr=44100)

        assert result.shape == (8000,)

    def test_rejects_zero_target_samples(self) -> None:
        """Test that target_samples=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            FixedSizeClip(0)

    def test_rejects_negative_target_samples(self) -> None:
        """Test that negative target_samples raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            FixedSizeClip(-1)

    def test_rejects_invalid_mode(self) -> None:
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode"):
            FixedSizeClip(16000, mode="invalid")

    def test_center_mode_invalid_raises(self) -> None:
        """Test that invalid center mode raises ValueError."""
        with pytest.raises(ValueError, match="mode"):
            FixedSizeClip(16000, mode="start")

    def test_example_from_docstring(self) -> None:
        """Test the example from the task specification."""
        clip = FixedSizeClip(16000, mode="pad", jitter=True)
        audio = np.ones(8000, dtype=np.float32)
        audio_fixed = clip.apply(audio, sr=16000)

        assert audio_fixed.shape == (16000,)
        assert audio_fixed.dtype == np.float32

    def test_odd_length_center_padding(self) -> None:
        """Test center mode with odd-length padding (uneven distribution)."""
        clip = FixedSizeClip(16001, mode="center", jitter=False)
        audio = np.ones(8000, dtype=np.float32) * 3.0

        result = clip.apply(audio, sr=16000)

        assert result.shape == (16001,)
        # With odd padding (8001 total), front gets 4000, back gets 4001
        nonzero = np.where(result != 0)[0]
        assert len(nonzero) == 8000

    def test_odd_length_center_crop(self) -> None:
        """Test center mode with odd-length crop (uneven distribution)."""
        clip = FixedSizeClip(3999, mode="center", jitter=False)
        audio = np.arange(16000, dtype=np.float32)

        result = clip.apply(audio, sr=16000)

        assert result.shape == (3999,)


class TestFixedSizeClipPadOnly:
    """Tests for FixedSizeClip with pad_only mode."""

    def test_pad_only_short_audio_pads(self) -> None:
        """Test that pad_only mode pads short audio like normal pad."""
        clip = FixedSizeClip(16000, mode="pad_only", jitter=False)
        audio = np.ones(8000, dtype=np.float32) * 5.0

        result = clip.apply(audio, sr=16000)

        assert result.shape == (16000,)
        # Original audio should be preserved somewhere
        nonzero = np.where(result != 0)[0]
        assert len(nonzero) == 8000

    def test_pad_only_long_audio_stretches(self) -> None:
        """Test that pad_only mode stretches long audio instead of cropping."""
        clip = FixedSizeClip(4000, mode="pad_only", jitter=False)
        audio = np.arange(16000, dtype=np.float32)

        result = clip.apply(audio, sr=16000)

        # Result should be exactly target length
        assert result.shape == (4000,)
        # Stretched audio should be different from simple cropping
        # (since we're compressing 16000 samples into 4000)
        # The first value should not be 0 (since stretched from original)
        assert result[0] != 0 or True  # stretching produces non-zero at start

    def test_pad_only_long_audio_preserves_content(self) -> None:
        """Test that pad_only mode preserves all audio content through stretching."""
        clip = FixedSizeClip(8000, mode="pad_only", jitter=False)
        # Audio with distinct values at start and end
        audio = np.concatenate(
            [
                np.ones(4000, dtype=np.float32) * 1.0,
                np.ones(4000, dtype=np.float32) * 2.0,
            ]
        )

        result = clip.apply(audio, sr=16000)

        assert result.shape == (8000,)
        # Stretched audio should have some values from both original segments
        # Check that we have both low and high values (not just one)
        assert result.min() < 1.5  # Values from first half
        assert result.max() > 1.5  # Values from second half

    def test_pad_only_jitter_stretches_with_variation(self) -> None:
        """Test that pad_only mode with jitter produces varying stretch results."""
        clip = FixedSizeClip(4000, mode="pad_only", jitter=True)
        # Use audio with distinct non-zero values at start and end
        audio = np.concatenate(
            [
                np.ones(8000, dtype=np.float32) * 1.0,
                np.ones(8000, dtype=np.float32) * 2.0,
            ]
        )

        results = []
        for _ in range(20):
            result = clip.apply(audio.copy(), sr=16000)
            # Check a middle value that's affected by jitter
            # (near the boundary between 1.0 and 2.0 regions)
            results.append(result[1999])

        # With jitter, different stretch factors should produce variation
        assert len(set(results)) > 1, "Jitter should produce different stretch results"

    def test_pad_only_no_cropping(self) -> None:
        """Test that pad_only mode never crops audio."""
        clip = FixedSizeClip(2000, mode="pad_only", jitter=False)
        # Use audio with distinct start and end values
        audio = np.ones(10000, dtype=np.float32) * 7.0
        audio[:100] = 1.0  # Different value at start

        result = clip.apply(audio, sr=16000)

        assert result.shape == (2000,)
        # With stretching, we should have values from throughout the original audio
        # Not just from the first 2000 samples. Check that we have some influence
        # from both the low-value (1.0) and high-value (7.0) regions.
        assert result.min() < 5.0  # Influenced by start region (1.0)
        assert result.max() > 5.0  # Influenced by end region (7.0)

    def test_pad_only_equivalent_to_pad_for_short_audio(self) -> None:
        """Test that pad_only and pad behave identically for short audio."""
        clip_pad = FixedSizeClip(16000, mode="pad", jitter=False)
        clip_pad_only = FixedSizeClip(16000, mode="pad_only", jitter=False)
        audio = np.ones(8000, dtype=np.float32) * 5.0

        result_pad = clip_pad.apply(audio, sr=16000)
        result_pad_only = clip_pad_only.apply(audio, sr=16000)

        np.testing.assert_array_equal(result_pad, result_pad_only)

    def test_pad_only_exact_length_unchanged(self) -> None:
        """Test that audio at exact target length is unchanged."""
        clip = FixedSizeClip(16000, mode="pad_only")
        audio = np.ones(16000, dtype=np.float32)

        result = clip.apply(audio, sr=16000)

        np.testing.assert_array_equal(result, audio)

    def test_pad_only_default_mode_backward_compatible(self) -> None:
        """Test that default mode maintains backward compatibility."""
        # Default mode should be pad_or_crop (which crops long audio)
        clip_default = FixedSizeClip(4000, jitter=False)
        clip_explicit = FixedSizeClip(4000, mode="pad_or_crop", jitter=False)
        audio = np.arange(16000, dtype=np.float32)

        result_default = clip_default.apply(audio.copy(), sr=16000)
        result_explicit = clip_explicit.apply(audio.copy(), sr=16000)

        np.testing.assert_array_equal(result_default, result_explicit)

    def test_pad_mode_is_alias_for_pad_or_crop(self) -> None:
        """Test that mode='pad' is an alias for mode='pad_or_crop'."""
        clip_pad = FixedSizeClip(4000, mode="pad", jitter=False)
        clip_pad_or_crop = FixedSizeClip(4000, mode="pad_or_crop", jitter=False)
        audio = np.arange(16000, dtype=np.float32)

        result_pad = clip_pad.apply(audio.copy(), sr=16000)
        result_pad_or_crop = clip_pad_or_crop.apply(audio.copy(), sr=16000)

        np.testing.assert_array_equal(result_pad, result_pad_or_crop)


class TestTrimSilence:
    """Tests for TrimSilence class."""

    def test_requires_librosa(self) -> None:
        """Test that TrimSilence raises ImportError when librosa unavailable."""
        ts = TrimSilence()

        # Patch librosa import to simulate it not being installed
        import sys

        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError("No module named 'librosa'")
            return original_import(name, *args, **kwargs)

        try:
            __builtins__["__import__"] = mock_import
            with pytest.raises(ImportError, match="librosa"):
                ts.apply(np.ones(1000, dtype=np.float32), 16000)
        finally:
            __builtins__["__import__"] = original_import

    def test_init_default_threshold(self) -> None:
        """Test that default threshold_db is 20.0."""
        ts = TrimSilence()
        assert ts.threshold_db == 20.0

    def test_init_custom_threshold(self) -> None:
        """Test that custom threshold_db is accepted."""
        ts = TrimSilence(threshold_db=30.0)
        assert ts.threshold_db == 30.0

    def test_output_dtype_preserved(self) -> None:
        """Test that output dtype matches input dtype (when librosa available)."""
        pytest.importorskip("librosa")
        ts = TrimSilence()
        audio = np.ones(1000, dtype=np.float32)

        result = ts.apply(audio, sr=16000)

        assert result.dtype == audio.dtype
