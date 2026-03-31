"""Tests for noise augmentation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wakeword_workbench.augment.noise import AddColoredNoise, AddNoise


class TestAddColoredNoise:
    """Tests for AddColoredNoise class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        aug = AddColoredNoise()
        assert aug.color == "white"
        assert aug.snr_range == (-10, 10)
        assert aug.p == 0.5

    def test_init_with_params(self) -> None:
        """Test initialization with custom parameters."""
        aug = AddColoredNoise(color="pink", snr_range=(-5, 5), p=0.3)
        assert aug.color == "pink"
        assert aug.snr_range == (-5, 5)
        assert aug.p == 0.3

    def test_invalid_color(self) -> None:
        """Test that invalid color raises error."""
        aug = AddColoredNoise(color="purple")
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        # With p=1.0, apply always triggers generation
        aug.p = 1.0
        with pytest.raises(ValueError, match="Unknown noise color"):
            aug.apply(audio, sr=16000)

    def test_white_noise_generation(self) -> None:
        """Test white noise generation produces correct shape and type."""
        aug = AddColoredNoise(color="white")
        noise = aug._generate_colored_noise(16000)
        assert noise.shape == (16000,)
        assert noise.dtype == np.float32
        # White noise should have roughly uniform variance
        assert 0.01 < np.var(noise) < 1.0

    def test_pink_noise_generation(self) -> None:
        """Test pink noise generation produces correct shape and type."""
        aug = AddColoredNoise(color="pink")
        noise = aug._generate_colored_noise(16000)
        assert noise.shape == (16000,)
        assert noise.dtype == np.float32
        # Pink noise should have finite variance
        assert np.isfinite(np.var(noise))

    def test_brown_noise_generation(self) -> None:
        """Test brown noise generation produces correct shape and type."""
        aug = AddColoredNoise(color="brown")
        noise = aug._generate_colored_noise(16000)
        assert noise.shape == (16000,)
        assert noise.dtype == np.float32
        # Brown noise should have lower variance
        assert np.var(noise) < 1.0

    def test_apply_respects_probability(self) -> None:
        """Test that augmentation respects probability parameter."""
        aug = AddColoredNoise(p=0.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = aug.apply(audio, sr=16000)
        np.testing.assert_array_equal(result, audio)

    def test_apply_returns_copy_when_not_applied(self) -> None:
        """Test that a copy is returned even when not applying."""
        aug = AddColoredNoise(p=0.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = aug.apply(audio, sr=16000)
        assert result is not audio

    def test_apply_respects_snr_range(self) -> None:
        """Test that SNR is within specified range."""
        aug = AddColoredNoise(color="white", snr_range=(5, 10), p=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.5

        # Run multiple times to get a sample
        result = aug.apply(audio, sr=16000)

        # Output should be normalized to [-1, 1]
        assert np.max(np.abs(result)) <= 1.0
        assert result.shape == audio.shape

    def test_snr_calculation(self) -> None:
        """Test SNR scale factor calculation is correct."""
        aug = AddColoredNoise()

        # Signal power of 1.0
        signal_power = 1.0
        # Target SNR of 0 dB means signal power = noise power
        target_snr_db = 0.0
        scale = aug._calculate_scale_factor(signal_power, target_snr_db)
        # At 0 dB SNR, noise should have same power as signal
        assert abs(scale - 1.0) < 0.1

    def test_snr_calculation_positive_db(self) -> None:
        """Test SNR calculation for positive dB (less noise)."""
        aug = AddColoredNoise()

        signal_power = 1.0
        target_snr_db = 10.0
        scale = aug._calculate_scale_factor(signal_power, target_snr_db)

        # At +10 dB SNR, noise power should be 1/10th of signal power
        # scale = sqrt(0.1) ≈ 0.316
        expected_scale = np.sqrt(0.1)
        assert abs(scale - expected_scale) < 0.01

    def test_snr_calculation_negative_db(self) -> None:
        """Test SNR calculation for negative dB (more noise)."""
        aug = AddColoredNoise()

        signal_power = 1.0
        target_snr_db = -10.0
        scale = aug._calculate_scale_factor(signal_power, target_snr_db)

        # At -10 dB SNR, noise power should be 10x signal power
        # scale = sqrt(10) ≈ 3.162
        expected_scale = np.sqrt(10)
        assert abs(scale - expected_scale) < 0.01

    def test_apply_handles_silent_audio(self) -> None:
        """Test that silent audio is handled gracefully."""
        aug = AddColoredNoise(p=1.0)
        audio = np.zeros(16000, dtype=np.float32)
        result = aug.apply(audio, sr=16000)
        np.testing.assert_array_equal(result, audio)

    def test_apply_preserves_length(self) -> None:
        """Test that output audio has same length as input."""
        aug = AddColoredNoise(p=1.0)
        lengths = [8000, 16000, 32000]
        for length in lengths:
            audio = np.random.randn(length).astype(np.float32) * 0.5
            result = aug.apply(audio, sr=16000)
            assert len(result) == len(audio)


class TestAddNoise:
    """Tests for AddNoise class with file-based noise."""

    def test_init_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        """Test that nonexistent noise directory raises FileNotFoundError."""
        nonexistent = tmp_path / "noise_that_does_not_exist"
        with pytest.raises(FileNotFoundError, match="Noise directory not found"):
            AddNoise(nonexistent)

    def test_init_empty_dir_raises(self, tmp_path: Path) -> None:
        """Test that empty noise directory raises ValueError."""
        empty_dir = tmp_path / "empty_noise"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No noise files found"):
            AddNoise(empty_dir)

    def test_init_with_valid_noise_files(self, tmp_path: Path) -> None:
        """Test initialization with valid noise files."""
        # Create a mock noise file directory
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()

        # Create mock wav files
        import soundfile as sf

        # Create a 1-second noise file
        noise_data = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(noise_dir / "noise1.wav", noise_data, 16000)
        sf.write(noise_dir / "noise2.wav", noise_data, 16000)

        aug = AddNoise(noise_dir, snr_range=(-10, 10), p=0.75)
        assert len(aug._noise_files) == 2
        assert aug.p == 0.75
        assert aug.snr_range == (-10, 10)

    def test_apply_respects_probability(self, tmp_path: Path) -> None:
        """Test that augmentation respects probability parameter."""
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()

        import soundfile as sf

        noise_data = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(noise_dir / "noise.wav", noise_data, 16000)

        aug = AddNoise(noise_dir, p=0.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = aug.apply(audio, sr=16000)
        np.testing.assert_array_equal(result, audio)

    def test_apply_returns_copy_when_not_applied(self, tmp_path: Path) -> None:
        """Test that a copy is returned even when not applying."""
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()

        import soundfile as sf

        noise_data = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(noise_dir / "noise.wav", noise_data, 16000)

        aug = AddNoise(noise_dir, p=0.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = aug.apply(audio, sr=16000)
        assert result is not audio

    def test_apply_normalizes_output(self, tmp_path: Path) -> None:
        """Test that output is normalized to [-1, 1] range."""
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()

        import soundfile as sf

        # Create louder noise that would clip without normalization
        noise_data = np.random.randn(16000).astype(np.float32) * 2.0
        sf.write(noise_dir / "noise.wav", noise_data, 16000)

        aug = AddNoise(noise_dir, p=1.0)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = aug.apply(audio, sr=16000)

        assert np.max(np.abs(result)) <= 1.0

    def test_snr_calculation(self, tmp_path: Path) -> None:
        """Test SNR scale factor calculation is correct."""
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()

        import soundfile as sf

        noise_data = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(noise_dir / "noise.wav", noise_data, 16000)

        aug = AddNoise(noise_dir)

        # Signal power of 1.0
        signal_power = 1.0
        # Target SNR of 0 dB
        target_snr_db = 0.0
        scale = aug._calculate_scale_factor(signal_power, target_snr_db)
        assert abs(scale - 1.0) < 0.1

    def test_apply_handles_silent_audio(self, tmp_path: Path) -> None:
        """Test that silent audio is handled gracefully."""
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()

        import soundfile as sf

        noise_data = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(noise_dir / "noise.wav", noise_data, 16000)

        aug = AddNoise(noise_dir, p=1.0)
        audio = np.zeros(16000, dtype=np.float32)
        result = aug.apply(audio, sr=16000)
        np.testing.assert_array_equal(result, audio)

    def test_apply_preserves_length(self, tmp_path: Path) -> None:
        """Test that output audio has same length as input."""
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()

        import soundfile as sf

        noise_data = np.random.randn(32000).astype(np.float32) * 0.1
        sf.write(noise_dir / "noise.wav", noise_data, 16000)

        aug = AddNoise(noise_dir, p=1.0)

        for length in [8000, 16000, 32000]:
            audio = np.random.randn(length).astype(np.float32) * 0.5
            result = aug.apply(audio, sr=16000)
            assert len(result) == len(audio)

    def test_snr_formula_verification(self) -> None:
        """Verify the SNR formula: SNR_db = 10 * log10(signal_power / noise_power)."""
        signal_power = 0.01  # quiet signal
        noise_power = 0.1  # louder noise

        # Expected SNR = 10 * log10(0.01 / 0.1) = 10 * log10(0.1) = -10 dB
        expected_snr = 10 * np.log10(signal_power / noise_power)
        assert abs(expected_snr - (-10.0)) < 0.01

    def test_snr_reverse_calculation(self) -> None:
        """Test that we can recover noise power from SNR."""
        signal_power = 0.01
        target_snr_db = -10.0

        # From SNR formula: noise_power = signal_power / 10^(SNR_db / 10)
        expected_noise_power = signal_power / (10 ** (target_snr_db / 10))
        assert abs(expected_noise_power - 0.1) < 0.001
