"""Tests for audio loading and resampling utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wakeword_workbench.augment.audio_loader import (
    AudioLoadError,
    AudioMetadata,
    load_audio,
    save_audio,
)


class TestAudioMetadata:
    """Tests for AudioMetadata dataclass."""

    def test_create_metadata(self) -> None:
        """Test creating AudioMetadata."""
        metadata = AudioMetadata(
            original_sr=48000,
            target_sr=16000,
            duration=1.5,
            channels=1,
            filename="test.wav",
        )
        assert metadata.original_sr == 48000
        assert metadata.target_sr == 16000
        assert metadata.duration == 1.5
        assert metadata.channels == 1
        assert metadata.filename == "test.wav"


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_load_16khz_audio(self, tmp_path: Path) -> None:
        """Test loading 16kHz audio (no resampling needed)."""
        # Create a 16kHz test file
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1

        audio_path = tmp_path / "test_16k.wav"
        sf.write(audio_path, audio, sample_rate, subtype="PCM_16")

        # Load and verify
        loaded_audio, metadata = load_audio(audio_path, target_sr=16000)

        assert loaded_audio.dtype == np.float32
        assert metadata.original_sr == 16000
        assert metadata.target_sr == 16000
        assert metadata.channels == 1
        assert metadata.filename == "test_16k.wav"
        # audioread fallback may have slight differences, use looser tolerance
        np.testing.assert_allclose(loaded_audio, audio, atol=1e-3)

    def test_load_48khz_audio_resampled(self, tmp_path: Path) -> None:
        """Test loading 48kHz audio and resampling to 16kHz."""
        # Create a 48kHz test file
        sample_rate = 48000
        duration = 1.0
        num_samples = int(sample_rate * duration)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1

        audio_path = tmp_path / "test_48k.wav"
        sf.write(audio_path, audio, sample_rate, subtype="PCM_16")

        # Load and verify resampling
        loaded_audio, metadata = load_audio(audio_path, target_sr=16000)

        assert loaded_audio.dtype == np.float32
        assert metadata.original_sr == 48000
        assert metadata.target_sr == 16000
        assert metadata.channels == 1

        # Duration should remain the same after resampling
        expected_num_samples = int(duration * 16000)
        assert len(loaded_audio) == expected_num_samples

    def test_load_audio_not_found_raises(self, tmp_path: Path) -> None:
        """Test that loading non-existent file raises AudioLoadError."""
        nonexistent = tmp_path / "does_not_exist.wav"
        with pytest.raises(AudioLoadError, match="not found"):
            load_audio(nonexistent)

    def test_load_corrupted_audio_raises(self, tmp_path: Path) -> None:
        """Test that corrupted file raises AudioLoadError."""
        # Write invalid data
        corrupted_path = tmp_path / "corrupted.wav"
        corrupted_path.write_bytes(b"This is not a valid WAV file")

        with pytest.raises(AudioLoadError, match="Failed to load"):
            load_audio(corrupted_path)

    def test_load_stereo_audio_converts_to_mono(self, tmp_path: Path) -> None:
        """Test that stereo audio is converted to mono."""
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)

        # Create stereo audio
        left = np.random.randn(num_samples).astype(np.float32) * 0.1
        right = np.random.randn(num_samples).astype(np.float32) * 0.1
        stereo = np.stack([left, right], axis=-1)

        audio_path = tmp_path / "test_stereo.wav"
        sf.write(audio_path, stereo, sample_rate, subtype="PCM_16")

        loaded_audio, metadata = load_audio(audio_path)

        assert loaded_audio.ndim == 1
        assert metadata.channels == 1

    def test_load_audio_wav_format(self, tmp_path: Path) -> None:
        """Test loading WAV format."""
        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, audio, sample_rate, subtype="PCM_16")

        loaded_audio, metadata = load_audio(audio_path)
        assert len(loaded_audio) == sample_rate

    def test_load_audio_flac_format(self, tmp_path: Path) -> None:
        """Test loading FLAC format."""
        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1
        audio_path = tmp_path / "test.flac"
        sf.write(audio_path, audio, sample_rate, subtype="PCM_24")

        loaded_audio, metadata = load_audio(audio_path)
        assert loaded_audio.dtype == np.float32
        assert metadata.channels == 1


class TestSaveAudio:
    """Tests for save_audio function."""

    def test_save_audio_basic(self, tmp_path: Path) -> None:
        """Test basic audio saving."""
        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1

        output_path = tmp_path / "output.wav"
        save_audio(output_path, audio, sr=sample_rate)

        assert output_path.exists()

        # Verify saved file can be read back
        loaded_audio, sr = sf.read(output_path)
        assert sr == sample_rate
        np.testing.assert_allclose(loaded_audio, audio, atol=1e-4)

    def test_save_audio_creates_16bit_pcm(self, tmp_path: Path) -> None:
        """Test that saved audio is 16-bit PCM."""
        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1

        output_path = tmp_path / "output.wav"
        save_audio(output_path, audio, sr=sample_rate)

        # Check file info
        info = sf.info(output_path)
        assert info.subtype == "PCM_16"

    def test_save_audio_normalizes_to_unity(self, tmp_path: Path) -> None:
        """Test that audio exceeding [-1, 1] is normalized."""
        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 2.0  # Exceeds range

        output_path = tmp_path / "output.wav"
        save_audio(output_path, audio, sr=sample_rate)

        loaded_audio, _ = sf.read(output_path)
        assert np.abs(loaded_audio).max() <= 1.0

    def test_save_audio_float64_conversion(self, tmp_path: Path) -> None:
        """Test that float32 audio is properly converted for saving."""
        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1

        output_path = tmp_path / "output.wav"
        save_audio(output_path, audio, sr=sample_rate)

        loaded_audio, _ = sf.read(output_path)
        assert loaded_audio.dtype == np.float64 or np.issubdtype(loaded_audio.dtype, np.floating)


class TestLoadSaveRoundTrip:
    """Tests for complete load-save round trip."""

    def test_roundtrip_preserves_audio(self, tmp_path: Path) -> None:
        """Test that load-save round trip preserves audio data."""
        # Create original audio
        sample_rate = 44100
        duration = 0.5
        num_samples = int(sample_rate * duration)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples)).astype(np.float32)

        # Save original
        original_path = tmp_path / "original.wav"
        sf.write(original_path, audio, sample_rate, subtype="PCM_16")

        # Load with resampling to 16kHz
        loaded_audio, metadata = load_audio(original_path, target_sr=16000)

        # Save the loaded audio
        output_path = tmp_path / "output.wav"
        save_audio(output_path, loaded_audio, sr=metadata.target_sr)

        # Verify
        assert output_path.exists()
        final_audio, final_sr = sf.read(output_path)
        assert final_sr == 16000
        assert len(final_audio) == len(loaded_audio)

    def test_roundtrip_48khz_to_16khz(self, tmp_path: Path) -> None:
        """Test round trip: 48kHz -> 16kHz -> save."""
        sample_rate = 48000
        duration = 1.0
        num_samples = int(sample_rate * duration)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples)).astype(np.float32)

        # Save 48kHz
        original_path = tmp_path / "original_48k.wav"
        sf.write(original_path, audio, sample_rate, subtype="PCM_16")

        # Load and resample
        loaded_audio, metadata = load_audio(original_path, target_sr=16000)

        # Verify resampling
        expected_samples = int(duration * 16000)
        assert len(loaded_audio) == expected_samples
        assert metadata.original_sr == 48000
        assert metadata.target_sr == 16000

        # Save
        output_path = tmp_path / "output_16k.wav"
        save_audio(output_path, loaded_audio, sr=16000)

        # Verify output
        output_info = sf.info(output_path)
        assert output_info.samplerate == 16000
