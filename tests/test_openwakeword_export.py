"""Tests for openWakeWord numpy export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.export.openwakeword import (
    OpenWakeWordExportError,
    export_to_numpy,
    validate_export,
)


def _create_test_audio(path: Path, duration_samples: int) -> Path:
    """Create a test audio file."""
    audio = np.random.randn(duration_samples).astype(np.float32) * 0.1
    sf.write(path, audio, 16000, subtype="PCM_16")
    return path


class TestExportToNumpy:
    """Tests for export_to_numpy function."""

    def test_export_variable_length_raw(self, tmp_path: Path) -> None:
        """Test exporting raw audio with variable lengths (padded to max)."""
        # Create manifest with different audio lengths
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        audio1 = _create_test_audio(audio_dir / "audio1.wav", 8000)  # 0.5s
        audio2 = _create_test_audio(audio_dir / "audio2.wav", 16000)  # 1.0s
        audio3 = _create_test_audio(audio_dir / "audio3.wav", 24000)  # 1.5s

        manifest = Manifest(
            [
                ManifestEntry(
                    path=str(audio1), label=1, text="test 1", duration_ms=500, sample_rate=16000
                ),
                ManifestEntry(
                    path=str(audio2), label=0, text="test 2", duration_ms=1000, sample_rate=16000
                ),
                ManifestEntry(
                    path=str(audio3), label=1, text="test 3", duration_ms=1500, sample_rate=16000
                ),
            ]
        )

        output_dir = tmp_path / "output"
        X_path, y_path = export_to_numpy(manifest, output_dir, split="train")

        # Verify files exist
        assert X_path.exists()
        assert y_path.exists()

        # Load and verify shapes
        X = np.load(X_path)
        y = np.load(y_path)

        assert X.shape[0] == 3  # 3 samples
        assert y.shape == (3,)
        np.testing.assert_array_equal(y, [1, 0, 1])

        # Variable length: padded to max (24000 samples)
        assert X.shape[1] == 24000
        assert X.dtype == np.float32

    def test_export_fixed_length_raw(self, tmp_path: Path) -> None:
        """Test exporting raw audio with fixed length padding."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # Create audio of different lengths
        audio1 = _create_test_audio(audio_dir / "audio1.wav", 8000)  # shorter
        audio2 = _create_test_audio(audio_dir / "audio2.wav", 24000)  # longer

        manifest = Manifest(
            [
                ManifestEntry(
                    path=str(audio1), label=1, text="short", duration_ms=500, sample_rate=16000
                ),
                ManifestEntry(
                    path=str(audio2), label=0, text="long", duration_ms=1500, sample_rate=16000
                ),
            ]
        )

        output_dir = tmp_path / "output"
        fixed_length = 16000  # 1 second
        X_path, y_path = export_to_numpy(
            manifest, output_dir, split="train", fixed_length=fixed_length
        )

        # Load and verify
        X = np.load(X_path)
        y = np.load(y_path)

        assert X.shape == (2, fixed_length)
        assert y.shape == (2,)
        np.testing.assert_array_equal(y, [1, 0])

        # Both should be exactly 16000 samples
        assert X[0].shape == (fixed_length,)
        assert X[1].shape == (fixed_length,)

    def test_export_fixed_length_longer_than_all(self, tmp_path: Path) -> None:
        """Test that padding works when all audio is shorter than fixed_length."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # Create short audio
        audio = _create_test_audio(audio_dir / "audio.wav", 4000)  # 0.25s

        manifest = Manifest(
            [
                ManifestEntry(
                    path=str(audio), label=1, text="short", duration_ms=250, sample_rate=16000
                ),
            ]
        )

        output_dir = tmp_path / "output"
        fixed_length = 16000  # 1 second
        X_path, y_path = export_to_numpy(
            manifest, output_dir, split="train", fixed_length=fixed_length
        )

        X = np.load(X_path)
        assert X.shape == (1, fixed_length)
        # Padded portion should be zeros
        assert np.all(X[0][4000:] == 0)

    def test_export_fixed_length_shorter_than_all(self, tmp_path: Path) -> None:
        """Test that cropping works when all audio is longer than fixed_length."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # Create long audio
        audio = _create_test_audio(audio_dir / "audio.wav", 32000)  # 2s

        manifest = Manifest(
            [
                ManifestEntry(
                    path=str(audio), label=1, text="long", duration_ms=2000, sample_rate=16000
                ),
            ]
        )

        output_dir = tmp_path / "output"
        fixed_length = 16000  # 1 second
        X_path, y_path = export_to_numpy(
            manifest, output_dir, split="train", fixed_length=fixed_length
        )

        X = np.load(X_path)
        assert X.shape == (1, fixed_length)

    def test_export_mel_format(self, tmp_path: Path) -> None:
        """Test exporting mel-spectrogram features."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        audio = _create_test_audio(audio_dir / "audio.wav", 16000)  # 1 second

        manifest = Manifest(
            [
                ManifestEntry(
                    path=str(audio), label=1, text="test", duration_ms=1000, sample_rate=16000
                ),
            ]
        )

        output_dir = tmp_path / "output"
        X_path, y_path = export_to_numpy(manifest, output_dir, split="train", format="mel")

        X = np.load(X_path)
        y = np.load(y_path)

        # Mel spectrogram: (n_samples, time_steps, n_mels)
        assert X.shape[2] == 96  # n_mels
        assert X.shape[0] == 1  # 1 sample
        assert X.shape[1] >= 1  # At least 1 time step
        assert y.shape == (1,)
        assert X.dtype == np.float32

    def test_export_mel_fixed_length(self, tmp_path: Path) -> None:
        """Test exporting mel features with fixed time steps."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        audio = _create_test_audio(audio_dir / "audio.wav", 16000)  # 1 second

        manifest = Manifest(
            [
                ManifestEntry(
                    path=str(audio), label=1, text="test", duration_ms=1000, sample_rate=16000
                ),
            ]
        )

        output_dir = tmp_path / "output"
        # 160 samples = 1 second at 160 hop length
        fixed_time_steps = 100
        X_path, y_path = export_to_numpy(
            manifest, output_dir, split="train", format="mel", fixed_length=fixed_time_steps
        )

        X = np.load(X_path)
        # Shape: (1, time_steps, n_mels) - mel features transposed
        assert X.shape == (1, fixed_time_steps, 96)

    def test_export_custom_split_name(self, tmp_path: Path) -> None:
        """Test exporting with custom split name."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        audio = _create_test_audio(audio_dir / "audio.wav", 16000)

        manifest = Manifest(
            [
                ManifestEntry(
                    path=str(audio), label=1, text="test", duration_ms=1000, sample_rate=16000
                ),
            ]
        )

        output_dir = tmp_path / "output"
        X_path, y_path = export_to_numpy(manifest, output_dir, split="validation")

        assert X_path.name == "X_validation.npy"
        assert y_path.name == "y_validation.npy"

    def test_export_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that export creates output directory if it doesn't exist."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        audio = _create_test_audio(audio_dir / "audio.wav", 16000)

        manifest = Manifest(
            [
                ManifestEntry(
                    path=str(audio), label=1, text="test", duration_ms=1000, sample_rate=16000
                ),
            ]
        )

        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        X_path, y_path = export_to_numpy(manifest, output_dir, split="train", fixed_length=16000)

        assert output_dir.exists()
        assert X_path.exists()
        assert y_path.exists()


class TestValidateExport:
    """Tests for validate_export function."""

    def test_validate_valid_export(self, tmp_path: Path) -> None:
        """Test validating a valid export."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        audio = _create_test_audio(audio_dir / "audio.wav", 16000)

        manifest = Manifest(
            [
                ManifestEntry(
                    path=str(audio), label=1, text="test", duration_ms=1000, sample_rate=16000
                ),
                ManifestEntry(
                    path=str(audio), label=0, text="neg", duration_ms=1000, sample_rate=16000
                ),
            ]
        )

        output_dir = tmp_path / "output"
        export_to_numpy(manifest, output_dir, split="train", fixed_length=16000)

        result = validate_export(output_dir, split="train")

        assert "X" in result
        assert "y" in result
        assert result["X"].shape[0] == 2
        assert result["y"].shape == (2,)

    def test_validate_missing_x_file(self, tmp_path: Path) -> None:
        """Test that missing X file raises error."""
        y_path = tmp_path / "y_train.npy"
        np.save(y_path, np.array([1, 0]))

        with pytest.raises(OpenWakeWordExportError, match="X file not found"):
            validate_export(tmp_path, split="train")

    def test_validate_missing_y_file(self, tmp_path: Path) -> None:
        """Test that missing y file raises error."""
        X_path = tmp_path / "X_train.npy"
        np.save(X_path, np.array([[1, 2, 3], [4, 5, 6]]))

        with pytest.raises(OpenWakeWordExportError, match="y file not found"):
            validate_export(tmp_path, split="train")

    def test_validate_invalid_labels(self, tmp_path: Path) -> None:
        """Test that invalid labels raise error."""
        X_path = tmp_path / "X_train.npy"
        y_path = tmp_path / "y_train.npy"
        np.save(X_path, np.array([[1, 2, 3], [4, 5, 6]]))
        np.save(y_path, np.array([2, 3]))  # Invalid labels

        with pytest.raises(OpenWakeWordExportError, match="Labels must be 0 or 1"):
            validate_export(tmp_path, split="train")

    def test_validate_mismatched_lengths(self, tmp_path: Path) -> None:
        """Test that mismatched X and y lengths raise error."""
        X_path = tmp_path / "X_train.npy"
        y_path = tmp_path / "y_train.npy"
        np.save(X_path, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))  # 3 samples
        np.save(y_path, np.array([1, 0]))  # 2 labels

        with pytest.raises(OpenWakeWordExportError, match="doesn't match"):
            validate_export(tmp_path, split="train")


class TestExportErrors:
    """Tests for export error handling."""

    def test_export_empty_manifest_raises(self, tmp_path: Path) -> None:
        """Test that empty manifest raises error."""
        manifest = Manifest()

        with pytest.raises(OpenWakeWordExportError, match="empty"):
            export_to_numpy(manifest, tmp_path / "output", split="train")

    def test_export_nonexistent_audio_raises(self, tmp_path: Path) -> None:
        """Test that nonexistent audio file raises error."""
        manifest = Manifest(
            [
                ManifestEntry(
                    path=str(tmp_path / "nonexistent.wav"),
                    label=1,
                    text="missing",
                    duration_ms=1000,
                    sample_rate=16000,
                ),
            ]
        )

        with pytest.raises(OpenWakeWordExportError, match="Failed to export"):
            export_to_numpy(manifest, tmp_path / "output", split="train")
