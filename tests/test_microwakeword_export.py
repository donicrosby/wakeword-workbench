"""Tests for microWakeWord export module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.export.microwakeword import (
    MicroWakeWordExportError,
    compute_mel_spectrogram,
    export_to_mmap,
    export_with_features,
    load_mmap,
    validate_mmap,
)


def create_audio_file(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    """Create a test audio file."""
    num_samples = int(sample_rate * duration_sec)
    audio = np.random.randn(num_samples).astype(np.float32) * 0.1
    sf.write(path, audio, sample_rate, subtype="PCM_16")


class TestComputeMelSpectrogram:
    """Tests for compute_mel_spectrogram function."""

    def test_basic_mel_spectrogram(self) -> None:
        """Test basic mel-spectrogram computation."""
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1

        features = compute_mel_spectrogram(audio, sample_rate=sample_rate)

        assert features.dtype == np.float32
        assert features.shape[0] == 40  # n_mels

    def test_different_n_mels(self) -> None:
        """Test different number of mel bins."""
        sample_rate = 16000
        duration = 0.5
        num_samples = int(sample_rate * duration)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1

        features = compute_mel_spectrogram(audio, sample_rate=sample_rate, n_mels=80)

        assert features.shape[0] == 80

    def test_hop_length_affects_frames(self) -> None:
        """Test that different hop lengths produce different frame counts."""
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1

        features_short = compute_mel_spectrogram(audio, hop_length=160)
        features_long = compute_mel_spectrogram(audio, hop_length=480)

        # Longer hop length = fewer frames
        assert features_short.shape[1] > features_long.shape[1]


class TestExportToMmap:
    """Tests for export_to_mmap function."""

    def test_export_single_positive(self, tmp_path: Path) -> None:
        """Test exporting a single positive sample."""
        # Create audio file
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        audio_path = audio_dir / "positive.wav"
        create_audio_file(audio_path, duration_sec=1.0)

        # Create manifest
        manifest = Manifest(
            [
                ManifestEntry(
                    path="positive.wav",
                    label=1,
                    text="hey test",
                    duration_ms=1000,
                )
            ]
        )

        # Export
        output_dir = tmp_path / "output"
        result = export_to_mmap(manifest, output_dir, split="train", audio_dir=audio_dir)

        # Verify files created
        assert result["data"].exists()
        assert result["indices"].exists()
        assert result["labels"].exists()

        # Verify shapes
        data, indices, labels = load_mmap(result["data"], result["indices"], result["labels"])
        assert indices.shape == (1, 2)
        assert labels.shape == (1,)
        assert labels[0] == 1

    def test_export_multiple_samples(self, tmp_path: Path) -> None:
        """Test exporting multiple samples with different labels."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # Create audio files with different durations
        create_audio_file(audio_dir / "pos1.wav", duration_sec=1.0)
        create_audio_file(audio_dir / "pos2.wav", duration_sec=0.5)
        create_audio_file(audio_dir / "neg1.wav", duration_sec=1.5)
        create_audio_file(audio_dir / "neg2.wav", duration_sec=0.8)

        # Create manifest
        manifest = Manifest(
            [
                ManifestEntry(path="pos1.wav", label=1, text="test", duration_ms=1000),
                ManifestEntry(path="pos2.wav", label=1, text="test", duration_ms=500),
                ManifestEntry(path="neg1.wav", label=0, text="test", duration_ms=1500),
                ManifestEntry(path="neg2.wav", label=0, text="test", duration_ms=800),
            ]
        )

        # Export
        output_dir = tmp_path / "output"
        result = export_to_mmap(manifest, output_dir, split="train", audio_dir=audio_dir)

        # Verify
        data, indices, labels = load_mmap(result["data"], result["indices"], result["labels"])
        assert indices.shape == (4, 2)
        assert labels.shape == (4,)
        np.testing.assert_array_equal(labels, [1, 1, 0, 0])

        # Verify indices are contiguous and valid
        expected_total_samples = 16000 * (1.0 + 0.5 + 1.5 + 0.8)
        assert len(data) == expected_total_samples

        # Check start/end indices
        np.testing.assert_array_equal(indices[0], [0, 16000])  # 1.0 sec
        np.testing.assert_array_equal(indices[1], [16000, 24000])  # 0.5 sec
        np.testing.assert_array_equal(indices[2], [24000, 48000])  # 1.5 sec

    def test_export_empty_manifest(self, tmp_path: Path) -> None:
        """Test exporting empty manifest creates empty files."""
        manifest = Manifest()

        output_dir = tmp_path / "output"
        result = export_to_mmap(manifest, output_dir, split="train")

        # Verify files created
        assert result["data"].exists()
        assert result["indices"].exists()
        assert result["labels"].exists()

        # Verify shapes
        data, indices, labels = load_mmap(result["data"], result["indices"], result["labels"])
        assert len(indices) == 0
        assert len(labels) == 0

    def test_export_missing_audio_file_raises(self, tmp_path: Path) -> None:
        """Test that missing audio file raises error."""
        manifest = Manifest(
            [ManifestEntry(path="nonexistent.wav", label=1, text="test", duration_ms=1000)]
        )

        output_dir = tmp_path / "output"
        with pytest.raises(MicroWakeWordExportError, match="Audio file not found"):
            export_to_mmap(manifest, output_dir, split="train")

    def test_export_creates_correct_split_name(self, tmp_path: Path) -> None:
        """Test that split name is used in output files."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        create_audio_file(audio_dir / "test.wav", duration_sec=0.5)

        manifest = Manifest([ManifestEntry(path="test.wav", label=0, text="test", duration_ms=500)])

        output_dir = tmp_path / "output"

        # Export with different splits
        export_to_mmap(manifest, output_dir, split="train", audio_dir=audio_dir)
        assert (output_dir / "train_data.mmap").exists()
        assert (output_dir / "train_indices.npy").exists()
        assert (output_dir / "train_labels.npy").exists()

        export_to_mmap(manifest, output_dir, split="val", audio_dir=audio_dir)
        assert (output_dir / "val_data.mmap").exists()
        assert (output_dir / "val_indices.npy").exists()
        assert (output_dir / "val_labels.npy").exists()


class TestExportWithFeatures:
    """Tests for export_with_features function."""

    def test_export_with_mel_features(self, tmp_path: Path) -> None:
        """Test exporting with mel-spectrogram features."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        create_audio_file(audio_dir / "test.wav", duration_sec=1.0)

        manifest = Manifest(
            [ManifestEntry(path="test.wav", label=1, text="test", duration_ms=1000)]
        )

        output_dir = tmp_path / "output"
        result = export_with_features(
            manifest, output_dir, split="train", audio_dir=audio_dir, n_mels=40
        )

        # Verify all files created
        assert result["data"].exists()
        assert result["indices"].exists()
        assert result["labels"].exists()
        assert result["durations"].exists()
        assert result["shape"].exists()

        # Load and verify feature data
        data = np.load(result["data"])
        assert data.ndim == 2
        assert data.shape[1] == 40  # n_mels

        # Verify feature shape: for 1 sec at 16kHz with hop_length=480, we get ~33 frames
        # (16000 / 480 = 33.33)
        # Shape should be (n_frames, 40)
        assert data.shape[1] == 40  # n_mels

        durations = np.load(result["durations"])
        assert len(durations) == 1

    def test_export_features_multiple_clips(self, tmp_path: Path) -> None:
        """Test exporting features with multiple clips."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        create_audio_file(audio_dir / "clip1.wav", duration_sec=0.5)
        create_audio_file(audio_dir / "clip2.wav", duration_sec=1.0)

        manifest = Manifest(
            [
                ManifestEntry(path="clip1.wav", label=1, text="test", duration_ms=500),
                ManifestEntry(path="clip2.wav", label=0, text="test", duration_ms=1000),
            ]
        )

        output_dir = tmp_path / "output"
        result = export_with_features(manifest, output_dir, split="train", audio_dir=audio_dir)

        data = np.load(result["data"])
        indices = np.load(result["indices"])
        labels = np.load(result["labels"])
        durations = np.load(result["durations"])

        # Verify data is 2D
        assert data.ndim == 2
        assert data.shape[1] == 40  # n_mels

        # Verify durations array
        assert len(durations) == 2
        assert durations[0] < durations[1]  # 0.5 sec has fewer frames than 1.0 sec


class TestLoadMmap:
    """Tests for load_mmap function."""

    def test_load_existing_files(self, tmp_path: Path) -> None:
        """Test loading valid mmap files."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        create_audio_file(audio_dir / "test.wav", duration_sec=1.0)

        manifest = Manifest(
            [ManifestEntry(path="test.wav", label=1, text="test", duration_ms=1000)]
        )

        output_dir = tmp_path / "output"
        result = export_to_mmap(manifest, output_dir, split="train", audio_dir=audio_dir)

        data, indices, labels = load_mmap(result["data"], result["indices"], result["labels"])

        assert isinstance(data, np.memmap)
        assert data.dtype == np.float32
        assert indices.shape == (1, 2)
        assert labels.shape == (1,)
        assert labels[0] == 1

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Test loading non-existent file raises error."""
        with pytest.raises(MicroWakeWordExportError, match="Data file not found"):
            load_mmap(
                tmp_path / "missing.mmap",
                tmp_path / "missing.npy",
                tmp_path / "missing.npy",
            )


class TestValidateMmap:
    """Tests for validate_mmap function."""

    def test_validate_valid_files(self, tmp_path: Path) -> None:
        """Test validation of valid mmap files."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        create_audio_file(audio_dir / "test.wav", duration_sec=1.0)

        manifest = Manifest(
            [
                ManifestEntry(path="test.wav", label=1, text="test", duration_ms=1000),
                ManifestEntry(path="test.wav", label=0, text="test", duration_ms=1000),
            ]
        )

        output_dir = tmp_path / "output"
        result = export_to_mmap(manifest, output_dir, split="train", audio_dir=audio_dir)

        validation = validate_mmap(
            result["data"],
            result["indices"],
            result["labels"],
            expected_count=2,
        )

        assert validation["files_exist"] is True
        assert validation["indices_shape"] is True
        assert validation["labels_shape"] is True
        assert validation["counts_match"] is True
        assert validation["expected_count"] is True
        assert validation["valid_ranges"] is True
        assert validation["valid_labels"] is True

    def test_validate_with_expected_count(self, tmp_path: Path) -> None:
        """Test validation with expected count check."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        create_audio_file(audio_dir / "test.wav", duration_sec=1.0)

        manifest = Manifest(
            [
                ManifestEntry(path="test.wav", label=1, text="test", duration_ms=1000),
            ]
        )

        output_dir = tmp_path / "output"
        result = export_to_mmap(manifest, output_dir, split="train", audio_dir=audio_dir)

        # Correct count
        validation = validate_mmap(
            result["data"],
            result["indices"],
            result["labels"],
            expected_count=1,
        )
        assert validation["expected_count"] is True

        # Wrong count
        validation = validate_mmap(
            result["data"],
            result["indices"],
            result["labels"],
            expected_count=5,
        )
        assert validation["expected_count"] is False

    def test_validate_missing_files(self, tmp_path: Path) -> None:
        """Test validation of missing files."""
        validation = validate_mmap(
            tmp_path / "missing.mmap",
            tmp_path / "missing.npy",
            tmp_path / "missing.npy",
        )
        assert validation["files_exist"] is False


class TestMmapDataIntegrity:
    """Tests for data integrity in mmap format."""

    def test_audio_data_preserved(self, tmp_path: Path) -> None:
        """Test that audio data is correctly preserved in export."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # Create specific audio pattern
        sample_rate = 16000
        duration = 0.5
        num_samples = int(sample_rate * duration)
        original_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, num_samples)).astype(
            np.float32
        )
        sf.write(audio_dir / "sine.wav", original_audio, sample_rate, subtype="PCM_16")

        manifest = Manifest([ManifestEntry(path="sine.wav", label=1, text="sine", duration_ms=500)])

        output_dir = tmp_path / "output"
        result = export_to_mmap(manifest, output_dir, split="train", audio_dir=audio_dir)

        # Load and verify
        data, indices, _ = load_mmap(result["data"], result["indices"], result["labels"])

        extracted = np.array(data[indices[0][0] : indices[0][1]])

        # Check length
        assert len(extracted) == len(original_audio)

        # Check values are approximately equal (allowing for small resampling differences)
        np.testing.assert_allclose(extracted[:100], original_audio[:100], rtol=1e-4, atol=1e-4)

    def test_multiple_clips_concatenated_correctly(self, tmp_path: Path) -> None:
        """Test that multiple clips are correctly concatenated."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # Create two distinct audio patterns
        sample_rate = 16000
        t1 = np.linspace(0, 0.5, int(sample_rate * 0.5))
        audio1 = np.sin(2 * np.pi * 440 * t1).astype(np.float32)
        sf.write(audio_dir / "tone1.wav", audio1, sample_rate, subtype="PCM_16")

        t2 = np.linspace(0, 0.25, int(sample_rate * 0.25))
        audio2 = np.sin(2 * np.pi * 880 * t2).astype(np.float32)
        sf.write(audio_dir / "tone2.wav", audio2, sample_rate, subtype="PCM_16")

        manifest = Manifest(
            [
                ManifestEntry(path="tone1.wav", label=1, text="440Hz", duration_ms=500),
                ManifestEntry(path="tone2.wav", label=0, text="880Hz", duration_ms=250),
            ]
        )

        output_dir = tmp_path / "output"
        result = export_to_mmap(manifest, output_dir, split="train", audio_dir=audio_dir)

        # Load and verify
        data, indices, labels = load_mmap(result["data"], result["indices"], result["labels"])

        # Extract each clip
        clip1 = np.array(data[indices[0][0] : indices[0][1]])
        clip2 = np.array(data[indices[1][0] : indices[1][1]])

        # Verify shapes
        assert len(clip1) == len(audio1)
        assert len(clip2) == len(audio2)

        # Verify data integrity (using allclose due to librosa resampling differences)
        np.testing.assert_allclose(clip1, audio1, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(clip2, audio2, rtol=1e-4, atol=1e-4)
