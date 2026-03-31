"""MicroWakeWord format exporter."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from wakeword_workbench.augment.audio_loader import load_audio
from wakeword_workbench.dataset.metadata import Manifest


class MicroWakeWordExportError(Exception):
    """Raised when microWakeWord export fails."""

    pass


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 40,
    hop_length: int = 480,
    n_fft: int = 2048,
) -> np.ndarray:
    """Compute mel-spectrogram features from audio.

    Args:
        audio: Audio samples as float32 array in [-1, 1] range.
        sample_rate: Sample rate in Hz (default: 16000).
        n_mels: Number of mel filterbanks (default: 40).
        hop_length: Number of samples between frames (default: 480 = 30ms at 16kHz).
        n_fft: FFT window size (default: 2048).

    Returns:
        Mel-spectrogram features as float32 array of shape (n_mels, n_frames).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel.astype(np.float32)


def export_to_mmap(
    manifest: Manifest,
    output_dir: Path,
    split: str = "train",
    audio_dir: Path | None = None,
) -> dict[str, Path]:
    """Export manifest to microWakeWord Ragged Mmap format.

    Creates three files:
    - {split}_data.mmap: Concatenated raw audio bytes (float32)
    - {split}_indices.npy: Start/end indices for each clip (N x 2)
    - {split}_labels.npy: Labels for each clip (N,)

    Args:
        manifest: Manifest containing audio file entries.
        output_dir: Directory to write output files.
        split: Split name for output files (default: "train").
        audio_dir: Base directory for audio files in manifest (optional).

    Returns:
        Dictionary mapping file type to output Path.

    Raises:
        MicroWakeWordExportError: If export fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / f"{split}_data.mmap"
    indices_path = output_dir / f"{split}_indices.npy"
    labels_path = output_dir / f"{split}_labels.npy"

    entries = list(manifest)
    if len(entries) == 0:
        # Create empty files for empty manifest
        _write_empty_files(data_path, indices_path, labels_path)
        return {
            "data": data_path,
            "indices": indices_path,
            "labels": labels_path,
        }

    # Collect audio data and metadata
    audio_clips: list[np.ndarray] = []
    indices: list[tuple[int, int]] = []
    labels: list[int] = []

    for entry in entries:
        # Determine audio file path
        if audio_dir is not None:
            audio_path = audio_dir / entry.path
        else:
            audio_path = Path(entry.path)

        if not audio_path.exists():
            raise MicroWakeWordExportError(f"Audio file not found: {audio_path}")

        try:
            audio, _ = load_audio(audio_path, target_sr=16000)
        except Exception as e:
            raise MicroWakeWordExportError(f"Failed to load audio '{audio_path}': {e}") from e

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        start_idx = sum(len(clip) for clip in audio_clips)
        end_idx = start_idx + len(audio)

        audio_clips.append(audio)
        indices.append((start_idx, end_idx))
        labels.append(entry.label)

    # Concatenate all audio clips
    concatenated_audio = np.concatenate(audio_clips).astype(np.float32)

    # Write audio data directly (compatible with memory mapping by loader)
    try:
        concatenated_audio.tofile(data_path)
    except OSError as e:
        raise MicroWakeWordExportError(f"Failed to write data file: {e}") from e

    # Save indices as numpy array (N x 2)
    indices_array = np.array(indices, dtype=np.int64)
    try:
        np.save(indices_path, indices_array)
    except OSError as e:
        raise MicroWakeWordExportError(f"Failed to write indices file: {e}") from e

    # Save labels as numpy array
    labels_array = np.array(labels, dtype=np.int32)
    try:
        np.save(labels_path, labels_array)
    except OSError as e:
        raise MicroWakeWordExportError(f"Failed to write labels file: {e}") from e

    return {
        "data": data_path,
        "indices": indices_path,
        "labels": labels_path,
    }


def export_with_features(
    manifest: Manifest,
    output_dir: Path,
    split: str = "train",
    audio_dir: Path | None = None,
    n_mels: int = 40,
    hop_length: int = 480,
) -> dict[str, Path]:
    """Export manifest to microWakeWord format with mel-spectrogram features.

    Creates five files:
    - {split}_data.npy: Concatenated mel-spectrogram features (2D array)
    - {split}_indices.npy: Start/end indices for each clip (N x 2)
    - {split}_labels.npy: Labels for each clip (N,)
    - {split}_durations.npy: Number of frames per clip (N,)
    - {split}_shape.npy: Shape metadata for 2D feature array (n_frames, n_features)

    Args:
        manifest: Manifest containing audio file entries.
        output_dir: Directory to write output files.
        split: Split name for output files (default: "train").
        audio_dir: Base directory for audio files in manifest (optional).
        n_mels: Number of mel filterbanks (default: 40).
        hop_length: Hop length in samples (default: 480 = 30ms at 16kHz).

    Returns:
        Dictionary mapping file type to output Path.

    Raises:
        MicroWakeWordExportError: If export fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / f"{split}_data.npy"
    indices_path = output_dir / f"{split}_indices.npy"
    labels_path = output_dir / f"{split}_labels.npy"
    durations_path = output_dir / f"{split}_durations.npy"
    shape_path = output_dir / f"{split}_shape.npy"

    entries = list(manifest)
    if len(entries) == 0:
        _write_empty_feature_files(data_path, indices_path, labels_path, durations_path, shape_path)
        return {
            "data": data_path,
            "indices": indices_path,
            "labels": labels_path,
            "durations": durations_path,
            "shape": shape_path,
        }

    feature_clips: list[np.ndarray] = []
    indices: list[tuple[int, int]] = []
    labels: list[int] = []
    durations: list[int] = []

    for entry in entries:
        # Determine audio file path
        if audio_dir is not None:
            audio_path = audio_dir / entry.path
        else:
            audio_path = Path(entry.path)

        if not audio_path.exists():
            raise MicroWakeWordExportError(f"Audio file not found: {audio_path}")

        try:
            audio, _ = load_audio(audio_path, target_sr=16000)
        except Exception as e:
            raise MicroWakeWordExportError(f"Failed to load audio '{audio_path}': {e}") from e

        # Compute mel-spectrogram
        features = compute_mel_spectrogram(
            audio,
            sample_rate=16000,
            n_mels=n_mels,
            hop_length=hop_length,
        )
        # Transpose to get (time, n_mels) format
        features = features.T.astype(np.float32)

        start_idx = sum(len(clip) for clip in feature_clips)
        end_idx = start_idx + len(features)

        feature_clips.append(features)
        indices.append((start_idx, end_idx))
        labels.append(entry.label)
        durations.append(len(features))

    # Concatenate all features (each clip has shape (n_frames, n_mels))
    concatenated_features = np.concatenate(feature_clips, axis=0).astype(np.float32)

    # Save features as numpy array (preserves 2D shape)
    try:
        np.save(data_path, concatenated_features)
    except Exception as e:
        raise MicroWakeWordExportError(f"Failed to write data file: {e}") from e

    # Save indices
    indices_array = np.array(indices, dtype=np.int64)
    np.save(indices_path, indices_array)

    # Save labels
    labels_array = np.array(labels, dtype=np.int32)
    np.save(labels_path, labels_array)

    # Save durations
    durations_array = np.array(durations, dtype=np.int64)
    np.save(durations_path, durations_array)

    # Save shape metadata
    shape_array = np.array(concatenated_features.shape, dtype=np.int64)
    np.save(shape_path, shape_array)

    return {
        "data": data_path,
        "indices": indices_path,
        "labels": labels_path,
        "durations": durations_path,
        "shape": shape_path,
    }


def load_mmap(
    data_path: Path,
    indices_path: Path,
    labels_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load microWakeWord mmap format files.

    Args:
        data_path: Path to *_data.mmap file.
        indices_path: Path to *_indices.npy file.
        labels_path: Path to *_labels.npy file.

    Returns:
        Tuple of (data array, indices array, labels array).

    Raises:
        MicroWakeWordExportError: If files cannot be loaded.
    """
    if not data_path.exists():
        raise MicroWakeWordExportError(f"Data file not found: {data_path}")
    if not indices_path.exists():
        raise MicroWakeWordExportError(f"Indices file not found: {indices_path}")
    if not labels_path.exists():
        raise MicroWakeWordExportError(f"Labels file not found: {labels_path}")

    try:
        # Check if file is empty - can't mmap empty file
        if data_path.stat().st_size == 0:
            data = np.array([], dtype=np.float32)
        else:
            data = np.memmap(data_path, dtype=np.float32, mode="r")
    except Exception as e:
        raise MicroWakeWordExportError(f"Failed to load data: {e}") from e

    try:
        indices = np.load(indices_path)
    except Exception as e:
        raise MicroWakeWordExportError(f"Failed to load indices: {e}") from e

    try:
        labels = np.load(labels_path)
    except Exception as e:
        raise MicroWakeWordExportError(f"Failed to load labels: {e}") from e

    return data, indices, labels


def validate_mmap(
    data_path: Path,
    indices_path: Path,
    labels_path: Path,
    expected_count: int | None = None,
) -> dict[str, bool]:
    """Validate microWakeWord mmap format files.

    Args:
        data_path: Path to *_data.mmap file.
        indices_path: Path to *_indices.npy file.
        labels_path: Path to *_labels.npy file.
        expected_count: Expected number of clips (optional).

    Returns:
        Dictionary with validation results.
    """
    results: dict[str, bool] = {}

    try:
        data, indices, labels = load_mmap(data_path, indices_path, labels_path)
        results["files_exist"] = True
    except MicroWakeWordExportError:
        results["files_exist"] = False
        return results

    # Check shapes
    results["indices_shape"] = indices.ndim == 2 and indices.shape[1] == 2
    results["labels_shape"] = labels.ndim == 1

    # Check consistency
    n_clips = len(labels)
    results["counts_match"] = n_clips == len(indices)

    if expected_count is not None:
        results["expected_count"] = n_clips == expected_count

    # Check indices validity
    if indices.shape[0] > 0:
        start_indices = indices[:, 0]
        end_indices = indices[:, 1]
        results["valid_ranges"] = bool(
            np.all(start_indices >= 0)
            and np.all(end_indices <= len(data))
            and np.all(start_indices <= end_indices)
        )
    else:
        results["valid_ranges"] = True

    # Check labels values
    results["valid_labels"] = bool(np.all((labels == 0) | (labels == 1)))

    return results


def _write_empty_files(
    data_path: Path,
    indices_path: Path,
    labels_path: Path,
) -> None:
    """Write empty mmap format files for empty manifest."""
    # Write empty data file using tofile (creates a valid empty binary file)
    empty_audio = np.array([], dtype=np.float32)
    empty_audio.tofile(data_path)

    # Write empty arrays
    np.save(indices_path, np.array([], dtype=np.int64).reshape(0, 2))
    np.save(labels_path, np.array([], dtype=np.int32))


def _write_empty_feature_files(
    data_path: Path,
    indices_path: Path,
    labels_path: Path,
    durations_path: Path,
    shape_path: Path,
) -> None:
    """Write empty feature export files for empty manifest."""
    # Write empty 2D array (0, n_mels) - n_mels defaults to 40
    empty_features = np.zeros((0, 40), dtype=np.float32)
    np.save(data_path, empty_features)

    # Write empty arrays
    np.save(indices_path, np.array([], dtype=np.int64).reshape(0, 2))
    np.save(labels_path, np.array([], dtype=np.int32))
    np.save(durations_path, np.array([], dtype=np.int64))
    np.save(shape_path, np.array([0, 40], dtype=np.int64))
