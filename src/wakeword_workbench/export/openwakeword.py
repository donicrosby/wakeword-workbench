"""OpenWakeWord format exporter.

Exports manifest data to numpy arrays compatible with openWakeWord training.

Format:
    - X_{split}.npy: Audio features array
    - y_{split}.npy: Label array (0/1)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np

from wakeword_workbench.augment.audio_loader import AudioLoadError, load_audio
from wakeword_workbench.augment.padding import FixedSizeClip
from wakeword_workbench.dataset.metadata import Manifest


class OpenWakeWordExportError(Exception):
    """Raised when openWakeWord export fails."""

    pass


@dataclass
class ExportConfig:
    """Configuration for openWakeWord export."""

    sample_rate: int = 16000
    format: Literal["raw", "mel"] = "raw"
    n_mels: int = 96
    n_fft: int = 512
    hop_length: int = 160  # ~10ms at 16kHz
    fixed_length: int | None = None


def _extract_mel_features(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 96,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """Extract mel-spectrogram features.

    Args:
        audio: Audio array (samples,).
        sr: Sample rate.
        n_mels: Number of mel bands.
        n_fft: FFT window size.
        hop_length: Hop length in samples.

    Returns:
        Mel-spectrogram features (n_mels, time_steps).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=0,
        fmax=8000,
    )
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel.astype(np.float32)


def export_to_numpy(
    manifest: Manifest,
    output_dir: Path | str,
    split: str = "train",
    fixed_length: int | None = None,
    format: Literal["raw", "mel"] = "raw",
    sample_rate: int = 16000,
) -> tuple[Path, Path]:
    """Export manifest to openWakeWord numpy format.

    Args:
        manifest: Manifest containing audio entries.
        output_dir: Directory to save numpy files.
        split: Dataset split name (e.g., "train", "val", "test").
        fixed_length: If set, pad/crop audio to this number of samples.
                     For mel format, this is in time frames.
        format: Export format - "raw" (audio samples) or "mel" (mel-spectrogram).
        sample_rate: Target sample rate for audio.

    Returns:
        Tuple of (X_path, y_path) pointing to the saved files.

    Raises:
        OpenWakeWordExportError: If export fails.

    Example:
        >>> manifest = Manifest.load(Path("train_manifest.jsonl"))
        >>> export_to_numpy(manifest, Path("output/"), split="train")
        (PosixPath('output/X_train.npy'), PosixPath('output/y_train.npy'))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_path = output_dir / f"X_{split}.npy"
    y_path = output_dir / f"y_{split}.npy"

    audio_arrays: list[np.ndarray] = []
    labels: list[int] = []

    clipper = (
        FixedSizeClip(fixed_length, mode="pad", jitter=False) if fixed_length is not None else None
    )

    for i, entry in enumerate(manifest):
        try:
            # Load audio file
            audio, _ = load_audio(entry.path, target_sr=sample_rate)

            if format == "mel":
                # Extract mel features
                mel = _extract_mel_features(
                    audio,
                    sr=sample_rate,
                    n_mels=96,
                    n_fft=512,
                    hop_length=160,
                )
                # Transpose to (time_steps, n_mels) for openWakeWord
                features = mel.T
            else:
                # Raw audio
                features = audio

            # Apply fixed-length padding/cropping
            if clipper is not None:
                if format == "mel":
                    # For mel, pad/crop to fixed time frames
                    assert fixed_length is not None  # Type guard for type checker
                    # Transpose, clip, transpose back
                    features_t = features.T  # (n_mels, time)
                    clipped = np.zeros((features_t.shape[0], fixed_length), dtype=np.float32)
                    min_time = min(features_t.shape[1], fixed_length)
                    clipped[:, :min_time] = features_t[:, :min_time]
                    features = clipped.T  # (time, n_mels)
                else:
                    features = clipper.apply(features, sample_rate)

            audio_arrays.append(features.astype(np.float32))
            labels.append(entry.label)

        except (AudioLoadError, OSError) as e:
            raise OpenWakeWordExportError(f"Failed to export entry {i} ({entry.path}): {e}") from e

    if not audio_arrays:
        raise OpenWakeWordExportError("Manifest is empty - nothing to export")

    # Stack into arrays - handle variable-length by padding to max
    if fixed_length is not None:
        # All same length due to fixed_length constraint
        X = np.stack(audio_arrays)
    else:
        # Variable length - pad to max length
        max_len = max(arr.shape[0] for arr in audio_arrays)
        padded_arrays = []
        for arr in audio_arrays:
            if arr.shape[0] < max_len:
                pad_width = max_len - arr.shape[0]
                arr = np.pad(arr, (0, pad_width), mode="constant", constant_values=0)
            padded_arrays.append(arr)
        X = np.stack(padded_arrays)

    y = np.array(labels, dtype=np.int64)

    # Validate shapes
    if X.ndim == 1:
        raise OpenWakeWordExportError(f"Expected at least 2D array, got shape {X.shape}")

    if len(y) != len(X):
        raise OpenWakeWordExportError(
            f"Label count ({len(y)}) doesn't match audio count ({len(X)})"
        )

    # Save arrays
    try:
        np.save(X_path, X)
        np.save(y_path, y)
    except OSError as e:
        raise OpenWakeWordExportError(f"Failed to save numpy files: {e}") from e

    return X_path, y_path


def validate_export(output_dir: Path | str, split: str = "train") -> dict[str, np.ndarray]:
    """Load and validate exported openWakeWord files.

    Args:
        output_dir: Directory containing the export files.
        split: Split name used during export.

    Returns:
        Dict with 'X' and 'y' arrays.

    Raises:
        OpenWakeWordExportError: If files don't exist or are invalid.
    """
    output_dir = Path(output_dir)
    X_path = output_dir / f"X_{split}.npy"
    y_path = output_dir / f"y_{split}.npy"

    if not X_path.exists():
        raise OpenWakeWordExportError(f"X file not found: {X_path}")
    if not y_path.exists():
        raise OpenWakeWordExportError(f"y file not found: {y_path}")

    try:
        X = np.load(X_path, allow_pickle=False)
        y = np.load(y_path, allow_pickle=False)
    except Exception as e:
        raise OpenWakeWordExportError(f"Failed to load numpy files: {e}") from e

    # Validate shapes
    if X.ndim < 2:
        raise OpenWakeWordExportError(f"X must be at least 2D, got shape {X.shape}")

    if len(y) != len(X):
        raise OpenWakeWordExportError(f"y length ({len(y)}) doesn't match X batch size ({len(X)})")

    # Validate labels
    unique_labels = np.unique(y)
    if not all(label in (0, 1) for label in unique_labels):
        raise OpenWakeWordExportError(f"Labels must be 0 or 1, got unique values: {unique_labels}")

    return {"X": X, "y": y}
