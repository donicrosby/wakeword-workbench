"""Audio loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray


class AudioLoadError(Exception):
    """Raised when audio loading fails due to corruption or unsupported format."""

    pass


@dataclass
class AudioMetadata:
    """Metadata about an audio file."""

    original_sr: int
    target_sr: int
    duration: float
    channels: int
    filename: str


def load_audio(
    path: Path | str, target_sr: int = 16000
) -> tuple[NDArray[np.float32], AudioMetadata]:
    """Load an audio file and resample to the target sample rate.

    Args:
        path: Path to the audio file.
        target_sr: Target sample rate in Hz (default: 16000).

    Returns:
        A tuple of (audio_data, metadata) where audio_data is a float32 numpy array
        with values in [-1.0, 1.0].

    Raises:
        AudioLoadError: If the file cannot be loaded or is corrupted.
    """
    path = Path(path)

    if not path.exists():
        raise AudioLoadError(f"Audio file not found: {path}")

    try:
        soundfile = import_module("soundfile")
        info = soundfile.info(path)
        audio, original_sr = soundfile.read(path, always_2d=False, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
        channels = info.channels
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio file '{path}': {e}") from e

    if audio.ndim == 0:
        raise AudioLoadError(f"Failed to load audio file '{path}': no audio samples found")

    # Convert to mono if stereo (common for wake word processing)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
        channels = 1

    # Ensure float32
    audio = np.asarray(audio, dtype=np.float32)

    # Resample if needed
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
        audio = np.asarray(audio, dtype=np.float32)

    # Calculate duration
    duration = len(audio) / target_sr

    metadata = AudioMetadata(
        original_sr=original_sr,
        target_sr=target_sr,
        duration=duration,
        channels=channels,
        filename=path.name,
    )

    return audio, metadata


def save_audio(path: Path | str, audio: NDArray[np.float32], sr: int) -> None:
    """Save audio data to a WAV file.

    Args:
        path: Path where to save the audio file.
        audio: Audio data as a numpy array (float32 in [-1.0, 1.0] range).
        sr: Sample rate in Hz.

    Raises:
        AudioLoadError: If the file cannot be saved.
    """
    path = Path(path)

    # Ensure float64 for soundfile
    if audio.dtype != np.float64:
        audio = audio.astype(np.float64)

    # Normalize to [-1, 1] range just in case
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val

    try:
        # Save as 16-bit PCM WAV
        soundfile = import_module("soundfile")
        soundfile.write(path, audio, sr, subtype="PCM_16")
    except Exception as e:
        raise AudioLoadError(f"Failed to save audio to '{path}': {e}") from e
