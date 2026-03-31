"""Audio loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


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


def load_audio(path: Path | str, target_sr: int = 16000) -> tuple[np.ndarray, AudioMetadata]:
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
        # Load audio with librosa (handles WAV, FLAC, OGG, MP3, and more)
        # sr=None preserves original sample rate, then we resample
        audio, original_sr = librosa.load(path, sr=None, mono=False)
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio file '{path}': {e}") from e

    # Get metadata for channels
    try:
        info = sf.info(path)
        channels = info.channels
    except Exception:
        # Fallback: librosa loads mono as 1D, stereo as 2D
        channels = 1 if audio.ndim == 1 else audio.ndim

    # Convert to mono if stereo (common for wake word processing)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)
        channels = 1

    # Ensure float32
    audio = audio.astype(np.float32)

    # Resample if needed
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

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


def save_audio(path: Path | str, audio: np.ndarray, sr: int) -> None:
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
        sf.write(path, audio, sr, subtype="PCM_16")
    except Exception as e:
        raise AudioLoadError(f"Failed to save audio to '{path}': {e}") from e
