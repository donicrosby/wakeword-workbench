"""Long audio processor with sliding window inference and chunked loading.

This module provides memory-efficient processing of long audio files (hours)
for wake word detection using sliding window inference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import librosa
import numpy as np


@dataclass
class WindowPrediction:
    """Prediction result for a single audio window.

    Attributes:
        timestamp: Time in seconds from the start of the audio file
        prediction: Model confidence score (typically 0-1)
        window_start: Sample index where the window starts
        window_end: Sample index where the window ends
    """

    timestamp: float
    prediction: float
    window_start: int
    window_end: int


def process_long_audio(
    model: Callable[[np.ndarray], float],
    audio_path: Path | str,
    window_size: int = 16000,
    hop_size: int = 8000,
    sample_rate: int = 16000,
    chunk_duration: float = 30.0,
) -> list[WindowPrediction]:
    """Process long audio file with sliding window inference.

    This function processes audio files of arbitrary length (even hours)
    without loading the entire file into memory. It uses chunked loading
    and processes audio in overlapping windows.

    Args:
        model: Callable that takes audio samples and returns confidence score
        audio_path: Path to the audio file to process
        window_size: Size of each window in samples (default: 16000 = 1s at 16kHz)
        hop_size: Step size between consecutive windows in samples (default: 8000 = 0.5s)
        sample_rate: Target sample rate for audio loading (default: 16000)
        chunk_duration: Duration of each chunk to load in seconds (default: 30.0)

    Returns:
        List of WindowPrediction objects with timestamps and predictions

    Raises:
        FileNotFoundError: If the audio file does not exist
        ValueError: If window_size <= 0, hop_size <= 0, or chunk_duration <= 0

    Example:
        >>> def dummy_model(audio):
        ...     return np.mean(audio)  # Simple example
        >>> predictions = process_long_audio(
        ...     model=dummy_model,
        ...     audio_path="audio.wav",
        ...     window_size=16000,
        ...     hop_size=8000,
        ... )
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")
    if chunk_duration <= 0:
        raise ValueError(f"chunk_duration must be positive, got {chunk_duration}")

    # Get audio file duration
    duration = librosa.get_duration(path=str(audio_path))

    # Handle very short audio (process in one go)
    if duration <= chunk_duration:
        audio, _ = librosa.load(
            str(audio_path),
            sr=sample_rate,
            mono=True,
        )
        return _process_audio_array(model, audio, window_size, hop_size, sample_rate)

    # Process in chunks for memory efficiency
    return _process_in_chunks(
        model,
        audio_path,
        window_size,
        hop_size,
        sample_rate,
        chunk_duration,
        duration,
    )


def _process_audio_array(
    model: Callable[[np.ndarray], float],
    audio: np.ndarray,
    window_size: int,
    hop_size: int,
    sample_rate: int,
    offset_samples: int = 0,
) -> list[WindowPrediction]:
    """Process audio array with sliding windows.

    Args:
        model: Model callable
        audio: Audio samples array
        window_size: Window size in samples
        hop_size: Hop size in samples
        sample_rate: Sample rate for timestamp calculation
        offset_samples: Sample offset from start of original file

    Returns:
        List of WindowPrediction objects
    """
    predictions = []
    num_samples = len(audio)

    # Calculate number of windows ensuring we cover the entire audio
    # The last window should include the final samples
    if num_samples <= window_size:
        # Audio is shorter than or equal to window size - process once
        window_start = 0
        window_end = num_samples
        window = audio

        # Pad if necessary for model consistency
        if len(window) < window_size:
            window = np.pad(window, (0, window_size - len(window)), mode="constant")

        prediction = model(window)
        timestamp = (offset_samples + window_start) / sample_rate

        predictions.append(
            WindowPrediction(
                timestamp=timestamp,
                prediction=float(prediction),
                window_start=offset_samples + window_start,
                window_end=offset_samples + window_end,
            )
        )
    else:
        # Process with sliding windows
        num_windows = (num_samples - window_size) // hop_size + 1

        for i in range(num_windows):
            window_start = i * hop_size
            window_end = window_start + window_size
            window = audio[window_start:window_end]

            prediction = model(window)
            timestamp = (offset_samples + window_start) / sample_rate

            predictions.append(
                WindowPrediction(
                    timestamp=timestamp,
                    prediction=float(prediction),
                    window_start=offset_samples + window_start,
                    window_end=offset_samples + window_end,
                )
            )

        # Handle the final partial window if there's remaining audio
        remaining = num_samples - num_windows * hop_size
        if remaining > 0 and remaining < window_size:
            window_start = num_samples - window_size
            window_end = num_samples
            window = audio[window_start:window_end]

            # Only add if this window wasn't already processed
            last_end = predictions[-1].window_end - offset_samples if predictions else 0
            if window_end > last_end:
                prediction = model(window)
                timestamp = (offset_samples + window_start) / sample_rate

                predictions.append(
                    WindowPrediction(
                        timestamp=timestamp,
                        prediction=float(prediction),
                        window_start=offset_samples + window_start,
                        window_end=offset_samples + window_end,
                    )
                )

    return predictions


def _process_in_chunks(
    model: Callable[[np.ndarray], float],
    audio_path: Path,
    window_size: int,
    hop_size: int,
    sample_rate: int,
    chunk_duration: float,
    total_duration: float,
) -> list[WindowPrediction]:
    """Process audio in chunks for memory efficiency.

    Uses overlapping chunks to ensure continuity at chunk boundaries.

    Args:
        model: Model callable
        audio_path: Path to audio file
        window_size: Window size in samples
        hop_size: Hop size in samples
        sample_rate: Sample rate
        chunk_duration: Duration of each chunk in seconds
        total_duration: Total duration of audio file in seconds

    Returns:
        List of WindowPrediction objects
    """
    predictions = []
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = window_size  # Overlap by window size to handle boundaries

    # Process chunks
    offset = 0.0
    while offset < total_duration:
        # Calculate chunk boundaries with overlap
        chunk_start = max(0.0, offset - (overlap_samples / sample_rate) if offset > 0 else 0.0)
        current_chunk_duration = min(
            chunk_duration + (overlap_samples / sample_rate if offset > 0 else 0),
            total_duration - chunk_start,
        )

        # Load chunk
        audio, _ = librosa.load(
            str(audio_path),
            sr=sample_rate,
            mono=True,
            offset=chunk_start,
            duration=current_chunk_duration,
        )

        # Calculate offset in samples for this chunk
        offset_samples = int(chunk_start * sample_rate)

        # Determine valid processing range (excluding overlap padding)
        valid_start = 0
        if offset == 0:
            # First chunk: process all except last window_size samples
            # (will be reprocessed with next chunk)
            valid_end = len(audio)
            if total_duration > chunk_duration:
                valid_end = max(0, len(audio) - overlap_samples)
        elif offset + chunk_duration >= total_duration:
            # Last chunk: process from the overlap point
            valid_start = int((offset - chunk_start) * sample_rate)
            valid_end = len(audio)
        else:
            # Middle chunk: process middle section
            valid_start = int((offset - chunk_start) * sample_rate)
            valid_end = len(audio) - overlap_samples

        # Adjust processing range
        if offset == 0:
            process_audio = audio[:valid_end] if valid_end > 0 else audio
        else:
            process_audio = (
                audio[valid_start:valid_end] if valid_end > valid_start else np.array([])
            )

        # Process this chunk
        if len(process_audio) > 0:
            chunk_predictions = _process_audio_array(
                model,
                process_audio,
                window_size,
                hop_size,
                sample_rate,
                offset_samples=offset_samples + (0 if offset == 0 else valid_start),
            )
            predictions.extend(chunk_predictions)

        # Move to next chunk
        offset += chunk_duration

        # Safety check to prevent infinite loop
        if current_chunk_duration <= 0:
            break

    # Remove any duplicate predictions (can occur at chunk boundaries)
    predictions = _deduplicate_predictions(predictions)

    return predictions


def _deduplicate_predictions(
    predictions: list[WindowPrediction],
) -> list[WindowPrediction]:
    """Remove duplicate predictions based on window start position.

    Args:
        predictions: List of predictions (may contain duplicates)

    Returns:
        List of unique predictions sorted by window_start
    """
    seen_starts = set()
    unique_predictions = []

    for pred in predictions:
        if pred.window_start not in seen_starts:
            seen_starts.add(pred.window_start)
            unique_predictions.append(pred)

    # Sort by window_start
    unique_predictions.sort(key=lambda p: p.window_start)

    return unique_predictions
