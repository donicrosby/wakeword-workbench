"""False positive extractor for wake word detection.

Extracts audio clips where model predictions exceed a threshold,
saving them with metadata for use as hard negatives in training.
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .long_audio import WindowPrediction


@dataclass
class ExtractedClip:
    """Metadata for an extracted audio clip.

    Attributes:
        clip_path: Path to the saved audio file
        original_path: Path to the source audio file
        timestamp: Time in seconds from the start of the audio
        prediction: Model confidence score for this clip
        threshold: Threshold used for extraction
        duration: Duration of the clip in seconds
    """

    clip_path: Path
    original_path: Path
    timestamp: float
    prediction: float
    threshold: float
    duration: float


def extract_false_positives(
    predictions: list[WindowPrediction],
    audio_path: Path | str,
    threshold: float,
    output_dir: Path | str,
    cooldown_seconds: float = 1.0,
    sample_rate: int = 16000,
) -> list[ExtractedClip]:
    """Extract audio clips where predictions exceed the threshold.

    Scans through window predictions and extracts audio segments where
    the prediction confidence exceeds the specified threshold. Uses
    cooldown to deduplicate overlapping detections from the same
    false positive event.

    Args:
        predictions: List of WindowPrediction objects from long_audio processor
        audio_path: Path to the source audio file
        threshold: Minimum confidence score to extract (exclusive)
        output_dir: Directory to save extracted clips (must exist)
        cooldown_seconds: Minimum time between extractions to avoid
            duplicate clips from the same event (default: 1.0)
        sample_rate: Sample rate for audio loading/saving (default: 16000)

    Returns:
        List of ExtractedClip objects with paths and metadata

    Raises:
        FileNotFoundError: If audio_path does not exist
        NotADirectoryError: If output_dir is not a directory
        ValueError: If threshold is not between 0 and 1, or cooldown_seconds < 0

    Example:
        >>> from wakeword_workbench.mining.long_audio import process_long_audio
        >>> preds = process_long_audio(model, "recording.wav")
        >>> clips = extract_false_positives(preds, "recording.wav", 0.8, "fp_clips/")
        >>> for clip in clips:
        ...     print(f"Extracted: {clip.clip_path} @ {clip.timestamp}s")
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output directory does not exist: {output_dir}")

    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    if cooldown_seconds < 0:
        raise ValueError(f"cooldown_seconds must be non-negative, got {cooldown_seconds}")

    # Filter predictions above threshold and apply cooldown deduplication
    selected_predictions = _select_with_cooldown(predictions, threshold, cooldown_seconds)

    if not selected_predictions:
        return []

    # Load full audio for extraction
    audio, sr = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True,
    )

    extracted_clips: list[ExtractedClip] = []

    for pred in selected_predictions:
        clip_path = _generate_clip_path(output_dir, audio_path.stem, pred.timestamp)

        # Extract audio segment using window boundaries
        clip_audio = audio[pred.window_start : pred.window_end]
        duration = len(clip_audio) / sample_rate

        # Save as WAV
        sf.write(str(clip_path), clip_audio, sample_rate)

        extracted_clips.append(
            ExtractedClip(
                clip_path=clip_path,
                original_path=audio_path,
                timestamp=pred.timestamp,
                prediction=pred.prediction,
                threshold=threshold,
                duration=duration,
            )
        )

    return extracted_clips


def _select_with_cooldown(
    predictions: list[WindowPrediction],
    threshold: float,
    cooldown_seconds: float,
) -> list[WindowPrediction]:
    """Select predictions above threshold with cooldown deduplication.

    Args:
        predictions: All window predictions
        threshold: Minimum prediction to select
        cooldown_seconds: Minimum seconds between selections

    Returns:
        Filtered list of predictions
    """
    selected: list[WindowPrediction] = []
    last_extracted_time: float | None = None

    for pred in predictions:
        if pred.prediction <= threshold:
            continue

        # Check cooldown
        if last_extracted_time is not None:
            if pred.timestamp - last_extracted_time < cooldown_seconds:
                continue

        selected.append(pred)
        last_extracted_time = pred.timestamp

    return selected


def _generate_clip_path(
    output_dir: Path,
    stem: str,
    timestamp: float,
) -> Path:
    """Generate a unique filename for an extracted clip.

    Args:
        output_dir: Directory to save to
        stem: Base name from original file
        timestamp: Timestamp of the clip

    Returns:
        Unique Path with .wav extension
    """
    # Format: stem_T{seconds}.wav
    filename = f"{stem}_T{timestamp:.3f}.wav"
    candidate = output_dir / filename

    # If file exists, add counter suffix
    counter = 1
    while candidate.exists():
        filename = f"{stem}_T{timestamp:.3f}_{counter}.wav"
        candidate = output_dir / filename
        counter += 1

    return candidate
