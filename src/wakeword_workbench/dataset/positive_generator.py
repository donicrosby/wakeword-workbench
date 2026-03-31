"""Positive dataset generator for wake word training samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from wakeword_workbench.config import Config
from wakeword_workbench.logging_config import get_logger
from wakeword_workbench.tts.base import TTSError
from wakeword_workbench.tts.registry import get_backend

from .phrase_variants import generate_variants

log = get_logger(__name__)


class PositiveGeneratorError(Exception):
    """Raised when positive sample generation fails critically."""

    pass


class PositiveGenerator:
    """Generator for positive wake word training samples.

    Generates audio samples by synthesizing text variations of the wake word
    phrase using the configured TTS backend and voices.

    Attributes:
        config: Configuration object containing wake word and TTS settings.
        output_dir: Directory where generated samples will be saved.
    """

    def __init__(self, config: Config, output_dir: Path) -> None:
        """Initialize the positive sample generator.

        Args:
            config: Configuration object with wake_word, tts, and samples settings.
            output_dir: Directory path where generated audio files will be saved.

        Raises:
            PositiveGeneratorError: If critical initialization fails.
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self._wake_word = config.wake_word
        self._voices = config.tts.voices
        self._backend_name = config.tts.backend

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "positive_generator_init",
            wake_word=self._wake_word,
            backend=self._backend_name,
            voices=self._voices,
            output_dir=str(self.output_dir),
        )

    def generate(self, count: int) -> Path:
        """Generate positive wake word samples.

        Synthesizes audio samples for the wake word phrase using text variants,
        TTS voices, and saves them as WAV files with a JSONL manifest.

        Args:
            count: Number of samples to generate.

        Returns:
            Path to the generated JSONL manifest file.

        Raises:
            PositiveGeneratorError: If critical errors occur during generation.
        """
        if count <= 0:
            raise PositiveGeneratorError(f"count must be positive, got {count}")

        log.info("generating_positive_samples", count=count)

        # Get TTS backend
        try:
            backend = get_backend(self._backend_name)
        except Exception as e:
            raise PositiveGeneratorError(f"Failed to get TTS backend: {e}") from e

        # Generate text variants
        variants = generate_variants(self._wake_word)
        if not variants:
            # Fallback to just the wake word if no variants generated
            variants = [self._wake_word.replace("_", " ")]
            log.warning("no_variants_generated", fallback=variants[0])

        log.info("variants_generated", count=len(variants), variants=variants[:3])

        # Build generation list: (phrase, voice) combinations
        combinations: list[tuple[str, str]] = []
        for variant in variants:
            for voice in self._voices:
                combinations.append((variant, voice))

        # Calculate actual samples to generate (may be more or less than count)
        # We generate all combinations and return count of them
        samples_per_combination = max(1, count // len(combinations)) if combinations else 0
        total_needed = count

        manifest_entries: list[dict] = []
        file_index = 0
        generated_count = 0
        failed_count = 0

        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Generating positive samples...",
                total=total_needed,
            )

            # Generate samples
            for phrase, voice in combinations:
                if generated_count >= total_needed:
                    break

                # Set voice for this backend
                try:
                    backend.set_voice(voice)
                except TTSError as e:
                    log.warning("voice_set_failed", voice=voice, error=str(e))
                    continue

                # Generate samples for this combination
                for _sample_idx in range(samples_per_combination):
                    if generated_count >= total_needed:
                        break

                    try:
                        # Synthesize audio
                        result = backend.synthesize(phrase)

                        # Generate filename
                        filename = f"{self._wake_word}_{voice}_{file_index:04d}.wav"
                        file_path = self.output_dir / filename

                        # Ensure mono at 16000 Hz
                        audio = self._ensure_format(result.audio, result.sample_rate)
                        audio = self._ensure_mono(audio)

                        # Save WAV file
                        sf.write(file_path, audio, 16000)

                        # Calculate duration in milliseconds
                        duration_ms = int(len(audio) / 16000 * 1000)

                        # Add to manifest
                        manifest_entries.append(
                            {
                                "path": str(file_path),
                                "label": 1,
                                "text": phrase,
                                "voice": voice,
                                "duration_ms": duration_ms,
                            }
                        )

                        generated_count += 1
                        file_index += 1

                        progress.update(task, advance=1)

                    except TTSError as e:
                        failed_count += 1
                        log.warning(
                            "tts_synthesis_failed",
                            phrase=phrase,
                            voice=voice,
                            error=str(e),
                        )
                        continue
                    except Exception as e:
                        failed_count += 1
                        log.error(
                            "unexpected_error",
                            phrase=phrase,
                            voice=voice,
                            error=str(e),
                        )
                        continue

        # Write manifest
        manifest_path = self.output_dir / "positive_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + "\n")

        log.info(
            "generation_complete",
            total=generated_count,
            failed=failed_count,
            manifest=str(manifest_path),
        )

        return manifest_path

    def _ensure_format(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Ensure audio is at 16000 Hz sample rate.

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Current sample rate.

        Returns:
            Audio resampled to 16000 Hz if needed.
        """
        if sample_rate != 16000:
            # Resample to 16000 Hz
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            audio = audio.astype(np.float32)
        return audio

    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is mono (single channel).

        Args:
            audio: Audio samples (can be stereo).

        Returns:
            Audio with single channel if stereo.
        """
        if audio.ndim > 1:
            # Average channels to mono
            audio = audio.mean(axis=1)
        return audio
