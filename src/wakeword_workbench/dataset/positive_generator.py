"""Positive dataset generator for wake word training samples."""

from __future__ import annotations

import inspect
import json
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from wakeword_workbench.config import Config, TTSProviderConfig
from wakeword_workbench.logging_config import get_logger
from wakeword_workbench.tts.base import TTSBackend, TTSError
from wakeword_workbench.tts.pool import BackendPool
from wakeword_workbench.tts.registry import _BACKENDS, list_available_backends

log = get_logger(__name__)


class PositiveGeneratorError(Exception):
    """Raised when positive sample generation fails critically."""

    pass


def _create_backend_for_provider(provider: TTSProviderConfig) -> TTSBackend:
    """Create a TTS backend for a provider, forwarding supported runtime options.

    Args:
        provider: Provider configuration with backend/runtime settings.

    Returns:
        Instantiated TTS backend.

    Raises:
        TTSError: If the backend is unknown or cannot be instantiated.
    """
    list_available_backends()
    backend_name = provider.backend
    backend_cls = _BACKENDS.get(backend_name)
    if backend_cls is None:
        available = ", ".join(sorted(_BACKENDS)) or "none"
        raise TTSError(f"Unknown TTS backend: '{backend_name}'. Available backends: {available}")

    signature = inspect.signature(backend_cls.__init__)
    kwargs: dict[str, Any] = {}

    if "speed" in signature.parameters:
        kwargs["speed"] = provider.speed
    elif provider.speed != 1.0:
        log.warning("backend_no_speed_support", backend=backend_name, speed=provider.speed)

    if "acceleration" in signature.parameters:
        kwargs["acceleration"] = provider.acceleration
    elif provider.acceleration != "cpu" and "use_cuda" not in signature.parameters:
        log.warning(
            "backend_no_acceleration_support",
            backend=backend_name,
            acceleration=provider.acceleration,
        )

    if "use_cuda" in signature.parameters:
        kwargs["use_cuda"] = provider.acceleration == "cuda"

    if "device" in signature.parameters and provider.device is not None:
        kwargs["device"] = provider.device
    elif provider.device is not None:
        log.warning("backend_no_device_support", backend=backend_name, device=provider.device)

    if "model_path" in signature.parameters and provider.model_path is not None:
        kwargs["model_path"] = provider.model_path

    backend_factory: Any = backend_cls
    return cast(TTSBackend, backend_factory(**kwargs))


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
        self._providers = config.tts.providers
        self._backend_pool = BackendPool()

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "positive_generator_init",
            wake_word=self._wake_word,
            providers=[provider.backend for provider in self._providers],
            voices=config.tts.get_all_voices(),
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

        variants = self._get_positive_phrases()
        log.info("positive_phrases_selected", count=len(variants), phrases=variants[:3])

        # Phase 1: Build generation list from provider/phrase/voice combinations.
        combinations: list[tuple[TTSProviderConfig, str, str]] = []
        for variant in variants:
            for provider in self._providers:
                for voice in provider.voices:
                    combinations.append((provider, variant, voice))

        if not combinations:
            raise PositiveGeneratorError("No TTS provider/voice combinations available")

        total_needed = count

        # Keep voice diversity by assigning samples in round-robin order first.
        assigned_combinations = [combinations[i % len(combinations)] for i in range(total_needed)]

        # Phase 2: Group execution by backend/runtime/voice. This minimizes expensive
        # model/voice switches while keeping the original round-robin distribution.
        grouped_combinations = sorted(
            assigned_combinations,
            key=lambda combo: (
                combo[0].backend,
                combo[0].acceleration,
                combo[2],
            ),
        )

        def provider_instance_key(
            provider: TTSProviderConfig,
        ) -> tuple[str, float, str, str | None, str | None]:
            key_getter = cast(Any, getattr(provider, "backend_instance_key", None))
            if callable(key_getter):
                return cast(tuple[str, float, str, str | None, str | None], key_getter())

            backend, _voices, speed, acceleration, device, model_path = provider.backend_cache_key()
            return (backend, speed, acceleration, device, model_path)

        backends_by_key: dict[tuple[str, float, str, str | None, str | None], TTSBackend] = {}
        for provider, _phrase, _voice in grouped_combinations:
            key = provider_instance_key(provider)
            if key in backends_by_key:
                continue
            try:
                backends_by_key[key] = self._backend_pool.get(provider)
            except Exception as e:
                raise PositiveGeneratorError(f"Failed to get TTS backend: {e}") from e

        manifest_entries: list[dict] = []
        file_index = 0
        generated_count = 0
        failed_count = 0
        active_group: tuple[str, str, str] | None = None
        active_voice_by_backend: dict[tuple[str, float, str, str | None, str | None], str] = {}

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

            # Generate samples using grouped execution order.
            for provider, phrase, voice in grouped_combinations:
                backend_key = provider_instance_key(provider)
                backend = backends_by_key[backend_key]

                current_group = (provider.backend, provider.acceleration, voice)
                if current_group != active_group:
                    active_group = current_group
                    progress.update(
                        task,
                        description=(
                            "[cyan]Generating positive samples... "
                            f"({provider.backend}/{provider.acceleration}/{voice})"
                        ),
                    )

                # Set voice for this backend
                try:
                    if active_voice_by_backend.get(backend_key) != voice:
                        backend.set_voice(voice)
                        active_voice_by_backend[backend_key] = voice
                except TTSError as e:
                    log.warning("voice_set_failed", voice=voice, error=str(e))
                    continue
                except NotImplementedError as e:
                    log.warning(
                        "voice_set_not_supported",
                        backend=provider.backend,
                        voice=voice,
                        error=str(e),
                    )

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
                    soundfile = import_module("soundfile")
                    soundfile.write(file_path, audio, 16000)

                    # Calculate duration in milliseconds
                    duration_ms = int(len(audio) / 16000 * 1000)

                    # Add to manifest
                    manifest_entries.append(
                        {
                            "path": filename,
                            "label": 1,
                            "text": phrase,
                            "voice": voice,
                            "backend": provider.backend,
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
                        backend=provider.backend,
                        phrase=phrase,
                        voice=voice,
                        error=str(e),
                    )
                    continue
                except Exception as e:
                    failed_count += 1
                    log.error(
                        "unexpected_error",
                        backend=provider.backend,
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

    def _get_positive_phrases(self) -> list[str]:
        """Return the explicit positive phrases to synthesize.

        Defaults to the exact wake word only. Optional variants must be provided
        explicitly by the end user in config.
        """
        if self.config.wake_word_variants is not None:
            return self.config.wake_word_variants.copy()
        return [self._wake_word.replace("_", " ")]

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
            librosa = import_module("librosa")
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
