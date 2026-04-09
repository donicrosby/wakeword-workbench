"""Kokoro TTS backend implementation."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from typing import Any, ClassVar, Protocol, cast

import numpy as np

# Optional dependency - graceful fallback if not installed
try:
    pykokoro = import_module("pykokoro")
    GenerationConfig = pykokoro.GenerationConfig
    KokoroPipeline = pykokoro.KokoroPipeline
    PipelineConfig = pykokoro.PipelineConfig

    _PYKOKORO_AVAILABLE = True
except ImportError:
    _PYKOKORO_AVAILABLE = False
    KokoroPipeline = None  # type: ignore[assignment, misc]
    PipelineConfig = None  # type: ignore[assignment, misc]
    GenerationConfig = None  # type: ignore[assignment, misc]

from .base import BackendNotAvailableError, TTSBackend, TTSError, TTSResult
from .cache import get_default_cache

# Kokoro generates at 24000 Hz, we need to resample to 16000 Hz
_KOKORO_SAMPLE_RATE = 24000
_TARGET_SAMPLE_RATE = 16000

# All English voices from upstream Kokoro-82M (American + British)
# See: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
_COMMON_VOICES = [
    # American English — Female
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    # American English — Male
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    # British English — Female
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    # British English — Male
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
]


class _KokoroAudioResult(Protocol):
    audio: Any
    sample_rate: int


class _KokoroPipelineLike(Protocol):
    def run(self, text: str, voice: str, generation: Any) -> _KokoroAudioResult: ...


class KokoroBackend(TTSBackend):
    """Kokoro TTS backend using pykokoro.

    Kokoro is a lightweight, fast TTS engine that outputs audio at 24000 Hz.
    This backend resamples to 16000 Hz for wake word detection compatibility.

    Args:
        voice: Voice identifier (default: "af_sarah").
        speed: Speech speed multiplier (default: 1.0).

    Raises:
        BackendNotAvailableError: If pykokoro is not installed.
        TTSError: If initialization fails.
    """

    _pipeline: ClassVar[_KokoroPipelineLike | None] = None
    _pipelines: ClassVar[dict[tuple[str, str | None], _KokoroPipelineLike]] = {}

    def __init__(
        self,
        voice: str = "af_sarah",
        speed: float = 1.0,
        acceleration: str = "cpu",
        device: str | None = None,
    ) -> None:
        if not _PYKOKORO_AVAILABLE:
            raise BackendNotAvailableError(
                "pykokoro is not installed. Install it with: uv sync --extra kokoro"
            )

        if speed <= 0:
            raise TTSError(f"speed must be positive, got {speed}")

        normalized_acceleration = acceleration.strip().lower()
        if normalized_acceleration == "migraphx":
            raise TTSError("MIGraphX is not supported for Kokoro inference")
        if normalized_acceleration not in {"cpu", "cuda", "openvino"}:
            raise TTSError(
                "Kokoro acceleration must be one of: cpu, cuda, openvino; "
                f"got {normalized_acceleration!r}"
            )

        self._voice = voice
        self._speed = speed
        self._acceleration = normalized_acceleration
        self._device = device

        # Initialize pipeline lazily to avoid loading models at import time
        pipeline_key = (self._acceleration, self._device)
        try:
            if pipeline_key == ("cpu", None):
                if KokoroBackend._pipeline is None:
                    KokoroBackend._pipeline = self._build_pipeline(
                        acceleration=self._acceleration,
                        device=self._device,
                    )
            elif pipeline_key not in KokoroBackend._pipelines:
                KokoroBackend._pipelines[pipeline_key] = self._build_pipeline(
                    acceleration=self._acceleration,
                    device=self._device,
                )
        except Exception as e:
            raise TTSError(f"Failed to initialize Kokoro pipeline: {e}") from e

        # Validate voice is available
        if voice not in self.list_voices():
            available = ", ".join(self.list_voices())
            raise TTSError(f"Voice '{voice}' not available. Available: {available}")

    @staticmethod
    @contextmanager
    def _temporary_provider_env(provider_name: str | None) -> Iterator[None]:
        previous = os.environ.get("ONNX_PROVIDER")
        try:
            if provider_name is None:
                os.environ.pop("ONNX_PROVIDER", None)
            else:
                os.environ["ONNX_PROVIDER"] = provider_name
            yield
        finally:
            if previous is None:
                os.environ.pop("ONNX_PROVIDER", None)
            else:
                os.environ["ONNX_PROVIDER"] = previous

    @classmethod
    def _provider_name(cls, acceleration: str) -> str | None:
        if acceleration == "cpu":
            return None
        if acceleration == "cuda":
            return "cuda"
        if acceleration == "openvino":
            return "openvino"
        raise TTSError(f"Unsupported Kokoro acceleration: {acceleration}")

    @classmethod
    def _build_pipeline(cls, acceleration: str, device: str | None) -> _KokoroPipelineLike:
        pipeline_cls: Any = KokoroPipeline
        pipeline_config_cls: Any = PipelineConfig
        provider_name = cls._provider_name(acceleration)

        config_kwargs: dict[str, Any] = {}
        config_signature = inspect.signature(pipeline_config_cls)
        if provider_name is not None:
            if "provider" in config_signature.parameters:
                config_kwargs["provider"] = provider_name
            elif "providers" in config_signature.parameters:
                config_kwargs["providers"] = [provider_name]

        if device is not None:
            if "device" in config_signature.parameters:
                config_kwargs["device"] = device
            elif "device_type" in config_signature.parameters:
                config_kwargs["device_type"] = device

        with cls._temporary_provider_env(provider_name):
            pipeline_config = pipeline_config_cls(**config_kwargs)
            return cast(_KokoroPipelineLike, pipeline_cls(pipeline_config))

    @classmethod
    def is_available(cls) -> bool:
        """Check if pykokoro is installed and functional.

        Returns:
            True if pykokoro is available, False otherwise.
        """
        return _PYKOKORO_AVAILABLE

    def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech from text using Kokoro.

        Args:
            text: The text to synthesize.

        Returns:
            TTSResult with audio at 16000 Hz, normalized to [-1, 1].

        Raises:
            TTSError: If synthesis fails.
        """
        if not text or not text.strip():
            raise TTSError("Cannot synthesize empty text")

        # Check cache first
        cache = get_default_cache()
        cached_result = cache.get(
            text,
            self._voice,
            "kokoro",
            self._speed,
            options={
                "acceleration": self._acceleration,
                "device": self._device or "",
            },
        )
        if cached_result is not None:
            return cached_result

        pipeline_key = (self._acceleration, self._device)
        pipeline = (
            self._pipeline if pipeline_key == ("cpu", None) else self._pipelines.get(pipeline_key)
        )
        if pipeline is None:
            raise TTSError("Kokoro pipeline not initialized")

        try:
            # Generate audio using pykokoro with speed
            generation_config_cls: Any = GenerationConfig
            audio_result = pipeline.run(
                text,
                voice=self._voice,
                generation=generation_config_cls(speed=self._speed),
            )

            # Extract audio and sample rate from AudioResult
            audio_24k = audio_result.audio
            source_sr = audio_result.sample_rate

            # Ensure float32
            audio_24k = np.asarray(audio_24k, dtype=np.float32)

            # Resample if needed
            import librosa  # type: ignore[attr-defined]

            if source_sr != _TARGET_SAMPLE_RATE:
                audio_16k = librosa.resample(
                    audio_24k,
                    orig_sr=source_sr,
                    target_sr=_TARGET_SAMPLE_RATE,
                )
            else:
                audio_16k = audio_24k

            # Ensure float32 after resampling
            audio_16k = np.asarray(audio_16k, dtype=np.float32)

            # Normalize to [-1, 1] if needed
            max_val = np.abs(audio_16k).max()
            if max_val > 1.0:
                audio_16k = audio_16k / max_val

            # Calculate duration
            duration = len(audio_16k) / _TARGET_SAMPLE_RATE

            result = TTSResult(
                audio=audio_16k,
                sample_rate=_TARGET_SAMPLE_RATE,
                duration=duration,
            )

            # Store in cache
            cache.put(
                text,
                self._voice,
                "kokoro",
                result,
                self._speed,
                options={
                    "acceleration": self._acceleration,
                    "device": self._device or "",
                },
            )

            return result

        except Exception as e:
            raise TTSError(f"Kokoro synthesis failed: {e}") from e

    def set_voice(self, voice: str) -> None:
        """Set the voice for synthesis.

        Args:
            voice: The voice identifier to use.

        Raises:
            TTSError: If the voice is not available.
        """
        available = self.list_voices()
        if voice not in available:
            raise TTSError(f"Voice '{voice}' not available. Available: {', '.join(available)}")
        self._voice = voice

    def list_voices(self) -> list[str]:
        """List available Kokoro voices.

        Returns:
            List of voice identifiers available for Kokoro TTS.
        """
        return _COMMON_VOICES.copy()


# Register this backend
KokoroBackend.register_backend("kokoro")
