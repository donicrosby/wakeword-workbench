"""Kokoro TTS backend implementation."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

# Optional dependency - graceful fallback if not installed
try:
    from pykokoro import KokoroPipeline

    _PYKOKORO_AVAILABLE = True
except ImportError:
    _PYKOKORO_AVAILABLE = False
    KokoroPipeline = None  # type: ignore[assignment, misc]

from .base import BackendNotAvailableError, TTSBackend, TTSError, TTSResult
from .cache import get_default_cache

# Kokoro generates at 24000 Hz, we need to resample to 16000 Hz
_KOKORO_SAMPLE_RATE = 24000
_TARGET_SAMPLE_RATE = 16000

# Common Kokoro voices available
_COMMON_VOICES = [
    "af_sarah",
    "af_nicole",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_lewis",
    "bm_george",
]


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

    _pipeline: ClassVar[KokoroPipeline | None] = None  # type: ignore[type-arg]

    def __init__(self, voice: str = "af_sarah", speed: float = 1.0) -> None:
        if not _PYKOKORO_AVAILABLE:
            raise BackendNotAvailableError(
                "pykokoro is not installed. Install it with: uv sync --extra kokoro"
            )

        if speed <= 0:
            raise TTSError(f"speed must be positive, got {speed}")

        self._voice = voice
        self._speed = speed

        # Initialize pipeline lazily to avoid loading models at import time
        if KokoroBackend._pipeline is None:
            try:
                KokoroBackend._pipeline = KokoroPipeline()  # type: ignore[operator]
            except Exception as e:
                raise TTSError(f"Failed to initialize Kokoro pipeline: {e}") from e

        # Validate voice is available
        if voice not in self.list_voices():
            available = ", ".join(self.list_voices())
            raise TTSError(f"Voice '{voice}' not available. Available: {available}")

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
        cached_result = cache.get(text, self._voice, "kokoro", self._speed)
        if cached_result is not None:
            return cached_result

        try:
            # Generate audio using pykokoro
            audio_24k = self._pipeline.generate(text, voice=self._voice, speed=self._speed)  # type: ignore[union-attr]

            # Convert to numpy array if needed
            if not isinstance(audio_24k, np.ndarray):
                audio_24k = np.array(audio_24k, dtype=np.float32)

            # Ensure float32
            audio_24k = audio_24k.astype(np.float32)

            # Resample from 24000 Hz to 16000 Hz using librosa
            import librosa  # type: ignore[attr-defined]

            audio_16k = librosa.resample(
                audio_24k,
                orig_sr=_KOKORO_SAMPLE_RATE,
                target_sr=_TARGET_SAMPLE_RATE,
            )

            # Ensure float32 after resampling
            audio_16k = audio_16k.astype(np.float32)

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
            cache.put(text, self._voice, "kokoro", result, self._speed)

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
