"""Piper TTS backend implementation."""

from __future__ import annotations

import urllib.request
import wave
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

import numpy as np

from wakeword_workbench.logging_config import get_logger
from wakeword_workbench.tts.base import (
    BackendNotAvailableError,
    TTSBackend,
    TTSError,
    TTSResult,
)

from .cache import get_default_cache

log = get_logger(__name__)

# Try to import piper, but allow graceful fallback
try:
    PiperVoice = import_module("piper").PiperVoice

    _PIPER_AVAILABLE = True
except ImportError:
    _PIPER_AVAILABLE = False
    PiperVoice = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from piper import PiperVoice  # type: ignore[import-untyped]

# Default Piper model configuration
_DEFAULT_MODEL_NAME = "en_US-lessac-medium"
_DEFAULT_MODEL_URL = (
    f"https://github.com/rhasspy/piper/releases/download/2024.11.14-2/{_DEFAULT_MODEL_NAME}.onnx"
)
_PIPER_MODEL_CACHE_DIR = Path.home() / ".cache" / "wakeword_workbench" / "piper_models"

_PIPER_HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Known high-quality English voices available from the Piper voices repository.
# Each entry is a voice key that maps to a model at:
#   {_PIPER_HF_BASE_URL}/en/en_US/{name}/{quality}/{voice_key}.onnx
_KNOWN_VOICES: list[str] = [
    "en_US-lessac-high",
    "en_US-lessac-medium",
    "en_US-lessac-low",
    "en_US-ryan-high",
    "en_US-ryan-medium",
    "en_US-ryan-low",
    "en_US-ljspeech-high",
    "en_US-ljspeech-medium",
    "en_US-libritts-high",
    "en_US-libritts_r-medium",
    "en_US-amy-medium",
    "en_US-amy-low",
    "en_US-arctic-medium",
    "en_US-hfc_female-medium",
    "en_US-hfc_male-medium",
    "en_US-joe-medium",
    "en_US-kusal-medium",
    "en_US-kristin-medium",
    "en_GB-alan-medium",
    "en_GB-alan-low",
    "en_GB-alba-medium",
    "en_GB-aru-medium",
    "en_GB-cori-medium",
    "en_GB-cori-high",
    "en_GB-jenny_dioco-medium",
    "en_GB-northern_english_male-medium",
    "en_GB-semaine-medium",
    "en_GB-southern_english_female-low",
    "en_GB-vctk-medium",
]


class _PiperVoiceLike(Protocol):
    def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> None: ...


class PiperBackend(TTSBackend):
    """Piper TTS backend implementation.

    Piper outputs audio at 22050 Hz, which is resampled to 16000 Hz
    to match the target format for wake word detection models.

    When model_path is None, automatically downloads the default model
    (en_US-lessac-medium) to the cache directory.

    Attributes:
        model_path: Path to the Piper ONNX model file.
        use_cuda: Whether to use CUDA for inference (if available).
        _voice: The loaded PiperVoice instance.
    """

    _PIPER_SAMPLE_RATE: ClassVar[int] = 22050
    """Piper's native output sample rate."""

    _TARGET_SAMPLE_RATE: ClassVar[int] = 16000
    """Target sample rate for wake word models."""

    def __init__(self, model_path: str | None = None, use_cuda: bool = False) -> None:
        """Initialize the Piper backend.

        Args:
            model_path: Path to the Piper ONNX model file (.onnx).
                If None, downloads the default model automatically.
            use_cuda: Whether to use CUDA for inference (default False).

        Raises:
            BackendNotAvailableError: If piper is not installed.
            TTSError: If the model cannot be loaded or downloaded.
        """
        if not self.is_available():
            raise BackendNotAvailableError(
                "Piper TTS is not available. Install it with: pip install piper-tts"
            )

        self.use_cuda = use_cuda
        self._voice: _PiperVoiceLike | None = None

        # Resolve model_path: use provided path or download default
        if model_path is None:
            self.model_path = self._download_default_model()
        else:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise TTSError(
                    f"Piper model not found: {model_path}. "
                    f"Hint: omit model_path to auto-download the default model."
                )

        try:
            # Load the Piper voice model
            piper_voice_class: Any = PiperVoice
            self._voice = cast(_PiperVoiceLike, piper_voice_class.load(str(self.model_path)))
        except Exception as e:
            raise TTSError(f"Failed to load Piper model: {e}") from e

    @classmethod
    def _download_default_model(cls) -> Path:
        """Download the default Piper model if not cached.

        Returns:
            Path to the downloaded model file.

        Raises:
            TTSError: If download fails.
        """
        model_path = _PIPER_MODEL_CACHE_DIR / f"{_DEFAULT_MODEL_NAME}.onnx"

        if model_path.exists():
            return model_path

        # Ensure cache directory exists
        _PIPER_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        log.info("piper_downloading_default_model", dest=str(model_path))
        try:
            urllib.request.urlretrieve(_DEFAULT_MODEL_URL, model_path)
            return model_path
        except Exception as e:
            raise TTSError(
                f"Failed to download default Piper model. "
                f"Please provide a model_path explicitly or download from: {_DEFAULT_MODEL_URL}"
            ) from e

    def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: The text to synthesize.

        Returns:
            TTSResult containing 16000 Hz audio data.

        Raises:
            TTSError: If synthesis fails.
        """
        if self._voice is None:
            raise TTSError("Piper voice not loaded. Cannot synthesize.")

        if not text.strip():
            raise TTSError("Cannot synthesize empty text.")

        # Check cache first
        cache = get_default_cache()
        cached_result = cache.get(
            text, self.model_path.stem if self.model_path else "default", "piper", 1.0
        )
        if cached_result is not None:
            return cached_result

        try:
            # Synthesize WAV audio via wave.open() writer
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)
            wav_buffer.seek(0)

            # Read the WAV data using soundfile
            soundfile = import_module("soundfile")
            audio_data, sample_rate = soundfile.read(wav_buffer, dtype="float32")

            # Handle stereo audio (convert to mono)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Ensure float32
            audio_data = np.asarray(audio_data, dtype=np.float32)

            # Resample from 22050 Hz to 16000 Hz using scipy
            audio_data = self._resample(audio_data, sample_rate, self._TARGET_SAMPLE_RATE)

            # Clip to ensure normalized audio (handles numerical errors from resampling)
            audio_data = np.clip(audio_data, -1.0, 1.0)

            # Calculate duration
            duration = len(audio_data) / self._TARGET_SAMPLE_RATE

            result = TTSResult(
                audio=audio_data,
                sample_rate=self._TARGET_SAMPLE_RATE,
                duration=duration,
            )

            # Store in cache
            cache.put(
                text, self.model_path.stem if self.model_path else "default", "piper", result, 1.0
            )

            return result

        except Exception as e:
            raise TTSError(f"Piper synthesis failed: {e}") from e

    def set_voice(self, voice: str) -> None:
        """Download (if needed) and load a different Piper voice model.

        Args:
            voice: Voice key like ``en_US-lessac-high``. Must be in
                :data:`_KNOWN_VOICES`.

        Raises:
            TTSError: If the voice is unknown or download/load fails.
        """
        if voice not in _KNOWN_VOICES:
            raise TTSError(
                f"Voice '{voice}' not available. "
                f"Available: {', '.join(_KNOWN_VOICES[:5])}... ({len(_KNOWN_VOICES)} total)"
            )

        model_path = self._download_voice_model(voice)
        try:
            piper_voice_class: Any = PiperVoice
            self._voice = cast(_PiperVoiceLike, piper_voice_class.load(str(model_path)))
            self.model_path = model_path
            log.info("piper_voice_loaded", voice=voice, model_path=str(model_path))
        except Exception as e:
            raise TTSError(f"Failed to load Piper model for voice '{voice}': {e}") from e

    @staticmethod
    def _download_voice_model(voice: str) -> Path:
        """Download a Piper voice model from HuggingFace if not cached.

        Args:
            voice: Voice key like ``en_US-lessac-high``.

        Returns:
            Path to the local ``.onnx`` model file.

        Raises:
            TTSError: If download fails.
        """
        onnx_path = _PIPER_MODEL_CACHE_DIR / f"{voice}.onnx"
        json_path = _PIPER_MODEL_CACHE_DIR / f"{voice}.onnx.json"

        if onnx_path.exists() and json_path.exists():
            return onnx_path

        _PIPER_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Voice key format: en_US-lessac-high -> locale=en/en_US, name=lessac, quality=high
        parts = voice.split("-")
        if len(parts) < 3:
            raise TTSError(
                f"Invalid voice key format '{voice}': expected <locale>-<name>-<quality>"
            )

        locale = parts[0]
        name = parts[1]
        quality = parts[2]
        lang = locale.split("_")[0]

        for suffix, local_path in [(".onnx", onnx_path), (".onnx.json", json_path)]:
            url = f"{_PIPER_HF_BASE_URL}/{lang}/{locale}/{name}/{quality}/{voice}{suffix}"
            log.info("piper_downloading_model", url=url, dest=str(local_path))
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                local_path.unlink(missing_ok=True)
                raise TTSError(f"Failed to download Piper voice '{voice}' from {url}: {e}") from e

        return onnx_path

    def list_voices(self) -> list[str]:
        """Return known Piper English voice names."""
        return _KNOWN_VOICES.copy()

    @staticmethod
    def list_voices_in_directory(directory: str | Path) -> list[str]:
        """List available Piper voice models in a directory.

        Args:
            directory: Directory to scan for .onnx model files.

        Returns:
            List of model file paths (excluding extension).
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        voices = []
        for model_file in directory.glob("*.onnx"):
            # Return model name without extension
            voices.append(str(model_file.with_suffix("")))
        return sorted(voices)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio from one sample rate to another.

        Uses scipy's signal processing for high-quality resampling.

        Args:
            audio: Input audio array.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio array.
        """
        if orig_sr == target_sr:
            return audio

        scipy_signal = import_module("scipy.signal")

        # Use rational approximation for common sample rate conversions
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd

        return np.asarray(scipy_signal.resample_poly(audio, up, down), dtype=np.float32)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Piper TTS is available in the current environment.

        Returns:
            True if piper is installed, False otherwise.
        """
        return _PIPER_AVAILABLE
