"""Piper TTS backend implementation."""

from __future__ import annotations

import urllib.request
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import soundfile as sf

from wakeword_workbench.tts.base import (
    BackendNotAvailableError,
    TTSBackend,
    TTSError,
    TTSResult,
)

# Try to import piper, but allow graceful fallback
try:
    from piper import PiperVoice

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
        self._voice: "PiperVoice" | None = None  # type: ignore[name-defined]

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
            self._voice = PiperVoice.load(str(self.model_path))  # type: ignore[union-attr]
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

        print(f"Downloading Piper default model to {model_path}...")
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

        try:
            # Synthesize WAV audio
            wav_buffer = BytesIO()
            self._voice.synthesize_wav(text, wav_buffer)
            wav_buffer.seek(0)

            # Read the WAV data using soundfile
            audio_data, sample_rate = sf.read(wav_buffer, dtype="float32")

            # Handle stereo audio (convert to mono)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Ensure float32
            audio_data = audio_data.astype(np.float32)

            # Resample from 22050 Hz to 16000 Hz using scipy
            audio_data = self._resample(audio_data, sample_rate, self._TARGET_SAMPLE_RATE)

            # Clip to ensure normalized audio (handles numerical errors from resampling)
            audio_data = np.clip(audio_data, -1.0, 1.0)

            # Calculate duration
            duration = len(audio_data) / self._TARGET_SAMPLE_RATE

            return TTSResult(
                audio=audio_data,
                sample_rate=self._TARGET_SAMPLE_RATE,
                duration=duration,
            )

        except Exception as e:
            raise TTSError(f"Piper synthesis failed: {e}") from e

    def set_voice(self, voice: str) -> None:
        """Set the voice for synthesis.

        Note:
            Piper requires voice selection at model load time.
            Changing the voice requires loading a different model file.

        Args:
            voice: The voice identifier (ignored for Piper).

        Raises:
            NotImplementedError: Always, since voice change requires model reload.
        """
        raise NotImplementedError(
            "Piper does not support runtime voice changes. "
            "Load a different model file to use a different voice."
        )

    def list_voices(self) -> list[str]:
        """List available voices for this backend.

        Returns:
            Empty list. Piper voices are determined by model files.
            Use the model directory to scan for available .onnx files.
        """
        return []

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

        from scipy.signal import resample_poly

        # Use rational approximation for common sample rate conversions
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd

        return resample_poly(audio, up, down).astype(np.float32)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Piper TTS is available in the current environment.

        Returns:
            True if piper is installed, False otherwise.
        """
        return _PIPER_AVAILABLE
