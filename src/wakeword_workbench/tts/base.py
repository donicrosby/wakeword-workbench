"""Base TTS interface and abstract backend class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass
class TTSResult:
    """Result of a TTS synthesis operation.

    Attributes:
        audio: Raw audio samples as numpy array (float32, normalized to [-1, 1]).
        sample_rate: Sample rate of the audio in Hz (target 16000 Hz).
        duration: Duration of the audio in seconds.
    """

    audio: np.ndarray
    sample_rate: int
    duration: float

    def __post_init__(self) -> None:
        """Validate the TTSResult fields."""
        if self.audio.dtype != np.float32:
            raise TTSError(f"audio must be float32, got {self.audio.dtype}")
        if not (-1.0 <= self.audio.min() <= self.audio.max() <= 1.0):
            raise TTSError("audio samples must be normalized to [-1, 1]")
        if self.sample_rate <= 0:
            raise TTSError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.duration <= 0:
            raise TTSError(f"duration must be positive, got {self.duration}")
        # Validate duration matches audio length
        expected_duration = len(self.audio) / self.sample_rate
        if not np.isclose(self.duration, expected_duration, rtol=1e-3):
            raise TTSError(
                f"duration {self.duration} does not match audio length "
                f"({len(self.audio)} samples / {self.sample_rate} Hz = {expected_duration:.3f}s)"
            )


class TTSError(Exception):
    """Base exception for TTS-related errors."""

    pass


class BackendNotAvailableError(TTSError):
    """Raised when a requested TTS backend is not available."""

    pass


class TTSBackend(ABC):
    """Abstract base class for TTS backends.

    All TTS backends must implement the synthesize, set_voice, and list_voices
    methods. The is_available class method can be overridden if the backend
    has external dependencies.

    Example:
        ```python
        from wakeword_workbench.tts.base import TTSBackend, TTSResult

        class MyBackend(TTSBackend):
            def synthesize(self, text: str) -> TTSResult:
                # Implementation here
                pass

            def set_voice(self, voice: str) -> None:
                # Implementation here
                pass

            def list_voices(self) -> list[str]:
                # Implementation here
                return []
        ```
    """

    _backend_registry: ClassVar[dict[str, type[TTSBackend]]] = {}

    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: The text to synthesize.

        Returns:
            TTSResult containing the audio data.

        Raises:
            TTSError: If synthesis fails.
        """
        ...

    @abstractmethod
    def set_voice(self, voice: str) -> None:
        """Set the voice for synthesis.

        Args:
            voice: The voice identifier to use.

        Raises:
            TTSError: If the voice is not available.
        """
        ...

    @abstractmethod
    def list_voices(self) -> list[str]:
        """List available voices for this backend.

        Returns:
            List of voice identifiers available for this backend.
        """
        ...

    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend is available in the current environment.

        Override this method in subclasses to check for required dependencies
        or resources (e.g., installed packages, model files).

        Returns:
            True if the backend can be used, False otherwise.
        """
        return True

    @classmethod
    def register_backend(cls, name: str) -> None:
        """Register this backend with the factory under the given name.

        Args:
            name: The name to register the backend under.
        """
        if not name:
            raise TTSError("Backend name cannot be empty")
        if name in cls._backend_registry:
            raise TTSError(f"Backend '{name}' is already registered")
        cls._backend_registry[name] = cls

    @classmethod
    def unregister_backend(cls, name: str) -> None:
        """Unregister a backend from the factory.

        Args:
            name: The name of the backend to unregister.
        """
        if name in cls._backend_registry:
            del cls._backend_registry[name]


def create_backend(name: str) -> TTSBackend:
    """Factory function to create a TTS backend by name.

    Args:
        name: The name of the backend to create (e.g., "kokoro", "piper").

    Returns:
        An instance of the requested TTSBackend.

    Raises:
        TTSError: If the backend name is unknown or the backend is not available.
        BackendNotAvailableError: If the backend cannot be instantiated
            (e.g., dependencies not installed).
    """
    if name not in TTSBackend._backend_registry:
        available = ", ".join(sorted(TTSBackend._backend_registry.keys()))
        raise TTSError(f"Unknown TTS backend: '{name}'. Available backends: {available or 'none'}")

    backend_cls = TTSBackend._backend_registry[name]

    if not backend_cls.is_available():
        raise BackendNotAvailableError(
            f"TTS backend '{name}' is not available. "
            f"Install the required dependencies or check the installation."
        )

    return backend_cls()
