"""Tests for TTS base module."""

import numpy as np
import pytest

from wakeword_workbench.tts.base import (
    BackendNotAvailableError,
    TTSBackend,
    TTSError,
    TTSResult,
    create_backend,
)

# --- TTSResult Tests ---


class TestTTSResultValidation:
    """Tests for TTSResult validation."""

    def test_valid_result(self) -> None:
        """Valid TTSResult should be created successfully."""
        audio = np.random.uniform(-0.5, 0.5, size=16000).astype(np.float32)
        result = TTSResult(audio=audio, sample_rate=16000, duration=1.0)
        assert result.audio is audio
        assert result.sample_rate == 16000
        assert result.duration == 1.0

    def test_audio_must_be_float32(self) -> None:
        """TTSResult should reject non-float32 audio."""
        audio = np.random.randn(16000).astype(np.float64) * 0.5
        with pytest.raises(TTSError, match="audio must be float32"):
            TTSResult(audio=audio, sample_rate=16000, duration=1.0)

    def test_audio_must_be_normalized(self) -> None:
        """TTSResult should reject audio outside [-1, 1] range."""
        audio = np.array([0.5, 1.5, -0.5], dtype=np.float32)  # 1.5 is out of range
        with pytest.raises(TTSError, match="audio samples must be normalized to"):
            TTSResult(audio=audio, sample_rate=16000, duration=0.0001875)

    def test_sample_rate_must_be_positive(self) -> None:
        """TTSResult should reject non-positive sample rates."""
        audio = np.zeros(100, dtype=np.float32)
        with pytest.raises(TTSError, match="sample_rate must be positive"):
            TTSResult(audio=audio, sample_rate=0, duration=0.01)
        with pytest.raises(TTSError, match="sample_rate must be positive"):
            TTSResult(audio=audio, sample_rate=-16000, duration=-0.01)

    def test_duration_must_be_positive(self) -> None:
        """TTSResult should reject non-positive duration."""
        audio = np.zeros(100, dtype=np.float32)
        with pytest.raises(TTSError, match="duration must be positive"):
            TTSResult(audio=audio, sample_rate=16000, duration=0.0)
        with pytest.raises(TTSError, match="duration must be positive"):
            TTSResult(audio=audio, sample_rate=16000, duration=-1.0)

    def test_duration_must_match_audio_length(self) -> None:
        """TTSResult should reject when duration doesn't match audio length."""
        audio = np.zeros(32000, dtype=np.float32)  # 2 seconds at 16kHz
        with pytest.raises(TTSError, match="does not match audio length"):
            TTSResult(audio=audio, sample_rate=16000, duration=1.0)  # Wrong duration


# --- TTSBackend ABC Tests ---


class TestTTSBackendABC:
    """Tests for TTSBackend abstract base class."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """TTSBackend should raise TypeError when instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            TTSBackend()  # type: ignore

    def test_concrete_backend_can_be_instantiated(self) -> None:
        """A concrete implementation of TTSBackend should be instantiable."""

        class ConcreteBackend(TTSBackend):
            def synthesize(self, text: str) -> TTSResult:
                return TTSResult(
                    audio=np.zeros(16000, dtype=np.float32),
                    sample_rate=16000,
                    duration=1.0,
                )

            def set_voice(self, voice: str) -> None:
                pass

            def list_voices(self) -> list[str]:
                return ["voice1", "voice2"]

        backend = ConcreteBackend()
        assert backend is not None

    def test_abstract_methods_must_be_implemented(self) -> None:
        """Subclass without all abstract methods should not be instantiable."""

        class IncompleteBackend(TTSBackend):
            def synthesize(self, text: str) -> TTSResult:
                return TTSResult(
                    audio=np.zeros(16000, dtype=np.float32),
                    sample_rate=16000,
                    duration=1.0,
                )

            # Missing set_voice and list_voices

        with pytest.raises(TypeError, match="abstract"):
            IncompleteBackend()  # type: ignore

    def test_is_available_default_returns_true(self) -> None:
        """TTSBackend.is_available() should return True by default."""
        assert TTSBackend.is_available() is True

    def test_is_available_can_be_overridden(self) -> None:
        """is_available() should be overridable in subclasses."""

        class UnavailableBackend(TTSBackend):
            @classmethod
            def is_available(cls) -> bool:
                return False

            def synthesize(self, text: str) -> TTSResult:
                return TTSResult(
                    audio=np.zeros(16000, dtype=np.float32),
                    sample_rate=16000,
                    duration=1.0,
                )

            def set_voice(self, voice: str) -> None:
                pass

            def list_voices(self) -> list[str]:
                return []

        assert UnavailableBackend.is_available() is False


# --- Factory Function Tests ---


class TestCreateBackend:
    """Tests for create_backend factory function."""

    def test_unknown_backend_raises_tts_error(self) -> None:
        """create_backend should raise TTSError for unknown backend names."""
        with pytest.raises(TTSError, match="Unknown TTS backend"):
            create_backend("nonexistent")

    def test_unknown_backend_shows_available_backends(self) -> None:
        """Error message should list available backends when possible."""
        with pytest.raises(TTSError, match="Available backends:"):
            create_backend("unknown_backend")

    def test_backend_not_available_raises_specific_error(self) -> None:
        """Backend that returns is_available=False should raise BackendNotAvailableError."""

        class MockUnavailableBackend(TTSBackend):
            @classmethod
            def is_available(cls) -> bool:
                return False

            def synthesize(self, text: str) -> TTSResult:
                return TTSResult(
                    audio=np.zeros(16000, dtype=np.float32),
                    sample_rate=16000,
                    duration=1.0,
                )

            def set_voice(self, voice: str) -> None:
                pass

            def list_voices(self) -> list[str]:
                return []

        MockUnavailableBackend.register_backend("mock_unavailable")

        try:
            with pytest.raises(BackendNotAvailableError, match="not available"):
                create_backend("mock_unavailable")
        finally:
            MockUnavailableBackend.unregister_backend("mock_unavailable")

    def test_registered_backend_can_be_created(self) -> None:
        """A registered backend should be creatable via the factory."""

        class MockBackend(TTSBackend):
            def synthesize(self, text: str) -> TTSResult:
                return TTSResult(
                    audio=np.zeros(16000, dtype=np.float32),
                    sample_rate=16000,
                    duration=1.0,
                )

            def set_voice(self, voice: str) -> None:
                pass

            def list_voices(self) -> list[str]:
                return ["mock_voice"]

        MockBackend.register_backend("mock")

        try:
            backend = create_backend("mock")
            assert isinstance(backend, MockBackend)
            assert backend.list_voices() == ["mock_voice"]
        finally:
            MockBackend.unregister_backend("mock")

    def test_cannot_register_empty_backend_name(self) -> None:
        """register_backend should reject empty names."""

        class TestBackend(TTSBackend):
            def synthesize(self, text: str) -> TTSResult:
                return TTSResult(
                    audio=np.zeros(16000, dtype=np.float32),
                    sample_rate=16000,
                    duration=1.0,
                )

            def set_voice(self, voice: str) -> None:
                pass

            def list_voices(self) -> list[str]:
                return []

        with pytest.raises(TTSError, match="Backend name cannot be empty"):
            TestBackend.register_backend("")

    def test_cannot_register_same_backend_twice(self) -> None:
        """register_backend should prevent duplicate registration."""

        class TestBackend(TTSBackend):
            def synthesize(self, text: str) -> TTSResult:
                return TTSResult(
                    audio=np.zeros(16000, dtype=np.float32),
                    sample_rate=16000,
                    duration=1.0,
                )

            def set_voice(self, voice: str) -> None:
                pass

            def list_voices(self) -> list[str]:
                return []

        TestBackend.register_backend("duplicate_test")

        try:
            with pytest.raises(TTSError, match="already registered"):
                TestBackend.register_backend("duplicate_test")
        finally:
            TestBackend.unregister_backend("duplicate_test")

    def test_unregister_removes_backend(self) -> None:
        """unregister_backend should remove the backend from registry."""
        TTSBackend.unregister_backend("mock")  # Should not raise

        class AnotherMock(TTSBackend):
            def synthesize(self, text: str) -> TTSResult:
                return TTSResult(
                    audio=np.zeros(16000, dtype=np.float32),
                    sample_rate=16000,
                    duration=1.0,
                )

            def set_voice(self, voice: str) -> None:
                pass

            def list_voices(self) -> list[str]:
                return []

        AnotherMock.register_backend("unregister_test")

        try:
            # Should work
            backend = create_backend("unregister_test")
            assert isinstance(backend, AnotherMock)

            # Unregister
            AnotherMock.unregister_backend("unregister_test")

            # Should now fail
            with pytest.raises(TTSError, match="Unknown TTS backend"):
                create_backend("unregister_test")
        finally:
            AnotherMock.unregister_backend("unregister_test")


# --- Exception Inheritance Tests ---


class TestExceptionInheritance:
    """Tests for exception class hierarchy."""

    def test_tts_error_inherits_from_exception(self) -> None:
        """TTSError should be usable as a standard exception."""
        error = TTSError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"

    def test_backend_not_available_error_inherits_from_tts_error(self) -> None:
        """BackendNotAvailableError should inherit from TTSError."""
        error = BackendNotAvailableError("backend not found")
        assert isinstance(error, TTSError)
        assert isinstance(error, Exception)
        assert str(error) == "backend not found"

    def test_can_catch_tts_error_for_all_tts_exceptions(self) -> None:
        """Catching TTSError should also catch BackendNotAvailableError."""
        with pytest.raises(TTSError):
            raise BackendNotAvailableError("caught by TTSError")
