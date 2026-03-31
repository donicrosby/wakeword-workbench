"""Tests for TTS registry module."""

import numpy as np
import pytest

from wakeword_workbench.tts import (
    BackendNotAvailableError,
    UnknownBackendError,
    TTSBackend,
    TTSResult,
    get_backend,
    list_all_backends,
    list_available_backends,
    register_backend,
    unregister_backend,
)


# Mock backend for testing
class MockAvailableBackend(TTSBackend):
    """A mock backend that is always available."""

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


class MockUnavailableBackend(TTSBackend):
    """A mock backend that is never available."""

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


# --- Registration Tests ---


class TestRegisterBackend:
    """Tests for register_backend function."""

    def test_register_backend_adds_to_registry(self) -> None:
        """register_backend should add the backend to the registry."""
        test_name = "test_register_available"

        # Ensure it's not registered
        unregister_backend(test_name)

        try:
            register_backend(test_name, MockAvailableBackend)
            available = list_all_backends()
            assert test_name in available
        finally:
            unregister_backend(test_name)

    def test_register_empty_name_raises_error(self) -> None:
        """register_backend should reject empty names."""
        with pytest.raises(UnknownBackendError, match="Backend name cannot be empty"):
            register_backend("", MockAvailableBackend)

    def test_register_duplicate_raises_error(self) -> None:
        """register_backend should reject duplicate registration."""
        test_name = "test_duplicate"

        # Ensure it's not registered
        unregister_backend(test_name)

        try:
            register_backend(test_name, MockAvailableBackend)
            with pytest.raises(Exception, match="already registered"):
                register_backend(test_name, MockAvailableBackend)
        finally:
            unregister_backend(test_name)


# --- Unregistration Tests ---


class TestUnregisterBackend:
    """Tests for unregister_backend function."""

    def test_unregister_removes_backend(self) -> None:
        """unregister_backend should remove the backend from the registry."""
        test_name = "test_unregister"

        # Register first
        register_backend(test_name, MockAvailableBackend)
        assert test_name in list_all_backends()

        # Unregister
        unregister_backend(test_name)
        assert test_name not in list_all_backends()

    def test_unregister_nonexistent_does_not_raise(self) -> None:
        """unregister_backend should not raise for unknown backends."""
        unregister_backend("nonexistent_backend_xyz")  # Should not raise


# --- List Backends Tests ---


class TestListBackends:
    """Tests for list_available_backends and list_all_backends."""

    def test_list_available_filters_unavailable(self) -> None:
        """list_available_backends should only return available backends."""
        available_name = "test_available_list"
        unavailable_name = "test_unavailable_list"

        unregister_backend(available_name)
        unregister_backend(unavailable_name)

        try:
            register_backend(available_name, MockAvailableBackend)
            register_backend(unavailable_name, MockUnavailableBackend)

            available = list_available_backends()
            assert available_name in available
            assert unavailable_name not in available
        finally:
            unregister_backend(available_name)
            unregister_backend(unavailable_name)

    def test_list_all_includes_unavailable(self) -> None:
        """list_all_backends should return all registered backends."""
        available_name = "test_all_available"
        unavailable_name = "test_all_unavailable"

        unregister_backend(available_name)
        unregister_backend(unavailable_name)

        try:
            register_backend(available_name, MockAvailableBackend)
            register_backend(unavailable_name, MockUnavailableBackend)

            all_backends = list_all_backends()
            assert available_name in all_backends
            assert unavailable_name in all_backends
        finally:
            unregister_backend(available_name)
            unregister_backend(unavailable_name)

    def test_list_backends_returns_sorted(self) -> None:
        """list_available_backends should return sorted list."""
        backends = list_available_backends()
        assert backends == sorted(backends)


# --- Get Backend Tests ---


class TestGetBackend:
    """Tests for get_backend function."""

    def test_get_backend_returns_instance(self) -> None:
        """get_backend should return an instance of the backend."""
        test_name = "test_get_instance"

        unregister_backend(test_name)

        try:
            register_backend(test_name, MockAvailableBackend)
            backend = get_backend(test_name)
            assert isinstance(backend, MockAvailableBackend)
        finally:
            unregister_backend(test_name)

    def test_get_unknown_backend_raises_error(self) -> None:
        """get_backend should raise UnknownBackendError for unknown names."""
        with pytest.raises(UnknownBackendError, match="Unknown TTS backend"):
            get_backend("definitely_does_not_exist_backend")

    def test_get_unknown_shows_available(self) -> None:
        """Error message should list available backends."""
        with pytest.raises(UnknownBackendError, match="Available backends:"):
            get_backend("unknown_backend_xyz")

    def test_get_unavailable_backend_raises_error(self) -> None:
        """get_backend should raise BackendNotAvailableError for unavailable backends."""
        test_name = "test_unavailable_get"

        unregister_backend(test_name)

        try:
            register_backend(test_name, MockUnavailableBackend)
            with pytest.raises(BackendNotAvailableError, match="not available"):
                get_backend(test_name)
        finally:
            unregister_backend(test_name)


# --- Lazy Loading Tests ---


class TestLazyLoading:
    """Tests for lazy loading behavior."""

    def test_discovery_only_runs_once(self) -> None:
        """Auto-discovery should only run once."""
        from wakeword_workbench.tts.registry import _DISCOVERY_RUN

        initial_backends = list_all_backends()

        # Run again - should be the same
        second_backends = list_all_backends()
        assert initial_backends == second_backends


# --- Integration Tests ---


class TestRegistryIntegration:
    """Integration tests for the registry."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: register, list, get, unregister."""
        test_name = "test_integration"

        # Ensure clean state
        unregister_backend(test_name)

        try:
            # Should not be available before registration
            with pytest.raises(UnknownBackendError):
                get_backend(test_name)

            # Register
            register_backend(test_name, MockAvailableBackend)

            # Should appear in lists
            assert test_name in list_all_backends()

            # Should be gettable
            backend = get_backend(test_name)
            assert isinstance(backend, MockAvailableBackend)

            # Unregister
            unregister_backend(test_name)

            # Should not be available after unregistration
            with pytest.raises(UnknownBackendError):
                get_backend(test_name)
        finally:
            unregister_backend(test_name)

    def test_example_usage_from_docstring(self) -> None:
        """Test the example usage from the module docstring works."""
        test_name = "test_example"

        unregister_backend(test_name)

        try:
            # Register a test backend
            register_backend(test_name, MockAvailableBackend)

            # Test the pattern from the docstring
            backends = list_available_backends()
            assert test_name in backends

            backend = get_backend(test_name)
            audio = backend.synthesize("hello")
            assert isinstance(audio, TTSResult)
            assert audio.sample_rate == 16000
        finally:
            unregister_backend(test_name)


# --- Error Handling Tests ---


class TestErrorHandling:
    """Tests for error handling."""

    def test_unknown_backend_error_inheritance(self) -> None:
        """UnknownBackendError should inherit from TTSError."""
        from wakeword_workbench.tts.base import TTSError

        error = UnknownBackendError("test")
        assert isinstance(error, TTSError)
        assert isinstance(error, Exception)

    def test_error_can_be_caught_as_tts_error(self) -> None:
        """UnknownBackendError should be catchable as TTSError."""
        from wakeword_workbench.tts.base import TTSError

        with pytest.raises(TTSError):
            get_backend("nonexistent_backend_for_error_test")

    def test_backend_not_available_error_inheritance(self) -> None:
        """BackendNotAvailableError should still work from base module."""
        test_name = "test_error_inheritance"

        unregister_backend(test_name)

        try:
            register_backend(test_name, MockUnavailableBackend)
            with pytest.raises(BackendNotAvailableError):
                get_backend(test_name)
        finally:
            unregister_backend(test_name)
