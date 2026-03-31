"""TTS backend registry with lazy loading and auto-discovery."""

from __future__ import annotations

import importlib

from .base import BackendNotAvailableError, TTSBackend, TTSError

# Internal registry mapping backend name to backend class
_BACKENDS: dict[str, type[TTSBackend]] = {}

# Flag to track if auto-discovery has been run
_DISCOVERY_RUN: bool = False

# Backend modules to try during auto-discovery
_BACKEND_MODULES: list[tuple[str, str]] = [
    ("kokoro", "wakeword_workbench.tts.kokoro_backend"),
    ("piper", "wakeword_workbench.tts.piper_backend"),
]


class UnknownBackendError(TTSError):
    """Raised when a requested TTS backend is not registered.

    Inherits from TTSError for backwards compatibility.
    """

    pass


def _discover_backends() -> None:
    """Auto-discover and register available TTS backends.

    This function attempts to import backend modules and register them
    if they are available (is_available() returns True).
    Lazy loading ensures backends are only imported when needed.
    """
    global _DISCOVERY_RUN
    if _DISCOVERY_RUN:
        return

    _DISCOVERY_RUN = True

    for backend_name, module_name in _BACKEND_MODULES:
        # Skip if already registered
        if backend_name in _BACKENDS:
            continue

        try:
            # Lazy import - only loads the module when discovering
            module = importlib.import_module(module_name)

            # Get the backend class from the module
            # Convention: module contains a class named {BackendName}Backend
            backend_class_name = f"{backend_name.title().replace('kokoro', 'Kokoro').replace('piper', 'Piper')}Backend"
            if hasattr(module, backend_class_name):
                backend_cls = getattr(module, backend_class_name)
                if isinstance(backend_cls, type) and issubclass(backend_cls, TTSBackend):
                    # Only register if available
                    if backend_cls.is_available():
                        _BACKENDS[backend_name] = backend_cls
        except ImportError:
            # Backend module not available (optional dependency not installed)
            pass
        except Exception:
            # Silently skip backends that fail to load
            # This allows the registry to work even if some backends are broken
            pass


def register_backend(name: str, backend_class: type[TTSBackend]) -> None:
    """Register a TTS backend class under a given name.

    Args:
        name: The name to register the backend under (e.g., "kokoro", "piper").
        backend_class: The TTSBackend subclass to register.

    Raises:
        UnknownBackendError: If name is empty.
        TTSError: If the backend is already registered.
    """
    if not name:
        raise UnknownBackendError("Backend name cannot be empty")
    if name in _BACKENDS:
        raise TTSError(f"Backend '{name}' is already registered")
    _BACKENDS[name] = backend_class


def unregister_backend(name: str) -> None:
    """Unregister a TTS backend by name.

    Args:
        name: The name of the backend to unregister.
    """
    _BACKENDS.pop(name, None)


def list_available_backends() -> list[str]:
    """List all registered and available TTS backends.

    Runs auto-discovery if not already done, then returns backends
    that report is_available() as True.

    Returns:
        List of available backend names (e.g., ["kokoro"]).
    """
    _discover_backends()

    available = []
    for name, backend_cls in _BACKENDS.items():
        if backend_cls.is_available():
            available.append(name)
    return sorted(available)


def list_all_backends() -> list[str]:
    """List all registered TTS backends (including unavailable ones).

    This is useful for debugging and diagnostics.

    Returns:
        List of all registered backend names.
    """
    _discover_backends()
    return sorted(_BACKENDS.keys())


def get_backend(name: str) -> TTSBackend:
    """Get an instance of a TTS backend by name.

    Args:
        name: The name of the backend to instantiate (e.g., "kokoro").

    Returns:
        An instance of the requested TTSBackend.

    Raises:
        UnknownBackendError: If the backend name is not registered.
        BackendNotAvailableError: If the backend is not available
            (e.g., optional dependencies not installed).
    """
    _discover_backends()

    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys())) or "none"
        raise UnknownBackendError(f"Unknown TTS backend: '{name}'. Available backends: {available}")

    backend_cls = _BACKENDS[name]

    if not backend_cls.is_available():
        raise BackendNotAvailableError(
            f"TTS backend '{name}' is not available. Install the required dependencies."
        )

    return backend_cls()


def get_registered_backends() -> dict[str, type[TTSBackend]]:
    """Get a copy of the internal registry.

    Returns:
        A copy of the backend registry dictionary.
    """
    _discover_backends()
    return _BACKENDS.copy()
