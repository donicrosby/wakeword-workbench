"""Text-to-Speech backend modules."""

from .base import (
    BackendNotAvailableError,
    TTSBackend,
    TTSError,
    TTSResult,
)
from .cache import (
    DEFAULT_CACHE_DIR,
    TTSCache,
    get_default_cache,
    reset_default_cache,
)
from .kokoro_backend import KokoroBackend
from .registry import (
    UnknownBackendError,
    get_backend,
    list_all_backends,
    list_available_backends,
    register_backend,
    unregister_backend,
)

__all__ = [
    "BackendNotAvailableError",
    "DEFAULT_CACHE_DIR",
    "get_backend",
    "get_default_cache",
    "KokoroBackend",
    "list_all_backends",
    "list_available_backends",
    "register_backend",
    "reset_default_cache",
    "TTSBackend",
    "TTSCache",
    "TTSError",
    "TTSResult",
    "UnknownBackendError",
    "unregister_backend",
]
