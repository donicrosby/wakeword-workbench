"""Thread-safe backend pooling for TTS provider reuse.

Usage pattern:
    Build one ``BackendPool`` per generation run, then call ``get(provider)`` for each
    provider selection. The returned backend instance is cached by provider runtime
    configuration and wrapped with an instance-level lock, so concurrent calls are
    serialized per backend instance.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from wakeword_workbench.config import TTSProviderConfig

from .base import TTSBackend, TTSResult
from .registry import get_backend

BackendPoolKey = tuple[str, float, str, str | None, str | None]


@dataclass
class _LockedBackend(TTSBackend):
    backend: TTSBackend
    lock: threading.Lock

    def synthesize(self, text: str) -> TTSResult:
        with self.lock:
            return self.backend.synthesize(text)

    def set_voice(self, voice: str) -> None:
        with self.lock:
            self.backend.set_voice(voice)

    def list_voices(self) -> list[str]:
        with self.lock:
            return self.backend.list_voices()


class BackendPool:
    """Cache and reuse TTS backends across provider selections.

    ``BackendPool`` is specific to wakeword TTS backends. It caches one backend instance
    per provider runtime configuration key, wraps that backend in a lock-protected proxy,
    and returns the same proxy for repeated requests with the same key.
    """

    def __init__(self, max_voices_per_backend: int = 5) -> None:
        if max_voices_per_backend <= 0:
            raise ValueError("max_voices_per_backend must be positive")

        self.max_voices_per_backend = max_voices_per_backend
        self._pool: dict[BackendPoolKey, TTSBackend] = {}
        self._instance_locks: dict[BackendPoolKey, threading.Lock] = {}
        self._pool_lock = threading.Lock()

    def get(self, provider: TTSProviderConfig) -> TTSBackend:
        key_getter = cast(
            Callable[[], BackendPoolKey] | None, getattr(provider, "backend_instance_key", None)
        )
        if callable(key_getter):
            key = key_getter()
        else:
            backend, _voices, speed, acceleration, device, model_path = provider.backend_cache_key()
            key = (backend, speed, acceleration, device, model_path)

        with self._pool_lock:
            cached = self._pool.get(key)
            if cached is not None:
                return cached

            instance_lock = threading.Lock()
            backend = get_backend(provider.backend)
            locked_backend = _LockedBackend(backend=backend, lock=instance_lock)
            self._instance_locks[key] = instance_lock
            self._pool[key] = locked_backend
            return locked_backend

    def clear(self) -> None:
        with self._pool_lock:
            self._pool.clear()
            self._instance_locks.clear()
