"""TTS synthesis caching with LRU eviction."""

from __future__ import annotations

import json
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from wakeword_workbench.tts.base import TTSResult

if TYPE_CHECKING:
    pass

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "wakeword_workbench" / "tts"
DEFAULT_MAX_SIZE_MB = 1000


class TTSCache:
    """LRU cache for TTS synthesis results.

    Stores cached audio in ~/.cache/wakeword_workbench/tts/ using:
    - SHA256 hash-based keys for deterministic file names
    - .npy files for audio data (secure, no pickle)
    - .json files for metadata (sample_rate, duration)

    Evicts least-recently-used entries when size limit is exceeded.

    Example:
        cache = TTSCache(max_size_mb=500)
        result = cache.get("hello world", "af_heart", "kokoro")
        if result is None:
            result = backend.synthesize("hello world")
            cache.put("hello world", "af_heart", "kokoro", result)
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
    ) -> None:
        """Initialize the TTS cache.

        Args:
            cache_dir: Directory for cache storage. Defaults to
                ~/.cache/wakeword_workbench/tts/
            max_size_mb: Maximum cache size in megabytes. LRU eviction
                triggers when this limit is approached.
        """
        self._cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._lru: OrderedDict[str, float] = OrderedDict()
        self._load_lru_index()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get(self, text: str, voice: str, backend: str, speed: float = 1.0) -> TTSResult | None:
        """Retrieve a cached TTS result.

        Args:
            text: The text that was synthesized.
            voice: The voice identifier used.
            backend: The TTS backend name.
            speed: Synthesis speed (default 1.0).

        Returns:
            TTSResult if found in cache, None otherwise.
            Updates LRU order on cache hit.
        """
        key = self._make_key(text, voice, backend, speed)
        audio_path = self._audio_path(key)
        meta_path = self._meta_path(key)

        if not audio_path.exists() or not meta_path.exists():
            return None

        try:
            audio = np.load(str(audio_path), mmap_mode="r")
            audio = audio.astype(np.float32)
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            result = TTSResult(
                audio=audio,
                sample_rate=meta["sample_rate"],
                duration=meta["duration"],
            )
            self._touch(key)
            return result
        except Exception:
            # Corrupted cache entry — treat as miss
            self._remove_entry(key)
            return None

    def put(
        self,
        text: str,
        voice: str,
        backend: str,
        result: TTSResult,
        speed: float = 1.0,
    ) -> None:
        """Store a TTS result in the cache.

        Args:
            text: The text that was synthesized.
            voice: The voice identifier used.
            backend: The TTS backend name.
            result: The synthesis result to cache.
            speed: Synthesis speed (default 1.0).
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._make_key(text, voice, backend, speed)
        audio_path = self._audio_path(key)
        meta_path = self._meta_path(key)

        np.save(str(audio_path), result.audio)
        meta = {
            "sample_rate": result.sample_rate,
            "duration": result.duration,
            "text": text,
            "voice": voice,
            "backend": backend,
            "speed": speed,
            "cached_at": time.time(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        self._touch(key)
        self._enforce_limit()

    def clear(self) -> None:
        """Remove all cached files and reset the cache."""
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lru.clear()
        self._save_lru_index()

    def get_stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dict with keys:
                - num_entries: Total cached entries.
                - size_mb: Total cache size in MB.
                - max_size_mb: Size limit in MB.
                - oldest_entry: Timestamp of oldest cached entry.
                - newest_entry: Timestamp of newest cached entry.
        """
        if not self._cache_dir.exists():
            return {
                "num_entries": 0,
                "size_mb": 0.0,
                "max_size_mb": self._max_size_bytes / (1024 * 1024),
                "oldest_entry": None,
                "newest_entry": None,
            }

        meta_files = [f for f in self._cache_dir.glob("*.json") if f.name != ".lru_index.json"]
        cached_at_list = []
        total_size = 0

        for meta_file in meta_files:
            try:
                total_size += meta_file.stat().st_size
                audio_file = meta_file.with_suffix(".npy")
                if audio_file.exists():
                    total_size += audio_file.stat().st_size
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                cached_at_list.append(meta.get("cached_at", 0))
            except Exception:
                continue

        cached_at_list.sort()
        return {
            "num_entries": len(cached_at_list),
            "size_mb": total_size / (1024 * 1024),
            "max_size_mb": self._max_size_bytes / (1024 * 1024),
            "oldest_entry": cached_at_list[0] if cached_at_list else None,
            "newest_entry": cached_at_list[-1] if cached_at_list else None,
        }

    # -------------------------------------------------------------------------
    # Key generation
    # -------------------------------------------------------------------------

    def _make_key(
        self,
        text: str,
        voice: str,
        backend: str,
        speed: float = 1.0,
    ) -> str:
        """Generate a SHA256 cache key from synthesis parameters.

        Args:
            text: Input text.
            voice: Voice identifier.
            backend: Backend name.
            speed: Synthesis speed.

        Returns:
            64-character hex string (SHA256).
        """
        import hashlib

        data = f"{text}\x00{voice}\x00{backend}\x00{speed}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------

    def _audio_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.npy"

    def _meta_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    # -------------------------------------------------------------------------
    # LRU management
    # -------------------------------------------------------------------------

    def _touch(self, key: str) -> None:
        """Update LRU order on access."""
        if key in self._lru:
            del self._lru[key]
        self._lru[key] = time.time()
        self._save_lru_index()

    def _remove_entry(self, key: str) -> None:
        """Remove a single cache entry."""
        audio_path = self._audio_path(key)
        meta_path = self._meta_path(key)
        audio_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        self._lru.pop(key, None)
        self._save_lru_index()

    def _enforce_limit(self) -> None:
        """Evict LRU entries until cache is under size limit."""
        while self._current_size() > self._max_size_bytes and self._lru:
            oldest_key = next(iter(self._lru))
            self._remove_entry(oldest_key)

    def _current_size(self) -> int:
        """Calculate current cache size in bytes."""
        if not self._cache_dir.exists():
            return 0
        total = 0
        for f in self._cache_dir.iterdir():
            if f.is_file():
                total += f.stat().st_size
        return total

    # -------------------------------------------------------------------------
    # LRU index persistence
    # -------------------------------------------------------------------------

    def _load_lru_index(self) -> None:
        """Load LRU index from disk."""
        index_path = self._cache_dir / ".lru_index.json"
        if index_path.exists():
            try:
                with open(index_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._lru = OrderedDict(
                    (k, data[k]) for k in sorted(data, key=data.__getitem__) if k
                )
            except Exception:
                self._lru = OrderedDict()
        else:
            self._lru = OrderedDict()

    def _save_lru_index(self) -> None:
        """Persist LRU index to disk."""
        index_path = self._cache_dir / ".lru_index.json"
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(dict(self._lru), f)
        except Exception:
            pass


# -------------------------------------------------------------------------
# Module-level singleton (lazy)
# -------------------------------------------------------------------------

_default_cache: TTSCache | None = None


def get_default_cache() -> TTSCache:
    """Get the module-level default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = TTSCache()
    return _default_cache


def reset_default_cache() -> None:
    """Reset the module-level cache (useful for testing)."""
    global _default_cache
    _default_cache = None
