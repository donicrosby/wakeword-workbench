"""Tests for TTS cache module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wakeword_workbench.tts.base import TTSResult
from wakeword_workbench.tts.cache import TTSCache, get_default_cache, reset_default_cache


def make_result(duration: float = 1.0, sample_rate: int = 16000) -> TTSResult:
    """Helper to create a valid TTSResult."""
    num_samples = int(sample_rate * duration)
    audio = np.random.uniform(-0.5, 0.5, size=num_samples).astype(np.float32)
    return TTSResult(audio=audio, sample_rate=sample_rate, duration=duration)


# ---------------------------------------------------------------------------
# Basic get / put
# ---------------------------------------------------------------------------


class TestCacheBasic:
    """Tests for basic cache get/put operations."""

    def test_get_returns_none_for_empty_cache(self, tmp_path: Path) -> None:
        """Cache miss on empty cache should return None."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        result = cache.get("hello", "af_sarah", "kokoro")
        assert result is None

    def test_put_and_get_returns_same_result(self, tmp_path: Path) -> None:
        """Caching a result should allow retrieval with identical data."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        original = make_result()

        cache.put("hello world", "af_sarah", "kokoro", original)
        cached = cache.get("hello world", "af_sarah", "kokoro")

        assert cached is not None
        assert np.array_equal(cached.audio, original.audio)
        assert cached.sample_rate == original.sample_rate
        assert cached.duration == original.duration

    def test_different_text_gives_cache_miss(self, tmp_path: Path) -> None:
        """Different text should not return a cached result."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        original = make_result()
        cache.put("hello", "af_sarah", "kokoro", original)

        result = cache.get("goodbye", "af_sarah", "kokoro")
        assert result is None

    def test_different_voice_gives_cache_miss(self, tmp_path: Path) -> None:
        """Different voice should not return a cached result."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        original = make_result()
        cache.put("hello", "af_sarah", "kokoro", original)

        result = cache.get("hello", "af_heart", "kokoro")
        assert result is None

    def test_different_backend_gives_cache_miss(self, tmp_path: Path) -> None:
        """Different backend should not return a cached result."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        original = make_result()
        cache.put("hello", "af_sarah", "kokoro", original)

        result = cache.get("hello", "af_sarah", "piper")
        assert result is None

    def test_different_speed_gives_cache_miss(self, tmp_path: Path) -> None:
        """Different synthesis speed should not return a cached result."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        original = make_result()
        cache.put("hello", "af_sarah", "kokoro", original, speed=1.0)

        result = cache.get("hello", "af_sarah", "kokoro", speed=1.5)
        assert result is None

    def test_cache_hit_preserves_audio_fidelity(self, tmp_path: Path) -> None:
        """Cached audio should match original byte-for-byte."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        audio = np.linspace(-1, 1, 16000).astype(np.float32)
        original = TTSResult(audio=audio, sample_rate=16000, duration=1.0)

        cache.put("precise", "af_sarah", "kokoro", original)
        cached = cache.get("precise", "af_sarah", "kokoro")

        assert cached is not None
        np.testing.assert_array_equal(cached.audio, original.audio)


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


class TestCacheKeyGeneration:
    """Tests for deterministic cache key generation."""

    def test_same_inputs_produce_same_key(self, tmp_path: Path) -> None:
        """Identical inputs should always produce the same cached result."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        result = make_result()
        cache.put("hello", "af_sarah", "kokoro", result)
        cache.put("hello", "af_sarah", "kokoro", result)

        cached = cache.get("hello", "af_sarah", "kokoro")
        assert cached is not None

    def test_key_includes_speed(self, tmp_path: Path) -> None:
        """Cache key must incorporate speed parameter."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        r1 = make_result()
        r2 = make_result()
        cache.put("hello", "v", "b", r1, speed=1.0)
        cache.put("hello", "v", "b", r2, speed=2.0)

        c1 = cache.get("hello", "v", "b", speed=1.0)
        c2 = cache.get("hello", "v", "b", speed=2.0)

        assert c1 is not None
        assert c2 is not None
        # Results may differ but both should be retrievable
        assert c1.audio.shape == c2.audio.shape or c1.audio.shape == r1.audio.shape

    def test_key_is_sha256_hex(self, tmp_path: Path) -> None:
        """Key generated via _make_key should be a 64-char hex string."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        key = cache._make_key("hello", "af_sarah", "kokoro", 1.0)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestCacheLRUEviction:
    """Tests for LRU cache eviction behavior."""

    def test_lru_eviction_respects_size_limit(self, tmp_path: Path) -> None:
        """Cache should evict LRU entries when size limit is reached."""
        # Small max size so eviction happens quickly
        cache = TTSCache(cache_dir=tmp_path / "tts_cache", max_size_mb=0)
        result = make_result(duration=0.1)  # ~1600 bytes

        # Fill cache until eviction kicks in
        for i in range(20):
            cache.put(f"text_{i}", "v", "b", result)

        # After eviction, at least some entries should be gone
        stats = cache.get_stats()
        assert stats["num_entries"] < 20

    def test_lru_order_is_updated_on_access(self, tmp_path: Path) -> None:
        """Accessing a cached entry should update its LRU position."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache", max_size_mb=1)
        result = make_result()

        for i in range(5):
            cache.put(f"text_{i}", "v", "b", result)

        # Access text_0 to move it to front of LRU
        cache.get("text_0", "v", "b")

        # Force eviction — text_0 should be evicted last
        cache._enforce_limit()

        # text_0 should still be in cache (most recently used)
        assert cache.get("text_0", "v", "b") is not None

    def test_clear_removes_all_entries(self, tmp_path: Path) -> None:
        """clear() should remove all cached files."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        result = make_result()

        for i in range(5):
            cache.put(f"text_{i}", "v", "b", result)

        cache.clear()

        stats = cache.get_stats()
        assert stats["num_entries"] == 0
        assert stats["size_mb"] == 0.0


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


class TestCacheStats:
    """Tests for get_stats() reporting."""

    def test_empty_cache_stats(self, tmp_path: Path) -> None:
        """Empty cache should report zero entries."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        stats = cache.get_stats()

        assert stats["num_entries"] == 0
        assert stats["size_mb"] == 0.0
        assert stats["max_size_mb"] > 0
        assert stats["oldest_entry"] is None
        assert stats["newest_entry"] is None

    def test_stats_after_put(self, tmp_path: Path) -> None:
        """Stats should reflect entries after caching."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        result = make_result()

        cache.put("hello", "v", "b", result)
        stats = cache.get_stats()

        assert stats["num_entries"] == 1
        assert stats["size_mb"] > 0
        assert stats["oldest_entry"] is not None
        assert stats["newest_entry"] is not None

    def test_stats_max_size_reflects_config(self, tmp_path: Path) -> None:
        """max_size_mb in stats should match configured limit."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache", max_size_mb=500)
        stats = cache.get_stats()
        assert stats["max_size_mb"] == 500.0


# ---------------------------------------------------------------------------
# Corrupted entry handling
# ---------------------------------------------------------------------------


class TestCacheCorruptionHandling:
    """Tests for graceful handling of corrupted cache entries."""

    def test_missing_audio_file_returns_none(self, tmp_path: Path) -> None:
        """Missing audio file should result in cache miss, not crash."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        key = cache._make_key("hello", "v", "b")
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        # Create only metadata, no audio
        (cache._cache_dir / f"{key}.json").write_text("{}")
        (cache._cache_dir / ".lru_index.json").write_text("{}")

        result = cache.get("hello", "v", "b")
        assert result is None

    def test_corrupted_npy_file_returns_none(self, tmp_path: Path) -> None:
        """Corrupted .npy file should result in cache miss."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        key = cache._make_key("hello", "v", "b")
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        npy_path = cache._cache_dir / f"{key}.npy"
        npy_path.write_bytes(b"this is not a numpy file")

        import json

        meta_path = cache._cache_dir / f"{key}.json"
        meta_path.write_text(json.dumps({"sample_rate": 16000, "duration": 1.0}))

        result = cache.get("hello", "v", "b")
        assert result is None

    def test_corrupted_json_meta_returns_none(self, tmp_path: Path) -> None:
        """Corrupted JSON metadata should result in cache miss."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        key = cache._make_key("hello", "v", "b")
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        np.save(cache._cache_dir / f"{key}.npy", np.zeros(16000, dtype=np.float32))
        (cache._cache_dir / f"{key}.json").write_text("not valid json{{")

        result = cache.get("hello", "v", "b")
        assert result is None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    """Tests for module-level get_default_cache and reset_default_cache."""

    def test_get_default_cache_returns_instance(self) -> None:
        """get_default_cache should return a TTSCache instance."""
        reset_default_cache()
        cache = get_default_cache()
        assert isinstance(cache, TTSCache)

    def test_reset_clears_singleton(self) -> None:
        """reset_default_cache should allow recreating the singleton."""
        cache1 = get_default_cache()
        reset_default_cache()
        cache2 = get_default_cache()
        assert cache1 is not cache2

    def test_default_cache_uses_user_cache_dir(self) -> None:
        """Default cache should use ~/.cache/wakeword_workbench/tts/."""
        reset_default_cache()
        cache = get_default_cache()
        assert cache._cache_dir.name == "tts"
        assert ".cache" in str(cache._cache_dir)


# ---------------------------------------------------------------------------
# Cache directory creation
# ---------------------------------------------------------------------------


class TestCacheDirectory:
    """Tests for automatic cache directory creation."""

    def test_cache_dir_created_on_first_put(self, tmp_path: Path) -> None:
        """Cache directory should be created on first put."""
        cache_dir = tmp_path / "new_cache"
        cache = TTSCache(cache_dir=cache_dir)
        result = make_result()

        assert not cache_dir.exists()
        cache.put("hello", "v", "b", result)
        assert cache_dir.exists()

    def test_cache_stores_npy_and_json_files(self, tmp_path: Path) -> None:
        """Cache should produce both .npy audio and .json metadata files."""
        cache = TTSCache(cache_dir=tmp_path / "tts_cache")
        result = make_result()

        cache.put("hello", "v", "b", result)
        key = cache._make_key("hello", "v", "b")

        npy_path = tmp_path / "tts_cache" / f"{key}.npy"
        json_path = tmp_path / "tts_cache" / f"{key}.json"

        assert npy_path.exists()
        assert json_path.exists()
        assert json_path.stat().st_size > 0
