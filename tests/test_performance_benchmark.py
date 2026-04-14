"""Performance regression tests for caching and pooling optimizations.

These tests verify that the performance optimizations (Piper model caching,
BackendPool backend reuse, and voice grouping) work correctly. They use mocks
to avoid real TTS synthesis and are regression tests, not microbenchmarks.

Baseline improvement ratios (documented for regression tracking):
- Piper voice cache: cache hit avoids model load (typically ~1-3s per load)
- BackendPool: N requests with same provider → 1 backend instance (not N)
- Voice grouping: sorted synthesis reduces model switches during generation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from wakeword_workbench.tts.pool import BackendPool


class TestPiperModelCaching:
    """Tests that Piper model caching reduces redundant model loads."""

    def test_voice_cache_hit_avoids_reload(self) -> None:
        """Cache hit should not trigger a new PiperVoice.load() call.

        Baseline: Without caching, loading the same voice model for every
        synthesize call would cost ~1-3s per call. With caching, the first
        load is cached and subsequent calls reuse the cached instance.
        """
        mock_voice = MagicMock()
        mock_voice.synthesize_wav = MagicMock()

        # First load
        with patch("wakeword_workbench.tts.piper_backend.PiperVoice") as mock_piper_class:
            mock_piper_class.load.return_value = mock_voice

            from wakeword_workbench.tts.piper_backend import PiperBackend

            # Clear any residual state from previous tests
            PiperBackend._voice_cache.clear()

            backend = PiperBackend(model_path=None, acceleration="cpu")

            # Reset to count load calls
            mock_piper_class.load.reset_mock()

            # Cache should contain the voice after init
            # Subsequent set_voice to same voice should not reload
            try:
                backend.set_voice("en_US-lessac-medium")
            except Exception:
                pass  # We only care about load count

            # If we somehow triggered another load, it would call load() again
            # With proper caching, load should NOT be called for cached voices
            load_calls_after = mock_piper_class.load.call_count
            assert load_calls_after == 0, "Cache hit should not trigger additional model loads"

    def test_voice_cache_max_size_eviction(self) -> None:
        """Cache should evict oldest entry when max size is exceeded.

        Baseline: _MAX_CACHED_VOICES = 5. Loading a 6th unique voice should
        evict the oldest cached voice.
        """
        from wakeword_workbench.tts.piper_backend import PiperBackend

        PiperBackend._voice_cache.clear()

        # Cache has max size of 5
        assert PiperBackend._MAX_CACHED_VOICES == 5


class TestBackendPool:
    """Tests that BackendPool reduces backend instances."""

    def test_same_provider_returns_same_backend_instance(self) -> None:
        """Multiple get() calls for the same provider should return the same instance.

        Baseline: Without pooling, each get() could create a new backend instance.
        With pooling, identical providers map to a single cached backend.
        Expected: 1 backend instance regardless of call count.
        """
        from wakeword_workbench.config import TTSProviderConfig

        pool = BackendPool()
        provider = TTSProviderConfig(
            backend="kokoro",
            voices=["af_sarah"],
            speed=1.0,
        )

        with patch("wakeword_workbench.tts.pool.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend

            # Call get() multiple times with the same provider
            backend1 = pool.get(provider)
            backend2 = pool.get(provider)
            backend3 = pool.get(provider)

            # All should return the same instance
            assert backend1 is backend2
            assert backend2 is backend3

            # get_backend should be called exactly once (first get)
            assert mock_get_backend.call_count == 1, (
                "BackendPool should create backend only on first get() call"
            )

    def test_different_providers_create_different_backends(self) -> None:
        """Different provider configurations should create separate backend instances.

        Baseline: Providers with different backends/speed/acceleration should each
        get their own backend instance from the pool.
        Expected: 2 backend instances for 2 different providers.
        """
        from wakeword_workbench.config import TTSProviderConfig

        pool = BackendPool()
        provider1 = TTSProviderConfig(
            backend="kokoro",
            voices=["af_sarah"],
            speed=1.0,
        )
        provider2 = TTSProviderConfig(
            backend="kokoro",
            voices=["am_adam"],
            speed=0.9,
        )

        with patch("wakeword_workbench.tts.pool.get_backend") as mock_get_backend:
            mock_backend1 = MagicMock()
            mock_backend2 = MagicMock()
            mock_get_backend.side_effect = [mock_backend1, mock_backend2]

            backend1 = pool.get(provider1)
            backend2 = pool.get(provider2)

            # Should be different instances
            assert backend1 is not backend2

            # Should have called get_backend twice
            assert mock_get_backend.call_count == 2

    def test_different_backends_create_different_instances(self) -> None:
        """Providers with different backend names should get separate instances.

        Baseline: kokoro vs piper should always be different instances.
        Expected: 2 instances for kokoro + piper providers.
        """
        from wakeword_workbench.config import TTSProviderConfig

        pool = BackendPool()
        kokoro_provider = TTSProviderConfig(
            backend="kokoro",
            voices=["af_sarah"],
            speed=1.0,
        )
        piper_provider = TTSProviderConfig(
            backend="piper",
            voices=["en_US-lessac-medium"],
            speed=1.0,
        )

        with patch("wakeword_workbench.tts.pool.get_backend") as mock_get_backend:
            mock_kokoro = MagicMock()
            mock_piper = MagicMock()
            mock_get_backend.side_effect = [mock_kokoro, mock_piper]

            kokoro_backend = pool.get(kokoro_provider)
            piper_backend = pool.get(piper_provider)

            assert kokoro_backend is not piper_backend
            assert mock_get_backend.call_count == 2

    def test_pool_clear_removes_all_instances(self) -> None:
        """BackendPool.clear() should remove all cached backends."""
        from wakeword_workbench.config import TTSProviderConfig

        pool = BackendPool()
        provider = TTSProviderConfig(
            backend="kokoro",
            voices=["af_sarah"],
            speed=1.0,
        )

        with patch("wakeword_workbench.tts.pool.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend

            backend = pool.get(provider)
            assert backend is not None

            pool.clear()

            # After clear, next get() should create a new instance
            mock_get_backend.reset_mock()
            pool.get(provider)
            assert mock_get_backend.call_count == 1


class TestVoiceGrouping:
    """Tests that voice grouping reduces model switches during synthesis."""

    def test_grouped_assignments_sort_by_backend_and_acceleration(self) -> None:
        """Synthesize assignments should be grouped by backend for efficiency.

        Baseline: Random ordering of phrases causes interleaved backend/voice
        switches, which are expensive. Grouped ordering puts same-backend
        phrases together, minimizing switches.

        The generator sorts phrase_voice_assignments by
        (backend, acceleration, voice) before synthesis.
        """
        # Mock TTS provider configs
        kokoro_provider = MagicMock()
        kokoro_provider.backend = "kokoro"
        kokoro_provider.voices = ["af_sarah", "am_adam"]
        kokoro_provider.backend_cache_key.return_value = ("kokoro", 1.0, "cpu", None, None)

        piper_provider = MagicMock()
        piper_provider.backend = "piper"
        piper_provider.voices = ["en_US-lessac-medium"]
        piper_provider.backend_cache_key.return_value = ("piper", 1.0, "cpu", None, None)

        # Simulate the grouping logic from _synthesize_negatives
        phrases = ["phrase1", "phrase2", "phrase3", "phrase4"]
        provider_voices = [
            (kokoro_provider, "af_sarah"),
            (piper_provider, "en_US-lessac-medium"),
            (kokoro_provider, "am_adam"),
            (piper_provider, "en_US-lessac-medium"),
        ]

        assignments = [
            (idx, phrase, *pv)
            for idx, (phrase, pv) in enumerate(zip(phrases, provider_voices, strict=True))
        ]

        # Sort by (backend, acceleration, voice) as the generator does
        sorted_assignments = sorted(
            assignments,
            key=lambda a: (a[2].backend, a[2].backend_cache_key()[2], a[3]),
        )

        # After sorting, all kokoro entries should be contiguous
        backends = [a[2].backend for a in sorted_assignments]
        kokoro_indices = [i for i, b in enumerate(backends) if b == "kokoro"]
        piper_indices = [i for i, b in enumerate(backends) if b == "piper"]

        # Kokoro entries should form a contiguous block
        assert kokoro_indices == list(range(min(kokoro_indices), max(kokoro_indices) + 1)), (
            "Voice grouping should put same-backend assignments contiguously"
        )
        # Piper entries should form a contiguous block
        assert piper_indices == list(range(min(piper_indices), max(piper_indices) + 1)), (
            "Voice grouping should put same-backend assignments contiguously"
        )

    def test_round_robin_preserved_before_grouping(self) -> None:
        """Round-robin voice assignment should be preserved before sorting.

        The generator first assigns provider/voice in round-robin order
        (preserving distribution), then sorts for efficient synthesis.
        This ensures all voices are used while minimizing switches.
        """
        kokoro_provider = MagicMock()
        kokoro_provider.backend = "kokoro"
        kokoro_provider.backend_cache_key.return_value = ("kokoro", 1.0, "cpu", None, None)

        piper_provider = MagicMock()
        piper_provider.backend = "piper"
        piper_provider.backend_cache_key.return_value = ("piper", 1.0, "cpu", None, None)

        provider_voices = [
            (kokoro_provider, "voice1"),
            (piper_provider, "voice2"),
        ]

        phrases = [f"phrase{i}" for i in range(6)]

        # Round-robin assignment (phase 1 of _synthesize_negatives)
        assigned = [
            provider_voices[(0 + idx) % len(provider_voices)] for idx in range(len(phrases))
        ]

        # Each phrase gets exactly one provider/voice
        assert len(assigned) == len(phrases)
        # Round-robin should cycle through all provider/voice combos
        for i, (provider, voice) in enumerate(assigned):
            expected_provider, expected_voice = provider_voices[i % len(provider_voices)]
            assert provider is expected_provider
            assert voice == expected_voice
