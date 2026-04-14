from __future__ import annotations

import threading
import time
from importlib import import_module
from unittest.mock import Mock, patch

import numpy as np

TTSProviderConfig = import_module("wakeword_workbench.config").TTSProviderConfig
_tts_base = import_module("wakeword_workbench.tts.base")
TTSBackend = _tts_base.TTSBackend
TTSResult = _tts_base.TTSResult
BackendPool = import_module("wakeword_workbench.tts.pool").BackendPool


class _MockBackend(TTSBackend):
    def synthesize(self, text: str) -> TTSResult:
        del text
        return TTSResult(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000, duration=1.0)

    def set_voice(self, voice: str) -> None:
        del voice

    def list_voices(self) -> list[str]:
        return ["mock"]


class _ConcurrentBackend(TTSBackend):
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._active_calls = 0
        self.max_concurrent = 0

    def synthesize(self, text: str) -> TTSResult:
        del text
        with self._state_lock:
            self._active_calls += 1
            self.max_concurrent = max(self.max_concurrent, self._active_calls)
        time.sleep(0.01)
        with self._state_lock:
            self._active_calls -= 1
        return TTSResult(audio=np.zeros(16000, dtype=np.float32), sample_rate=16000, duration=1.0)

    def set_voice(self, voice: str) -> None:
        del voice

    def list_voices(self) -> list[str]:
        return ["mock"]


class TestBackendPool:
    def test_get_returns_same_instance_for_identical_provider_config(self) -> None:
        pool = BackendPool()
        provider_1 = TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.0)
        provider_2 = TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.0)

        with patch(
            "wakeword_workbench.tts.pool.get_backend", return_value=_MockBackend()
        ) as get_mock:
            instance_1 = pool.get(provider_1)
            instance_2 = pool.get(provider_2)

        assert instance_1 is instance_2
        assert get_mock.call_count == 1

    def test_get_returns_different_instance_for_different_configs(self) -> None:
        pool = BackendPool()
        provider_1 = TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.0)
        provider_2 = TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.25)

        with patch(
            "wakeword_workbench.tts.pool.get_backend",
            side_effect=[_MockBackend(), _MockBackend()],
        ) as get_mock:
            instance_1 = pool.get(provider_1)
            instance_2 = pool.get(provider_2)

        assert instance_1 is not instance_2
        assert get_mock.call_count == 2

    def test_get_is_thread_safe_for_concurrent_access(self) -> None:
        pool = BackendPool()
        provider = TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.0)
        provider.backend_instance_key = Mock(return_value=("kokoro", 1.0, "cpu", None, None))  # type: ignore[attr-defined]

        created = 0
        created_lock = threading.Lock()

        def create_backend(_: str) -> TTSBackend:
            nonlocal created
            time.sleep(0.01)
            with created_lock:
                created += 1
            return _MockBackend()

        results: list[TTSBackend] = []
        errors: list[Exception] = []
        results_lock = threading.Lock()

        def worker() -> None:
            try:
                instance = pool.get(provider)
                with results_lock:
                    results.append(instance)
            except Exception as exc:
                errors.append(exc)

        with patch("wakeword_workbench.tts.pool.get_backend", side_effect=create_backend):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert not errors
        assert len(results) == 10
        assert len({id(result) for result in results}) == 1
        assert created == 1

    def test_clear_empties_cached_backends(self) -> None:
        pool = BackendPool()
        provider_1 = TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.0)
        provider_2 = TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.1)
        provider_3 = TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.2)

        with patch(
            "wakeword_workbench.tts.pool.get_backend",
            side_effect=[_MockBackend(), _MockBackend(), _MockBackend(), _MockBackend()],
        ):
            first = pool.get(provider_1)
            pool.get(provider_2)
            pool.get(provider_3)
            assert len(pool._pool) == 3

            pool.clear()
            assert len(pool._pool) == 0

            second = pool.get(provider_1)

        assert first is not second

    def test_get_returns_lock_wrapped_backend_for_serialized_inference(self) -> None:
        pool = BackendPool()
        provider = TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.0)
        raw_backend = _ConcurrentBackend()

        with patch("wakeword_workbench.tts.pool.get_backend", return_value=raw_backend):
            backend = pool.get(provider)

        threads = [threading.Thread(target=backend.synthesize, args=("hello",)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert raw_backend.max_concurrent == 1
