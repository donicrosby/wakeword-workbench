"""Tests for PiperBackend TTS backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wakeword_workbench.tts.base import BackendNotAvailableError, TTSError, TTSResult
from wakeword_workbench.tts.cache import TTSCache, reset_default_cache


class TestPiperBackendAvailability:
    """Tests for PiperBackend availability checks."""

    def test_is_available_returns_bool(self) -> None:
        """is_available() should return a boolean."""
        # Import fresh to avoid cached module state
        import importlib

        import wakeword_workbench.tts.piper_backend as piper_module

        importlib.reload(piper_module)

        result = piper_module.PiperBackend.is_available()
        assert isinstance(result, bool)

    def test_is_available_false_when_not_installed(self) -> None:
        """is_available() returns False when piper is not installed."""
        # Patch _PIPER_AVAILABLE at module level
        import wakeword_workbench.tts.piper_backend as piper_module

        piper_module._PIPER_AVAILABLE = False
        try:
            assert piper_module.PiperBackend.is_available() is False
        finally:
            piper_module._PIPER_AVAILABLE = True  # Restore

    def test_is_available_true_when_installed(self) -> None:
        """is_available() returns True when piper is installed."""
        import wakeword_workbench.tts.piper_backend as piper_module

        piper_module._PIPER_AVAILABLE = True
        try:
            assert piper_module.PiperBackend.is_available() is True
        finally:
            piper_module._PIPER_AVAILABLE = True  # Keep True


class TestPiperBackendInitialization:
    """Tests for PiperBackend initialization."""

    def test_raises_when_piper_not_available(self, tmp_path: Path) -> None:
        """Should raise BackendNotAvailableError if piper is not installed."""
        import wakeword_workbench.tts.piper_backend as piper_module

        piper_module._PIPER_AVAILABLE = False
        try:
            with pytest.raises(BackendNotAvailableError) as exc_info:
                piper_module.PiperBackend(model_path=str(tmp_path / "model.onnx"))
            assert "Install it with: pip install piper-tts" in str(exc_info.value)
        finally:
            piper_module._PIPER_AVAILABLE = True

    def test_raises_when_model_not_found(self, tmp_path: Path) -> None:
        """Should raise TTSError if model file doesn't exist."""
        import wakeword_workbench.tts.piper_backend as piper_module

        piper_module._PIPER_AVAILABLE = True

        with pytest.raises(TTSError) as exc_info:
            piper_module.PiperBackend(model_path=str(tmp_path / "nonexistent.onnx"))
        assert "model not found" in str(exc_info.value)

    def test_default_model_download_reuses_voice_repository(self, tmp_path: Path) -> None:
        """Default model download should use the same voice repository path as named voices."""
        import wakeword_workbench.tts.piper_backend as piper_module

        onnx_path = tmp_path / "en_US-lessac-medium.onnx"
        onnx_path.touch()

        mock_voice = MagicMock()
        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_voice):
            with patch.object(
                piper_module.PiperBackend,
                "_download_voice_model",
                return_value=onnx_path,
            ) as mock_download_voice:
                backend = piper_module.PiperBackend(model_path=None)

        mock_download_voice.assert_called_once_with("en_US-lessac-medium")
        assert backend.model_path == onnx_path

    def test_download_voice_model_uses_hugging_face_cache(self) -> None:
        """Voice downloads should go through hf_hub_download and return the cached ONNX path."""
        import wakeword_workbench.tts.piper_backend as piper_module

        cached_onnx_path = "/tmp/hf-cache/en_US-lessac-medium.onnx"
        mock_hf_module = MagicMock()
        mock_hf_module.hf_hub_download.side_effect = [
            cached_onnx_path,
            "/tmp/hf-cache/en_US-lessac-medium.onnx.json",
        ]

        with patch.object(piper_module, "import_module", return_value=mock_hf_module):
            result = piper_module.PiperBackend._download_voice_model("en_US-lessac-medium")

        assert result == Path(cached_onnx_path)
        assert mock_hf_module.hf_hub_download.call_count == 2
        first_call = mock_hf_module.hf_hub_download.call_args_list[0]
        assert first_call.kwargs["repo_id"] == "rhasspy/piper-voices"
        assert first_call.kwargs["filename"] == "en/en_US/lessac/medium/en_US-lessac-medium.onnx"


class TestPiperBackendSynthesize:
    """Tests for PiperBackend.synthesize()."""

    @pytest.fixture
    def mock_piper_voice(self) -> MagicMock:
        """Create a mock PiperVoice for testing."""
        mock_voice = MagicMock()
        return mock_voice

    def test_synthesize_returns_tts_result(
        self,
        tmp_path: Path,
        mock_piper_voice: MagicMock,
    ) -> None:
        """synthesize() should return a TTSResult with 16000 Hz audio."""
        import wave

        import soundfile as sf

        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        sample_audio = np.sin(np.linspace(0, 2 * np.pi, 22050)).astype(np.float32)

        def _write_wav(text: str, wav_file: wave.Wave_write) -> None:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            int16_data = (sample_audio * 32767).astype(np.int16)
            wav_file.writeframes(int16_data.tobytes())

        mock_piper_voice.load.return_value.synthesize_wav = MagicMock(side_effect=_write_wav)

        def mock_sf_read(file, dtype=None):
            return sample_audio, 22050

        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_piper_voice):
            with patch.object(sf, "read", mock_sf_read):
                backend = piper_module.PiperBackend(model_path=str(model_path))
                result = backend.synthesize("hello world")

        assert isinstance(result.audio, np.ndarray)
        assert result.sample_rate == 16000
        assert result.duration > 0
        # After resampling, should be ~16000 samples for 1 second
        assert len(result.audio) == int(16000 * result.duration)

    def test_synthesize_empty_text_raises_error(
        self, tmp_path: Path, mock_piper_voice: MagicMock
    ) -> None:
        """synthesize() should raise TTSError for empty text."""
        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_piper_voice):
            backend = piper_module.PiperBackend(model_path=str(model_path))
            with pytest.raises(TTSError) as exc_info:
                backend.synthesize("")
            assert "empty text" in str(exc_info.value)

    def test_synthesize_whitespace_only_raises_error(
        self, tmp_path: Path, mock_piper_voice: MagicMock
    ) -> None:
        """synthesize() should raise TTSError for whitespace-only text."""
        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_piper_voice):
            backend = piper_module.PiperBackend(model_path=str(model_path))
            with pytest.raises(TTSError) as exc_info:
                backend.synthesize("   ")
            assert "empty text" in str(exc_info.value)


class TestPiperBackendSetVoice:
    """Tests for PiperBackend.set_voice()."""

    def test_set_voice_loads_new_model(self, tmp_path: Path) -> None:
        """set_voice() should download and load the requested voice model."""
        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        mock_voice = MagicMock()
        piper_module._PIPER_AVAILABLE = True

        onnx_file = tmp_path / "en_US-lessac-high.onnx"
        onnx_file.touch()

        with patch.object(piper_module, "PiperVoice", mock_voice):
            with patch.object(
                piper_module.PiperBackend,
                "_download_voice_model",
                return_value=onnx_file,
            ):
                backend = piper_module.PiperBackend(model_path=str(model_path))
                backend.set_voice("en_US-lessac-high")

        assert mock_voice.load.call_count == 2
        assert backend.model_path == onnx_file

    def test_set_voice_unknown_raises_error(self, tmp_path: Path) -> None:
        """set_voice() should raise TTSError for unknown voices."""
        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        mock_voice = MagicMock()
        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_voice):
            backend = piper_module.PiperBackend(model_path=str(model_path))
            with pytest.raises(TTSError) as exc_info:
                backend.set_voice("nonexistent_voice")
            assert "not available" in str(exc_info.value)

    def test_set_voice_download_failure_raises_error(self, tmp_path: Path) -> None:
        """set_voice() should raise TTSError when model download fails."""
        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        mock_voice = MagicMock()
        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_voice):
            with patch.object(
                piper_module.PiperBackend,
                "_download_voice_model",
                side_effect=TTSError("Failed to download Piper voice 'en_US-lessac-high'"),
            ):
                backend = piper_module.PiperBackend(model_path=str(model_path))
                with pytest.raises(TTSError) as exc_info:
                    backend.set_voice("en_US-lessac-high")
                assert "Failed to download" in str(exc_info.value)


class TestPiperBackendListVoices:
    """Tests for PiperBackend.list_voices()."""

    def test_list_voices_returns_known_voices(self, tmp_path: Path) -> None:
        """list_voices() should return known Piper voice names."""
        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        mock_voice = MagicMock()
        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_voice):
            backend = piper_module.PiperBackend(model_path=str(model_path))
            voices = backend.list_voices()
            assert len(voices) > 0
            assert "en_US-lessac-high" in voices
            assert "en_US-ryan-high" in voices
            assert voices == piper_module._KNOWN_VOICES

    def test_list_voices_in_directory_finds_models(self, tmp_path: Path) -> None:
        """list_voices_in_directory() should find .onnx model files."""
        import wakeword_workbench.tts.piper_backend as piper_module

        # Create mock model files
        (tmp_path / "voice1.onnx").touch()
        (tmp_path / "voice2.onnx").touch()
        (tmp_path / "config.json").touch()  # Should be ignored

        voices = piper_module.PiperBackend.list_voices_in_directory(tmp_path)

        assert len(voices) == 2
        assert str(tmp_path / "voice1") in voices
        assert str(tmp_path / "voice2") in voices


class TestPiperBackendResample:
    """Tests for the resampling functionality."""

    def test_resample_no_change_for_same_rate(self) -> None:
        """Should return same audio when sample rates match."""
        import wakeword_workbench.tts.piper_backend as piper_module

        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = piper_module.PiperBackend._resample(audio, 16000, 16000)
        np.testing.assert_array_almost_equal(audio, result)

    def test_resample_changes_rate(self) -> None:
        """Should correctly resample audio."""
        import wakeword_workbench.tts.piper_backend as piper_module

        # Create 1 second of audio at 22050 Hz (22050 samples)
        audio = np.sin(np.linspace(0, 2 * np.pi, 22050)).astype(np.float32)

        # Resample to 16000 Hz
        result = piper_module.PiperBackend._resample(audio, 22050, 16000)

        # Should be approximately 16000 samples for 1 second
        assert abs(len(result) - 16000) < 100  # Within 100 samples tolerance
        assert result.dtype == np.float32


class TestPiperBackendTTSResultValidation:
    """Tests for TTSResult validation in synthesize()."""

    @pytest.fixture
    def mock_piper_voice(self) -> MagicMock:
        """Create a mock PiperVoice for testing."""
        mock_voice = MagicMock()
        return mock_voice

    def test_result_audio_is_normalized(
        self,
        tmp_path: Path,
        mock_piper_voice: MagicMock,
    ) -> None:
        """TTSResult audio should be normalized to [-1, 1]."""
        import wave

        import soundfile as sf

        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        sample_audio = np.sin(np.linspace(0, 2 * np.pi, 22050)).astype(np.float32)

        def _write_wav(text: str, wav_file: wave.Wave_write) -> None:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            int16_data = (sample_audio * 32767).astype(np.int16)
            wav_file.writeframes(int16_data.tobytes())

        mock_piper_voice.load.return_value.synthesize_wav = MagicMock(side_effect=_write_wav)

        def mock_sf_read(file, dtype=None):
            return sample_audio, 22050

        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_piper_voice):
            with patch.object(sf, "read", mock_sf_read):
                backend = piper_module.PiperBackend(model_path=str(model_path))
                result = backend.synthesize("test")

        # Check normalization
        assert result.audio.min() >= -1.0
        assert result.audio.max() <= 1.0

    def test_result_duration_matches_audio_length(
        self,
        tmp_path: Path,
        mock_piper_voice: MagicMock,
    ) -> None:
        """TTSResult duration should match actual audio length."""
        import wave

        import soundfile as sf

        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        sample_audio = np.sin(np.linspace(0, np.pi, 11025)).astype(np.float32)

        def _write_wav(text: str, wav_file: wave.Wave_write) -> None:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            int16_data = (sample_audio * 32767).astype(np.int16)
            wav_file.writeframes(int16_data.tobytes())

        mock_piper_voice.load.return_value.synthesize_wav = MagicMock(side_effect=_write_wav)

        def mock_sf_read(file, dtype=None):
            return sample_audio, 22050

        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_piper_voice):
            with patch.object(sf, "read", mock_sf_read):
                backend = piper_module.PiperBackend(model_path=str(model_path))
                result = backend.synthesize("test")

        # Check duration matches
        expected_duration = len(result.audio) / 16000
        assert abs(result.duration - expected_duration) < 0.01


class TestPiperBackendCaching:
    """Test PiperBackend caching integration."""

    @pytest.fixture
    def mock_cache(self, tmp_path: Path) -> TTSCache:
        """Create a fresh cache for testing."""
        reset_default_cache()
        return TTSCache(cache_dir=tmp_path / "tts_cache")

    @pytest.fixture
    def mock_piper_voice(self) -> MagicMock:
        """Create a mock PiperVoice for testing."""
        mock_voice = MagicMock()
        return mock_voice

    def test_synthesize_caches_result(
        self,
        tmp_path: Path,
        mock_piper_voice: MagicMock,
        mock_cache: TTSCache,
    ) -> None:
        """Test that synthesize stores result in cache after synthesis."""
        import wave

        import soundfile as sf

        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        sample_audio = np.sin(np.linspace(0, 2 * np.pi, 22050)).astype(np.float32)

        def _write_wav(text: str, wav_file: wave.Wave_write) -> None:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            int16_data = (sample_audio * 32767).astype(np.int16)
            wav_file.writeframes(int16_data.tobytes())

        mock_piper_voice.load.return_value.synthesize_wav = MagicMock(side_effect=_write_wav)

        def mock_sf_read(file, dtype=None):
            return sample_audio, 22050

        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_piper_voice):
            with patch.object(sf, "read", mock_sf_read):
                with patch(
                    "wakeword_workbench.tts.piper_backend.get_default_cache",
                    return_value=mock_cache,
                ):
                    backend = piper_module.PiperBackend(model_path=str(model_path))
                    result = backend.synthesize("hello world")

                    cached = mock_cache.get(
                        "hello world",
                        "test_model",
                        "piper",
                        1.0,
                        options={
                            "acceleration": "cpu",
                            "model_path": str(model_path),
                        },
                    )
                    assert cached is not None
                    np.testing.assert_array_equal(cached.audio, result.audio)

    def test_synthesize_returns_cached_result(
        self,
        tmp_path: Path,
        mock_piper_voice: MagicMock,
        mock_cache: TTSCache,
    ) -> None:
        """Test that synthesize returns cached result without synthesis."""
        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        # Pre-populate cache
        cached_audio = np.zeros(16000, dtype=np.float32)
        cached_result = TTSResult(
            audio=cached_audio,
            sample_rate=16000,
            duration=1.0,
        )
        mock_cache.put(
            "hello world",
            "test_model",
            "piper",
            cached_result,
            1.0,
            options={
                "acceleration": "cpu",
                "model_path": str(model_path),
            },
        )

        mock_piper_voice.synthesize_wav = MagicMock()
        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_piper_voice):
            with patch(
                "wakeword_workbench.tts.piper_backend.get_default_cache", return_value=mock_cache
            ):
                backend = piper_module.PiperBackend(model_path=str(model_path))
                result = backend.synthesize("hello world")

                # Voice synthesis should NOT have been called
                mock_piper_voice.load.return_value.synthesize_wav.assert_not_called()

                np.testing.assert_array_equal(result.audio, cached_audio)
