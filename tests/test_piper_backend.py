"""Tests for PiperBackend TTS backend."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wakeword_workbench.tts.base import BackendNotAvailableError, TTSError


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
        import wakeword_workbench.tts.piper_backend as piper_module
        import soundfile as sf

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        # Create sample audio data (1 second at 22050 Hz)
        sample_audio = np.sin(np.linspace(0, 2 * np.pi, 22050)).astype(np.float32)

        # Mock synthesize_wav (no-op since we're mocking sf.read)
        mock_piper_voice.synthesize_wav = MagicMock()

        # Mock soundfile.read to return our sample audio
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

    def test_set_voice_raises_not_implemented(self, tmp_path: Path) -> None:
        """set_voice() should always raise NotImplementedError."""
        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        mock_voice = MagicMock()
        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_voice):
            backend = piper_module.PiperBackend(model_path=str(model_path))
            with pytest.raises(NotImplementedError) as exc_info:
                backend.set_voice("any_voice")
            assert "runtime voice changes" in str(exc_info.value)


class TestPiperBackendListVoices:
    """Tests for PiperBackend.list_voices()."""

    def test_list_voices_returns_empty_list(self, tmp_path: Path) -> None:
        """list_voices() should return empty list for Piper."""
        import wakeword_workbench.tts.piper_backend as piper_module

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        mock_voice = MagicMock()
        piper_module._PIPER_AVAILABLE = True
        with patch.object(piper_module, "PiperVoice", mock_voice):
            backend = piper_module.PiperBackend(model_path=str(model_path))
            assert backend.list_voices() == []

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
        import wakeword_workbench.tts.piper_backend as piper_module
        import soundfile as sf

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        # Create sample audio data that's already normalized
        sample_audio = np.sin(np.linspace(0, 2 * np.pi, 22050)).astype(np.float32)

        # Mock synthesize_wav
        mock_piper_voice.synthesize_wav = MagicMock()

        # Mock soundfile.read
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
        import wakeword_workbench.tts.piper_backend as piper_module
        import soundfile as sf

        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        # Create sample audio data (0.5 seconds at 22050 Hz)
        sample_audio = np.sin(np.linspace(0, np.pi, 11025)).astype(np.float32)

        # Mock synthesize_wav
        mock_piper_voice.synthesize_wav = MagicMock()

        # Mock soundfile.read
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
