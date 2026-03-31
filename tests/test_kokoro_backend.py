"""Tests for Kokoro TTS backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wakeword_workbench.tts.base import BackendNotAvailableError, TTSError, TTSResult
from wakeword_workbench.tts.kokoro_backend import KokoroBackend


class TestKokoroBackendAvailability:
    """Test backend availability detection."""

    def test_is_available_when_pykokoro_installed(self) -> None:
        """Test is_available returns True when pykokoro is installed."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            assert KokoroBackend.is_available() is True

    def test_is_available_when_pykokoro_not_installed(self) -> None:
        """Test is_available returns False when pykokoro is not installed."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", False):
            assert KokoroBackend.is_available() is False


class TestKokoroBackendInit:
    """Test KokoroBackend initialization."""

    def test_init_raises_when_pykokoro_not_available(self) -> None:
        """Test that __init__ raises BackendNotAvailableError if pykokoro is missing."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", False):
            with pytest.raises(BackendNotAvailableError) as exc_info:
                KokoroBackend()
            assert "pykokoro is not installed" in str(exc_info.value)

    def test_init_with_valid_voice(self) -> None:
        """Test initialization with a valid voice."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            with patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline"):
                backend = KokoroBackend(voice="af_sarah", speed=1.0)
                assert backend._voice == "af_sarah"
                assert backend._speed == 1.0

    def test_init_with_invalid_speed(self) -> None:
        """Test that __init__ raises TTSError for invalid speed."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            with pytest.raises(TTSError) as exc_info:
                KokoroBackend(speed=0)
            assert "speed must be positive" in str(exc_info.value)

    def test_init_with_invalid_voice(self) -> None:
        """Test that __init__ raises TTSError for invalid voice."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            with patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline"):
                with pytest.raises(TTSError) as exc_info:
                    KokoroBackend(voice="invalid_voice")
                assert "Voice 'invalid_voice' not available" in str(exc_info.value)


class TestKokoroBackendSynthesize:
    """Test KokoroBackend.synthesize()."""

    def test_synthesize_empty_text_raises(self) -> None:
        """Test that synthesize raises TTSError for empty text."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            with patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline"):
                backend = KokoroBackend()
                with pytest.raises(TTSError) as exc_info:
                    backend.synthesize("")
                assert "Cannot synthesize empty text" in str(exc_info.value)

    def test_synthesize_whitespace_text_raises(self) -> None:
        """Test that synthesize raises TTSError for whitespace-only text."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            with patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline"):
                backend = KokoroBackend()
                with pytest.raises(TTSError) as exc_info:
                    backend.synthesize("   ")
                assert "Cannot synthesize empty text" in str(exc_info.value)

    @patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline")
    def test_synthesize_returns_tts_result(self, mock_pipeline: MagicMock) -> None:
        """Test that synthesize returns a valid TTSResult."""
        # Create mock audio at 24000 Hz (Kokoro native rate)
        mock_audio = np.random.randn(24000).astype(np.float32) * 0.5  # 1 second at 24kHz

        # Setup mock pipeline
        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_audio
        mock_pipeline.return_value = mock_instance

        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            # Clear pipeline cache
            KokoroBackend._pipeline = None
            backend = KokoroBackend()

            result = backend.synthesize("Hello world")

            assert isinstance(result, TTSResult)
            assert result.sample_rate == 16000
            assert result.audio.dtype == np.float32
            # Verify audio is normalized to [-1, 1]
            assert result.audio.min() >= -1.0
            assert result.audio.max() <= 1.0
            # Duration should be approximately 1.0 seconds
            assert 0.9 < result.duration < 1.1

    @patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline")
    def test_synthesize_resamples_to_16khz(self, mock_pipeline: MagicMock) -> None:
        """Test that synthesize resamples audio from 24kHz to 16kHz."""
        # Create mock audio at 24000 Hz
        mock_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_audio
        mock_pipeline.return_value = mock_instance

        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            KokoroBackend._pipeline = None
            backend = KokoroBackend()

            result = backend.synthesize("Test")

            # At 16kHz, 1 second should have 16000 samples
            assert len(result.audio) == 16000
            assert result.sample_rate == 16000

    @patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline")
    def test_synthesize_uses_voice_and_speed(self, mock_pipeline: MagicMock) -> None:
        """Test that synthesize uses the configured voice and speed."""
        mock_audio = np.zeros(24000, dtype=np.float32)
        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_audio
        mock_pipeline.return_value = mock_instance

        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            KokoroBackend._pipeline = None
            backend = KokoroBackend(voice="af_nicole", speed=1.2)

            backend.synthesize("Test")

            mock_instance.generate.assert_called_once_with("Test", voice="af_nicole", speed=1.2)

    @patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline")
    def test_synthesize_normalizes_audio(self, mock_pipeline: MagicMock) -> None:
        """Test that synthesize normalizes audio to [-1, 1]."""
        # Create audio with values > 1.0
        mock_audio = np.random.randn(24000).astype(np.float32) * 2.0  # exceeds [-1, 1]

        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_audio
        mock_pipeline.return_value = mock_instance

        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            KokoroBackend._pipeline = None
            backend = KokoroBackend()

            result = backend.synthesize("Test")

            # Audio should be normalized to [-1, 1]
            assert result.audio.min() >= -1.0
            assert result.audio.max() <= 1.0


class TestKokoroBackendVoiceManagement:
    """Test voice management methods."""

    def test_set_voice_valid(self) -> None:
        """Test setting a valid voice."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            with patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline"):
                backend = KokoroBackend()
                backend.set_voice("am_michael")
                assert backend._voice == "am_michael"

    def test_set_voice_invalid(self) -> None:
        """Test setting an invalid voice raises TTSError."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            with patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline"):
                backend = KokoroBackend()
                with pytest.raises(TTSError) as exc_info:
                    backend.set_voice("invalid_voice")
                assert "Voice 'invalid_voice' not available" in str(exc_info.value)

    def test_list_voices(self) -> None:
        """Test that list_voices returns the expected voices."""
        with patch("wakeword_workbench.tts.kokoro_backend._PYKOKORO_AVAILABLE", True):
            with patch("wakeword_workbench.tts.kokoro_backend.KokoroPipeline"):
                backend = KokoroBackend()
                voices = backend.list_voices()
                assert isinstance(voices, list)
                assert "af_sarah" in voices
                assert "am_michael" in voices
                # Verify it returns a copy
                voices.append("test_voice")
                assert "test_voice" not in backend.list_voices()


class TestKokoroBackendRegistration:
    """Test backend registration."""

    def test_backend_registered(self) -> None:
        """Test that KokoroBackend is registered under 'kokoro'."""
        from wakeword_workbench.tts.base import TTSBackend

        assert "kokoro" in TTSBackend._backend_registry
        assert TTSBackend._backend_registry["kokoro"] is KokoroBackend
