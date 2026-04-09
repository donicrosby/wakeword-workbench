"""Tests for positive_generator module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as _real_sf_module

from wakeword_workbench.config import (
    AugmentationConfig,
    Config,
    OutputConfig,
    SamplesConfig,
    TTSConfig,
    TTSProviderConfig,
)
from wakeword_workbench.dataset.positive_generator import PositiveGenerator, PositiveGeneratorError
from wakeword_workbench.tts.base import TTSResult

_real_write = _real_sf_module.write


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    """Create a mock configuration for testing."""
    samples = SamplesConfig(positives=100, negatives_multiplier=5)
    tts = TTSConfig(
        providers=[
            TTSProviderConfig(backend="kokoro", voices=["af_sarah", "am_adam"], speed=1.0),
        ]
    )
    augmentation = AugmentationConfig(
        noise_snr=[-10, 10],
        reverb_probability=0.5,
        gain_range=[-45, 0],
    )
    output = OutputConfig(path=str(tmp_path / "output"), format=["microwakeword"])
    return Config(
        wake_word="hey_vera",
        samples=samples,
        tts=tts,
        augmentation=augmentation,
        output=output,
    )


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    out_dir = tmp_path / "samples"
    out_dir.mkdir(parents=True)
    return out_dir


@pytest.fixture
def mock_tts_result() -> TTSResult:
    """Create a mock TTS result."""
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, num_samples)).astype(np.float32)
    return TTSResult(audio=audio, sample_rate=sample_rate, duration=duration_sec)


class TestPositiveGenerator:
    """Tests for PositiveGenerator class."""

    def test_init_creates_output_directory(self, mock_config: Config, tmp_path: Path) -> None:
        """Test that initialization creates output directory."""
        output_dir = tmp_path / "test_output"
        generator = PositiveGenerator(mock_config, output_dir)
        assert output_dir.exists()
        assert generator.output_dir == output_dir

    def test_init_stores_config(self, mock_config: Config, output_dir: Path) -> None:
        """Test that initialization stores configuration."""
        generator = PositiveGenerator(mock_config, output_dir)
        assert generator.config == mock_config
        assert generator._wake_word == "hey_vera"
        assert len(generator._providers) == 1
        assert generator._providers[0].backend == "kokoro"
        assert generator._providers[0].voices == ["af_sarah", "am_adam"]

    def test_generate_raises_on_invalid_count(self, mock_config: Config, output_dir: Path) -> None:
        """Test that generate raises error on invalid count."""
        generator = PositiveGenerator(mock_config, output_dir)
        with pytest.raises(PositiveGeneratorError, match="count must be positive"):
            generator.generate(0)
        with pytest.raises(PositiveGeneratorError, match="count must be positive"):
            generator.generate(-1)

    @patch("soundfile.write")
    @patch("wakeword_workbench.dataset.positive_generator._create_backend_for_provider")
    def test_generate_success(
        self,
        mock_create_backend: MagicMock,
        mock_sf_write: MagicMock,
        mock_config: Config,
        output_dir: Path,
        mock_tts_result: TTSResult,
    ) -> None:
        """Test successful sample generation."""
        mock_backend = MagicMock()
        mock_backend.synthesize.return_value = mock_tts_result
        mock_backend.set_voice.return_value = None
        mock_create_backend.return_value = mock_backend

        # Make sf.write actually write files
        def real_write(path, audio, sr):
            _real_write(path, audio, sr)

        mock_sf_write.side_effect = real_write

        # Generate samples
        manifest_path = PositiveGenerator(mock_config, output_dir).generate(4)

        # Verify manifest was created
        assert manifest_path.exists()
        assert manifest_path.name == "positive_manifest.jsonl"

        # Verify manifest contents
        with open(manifest_path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        assert len(entries) == 4

        # Verify each entry has required fields
        for entry in entries:
            assert "path" in entry
            assert "label" in entry
            assert entry["label"] == 1
            assert "text" in entry
            assert entry["text"] == "hey vera"
            assert "voice" in entry
            assert entry["voice"] in ["af_sarah", "am_adam"]
            assert entry["backend"] == "kokoro"
            assert "duration_ms" in entry
            assert entry["duration_ms"] > 0

        mock_create_backend.assert_called_with(mock_config.tts.providers[0])

        # Verify WAV files were created
        wav_files = list(output_dir.glob("*.wav"))
        assert len(wav_files) == 4

        # Verify file naming convention
        for wav_file in wav_files:
            assert wav_file.stem.startswith("hey_vera_")

    @patch("soundfile.write")
    @patch("wakeword_workbench.dataset.positive_generator._create_backend_for_provider")
    def test_generate_handles_tts_error(
        self,
        mock_create_backend: MagicMock,
        mock_sf_write: MagicMock,
        mock_config: Config,
        output_dir: Path,
    ) -> None:
        """Test that TTS errors are logged but don't stop generation."""
        from wakeword_workbench.tts.base import TTSError

        mock_backend = MagicMock()
        mock_backend.set_voice.return_value = None

        # Make first call fail, second succeed
        call_count = [0]
        sample_rate = 16000
        duration_sec = 1.0
        num_samples = int(sample_rate * duration_sec)
        success_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, num_samples)).astype(
            np.float32
        )
        success_result = TTSResult(
            audio=success_audio, sample_rate=sample_rate, duration=duration_sec
        )

        def synthesize_side_effect(text: str) -> TTSResult:
            call_count[0] += 1
            if call_count[0] == 1:
                raise TTSError("Simulated TTS failure")
            return success_result

        mock_backend.synthesize.side_effect = synthesize_side_effect
        mock_create_backend.return_value = mock_backend

        # Generate samples (should succeed despite one failure)
        manifest_path = PositiveGenerator(mock_config, output_dir).generate(3)

        # Verify manifest was created with remaining successful samples
        assert manifest_path.exists()
        with open(manifest_path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        # Should have fewer entries due to one failure
        assert len(entries) <= 3

    @patch("soundfile.write")
    @patch("wakeword_workbench.dataset.positive_generator._create_backend_for_provider")
    def test_generate_returns_correct_manifest_path(
        self,
        mock_create_backend: MagicMock,
        mock_sf_write: MagicMock,
        mock_config: Config,
        output_dir: Path,
        mock_tts_result: TTSResult,
    ) -> None:
        """Test that manifest path is correctly returned."""
        mock_backend = MagicMock()
        mock_backend.synthesize.return_value = mock_tts_result
        mock_backend.set_voice.return_value = None
        mock_create_backend.return_value = mock_backend

        generator = PositiveGenerator(mock_config, output_dir)
        manifest_path = generator.generate(1)

        assert manifest_path == output_dir / "positive_manifest.jsonl"

    @patch("soundfile.write")
    @patch("wakeword_workbench.dataset.positive_generator._create_backend_for_provider")
    def test_generate_uses_exact_wake_word_by_default(
        self,
        mock_create_backend: MagicMock,
        mock_sf_write: MagicMock,
        mock_config: Config,
        output_dir: Path,
        mock_tts_result: TTSResult,
    ) -> None:
        """Default generation should use only the exact configured wake word."""
        mock_backend = MagicMock()
        mock_backend.synthesize.return_value = mock_tts_result
        mock_backend.set_voice.return_value = None
        mock_create_backend.return_value = mock_backend

        manifest_path = PositiveGenerator(mock_config, output_dir).generate(1)
        assert manifest_path.exists()
        mock_backend.synthesize.assert_called_once_with("hey vera")

    @patch("soundfile.write")
    @patch("wakeword_workbench.dataset.positive_generator._create_backend_for_provider")
    def test_generate_uses_explicit_wake_word_variants(
        self,
        mock_create_backend: MagicMock,
        mock_sf_write: MagicMock,
        mock_config: Config,
        output_dir: Path,
        mock_tts_result: TTSResult,
    ) -> None:
        """Explicitly configured wake word variants should be used as provided."""
        mock_config.wake_word_variants = ["hey vera", "hey, vera"]
        mock_backend = MagicMock()
        mock_backend.synthesize.return_value = mock_tts_result
        mock_backend.set_voice.return_value = None
        mock_create_backend.return_value = mock_backend

        manifest_path = PositiveGenerator(mock_config, output_dir).generate(4)

        assert manifest_path.exists()
        used_phrases = {call.args[0] for call in mock_backend.synthesize.call_args_list}
        assert used_phrases == {"hey vera", "hey, vera"}

    def test_generate_raises_on_backend_failure(
        self, mock_config: Config, output_dir: Path
    ) -> None:
        """Test that backend initialization failure raises error."""
        with patch(
            "wakeword_workbench.dataset.positive_generator._create_backend_for_provider",
            side_effect=Exception("Backend not available"),
        ):
            generator = PositiveGenerator(mock_config, output_dir)
            with pytest.raises(PositiveGeneratorError, match="Failed to get TTS backend"):
                generator.generate(1)

    @patch("soundfile.write")
    @patch("wakeword_workbench.dataset.positive_generator._create_backend_for_provider")
    def test_wav_files_are_16khz_mono(
        self,
        mock_create_backend: MagicMock,
        mock_sf_write: MagicMock,
        mock_config: Config,
        output_dir: Path,
    ) -> None:
        """Test that generated WAV files are 16000 Hz mono."""
        import soundfile as sf

        # Create stereo audio result
        sample_rate = 24000
        duration_sec = 1.0
        num_samples = int(sample_rate * duration_sec)
        stereo_audio = np.column_stack(
            [
                np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, num_samples)).astype(
                    np.float32
                ),
                np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, num_samples)).astype(
                    np.float32
                ),
            ]
        )
        stereo_result = TTSResult(
            audio=stereo_audio, sample_rate=sample_rate, duration=duration_sec
        )

        mock_backend = MagicMock()
        mock_backend.synthesize.return_value = stereo_result
        mock_backend.set_voice.return_value = None
        mock_create_backend.return_value = mock_backend

        # Make sf.write actually write files so we can read them back
        def real_write(path, audio, sr):
            # Convert stereo to mono and resample to 16kHz like the source does
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            _real_write(path, audio.astype(np.float32), 16000)

        mock_sf_write.side_effect = real_write

        # Generate sample
        manifest_path = PositiveGenerator(mock_config, output_dir).generate(1)
        assert manifest_path.exists()

        # Check WAV file format
        wav_files = list(output_dir.glob("*.wav"))
        assert len(wav_files) == 1

        audio, sr = sf.read(wav_files[0])
        assert sr == 16000
        assert audio.ndim == 1  # Mono

    @patch("soundfile.write")
    @patch("wakeword_workbench.dataset.positive_generator._create_backend_for_provider")
    def test_generate_uses_multiple_providers(
        self,
        mock_create_backend: MagicMock,
        mock_sf_write: MagicMock,
        tmp_path: Path,
        output_dir: Path,
        mock_tts_result: TTSResult,
    ) -> None:
        """Test that generation includes all configured providers."""
        config = Config(
            wake_word="hey_vera",
            samples=SamplesConfig(positives=100, negatives_multiplier=5),
            tts=TTSConfig(
                providers=[
                    TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.0),
                    TTSProviderConfig(backend="piper", voices=["en_US-amy-low"], speed=1.0),
                ]
            ),
            augmentation=AugmentationConfig(
                noise_snr=[-10, 10],
                reverb_probability=0.5,
                gain_range=[-45, 0],
            ),
            output=OutputConfig(path=str(tmp_path / "output"), format=["microwakeword"]),
        )

        kokoro_backend = MagicMock()
        kokoro_backend.synthesize.return_value = mock_tts_result
        kokoro_backend.set_voice.return_value = None

        piper_backend = MagicMock()
        piper_backend.synthesize.return_value = mock_tts_result
        piper_backend.set_voice.side_effect = NotImplementedError(
            "runtime voice changes unsupported"
        )

        def create_backend(provider: TTSProviderConfig) -> MagicMock:
            if provider.backend == "kokoro":
                return kokoro_backend
            if provider.backend == "piper":
                return piper_backend
            raise AssertionError(f"unexpected backend {provider.backend}")

        mock_create_backend.side_effect = create_backend
        mock_sf_write.side_effect = lambda path, audio, sr: _real_write(path, audio, sr)

        manifest_path = PositiveGenerator(config, output_dir).generate(2)

        with open(manifest_path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        assert {entry["backend"] for entry in entries} == {"kokoro", "piper"}
        assert {entry["voice"] for entry in entries} == {"af_sarah", "en_US-amy-low"}
        assert {entry["text"] for entry in entries} == {"hey vera"}

    @patch("wakeword_workbench.dataset.positive_generator.list_available_backends")
    def test_create_backend_for_provider_passes_supported_runtime_options(
        self,
        mock_list_available_backends: MagicMock,
    ) -> None:
        """Test that supported runtime options are passed to backend constructors."""
        from wakeword_workbench.dataset.positive_generator import _create_backend_for_provider

        class SpeedAwareBackend:
            last_speed: float | None = None
            last_acceleration: str | None = None
            last_device: str | None = None
            last_model_path: str | None = None
            last_use_cuda: bool | None = None

            def __init__(
                self,
                speed: float,
                acceleration: str,
                device: str,
                model_path: str,
                use_cuda: bool,
            ) -> None:
                self.speed = speed
                SpeedAwareBackend.last_speed = speed
                SpeedAwareBackend.last_acceleration = acceleration
                SpeedAwareBackend.last_device = device
                SpeedAwareBackend.last_model_path = model_path
                SpeedAwareBackend.last_use_cuda = use_cuda

        provider = TTSProviderConfig(
            backend="kokoro",
            voices=["af_sarah"],
            speed=1.5,
            acceleration="cuda",
            device="cuda:0",
            model_path="/tmp/model.onnx",
        )

        with patch.dict(
            "wakeword_workbench.dataset.positive_generator._BACKENDS",
            {"kokoro": SpeedAwareBackend},
            clear=True,
        ):
            backend = _create_backend_for_provider(provider)

        mock_list_available_backends.assert_called_once()
        assert isinstance(backend, SpeedAwareBackend)
        assert SpeedAwareBackend.last_speed == 1.5
        assert SpeedAwareBackend.last_acceleration == "cuda"
        assert SpeedAwareBackend.last_device == "cuda:0"
        assert SpeedAwareBackend.last_model_path == "/tmp/model.onnx"
        assert SpeedAwareBackend.last_use_cuda is True

    @patch("wakeword_workbench.dataset.positive_generator.log.warning")
    @patch("wakeword_workbench.dataset.positive_generator.list_available_backends")
    def test_create_backend_for_provider_skips_default_warnings_for_unsupported_backend(
        self,
        mock_list_available_backends: MagicMock,
        mock_warning: MagicMock,
    ) -> None:
        """Default CPU settings should not warn when backend lacks optional runtime args."""
        from wakeword_workbench.dataset.positive_generator import _create_backend_for_provider

        class NoSpeedBackend:
            def __init__(self) -> None:
                self.created = True

        provider = TTSProviderConfig(backend="piper", voices=["voice"], speed=1.0)

        with patch.dict(
            "wakeword_workbench.dataset.positive_generator._BACKENDS",
            {"piper": NoSpeedBackend},
            clear=True,
        ):
            backend = _create_backend_for_provider(provider)

        mock_list_available_backends.assert_called_once()
        mock_warning.assert_not_called()
        assert isinstance(backend, NoSpeedBackend)

    @patch("wakeword_workbench.dataset.positive_generator.log.warning")
    @patch("wakeword_workbench.dataset.positive_generator.list_available_backends")
    def test_create_backend_for_provider_warns_for_unsupported_runtime_options(
        self,
        mock_list_available_backends: MagicMock,
        mock_warning: MagicMock,
    ) -> None:
        """Non-default runtime settings should warn when backend ignores them."""
        from wakeword_workbench.dataset.positive_generator import _create_backend_for_provider

        class NoSpeedBackend:
            def __init__(self) -> None:
                self.created = True

        provider = TTSProviderConfig(
            backend="custom",
            voices=["voice"],
            speed=0.8,
            acceleration="openvino",
            device="GPU",
        )

        with patch.dict(
            "wakeword_workbench.dataset.positive_generator._BACKENDS",
            {"custom": NoSpeedBackend},
            clear=True,
        ):
            backend = _create_backend_for_provider(provider)

        mock_list_available_backends.assert_called_once()
        assert mock_warning.call_count == 3
        mock_warning.assert_any_call("backend_no_speed_support", backend="custom", speed=0.8)
        mock_warning.assert_any_call(
            "backend_no_acceleration_support",
            backend="custom",
            acceleration="openvino",
        )
        mock_warning.assert_any_call("backend_no_device_support", backend="custom", device="GPU")
        assert isinstance(backend, NoSpeedBackend)
