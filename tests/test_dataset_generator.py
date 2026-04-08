"""Tests for DatasetGenerator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wakeword_workbench.config import (
    AugmentationConfig,
    Config,
    OutputConfig,
    SamplesConfig,
    TTSConfig,
    TTSProviderConfig,
)
from wakeword_workbench.dataset.generator import (
    DatasetGenerator,
    GenerationResult,
    GeneratorConfigError,
    GeneratorError,
    GeneratorIOError,
    GeneratorMergeError,
    GeneratorSplitError,
    GeneratorTTSError,
)
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.tts.base import TTSResult


class TestGeneratorErrorHierarchy:
    """Test GeneratorError exception hierarchy."""

    def test_generator_error_is_exception(self) -> None:
        """GeneratorError should inherit from Exception."""
        error = GeneratorError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_generator_config_error_inherits_from_generator_error(self) -> None:
        """GeneratorConfigError should inherit from GeneratorError."""
        error = GeneratorConfigError("config error")
        assert isinstance(error, GeneratorError)
        assert isinstance(error, Exception)
        assert str(error) == "config error"

    def test_generator_tts_error_inherits_from_generator_error(self) -> None:
        """GeneratorTTSError should inherit from GeneratorError."""
        error = GeneratorTTSError("TTS error")
        assert isinstance(error, GeneratorError)
        assert isinstance(error, Exception)
        assert str(error) == "TTS error"

    def test_generator_merge_error_inherits_from_generator_error(self) -> None:
        """GeneratorMergeError should inherit from GeneratorError."""
        error = GeneratorMergeError("merge error")
        assert isinstance(error, GeneratorError)
        assert isinstance(error, Exception)
        assert str(error) == "merge error"

    def test_generator_split_error_inherits_from_generator_error(self) -> None:
        """GeneratorSplitError should inherit from GeneratorError."""
        error = GeneratorSplitError("split error")
        assert isinstance(error, GeneratorError)
        assert isinstance(error, Exception)
        assert str(error) == "split error"

    def test_generator_io_error_inherits_from_generator_error(self) -> None:
        """GeneratorIOError should inherit from GeneratorError."""
        error = GeneratorIOError("IO error")
        assert isinstance(error, GeneratorError)
        assert isinstance(error, Exception)
        assert str(error) == "IO error"

    def test_error_can_be_caught_as_base_generator_error(self) -> None:
        """Specific errors should be catchable as GeneratorError."""
        with pytest.raises(GeneratorError, match="specific error"):
            raise GeneratorTTSError("specific error")


class TestGenerationResult:
    """Test GenerationResult dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """GenerationResult should be creatable with required fields."""
        output_dir = Path("/tmp/test_output")
        train_manifest = Manifest()
        val_manifest = Manifest()
        test_manifest = Manifest()

        result = GenerationResult(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            total_positives=100,
            total_negatives=500,
            total_entries=600,
            train_count=420,
            val_count=90,
            test_count=90,
            target_ratio=5.0,
            actual_ratio=5.0,
            output_dir=output_dir,
            generation_time_seconds=1.5,
        )

        assert result.total_positives == 100
        assert result.total_negatives == 500
        assert result.total_entries == 600
        assert result.train_count == 420
        assert result.val_count == 90
        assert result.test_count == 90
        assert result.target_ratio == 5.0
        assert result.actual_ratio == 5.0
        assert result.output_dir == output_dir
        assert result.generation_time_seconds == 1.5
        assert result.warnings == []

    def test_creation_with_warnings(self) -> None:
        """GenerationResult should accept warnings list."""
        output_dir = Path("/tmp/test_output")
        train_manifest = Manifest()
        val_manifest = Manifest()
        test_manifest = Manifest()

        result = GenerationResult(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            total_positives=100,
            total_negatives=500,
            total_entries=600,
            train_count=420,
            val_count=90,
            test_count=90,
            target_ratio=5.0,
            actual_ratio=5.0,
            output_dir=output_dir,
            generation_time_seconds=1.5,
            warnings=["Test warning 1", "Test warning 2"],
        )

        assert len(result.warnings) == 2
        assert "Test warning 1" in result.warnings
        assert "Test warning 2" in result.warnings

    def test_has_warnings_returns_true_when_warnings_present(self) -> None:
        """has_warnings should return True when warnings exist."""
        output_dir = Path("/tmp/test_output")
        result = GenerationResult(
            train_manifest=Manifest(),
            val_manifest=Manifest(),
            test_manifest=Manifest(),
            total_positives=100,
            total_negatives=500,
            total_entries=600,
            train_count=420,
            val_count=90,
            test_count=90,
            target_ratio=5.0,
            actual_ratio=5.0,
            output_dir=output_dir,
            generation_time_seconds=1.5,
            warnings=["Some warning"],
        )
        assert result.has_warnings() is True

    def test_has_warnings_returns_false_when_no_warnings(self) -> None:
        """has_warnings should return False when no warnings."""
        output_dir = Path("/tmp/test_output")
        result = GenerationResult(
            train_manifest=Manifest(),
            val_manifest=Manifest(),
            test_manifest=Manifest(),
            total_positives=100,
            total_negatives=500,
            total_entries=600,
            train_count=420,
            val_count=90,
            test_count=90,
            target_ratio=5.0,
            actual_ratio=5.0,
            output_dir=output_dir,
            generation_time_seconds=1.5,
        )
        assert result.has_warnings() is False

    def test_summary_returns_dict(self) -> None:
        """summary should return a dictionary with expected keys."""
        output_dir = Path("/tmp/test_output")
        result = GenerationResult(
            train_manifest=Manifest(),
            val_manifest=Manifest(),
            test_manifest=Manifest(),
            total_positives=100,
            total_negatives=500,
            total_entries=600,
            train_count=420,
            val_count=90,
            test_count=90,
            target_ratio=5.0,
            actual_ratio=5.0,
            output_dir=output_dir,
            generation_time_seconds=1.5,
            warnings=["Warning 1"],
        )

        summary = result.summary()
        assert summary["total_entries"] == 600
        assert summary["positives"] == 100
        assert summary["negatives"] == 500
        assert summary["ratio"] == 5.0
        assert summary["splits"]["train"] == 420
        assert summary["splits"]["val"] == 90
        assert summary["splits"]["test"] == 90
        assert str(output_dir) in summary["output_dir"]
        assert summary["warnings"] == 1


class TestDatasetGeneratorInstantiation:
    """Test DatasetGenerator initialization."""

    def _make_config(self, **overrides) -> Config:
        """Create a config with optional overrides."""
        samples = SamplesConfig(
            positives=overrides.get("positives", 1000),
            negatives_multiplier=overrides.get("negatives_multiplier", 5),
        )
        tts = TTSConfig(
            providers=overrides.get(
                "providers",
                [
                    TTSProviderConfig(
                        backend=overrides.get("backend", "kokoro"),
                        voices=overrides.get("voices", ["af_sarah"]),
                        speed=overrides.get("speed", 1.0),
                    )
                ],
            ),
        )
        augmentation = AugmentationConfig(
            noise_snr=[-10, 10],
            reverb_probability=0.5,
            gain_range=[-45, 0],
        )
        output = OutputConfig(
            path=overrides.get("output_path", "/tmp/test_output"),
            format=["microwakeword"],
        )
        return Config(
            wake_word=overrides.get("wake_word", "hey_vera"),
            samples=samples,
            tts=tts,
            augmentation=augmentation,
            output=output,
        )

    def test_instantiation_with_valid_config(self, sample_config: Config) -> None:
        """DatasetGenerator should instantiate with valid config."""
        generator = DatasetGenerator(sample_config)
        assert generator.config == sample_config
        assert generator.output_dir == Path(sample_config.output.path)

    def test_instantiation_with_custom_output_dir(
        self, sample_config: Config, tmp_path: Path
    ) -> None:
        """DatasetGenerator should use custom output_dir when provided."""
        custom_output = tmp_path / "custom_output"
        generator = DatasetGenerator(sample_config, output_dir=custom_output)
        assert generator.output_dir == custom_output

    def test_instantiation_raises_on_whitespace_only_wake_word(self, tmp_path: Path) -> None:
        """DatasetGenerator should raise GeneratorConfigError for whitespace-only wake_word."""
        # Config accepts whitespace-only strings, but DatasetGenerator rejects them after strip()
        config = self._make_config(wake_word="   ")
        with pytest.raises(GeneratorConfigError, match="wake_word cannot be empty"):
            DatasetGenerator(config)

    def test_instantiation_with_valid_config_values(self, sample_config: Config) -> None:
        """DatasetGenerator should accept valid config values (Config validates at creation time)."""
        # These tests document that Config validates its values at creation,
        # so DatasetGenerator doesn't need to re-validate them
        # (ConfigError is raised during Config creation, not DatasetGenerator init)
        assert sample_config.wake_word == "hey_vera"
        assert sample_config.samples.positives == 1000
        assert sample_config.samples.negatives_multiplier == 5
        assert len(sample_config.tts.providers) == 1

    def test_config_validation_occurs_at_config_creation_time(self) -> None:
        """Config validates values at creation, so DatasetGenerator re-validation is redundant."""
        # Config classes validate in __post_init__, so these raise ConfigError (not GeneratorConfigError)
        # These are tested in test_config.py - this test documents the architecture
        from wakeword_workbench.config import (
            ConfigError,
            SamplesConfig,
            TTSConfig,
            TTSProviderConfig,
        )

        with pytest.raises(ConfigError):
            SamplesConfig(positives=0, negatives_multiplier=5)

        with pytest.raises(ConfigError):
            SamplesConfig(positives=1000, negatives_multiplier=0)

        with pytest.raises(ConfigError):
            TTSConfig(providers=[TTSProviderConfig(backend="kokoro", voices=[])])


class TestDatasetGeneratorGenerate:
    """Test DatasetGenerator.generate() method."""

    @pytest.fixture
    def mock_generator(self, sample_config: Config, tmp_dataset_dir: Path) -> DatasetGenerator:
        """Create a DatasetGenerator with mocked dependencies."""
        return DatasetGenerator(sample_config, output_dir=tmp_dataset_dir)

    def test_generate_raises_on_invalid_positive_count(
        self, mock_generator: DatasetGenerator
    ) -> None:
        """generate should raise GeneratorConfigError for zero positive_count."""
        with pytest.raises(GeneratorConfigError, match="positive_count must be positive"):
            mock_generator.generate(positive_count=0)

    def test_generate_raises_on_negative_positive_count(
        self, mock_generator: DatasetGenerator
    ) -> None:
        """generate should raise GeneratorConfigError for negative positive_count."""
        with pytest.raises(GeneratorConfigError, match="positive_count must be positive"):
            mock_generator.generate(positive_count=-1)

    def test_generate_raises_on_invalid_negatives_multiplier(
        self, mock_generator: DatasetGenerator
    ) -> None:
        """generate should raise GeneratorConfigError for zero negatives_multiplier."""
        with pytest.raises(GeneratorConfigError, match="negatives_multiplier must be positive"):
            mock_generator.generate(negatives_multiplier=0)

    def test_generate_raises_on_invalid_ratio_sum(self, mock_generator: DatasetGenerator) -> None:
        """generate should raise GeneratorConfigError if ratios don't sum to 1.0."""
        with pytest.raises(
            GeneratorConfigError, match="train_ratio \\+ val_ratio \\+ test_ratio must sum to 1.0"
        ):
            mock_generator.generate(train_ratio=0.6, val_ratio=0.2, test_ratio=0.1)

    def test_generate_raises_on_invalid_ratio(self, mock_generator: DatasetGenerator) -> None:
        """generate should raise GeneratorConfigError for non-positive ratio."""
        with pytest.raises(GeneratorConfigError, match="ratio must be positive"):
            mock_generator.generate(ratio=0)

    @patch.object(DatasetGenerator, "_save_splits")
    @patch("wakeword_workbench.dataset.generator.PositiveGenerator")
    @patch("wakeword_workbench.dataset.generator.Manifest")
    @patch("wakeword_workbench.dataset.generator._create_backend_with_speed")
    @patch("wakeword_workbench.dataset.generator.merge")
    @patch("wakeword_workbench.dataset.generator.split")
    def test_generate_success(
        self,
        mock_split: MagicMock,
        mock_merge: MagicMock,
        mock_create_backend: MagicMock,
        mock_manifest: MagicMock,
        mock_pos_gen: MagicMock,
        mock_save_splits: MagicMock,
        mock_generator: DatasetGenerator,
    ) -> None:
        """generate should return GenerationResult on success."""
        # Setup mock positive generator
        mock_pos_instance = MagicMock()
        mock_pos_instance.generate.return_value = Path("/tmp/pos_manifest.jsonl")
        mock_pos_gen.return_value = mock_pos_instance

        # Setup mock manifest loading
        pos_entries = [
            ManifestEntry(path="pos1.wav", label=1, text="hey vera", voice="af_sarah"),
            ManifestEntry(path="pos2.wav", label=1, text="hey vera", voice="af_sarah"),
        ]
        neg_entries = [
            ManifestEntry(path="neg1.wav", label=0, text="hey vera voice", voice="af_sarah"),
        ]

        mock_pos_manifest = Manifest(pos_entries)
        mock_neg_manifest = Manifest(neg_entries)
        mock_combined = Manifest(pos_entries + neg_entries)

        def load_side_effect(path: Path | str) -> Manifest:
            if "pos" in str(path):
                return mock_pos_manifest
            return mock_neg_manifest

        mock_manifest.load.side_effect = load_side_effect

        # Setup mock TTS backend
        mock_backend = MagicMock()
        mock_backend.synthesize.return_value = TTSResult(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            duration=1.0,
        )
        mock_create_backend.return_value = mock_backend

        # Setup mock merge
        mock_merge.return_value = mock_combined

        # Setup mock split
        train_entries = [
            ManifestEntry(path="train1.wav", label=1, text="hey vera", duration_ms=1000)
        ]
        val_entries = [ManifestEntry(path="val1.wav", label=1, text="hey vera", duration_ms=1000)]
        test_entries = [ManifestEntry(path="test1.wav", label=1, text="hey vera", duration_ms=1000)]
        mock_split.return_value = (
            Manifest(train_entries),
            Manifest(val_entries),
            Manifest(test_entries),
        )

        # Execute
        result = mock_generator.generate(positive_count=2, negatives_multiplier=1, ratio=1.0)

        # Verify
        assert isinstance(result, GenerationResult)
        assert result.total_positives >= 1
        assert result.total_negatives >= 1
        assert result.total_entries >= 2
        assert result.output_dir == mock_generator.output_dir
        mock_create_backend.assert_called()


class TestDatasetGeneratorNegativePhrases:
    """Test negative phrase generation methods."""

    @pytest.fixture
    def mock_generator(self, sample_config: Config, tmp_dataset_dir: Path) -> DatasetGenerator:
        """Create a DatasetGenerator with mocked dependencies."""
        return DatasetGenerator(sample_config, output_dir=tmp_dataset_dir)

    def test_generate_negative_phrases_returns_empty_for_zero(
        self, mock_generator: DatasetGenerator
    ) -> None:
        """_generate_negative_phrases should return empty list for count=0."""
        result = mock_generator._generate_negative_phrases(0)
        assert result == []

    @patch("wakeword_workbench.dataset.generator.generate_synthetic_negatives")
    def test_generate_synthetic_phrases_calls_correct_function(
        self, mock_synth: MagicMock, mock_generator: DatasetGenerator
    ) -> None:
        """_generate_synthetic_phrases should call generate_synthetic_negatives."""
        mock_synth.return_value = ["phrase1", "phrase2"]
        result = mock_generator._generate_synthetic_phrases(5)
        mock_synth.assert_called_once_with(count=5, wake_word=mock_generator.config.wake_word)
        assert result == ["phrase1", "phrase2"]

    def test_generate_synthetic_phrases_returns_empty_for_zero(
        self, mock_generator: DatasetGenerator
    ) -> None:
        """_generate_synthetic_phrases should return empty list for count=0."""
        result = mock_generator._generate_synthetic_phrases(0)
        assert result == []

    @patch("wakeword_workbench.dataset.generator.generate_confusions")
    def test_generate_confusion_phrases_calls_generate_confusions(
        self, mock_confuse: MagicMock, mock_generator: DatasetGenerator
    ) -> None:
        """_generate_confusion_phrases should call generate_confusions."""
        mock_confuse.return_value = ["confusion1", "confusion2"]
        result = mock_generator._generate_confusion_phrases(5)
        mock_confuse.assert_called_once_with(mock_generator.config.wake_word, count=5)
        assert result == ["confusion1", "confusion2"]

    def test_generate_confusion_phrases_returns_empty_for_zero(
        self, mock_generator: DatasetGenerator
    ) -> None:
        """_generate_confusion_phrases should return empty list for count=0."""
        result = mock_generator._generate_confusion_phrases(0)
        assert result == []

    @patch("wakeword_workbench.dataset.generator.generate_confusions")
    def test_generate_confusion_phrases_raises_on_import_error(
        self, mock_confuse: MagicMock, mock_generator: DatasetGenerator
    ) -> None:
        """_generate_confusion_phrases should raise GeneratorError on ImportError."""
        mock_confuse.side_effect = ImportError("jellyfish not installed")
        with pytest.raises(GeneratorError, match="Failed generating confusion phrases"):
            mock_generator._generate_confusion_phrases(5)


class TestDatasetGeneratorVoiceSelection:
    """Test voice selection methods."""

    @pytest.fixture
    def mock_generator(self, sample_config: Config) -> DatasetGenerator:
        """Create a DatasetGenerator for testing voice selection."""
        return DatasetGenerator(sample_config)

    def test_select_voice_round_robin(self, mock_generator: DatasetGenerator) -> None:
        """_select_voice_for_phrase should round-robin through voices."""
        provider_voices = mock_generator.config.tts.get_all_voices()
        assert len(provider_voices) >= 1

        # First call
        provider1, voice1 = mock_generator._select_voice_for_phrase("phrase 1")
        assert (provider1.backend, voice1) == provider_voices[0]

        # Second call
        provider2, voice2 = mock_generator._select_voice_for_phrase("phrase 2")
        if len(provider_voices) > 1:
            assert (provider2.backend, voice2) == provider_voices[1]
        else:
            assert (provider2.backend, voice2) == provider_voices[0]

    def test_select_voice_wraps_around(self, mock_generator: DatasetGenerator) -> None:
        """_select_voice_for_phrase should wrap around after all voices used."""
        provider_voices = mock_generator.config.tts.get_all_voices()
        if len(provider_voices) < 2:
            pytest.skip("Need at least 2 voices to test wrapping")

        # Call more times than there are voices
        for i in range(len(provider_voices) + 1):
            mock_generator._select_voice_for_phrase(f"phrase {i}")

        # Next call should wrap to first voice
        provider, voice = mock_generator._select_voice_for_phrase("wrapped phrase")
        assert (provider.backend, voice) == provider_voices[0]

    def test_select_voice_round_robin_across_multiple_providers(self, tmp_path: Path) -> None:
        """_select_voice_for_phrase should cycle across all backend-voice pairs."""
        config = Config(
            wake_word="hey_vera",
            samples=SamplesConfig(positives=100, negatives_multiplier=5),
            tts=TTSConfig(
                providers=[
                    TTSProviderConfig(backend="kokoro", voices=["v1", "v2"], speed=1.0),
                    TTSProviderConfig(backend="piper", voices=["v3"], speed=1.0),
                ]
            ),
            augmentation=AugmentationConfig(
                noise_snr=[-10, 10],
                reverb_probability=0.5,
                gain_range=[-45, 0],
            ),
            output=OutputConfig(path=str(tmp_path / "output"), format=["microwakeword"]),
        )
        generator = DatasetGenerator(config, output_dir=tmp_path / "dataset")

        sequence = [generator._select_voice_for_phrase(f"phrase {i}") for i in range(6)]

        assert [(provider.backend, voice) for provider, voice in sequence] == [
            ("kokoro", "v1"),
            ("kokoro", "v2"),
            ("piper", "v3"),
            ("kokoro", "v1"),
            ("kokoro", "v2"),
            ("piper", "v3"),
        ]


class TestDatasetGeneratorHelpers:
    """Test helper methods."""

    @pytest.fixture
    def mock_generator(self, sample_config: Config, tmp_dataset_dir: Path) -> DatasetGenerator:
        """Create a DatasetGenerator for testing."""
        return DatasetGenerator(sample_config, output_dir=tmp_dataset_dir)

    def test_ensure_output_dir_creates_directory(self, mock_generator: DatasetGenerator) -> None:
        """_ensure_output_dir should create the directory if it doesn't exist."""
        mock_generator._ensure_output_dir()
        assert mock_generator.output_dir.exists()
        assert mock_generator.output_dir.is_dir()

    def test_ensure_output_dir_returns_path(self, mock_generator: DatasetGenerator) -> None:
        """_ensure_output_dir should return the output_dir path."""
        result = mock_generator._ensure_output_dir()
        assert result == mock_generator.output_dir

    def test_ensure_mono_returns_same_for_1d_array(self, mock_generator: DatasetGenerator) -> None:
        """_ensure_mono should return same array for 1D input."""
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = mock_generator._ensure_mono(audio)
        np.testing.assert_array_equal(result, audio)

    def test_ensure_mono_averages_multichannel(self, mock_generator: DatasetGenerator) -> None:
        """_ensure_mono should average channels for multi-channel input."""
        audio = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = mock_generator._ensure_mono(audio)
        expected = np.array([1.5, 3.5, 5.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_ensure_format_returns_same_for_16khz(self, mock_generator: DatasetGenerator) -> None:
        """_ensure_format should return same array for 16kHz input."""
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = mock_generator._ensure_format(audio, 16000)
        np.testing.assert_array_equal(result, audio)

    @patch("wakeword_workbench.dataset.generator.import_module")
    def test_ensure_format_resamples_different_sample_rates(
        self, mock_import: MagicMock, mock_generator: DatasetGenerator
    ) -> None:
        """_ensure_format should resample audio from different sample rates."""
        mock_librosa = MagicMock()
        mock_import.return_value = mock_librosa

        audio = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        mock_librosa.resample.return_value = np.array([1.0, 2.0], dtype=np.float32)

        result = mock_generator._ensure_format(audio, 32000)

        mock_librosa.resample.assert_called_once_with(audio, orig_sr=32000, target_sr=16000)
        assert result.dtype == np.float32


class TestDatasetGeneratorMerge:
    """Test merge-related methods."""

    @pytest.fixture
    def mock_generator(self, sample_config: Config, tmp_dataset_dir: Path) -> DatasetGenerator:
        """Create a DatasetGenerator for testing."""
        return DatasetGenerator(sample_config, output_dir=tmp_dataset_dir)

    def test_merge_manifests_calls_merge_function(self, mock_generator: DatasetGenerator) -> None:
        """_merge_manifests should call the merge function."""
        pos_manifest = Manifest(
            [
                ManifestEntry(path="pos1.wav", label=1, text="hey vera"),
            ]
        )
        neg_manifest = Manifest(
            [
                ManifestEntry(path="neg1.wav", label=0, text="not hey vera"),
            ]
        )

        with patch("wakeword_workbench.dataset.generator.merge") as mock_merge:
            mock_merge.return_value = Manifest()
            mock_generator._validate_files = False
            mock_generator._merge_manifests(pos_manifest, neg_manifest, ratio=5.0)

            mock_merge.assert_called_once_with(
                pos_manifest,
                neg_manifest,
                ratio=5.0,
                validate_files=False,
            )

    def test_merge_manifests_raises_on_merger_error(self, mock_generator: DatasetGenerator) -> None:
        """_merge_manifests should raise GeneratorMergeError on MergerError."""
        from wakeword_workbench.dataset.merger import MergerError

        pos_manifest = Manifest()
        neg_manifest = Manifest()

        with patch("wakeword_workbench.dataset.generator.merge") as mock_merge:
            mock_merge.side_effect = MergerError("merge failed")
            with pytest.raises(GeneratorMergeError, match="Failed merging manifests"):
                mock_generator._merge_manifests(pos_manifest, neg_manifest, ratio=5.0)


class TestDatasetGeneratorSplit:
    """Test split-related methods."""

    @pytest.fixture
    def mock_generator(self, sample_config: Config, tmp_dataset_dir: Path) -> DatasetGenerator:
        """Create a DatasetGenerator for testing."""
        return DatasetGenerator(sample_config, output_dir=tmp_dataset_dir)

    def test_split_manifest_calls_split_function(self, mock_generator: DatasetGenerator) -> None:
        """_split_manifest should call the split function."""
        manifest = Manifest(
            [
                ManifestEntry(path="file1.wav", label=1, text="hey vera"),
            ]
        )

        with patch("wakeword_workbench.dataset.generator.split") as mock_split:
            train = Manifest([ManifestEntry(path="t1.wav", label=1, text="hey vera")])
            val = Manifest([ManifestEntry(path="v1.wav", label=1, text="hey vera")])
            test = Manifest([ManifestEntry(path="e1.wav", label=1, text="hey vera")])
            mock_split.return_value = (train, val, test)

            result = mock_generator._split_manifest(manifest, 0.7, 0.15, 0.15)

            mock_split.assert_called_once_with(
                manifest,
                train=0.7,
                val=0.15,
                test=0.15,
                by="speaker",
            )
            assert len(result) == 3

    def test_split_manifest_raises_on_split_validation_error(
        self, mock_generator: DatasetGenerator
    ) -> None:
        """_split_manifest should raise GeneratorSplitError on SplitValidationError."""
        from wakeword_workbench.dataset.splitter import SplitValidationError

        manifest = Manifest()

        with patch("wakeword_workbench.dataset.generator.split") as mock_split:
            mock_split.side_effect = SplitValidationError("validation failed")
            with pytest.raises(GeneratorSplitError, match="Failed splitting manifest"):
                mock_generator._split_manifest(manifest, 0.7, 0.15, 0.15)


class TestDatasetGeneratorSave:
    """Test save-related methods."""

    @pytest.fixture
    def mock_generator(self, sample_config: Config, tmp_dataset_dir: Path) -> DatasetGenerator:
        """Create a DatasetGenerator for testing."""
        return DatasetGenerator(sample_config, output_dir=tmp_dataset_dir)

    def test_save_splits_calls_save_on_manifests(self, mock_generator: DatasetGenerator) -> None:
        """_save_splits should call save on each manifest."""
        train = Manifest(
            [ManifestEntry(path="train.wav", label=1, text="hey vera", duration_ms=1000)]
        )
        val = Manifest([ManifestEntry(path="val.wav", label=1, text="hey vera", duration_ms=1000)])
        test = Manifest(
            [ManifestEntry(path="test.wav", label=1, text="hey vera", duration_ms=1000)]
        )

        mock_generator._save_splits(train, val, test)

        assert (mock_generator.output_dir / "train.jsonl").exists()
        assert (mock_generator.output_dir / "val.jsonl").exists()
        assert (mock_generator.output_dir / "test.jsonl").exists()

    def test_save_splits_raises_on_manifest_error(self, mock_generator: DatasetGenerator) -> None:
        """_save_splits should raise GeneratorIOError on ManifestError."""
        from wakeword_workbench.dataset.metadata import ManifestError

        train = MagicMock(spec=Manifest)
        train.save.side_effect = ManifestError("save failed")
        val = MagicMock(spec=Manifest)
        test = MagicMock(spec=Manifest)

        with pytest.raises(GeneratorIOError, match="Failed saving split manifests"):
            mock_generator._save_splits(train, val, test)
