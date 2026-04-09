"""Tests for configuration module."""

from pathlib import Path
from textwrap import dedent

import pytest

from wakeword_workbench.config import (
    AugmentationConfig,
    Config,
    ConfigError,
    NegativeConfusionConfig,
    NegativeGenerationConfig,
    NegativeSyntheticConfig,
    OutputConfig,
    SamplesConfig,
    TTSConfig,
    TTSProviderConfig,
    load_config,
)

# --- Valid Config Tests ---

MINIMAL_VALID_CONFIG = dedent("""
wake_word: "hey assistant"
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  providers:
    - backend: kokoro
      voices:
        - af_sarah
      speed: 1.0
      acceleration: cuda
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [microwakeword]
""")

FULL_VALID_CONFIG = dedent("""
wake_word: "hey assistant"
wake_word_variants: ["Hey Assistant", "hey assistant", " assistant "]
samples:
  positives: 1000
  negatives_multiplier: 5
negatives:
  custom_phrases: [archer, archer,  assistant]
  confusion:
    enabled: true
    weight: 0.75
    min_similarity: 0.8
  synthetic:
    enabled: true
    weight: 0.25
    strategy: topic
    min_word_count: 3
    max_word_count: 5
    topics: [technology, weather]
    word_list: [custom, negative, phrase, pool]
tts:
  providers:
    - backend: kokoro
      voices:
        - af_sarah
        - af_nicole
      speed: 1.0
    - backend: piper
      voices:
        - en_US-amy-low
      speed: 1.2
      acceleration: cuda
      model_path: /models/en_US-amy-low.onnx
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [microwakeword, openwakeword]
""")


def test_load_config_valid_minimal(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(MINIMAL_VALID_CONFIG)

    config = load_config(config_file)

    assert config.wake_word == "hey assistant"
    assert config.wake_word_variants is None
    assert config.samples.positives == 1000
    assert config.samples.negatives_multiplier == 5
    assert len(config.tts.providers) == 1
    assert config.tts.providers[0].backend == "kokoro"
    assert config.tts.providers[0].voices == ["af_sarah"]
    assert config.tts.providers[0].speed == 1.0
    assert config.tts.providers[0].acceleration == "cuda"
    assert config.tts.providers[0].device is None
    assert config.tts.providers[0].model_path is None
    assert config.augmentation.noise_snr == [-10, 10]
    assert config.augmentation.reverb_probability == 0.5
    assert config.augmentation.gain_range == [-45, 0]
    assert config.output.path == "./datasets"
    assert config.output.format == ["microwakeword"]
    assert config.negatives.confusion.enabled is True
    assert config.negatives.confusion.weight == 0.6
    assert config.negatives.confusion.min_similarity == 0.6
    assert config.negatives.synthetic.enabled is True
    assert config.negatives.synthetic.weight == 0.4
    assert config.negatives.synthetic.strategy == "random"
    assert config.negatives.synthetic.min_word_count == 2
    assert config.negatives.synthetic.max_word_count == 4
    assert config.negatives.synthetic.topics is None
    assert config.negatives.synthetic.word_list is None


def test_load_config_valid_full(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(FULL_VALID_CONFIG)

    config = load_config(config_file)

    assert config.wake_word == "hey assistant"
    assert config.wake_word_variants == ["Hey Assistant", "assistant"]
    assert len(config.tts.providers) == 2
    assert config.tts.providers[0].backend == "kokoro"
    assert config.tts.providers[0].voices == ["af_sarah", "af_nicole"]
    assert config.tts.providers[0].speed == 1.0
    assert config.tts.providers[0].acceleration == "cpu"
    assert config.tts.providers[1].backend == "piper"
    assert config.tts.providers[1].voices == ["en_US-amy-low"]
    assert config.tts.providers[1].speed == 1.2
    assert config.tts.providers[1].acceleration == "cuda"
    assert config.tts.providers[1].model_path == "/models/en_US-amy-low.onnx"
    assert config.output.format == ["microwakeword", "openwakeword"]
    assert config.negatives.confusion.weight == 0.75
    assert config.negatives.confusion.min_similarity == 0.8
    assert config.negatives.synthetic.weight == 0.25
    assert config.negatives.synthetic.strategy == "topic"
    assert config.negatives.synthetic.min_word_count == 3
    assert config.negatives.synthetic.max_word_count == 5
    assert config.negatives.synthetic.topics == ["technology", "weather"]
    assert config.negatives.synthetic.word_list == ["custom", "negative", "phrase", "pool"]
    assert config.negatives.custom_phrases == ["archer", "assistant"]


def test_load_config_string_path(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(MINIMAL_VALID_CONFIG)

    config = load_config(str(config_file))

    assert config.wake_word == "hey assistant"


def test_load_config_with_openvino_device(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
wake_word: "hey assistant"
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  providers:
    - backend: kokoro
      voices: [af_sarah]
      acceleration: openvino
      device: GPU
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [microwakeword]
""")
    )

    config = load_config(config_file)

    assert config.tts.providers[0].acceleration == "openvino"
    assert config.tts.providers[0].device == "GPU"


# --- File Not Found / Readable Tests ---


def test_load_config_file_not_found() -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(Path("/nonexistent/config.yaml"))


def test_load_config_empty_file(tmp_path: Path) -> None:
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")

    with pytest.raises(ConfigError, match="is empty"):
        load_config(config_file)


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content:")

    with pytest.raises(ConfigError, match="Failed to parse YAML"):
        load_config(config_file)


def test_load_config_rejects_migraphx_acceleration(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
wake_word: "hey assistant"
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  providers:
    - backend: kokoro
      voices: [af_sarah]
      acceleration: migraphx
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [microwakeword]
""")
    )

    with pytest.raises(ConfigError, match="MIGraphX acceleration is not supported"):
        load_config(config_file)


def test_load_config_rejects_piper_openvino(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
wake_word: "hey assistant"
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  providers:
    - backend: piper
      voices: [en_US-amy-low]
      acceleration: openvino
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [microwakeword]
""")
    )

    with pytest.raises(ConfigError, match="Piper supports CPU and CUDA acceleration"):
        load_config(config_file)


# --- Missing Field Tests ---


def test_load_config_missing_wake_word(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  backend: kokoro
  voices: [af_sarah]
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [microwakeword]
""")
    )
    with pytest.raises(ConfigError, match="Missing required field: wake_word"):
        load_config(config_file)


def test_load_config_missing_samples(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
wake_word: "hey assistant"
tts:
  providers:
    - backend: kokoro
      voices: [af_sarah]
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [microwakeword]
""")
    )
    with pytest.raises(ConfigError, match="Missing required field: samples"):
        load_config(config_file)


def test_load_config_missing_tts(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
wake_word: "hey assistant"
samples:
  positives: 1000
  negatives_multiplier: 5
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [microwakeword]
""")
    )
    with pytest.raises(ConfigError, match="Missing required field: tts"):
        load_config(config_file)


def test_load_config_missing_augmentation(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
wake_word: "hey assistant"
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  providers:
    - backend: kokoro
      voices: [af_sarah]
output:
  path: ./datasets
  format: [microwakeword]
""")
    )
    with pytest.raises(ConfigError, match="Missing required field: augmentation"):
        load_config(config_file)


def test_load_config_missing_output(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
wake_word: "hey assistant"
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  providers:
    - backend: kokoro
      voices: [af_sarah]
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
""")
    )
    with pytest.raises(ConfigError, match="Missing required field: output"):
        load_config(config_file)


# --- Validation Tests ---


def test_samples_positives_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="positives must be positive"):
        SamplesConfig(positives=0, negatives_multiplier=5)


def test_samples_negatives_multiplier_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="negatives_multiplier must be positive"):
        SamplesConfig(positives=1000, negatives_multiplier=0)


def test_tts_backend_cannot_be_empty() -> None:
    with pytest.raises(ConfigError, match="backend cannot be empty"):
        TTSProviderConfig(backend="", voices=["af_sarah"])


def test_tts_voices_cannot_be_empty() -> None:
    with pytest.raises(ConfigError, match="voices cannot be empty"):
        TTSProviderConfig(backend="kokoro", voices=[])


def test_tts_speed_range() -> None:
    with pytest.raises(ConfigError, match="speed must be between 0 and 3"):
        TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=0.0)
    with pytest.raises(ConfigError, match="speed must be between 0 and 3"):
        TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=5.0)


def test_tts_config_empty_providers() -> None:
    with pytest.raises(ConfigError, match="providers cannot be empty"):
        TTSConfig(providers=[])


def test_negative_confusion_weight_must_be_non_negative() -> None:
    with pytest.raises(ConfigError, match="confusion weight must be non-negative"):
        NegativeConfusionConfig(weight=-0.1)


def test_negative_confusion_min_similarity_range() -> None:
    with pytest.raises(ConfigError, match="confusion min_similarity must be between 0 and 1"):
        NegativeConfusionConfig(min_similarity=-0.1)
    with pytest.raises(ConfigError, match="confusion min_similarity must be between 0 and 1"):
        NegativeConfusionConfig(min_similarity=1.1)


def test_negative_synthetic_weight_must_be_non_negative() -> None:
    with pytest.raises(ConfigError, match="synthetic weight must be non-negative"):
        NegativeSyntheticConfig(weight=-0.1)


def test_negative_synthetic_strategy_validation() -> None:
    with pytest.raises(ConfigError, match="synthetic strategy must be one of"):
        NegativeSyntheticConfig(strategy="invalid")


def test_negative_synthetic_word_count_validation() -> None:
    with pytest.raises(ConfigError, match="synthetic min_word_count must be at least 1"):
        NegativeSyntheticConfig(min_word_count=0)
    with pytest.raises(
        ConfigError,
        match="synthetic max_word_count must be greater than or equal to min_word_count",
    ):
        NegativeSyntheticConfig(min_word_count=3, max_word_count=2)


def test_negative_generation_requires_enabled_source() -> None:
    with pytest.raises(ConfigError, match="at least one negative source must be enabled"):
        NegativeGenerationConfig(
            confusion=NegativeConfusionConfig(enabled=False),
            synthetic=NegativeSyntheticConfig(enabled=False),
        )


def test_negative_generation_requires_positive_enabled_weight() -> None:
    with pytest.raises(
        ConfigError, match="enabled negative sources must have a positive total weight"
    ):
        NegativeGenerationConfig(
            confusion=NegativeConfusionConfig(enabled=True, weight=0.0),
            synthetic=NegativeSyntheticConfig(enabled=False),
        )


def test_negative_generation_rejects_empty_custom_phrase_list() -> None:
    with pytest.raises(ConfigError, match="custom_phrases cannot be empty when provided"):
        NegativeGenerationConfig(custom_phrases=[])


def test_negative_generation_rejects_blank_custom_phrases() -> None:
    with pytest.raises(ConfigError, match="custom_phrases cannot contain empty values"):
        NegativeGenerationConfig(custom_phrases=["archer", "   "])


def test_negative_generation_normalizes_custom_phrases() -> None:
    config = NegativeGenerationConfig(custom_phrases=[" Archer ", "archer", "RJ"])

    assert config.custom_phrases == ["Archer", "RJ"]


def test_config_rejects_empty_wake_word_variants() -> None:
    with pytest.raises(ConfigError, match="wake_word_variants cannot be empty when provided"):
        Config(
            wake_word="hey assistant",
            wake_word_variants=[],
            samples=SamplesConfig(positives=10, negatives_multiplier=2),
            tts=TTSConfig(providers=[TTSProviderConfig(backend="kokoro", voices=["af_sarah"])]),
            augmentation=AugmentationConfig(
                noise_snr=[-10, 10],
                reverb_probability=0.5,
                gain_range=[-45, 0],
            ),
            output=OutputConfig(path="./datasets", format=["microwakeword"]),
        )


def test_config_normalizes_wake_word_variants() -> None:
    config = Config(
        wake_word="hey assistant",
        wake_word_variants=[" Hey Assistant ", "hey assistant", "assistant"],
        samples=SamplesConfig(positives=10, negatives_multiplier=2),
        tts=TTSConfig(providers=[TTSProviderConfig(backend="kokoro", voices=["af_sarah"])]),
        augmentation=AugmentationConfig(
            noise_snr=[-10, 10],
            reverb_probability=0.5,
            gain_range=[-45, 0],
        ),
        output=OutputConfig(path="./datasets", format=["microwakeword"]),
    )

    assert config.wake_word_variants == ["Hey Assistant", "assistant"]


def test_augmentation_noise_snr_requires_two_values() -> None:
    with pytest.raises(ConfigError, match="noise_snr must have exactly 2 values"):
        AugmentationConfig(noise_snr=[-10], reverb_probability=0.5, gain_range=[-45, 0])


def test_augmentation_noise_snr_min_max_order() -> None:
    with pytest.raises(ConfigError, match="noise_snr\\[0\\] must be <= noise_snr\\[1\\]"):
        AugmentationConfig(noise_snr=[10, -10], reverb_probability=0.5, gain_range=[-45, 0])


def test_augmentation_reverb_probability_range() -> None:
    with pytest.raises(ConfigError, match="reverb_probability must be between 0 and 1"):
        AugmentationConfig(noise_snr=[-10, 10], reverb_probability=-0.1, gain_range=[-45, 0])
    with pytest.raises(ConfigError, match="reverb_probability must be between 0 and 1"):
        AugmentationConfig(noise_snr=[-10, 10], reverb_probability=1.5, gain_range=[-45, 0])


def test_augmentation_gain_range_requires_two_values() -> None:
    with pytest.raises(ConfigError, match="gain_range must have exactly 2 values"):
        AugmentationConfig(noise_snr=[-10, 10], reverb_probability=0.5, gain_range=[-45])


def test_augmentation_gain_range_min_max_order() -> None:
    with pytest.raises(ConfigError, match="gain_range\\[0\\] must be <= gain_range\\[1\\]"):
        AugmentationConfig(noise_snr=[-10, 10], reverb_probability=0.5, gain_range=[0, -45])


def test_output_path_cannot_be_empty() -> None:
    with pytest.raises(ConfigError, match="path cannot be empty"):
        OutputConfig(path="", format=["microwakeword"])


def test_output_format_cannot_be_empty() -> None:
    with pytest.raises(ConfigError, match="format cannot be empty"):
        OutputConfig(path="./datasets", format=[])


def test_output_format_invalid_value(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
wake_word: "hey assistant"
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  providers:
    - backend: kokoro
      voices: [af_sarah]
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [invalid_format]
""")
    )
    with pytest.raises(ConfigError, match="output.format contains invalid format"):
        load_config(config_file)


def test_wake_word_cannot_be_empty(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
wake_word: ""
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  providers:
    - backend: kokoro
      voices: [af_sarah]
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [microwakeword]
""")
    )
    with pytest.raises(ConfigError, match="wake_word cannot be empty"):
        load_config(config_file)


# --- Edge Cases ---


def test_load_config_with_comments(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        dedent("""
# This is a comment
wake_word: "hey assistant"  # inline comment
samples:
  positives: 1000  # number of positive samples
  negatives_multiplier: 5
tts:
  providers:
    - backend: kokoro
      voices:
        - af_sarah  # voice 1
        - af_nicole  # voice 2
      speed: 1.0
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: ./datasets
  format: [microwakeword, openwakeword]
""")
    )
    config = load_config(config_file)
    assert config.wake_word == "hey assistant"
    assert len(config.tts.providers[0].voices) == 2


def test_config_error_inherits_from_exception() -> None:
    """ConfigError should be usable as a standard exception."""
    error = ConfigError("test error")
    assert isinstance(error, Exception)
    assert str(error) == "test error"
