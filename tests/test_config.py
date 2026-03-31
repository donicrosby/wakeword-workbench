"""Tests for configuration module."""

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from wakeword_workbench.config import (
    AugmentationConfig,
    Config,
    ConfigError,
    OutputConfig,
    SamplesConfig,
    TTSConfig,
    load_config,
)


# --- Valid Config Tests ---

MINIMAL_VALID_CONFIG = dedent("""
wake_word: "hey assistant"
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  backend: kokoro
  voices:
    - af_sarah
  speed: 1.0
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
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  backend: kokoro
  voices:
    - af_sarah
    - af_nicole
  speed: 1.0
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
    assert config.samples.positives == 1000
    assert config.samples.negatives_multiplier == 5
    assert config.tts.backend == "kokoro"
    assert config.tts.voices == ["af_sarah"]
    assert config.tts.speed == 1.0
    assert config.augmentation.noise_snr == [-10, 10]
    assert config.augmentation.reverb_probability == 0.5
    assert config.augmentation.gain_range == [-45, 0]
    assert config.output.path == "./datasets"
    assert config.output.format == ["microwakeword"]


def test_load_config_valid_full(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(FULL_VALID_CONFIG)

    config = load_config(config_file)

    assert config.wake_word == "hey assistant"
    assert config.tts.voices == ["af_sarah", "af_nicole"]
    assert config.output.format == ["microwakeword", "openwakeword"]


def test_load_config_string_path(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(MINIMAL_VALID_CONFIG)

    config = load_config(str(config_file))

    assert config.wake_word == "hey assistant"


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
  backend: kokoro
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
  backend: kokoro
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
        TTSConfig(backend="", voices=["af_sarah"])


def test_tts_voices_cannot_be_empty() -> None:
    with pytest.raises(ConfigError, match="voices cannot be empty"):
        TTSConfig(backend="kokoro", voices=[])


def test_tts_speed_range() -> None:
    with pytest.raises(ConfigError, match="speed must be between 0 and 3"):
        TTSConfig(backend="kokoro", voices=["af_sarah"], speed=0.0)
    with pytest.raises(ConfigError, match="speed must be between 0 and 3"):
        TTSConfig(backend="kokoro", voices=["af_sarah"], speed=5.0)


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
  backend: kokoro
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
  backend: kokoro
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
    assert len(config.tts.voices) == 2


def test_config_error_inherits_from_exception() -> None:
    """ConfigError should be usable as a standard exception."""
    error = ConfigError("test error")
    assert isinstance(error, Exception)
    assert str(error) == "test error"
