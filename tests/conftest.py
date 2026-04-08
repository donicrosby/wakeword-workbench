"""Pytest configuration and fixtures."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pytest
import yaml

from wakeword_workbench.config import (
    AugmentationConfig,
    Config,
    OutputConfig,
    SamplesConfig,
    TTSConfig,
    TTSProviderConfig,
)


@pytest.fixture
def tmp_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test datasets."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    return dataset_dir


@pytest.fixture
def sample_config(tmp_dataset_dir: Path) -> Config:
    """Create a sample Config object for testing."""
    samples = SamplesConfig(positives=1000, negatives_multiplier=5)
    tts = TTSConfig(
        providers=[
            TTSProviderConfig(backend="kokoro", voices=["af_sarah"], speed=1.0),
        ]
    )
    augmentation = AugmentationConfig(
        noise_snr=[-10, 10],
        reverb_probability=0.5,
        gain_range=[-45, 0],
    )
    output = OutputConfig(path=str(tmp_dataset_dir / "output"), format=["microwakeword"])
    return Config(
        wake_word="hey_vera",
        samples=samples,
        tts=tts,
        augmentation=augmentation,
        output=output,
    )


@pytest.fixture
def mock_audio() -> np.ndarray:
    """Generate mock audio data as numpy array."""
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples)
    frequency = 440.0
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    noise = np.random.normal(0, 0.01, num_samples).astype(np.float32)
    return audio + noise


@pytest.fixture
def sample_manifest(tmp_dataset_dir: Path) -> list[dict[str, Any]]:
    """Create sample manifest entries for testing."""
    manifest = [
        {
            "file_path": str(tmp_dataset_dir / "positive_1.wav"),
            "label": "positive",
            "duration_ms": 1000,
        },
        {
            "file_path": str(tmp_dataset_dir / "positive_2.wav"),
            "label": "positive",
            "duration_ms": 1200,
        },
        {
            "file_path": str(tmp_dataset_dir / "negative_1.wav"),
            "label": "negative",
            "duration_ms": 800,
        },
        {
            "file_path": str(tmp_dataset_dir / "negative_2.wav"),
            "label": "negative",
            "duration_ms": 1500,
        },
    ]
    return manifest


VALID_CONFIG_YAML = dedent("""
wake_word: "hey_vera"
samples:
  positives: 1000
  negatives_multiplier: 5
tts:
  providers:
    - backend: kokoro
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


@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    """Create a valid YAML config file for testing."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID_CONFIG_YAML)
    return config_path


@pytest.fixture
def invalid_config_file(tmp_path: Path) -> Path:
    """Create an invalid YAML config file for testing."""
    config_data = {
        "dataset": {
            "sample_rate": 99999,
        },
    }
    config_path = tmp_path / "invalid_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return config_path
