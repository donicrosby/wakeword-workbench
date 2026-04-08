"""Configuration management for WakeWord Workbench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

SUPPORTED_OUTPUT_FORMATS = {"microwakeword", "openwakeword"}


class ConfigError(Exception):
    """Raised when configuration validation fails."""

    pass


@dataclass
class SamplesConfig:
    """Samples configuration."""

    positives: int
    negatives_multiplier: int

    def __post_init__(self) -> None:
        if self.positives <= 0:
            raise ConfigError("positives must be positive")
        if self.negatives_multiplier <= 0:
            raise ConfigError("negatives_multiplier must be positive")


@dataclass
class TTSProviderConfig:
    """TTS provider configuration."""

    backend: str
    voices: list[str]
    speed: float = 1.0

    def __post_init__(self) -> None:
        if not self.backend:
            raise ConfigError("backend cannot be empty")
        if not self.voices:
            raise ConfigError("voices cannot be empty")
        if not 0 < self.speed <= 3:
            raise ConfigError("speed must be between 0 and 3")


@dataclass
class TTSConfig:
    """TTS configuration with multiple providers support."""

    providers: list[TTSProviderConfig]

    def __post_init__(self) -> None:
        if not self.providers:
            raise ConfigError("providers cannot be empty")
        for provider in self.providers:
            if not provider.backend:
                raise ConfigError("backend cannot be empty")
            if not provider.voices:
                raise ConfigError("voices cannot be empty")
            if not 0 < provider.speed <= 3:
                raise ConfigError("speed must be between 0 and 3")

    def get_all_voices(self) -> list[tuple[str, str]]:
        """Get all voices as (backend, voice) tuples.

        Returns:
            List of tuples where each tuple is (backend_name, voice_id).
        """
        voices: list[tuple[str, str]] = []
        for provider in self.providers:
            for voice in provider.voices:
                voices.append((provider.backend, voice))
        return voices


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""

    noise_snr: list[float]
    reverb_probability: float
    gain_range: list[float]

    def __post_init__(self) -> None:
        if len(self.noise_snr) != 2:
            raise ConfigError("noise_snr must have exactly 2 values")
        if self.noise_snr[0] > self.noise_snr[1]:
            raise ConfigError("noise_snr[0] must be <= noise_snr[1]")
        if not 0 <= self.reverb_probability <= 1:
            raise ConfigError("reverb_probability must be between 0 and 1")
        if len(self.gain_range) != 2:
            raise ConfigError("gain_range must have exactly 2 values")
        if self.gain_range[0] > self.gain_range[1]:
            raise ConfigError("gain_range[0] must be <= gain_range[1]")


@dataclass
class OutputConfig:
    """Output configuration."""

    path: str
    format: list[str]

    def __post_init__(self) -> None:
        if not self.path:
            raise ConfigError("path cannot be empty")
        if not self.format:
            raise ConfigError("format cannot be empty")
        for fmt in self.format:
            if fmt not in SUPPORTED_OUTPUT_FORMATS:
                raise ConfigError(f"output.format contains invalid format: {fmt}")


@dataclass
class Config:
    """Main configuration."""

    wake_word: str
    samples: SamplesConfig
    tts: TTSConfig
    augmentation: AugmentationConfig
    output: OutputConfig


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML: {e}") from e

    if data is None:
        raise ConfigError("Config file is empty")

    if not isinstance(data, dict):
        raise ConfigError("Failed to parse YAML: expected dictionary")

    if "wake_word" not in data:
        raise ConfigError("Missing required field: wake_word")
    if not data["wake_word"]:
        raise ConfigError("wake_word cannot be empty")

    if "samples" not in data:
        raise ConfigError("Missing required field: samples")
    samples_data = data["samples"]
    samples = SamplesConfig(
        positives=samples_data["positives"],
        negatives_multiplier=samples_data["negatives_multiplier"],
    )

    if "tts" not in data:
        raise ConfigError("Missing required field: tts")
    tts_data = data["tts"]

    if "providers" not in tts_data:
        raise ConfigError("Missing required field: tts.providers")

    providers_list = tts_data.get("providers", [])
    if not isinstance(providers_list, list):
        raise ConfigError("tts.providers must be a list")

    providers: list[TTSProviderConfig] = []
    for provider_data in providers_list:
        if not isinstance(provider_data, dict):
            raise ConfigError("Each provider must be a dictionary")
        provider = TTSProviderConfig(
            backend=provider_data["backend"],
            voices=provider_data["voices"],
            speed=provider_data.get("speed", 1.0),
        )
        providers.append(provider)

    tts = TTSConfig(providers=providers)

    if "augmentation" not in data:
        raise ConfigError("Missing required field: augmentation")
    aug_data = data["augmentation"]
    augmentation = AugmentationConfig(
        noise_snr=aug_data["noise_snr"],
        reverb_probability=aug_data["reverb_probability"],
        gain_range=aug_data["gain_range"],
    )

    if "output" not in data:
        raise ConfigError("Missing required field: output")
    output_data = data["output"]
    output = OutputConfig(
        path=output_data["path"],
        format=output_data["format"],
    )

    return Config(
        wake_word=data["wake_word"],
        samples=samples,
        tts=tts,
        augmentation=augmentation,
        output=output,
    )
