"""Configuration management for WakeWord Workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

yaml = import_module("yaml")

SUPPORTED_OUTPUT_FORMATS = {"microwakeword", "openwakeword"}
SUPPORTED_TTS_ACCELERATION = {"cpu", "cuda", "openvino"}


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
    acceleration: str = "cpu"
    device: str | None = None
    model_path: str | None = None

    def __post_init__(self) -> None:
        if not self.backend:
            raise ConfigError("backend cannot be empty")
        if not self.voices:
            raise ConfigError("voices cannot be empty")
        if not 0 < self.speed <= 3:
            raise ConfigError("speed must be between 0 and 3")

        self.acceleration = self.acceleration.strip().lower()
        if self.acceleration == "migraphx":
            raise ConfigError("MIGraphX acceleration is not supported for TTS providers")
        if self.acceleration not in SUPPORTED_TTS_ACCELERATION:
            supported = ", ".join(sorted(SUPPORTED_TTS_ACCELERATION))
            raise ConfigError(
                f"acceleration must be one of: {supported}; got {self.acceleration!r}"
            )

        if self.device is not None:
            self.device = self.device.strip()
            if not self.device:
                raise ConfigError("device cannot be empty when provided")

        if self.model_path is not None:
            self.model_path = self.model_path.strip()
            if not self.model_path:
                raise ConfigError("model_path cannot be empty when provided")

        backend_name = self.backend.strip().lower()
        if backend_name == "piper" and self.acceleration == "openvino":
            raise ConfigError("Piper supports CPU and CUDA acceleration, but not OpenVINO")

    def runtime_options(self) -> dict[str, str]:
        """Return normalized runtime options for backend creation and caching."""
        options = {"acceleration": self.acceleration}
        if self.device is not None:
            options["device"] = self.device
        if self.model_path is not None:
            options["model_path"] = self.model_path
        return options

    def backend_cache_key(self) -> tuple[str, tuple[str, ...], float, str, str | None, str | None]:
        """Return a stable cache key for backend instance reuse."""
        return (
            self.backend,
            tuple(self.voices),
            self.speed,
            self.acceleration,
            self.device,
            self.model_path,
        )

    def backend_instance_key(self) -> tuple[str, float, str, str | None, str | None]:
        """Return a key for backend instance reuse across different voice selections.

        This key excludes the `voices` field, allowing BackendPool to reuse
        the same backend instance for different voice selections.

        Returns:
            Tuple of (backend, speed, acceleration, device, model_path).
        """
        return (
            self.backend,
            self.speed,
            self.acceleration,
            self.device,
            self.model_path,
        )


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
class NegativeConfusionConfig:
    """Confusion-phrase negative generation settings."""

    enabled: bool = True
    weight: float = 0.6
    min_similarity: float = 0.6

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ConfigError("confusion weight must be non-negative")
        if not 0 <= self.min_similarity <= 1:
            raise ConfigError("confusion min_similarity must be between 0 and 1")


@dataclass
class NegativeSyntheticConfig:
    """Synthetic negative generation settings."""

    enabled: bool = True
    weight: float = 0.4
    strategy: str = "random"
    min_word_count: int = 2
    max_word_count: int = 4
    topics: list[str] | None = None
    word_list: list[str] | None = None

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ConfigError("synthetic weight must be non-negative")
        if self.strategy not in {"random", "sentence", "topic"}:
            raise ConfigError("synthetic strategy must be one of: random, sentence, topic")
        if self.min_word_count < 1:
            raise ConfigError("synthetic min_word_count must be at least 1")
        if self.max_word_count < self.min_word_count:
            raise ConfigError(
                "synthetic max_word_count must be greater than or equal to min_word_count"
            )
        if self.topics is not None and len(self.topics) == 0:
            raise ConfigError("synthetic topics cannot be empty when provided")
        if self.word_list is not None and len(self.word_list) == 0:
            raise ConfigError("synthetic word_list cannot be empty when provided")


@dataclass
class NegativeGenerationConfig:
    """Negative generation configuration."""

    confusion: NegativeConfusionConfig = field(default_factory=NegativeConfusionConfig)
    synthetic: NegativeSyntheticConfig = field(default_factory=NegativeSyntheticConfig)
    custom_phrases: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.confusion.enabled and not self.synthetic.enabled:
            raise ConfigError("at least one negative source must be enabled")

        enabled_weight = 0.0
        if self.confusion.enabled:
            enabled_weight += self.confusion.weight
        if self.synthetic.enabled:
            enabled_weight += self.synthetic.weight

        if enabled_weight <= 0:
            raise ConfigError("enabled negative sources must have a positive total weight")

        if self.custom_phrases is not None:
            if len(self.custom_phrases) == 0:
                raise ConfigError("custom_phrases cannot be empty when provided")
            normalized_custom_phrases: list[str] = []
            seen_phrases: set[str] = set()
            for phrase in self.custom_phrases:
                cleaned = phrase.strip()
                if not cleaned:
                    raise ConfigError("custom_phrases cannot contain empty values")
                normalized = cleaned.lower()
                if normalized in seen_phrases:
                    continue
                seen_phrases.add(normalized)
                normalized_custom_phrases.append(cleaned)
            self.custom_phrases = normalized_custom_phrases


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
    wake_word_variants: list[str] | None = None
    negatives: NegativeGenerationConfig = field(default_factory=NegativeGenerationConfig)

    def __post_init__(self) -> None:
        if self.wake_word_variants is not None:
            if len(self.wake_word_variants) == 0:
                raise ConfigError("wake_word_variants cannot be empty when provided")

            normalized_variants: list[str] = []
            seen_variants: set[str] = set()
            for variant in self.wake_word_variants:
                cleaned = " ".join(variant.split())
                if not cleaned:
                    raise ConfigError("wake_word_variants cannot contain empty values")
                normalized = cleaned.lower()
                if normalized in seen_variants:
                    continue
                seen_variants.add(normalized)
                normalized_variants.append(cleaned)
            self.wake_word_variants = normalized_variants


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
            acceleration=provider_data.get("acceleration", "cpu"),
            device=provider_data.get("device"),
            model_path=provider_data.get("model_path"),
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

    negatives_data = data.get("negatives", {})
    confusion_data = negatives_data.get("confusion", {})
    synthetic_data = negatives_data.get("synthetic", {})
    negatives = NegativeGenerationConfig(
        confusion=NegativeConfusionConfig(
            enabled=confusion_data.get("enabled", True),
            weight=confusion_data.get("weight", 0.6),
            min_similarity=confusion_data.get("min_similarity", 0.6),
        ),
        synthetic=NegativeSyntheticConfig(
            enabled=synthetic_data.get("enabled", True),
            weight=synthetic_data.get("weight", 0.4),
            strategy=synthetic_data.get("strategy", "random"),
            min_word_count=synthetic_data.get("min_word_count", 2),
            max_word_count=synthetic_data.get("max_word_count", 4),
            topics=synthetic_data.get("topics"),
            word_list=synthetic_data.get("word_list"),
        ),
        custom_phrases=negatives_data.get("custom_phrases"),
    )

    return Config(
        wake_word=data["wake_word"],
        wake_word_variants=data.get("wake_word_variants"),
        samples=samples,
        tts=tts,
        augmentation=augmentation,
        output=output,
        negatives=negatives,
    )
