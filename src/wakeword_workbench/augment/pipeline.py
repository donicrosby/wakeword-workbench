"""Augmentation pipeline orchestration.

Provides Compose class for chaining audio transforms, a Transform protocol,
config-based pipeline loading, and common preset pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from wakeword_workbench.logging_config import get_logger

log = get_logger(__name__)


@runtime_checkable
class Transform(Protocol):
    """Protocol for audio augmentation transforms.

    All transforms must implement the apply(audio, sr) method that returns
    a new audio array. Optionally, transforms may have a `p` property for
    application probability.
    """

    def apply(self, audio: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        """Apply transform to audio.

        Args:
            audio: Input audio as float32 array.
            sr: Sample rate in Hz.

        Returns:
            Transformed audio (may be new array, never mutates input).
        """
        ...


class Compose:
    """Compose multiple audio transforms into a sequential pipeline.

    Applies transforms in order, passing output of each to the next.
    Each transform is called with its own probability check if it has a `p` attribute.

    Args:
        transforms: List of transform objects that implement the Transform protocol.

    Example:
        >>> from wakeword_workbench.augment.gain import AdjustGain
        >>> from wakeword_workbench.augment.noise import AddNoise
        >>> pipeline = Compose([
        ...     AdjustGain(gain_range=(-45, 0)),
        ...     AddNoise(noise_dir, snr_range=(-10, 10), p=0.75),
        ... ])
        >>> audio_aug = pipeline.apply(audio, sr=16000)
    """

    def __init__(self, transforms: list[Any]) -> None:
        """Initialize pipeline with transforms.

        Args:
            transforms: List of transform objects.

        Raises:
            TypeError: If a transform doesn't implement the Transform protocol.
        """
        for i, transform in enumerate(transforms):
            if not isinstance(transform, Transform):
                raise TypeError(
                    f"Transform at index {i} ({type(transform).__name__}) "
                    "does not implement the Transform protocol. "
                    "Transforms must have apply(audio, sr) -> audio method."
                )
        self.transforms = transforms

    def apply(self, audio: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        """Apply all transforms in sequence.

        Args:
            audio: Input audio as float32 array.
            sr: Sample rate in Hz.

        Returns:
            Audio after all transforms applied (new array, no mutation).
        """
        result = np.asarray(audio, dtype=np.float32)
        for transform in self.transforms:
            result = transform.apply(result, sr)
        return result

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        transform_names = [type(t).__name__ for t in self.transforms]
        return f"Compose([{', '.join(transform_names)}])"

    def __len__(self) -> int:
        """Number of transforms in the pipeline."""
        return len(self.transforms)

    def __getitem__(self, index: int) -> Any:
        """Get transform by index."""
        return self.transforms[index]


# Registry of transform classes for config-based loading
_TRANSFORM_REGISTRY: dict[str, type] = {}


def _register_transform(cls: type) -> type:
    """Decorator to register a transform class in the registry.

    Usage:
        @_register_transform
        class MyTransform:
            def apply(self, audio, sr):
                ...
    """
    _TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls


def _get_transform_class(name: str) -> type:
    """Get transform class from registry by name.

    Args:
        name: Transform class name.

    Returns:
        Transform class.

    Raises:
        ValueError: If transform name is not registered.
    """
    if name not in _TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform: {name!r}. Available transforms: {list(_TRANSFORM_REGISTRY.keys())}"
        )
    return _TRANSFORM_REGISTRY[name]


def from_config(config: dict[str, Any] | str | Path) -> Compose:
    """Create a Compose pipeline from a configuration dict or file.

    Config format:
        {
            "transforms": [
                {"type": "AdjustGain", "kwargs": {"gain_range": [-45, 0]}},
                {"type": "AddColoredNoise", "kwargs": {"color": "pink", "p": 0.5}},
            ]
        }

    Args:
        config: Configuration dict, YAML file path, or JSON file path.

    Returns:
        Compose pipeline instance.

    Raises:
        ValueError: If config format is invalid.
        FileNotFoundError: If config file path doesn't exist.
    """
    # Load config from file if string/path
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        suffix = config_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            import yaml

            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        elif suffix == ".json":
            import json

            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")

        log.debug("pipeline_config_loaded", path=str(config_path))

    if not isinstance(config, dict):
        raise ValueError("Config must be a dict")

    transforms_config = config.get("transforms", [])
    if not isinstance(transforms_config, list):
        raise ValueError("Config 'transforms' must be a list")

    transforms: list[Any] = []
    for i, transform_cfg in enumerate(transforms_config):
        if not isinstance(transform_cfg, dict):
            raise ValueError(f"Transform at index {i} must be a dict with 'type' key")

        transform_type = transform_cfg.get("type")
        if not transform_type:
            raise ValueError(f"Transform at index {i} missing 'type' key")

        kwargs = transform_cfg.get("kwargs", {})

        try:
            transform_cls = _get_transform_class(transform_type)
        except ValueError:
            # Try to import from wakeword_workbench.augment
            try:
                import importlib

                module = importlib.import_module("wakeword_workbench.augment")
                if hasattr(module, transform_type):
                    transform_cls = getattr(module, transform_type)
                    _TRANSFORM_REGISTRY[transform_type] = transform_cls
                else:
                    raise
            except (ImportError, AttributeError) as err:
                raise ValueError(
                    f"Unknown transform type: {transform_type!r} at index {i}. "
                    f"Available: {list(_TRANSFORM_REGISTRY.keys())}"
                ) from err

        try:
            transforms.append(transform_cls(**kwargs))
        except TypeError as e:
            raise ValueError(f"Failed to instantiate {transform_type}: {e}") from e

    log.info(
        "pipeline_created",
        transform_count=len(transforms),
        transforms=[type(t).__name__ for t in transforms],
    )

    return Compose(transforms)


def _collect_presets() -> dict[str, tuple[str, dict[str, Any]]]:
    """Collect preset configurations.

    Returns:
        Dict mapping preset name to (description, config).
    """
    return {
        "minimal": (
            "Light augmentation: gain only",
            {
                "transforms": [
                    {
                        "type": "AdjustGain",
                        "kwargs": {"gain_range": [-20, 0], "p": 1.0},
                    },
                ]
            },
        ),
        "default": (
            "Standard augmentation: gain + colored noise",
            {
                "transforms": [
                    {
                        "type": "AdjustGain",
                        "kwargs": {"gain_range": [-45, 0], "p": 1.0},
                    },
                    {
                        "type": "AddColoredNoise",
                        "kwargs": {"color": "pink", "snr_range": (-5, 15), "p": 0.5},
                    },
                ]
            },
        ),
        "heavy": (
            "Aggressive augmentation: gain + multiple noise types",
            {
                "transforms": [
                    {
                        "type": "AdjustGain",
                        "kwargs": {"gain_range": [-45, 0], "p": 1.0},
                    },
                    {
                        "type": "AddColoredNoise",
                        "kwargs": {"color": "pink", "snr_range": (-10, 10), "p": 0.3},
                    },
                    {
                        "type": "AddColoredNoise",
                        "kwargs": {"color": "brown", "snr_range": (-15, 5), "p": 0.3},
                    },
                    {
                        "type": "AddColoredNoise",
                        "kwargs": {"color": "white", "snr_range": (-20, 0), "p": 0.2},
                    },
                ]
            },
        ),
    }


def minimal_pipeline() -> Compose:
    """Create a minimal augmentation pipeline.

    Returns:
        Pipeline with only gain adjustment for light augmentation.
    """
    from wakeword_workbench.augment.gain import AdjustGain

    _TRANSFORM_REGISTRY["AdjustGain"] = AdjustGain

    return from_config(_collect_presets()["minimal"][1])


def default_pipeline() -> Compose:
    """Create a standard augmentation pipeline.

    Returns:
        Pipeline with gain and colored noise for balanced augmentation.
    """
    from wakeword_workbench.augment.gain import AdjustGain
    from wakeword_workbench.augment.noise import AddColoredNoise

    _TRANSFORM_REGISTRY["AdjustGain"] = AdjustGain
    _TRANSFORM_REGISTRY["AddColoredNoise"] = AddColoredNoise

    return from_config(_collect_presets()["default"][1])


def heavy_pipeline() -> Compose:
    """Create an aggressive augmentation pipeline.

    Returns:
        Pipeline with gain and multiple noise types for heavy augmentation.
    """
    from wakeword_workbench.augment.gain import AdjustGain
    from wakeword_workbench.augment.noise import AddColoredNoise

    _TRANSFORM_REGISTRY["AdjustGain"] = AdjustGain
    _TRANSFORM_REGISTRY["AddColoredNoise"] = AddColoredNoise

    return from_config(_collect_presets()["heavy"][1])


# Auto-register built-in transforms
def _auto_register() -> None:
    """Auto-register built-in transforms."""
    try:
        from wakeword_workbench.augment.gain import AdjustGain, GainTransition, HardClip, SoftClip

        _TRANSFORM_REGISTRY.update(
            {
                "AdjustGain": AdjustGain,
                "GainTransition": GainTransition,
                "HardClip": HardClip,
                "SoftClip": SoftClip,
            }
        )
    except ImportError:
        pass

    try:
        from wakeword_workbench.augment.noise import AddColoredNoise, AddNoise

        _TRANSFORM_REGISTRY.update(
            {
                "AddColoredNoise": AddColoredNoise,
                "AddNoise": AddNoise,
            }
        )
    except ImportError:
        pass

    try:
        from wakeword_workbench.augment.reverb import AddReverb

        _TRANSFORM_REGISTRY["AddReverb"] = AddReverb
    except ImportError:
        pass

    try:
        from wakeword_workbench.augment.padding import FixedSizeClip, TrimSilence

        _TRANSFORM_REGISTRY.update(
            {
                "FixedSizeClip": FixedSizeClip,
                "TrimSilence": TrimSilence,
            }
        )
    except ImportError:
        pass


_auto_register()
