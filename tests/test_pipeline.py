"""Tests for augmentation pipeline module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from wakeword_workbench.augment.gain import AdjustGain, SoftClip
from wakeword_workbench.augment.pipeline import (
    Compose,
    Transform,
    default_pipeline,
    from_config,
    heavy_pipeline,
    minimal_pipeline,
)


class DummyTransform:
    """Dummy transform for testing."""

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return audio * self.factor


class NoOpTransform:
    """Transform that doesn't modify audio (for no-mutation tests)."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return audio.copy()


class TestTransformProtocol:
    """Tests for Transform protocol."""

    def test_adjust_gain_implements_protocol(self, mock_audio: np.ndarray) -> None:
        """AdjustGain should satisfy the Transform protocol."""
        transform = AdjustGain(gain_range=(-10, 0))
        assert isinstance(transform, Transform)
        result = transform.apply(mock_audio, sr=16000)
        assert isinstance(result, np.ndarray)

    def test_dummy_transform_implements_protocol(self, mock_audio: np.ndarray) -> None:
        """DummyTransform should satisfy the Transform protocol."""
        transform = DummyTransform(factor=2.0)
        assert isinstance(transform, Transform)
        result = transform.apply(mock_audio, sr=16000)
        assert isinstance(result, np.ndarray)


class TestCompose:
    """Tests for Compose class."""

    def test_compose_with_valid_transforms(self) -> None:
        """Compose should accept valid transforms."""
        pipeline = Compose([DummyTransform(), DummyTransform()])
        assert len(pipeline) == 2

    def test_compose_rejects_invalid_transform(self) -> None:
        """Compose should reject transforms that don't implement protocol."""

        class NotATransform:
            pass

        with pytest.raises(TypeError, match="does not implement the Transform protocol"):
            Compose([NotATransform()])

    def test_compose_rejects_invalid_transform_at_index(self) -> None:
        """Compose should report the correct index for invalid transforms."""

        class BadTransform:
            pass

        with pytest.raises(TypeError, match="index 1"):
            Compose([DummyTransform(), BadTransform(), DummyTransform()])

    def test_apply_passes_audio_through_transforms(self, mock_audio: np.ndarray) -> None:
        """Compose.apply should pass audio through each transform in order."""
        pipeline = Compose([DummyTransform(factor=2.0), DummyTransform(factor=3.0)])
        result = pipeline.apply(mock_audio, sr=16000)
        expected = mock_audio * 2.0 * 3.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_apply_returns_new_array(self, mock_audio: np.ndarray) -> None:
        """Compose.apply should not mutate the input."""
        pipeline = Compose([DummyTransform(factor=2.0)])
        original = mock_audio.copy()
        pipeline.apply(mock_audio, sr=16000)
        np.testing.assert_array_equal(mock_audio, original)

    def test_apply_empty_pipeline(self, mock_audio: np.ndarray) -> None:
        """Compose with no transforms should return copy of audio."""
        pipeline = Compose([])
        result = pipeline.apply(mock_audio, sr=16000)
        np.testing.assert_array_equal(result, mock_audio)

    def test_repr(self) -> None:
        """Compose should have a useful string representation."""
        pipeline = Compose([DummyTransform(), AdjustGain()])
        repr_str = repr(pipeline)
        assert "Compose" in repr_str
        assert "DummyTransform" in repr_str
        assert "AdjustGain" in repr_str

    def test_len(self) -> None:
        """Compose.__len__ should return number of transforms."""
        pipeline = Compose([DummyTransform(), DummyTransform(), DummyTransform()])
        assert len(pipeline) == 3

    def test_getitem(self) -> None:
        """Compose.__getitem__ should allow indexing."""
        pipeline = Compose([DummyTransform(1.0), DummyTransform(2.0), DummyTransform(3.0)])
        assert pipeline[0].factor == 1.0
        assert pipeline[1].factor == 2.0
        assert pipeline[2].factor == 3.0


class TestFromConfig:
    """Tests for from_config function."""

    def test_from_config_dict(self) -> None:
        """from_config should accept a dict configuration."""
        config = {
            "transforms": [
                {"type": "AdjustGain", "kwargs": {"gain_range": [-10, 0]}},
            ]
        }
        pipeline = from_config(config)
        assert len(pipeline) == 1
        assert isinstance(pipeline[0], AdjustGain)

    def test_from_config_yaml_file(self, tmp_path: Path, mock_audio: np.ndarray) -> None:
        """from_config should load from YAML files."""
        config = {
            "transforms": [
                {"type": "AdjustGain", "kwargs": {"gain_range": [-10, 0]}},
            ]
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        pipeline = from_config(config_file)
        assert len(pipeline) == 1
        result = pipeline.apply(mock_audio, sr=16000)
        assert isinstance(result, np.ndarray)

    def test_from_config_json_file(self, tmp_path: Path) -> None:
        """from_config should load from JSON files."""
        config = {
            "transforms": [
                {"type": "AdjustGain", "kwargs": {"gain_range": [-10, 0]}},
            ]
        }
        config_file = tmp_path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f)

        pipeline = from_config(config_file)
        assert len(pipeline) == 1

    def test_from_config_file_not_found(self) -> None:
        """from_config should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            from_config(Path("/nonexistent/config.yaml"))

    def test_from_config_invalid_format(self, tmp_path: Path) -> None:
        """from_config should raise ValueError for unsupported formats."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported config format"):
            from_config(config_file)

    def test_from_config_missing_type(self) -> None:
        """from_config should raise ValueError for missing 'type' key."""
        config = {"transforms": [{"kwargs": {}}]}
        with pytest.raises(ValueError, match="missing 'type'"):
            from_config(config)

    def test_from_config_unknown_transform(self) -> None:
        """from_config should raise ValueError for unknown transform types."""
        config = {
            "transforms": [
                {"type": "UnknownTransform", "kwargs": {}},
            ]
        }
        with pytest.raises(ValueError, match="Unknown transform"):
            from_config(config)

    def test_from_config_invalid_kwargs(self) -> None:
        """from_config should raise ValueError for invalid kwargs."""
        config = {
            "transforms": [
                {"type": "AdjustGain", "kwargs": {"invalid_param": 42}},
            ]
        }
        with pytest.raises(ValueError, match="Failed to instantiate"):
            from_config(config)

    def test_from_config_multiple_transforms(self) -> None:
        """from_config should handle multiple transforms."""
        config = {
            "transforms": [
                {"type": "AdjustGain", "kwargs": {"gain_range": [-20, 0]}},
                {"type": "SoftClip", "kwargs": {"threshold": 0.9}},
            ]
        }
        pipeline = from_config(config)
        assert len(pipeline) == 2
        assert isinstance(pipeline[0], AdjustGain)
        assert isinstance(pipeline[1], SoftClip)


class TestPresets:
    """Tests for preset pipeline functions."""

    def test_minimal_pipeline(self, mock_audio: np.ndarray) -> None:
        """minimal_pipeline should create a pipeline with only gain."""
        pipeline = minimal_pipeline()
        assert len(pipeline) >= 1
        result = pipeline.apply(mock_audio, sr=16000)
        assert isinstance(result, np.ndarray)

    def test_default_pipeline(self, mock_audio: np.ndarray) -> None:
        """default_pipeline should create a pipeline with gain and noise."""
        pipeline = default_pipeline()
        assert len(pipeline) >= 2
        result = pipeline.apply(mock_audio, sr=16000)
        assert isinstance(result, np.ndarray)

    def test_heavy_pipeline(self, mock_audio: np.ndarray) -> None:
        """heavy_pipeline should create a pipeline with multiple noise types."""
        pipeline = heavy_pipeline()
        assert len(pipeline) >= 4
        result = pipeline.apply(mock_audio, sr=16000)
        assert isinstance(result, np.ndarray)

    def test_presets_produce_different_results(self, mock_audio: np.ndarray) -> None:
        """Different presets should produce different augmentation results."""
        minimal = minimal_pipeline()
        heavy = heavy_pipeline()

        # Run multiple times to account for random augmentation
        minimal_results = [minimal.apply(mock_audio, sr=16000) for _ in range(5)]
        heavy_results = [heavy.apply(mock_audio, sr=16000) for _ in range(5)]

        # Heavy should generally produce more variation than minimal
        minimal_variances = [np.var(r - mock_audio) for r in minimal_results]
        heavy_variances = [np.var(r - mock_audio) for r in heavy_results]

        # At least some heavy results should have higher variance
        assert any(h > m for h in heavy_variances for m in minimal_variances)


class TestNoMutation:
    """Tests to verify input arrays are not mutated."""

    def test_compose_does_not_mutate_input(self, mock_audio: np.ndarray) -> None:
        """Compose.apply should never mutate the input array."""
        original = mock_audio.copy()
        pipeline = Compose([AdjustGain(gain_range=(-10, 0)), NoOpTransform()])

        # Apply multiple times
        for _ in range(10):
            pipeline.apply(mock_audio, sr=16000)

        np.testing.assert_array_equal(mock_audio, original)

    def test_individual_transforms_dont_mutate(self, mock_audio: np.ndarray) -> None:
        """Individual transforms should not mutate input when returning copies."""
        original = mock_audio.copy()
        pipeline = Compose([NoOpTransform(), NoOpTransform()])

        pipeline.apply(mock_audio, sr=16000)

        np.testing.assert_array_equal(mock_audio, original)
