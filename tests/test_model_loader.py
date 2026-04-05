"""Tests for ONNX model loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wakeword_workbench.mining.model_loader import (
    ModelLoadError,
    load_onnx_model,
)


class MockIONode:
    """Mock I/O node for onnxruntime InferenceSession."""

    def __init__(self, name: str, shape: list[int]) -> None:
        self.name = name
        self.shape = shape


class MockInferenceSession:
    """Mock onnxruntime InferenceSession."""

    def __init__(
        self,
        input_name: str = "input",
        output_name: str = "output",
        input_shape: list[int] | tuple[int, ...] | None = None,
        output_shape: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if input_shape is None:
            input_shape = [1, 16000]
        if output_shape is None:
            output_shape = [1, 1]
        self._input = MockIONode(input_name, list(input_shape))
        self._output = MockIONode(output_name, list(output_shape))
        self._run_result: list[np.ndarray] = [np.array([[0.5]])]
        self.get_inputs = MagicMock(return_value=[self._input])
        self.get_outputs = MagicMock(return_value=[self._output])
        self.run = MagicMock(side_effect=self._run)

    def _run(self, output_names: list[str], feed_dict: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self._run_result

    def set_run_result(self, result: np.ndarray) -> None:
        self._run_result = [result]


class TestModelLoadError:
    """Tests for ModelLoadError exception."""

    def test_file_not_found(self) -> None:
        """Test that non-existent file raises ModelLoadError."""
        with pytest.raises(ModelLoadError, match="not found"):
            load_onnx_model(Path("nonexistent_model.onnx"))

    def test_path_is_directory(self, tmp_path: Path) -> None:
        """Test that passing a directory raises ModelLoadError."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()

        with pytest.raises(ModelLoadError, match="not a file"):
            load_onnx_model(model_dir)

    def test_unreadable_file(self, tmp_path: Path) -> None:
        """Test that unreadable file raises ModelLoadError."""
        model_path = tmp_path / "unreadable.onnx"
        model_path.write_text("not readable")

        with patch("pathlib.Path.read_bytes", side_effect=OSError("Permission denied")):
            with pytest.raises(ModelLoadError, match="Cannot read"):
                load_onnx_model(model_path)

    def test_missing_onnxruntime(self, tmp_path: Path) -> None:
        """Test that missing onnxruntime raises helpful error."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx data")

        with patch(
            "wakeword_workbench.mining.model_loader._check_onnxruntime_available",
            side_effect=ModelLoadError("onnxruntime is required"),
        ):
            with pytest.raises(ModelLoadError, match="onnxruntime is required"):
                load_onnx_model(model_path)

    def test_invalid_onnx_format(self, tmp_path: Path) -> None:
        """Test that invalid ONNX format raises ModelLoadError."""
        model_path = tmp_path / "invalid.onnx"
        model_path.write_bytes(b"not valid onnx data")

        mock_session = MockInferenceSession()
        mock_session.get_inputs = MagicMock(side_effect=Exception("Invalid ONNX format"))

        with patch(
            "wakeword_workbench.mining.model_loader._check_onnxruntime_available",
            return_value=MagicMock(InferenceSession=lambda *a, **kw: mock_session),
        ):
            with pytest.raises(ModelLoadError, match="Failed to get model I/O info"):
                load_onnx_model(model_path)


class TestLoadOnnxModel:
    """Tests for load_onnx_model function."""

    def test_successful_load(self, tmp_path: Path) -> None:
        """Test successful model loading returns callable."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx data")

        mock_session = MockInferenceSession()
        mock_session.set_run_result(np.array([[0.85]]))

        with patch(
            "wakeword_workbench.mining.model_loader._check_onnxruntime_available",
            return_value=MagicMock(InferenceSession=lambda *a, **kw: mock_session),
        ):
            model = load_onnx_model(model_path)
            assert callable(model)

    def test_model_callable_returns_float(self, tmp_path: Path) -> None:
        """Test that returned callable returns a float."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx data")

        mock_session = MockInferenceSession()
        mock_session.set_run_result(np.array([[0.75]]))

        with patch(
            "wakeword_workbench.mining.model_loader._check_onnxruntime_available",
            return_value=MagicMock(InferenceSession=lambda *a, **kw: mock_session),
        ):
            model = load_onnx_model(model_path)
            audio = np.random.randn(16000).astype(np.float32)
            result = model(audio)

            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_inference_with_1d_audio(self, tmp_path: Path) -> None:
        """Test inference with 1D audio array (common case)."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx data")

        mock_session = MockInferenceSession()
        mock_session.set_run_result(np.array([[0.5]]))

        with patch(
            "wakeword_workbench.mining.model_loader._check_onnxruntime_available",
            return_value=MagicMock(InferenceSession=lambda *a, **kw: mock_session),
        ):
            model = load_onnx_model(model_path)
            audio = np.random.randn(16000).astype(np.float32)
            model(audio)

            assert mock_session.run.called

    def test_inference_with_2d_audio(self, tmp_path: Path) -> None:
        """Test inference with 2D audio array (already batched)."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx data")

        mock_session = MockInferenceSession()
        mock_session.set_run_result(np.array([[0.6]]))

        with patch(
            "wakeword_workbench.mining.model_loader._check_onnxruntime_available",
            return_value=MagicMock(InferenceSession=lambda *a, **kw: mock_session),
        ):
            model = load_onnx_model(model_path)
            audio = np.random.randn(1, 16000).astype(np.float32)
            result = model(audio)

            assert isinstance(result, float)

    def test_dtype_conversion(self, tmp_path: Path) -> None:
        """Test that non-float32 audio is converted."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx data")

        mock_session = MockInferenceSession()
        mock_session.set_run_result(np.array([[0.4]]))

        with patch(
            "wakeword_workbench.mining.model_loader._check_onnxruntime_available",
            return_value=MagicMock(InferenceSession=lambda *a, **kw: mock_session),
        ):
            model = load_onnx_model(model_path)
            audio = np.random.randn(16000).astype(np.float64)
            model(audio)

            assert mock_session.run.called
            call_args = mock_session.run.call_args[0]
            feed_dict = call_args[1]
            fed_audio = feed_dict[mock_session._input.name]
            assert fed_audio.dtype == np.float32

    def test_multiple_outputs_mean(self, tmp_path: Path) -> None:
        """Test that multiple output values are averaged."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx data")

        mock_session = MockInferenceSession(output_shape=[1, 3])
        mock_session.set_run_result(np.array([[0.2, 0.5, 0.8]]))

        with patch(
            "wakeword_workbench.mining.model_loader._check_onnxruntime_available",
            return_value=MagicMock(InferenceSession=lambda *a, **kw: mock_session),
        ):
            model = load_onnx_model(model_path)
            audio = np.random.randn(16000).astype(np.float32)
            result = model(audio)

            expected = (0.2 + 0.5 + 0.8) / 3
            assert abs(result - expected) < 1e-6

    def test_inference_error_propagates(self, tmp_path: Path) -> None:
        """Test that inference errors are wrapped as ModelLoadError."""
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx data")

        mock_session = MockInferenceSession()
        mock_session.run = MagicMock(side_effect=RuntimeError("Inference failed"))

        with patch(
            "wakeword_workbench.mining.model_loader._check_onnxruntime_available",
            return_value=MagicMock(InferenceSession=lambda *a, **kw: mock_session),
        ):
            model = load_onnx_model(model_path)
            audio = np.random.randn(16000).astype(np.float32)

            with pytest.raises(ModelLoadError, match="Inference failed"):
                model(audio)
