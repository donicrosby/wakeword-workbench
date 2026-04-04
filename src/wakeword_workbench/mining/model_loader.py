"""ONNX model loader for wake word detection inference.

This module provides utilities for loading ONNX models and wrapping them
in callable interfaces suitable for wake word detection tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from wakeword_workbench.logging_config import get_logger

log = get_logger(__name__)


class ModelLoadError(Exception):
    """Raised when ONNX model loading fails.

    This exception is raised for various model loading failures including:
    - File not found or unreadable
    - Invalid ONNX format
    - Missing onnxruntime dependency
    - Session creation errors
    """

    pass


def _check_onnxruntime_available() -> Any:
    """Check if onnxruntime is available and return it.

    Returns:
        The onnxruntime module.

    Raises:
        ModelLoadError: If onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort  # noqa: F401

        return ort
    except ImportError:
        raise ModelLoadError(
            "onnxruntime is required for model inference. Install it with: uv add onnxruntime"
        ) from None


def load_onnx_model(path: Path) -> Callable[[np.ndarray], float]:
    """Load an ONNX model and return a callable for inference.

    This function loads an ONNX model from the specified path and returns
    a callable that accepts audio samples as a numpy array and returns
    a confidence score.

    Args:
        path: Path to the ONNX model file.

    Returns:
        A callable that takes a numpy array of audio samples and returns
        a float confidence score between 0.0 and 1.0.

    Raises:
        ModelLoadError: If the model cannot be loaded, the file doesn't exist,
            or onnxruntime is not installed.

    Example:
        >>> model = load_onnx_model(Path("model.onnx"))
        >>> audio = np.random.randn(16000).astype(np.float32)
        >>> score = model(audio)
        >>> print(f"Confidence: {score:.4f}")
    """
    path = Path(path)

    # Check if file exists
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")

    # Check if file is readable
    if not path.is_file():
        raise ModelLoadError(f"Path is not a file: {path}")

    try:
        path.read_bytes()
    except OSError as e:
        raise ModelLoadError(f"Cannot read model file: {path}. Error: {e}") from e

    # Import onnxruntime (with graceful error handling)
    ort = _check_onnxruntime_available()

    log.debug("loading_onnx_model", path=str(path))

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as e:
        raise ModelLoadError(f"Failed to create inference session: {e}") from e

    # Get input and output names
    try:
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        if len(inputs) == 0:
            raise ModelLoadError("Model has no input tensors")

        if len(outputs) == 0:
            raise ModelLoadError("Model has no output tensors")

        input_name = inputs[0].name
        output_name = outputs[0].name

        log.debug(
            "onnx_model_loaded",
            path=str(path),
            input_shape=inputs[0].shape,
            output_shape=outputs[0].shape,
        )
    except Exception as e:
        raise ModelLoadError(f"Failed to get model I/O info: {e}") from e

    def _inference_wrapper(audio: np.ndarray) -> float:
        """Run inference on audio samples.

        Args:
            audio: Numpy array of audio samples.

        Returns:
            Confidence score as float.

        Raises:
            ModelLoadError: If inference fails.
        """
        try:
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Add batch dimension if needed (model expects [batch, samples])
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            # Run inference
            result = session.run([output_name], {input_name: audio})

            # Extract scalar score from output
            # Most wake word models output a single value per sample window
            # We take the mean across the batch dimension if present
            scores = result[0]
            if scores.ndim > 1:
                scores = scores.flatten()

            return float(scores.mean())

        except Exception as e:
            raise ModelLoadError(f"Inference failed: {e}") from e

    return _inference_wrapper
