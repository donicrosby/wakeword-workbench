"""Model format validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ValidationReport:
    """Report from validating an export directory.

    Attributes:
        valid: True if the export passed all validation checks.
        errors: List of critical errors that make the export invalid.
        warnings: List of non-critical issues found during validation.
        stats: Dictionary of statistics about the validated data.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate_microwakeword(export_dir: Path, split: str = "train") -> ValidationReport:
    """Validate a MicroWakeWord format export directory.

    Checks for:
    - Required files exist (train_data.mmap or train_data.npy, train_labels.npy, train_indices.npy)
    - Shapes match between files
    - Data integrity (no NaN/Inf values)
    - Valid array dtypes
    - Reasonable file sizes
    - Valid indices (within bounds, proper dtype)

    Args:
        export_dir: Path to the export directory to validate.
        split: Split name used during export (default: "train").

    Returns:
        ValidationReport with validation results.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    export_dir = Path(export_dir)

    # Check for mmap format (raw audio) or feature format (mel-spectrogram)
    data_mmap = export_dir / f"{split}_data.mmap"
    data_npy = export_dir / f"{split}_data.npy"
    indices_path = export_dir / f"{split}_indices.npy"
    labels_path = export_dir / f"{split}_labels.npy"

    # Determine which data format is present
    has_mmap = data_mmap.exists()
    has_data_npy = data_npy.exists()

    # Check if data file exists and is not empty
    mmap_has_content = has_mmap and data_mmap.stat().st_size > 0
    npy_has_content = has_data_npy and data_npy.stat().st_size > 0

    if not has_mmap and not has_data_npy:
        errors.append(f"Missing required file: {split}_data.mmap or {split}_data.npy")
    elif has_mmap and not mmap_has_content:
        errors.append(f"Empty file: {split}_data.mmap")
    elif has_data_npy and not npy_has_content:
        errors.append(f"Empty file: {split}_data.npy")

    if not indices_path.exists():
        errors.append(f"Missing required file: {split}_indices.npy")
    elif indices_path.stat().st_size == 0:
        errors.append(f"Empty file: {split}_indices.npy")

    if not labels_path.exists():
        errors.append(f"Missing required file: {split}_labels.npy")
    elif labels_path.stat().st_size == 0:
        errors.append(f"Empty file: {split}_labels.npy")

    if errors:
        return ValidationReport(valid=False, errors=errors, warnings=warnings, stats=stats)

    # Load arrays
    try:
        if mmap_has_content:
            # mmap format - load as memory-mapped array
            features = np.memmap(data_mmap, dtype=np.float32, mode="r")
        else:
            # npy format
            features = np.load(data_npy)
        labels = np.load(labels_path)
        indices = np.load(indices_path)
    except Exception as e:
        errors.append(f"Failed to load numpy files: {e}")
        return ValidationReport(valid=False, errors=errors, warnings=warnings, stats=stats)

    # Record stats
    stats["n_samples"] = len(labels)
    stats["feature_shape"] = list(features.shape)
    stats["feature_dtype"] = str(features.dtype)
    stats["labels_shape"] = list(labels.shape)
    stats["labels_dtype"] = str(labels.dtype)
    stats["indices_shape"] = list(indices.shape)
    stats["indices_dtype"] = str(indices.dtype)
    stats["is_mmap_format"] = mmap_has_content

    # Check shapes match
    if len(labels) != len(indices):
        errors.append(
            f"Shape mismatch: labels has {len(labels)} samples, indices has {len(indices)} samples"
        )

    # Validate feature array (2D for feature format, 1D for mmap format)
    if mmap_has_content:
        # mmap format stores raw audio as 1D
        if features.ndim != 1:
            errors.append(f"mmap data must be 1D (raw audio), got {features.ndim}D array")
    else:
        # npy format stores features as 2D
        if features.ndim != 2:
            errors.append(f"Features must be 2D, got {features.ndim}D array")
        elif features.shape[1] <= 0:
            errors.append(f"Invalid feature dimension: {features.shape[1]}")

    # Validate labels
    if labels.ndim != 1:
        errors.append(f"Labels must be 1D, got {labels.ndim}D array")

    unique_labels = np.unique(labels)
    if not set(unique_labels).issubset({0, 1}):
        errors.append(f"Labels must be 0 or 1, found unique values: {unique_labels.tolist()}")

    # Validate indices (2D (N, 2) for mmap, 1D for feature format)
    if mmap_has_content:
        # mmap format uses 2D indices (start, end) pairs
        if indices.ndim != 2:
            errors.append(f"mmap indices must be 2D (N x 2), got {indices.ndim}D array")
        elif indices.shape[1] != 2:
            errors.append(f"mmap indices must have 2 columns, got {indices.shape[1]} columns")
    else:
        # npy format uses 1D indices
        if indices.ndim != 1:
            errors.append(f"Indices must be 1D, got {indices.ndim}D array")

    if not np.issubdtype(indices.dtype, np.integer):
        errors.append(f"Indices must be integer type, got {indices.dtype}")

    # Check indices are within bounds (only for feature format)
    if not mmap_has_content and len(features) > 0:
        if indices.size > 0:
            min_idx = int(np.min(indices))
            max_idx = int(np.max(indices))
            if min_idx < 0:
                errors.append(f"Invalid index: {min_idx} (indices must be >= 0)")
            if max_idx >= len(features):
                errors.append(f"Invalid index: {max_idx} (indices must be < {len(features)})")

    # For mmap format, validate start/end pairs are valid
    if mmap_has_content and indices.ndim == 2 and indices.shape[0] > 0:
        start_indices = indices[:, 0]
        end_indices = indices[:, 1]
        if np.any(start_indices < 0):
            errors.append("mmap indices contain negative start values")
        if np.any(end_indices > len(features)):
            errors.append("mmap indices contain end values beyond data length")
        if np.any(start_indices > end_indices):
            errors.append("mmap indices contain start > end pairs")

    # Check for NaN/Inf values
    if np.any(np.isnan(features)):
        errors.append("Features contain NaN values")

    if np.any(np.isinf(features)):
        errors.append("Features contain Inf values")

    if np.any(np.isnan(labels)):
        errors.append("Labels contain NaN values")

    if np.any(np.isinf(labels)):
        errors.append("Labels contain Inf values")

    # Check file sizes (warn if suspiciously small or large)
    features_size = features.nbytes
    labels_size = labels.nbytes
    indices_size = indices.nbytes

    if features_size < 100:
        warnings.append(f"Features file suspiciously small: {features_size} bytes")

    stats["features_bytes"] = features_size
    stats["labels_bytes"] = labels_size
    stats["indices_bytes"] = indices_size

    # Check dtype validity
    if not np.issubdtype(features.dtype, np.floating):
        warnings.append(f"Features dtype is {features.dtype}, expected float type")

    if not np.issubdtype(labels.dtype, np.integer) and not np.issubdtype(labels.dtype, np.floating):
        warnings.append(f"Labels dtype is {labels.dtype}, expected numeric type")

    # Calculate class distribution
    n_positives = int(np.sum(labels == 1))
    n_negatives = int(np.sum(labels == 0))
    stats["n_positives"] = n_positives
    stats["n_negatives"] = n_negatives

    if n_positives == 0:
        warnings.append("No positive samples found in labels")
    if n_negatives == 0:
        warnings.append("No negative samples found in labels")

    return ValidationReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def validate_openwakeword(export_dir: Path, split: str = "train") -> ValidationReport:
    """Validate an OpenWakeWord format export directory.

    Checks for:
    - Required files exist (X_{split}.npy, y_{split}.npy)
    - X and y shapes match
    - Data integrity (no NaN/Inf values)
    - Valid labels are 0 or 1
    - Valid array dtypes
    - Reasonable file sizes

    Args:
        export_dir: Path to the export directory to validate.
        split: Split name used during export (default: "train").

    Returns:
        ValidationReport with validation results.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    export_dir = Path(export_dir)

    X_path = export_dir / f"X_{split}.npy"
    y_path = export_dir / f"y_{split}.npy"

    # Check required files exist
    if not X_path.exists():
        errors.append(f"Missing required file: {X_path.name}")
    elif X_path.stat().st_size == 0:
        errors.append(f"Empty file: {X_path.name}")

    if not y_path.exists():
        errors.append(f"Missing required file: {y_path.name}")
    elif y_path.stat().st_size == 0:
        errors.append(f"Empty file: {y_path.name}")

    if errors:
        return ValidationReport(valid=False, errors=errors, warnings=warnings, stats=stats)

    # Load arrays
    try:
        X = np.load(X_path)
        y = np.load(y_path)
    except Exception as e:
        errors.append(f"Failed to load numpy files: {e}")
        return ValidationReport(valid=False, errors=errors, warnings=warnings, stats=stats)

    # Record stats
    stats["n_samples"] = len(X)
    stats["X_shape"] = list(X.shape)
    stats["X_dtype"] = str(X.dtype)
    stats["y_shape"] = list(y.shape)
    stats["y_dtype"] = str(y.dtype)

    # Check shapes match
    if len(X) != len(y):
        errors.append(f"Shape mismatch: X has {len(X)} samples, y has {len(y)} samples")

    # Validate X array
    if X.ndim != 2:
        errors.append(f"X must be 2D, got {X.ndim}D array")
    elif X.shape[1] <= 0:
        errors.append(f"Invalid feature dimension: {X.shape[1]}")

    # Validate y array
    if y.ndim != 1:
        errors.append(f"y must be 1D, got {y.ndim}D array")

    # Check labels are 0 or 1
    unique_labels = np.unique(y)
    if not set(unique_labels).issubset({0, 1}):
        errors.append(f"Labels must be 0 or 1, found unique values: {unique_labels.tolist()}")

    # Check for NaN/Inf values
    if np.any(np.isnan(X)):
        errors.append("X contains NaN values")

    if np.any(np.isinf(X)):
        errors.append("X contains Inf values")

    if np.any(np.isnan(y)):
        errors.append("y contains NaN values")

    if np.any(np.isinf(y)):
        errors.append("y contains Inf values")

    # Check file sizes
    X_size = X.nbytes
    y_size = y.nbytes

    if X_size < 100:
        warnings.append(f"X file suspiciously small: {X_size} bytes")

    stats["X_bytes"] = X_size
    stats["y_bytes"] = y_size

    # Check dtype validity
    if not np.issubdtype(X.dtype, np.floating):
        warnings.append(f"X dtype is {X.dtype}, expected float type")

    if not np.issubdtype(y.dtype, np.integer) and not np.issubdtype(y.dtype, np.floating):
        warnings.append(f"y dtype is {y.dtype}, expected numeric type")

    # Calculate class distribution
    n_positives = int(np.sum(y == 1))
    n_negatives = int(np.sum(y == 0))
    stats["n_positives"] = n_positives
    stats["n_negatives"] = n_negatives

    if n_positives == 0:
        warnings.append("No positive samples found in y")
    if n_negatives == 0:
        warnings.append("No negative samples found in y")

    return ValidationReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )
