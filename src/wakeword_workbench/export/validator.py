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


def validate_microwakeword(export_dir: Path) -> ValidationReport:
    """Validate a MicroWakeWord format export directory.

    Checks for:
    - Required files exist (features.npy, labels.npy, indices.npy)
    - Shapes match between files
    - Data integrity (no NaN/Inf values)
    - Valid array dtypes
    - Reasonable file sizes
    - Valid indices (within bounds, proper dtype)

    Args:
        export_dir: Path to the export directory to validate.

    Returns:
        ValidationReport with validation results.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    export_dir = Path(export_dir)

    # Check required files exist
    required_files = ["features.npy", "labels.npy", "indices.npy"]
    for filename in required_files:
        file_path = export_dir / filename
        if not file_path.exists():
            errors.append(f"Missing required file: {filename}")
        elif file_path.stat().st_size == 0:
            errors.append(f"Empty file: {filename}")

    if errors:
        return ValidationReport(valid=False, errors=errors, warnings=warnings, stats=stats)

    # Load arrays
    try:
        features = np.load(export_dir / "features.npy")
        labels = np.load(export_dir / "labels.npy")
        indices = np.load(export_dir / "indices.npy")
    except Exception as e:
        errors.append(f"Failed to load numpy files: {e}")
        return ValidationReport(valid=False, errors=errors, warnings=warnings, stats=stats)

    # Record stats
    stats["n_samples"] = len(features)
    stats["feature_shape"] = list(features.shape)
    stats["feature_dtype"] = str(features.dtype)
    stats["labels_shape"] = list(labels.shape)
    stats["labels_dtype"] = str(labels.dtype)
    stats["indices_shape"] = list(indices.shape)
    stats["indices_dtype"] = str(indices.dtype)

    # Check shapes match
    if len(features) != len(labels):
        errors.append(
            f"Shape mismatch: features has {len(features)} samples, "
            f"labels has {len(labels)} samples"
        )

    if len(features) != len(indices):
        errors.append(
            f"Shape mismatch: features has {len(features)} samples, "
            f"indices has {len(indices)} samples"
        )

    # Validate feature array
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

    # Validate indices
    if indices.ndim != 1:
        errors.append(f"Indices must be 1D, got {indices.ndim}D array")

    if not np.issubdtype(indices.dtype, np.integer):
        errors.append(f"Indices must be integer type, got {indices.dtype}")

    # Check indices are within bounds
    if len(features) > 0:
        if indices.size > 0:
            min_idx = int(np.min(indices))
            max_idx = int(np.max(indices))
            if min_idx < 0:
                errors.append(f"Invalid index: {min_idx} (indices must be >= 0)")
            if max_idx >= len(features):
                errors.append(f"Invalid index: {max_idx} (indices must be < {len(features)})")

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


def validate_openwakeword(export_dir: Path) -> ValidationReport:
    """Validate an OpenWakeWord format export directory.

    Checks for:
    - Required files exist (X.npy, y.npy)
    - X and y shapes match
    - Data integrity (no NaN/Inf values)
    - Valid labels are 0 or 1
    - Valid array dtypes
    - Reasonable file sizes

    Args:
        export_dir: Path to the export directory to validate.

    Returns:
        ValidationReport with validation results.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    export_dir = Path(export_dir)

    # Check required files exist
    required_files = ["X.npy", "y.npy"]
    for filename in required_files:
        file_path = export_dir / filename
        if not file_path.exists():
            errors.append(f"Missing required file: {filename}")
        elif file_path.stat().st_size == 0:
            errors.append(f"Empty file: {filename}")

    if errors:
        return ValidationReport(valid=False, errors=errors, warnings=warnings, stats=stats)

    # Load arrays
    try:
        X = np.load(export_dir / "X.npy")
        y = np.load(export_dir / "y.npy")
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
