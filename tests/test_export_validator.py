"""Tests for export validator module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wakeword_workbench.export.validator import (
    ValidationReport,
    validate_microwakeword,
    validate_openwakeword,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_mww_dir(tmp_path: Path) -> Path:
    """Create a valid MicroWakeWord export directory."""
    export_dir = tmp_path / "mww_export"
    export_dir.mkdir()

    # Create valid data with split-based filenames
    features = np.random.randn(100, 40).astype(np.float32)
    labels = np.array([1 if i < 50 else 0 for i in range(100)], dtype=np.int32)
    indices = np.arange(100, dtype=np.int32)

    np.save(export_dir / "train_data.npy", features)
    np.save(export_dir / "train_labels.npy", labels)
    np.save(export_dir / "train_indices.npy", indices)

    return export_dir


@pytest.fixture
def valid_oww_dir(tmp_path: Path) -> Path:
    """Create a valid OpenWakeWord export directory."""
    export_dir = tmp_path / "oww_export"
    export_dir.mkdir()

    # Create valid data with split-based filenames
    X = np.random.randn(100, 40).astype(np.float32)
    y = np.array([1 if i < 50 else 0 for i in range(100)], dtype=np.int32)

    np.save(export_dir / "X_train.npy", X)
    np.save(export_dir / "y_train.npy", y)

    return export_dir


# ---------------------------------------------------------------------------
# ValidationReport tests
# ---------------------------------------------------------------------------


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_valid_report(self) -> None:
        """Test creating a valid report."""
        report = ValidationReport(
            valid=True,
            errors=[],
            warnings=["Minor warning"],
            stats={"n_samples": 100},
        )
        assert report.valid is True
        assert len(report.errors) == 0
        assert len(report.warnings) == 1

    def test_invalid_report(self) -> None:
        """Test creating an invalid report."""
        report = ValidationReport(
            valid=False,
            errors=["Critical error"],
            warnings=[],
            stats={},
        )
        assert report.valid is False
        assert "Critical error" in report.errors


# ---------------------------------------------------------------------------
# MicroWakeWord validation tests
# ---------------------------------------------------------------------------


class TestValidateMicrowakeword:
    """Tests for validate_microwakeword function."""

    def test_valid_export(self, valid_mww_dir: Path) -> None:
        """Test validation passes for a valid export."""
        report = validate_microwakeword(valid_mww_dir)
        assert report.valid is True
        assert len(report.errors) == 0
        assert report.stats["n_samples"] == 100
        assert report.stats["n_positives"] == 50
        assert report.stats["n_negatives"] == 50

    def test_missing_features_file(self, tmp_path: Path) -> None:
        """Test validation fails when train_data.npy is missing."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()
        np.save(export_dir / "train_labels.npy", np.array([0, 1]))
        np.save(export_dir / "train_indices.npy", np.array([0, 1]))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("train_data.mmap or train_data.npy" in err for err in report.errors)

    def test_missing_labels_file(self, tmp_path: Path) -> None:
        """Test validation fails when train_labels.npy is missing."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()
        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_indices.npy", np.arange(10))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("train_labels.npy" in err for err in report.errors)

    def test_missing_indices_file(self, tmp_path: Path) -> None:
        """Test validation fails when train_indices.npy is missing."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()
        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.array([0, 1] * 5))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("train_indices.npy" in err for err in report.errors)

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test validation fails for empty files."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()
        (export_dir / "train_data.npy").touch()
        np.save(export_dir / "train_labels.npy", np.array([0, 1]))
        np.save(export_dir / "train_indices.npy", np.arange(2))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("Empty file" in err for err in report.errors)

    def test_shape_mismatch_features_labels(self, tmp_path: Path) -> None:
        """Test validation fails when features and labels shapes don't match."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.array([0, 1] * 3))  # 6 labels
        np.save(export_dir / "train_indices.npy", np.arange(10))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("Shape mismatch" in err for err in report.errors)

    def test_shape_mismatch_features_indices(self, tmp_path: Path) -> None:
        """Test validation fails when features and indices shapes don't match."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.array([0, 1] * 5))
        np.save(export_dir / "train_indices.npy", np.arange(5))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("Shape mismatch" in err for err in report.errors)

    def test_features_wrong_ndim(self, tmp_path: Path) -> None:
        """Test validation fails when features is not 2D."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10))  # 1D array
        np.save(export_dir / "train_labels.npy", np.array([0, 1] * 5))
        np.save(export_dir / "train_indices.npy", np.arange(10))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("must be 2D" in err for err in report.errors)

    def test_labels_wrong_ndim(self, tmp_path: Path) -> None:
        """Test validation fails when labels is not 1D."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.random.randn(10, 1))  # 2D array
        np.save(export_dir / "train_indices.npy", np.arange(10))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("must be 1D" in err for err in report.errors)

    def test_invalid_label_values(self, tmp_path: Path) -> None:
        """Test validation fails when labels contain values other than 0 or 1."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]))
        np.save(export_dir / "train_indices.npy", np.arange(10))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("must be 0 or 1" in err for err in report.errors)

    def test_indices_non_integer(self, tmp_path: Path) -> None:
        """Test validation fails when indices is not integer type."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.array([0, 1] * 5))
        np.save(export_dir / "train_indices.npy", np.arange(10, dtype=np.float32))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("must be integer type" in err for err in report.errors)

    def test_negative_index(self, tmp_path: Path) -> None:
        """Test validation fails when indices contain negative values."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.array([0, 1] * 5))
        np.save(export_dir / "train_indices.npy", np.array([0, 1, -1, 3, 4, 5, 6, 7, 8, 9]))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("Invalid index" in err for err in report.errors)

    def test_out_of_bounds_index(self, tmp_path: Path) -> None:
        """Test validation fails when indices exceed array bounds."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.array([0, 1] * 5))
        np.save(export_dir / "train_indices.npy", np.array([0, 1, 10, 3, 4, 5, 6, 7, 8, 9]))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("Invalid index" in err for err in report.errors)

    def test_nan_in_features(self, tmp_path: Path) -> None:
        """Test validation fails when features contain NaN."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        features = np.random.randn(10, 40)
        features[0, 0] = np.nan
        np.save(export_dir / "train_data.npy", features)
        np.save(export_dir / "train_labels.npy", np.array([0, 1] * 5))
        np.save(export_dir / "train_indices.npy", np.arange(10))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("NaN" in err for err in report.errors)

    def test_inf_in_features(self, tmp_path: Path) -> None:
        """Test validation fails when features contain Inf."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        features = np.random.randn(10, 40)
        features[0, 0] = np.inf
        np.save(export_dir / "train_data.npy", features)
        np.save(export_dir / "train_labels.npy", np.array([0, 1] * 5))
        np.save(export_dir / "train_indices.npy", np.arange(10))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("Inf" in err for err in report.errors)

    def test_nan_in_labels(self, tmp_path: Path) -> None:
        """Test validation fails when labels contain NaN."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        labels = np.array([0.0, np.nan, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        np.save(export_dir / "train_labels.npy", labels)
        np.save(export_dir / "train_indices.npy", np.arange(10))

        report = validate_microwakeword(export_dir)
        assert report.valid is False
        assert any("Labels contain NaN" in err for err in report.errors)

    def test_no_positive_samples(self, tmp_path: Path) -> None:
        """Test warning when no positive samples found."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.zeros(10, dtype=np.int32))
        np.save(export_dir / "train_indices.npy", np.arange(10))

        report = validate_microwakeword(export_dir)
        assert report.valid is True
        assert any("No positive samples" in warn for warn in report.warnings)

    def test_no_negative_samples(self, tmp_path: Path) -> None:
        """Test warning when no negative samples found."""
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()

        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.ones(10, dtype=np.int32))
        np.save(export_dir / "train_indices.npy", np.arange(10))

        report = validate_microwakeword(export_dir)
        assert report.valid is True
        assert any("No negative samples" in warn for warn in report.warnings)


# ---------------------------------------------------------------------------
# OpenWakeWord validation tests
# ---------------------------------------------------------------------------


class TestValidateOpenwakeword:
    """Tests for validate_openwakeword function."""

    def test_valid_export(self, valid_oww_dir: Path) -> None:
        """Test validation passes for a valid export."""
        report = validate_openwakeword(valid_oww_dir)
        assert report.valid is True
        assert len(report.errors) == 0
        assert report.stats["n_samples"] == 100
        assert report.stats["n_positives"] == 50
        assert report.stats["n_negatives"] == 50

    def test_missing_X_file(self, tmp_path: Path) -> None:
        """Test validation fails when X_train.npy is missing."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()
        np.save(export_dir / "y_train.npy", np.array([0, 1]))

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("X_train.npy" in err for err in report.errors)

    def test_missing_y_file(self, tmp_path: Path) -> None:
        """Test validation fails when y_train.npy is missing."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()
        np.save(export_dir / "X_train.npy", np.random.randn(10, 40))

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("y_train.npy" in err for err in report.errors)

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test validation fails for empty files."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()
        (export_dir / "X_train.npy").touch()
        np.save(export_dir / "y_train.npy", np.array([0, 1]))

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("Empty file" in err for err in report.errors)

    def test_shape_mismatch_X_y(self, tmp_path: Path) -> None:
        """Test validation fails when X and y shapes don't match."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()

        np.save(export_dir / "X_train.npy", np.random.randn(10, 40))
        np.save(export_dir / "y_train.npy", np.array([0, 1] * 3))  # 6 labels

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("Shape mismatch" in err for err in report.errors)

    def test_X_wrong_ndim(self, tmp_path: Path) -> None:
        """Test validation fails when X is not 2D."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()

        np.save(export_dir / "X_train.npy", np.random.randn(10))  # 1D array
        np.save(export_dir / "y_train.npy", np.array([0, 1] * 5))

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("must be 2D" in err for err in report.errors)

    def test_y_wrong_ndim(self, tmp_path: Path) -> None:
        """Test validation fails when y is not 1D."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()

        np.save(export_dir / "X_train.npy", np.random.randn(10, 40))
        np.save(export_dir / "y_train.npy", np.random.randn(10, 1))  # 2D array

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("must be 1D" in err for err in report.errors)

    def test_invalid_label_values(self, tmp_path: Path) -> None:
        """Test validation fails when y contains values other than 0 or 1."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()

        np.save(export_dir / "X_train.npy", np.random.randn(10, 40))
        np.save(export_dir / "y_train.npy", np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]))

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("must be 0 or 1" in err for err in report.errors)

    def test_nan_in_X(self, tmp_path: Path) -> None:
        """Test validation fails when X contains NaN."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()

        X = np.random.randn(10, 40)
        X[0, 0] = np.nan
        np.save(export_dir / "X_train.npy", X)
        np.save(export_dir / "y_train.npy", np.array([0, 1] * 5))

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("NaN" in err for err in report.errors)

    def test_inf_in_X(self, tmp_path: Path) -> None:
        """Test validation fails when X contains Inf."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()

        X = np.random.randn(10, 40)
        X[0, 0] = np.inf
        np.save(export_dir / "X_train.npy", X)
        np.save(export_dir / "y_train.npy", np.array([0, 1] * 5))

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("Inf" in err for err in report.errors)

    def test_nan_in_y(self, tmp_path: Path) -> None:
        """Test validation fails when y contains NaN."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()

        np.save(export_dir / "X_train.npy", np.random.randn(10, 40))
        y = np.array([0.0, np.nan, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        np.save(export_dir / "y_train.npy", y)

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("y contains NaN" in err for err in report.errors)

    def test_no_positive_samples(self, tmp_path: Path) -> None:
        """Test warning when no positive samples found."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()

        np.save(export_dir / "X_train.npy", np.random.randn(10, 40))
        np.save(export_dir / "y_train.npy", np.zeros(10, dtype=np.int32))

        report = validate_openwakeword(export_dir)
        assert report.valid is True
        assert any("No positive samples" in warn for warn in report.warnings)

    def test_no_negative_samples(self, tmp_path: Path) -> None:
        """Test warning when no negative samples found."""
        export_dir = tmp_path / "oww_export"
        export_dir.mkdir()

        np.save(export_dir / "X_train.npy", np.random.randn(10, 40))
        np.save(export_dir / "y_train.npy", np.ones(10, dtype=np.int32))

        report = validate_openwakeword(export_dir)
        assert report.valid is True
        assert any("No negative samples" in warn for warn in report.warnings)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestValidatorIntegration:
    """Integration tests for the validator."""

    def test_corrupted_numpy_file(self, tmp_path: Path) -> None:
        """Test detection of corrupted numpy file."""
        export_dir = tmp_path / "corrupted_export"
        export_dir.mkdir()

        # Write invalid numpy file
        (export_dir / "X_train.npy").write_bytes(b"not a numpy file")
        np.save(export_dir / "y_train.npy", np.array([0, 1]))

        report = validate_openwakeword(export_dir)
        assert report.valid is False
        assert any("Failed to load" in err for err in report.errors)

    def test_stats_calculation(self, valid_mww_dir: Path) -> None:
        """Test that stats are correctly calculated."""
        report = validate_microwakeword(valid_mww_dir)

        assert "n_samples" in report.stats
        assert "feature_shape" in report.stats
        assert "labels_shape" in report.stats
        assert "n_positives" in report.stats
        assert "n_negatives" in report.stats
        assert "features_bytes" in report.stats

        assert report.stats["n_samples"] == 100
        assert report.stats["feature_shape"] == [100, 40]

    def test_both_formats_same_directory(self, tmp_path: Path) -> None:
        """Test validating directory with both file sets."""
        # Create a directory with MicroWakeWord files
        export_dir = tmp_path / "mww_export"
        export_dir.mkdir()
        np.save(export_dir / "train_data.npy", np.random.randn(10, 40))
        np.save(export_dir / "train_labels.npy", np.array([0, 1] * 5))
        np.save(export_dir / "train_indices.npy", np.arange(10))

        # MicroWakeWord validation should pass
        mww_report = validate_microwakeword(export_dir)
        assert mww_report.valid is True

        # OpenWakeWord validation should fail (missing X_train.npy, y_train.npy)
        oww_report = validate_openwakeword(export_dir)
        assert oww_report.valid is False
        assert any("X_train.npy" in err for err in oww_report.errors)
