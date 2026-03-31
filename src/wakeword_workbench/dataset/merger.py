"""Dataset merger for combining positive and negative datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wakeword_workbench.logging_config import get_logger

if TYPE_CHECKING:
    from .metadata import Manifest

log = get_logger(__name__)


class MergerError(Exception):
    """Base exception for merger operations."""

    pass


class PathCollisionError(MergerError):
    """Raised when the same path exists in both positive and negative manifests."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        super().__init__(f"Path collision(s) found: {', '.join(paths)}")


class MergerValidationError(MergerError):
    """Raised when merger validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


@dataclass
class MergeResult:
    """Result of a merge operation with metadata."""

    manifest: Manifest
    total_entries: int
    positive_count: int
    negative_count: int
    target_ratio: float | None
    actual_ratio: float
    warnings: list[str]

    def has_warnings(self) -> bool:
        """Return True if there are any warnings."""
        return len(self.warnings) > 0


# Re-export Manifest type for use in annotations
if TYPE_CHECKING:
    from .metadata import Manifest


def merge(
    pos_manifest: Manifest,
    neg_manifest: Manifest,
    ratio: float | None = None,
    seed: int | None = None,
    validate_files: bool = False,
) -> Manifest:
    """Combine positive and negative manifests with balanced class distribution.

    Args:
        pos_manifest: Manifest containing positive (label=1) entries.
        neg_manifest: Manifest containing negative (label=0) entries.
        ratio: Target ratio of negatives to positives (neg/pos).
               - 1.0 means equal positives and negatives
               - 2.0 means 2x negatives to positives
               - None means use all available entries from both manifests
        seed: Random seed for deterministic shuffling. If None, no shuffling.
        validate_files: If True, verify that referenced files exist on disk.

    Returns:
        A new Manifest with combined and optionally balanced entries.

    Raises:
        MergerError: If both manifests are empty.
        PathCollisionError: If the same path appears in both manifests.
        MergerValidationError: If file validation fails.

    Examples:
        >>> from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
        >>> pos = Manifest([ManifestEntry(path="p1.wav", label=1, text="hey marvin", duration_ms=1000)])
        >>> neg = Manifest([ManifestEntry(path="n1.wav", label=0, text="", duration_ms=2000)])
        >>> combined = merge(pos, neg, ratio=1.0, seed=42)
        >>> len(combined)
        2
    """
    log.debug("merge_start", pos_count=len(pos_manifest), neg_count=len(neg_manifest), ratio=ratio)

    # Extract entries from each manifest
    pos_entries = list(pos_manifest)
    neg_entries = list(neg_manifest)

    # Validate no path collisions
    _check_path_collisions(pos_entries, neg_entries)

    # Validate not both empty
    if not pos_entries and not neg_entries:
        raise MergerError("Both positive and negative manifests are empty")

    warnings_list: list[str] = []

    # Apply ratio balancing
    if ratio is not None:
        balanced_pos, balanced_neg, warn = _balance_by_ratio(pos_entries, neg_entries, ratio)
        warnings_list.extend(warn)
        entries = balanced_pos + balanced_neg
        log.debug(
            "ratio_balancing_applied",
            target_ratio=ratio,
            pos_kept=len(balanced_pos),
            neg_kept=len(balanced_neg),
        )
    else:
        entries = pos_entries + neg_entries
        log.debug("no_ratio_balancing", total_entries=len(entries))

    # Deterministic shuffle
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(entries)
        log.debug("shuffling_with_seed", seed=seed)

    # File existence validation (optional, off by default)
    if validate_files:
        file_errors = _validate_file_existence(entries)
        if file_errors:
            raise MergerValidationError(file_errors)

    result = _create_manifest_from_entries(entries)
    log.info("merge_complete", total_entries=len(result))
    return result


def merge_with_result(
    pos_manifest: Manifest,
    neg_manifest: Manifest,
    ratio: float | None = None,
    seed: int | None = None,
    validate_files: bool = False,
) -> MergeResult:
    """Merge manifests and return detailed result with statistics.

    Args:
        pos_manifest: Manifest containing positive (label=1) entries.
        neg_manifest: Manifest containing negative (label=0) entries.
        ratio: Target ratio of negatives to positives. None means use all.
        seed: Random seed for deterministic shuffling.
        validate_files: If True, verify that referenced files exist.

    Returns:
        A MergeResult containing the manifest and merge statistics.

    Raises:
        Same exceptions as merge().
    """
    log.debug(
        "merge_with_result_start",
        pos_count=len(pos_manifest),
        neg_count=len(neg_manifest),
        ratio=ratio,
    )

    pos_entries = list(pos_manifest)
    neg_entries = list(neg_manifest)

    _check_path_collisions(pos_entries, neg_entries)

    if not pos_entries and not neg_entries:
        raise MergerError("Both positive and negative manifests are empty")

    warnings_list: list[str] = []

    if ratio is not None:
        balanced_pos, balanced_neg, ratio_warnings = _balance_by_ratio(
            pos_entries, neg_entries, ratio
        )
        warnings_list.extend(ratio_warnings)
        entries = balanced_pos + balanced_neg
        actual_pos = len(balanced_pos)
        actual_neg = len(balanced_neg)
    else:
        entries = pos_entries + neg_entries
        actual_pos = len(pos_entries)
        actual_neg = len(neg_entries)

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(entries)

    if validate_files:
        file_errors = _validate_file_existence(entries)
        if file_errors:
            raise MergerValidationError(file_errors)

    manifest = _create_manifest_from_entries(entries)
    total = len(entries)
    actual_ratio = actual_neg / actual_pos if actual_pos > 0 else 0.0

    log.info(
        "merge_with_result_complete",
        total_entries=total,
        positive_count=actual_pos,
        negative_count=actual_neg,
        target_ratio=ratio,
        actual_ratio=round(actual_ratio, 2),
        warnings_count=len(warnings_list),
    )

    return MergeResult(
        manifest=manifest,
        total_entries=total,
        positive_count=actual_pos,
        negative_count=actual_neg,
        target_ratio=ratio,
        actual_ratio=actual_ratio,
        warnings=warnings_list,
    )


def _balance_by_ratio(
    pos_entries: list,
    neg_entries: list,
    ratio: float,
) -> tuple[list, list, list[str]]:
    """Balance entries by the given negative-to-positive ratio.

    Args:
        pos_entries: List of positive entries.
        neg_entries: List of negative entries.
        ratio: Target ratio (negatives / positives).

    Returns:
        Tuple of (balanced_pos, balanced_neg, warnings).
    """
    warnings: list[str] = []
    num_pos = len(pos_entries)
    num_neg = len(neg_entries)

    if num_pos == 0:
        warnings.append("No positive entries found, cannot apply ratio balancing")
        log.warning("ratio_balancing_no_positives")
        return [], neg_entries[:], warnings

    target_neg = int(num_pos * ratio)

    # Sample negatives
    if num_neg >= target_neg:
        # Have enough negatives - sample to target
        balanced_neg = _sample_entries(neg_entries, target_neg)
        if num_neg > target_neg:
            msg = f"Using {target_neg}/{num_neg} negatives to match ratio {ratio}. Consider generating more negatives."
            warnings.append(msg)
            log.warning(
                "ratio_balancing_undersample", target_neg=target_neg, available=num_neg, ratio=ratio
            )
    else:
        # Not enough negatives - use all and warn
        balanced_neg = list(neg_entries)
        actual = num_neg / num_pos if num_pos > 0 else 0
        msg = f"Not enough negatives ({num_neg}) for ratio {ratio} (need {target_neg}). Using all {num_neg} available negatives. Actual ratio: {actual:.2f}"
        warnings.append(msg)
        log.warning(
            "ratio_balancing_insufficient_negatives",
            available=num_neg,
            target=target_neg,
            ratio=ratio,
            actual_ratio=round(actual, 2),
        )

    return list(pos_entries), balanced_neg, warnings


def _sample_entries(entries: list, count: int) -> list:
    """Sample entries deterministically using reservoir sampling.

    When count < len(entries), uses a seeded random for determinism.
    """
    if count >= len(entries):
        return list(entries)

    # Use random.sample which is deterministic with same seed context
    rng = random.Random()
    return rng.sample(entries, count)


def _check_path_collisions(pos_entries: list, neg_entries: list) -> None:
    """Check for path collisions between positive and negative manifests.

    Raises:
        PathCollisionError: If any paths collide.
    """
    pos_paths = {e.path for e in pos_entries}
    neg_paths = {e.path for e in neg_entries}
    collisions = sorted(pos_paths & neg_paths)
    if collisions:
        raise PathCollisionError(collisions)


def _validate_file_existence(entries: list) -> list[str]:
    """Validate that all referenced files exist.

    Args:
        entries: List of manifest entries to validate.

    Returns:
        List of error messages for files that don't exist.
    """
    errors: list[str] = []
    for entry in entries:
        if not Path(entry.path).exists():
            errors.append(f"File not found: {entry.path}")
    return errors


def _create_manifest_from_entries(entries: list) -> Manifest:
    """Create a new Manifest from a list of entries.

    Args:
        entries: List of ManifestEntry objects.

    Returns:
        A new Manifest containing the entries.
    """
    from .metadata import Manifest

    return Manifest(entries)
