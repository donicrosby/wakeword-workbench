"""Dataset splitter with leak prevention and stratification."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wakeword_workbench.logging_config import get_logger

from .metadata import Manifest

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


class SplitterError(Exception):
    """Base exception for splitter operations."""

    pass


class SplitValidationError(SplitterError):
    """Raised when split validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Split validation failed: {'; '.join(errors)}")


@dataclass
class SpeakerGroup:
    """A group of entries belonging to the same speaker."""

    speaker_id: str
    entries: list  # List of ManifestEntry
    positive_count: int
    negative_count: int


def split(
    manifest: Manifest,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    by: str = "speaker",
    seed: int | None = None,
) -> tuple[Manifest, Manifest, Manifest]:
    """Split dataset into train/val/test with leak prevention and stratification.

    Args:
        manifest: The manifest to split.
        train: Proportion for training set (default: 0.7).
        val: Proportion for validation set (default: 0.15).
        test: Proportion for test set (default: 0.15).
        by: Grouping key - "speaker" or "voice" (default: "speaker").
        seed: Random seed for reproducibility (default: None).

    Returns:
        A tuple of (train_manifest, val_manifest, test_manifest).

    Raises:
        SplitValidationError: If split ratios don't sum to 1.0 or validation fails.
    """
    log.debug(
        "split_start",
        total_entries=len(manifest),
        train_ratio=train,
        val_ratio=val,
        test_ratio=test,
        by=by,
        seed=seed,
    )

    # Validate split ratios
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise SplitValidationError(
            [f"Split ratios must sum to 1.0, got {train} + {val} + {test} = {total}"]
        )

    # Handle empty manifest
    if len(manifest) == 0:
        log.info("split_complete_empty_manifest")
        return Manifest(), Manifest(), Manifest()

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Group entries by speaker
    speaker_groups = _group_by_speaker(manifest, by)
    log.debug("speaker_groups_created", group_count=len(speaker_groups))

    # Stratify by label while maintaining speaker grouping
    train_groups, val_groups, test_groups = _stratified_split(speaker_groups, train, val, test)

    # Build manifests from groups
    train_manifest = _build_manifest(train_groups)
    val_manifest = _build_manifest(val_groups)
    test_manifest = _build_manifest(test_groups)

    # Validate the split
    _validate_split(manifest, train_manifest, val_manifest, test_manifest, by)

    log.info(
        "split_complete",
        total=len(manifest),
        train_count=len(train_manifest),
        val_count=len(val_manifest),
        test_count=len(test_manifest),
    )

    return train_manifest, val_manifest, test_manifest


def _group_by_speaker(manifest: Manifest, by: str) -> list[SpeakerGroup]:
    """Group manifest entries by speaker/voice.

    Args:
        manifest: The manifest to group.
        by: Grouping key - "speaker" or "voice".

    Returns:
        List of SpeakerGroup objects.
    """
    log.debug("grouping_by_speaker", by=by, total_entries=len(manifest))

    # Use voice field as speaker identifier, or "unknown" if None
    groups: dict[str, list] = defaultdict(list)

    for entry in manifest:
        speaker_id = entry.voice if entry.voice else "unknown"
        groups[speaker_id].append(entry)

    # Convert to SpeakerGroup objects
    result: list[SpeakerGroup] = []
    for speaker_id, entries in groups.items():
        positive_count = sum(1 for e in entries if e.label == 1)
        negative_count = sum(1 for e in entries if e.label == 0)
        result.append(
            SpeakerGroup(
                speaker_id=speaker_id,
                entries=entries,
                positive_count=positive_count,
                negative_count=negative_count,
            )
        )

    log.debug(
        "speaker_groups_formed",
        speaker_count=len(result),
        unknown_count=sum(1 for g in result if g.speaker_id == "unknown"),
    )

    return result


def _stratified_split(
    groups: list[SpeakerGroup], train: float, val: float, test: float
) -> tuple[list[SpeakerGroup], list[SpeakerGroup], list[SpeakerGroup]]:
    """Split speaker groups into train/val/test with stratification.

    This function attempts to maintain class distribution across splits
    while ensuring all entries from a speaker stay together.

    Args:
        groups: List of speaker groups.
        train: Training proportion.
        val: Validation proportion.
        test: Test proportion.

    Returns:
        Tuple of (train_groups, val_groups, test_groups).
    """
    # Calculate target counts based on entries, not just speakers
    total_entries = sum(g.positive_count + g.negative_count for g in groups)
    total_positive = sum(g.positive_count for g in groups)

    if total_entries == 0:
        return [], [], list(groups)

    target_train_entries = int(total_entries * train)
    target_val_entries = int(total_entries * val)
    # Separate groups by their label composition
    all_positive = [g for g in groups if g.positive_count > 0 and g.negative_count == 0]
    all_negative = [g for g in groups if g.negative_count > 0 and g.positive_count == 0]
    mixed = [g for g in groups if g.positive_count > 0 and g.negative_count > 0]

    # Shuffle each category
    random.shuffle(all_positive)
    random.shuffle(all_negative)
    random.shuffle(mixed)

    train_groups: list[SpeakerGroup] = []
    val_groups: list[SpeakerGroup] = []
    test_groups: list[SpeakerGroup] = []

    train_positive = 0
    train_negative = 0

    def add_to_train(g: SpeakerGroup) -> None:
        nonlocal train_positive, train_negative
        train_groups.append(g)
        train_positive += g.positive_count
        train_negative += g.negative_count

    def add_to_val(g: SpeakerGroup) -> None:
        val_groups.append(g)

    def add_to_test(g: SpeakerGroup) -> None:
        test_groups.append(g)

    # Assign pure positive groups with round-robin to distribute them
    pos_idx = 0
    for g in all_positive:
        if (
            pos_idx % 3 == 0
            and (train_positive + g.positive_count)
            <= target_train_entries + target_val_entries * 0.3
        ):
            add_to_train(g)
        elif pos_idx % 3 == 1:
            add_to_val(g)
        else:
            add_to_test(g)
        pos_idx += 1

    # Assign pure negative groups with round-robin
    neg_idx = 0
    for g in all_negative:
        if neg_idx % 3 == 0 and (train_negative + g.negative_count) <= target_train_entries * 1.2:
            add_to_train(g)
        elif neg_idx % 3 == 1:
            add_to_val(g)
        else:
            add_to_test(g)
        neg_idx += 1

    # Assign mixed groups - try to balance positive/negative ratio
    for g in mixed:
        current_train_positive_ratio = (
            (train_positive / (train_positive + train_negative))
            if (train_positive + train_negative) > 0
            else 0.5
        )
        target_ratio = total_positive / total_entries

        if abs(current_train_positive_ratio - target_ratio) > abs(
            current_train_positive_ratio
            + (g.positive_count / (g.positive_count + g.negative_count))
            - 2 * target_ratio
        ):
            add_to_train(g)
        elif len(val_groups) < len(groups) * val:
            add_to_val(g)
        else:
            add_to_test(g)

    # If we have very few groups, ensure distribution
    if len(groups) <= 3:
        remaining = [g for g in groups if g not in train_groups + val_groups + test_groups]
        for i, g in enumerate(remaining):
            if i == 0:
                add_to_train(g)
            elif i == 1:
                add_to_val(g)
            else:
                add_to_test(g)

    return train_groups, val_groups, test_groups


def _build_manifest(groups: list[SpeakerGroup]) -> Manifest:
    """Build a Manifest from a list of SpeakerGroups.

    Args:
        groups: List of speaker groups.

    Returns:
        A Manifest containing all entries from the groups.
    """
    from .metadata import Manifest

    all_entries = []
    for group in groups:
        all_entries.extend(group.entries)

    return Manifest(all_entries)


def _validate_split(
    manifest: Manifest,
    train: Manifest,
    val: Manifest,
    test: Manifest,
    by: str,
) -> None:
    """Validate that the split is correct.

    Args:
        manifest: Original manifest.
        train: Training manifest.
        val: Validation manifest.
        test: Test manifest.
        by: Grouping key.

    Raises:
        SplitValidationError: If validation fails.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check no overlapping entries
    train_paths = {e.path for e in train}
    val_paths = {e.path for e in val}
    test_paths = {e.path for e in test}

    train_val_overlap = train_paths & val_paths
    train_test_overlap = train_paths & test_paths
    val_test_overlap = val_paths & test_paths

    if train_val_overlap:
        errors.append(f"Train/val overlap: {len(train_val_overlap)} entries")
    if train_test_overlap:
        errors.append(f"Train/test overlap: {len(train_test_overlap)} entries")
    if val_test_overlap:
        errors.append(f"Val/test overlap: {len(val_test_overlap)} entries")

    # Check all entries accounted for
    all_paths = train_paths | val_paths | test_paths
    original_paths = {e.path for e in manifest}

    if all_paths != original_paths:
        missing = original_paths - all_paths
        extra = all_paths - original_paths
        if missing:
            errors.append(f"Missing {len(missing)} entries after split")
        if extra:
            errors.append(f"Extra {len(extra)} entries after split")

    # Check no speaker leakage
    train_speakers = {e.voice for e in train if e.voice}
    val_speakers = {e.voice for e in val if e.voice}
    test_speakers = {e.voice for e in test if e.voice}

    train_val_speakers = train_speakers & val_speakers
    train_test_speakers = train_speakers & test_speakers
    val_test_speakers = val_speakers & test_speakers

    if train_val_speakers:
        errors.append(f"Speaker leakage: {len(train_val_speakers)} speakers in train and val")
    if train_test_speakers:
        errors.append(f"Speaker leakage: {len(train_test_speakers)} speakers in train and test")
    if val_test_speakers:
        errors.append(f"Speaker leakage: {len(val_test_speakers)} speakers in val and test")

    if errors:
        log.error("split_validation_failed", errors=errors)
        raise SplitValidationError(errors)

    if warnings:
        for w in warnings:
            log.warning("split_validation_warning", warning=w)
