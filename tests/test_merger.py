"""Tests for dataset merger module."""

from __future__ import annotations

from pathlib import Path

import pytest

from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.dataset.merger import (
    MergeResult,
    MergerError,
    MergerValidationError,
    PathCollisionError,
    merge,
    merge_with_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_pos(path: str, text: str = "hey marvin") -> ManifestEntry:
    """Create a positive manifest entry."""
    return ManifestEntry(path=path, label=1, text=text, duration_ms=1000)


def make_neg(path: str) -> ManifestEntry:
    """Create a negative manifest entry."""
    return ManifestEntry(path=path, label=0, text="", duration_ms=500)


def make_manifest(*entries: ManifestEntry) -> Manifest:
    """Create a manifest from entries."""
    return Manifest(list(entries))


# ---------------------------------------------------------------------------
# Ratio balancing
# ---------------------------------------------------------------------------


class TestMergeRatioBalancing:
    """Tests for ratio-based balancing."""

    def test_ratio_equal_pos_and_neg(self) -> None:
        """Test ratio=1.0 gives equal positives and negatives."""
        pos = make_manifest(make_pos("p1.wav"), make_pos("p2.wav"), make_pos("p3.wav"))
        neg = make_manifest(
            make_neg("n1.wav"),
            make_neg("n2.wav"),
            make_neg("n3.wav"),
            make_neg("n4.wav"),
            make_neg("n5.wav"),
            make_neg("n6.wav"),
        )
        result = merge(pos, neg, ratio=1.0)
        pos_count = sum(1 for e in result if e.label == 1)
        neg_count = sum(1 for e in result if e.label == 0)
        assert pos_count == 3
        assert neg_count == 3

    def test_ratio_2x_negatives(self) -> None:
        """Test ratio=2.0 gives 2x negatives to positives."""
        pos = make_manifest(make_pos("p1.wav"), make_pos("p2.wav"))
        neg = make_manifest(
            make_neg("n1.wav"),
            make_neg("n2.wav"),
            make_neg("n3.wav"),
            make_neg("n4.wav"),
            make_neg("n5.wav"),
            make_neg("n6.wav"),
        )
        result = merge(pos, neg, ratio=2.0)
        pos_count = sum(1 for e in result if e.label == 1)
        neg_count = sum(1 for e in result if e.label == 0)
        assert pos_count == 2
        assert neg_count == 4  # 2 * 2

    def test_ratio_with_fewer_negatives_than_needed(self) -> None:
        """Test using fewer negatives than target ratio with warning."""
        pos = make_manifest(make_pos("p1.wav"), make_pos("p2.wav"))
        neg = make_manifest(
            make_neg("n1.wav"), make_neg("n2.wav")
        )  # Only 2 negs, need 4 for ratio=2.0
        res = merge_with_result(pos, neg, ratio=2.0)
        assert len(res.warnings) == 1
        assert "Not enough negatives" in res.warnings[0]
        assert res.positive_count == 2
        assert res.negative_count == 2

    def test_ratio_with_more_negatives_than_needed(self) -> None:
        """Test using fewer negatives than available with warning."""
        pos = make_manifest(make_pos("p1.wav"))
        neg = make_manifest(make_neg("n1.wav"), make_neg("n2.wav"), make_neg("n3.wav"))
        res = merge_with_result(pos, neg, ratio=1.0)
        assert len(res.warnings) == 1
        assert "Using" in res.warnings[0]
        assert "negatives" in res.warnings[0]
        assert res.positive_count == 1
        assert res.negative_count == 1

    def test_ratio_none_uses_all(self) -> None:
        """Test ratio=None uses all entries from both manifests."""
        pos = make_manifest(make_pos("p1.wav"), make_pos("p2.wav"))
        neg = make_manifest(make_neg("n1.wav"), make_neg("n2.wav"), make_neg("n3.wav"))
        result = merge(pos, neg, ratio=None)
        assert len(result) == 5

    def test_ratio_with_only_positives(self) -> None:
        """Test ratio with only positives (no negatives)."""
        pos = make_manifest(make_pos("p1.wav"), make_pos("p2.wav"))
        neg = make_manifest()
        res = merge_with_result(pos, neg, ratio=1.0)
        assert len(res.warnings) == 1
        assert "Not enough negatives" in res.warnings[0]
        assert res.positive_count == 2
        assert res.negative_count == 0

    def test_merge_with_result_reports_stats(self) -> None:
        """Test merge_with_result returns correct statistics."""
        pos = make_manifest(make_pos("p1.wav"), make_pos("p2.wav"), make_pos("p3.wav"))
        neg = make_manifest(
            make_neg("n1.wav"),
            make_neg("n2.wav"),
            make_neg("n3.wav"),
            make_neg("n4.wav"),
            make_neg("n5.wav"),
            make_neg("n6.wav"),
        )
        res = merge_with_result(pos, neg, ratio=1.0)
        assert isinstance(res, MergeResult)
        assert res.total_entries == 6
        assert res.positive_count == 3
        assert res.negative_count == 3
        assert res.target_ratio == 1.0
        assert res.actual_ratio == 1.0


# ---------------------------------------------------------------------------
# Deterministic shuffle
# ---------------------------------------------------------------------------


class TestDeterministicShuffle:
    """Tests for deterministic shuffling with seeds."""

    def test_same_seed_gives_same_order(self) -> None:
        """Test that the same seed produces the same order."""
        pos = make_manifest(*[make_pos(f"p{i}.wav") for i in range(5)])
        neg = make_manifest(*[make_neg(f"n{i}.wav") for i in range(5)])

        result1 = merge(pos, neg, seed=42)
        result2 = merge(pos, neg, seed=42)

        paths1 = [e.path for e in result1]
        paths2 = [e.path for e in result2]
        assert paths1 == paths2

    def test_different_seed_gives_different_order(self) -> None:
        """Test that different seeds produce different orders."""
        pos = make_manifest(*[make_pos(f"p{i}.wav") for i in range(5)])
        neg = make_manifest(*[make_neg(f"n{i}.wav") for i in range(5)])

        result1 = merge(pos, neg, seed=42)
        result2 = merge(pos, neg, seed=99)

        paths1 = [e.path for e in result1]
        paths2 = [e.path for e in result2]
        assert paths1 != paths2

    def test_no_seed_preserves_original_order(self) -> None:
        """Test that no seed keeps entries in original order (pos first, then neg)."""
        pos = make_manifest(make_pos("p1.wav"), make_pos("p2.wav"))
        neg = make_manifest(make_neg("n1.wav"), make_neg("n2.wav"))
        result = merge(pos, neg, seed=None)
        paths = [e.path for e in result]
        # Without seed, no shuffling occurs
        assert paths == ["p1.wav", "p2.wav", "n1.wav", "n2.wav"]


# ---------------------------------------------------------------------------
# Path collision detection
# ---------------------------------------------------------------------------


class TestPathCollisionDetection:
    """Tests for path collision detection."""

    def test_collision_raises_error(self) -> None:
        """Test that overlapping paths raise PathCollisionError."""
        pos = make_manifest(make_pos("shared.wav"), make_pos("p2.wav"))
        neg = make_manifest(make_neg("shared.wav"), make_neg("n2.wav"))
        with pytest.raises(PathCollisionError) as exc_info:
            merge(pos, neg)
        assert "shared.wav" in exc_info.value.paths

    def test_multiple_collisions_reports_all(self) -> None:
        """Test that multiple collisions are all reported."""
        pos = make_manifest(make_pos("a.wav"), make_pos("b.wav"))
        neg = make_manifest(make_neg("a.wav"), make_neg("b.wav"), make_neg("c.wav"))
        with pytest.raises(PathCollisionError) as exc_info:
            merge(pos, neg)
        assert set(exc_info.value.paths) == {"a.wav", "b.wav"}

    def test_no_collision_with_unique_paths(self) -> None:
        """Test that unique paths don't raise errors."""
        pos = make_manifest(make_pos("p1.wav"))
        neg = make_manifest(make_neg("n1.wav"))
        result = merge(pos, neg)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestMergerValidation:
    """Tests for merger validation."""

    def test_both_empty_raises(self) -> None:
        """Test that merging two empty manifests raises MergerError."""
        pos = make_manifest()
        neg = make_manifest()
        with pytest.raises(MergerError, match="Both positive and negative manifests are empty"):
            merge(pos, neg)

    def test_empty_positive_with_negatives(self) -> None:
        """Test merging with no positives but with negatives works."""
        pos = make_manifest()
        neg = make_manifest(make_neg("n1.wav"), make_neg("n2.wav"))
        result = merge(pos, neg)
        assert len(result) == 2

    def test_validate_files_missing(self, tmp_path: Path) -> None:
        """Test that file validation detects missing files."""
        pos = make_manifest(make_pos("p1.wav"))
        neg = make_manifest(make_neg("nonexistent.wav"))
        with pytest.raises(MergerValidationError) as exc_info:
            merge(pos, neg, validate_files=True)
        assert "File not found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    """Tests that metadata is preserved through merge."""

    def test_preserves_voice_metadata(self) -> None:
        """Test that voice metadata is preserved."""
        pos_entry = ManifestEntry(
            path="p1.wav",
            label=1,
            text="hey marvin",
            voice="af_sarah",
            duration_ms=1500,
            metadata={"source": "tts"},
        )
        pos = make_manifest(pos_entry)
        neg = make_manifest(make_neg("n1.wav"))
        result = merge(pos, neg)
        merged_entry = next(e for e in result if e.label == 1)
        assert merged_entry.voice == "af_sarah"
        assert merged_entry.metadata == {"source": "tts"}

    def test_preserves_all_entry_fields(self) -> None:
        """Test that all entry fields are preserved."""
        pos_entry = ManifestEntry(
            path="p1.wav",
            label=1,
            text="custom text",
            voice="bf_emma",
            duration_ms=2000,
            sample_rate=22050,
            metadata={"key": "value"},
        )
        pos = make_manifest(pos_entry)
        neg = make_manifest(make_neg("n1.wav"))
        result = merge(pos, neg)
        merged_entry = next(e for e in result if e.label == 1)
        assert merged_entry.path == "p1.wav"
        assert merged_entry.label == 1
        assert merged_entry.text == "custom text"
        assert merged_entry.voice == "bf_emma"
        assert merged_entry.duration_ms == 2000
        assert merged_entry.sample_rate == 22050
        assert merged_entry.metadata == {"key": "value"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMergerEdgeCases:
    """Tests for edge cases in merge operations."""

    def test_single_pos_single_neg(self) -> None:
        """Test merge with single entry of each type."""
        pos = make_manifest(make_pos("p.wav"))
        neg = make_manifest(make_neg("n.wav"))
        result = merge(pos, neg, ratio=1.0)
        assert len(result) == 2
        assert sum(1 for e in result if e.label == 1) == 1
        assert sum(1 for e in result if e.label == 0) == 1

    def test_ratio_fractional(self) -> None:
        """Test fractional ratio (0.5 means half as many negatives as positives)."""
        pos = make_manifest(*[make_pos(f"p{i}.wav") for i in range(4)])
        neg = make_manifest(*[make_neg(f"n{i}.wav") for i in range(10)])
        result = merge(pos, neg, ratio=0.5)
        pos_count = sum(1 for e in result if e.label == 1)
        neg_count = sum(1 for e in result if e.label == 0)
        assert pos_count == 4
        assert neg_count == 2  # int(4 * 0.5) = 2

    def test_ratio_zero_gives_no_negatives(self) -> None:
        """Test ratio=0 gives only positives."""
        pos = make_manifest(make_pos("p1.wav"), make_pos("p2.wav"))
        neg = make_manifest(make_neg("n1.wav"), make_neg("n2.wav"))
        result = merge(pos, neg, ratio=0.0)
        pos_count = sum(1 for e in result if e.label == 1)
        neg_count = sum(1 for e in result if e.label == 0)
        assert pos_count == 2
        assert neg_count == 0

    def test_merge_does_not_modify_original(self) -> None:
        """Test that merge doesn't modify the input manifests."""
        pos = make_manifest(make_pos("p1.wav"))
        neg = make_manifest(make_neg("n1.wav"))
        merge(pos, neg, ratio=1.0, seed=42)
        assert len(pos) == 1
        assert len(neg) == 1

    def test_large_ratio(self) -> None:
        """Test large ratio with limited negatives."""
        pos = make_manifest(make_pos("p1.wav"))
        neg = make_manifest(make_neg("n1.wav"), make_neg("n2.wav"))
        res = merge_with_result(pos, neg, ratio=10.0)
        # Can't get 10 negatives from 2
        assert len(res.warnings) == 1
        assert "Not enough negatives" in res.warnings[0]
        assert res.positive_count == 1
        assert res.negative_count == 2
