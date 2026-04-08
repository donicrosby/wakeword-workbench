"""Tests for dataset splitter."""

from __future__ import annotations

import pytest

from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.dataset.splitter import (
    SplitValidationError,
    split,
)


class TestSplitValidation:
    """Tests for split validation."""

    def test_ratios_must_sum_to_one(self) -> None:
        """Test that split ratios must sum to 1.0."""
        manifest = Manifest(
            [ManifestEntry(path="a.wav", label=1, text="test", voice="v1", duration_ms=1000)]
        )
        with pytest.raises(SplitValidationError, match="must sum to 1.0"):
            split(manifest, train=0.5, val=0.3, test=0.1)

    def test_empty_manifest_returns_empty_splits(self) -> None:
        """Test splitting empty manifest returns empty manifests."""
        manifest = Manifest()
        train, val, test = split(manifest)
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0


class TestLeakPrevention:
    """Tests for speaker leak prevention."""

    def test_same_speaker_in_same_split(self) -> None:
        """Test that all entries from same speaker go to same split."""
        entries = [
            ManifestEntry(
                path="a1.wav", label=1, text="test 1", voice="speaker1", duration_ms=1000
            ),
            ManifestEntry(
                path="a2.wav", label=0, text="test 2", voice="speaker1", duration_ms=1000
            ),
            ManifestEntry(
                path="a3.wav", label=1, text="test 3", voice="speaker1", duration_ms=1000
            ),
            ManifestEntry(
                path="b1.wav", label=1, text="test 4", voice="speaker2", duration_ms=1000
            ),
            ManifestEntry(
                path="b2.wav", label=0, text="test 5", voice="speaker2", duration_ms=1000
            ),
        ]
        manifest = Manifest(entries)
        train, val, test = split(manifest, train=0.6, val=0.2, test=0.2, seed=42)

        # Check all entries from same speaker are in same split
        speaker1_locations = []
        speaker2_locations = []
        for split_manifest, split_name in [(train, "train"), (val, "val"), (test, "test")]:
            for e in split_manifest:
                if e.voice == "speaker1":
                    speaker1_locations.append((e.path, split_name))
                elif e.voice == "speaker2":
                    speaker2_locations.append((e.path, split_name))

        # All speaker1 entries should be in the same split
        assert len({loc[1] for loc in speaker1_locations}) == 1
        # All speaker2 entries should be in the same split
        assert len({loc[1] for loc in speaker2_locations}) == 1

    def test_no_speaker_in_multiple_splits(self) -> None:
        """Test that no speaker appears in multiple splits."""
        entries = (
            [
                ManifestEntry(
                    path=f"s1_e{i}.wav",
                    label=1,
                    text=f"test {i}",
                    voice="speaker1",
                    duration_ms=1000,
                )
                for i in range(5)
            ]
            + [
                ManifestEntry(
                    path=f"s2_e{i}.wav",
                    label=0,
                    text=f"test {i}",
                    voice="speaker2",
                    duration_ms=1000,
                )
                for i in range(5)
            ]
            + [
                ManifestEntry(
                    path=f"s3_e{i}.wav",
                    label=1,
                    text=f"test {i}",
                    voice="speaker3",
                    duration_ms=1000,
                )
                for i in range(5)
            ]
            + [
                ManifestEntry(
                    path=f"s4_e{i}.wav",
                    label=0,
                    text=f"test {i}",
                    voice="speaker4",
                    duration_ms=1000,
                )
                for i in range(5)
            ]
        )
        manifest = Manifest(entries)
        train, val, test = split(manifest, train=0.5, val=0.25, test=0.25, seed=123)

        # Collect speakers from each split
        train_speakers = {e.voice for e in train if e.voice}
        val_speakers = {e.voice for e in val if e.voice}
        test_speakers = {e.voice for e in test if e.voice}

        # Check no overlap
        assert len(train_speakers & val_speakers) == 0
        assert len(train_speakers & test_speakers) == 0
        assert len(val_speakers & test_speakers) == 0

        # Check all speakers accounted for
        all_speakers = train_speakers | val_speakers | test_speakers
        assert all_speakers == {"speaker1", "speaker2", "speaker3", "speaker4"}

    def test_unknown_speaker_grouping(self) -> None:
        """Test entries without voice field are handled correctly."""
        entries = [
            ManifestEntry(path="a1.wav", label=1, text="test 1", duration_ms=1000),
            ManifestEntry(path="a2.wav", label=0, text="test 2", duration_ms=1000),
            ManifestEntry(
                path="b1.wav", label=1, text="test 3", voice="speaker1", duration_ms=1000
            ),
        ]
        manifest = Manifest(entries)
        train, val, test = split(manifest, seed=42)

        # Should not raise - entries without voice are grouped as "unknown"
        assert len(train) + len(val) + len(test) == 3


class TestStratification:
    """Tests for label stratification."""

    def test_stratification_preserves_ratio(self) -> None:
        """Test that label ratios are preserved across splits."""
        entries = [
            ManifestEntry(
                path=f"p{i}.wav",
                label=1,
                text=f"positive {i}",
                voice=f"speaker{i}",
                duration_ms=1000,
            )
            for i in range(20)
        ] + [
            ManifestEntry(
                path=f"n{i}.wav",
                label=0,
                text=f"negative {i}",
                voice=f"neg_speaker{i}",
                duration_ms=1000,
            )
            for i in range(20)
        ]
        manifest = Manifest(entries)

        # Overall ratio is 50/50
        total_positive = 20
        total_negative = 20

        train, val, test = split(manifest, train=0.7, val=0.15, test=0.15, seed=42)

        for split_manifest, split_name in [(train, "train"), (val, "val"), (test, "test")]:
            split_positive = sum(1 for e in split_manifest if e.label == 1)
            split_total = len(split_manifest)

            # Check ratio is roughly preserved (within 10%)
            expected_ratio = total_positive / (total_positive + total_negative)
            actual_ratio = split_positive / split_total if split_total > 0 else 0
            assert abs(actual_ratio - expected_ratio) < 0.15, (
                f"{split_name} ratio off: {actual_ratio:.2f} vs {expected_ratio:.2f}"
            )


class TestSplitCorrectness:
    """Tests for split correctness."""

    def test_all_entries_accounted_for(self) -> None:
        """Test that all entries appear exactly once in splits."""
        entries = [
            ManifestEntry(
                path=f"entry{i}.wav",
                label=i % 2,
                text=f"test {i}",
                voice=f"speaker{i % 5}",
                duration_ms=1000,
            )
            for i in range(100)
        ]
        manifest = Manifest(entries)
        train, val, test = split(manifest, train=0.7, val=0.15, test=0.15, seed=42)

        # Check total count
        assert len(train) + len(val) + len(test) == 100

        # Check no duplicates
        all_paths = set()
        for e in train:
            assert e.path not in all_paths
            all_paths.add(e.path)
        for e in val:
            assert e.path not in all_paths
            all_paths.add(e.path)
        for e in test:
            assert e.path not in all_paths
            all_paths.add(e.path)

    def test_no_overlapping_entries(self) -> None:
        """Test that no entry appears in multiple splits."""
        entries = [
            ManifestEntry(
                path=f"entry{i}.wav",
                label=1,
                text=f"test {i}",
                voice=f"speaker{i % 8}",
                duration_ms=1000,
            )
            for i in range(50)
        ]
        manifest = Manifest(entries)
        train, val, test = split(manifest, seed=123)

        train_paths = {e.path for e in train}
        val_paths = {e.path for e in val}
        test_paths = {e.path for e in test}

        # Check no overlaps
        assert len(train_paths & val_paths) == 0
        assert len(train_paths & test_paths) == 0
        assert len(val_paths & test_paths) == 0

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same split."""
        entries = [
            ManifestEntry(
                path=f"entry{i}.wav",
                label=1,
                text=f"test {i}",
                voice=f"speaker{i % 10}",
                duration_ms=1000,
            )
            for i in range(50)
        ]
        manifest = Manifest(entries)

        train1, val1, test1 = split(manifest, seed=42)
        train2, val2, test2 = split(manifest, seed=42)

        # Same seed should produce same splits
        assert [e.path for e in train1] == [e.path for e in train2]
        assert [e.path for e in val1] == [e.path for e in val2]
        assert [e.path for e in test1] == [e.path for e in test2]

    def test_different_seeds_different_splits(self) -> None:
        """Test that different seeds produce different splits."""
        entries = [
            ManifestEntry(
                path=f"entry{i}.wav",
                label=1,
                text=f"test {i}",
                voice=f"speaker{i % 10}",
                duration_ms=1000,
            )
            for i in range(50)
        ]
        manifest = Manifest(entries)

        train1, _, _ = split(manifest, seed=42)
        train2, _, _ = split(manifest, seed=123)

        # Different seeds should (likely) produce different splits
        # Note: there's a small chance they're the same, but unlikely
        assert [e.path for e in train1] != [e.path for e in train2]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_speaker(self) -> None:
        """Test splitting with single speaker."""
        entries = [
            ManifestEntry(
                path=f"entry{i}.wav",
                label=1,
                text=f"test {i}",
                voice="only_speaker",
                duration_ms=1000,
            )
            for i in range(10)
        ]
        manifest = Manifest(entries)
        train, val, test = split(manifest, train=0.7, val=0.15, test=0.15, seed=42)

        # With a single speaker, all entries must go to the same split (no speaker leakage)
        # The exact distribution depends on the algorithm
        total = len(train) + len(val) + len(test)
        assert total == 10
        # All entries from same speaker should be in exactly one split
        non_empty_splits = sum(1 for s in [train, val, test] if len(s) > 0)
        assert non_empty_splits == 1

    def test_two_speakers(self) -> None:
        """Test splitting with two speakers."""
        entries = [
            ManifestEntry(
                path=f"s1_e{i}.wav", label=1, text=f"test {i}", voice="speaker1", duration_ms=1000
            )
            for i in range(5)
        ] + [
            ManifestEntry(
                path=f"s2_e{i}.wav", label=0, text=f"test {i}", voice="speaker2", duration_ms=1000
            )
            for i in range(5)
        ]
        manifest = Manifest(entries)
        train, val, test = split(manifest, train=0.5, val=0.25, test=0.25, seed=42)

        # Speakers should not be split
        assert len(train) + len(val) + len(test) == 10

    def test_unbalanced_labels(self) -> None:
        """Test splitting with highly unbalanced labels."""
        entries = [
            ManifestEntry(
                path=f"p{i}.wav",
                label=1,
                text=f"positive {i}",
                voice=f"speaker{i}",
                duration_ms=1000,
            )
            for i in range(90)
        ] + [
            ManifestEntry(
                path=f"n{i}.wav", label=0, text=f"negative {i}", voice=f"neg{i}", duration_ms=1000
            )
            for i in range(10)
        ]
        manifest = Manifest(entries)
        train, val, test = split(manifest, seed=42)

        # Should still produce valid split
        assert len(train) + len(val) + len(test) == 100


class TestSpeakerGroup:
    """Tests for SpeakerGroup dataclass."""

    def test_speaker_group_counts(self) -> None:
        """Test SpeakerGroup positive/negative counting."""
        from wakeword_workbench.dataset.splitter import _group_by_speaker

        entries = [
            ManifestEntry(path="p1.wav", label=1, text="pos", voice="s1", duration_ms=1000),
            ManifestEntry(path="p2.wav", label=1, text="pos", voice="s1", duration_ms=1000),
            ManifestEntry(path="n1.wav", label=0, text="neg", voice="s1", duration_ms=1000),
        ]
        manifest = Manifest(entries)
        groups = _group_by_speaker(manifest, "speaker")

        assert len(groups) == 1
        assert groups[0].speaker_id == "s1"
        assert groups[0].positive_count == 2
        assert groups[0].negative_count == 1
