"""Tests for metadata module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wakeword_workbench.dataset.metadata import (
    Manifest,
    ManifestEntry,
    ManifestError,
    ManifestValidationError,
)


class TestManifestEntry:
    """Tests for ManifestEntry dataclass."""

    def test_create_minimal_entry(self) -> None:
        """Test creating entry with only required fields."""
        entry = ManifestEntry(path="audio.wav", label=1, text="hey marvin")
        assert entry.path == "audio.wav"
        assert entry.label == 1
        assert entry.text == "hey marvin"
        assert entry.voice is None
        assert entry.duration_ms == 0
        assert entry.sample_rate == 16000
        assert entry.metadata == {}

    def test_create_entry_with_all_fields(self) -> None:
        """Test creating entry with all fields."""
        entry = ManifestEntry(
            path="audio.wav",
            label=1,
            text="hey marvin",
            voice="af_sarah",
            duration_ms=1500,
            sample_rate=16000,
            metadata={"source": "tts"},
        )
        assert entry.voice == "af_sarah"
        assert entry.duration_ms == 1500
        assert entry.metadata == {"source": "tts"}

    def test_invalid_label_raises(self) -> None:
        """Test that invalid label raises ValueError."""
        with pytest.raises(ValueError, match="label must be 0 or 1"):
            ManifestEntry(path="audio.wav", label=2, text="test")

    def test_negative_duration_raises(self) -> None:
        """Test that negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_ms must be non-negative"):
            ManifestEntry(path="audio.wav", label=1, text="test", duration_ms=-1)

    def test_invalid_sample_rate_raises(self) -> None:
        """Test that invalid sample rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            ManifestEntry(path="audio.wav", label=1, text="test", sample_rate=0)


class TestManifest:
    """Tests for Manifest class."""

    def test_empty_manifest(self) -> None:
        """Test creating empty manifest."""
        manifest = Manifest()
        assert len(manifest) == 0

    def test_manifest_from_list(self) -> None:
        """Test creating manifest from list of entries."""
        entries = [
            ManifestEntry(path="a.wav", label=1, text="test 1"),
            ManifestEntry(path="b.wav", label=0, text="test 2"),
        ]
        manifest = Manifest(entries)
        assert len(manifest) == 2

    def test_add_entry(self) -> None:
        """Test adding single entry to manifest."""
        manifest = Manifest()
        entry = ManifestEntry(path="audio.wav", label=1, text="test")
        manifest.add(entry)
        assert len(manifest) == 1

    def test_extend_entries(self) -> None:
        """Test extending manifest with multiple entries."""
        manifest = Manifest()
        entries = [
            ManifestEntry(path="a.wav", label=1, text="test 1"),
            ManifestEntry(path="b.wav", label=0, text="test 2"),
        ]
        manifest.extend(entries)
        assert len(manifest) == 2

    def test_iteration(self) -> None:
        """Test iterating over manifest entries."""
        entries = [
            ManifestEntry(path="a.wav", label=1, text="test 1"),
            ManifestEntry(path="b.wav", label=0, text="test 2"),
        ]
        manifest = Manifest(entries)
        result = list(manifest)
        assert result == entries

    def test_len(self) -> None:
        """Test manifest length."""
        manifest = Manifest()
        assert len(manifest) == 0
        manifest.add(ManifestEntry(path="a.wav", label=1, text="test"))
        assert len(manifest) == 1


class TestManifestValidation:
    """Tests for manifest validation."""

    def test_validate_empty_manifest(self) -> None:
        """Test validation of empty manifest."""
        manifest = Manifest()
        errors = manifest.validate()
        assert errors == []

    def test_validate_valid_manifest(self) -> None:
        """Test validation of valid manifest."""
        manifest = Manifest(
            [
                ManifestEntry(path="a.wav", label=1, text="test", duration_ms=1000),
                ManifestEntry(path="b.wav", label=0, text="test", duration_ms=500),
            ]
        )
        errors = manifest.validate()
        assert errors == []

    def test_validate_absolute_path_fails(self) -> None:
        """Test validation catches absolute paths."""
        manifest = Manifest(
            [ManifestEntry(path="/absolute/path.wav", label=1, text="test", duration_ms=1000)]
        )
        errors = manifest.validate()
        assert len(errors) == 1
        assert "path must be relative" in errors[0]

    def test_validate_invalid_label_fails(self) -> None:
        """Test that creating entry with invalid label raises ValueError."""
        with pytest.raises(ValueError, match="label must be 0 or 1"):
            ManifestEntry(path="a.wav", label=2, text="test", duration_ms=1000)

    def test_validate_zero_duration_fails(self) -> None:
        """Test validation catches zero duration on save."""
        # Entry can be created with duration_ms=0
        entry = ManifestEntry(path="a.wav", label=1, text="test", duration_ms=0)
        manifest = Manifest([entry])
        # But validation fails when saving
        errors = manifest.validate()
        assert len(errors) == 1
        assert "duration_ms must be positive" in errors[0]

    def test_validate_multiple_errors(self) -> None:
        """Test validation catches multiple errors in saved manifest."""
        # Create manifest with entries that will fail validation
        # First entry has invalid label (caught by __post_init__)
        # Second entry has negative duration (caught by __post_init__)
        # Third entry has absolute path and zero duration (caught by validate)
        manifest = Manifest()
        manifest.add(
            ManifestEntry(path="relative.wav", label=0, text="test", duration_ms=1000)
        )  # valid
        manifest.add(
            ManifestEntry(path="/absolute.wav", label=1, text="test", duration_ms=0)
        )  # fails: absolute path + zero duration

        errors = manifest.validate()
        assert len(errors) == 2  # absolute path + zero duration


class TestManifestSaveLoad:
    """Tests for manifest save/load operations."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Test saving and loading manifest preserves data."""
        entries = [
            ManifestEntry(
                path="positive/hey_marvin.wav",
                label=1,
                text="hey marvin",
                voice="af_sarah",
                duration_ms=1500,
            ),
            ManifestEntry(
                path="negative/background_noise.wav",
                label=0,
                text="",
                duration_ms=2000,
            ),
        ]
        manifest = Manifest(entries)
        path = tmp_path / "manifest.jsonl"
        manifest.save(path)

        # Load and verify
        loaded = Manifest.load(path)
        assert len(loaded) == 2
        assert loaded._entries[0].path == "positive/hey_marvin.wav"
        assert loaded._entries[0].label == 1
        assert loaded._entries[0].text == "hey marvin"
        assert loaded._entries[0].voice == "af_sarah"
        assert loaded._entries[0].duration_ms == 1500
        assert loaded._entries[1].label == 0

    def test_save_omits_optional_defaults(self, tmp_path: Path) -> None:
        """Test that save omits optional fields with default values."""
        entry = ManifestEntry(path="a.wav", label=1, text="test", duration_ms=1000)
        manifest = Manifest([entry])
        path = tmp_path / "manifest.jsonl"
        manifest.save(path)

        # Check file content
        content = path.read_text()
        data = json.loads(content.strip())
        assert "voice" not in data
        assert "sample_rate" not in data
        assert "metadata" not in data

    def test_save_includes_non_default_values(self, tmp_path: Path) -> None:
        """Test that save includes non-default optional fields."""
        entry = ManifestEntry(
            path="a.wav",
            label=1,
            text="test",
            voice="voice1",
            duration_ms=1000,
            sample_rate=22050,
            metadata={"key": "value"},
        )
        manifest = Manifest([entry])
        path = tmp_path / "manifest.jsonl"
        manifest.save(path)

        content = path.read_text()
        data = json.loads(content.strip())
        assert data["voice"] == "voice1"
        assert data["duration_ms"] == 1000
        assert data["sample_rate"] == 22050
        assert data["metadata"] == {"key": "value"}

    def test_save_validation_fails_for_invalid(self, tmp_path: Path) -> None:
        """Test that save raises error for invalid manifest."""
        manifest = Manifest(
            [ManifestEntry(path="/abs.wav", label=1, text="test", duration_ms=1000)]
        )
        path = tmp_path / "manifest.jsonl"
        with pytest.raises(ManifestValidationError) as exc_info:
            manifest.save(path)
        assert "path must be relative" in str(exc_info.value)

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Test that loading nonexistent file raises error."""
        path = tmp_path / "nonexistent.jsonl"
        with pytest.raises(ManifestError, match="not found"):
            Manifest.load(path)

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        """Test that loading invalid JSON raises error."""
        path = tmp_path / "invalid.jsonl"
        path.write_text("not valid json\n")
        with pytest.raises(ManifestError, match="Failed to parse line 1"):
            Manifest.load(path)

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Test loading empty file creates empty manifest."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        manifest = Manifest.load(path)
        assert len(manifest) == 0

    def test_load_file_with_blank_lines(self, tmp_path: Path) -> None:
        """Test loading file with blank lines ignores them."""
        path = tmp_path / "manifest.jsonl"
        path.write_text(
            '{"path": "a.wav", "label": 1, "text": "test"}\n\n{"path": "b.wav", "label": 0, "text": "test2"}\n\n'
        )
        manifest = Manifest.load(path)
        assert len(manifest) == 2


class TestManifestMerge:
    """Tests for manifest merge operations."""

    def test_merge_two_manifests(self) -> None:
        """Test merging two manifests."""
        manifest1 = Manifest(
            [
                ManifestEntry(path="a.wav", label=1, text="test 1"),
            ]
        )
        manifest2 = Manifest(
            [
                ManifestEntry(path="b.wav", label=0, text="test 2"),
                ManifestEntry(path="c.wav", label=1, text="test 3"),
            ]
        )
        merged = manifest1.merge(manifest2)
        assert len(merged) == 3
        assert [e.path for e in merged] == ["a.wav", "b.wav", "c.wav"]

    def test_merge_preserves_original(self) -> None:
        """Test that merge doesn't modify original manifests."""
        manifest1 = Manifest([ManifestEntry(path="a.wav", label=1, text="test")])
        manifest2 = Manifest([ManifestEntry(path="b.wav", label=0, text="test")])
        manifest1.merge(manifest2)
        assert len(manifest1) == 1
        assert len(manifest2) == 1

    def test_merge_with_empty_manifest(self) -> None:
        """Test merging with empty manifest."""
        manifest1 = Manifest([ManifestEntry(path="a.wav", label=1, text="test")])
        manifest2 = Manifest()
        merged = manifest1.merge(manifest2)
        assert len(merged) == 1

    def test_merge_two_empty_manifests(self) -> None:
        """Test merging two empty manifests."""
        manifest1 = Manifest()
        manifest2 = Manifest()
        merged = manifest1.merge(manifest2)
        assert len(merged) == 0


class TestManifestJsonlFormat:
    """Tests for JSONL format compliance."""

    def test_jsonl_one_object_per_line(self, tmp_path: Path) -> None:
        """Test that output has one JSON object per line."""
        entries = [
            ManifestEntry(path="a.wav", label=1, text="test 1", duration_ms=1000),
            ManifestEntry(path="b.wav", label=0, text="test 2", duration_ms=2000),
        ]
        manifest = Manifest(entries)
        path = tmp_path / "manifest.jsonl"
        manifest.save(path)

        content = path.read_text()
        lines = [line for line in content.strip().split("\n") if line]
        assert len(lines) == 2

        for line in lines:
            json.loads(line)  # Should not raise

    def test_jsonl_parseable_by_jsonl_parser(self, tmp_path: Path) -> None:
        """Test that saved file is parseable by standard JSON parser."""
        entries = [
            ManifestEntry(path="test.wav", label=1, text="hello", voice="v1", duration_ms=1000),
        ]
        manifest = Manifest(entries)
        path = tmp_path / "manifest.jsonl"
        manifest.save(path)

        # Read as JSONL and verify each line is valid JSON
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Line {i} is not valid JSON: {e}")
