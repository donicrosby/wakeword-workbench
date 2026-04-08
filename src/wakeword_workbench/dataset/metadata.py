"""Dataset metadata management with JSONL support."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ManifestError(Exception):
    """Base exception for manifest operations."""

    pass


class ManifestValidationError(ManifestError):
    """Raised when manifest validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


@dataclass
class ManifestEntry:
    """A single entry in a dataset manifest."""

    path: str
    label: int
    text: str
    voice: str | None = None
    backend: str | None = None
    duration_ms: int = 0
    sample_rate: int = 16000
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate entry fields."""
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label}")
        if self.duration_ms < 0:
            raise ValueError(f"duration_ms must be non-negative, got {self.duration_ms}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")


class Manifest:
    """A collection of manifest entries with I/O and validation support."""

    def __init__(self, entries: list[ManifestEntry] | None = None) -> None:
        """Initialize manifest with optional entries.

        Args:
            entries: Initial list of manifest entries.
        """
        self._entries: list[ManifestEntry] = list(entries) if entries else []

    def add(self, entry: ManifestEntry) -> None:
        """Add a single entry to the manifest.

        Args:
            entry: The manifest entry to add.
        """
        self._entries.append(entry)

    def extend(self, entries: list[ManifestEntry]) -> None:
        """Add multiple entries to the manifest.

        Args:
            entries: List of manifest entries to add.
        """
        self._entries.extend(entries)

    def validate(self) -> list[str]:
        """Validate all entries in the manifest.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        for i, entry in enumerate(self._entries):
            # Check required fields present
            if not entry.path:
                errors.append(f"Entry {i}: path is required")
            # Check path is relative (not absolute)
            elif Path(entry.path).is_absolute():
                errors.append(f"Entry {i}: path must be relative, got absolute path '{entry.path}'")

            if entry.label not in (0, 1):
                errors.append(f"Entry {i}: label must be 0 or 1, got {entry.label}")

            if entry.duration_ms <= 0:
                errors.append(f"Entry {i}: duration_ms must be positive, got {entry.duration_ms}")

            if entry.sample_rate <= 0:
                errors.append(f"Entry {i}: sample_rate must be positive, got {entry.sample_rate}")

        return errors

    def save(self, path: Path | str) -> None:
        """Save manifest to a JSONL file.

        Args:
            path: Path to save the JSONL file.

        Raises:
            ManifestValidationError: If validation fails.
        """
        path = Path(path)
        errors = self.validate()
        if errors:
            raise ManifestValidationError(errors)

        with open(path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                # Convert to dict, excluding metadata if empty to keep file clean
                entry_dict = {
                    "path": entry.path,
                    "label": entry.label,
                    "text": entry.text,
                }
                if entry.voice is not None:
                    entry_dict["voice"] = entry.voice
                if entry.backend is not None:
                    entry_dict["backend"] = entry.backend
                if entry.duration_ms > 0:
                    entry_dict["duration_ms"] = entry.duration_ms
                if entry.sample_rate != 16000:
                    entry_dict["sample_rate"] = entry.sample_rate
                if entry.metadata:
                    entry_dict["metadata"] = entry.metadata
                f.write(json.dumps(entry_dict, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> Manifest:
        """Load manifest from a JSONL file.

        Args:
            path: Path to the JSONL file.

        Returns:
            A new Manifest instance with loaded entries.

        Raises:
            ManifestError: If the file cannot be read or parsed.
        """
        path = Path(path)
        if not path.exists():
            raise ManifestError(f"Manifest file not found: {path}")

        entries: list[ManifestEntry] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = ManifestEntry(
                            path=data.get("path", ""),
                            label=data.get("label", 0),
                            text=data.get("text", ""),
                            voice=data.get("voice"),
                            backend=data.get("backend"),
                            duration_ms=data.get("duration_ms", 0),
                            sample_rate=data.get("sample_rate", 16000),
                            metadata=data.get("metadata", {}),
                        )
                        entries.append(entry)
                    except (json.JSONDecodeError, ValueError) as e:
                        raise ManifestError(f"Failed to parse line {line_num}: {e}") from e
        except OSError as e:
            raise ManifestError(f"Failed to read manifest file: {e}") from e

        return cls(entries)

    def merge(self, other: Manifest) -> Manifest:
        """Merge two manifests.

        Args:
            other: Another manifest to merge with this one.

        Returns:
            A new Manifest containing entries from both manifests.
        """
        combined = list(self._entries) + list(other._entries)
        return Manifest(combined)

    def __len__(self) -> int:
        """Return the number of entries in the manifest."""
        return len(self._entries)

    def __iter__(self) -> Iterator[ManifestEntry]:
        """Return an iterator over manifest entries."""
        return iter(self._entries)
