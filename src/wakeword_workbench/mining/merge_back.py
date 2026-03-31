"""Dataset merge-back utility for adding extracted hard negatives to training dataset."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry, ManifestError


@dataclass
class MergeResult:
    """Result of a merge-back operation."""

    added_count: int
    total_count: int
    backup_path: Path | None
    errors: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MergeBackError(Exception):
    """Raised when merge-back operation fails."""

    pass


def add_to_training(
    new_negatives_manifest: Path | str,
    training_manifest: Path | str,
    backup: bool = True,
) -> MergeResult:
    """Add newly extracted hard negatives to the training dataset.

    Validates new negatives before adding, creates a timestamped backup
    of the training manifest, and appends valid new negatives.

    Args:
        new_negatives_manifest: Path to JSONL manifest of new negative samples.
        training_manifest: Path to existing training manifest (JSONL).
        backup: Whether to create a timestamped backup before modifying.

    Returns:
        MergeResult with statistics about the merge operation.

    Raises:
        MergeBackError: If validation fails or operation cannot complete.
    """
    errors: list[str] = []
    added_count = 0
    backup_path: Path | None = None

    new_negatives_path = Path(new_negatives_manifest)
    training_path = Path(training_manifest)

    # Validate input paths exist
    if not new_negatives_path.exists():
        raise MergeBackError(f"New negatives manifest not found: {new_negatives_path}")

    if not training_path.exists():
        raise MergeBackError(f"Training manifest not found: {training_path}")

    # Load existing training manifest
    try:
        training = Manifest.load(training_path)
    except ManifestError as e:
        raise MergeBackError(f"Failed to load training manifest: {e}") from e

    # Build set of existing paths for duplicate detection
    existing_paths: set[str] = {entry.path for entry in training}

    # Load and validate new negatives
    new_entries: list[ManifestEntry] = []
    try:
        with open(new_negatives_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                # Extract required path
                file_path = data.get("path", "")
                if not file_path:
                    errors.append(f"Line {line_num}: Missing 'path' field")
                    continue

                # Check if file exists
                audio_file = Path(file_path)
                if not audio_file.exists():
                    # Try relative to the manifest's directory
                    manifest_dir = new_negatives_path.parent
                    audio_file = manifest_dir / file_path
                    if not audio_file.exists():
                        errors.append(f"Line {line_num}: Audio file not found: {file_path}")
                        continue

                # Check for duplicates
                if file_path in existing_paths:
                    errors.append(f"Line {line_num}: Duplicate entry skipped: {file_path}")
                    continue

                # Validate label
                label = data.get("label", 0)
                if label not in (0, 1):
                    errors.append(f"Line {line_num}: Invalid label {label}, expected 0 or 1")
                    continue

                # Create manifest entry
                try:
                    entry = ManifestEntry(
                        path=file_path,
                        label=label,
                        text=data.get("text", ""),
                        voice=data.get("voice"),
                        duration_ms=data.get("duration_ms", 0),
                        sample_rate=data.get("sample_rate", 16000),
                        metadata=data.get("metadata", {}),
                    )
                    new_entries.append(entry)
                    existing_paths.add(file_path)
                except ValueError as e:
                    errors.append(f"Line {line_num}: Invalid entry - {e}")
                    continue

    except OSError as e:
        raise MergeBackError(f"Failed to read new negatives manifest: {e}") from e

    # Abort if validation produced errors that make merging unsafe
    # (missing files are warnings, not blockers)
    critical_errors = [e for e in errors if "Audio file not found" in e]
    if len(critical_errors) > len(new_entries):
        raise MergeBackError(f"Too many missing files ({len(critical_errors)}), aborting merge")

    # Abort if no valid entries to add
    if not new_entries:
        return MergeResult(
            added_count=0,
            total_count=len(training),
            backup_path=None,
            errors=errors,
        )

    # Create backup if requested
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"training_manifest.{timestamp}.backup.jsonl"
        backup_path = training_path.parent / backup_filename

        try:
            shutil.copy2(training_path, backup_path)
            backup_path = backup_path.resolve()
        except OSError as e:
            raise MergeBackError(f"Failed to create backup: {e}") from e

    # Add new entries to training manifest
    training.extend(new_entries)
    added_count = len(new_entries)

    # Save updated manifest
    try:
        training.save(training_path)
    except ManifestError as e:
        # Restore from backup if save fails
        if backup and backup_path and backup_path.exists():
            shutil.copy2(backup_path, training_path)
        raise MergeBackError(f"Failed to save training manifest: {e}") from e

    return MergeResult(
        added_count=added_count,
        total_count=len(training),
        backup_path=backup_path,
        errors=errors,
    )
