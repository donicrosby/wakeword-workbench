"""Tests for merge CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from wakeword_workbench.cli import app

runner = CliRunner()


class TestMergeCommandHelp:
    """Tests for merge command help output."""

    def test_merge_command_help(self) -> None:
        """Test that merge command shows help with all options."""
        result = runner.invoke(app, ["merge", "--help"])
        assert result.exit_code == 0
        assert "merge" in result.stdout
        assert "--source" in result.stdout
        assert "--target" in result.stdout
        assert "--backup" in result.stdout
        assert "hard negatives" in result.stdout.lower()


class TestMergeCommandValidation:
    """Tests for merge command input validation."""

    def test_merge_command_source_not_jsonl(self, tmp_path: Path) -> None:
        """Test merge fails when source is not a JSONL file."""
        source = tmp_path / "source.txt"
        source.write_text("not jsonl")
        target = tmp_path / "target.jsonl"
        target.write_text('{"path": "test.wav", "label": 0}\n')

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(target),
            ],
        )
        assert result.exit_code == 2
        assert "must be a JSONL file" in result.stdout

    def test_merge_command_target_not_jsonl(self, tmp_path: Path) -> None:
        """Test merge fails when target is not a JSONL file."""
        source = tmp_path / "source.jsonl"
        source.write_text('{"path": "test.wav", "label": 0}\n')
        target = tmp_path / "target.txt"
        target.write_text("not jsonl")

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(target),
            ],
        )
        assert result.exit_code == 2
        assert "must be a JSONL file" in result.stdout

    def test_merge_command_missing_target(self, tmp_path: Path) -> None:
        """Test merge fails when target manifest doesn't exist."""
        source = tmp_path / "source.jsonl"
        source.write_text('{"path": "test.wav", "label": 0}\n')
        target = tmp_path / "nonexistent.jsonl"

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(target),
            ],
        )
        assert result.exit_code == 2
        assert "not found" in result.stdout.lower()


class TestMergeCommandExecution:
    """Tests for merge command execution with mocked dependencies."""

    @patch("wakeword_workbench.cli.add_to_training")
    def test_merge_command_success(self, mock_add: MagicMock, tmp_path: Path) -> None:
        """Test successful merge without backup."""
        source = tmp_path / "source.jsonl"
        source.write_text('{"path": "neg.wav", "label": 0}\n')
        target = tmp_path / "target.jsonl"
        target.write_text('{"path": "train.wav", "label": 1}\n')

        # Mock MergeResult
        mock_result = MagicMock()
        mock_result.added_count = 1
        mock_result.total_count = 3
        mock_result.backup_path = None
        mock_result.errors = []
        mock_add.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(target),
            ],
        )

        assert result.exit_code == 0
        assert "Source entries" in result.stdout
        assert "Total in target" in result.stdout
        assert "completed successfully" in result.stdout

        mock_add.assert_called_once_with(
            new_negatives_manifest=source,
            training_manifest=target,
            backup=False,
        )

    @patch("wakeword_workbench.cli.add_to_training")
    def test_merge_command_with_backup(self, mock_add: MagicMock, tmp_path: Path) -> None:
        """Test merge with backup flag creates backup."""
        source = tmp_path / "source.jsonl"
        source.write_text('{"path": "neg.wav", "label": 0}\n')
        target = tmp_path / "target.jsonl"
        target.write_text('{"path": "train.wav", "label": 1}\n')

        backup_path = tmp_path / "training_manifest.20240101_120000.backup.jsonl"
        mock_result = MagicMock()
        mock_result.added_count = 2
        mock_result.total_count = 5
        mock_result.backup_path = backup_path
        mock_result.errors = []
        mock_add.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(target),
                "--backup",
            ],
        )

        assert result.exit_code == 0
        assert "Backup created" in result.stdout
        assert str(backup_path) in result.stdout

        mock_add.assert_called_once_with(
            new_negatives_manifest=source,
            training_manifest=target,
            backup=True,
        )

    @patch("wakeword_workbench.cli.add_to_training")
    def test_merge_command_with_warnings(self, mock_add: MagicMock, tmp_path: Path) -> None:
        """Test merge displays warnings when present."""
        source = tmp_path / "source.jsonl"
        source.write_text('{"path": "neg.wav", "label": 0}\n')
        target = tmp_path / "target.jsonl"
        target.write_text('{"path": "train.wav", "label": 1}\n')

        mock_result = MagicMock()
        mock_result.added_count = 1
        mock_result.total_count = 2
        mock_result.backup_path = None
        mock_result.errors = [
            "Line 2: Audio file not found: missing.wav",
            "Line 3: Duplicate entry skipped: dup.wav",
        ]
        mock_add.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(target),
            ],
        )

        assert result.exit_code == 0
        assert "Warnings" in result.stdout
        assert "Audio file not found" in result.stdout
        assert "Duplicate entry" in result.stdout

    @patch("wakeword_workbench.cli.add_to_training")
    def test_merge_command_many_warnings_truncated(
        self, mock_add: MagicMock, tmp_path: Path
    ) -> None:
        """Test merge truncates warnings when more than 5."""
        source = tmp_path / "source.jsonl"
        source.write_text('{"path": "neg.wav", "label": 0}\n')
        target = tmp_path / "target.jsonl"
        target.write_text('{"path": "train.wav", "label": 1}\n')

        mock_result = MagicMock()
        mock_result.added_count = 1
        mock_result.total_count = 2
        mock_result.backup_path = None
        mock_result.errors = [f"Line {i}: Error message" for i in range(1, 8)]
        mock_add.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(target),
            ],
        )

        assert result.exit_code == 0
        assert "Warnings (7)" in result.stdout
        assert "and 2 more" in result.stdout


class TestMergeCommandErrorHandling:
    """Tests for merge command error handling."""

    @patch("wakeword_workbench.cli.add_to_training")
    def test_merge_command_merge_error(self, mock_add: MagicMock, tmp_path: Path) -> None:
        """Test handling of MergeBackError."""
        from wakeword_workbench.mining.merge_back import MergeBackError

        source = tmp_path / "source.jsonl"
        source.write_text('{"path": "neg.wav", "label": 0}\n')
        target = tmp_path / "target.jsonl"
        target.write_text('{"path": "train.wav", "label": 1}\n')

        mock_add.side_effect = MergeBackError("Failed to read manifest")

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(target),
            ],
        )

        assert result.exit_code == 1
        assert "Failed to read manifest" in result.stdout

    @patch("wakeword_workbench.cli.add_to_training")
    def test_merge_command_too_many_missing_files(
        self, mock_add: MagicMock, tmp_path: Path
    ) -> None:
        """Test handling when too many files are missing."""
        from wakeword_workbench.mining.merge_back import MergeBackError

        source = tmp_path / "source.jsonl"
        source.write_text('{"path": "neg.wav", "label": 0}\n')
        target = tmp_path / "target.jsonl"
        target.write_text('{"path": "train.wav", "label": 1}\n')

        mock_add.side_effect = MergeBackError("Too many missing files (5), aborting merge")

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(target),
            ],
        )

        assert result.exit_code == 1
        assert "Too many missing files" in result.stdout


class TestMergeCommandEdgeCases:
    """Tests for merge command edge cases."""

    @patch("wakeword_workbench.cli.add_to_training")
    def test_merge_command_no_entries_added(self, mock_add: MagicMock, tmp_path: Path) -> None:
        """Test merge when no new entries are added."""
        source = tmp_path / "source.jsonl"
        source.write_text('{"path": "neg.wav", "label": 0}\n')
        target = tmp_path / "target.jsonl"
        target.write_text('{"path": "train.wav", "label": 1}\n')

        mock_result = MagicMock()
        mock_result.added_count = 0
        mock_result.total_count = 5
        mock_result.backup_path = None
        mock_result.errors = ["All entries were duplicates"]
        mock_add.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(target),
            ],
        )

        assert result.exit_code == 0
        assert "Source entries   : 0" in result.stdout
        assert "completed successfully" in result.stdout
