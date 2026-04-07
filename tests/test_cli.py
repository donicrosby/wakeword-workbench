"""Tests for CLI module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from wakeword_workbench.cli import app

runner = CliRunner()


class TestCliHelp:
    """Tests for CLI help output."""

    def test_cli_help(self) -> None:
        """Test that CLI shows help when called with --help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "WakeWord Workbench" in result.stdout
        assert "run" in result.stdout
        assert "validate" in result.stdout


class TestCliVersion:
    """Tests for CLI version command."""

    def test_cli_version(self) -> None:
        """Test that version command shows correct version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout


class TestValidateCommand:
    """Tests for validate command."""

    def test_validate_command_missing_file(self) -> None:
        """Test validate command fails when config file doesn't exist."""
        result = runner.invoke(app, ["validate", "/nonexistent/config.yaml"])
        assert result.exit_code == 2
        assert "does not exist" in result.stdout.lower() or "not found" in result.stdout.lower()

    def test_validate_command_with_valid_config(self, valid_config_file: Path) -> None:
        """Test validate command succeeds with valid config file."""
        result = runner.invoke(app, ["validate", str(valid_config_file)])
        assert result.exit_code == 0
        assert "Config validation passed" in result.stdout

    def test_validate_command_with_invalid_config(self, tmp_path: Path) -> None:
        """Test validate command fails with invalid config content."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content:")
        result = runner.invoke(app, ["validate", str(invalid_config)])
        assert result.exit_code == 2


class TestRunCommand:
    """Tests for run command."""

    def test_run_command_missing_file(self) -> None:
        """Test run command fails when config file doesn't exist."""
        result = runner.invoke(app, ["run", "/nonexistent/config.yaml"])
        assert result.exit_code == 2
        assert "does not exist" in result.stdout.lower() or "not found" in result.stdout.lower()

    def test_run_command_with_valid_config(self, valid_config_file: Path) -> None:
        """Test run command works with valid config file."""
        from wakeword_workbench.dataset.generator import GenerationResult

        # Mock the generation result
        mock_result = MagicMock(spec=GenerationResult)
        mock_result.total_positives = 1000
        mock_result.total_negatives = 5000
        mock_result.total_entries = 6000
        mock_result.actual_ratio = 5.0
        mock_result.output_dir = "/fake/output"
        mock_result.generation_time_seconds = 10.5
        mock_result.has_warnings.return_value = False
        mock_result.summary.return_value = {
            "total_entries": 6000,
            "positives": 1000,
            "negatives": 5000,
            "ratio": 5.0,
            "splits": {"train": 4200, "val": 900, "test": 900},
            "output_dir": "/fake/output",
            "warnings": 0,
        }

        with patch("wakeword_workbench.cli.DatasetGenerator") as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator.generate.return_value = mock_result
            mock_generator_class.return_value = mock_generator

            result = runner.invoke(app, ["run", str(valid_config_file)])

        assert result.exit_code == 0
        assert "Generation Results" in result.stdout or "completed successfully" in result.stdout
