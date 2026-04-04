"""Tests for mine CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from wakeword_workbench.cli import app

runner = CliRunner()


class MockExtractedClip:
    """Mock ExtractedClip for testing."""

    def __init__(
        self,
        clip_path: Path,
        original_path: Path,
        timestamp: float,
        prediction: float,
        threshold: float,
        duration: float,
    ):
        self.clip_path = clip_path
        self.original_path = original_path
        self.timestamp = timestamp
        self.prediction = prediction
        self.threshold = threshold
        self.duration = duration


class TestMineCommandHelp:
    """Tests for mine command help output."""

    def test_mine_command_help(self) -> None:
        """Test that mine command shows help with all options."""
        result = runner.invoke(app, ["mine", "--help"])
        assert result.exit_code == 0
        assert "mine" in result.stdout
        assert "--model" in result.stdout
        assert "--audio" in result.stdout
        assert "--threshold" in result.stdout
        assert "--output" in result.stdout
        assert "hard negatives" in result.stdout.lower()


class TestMineCommandValidation:
    """Tests for mine command input validation."""

    def test_mine_command_missing_model(self, tmp_path: Path) -> None:
        """Test mine command fails when model file doesn't exist."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                "mine",
                "--model",
                str(tmp_path / "nonexistent.onnx"),
                "--audio",
                str(tmp_path / "*.wav"),
                "--output",
                str(output_dir),
            ],
        )
        # Typer validates model path with exists=True before our code runs
        # So we just check the exit code (2 = config error)
        assert result.exit_code == 2

    def test_mine_command_no_matching_audio(self, tmp_path: Path) -> None:
        """Test mine command fails when no audio files match pattern."""
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "mine",
                "--model",
                str(model_file),
                "--audio",
                str(tmp_path / "*.wav"),
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 2
        assert "no audio files found" in result.stdout.lower()

    def test_mine_command_invalid_threshold_high(self, tmp_path: Path) -> None:
        """Test mine command fails with threshold > 1.0."""
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "mine",
                "--model",
                str(model_file),
                "--audio",
                str(tmp_path / "*.wav"),
                "--output",
                str(output_dir),
                "--threshold",
                "1.5",
            ],
        )
        assert result.exit_code != 0

    def test_mine_command_invalid_threshold_low(self, tmp_path: Path) -> None:
        """Test mine command fails with threshold < 0.0."""
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "mine",
                "--model",
                str(model_file),
                "--audio",
                str(tmp_path / "*.wav"),
                "--output",
                str(output_dir),
                "--threshold",
                "-0.5",
            ],
        )
        assert result.exit_code != 0


class TestMineCommandExecution:
    """Tests for mine command execution with mocked dependencies."""

    @patch("wakeword_workbench.cli.load_onnx_model")
    @patch("wakeword_workbench.cli.process_long_audio")
    @patch("wakeword_workbench.cli.extract_false_positives")
    def test_mine_command_success(
        self,
        mock_extract: MagicMock,
        mock_process: MagicMock,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test successful mining with mocked functions."""
        # Create model file
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")

        # Create audio file
        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")

        # Create output directory
        output_dir = tmp_path / "output"

        # Mock model function
        mock_model = MagicMock(return_value=0.85)
        mock_load_model.return_value = mock_model

        # Mock process_long_audio to return predictions
        mock_prediction = MagicMock()
        mock_prediction.timestamp = 5.0
        mock_prediction.prediction = 0.85
        mock_prediction.window_start = 80000
        mock_prediction.window_end = 96000
        mock_process.return_value = [mock_prediction]

        # Mock extract_false_positives to return a clip
        clip_path = output_dir / "audio_T5.000.wav"
        mock_clip = MockExtractedClip(
            clip_path=clip_path,
            original_path=audio_file,
            timestamp=5.0,
            prediction=0.85,
            threshold=0.7,
            duration=1.0,
        )
        mock_extract.return_value = [mock_clip]

        result = runner.invoke(
            app,
            [
                "mine",
                "--model",
                str(model_file),
                "--audio",
                str(tmp_path / "*.wav"),
                "--output",
                str(output_dir),
                "--threshold",
                "0.7",
            ],
        )

        assert result.exit_code == 0
        assert "Extracted 1 clips" in result.stdout
        assert "Saved manifest" in result.stdout

        # Verify mocks were called
        mock_load_model.assert_called_once_with(model_file)
        mock_process.assert_called_once()
        mock_extract.assert_called_once()

        # Verify manifest was created
        manifest_path = output_dir / "manifest.jsonl"
        assert manifest_path.exists()

    @patch("wakeword_workbench.cli.load_onnx_model")
    @patch("wakeword_workbench.cli.process_long_audio")
    @patch("wakeword_workbench.cli.extract_false_positives")
    def test_mine_command_multiple_files(
        self,
        mock_extract: MagicMock,
        mock_process: MagicMock,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test mining with multiple audio files."""
        # Create model file
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")

        # Create multiple audio files
        audio_file1 = tmp_path / "audio1.wav"
        audio_file1.write_text("fake audio 1")
        audio_file2 = tmp_path / "audio2.wav"
        audio_file2.write_text("fake audio 2")

        output_dir = tmp_path / "output"

        # Mock model function
        mock_model = MagicMock(return_value=0.9)
        mock_load_model.return_value = mock_model

        # Mock predictions
        mock_prediction = MagicMock()
        mock_prediction.timestamp = 2.0
        mock_prediction.prediction = 0.9
        mock_prediction.window_start = 32000
        mock_prediction.window_end = 48000
        mock_process.return_value = [mock_prediction]

        # Mock clips
        mock_clip1 = MockExtractedClip(
            clip_path=output_dir / "audio1_T2.000.wav",
            original_path=audio_file1,
            timestamp=2.0,
            prediction=0.9,
            threshold=0.7,
            duration=1.0,
        )
        mock_clip2 = MockExtractedClip(
            clip_path=output_dir / "audio2_T2.000.wav",
            original_path=audio_file2,
            timestamp=2.0,
            prediction=0.9,
            threshold=0.7,
            duration=1.0,
        )
        mock_extract.side_effect = [[mock_clip1], [mock_clip2]]

        result = runner.invoke(
            app,
            [
                "mine",
                "--model",
                str(model_file),
                "--audio",
                str(tmp_path / "*.wav"),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert "Extracted 2 clips" in result.stdout
        assert mock_process.call_count == 2
        assert mock_extract.call_count == 2

    @patch("wakeword_workbench.cli.load_onnx_model")
    def test_mine_command_model_load_error(
        self,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test handling of model load errors."""
        from wakeword_workbench.mining.model_loader import ModelLoadError

        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")
        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")
        output_dir = tmp_path / "output"

        mock_load_model.side_effect = ModelLoadError("Invalid model format")

        result = runner.invoke(
            app,
            [
                "mine",
                "--model",
                str(model_file),
                "--audio",
                str(tmp_path / "*.wav"),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 1
        assert "Failed to load model" in result.stdout

    @patch("wakeword_workbench.cli.load_onnx_model")
    @patch("wakeword_workbench.cli.process_long_audio")
    def test_mine_command_processing_error(
        self,
        mock_process: MagicMock,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test handling of audio processing errors."""
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")
        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_process.side_effect = FileNotFoundError("Audio file corrupted")

        result = runner.invoke(
            app,
            [
                "mine",
                "--model",
                str(model_file),
                "--audio",
                str(tmp_path / "*.wav"),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 1
        assert "Failed to process" in result.stdout


class TestMineCommandManifest:
    """Tests for manifest generation."""

    @patch("wakeword_workbench.cli.load_onnx_model")
    @patch("wakeword_workbench.cli.process_long_audio")
    @patch("wakeword_workbench.cli.extract_false_positives")
    def test_manifest_content(
        self,
        mock_extract: MagicMock,
        mock_process: MagicMock,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that manifest contains correct entries."""
        from wakeword_workbench.dataset.metadata import Manifest

        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")
        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")
        output_dir = tmp_path / "output"

        mock_model = MagicMock(return_value=0.8)
        mock_load_model.return_value = mock_model

        mock_prediction = MagicMock()
        mock_prediction.timestamp = 3.5
        mock_prediction.prediction = 0.8
        mock_prediction.window_start = 56000
        mock_prediction.window_end = 72000
        mock_process.return_value = [mock_prediction]

        clip_path = output_dir / "audio_T3.500.wav"
        mock_clip = MockExtractedClip(
            clip_path=clip_path,
            original_path=audio_file,
            timestamp=3.5,
            prediction=0.8,
            threshold=0.7,
            duration=1.0,
        )
        mock_extract.return_value = [mock_clip]

        result = runner.invoke(
            app,
            [
                "mine",
                "--model",
                str(model_file),
                "--audio",
                str(tmp_path / "*.wav"),
                "--output",
                str(output_dir),
                "--threshold",
                "0.7",
            ],
        )

        assert result.exit_code == 0

        # Load and verify manifest
        manifest_path = output_dir / "manifest.jsonl"
        manifest = Manifest.load(manifest_path)
        assert len(manifest) == 1

        entry = list(manifest)[0]
        assert entry.label == 0  # Hard negative
        assert entry.duration_ms == 1000  # 1.0 second
        assert entry.sample_rate == 16000
        assert entry.metadata["timestamp"] == 3.5
        assert entry.metadata["prediction"] == 0.8
        assert entry.metadata["threshold"] == 0.7

    @patch("wakeword_workbench.cli.load_onnx_model")
    @patch("wakeword_workbench.cli.process_long_audio")
    @patch("wakeword_workbench.cli.extract_false_positives")
    def test_manifest_no_clips(
        self,
        mock_extract: MagicMock,
        mock_process: MagicMock,
        mock_load_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test manifest generation when no clips are extracted."""
        from wakeword_workbench.dataset.metadata import Manifest

        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")
        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")
        output_dir = tmp_path / "output"

        mock_model = MagicMock(return_value=0.3)  # Low confidence
        mock_load_model.return_value = mock_model

        mock_prediction = MagicMock()
        mock_prediction.timestamp = 1.0
        mock_prediction.prediction = 0.3
        mock_prediction.window_start = 16000
        mock_prediction.window_end = 32000
        mock_process.return_value = [mock_prediction]

        # No clips extracted (below threshold)
        mock_extract.return_value = []

        result = runner.invoke(
            app,
            [
                "mine",
                "--model",
                str(model_file),
                "--audio",
                str(tmp_path / "*.wav"),
                "--output",
                str(output_dir),
                "--threshold",
                "0.7",
            ],
        )

        assert result.exit_code == 0
        assert "Extracted 0 clips" in result.stdout

        # Manifest should still be created but empty
        manifest_path = output_dir / "manifest.jsonl"
        manifest = Manifest.load(manifest_path)
        assert len(manifest) == 0
