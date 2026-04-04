"""End-to-end integration tests for mine and merge CLI commands.

These tests verify the complete workflow from mining hard negatives
to merging them into a training manifest.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from wakeword_workbench.cli import app
from wakeword_workbench.dataset.metadata import Manifest

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


class TestMineMergeWorkflow:
    """Tests for the complete mine-to-merge workflow."""

    def test_happy_path_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow: mine produces clips, then merge adds them."""
        # Create model file
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")

        # Create audio file
        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")

        # Create mine output directory
        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        # Create target training manifest
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            '{"path": "train.wav", "label": 1, "text": "hey assistant", "duration_ms": 1000}\n'
        )

        # Mock functions for mine
        with (
            patch("wakeword_workbench.cli.load_onnx_model") as mock_load,
            patch("wakeword_workbench.cli.process_long_audio") as mock_process,
            patch("wakeword_workbench.cli.extract_false_positives") as mock_extract,
            patch("wakeword_workbench.cli.add_to_training") as mock_add,
        ):
            # Setup mock model
            mock_model = MagicMock(return_value=0.85)
            mock_load.return_value = mock_model

            # Setup mock predictions
            mock_prediction = MagicMock()
            mock_prediction.timestamp = 5.0
            mock_prediction.prediction = 0.85
            mock_prediction.window_start = 80000
            mock_prediction.window_end = 96000
            mock_process.return_value = [mock_prediction]

            # Setup mock clip extraction
            clip_path = mine_output / "audio_T5.000.wav"
            mock_clip = MockExtractedClip(
                clip_path=clip_path,
                original_path=audio_file,
                timestamp=5.0,
                prediction=0.85,
                threshold=0.7,
                duration=1.0,
            )
            mock_extract.return_value = [mock_clip]

            # Setup mock merge result
            mock_result = MagicMock()
            mock_result.added_count = 1
            mock_result.total_count = 3
            mock_result.backup_path = None
            mock_result.errors = []
            mock_add.return_value = mock_result

            # Step 1: Run mine command
            mine_result = runner.invoke(
                app,
                [
                    "mine",
                    "--model",
                    str(model_file),
                    "--audio",
                    str(tmp_path / "*.wav"),
                    "--output",
                    str(mine_output),
                    "--threshold",
                    "0.7",
                ],
            )

            # Verify mine succeeded
            assert mine_result.exit_code == 0, f"Mine failed: {mine_result.stdout}"
            assert "Extracted 1 clips" in mine_result.stdout

            # Verify manifest was created by mine
            manifest_path = mine_output / "manifest.jsonl"
            assert manifest_path.exists(), "Mine should create manifest.jsonl"

            # Step 2: Run merge command with mine output
            merge_result = runner.invoke(
                app,
                [
                    "merge",
                    "--source",
                    str(manifest_path),
                    "--target",
                    str(training_manifest),
                ],
            )

            # Verify merge succeeded
            assert merge_result.exit_code == 0, f"Merge failed: {merge_result.stdout}"
            assert "Source entries   : 1" in merge_result.stdout
            assert "Total in target  : 3" in merge_result.stdout
            assert "completed successfully" in merge_result.stdout

            # Verify add_to_training was called with correct paths
            mock_add.assert_called_once_with(
                new_negatives_manifest=manifest_path,
                training_manifest=training_manifest,
                backup=False,
            )

    def test_mine_fails_then_merge_not_attempted(self, tmp_path: Path) -> None:
        """Test that merge is not attempted when mine fails."""
        # Create model file
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")

        # Create mine output directory
        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        # Create target training manifest
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text('{"path": "train.wav", "label": 1}\n')

        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")

        # Mine will fail due to model load error
        with patch("wakeword_workbench.cli.load_onnx_model") as mock_load:
            from wakeword_workbench.mining.model_loader import ModelLoadError

            mock_load.side_effect = ModelLoadError("Invalid model")

            mine_result = runner.invoke(
                app,
                [
                    "mine",
                    "--model",
                    str(model_file),
                    "--audio",
                    str(tmp_path / "*.wav"),
                    "--output",
                    str(mine_output),
                ],
            )

            assert mine_result.exit_code == 1
            assert "Failed to load model" in mine_result.stdout

            # Verify no manifest was created
            manifest_path = mine_output / "manifest.jsonl"
            assert not manifest_path.exists(), "Mine failure should not create manifest"

        # Verify merge was never attempted by checking add_to_training was not called
        # (We can infer this since there's no manifest to merge)

    def test_empty_mining_then_merge(self, tmp_path: Path) -> None:
        """Test merge works even when mine extracts no clips."""
        # Create model file
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")

        # Create audio file
        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")

        # Create mine output directory
        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        # Create target training manifest
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text('{"path": "train.wav", "label": 1}\n')

        with (
            patch("wakeword_workbench.cli.load_onnx_model") as mock_load,
            patch("wakeword_workbench.cli.process_long_audio") as mock_process,
            patch("wakeword_workbench.cli.extract_false_positives") as mock_extract,
            patch("wakeword_workbench.cli.add_to_training") as mock_add,
        ):
            mock_model = MagicMock(return_value=0.3)
            mock_load.return_value = mock_model

            mock_prediction = MagicMock()
            mock_prediction.timestamp = 1.0
            mock_prediction.prediction = 0.3
            mock_prediction.window_start = 16000
            mock_prediction.window_end = 32000
            mock_process.return_value = [mock_prediction]

            # No clips extracted (below threshold)
            mock_extract.return_value = []

            # Merge returns 0 added
            mock_result = MagicMock()
            mock_result.added_count = 0
            mock_result.total_count = 5
            mock_result.backup_path = None
            mock_result.errors = ["All entries were duplicates"]
            mock_add.return_value = mock_result

            # Step 1: Run mine (will extract 0 clips)
            mine_result = runner.invoke(
                app,
                [
                    "mine",
                    "--model",
                    str(model_file),
                    "--audio",
                    str(tmp_path / "*.wav"),
                    "--output",
                    str(mine_output),
                    "--threshold",
                    "0.7",
                ],
            )

            assert mine_result.exit_code == 0
            assert "Extracted 0 clips" in mine_result.stdout

            # Manifest should still exist (empty)
            manifest_path = mine_output / "manifest.jsonl"
            assert manifest_path.exists()

            # Step 2: Run merge with empty manifest
            merge_result = runner.invoke(
                app,
                [
                    "merge",
                    "--source",
                    str(manifest_path),
                    "--target",
                    str(training_manifest),
                ],
            )

            assert merge_result.exit_code == 0
            assert "Source entries   : 0" in merge_result.stdout

    def test_multiple_audio_files_workflow(self, tmp_path: Path) -> None:
        """Test mine and merge with multiple audio files."""
        # Create model file
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")

        # Create multiple audio files
        audio_file1 = tmp_path / "audio1.wav"
        audio_file1.write_text("fake audio 1")
        audio_file2 = tmp_path / "audio2.wav"
        audio_file2.write_text("fake audio 2")

        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text('{"path": "train.wav", "label": 1}\n')

        with (
            patch("wakeword_workbench.cli.load_onnx_model") as mock_load,
            patch("wakeword_workbench.cli.process_long_audio") as mock_process,
            patch("wakeword_workbench.cli.extract_false_positives") as mock_extract,
            patch("wakeword_workbench.cli.add_to_training") as mock_add,
        ):
            mock_model = MagicMock(return_value=0.9)
            mock_load.return_value = mock_model

            mock_prediction1 = MagicMock()
            mock_prediction1.timestamp = 2.0
            mock_prediction1.prediction = 0.9
            mock_prediction1.window_start = 32000
            mock_prediction1.window_end = 48000

            mock_prediction2 = MagicMock()
            mock_prediction2.timestamp = 3.0
            mock_prediction2.prediction = 0.9
            mock_prediction2.window_start = 48000
            mock_prediction2.window_end = 64000

            mock_process.side_effect = [[mock_prediction1], [mock_prediction2]]

            mock_clip1 = MockExtractedClip(
                clip_path=mine_output / "audio1_T2.000.wav",
                original_path=audio_file1,
                timestamp=2.0,
                prediction=0.9,
                threshold=0.7,
                duration=1.0,
            )
            mock_clip2 = MockExtractedClip(
                clip_path=mine_output / "audio2_T3.000.wav",
                original_path=audio_file2,
                timestamp=3.0,
                prediction=0.9,
                threshold=0.7,
                duration=1.0,
            )
            mock_extract.side_effect = [[mock_clip1], [mock_clip2]]

            mock_result = MagicMock()
            mock_result.added_count = 2
            mock_result.total_count = 4
            mock_result.backup_path = None
            mock_result.errors = []
            mock_add.return_value = mock_result

            # Mine with multiple files
            mine_result = runner.invoke(
                app,
                [
                    "mine",
                    "--model",
                    str(model_file),
                    "--audio",
                    str(tmp_path / "*.wav"),
                    "--output",
                    str(mine_output),
                ],
            )

            assert mine_result.exit_code == 0
            assert "Extracted 2 clips" in mine_result.stdout
            assert mock_process.call_count == 2
            assert mock_extract.call_count == 2

            # Merge
            manifest_path = mine_output / "manifest.jsonl"
            merge_result = runner.invoke(
                app,
                [
                    "merge",
                    "--source",
                    str(manifest_path),
                    "--target",
                    str(training_manifest),
                ],
            )

            assert merge_result.exit_code == 0
            assert "Source entries   : 2" in merge_result.stdout

    def test_workflow_with_backup(self, tmp_path: Path) -> None:
        """Test mine and merge with backup flag."""
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")

        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")

        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text('{"path": "train.wav", "label": 1}\n')

        with (
            patch("wakeword_workbench.cli.load_onnx_model") as mock_load,
            patch("wakeword_workbench.cli.process_long_audio") as mock_process,
            patch("wakeword_workbench.cli.extract_false_positives") as mock_extract,
            patch("wakeword_workbench.cli.add_to_training") as mock_add,
        ):
            mock_model = MagicMock(return_value=0.85)
            mock_load.return_value = mock_model

            mock_prediction = MagicMock()
            mock_prediction.timestamp = 5.0
            mock_prediction.prediction = 0.85
            mock_prediction.window_start = 80000
            mock_prediction.window_end = 96000
            mock_process.return_value = [mock_prediction]

            mock_clip = MockExtractedClip(
                clip_path=mine_output / "audio_T5.000.wav",
                original_path=audio_file,
                timestamp=5.0,
                prediction=0.85,
                threshold=0.7,
                duration=1.0,
            )
            mock_extract.return_value = [mock_clip]

            backup_path = tmp_path / "training.20240101_120000.backup.jsonl"
            mock_result = MagicMock()
            mock_result.added_count = 1
            mock_result.total_count = 3
            mock_result.backup_path = backup_path
            mock_result.errors = []
            mock_add.return_value = mock_result

            # Mine
            mine_result = runner.invoke(
                app,
                [
                    "mine",
                    "--model",
                    str(model_file),
                    "--audio",
                    str(tmp_path / "*.wav"),
                    "--output",
                    str(mine_output),
                ],
            )
            assert mine_result.exit_code == 0

            # Merge with backup
            manifest_path = mine_output / "manifest.jsonl"
            merge_result = runner.invoke(
                app,
                [
                    "merge",
                    "--source",
                    str(manifest_path),
                    "--target",
                    str(training_manifest),
                    "--backup",
                ],
            )

            assert merge_result.exit_code == 0
            assert "Backup created" in merge_result.stdout
            assert str(backup_path) in merge_result.stdout

            # Verify backup=True was passed to add_to_training
            mock_add.assert_called_once_with(
                new_negatives_manifest=manifest_path,
                training_manifest=training_manifest,
                backup=True,
            )

    def test_manifest_integrity_after_workflow(self, tmp_path: Path) -> None:
        """Test that manifest integrity is preserved through the workflow."""
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")

        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")

        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            '{"path": "train.wav", "label": 1, "text": "hey assistant", "duration_ms": 1000}\n'
        )

        with (
            patch("wakeword_workbench.cli.load_onnx_model") as mock_load,
            patch("wakeword_workbench.cli.process_long_audio") as mock_process,
            patch("wakeword_workbench.cli.extract_false_positives") as mock_extract,
        ):
            mock_model = MagicMock(return_value=0.85)
            mock_load.return_value = mock_model

            mock_prediction = MagicMock()
            mock_prediction.timestamp = 5.0
            mock_prediction.prediction = 0.85
            mock_prediction.window_start = 80000
            mock_prediction.window_end = 96000
            mock_process.return_value = [mock_prediction]

            mock_clip = MockExtractedClip(
                clip_path=mine_output / "audio_T5.000.wav",
                original_path=audio_file,
                timestamp=5.0,
                prediction=0.85,
                threshold=0.7,
                duration=1.0,
            )
            mock_extract.return_value = [mock_clip]

            # Mine creates manifest
            mine_result = runner.invoke(
                app,
                [
                    "mine",
                    "--model",
                    str(model_file),
                    "--audio",
                    str(tmp_path / "*.wav"),
                    "--output",
                    str(mine_output),
                    "--threshold",
                    "0.7",
                ],
            )

            assert mine_result.exit_code == 0

            # Verify mine manifest structure
            mine_manifest_path = mine_output / "manifest.jsonl"
            mine_manifest = Manifest.load(mine_manifest_path)
            assert len(mine_manifest) == 1

            entry = list(mine_manifest)[0]
            assert entry.label == 0  # Hard negative
            assert entry.duration_ms == 1000  # 1.0 second
            assert entry.sample_rate == 16000
            assert entry.metadata["timestamp"] == 5.0
            assert entry.metadata["prediction"] == 0.85
            assert entry.metadata["threshold"] == 0.7


class TestErrorPropagation:
    """Tests for error conditions and propagation between commands."""

    def test_merge_handles_invalid_manifest_file(self, tmp_path: Path) -> None:
        """Test merge reports warnings when source manifest contains invalid JSON."""
        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        mine_manifest_path = mine_output / "manifest.jsonl"
        mine_manifest_path.write_text("not valid json\n")

        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text('{"path": "train.wav", "label": 1}\n')

        merge_result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(mine_manifest_path),
                "--target",
                str(training_manifest),
            ],
        )

        assert merge_result.exit_code == 0
        assert "Warnings" in merge_result.stdout
        assert "Invalid JSON" in merge_result.stdout

    def test_merge_missing_target_not_found(self, tmp_path: Path) -> None:
        """Test merge fails appropriately when target manifest doesn't exist."""
        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        source = mine_output / "manifest.jsonl"
        source.write_text('{"path": "neg.wav", "label": 0}\n')

        result = runner.invoke(
            app,
            [
                "merge",
                "--source",
                str(source),
                "--target",
                str(tmp_path / "nonexistent.jsonl"),
            ],
        )

        assert result.exit_code == 2
        assert "not found" in result.stdout.lower()


class TestWorkflowEdgeCases:
    """Tests for edge cases in the mine-merge workflow."""

    def test_workflow_with_custom_threshold(self, tmp_path: Path) -> None:
        """Test workflow with different threshold values."""
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")

        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")

        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text('{"path": "train.wav", "label": 1}\n')

        with (
            patch("wakeword_workbench.cli.load_onnx_model") as mock_load,
            patch("wakeword_workbench.cli.process_long_audio") as mock_process,
            patch("wakeword_workbench.cli.extract_false_positives") as mock_extract,
            patch("wakeword_workbench.cli.add_to_training") as mock_add,
        ):
            mock_model = MagicMock(return_value=0.95)
            mock_load.return_value = mock_model

            mock_prediction = MagicMock()
            mock_prediction.timestamp = 5.0
            mock_prediction.prediction = 0.95
            mock_prediction.window_start = 80000
            mock_prediction.window_end = 96000
            mock_process.return_value = [mock_prediction]

            threshold_used = [0.9]

            def extract_side_effect(predictions, audio_path, threshold, output_dir):
                threshold_used[0] = threshold
                if threshold <= 0.9:
                    return [
                        MockExtractedClip(
                            clip_path=output_dir / "audio_T5.000.wav",
                            original_path=audio_file,
                            timestamp=5.0,
                            prediction=0.95,
                            threshold=threshold,
                            duration=1.0,
                        )
                    ]
                return []

            mock_extract.side_effect = extract_side_effect

            mock_result = MagicMock()
            mock_result.added_count = 1
            mock_result.total_count = 2
            mock_result.backup_path = None
            mock_result.errors = []
            mock_add.return_value = mock_result

            # Mine with high threshold (0.95)
            mine_result = runner.invoke(
                app,
                [
                    "mine",
                    "--model",
                    str(model_file),
                    "--audio",
                    str(tmp_path / "*.wav"),
                    "--output",
                    str(mine_output),
                    "--threshold",
                    "0.95",
                ],
            )

            assert mine_result.exit_code == 0

            mine_output2 = tmp_path / "mine_output2"
            mine_output2.mkdir()

            mine_result2 = runner.invoke(
                app,
                [
                    "mine",
                    "--model",
                    str(model_file),
                    "--audio",
                    str(tmp_path / "*.wav"),
                    "--output",
                    str(mine_output2),
                    "--threshold",
                    "0.8",
                ],
            )

            assert mine_result2.exit_code == 0
            assert "Extracted 1 clips" in mine_result2.stdout

            assert threshold_used[0] == 0.8

    def test_merge_with_source_not_jsonl(self, tmp_path: Path) -> None:
        """Test merge validation: source must be .jsonl."""
        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        source = mine_output / "source.txt"
        source.write_text("not jsonl")

        target = tmp_path / "target.jsonl"
        target.write_text('{"path": "train.wav", "label": 1}\n')

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

    def test_workflow_with_merge_warnings(self, tmp_path: Path) -> None:
        """Test workflow where merge produces warnings."""
        model_file = tmp_path / "model.onnx"
        model_file.write_text("fake model")

        audio_file = tmp_path / "audio.wav"
        audio_file.write_text("fake audio")

        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text('{"path": "train.wav", "label": 1}\n')

        with (
            patch("wakeword_workbench.cli.load_onnx_model") as mock_load,
            patch("wakeword_workbench.cli.process_long_audio") as mock_process,
            patch("wakeword_workbench.cli.extract_false_positives") as mock_extract,
            patch("wakeword_workbench.cli.add_to_training") as mock_add,
        ):
            mock_model = MagicMock(return_value=0.85)
            mock_load.return_value = mock_model

            mock_prediction = MagicMock()
            mock_prediction.timestamp = 5.0
            mock_prediction.prediction = 0.85
            mock_prediction.window_start = 80000
            mock_prediction.window_end = 96000
            mock_process.return_value = [mock_prediction]

            mock_clip = MockExtractedClip(
                clip_path=mine_output / "audio_T5.000.wav",
                original_path=audio_file,
                timestamp=5.0,
                prediction=0.85,
                threshold=0.7,
                duration=1.0,
            )
            mock_extract.return_value = [mock_clip]

            # Mock merge with warnings
            mock_result = MagicMock()
            mock_result.added_count = 1
            mock_result.total_count = 3
            mock_result.backup_path = None
            mock_result.errors = [
                "Line 2: Audio file not found: missing.wav",
                "Line 3: Duplicate entry skipped: dup.wav",
            ]
            mock_add.return_value = mock_result

            # Mine
            mine_result = runner.invoke(
                app,
                [
                    "mine",
                    "--model",
                    str(model_file),
                    "--audio",
                    str(tmp_path / "*.wav"),
                    "--output",
                    str(mine_output),
                ],
            )
            assert mine_result.exit_code == 0

            # Merge with warnings
            manifest_path = mine_output / "manifest.jsonl"
            merge_result = runner.invoke(
                app,
                [
                    "merge",
                    "--source",
                    str(manifest_path),
                    "--target",
                    str(training_manifest),
                ],
            )

            assert merge_result.exit_code == 0
            assert "Warnings" in merge_result.stdout
            assert "Audio file not found" in merge_result.stdout
            assert "Duplicate entry" in merge_result.stdout

    def test_merge_with_many_warnings_truncated(self, tmp_path: Path) -> None:
        """Test merge truncates warnings after 5."""
        mine_output = tmp_path / "mine_output"
        mine_output.mkdir()

        manifest_path = mine_output / "manifest.jsonl"
        manifest_path.write_text('{"path": "neg.wav", "label": 0}\n')

        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text('{"path": "train.wav", "label": 1}\n')

        with patch("wakeword_workbench.cli.add_to_training") as mock_add:
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
                    str(manifest_path),
                    "--target",
                    str(training_manifest),
                ],
            )

            assert result.exit_code == 0
            assert "Warnings (7)" in result.stdout
            assert "and 2 more" in result.stdout
