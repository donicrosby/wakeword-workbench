"""Tests for mining module components.

This module tests:
- long_audio.py: ProcessLongAudio with sliding window inference
- extractor.py: ExtractFalsePositives for mining hard negatives
- merge_back.py: AddToTraining for merging mined negatives back
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wakeword_workbench.mining.extractor import (
    ExtractedClip,
    _generate_clip_path,
    _select_with_cooldown,
    extract_false_positives,
)
from wakeword_workbench.mining.long_audio import (
    WindowPrediction,
    _deduplicate_predictions,
    _process_audio_array,
    _process_in_chunks,
    process_long_audio,
)
from wakeword_workbench.mining.merge_back import (
    MergeBackError,
    MergeResult,
    add_to_training,
)


class TestWindowPrediction:
    """Tests for WindowPrediction dataclass."""

    def test_window_prediction_creation(self) -> None:
        """Test creating a WindowPrediction instance."""
        pred = WindowPrediction(
            timestamp=1.5,
            prediction=0.85,
            window_start=24000,
            window_end=40000,
        )
        assert pred.timestamp == 1.5
        assert pred.prediction == 0.85
        assert pred.window_start == 24000
        assert pred.window_end == 40000


class TestProcessAudioArray:
    """Tests for _process_audio_array helper function."""

    def test_short_audio_single_window(self) -> None:
        """Audio shorter than window size should produce single prediction."""
        mock_model = MagicMock(return_value=0.75)
        audio = np.ones(8000, dtype=np.float32)  # 0.5s at 16kHz
        window_size = 16000  # 1s
        hop_size = 8000
        sample_rate = 16000

        results = _process_audio_array(mock_model, audio, window_size, hop_size, sample_rate)

        assert len(results) == 1
        assert results[0].timestamp == 0.0
        assert results[0].prediction == 0.75
        assert results[0].window_start == 0
        assert results[0].window_end == 8000
        mock_model.assert_called_once()

    def test_short_audio_padding(self) -> None:
        """Audio shorter than window should be padded with zeros."""
        mock_model = MagicMock(return_value=0.5)
        audio = np.ones(8000, dtype=np.float32)
        window_size = 16000
        hop_size = 8000
        sample_rate = 16000

        _process_audio_array(mock_model, audio, window_size, hop_size, sample_rate)

        # Check that model received padded audio
        call_args = mock_model.call_args[0][0]
        assert len(call_args) == window_size
        assert np.all(call_args[8000:] == 0)  # Padded region should be zeros

    def test_multiple_sliding_windows(self) -> None:
        """Audio longer than window should produce multiple predictions."""
        # Need 4 return values: 3 regular windows + 1 final partial window
        mock_model = MagicMock(side_effect=[0.6, 0.7, 0.8, 0.75])
        # 2.5 seconds of audio = 40000 samples
        audio = np.ones(40000, dtype=np.float32)
        window_size = 16000  # 1s
        hop_size = 8000  # 0.5s
        sample_rate = 16000

        results = _process_audio_array(mock_model, audio, window_size, hop_size, sample_rate)

        # Should have 3 windows: samples 0-16000, 8000-24000, 16000-32000
        # Plus final partial window 24000-40000
        assert len(results) == 4
        assert results[0].timestamp == 0.0
        assert results[1].timestamp == 0.5
        assert results[2].timestamp == 1.0
        assert results[3].timestamp == 1.5  # 24000 / 16000 = 1.5

    def test_offset_samples(self) -> None:
        """Offset should be added to window positions."""
        mock_model = MagicMock(return_value=0.5)
        audio = np.ones(16000, dtype=np.float32)
        window_size = 16000
        hop_size = 8000
        sample_rate = 16000
        offset = 32000  # 2 seconds offset

        results = _process_audio_array(
            mock_model, audio, window_size, hop_size, sample_rate, offset_samples=offset
        )

        assert results[0].window_start == offset
        assert results[0].window_end == offset + 16000
        assert results[0].timestamp == 2.0  # 32000 / 16000

    def test_exact_window_boundary(self) -> None:
        """Audio exactly matching window boundaries."""
        mock_model = MagicMock(return_value=0.9)
        audio = np.ones(32000, dtype=np.float32)  # Exactly 2 windows
        window_size = 16000
        hop_size = 16000  # No overlap
        sample_rate = 16000

        results = _process_audio_array(mock_model, audio, window_size, hop_size, sample_rate)

        assert len(results) == 2
        assert results[0].window_start == 0
        assert results[1].window_start == 16000

    def test_empty_audio(self) -> None:
        """Empty audio array should produce single result with padding."""
        mock_model = MagicMock(return_value=0.0)
        audio = np.array([], dtype=np.float32)
        window_size = 16000
        hop_size = 8000
        sample_rate = 16000

        results = _process_audio_array(mock_model, audio, window_size, hop_size, sample_rate)

        assert len(results) == 1
        assert results[0].window_start == 0
        assert results[0].window_end == 0


class TestDeduplicatePredictions:
    """Tests for _deduplicate_predictions helper function."""

    def test_no_duplicates(self) -> None:
        """List with no duplicates should remain unchanged."""
        preds = [
            WindowPrediction(timestamp=0.0, prediction=0.5, window_start=0, window_end=16000),
            WindowPrediction(timestamp=0.5, prediction=0.6, window_start=8000, window_end=24000),
            WindowPrediction(timestamp=1.0, prediction=0.7, window_start=16000, window_end=32000),
        ]

        result = _deduplicate_predictions(preds)

        assert len(result) == 3

    def test_removes_duplicates(self) -> None:
        """Duplicate window_start values should be removed."""
        preds = [
            WindowPrediction(timestamp=0.0, prediction=0.5, window_start=0, window_end=16000),
            WindowPrediction(
                timestamp=0.0, prediction=0.6, window_start=0, window_end=16000
            ),  # Duplicate
            WindowPrediction(timestamp=0.5, prediction=0.7, window_start=8000, window_end=24000),
        ]

        result = _deduplicate_predictions(preds)

        assert len(result) == 2
        assert result[0].prediction == 0.5  # First one kept

    def test_sorts_by_window_start(self) -> None:
        """Results should be sorted by window_start."""
        preds = [
            WindowPrediction(timestamp=1.0, prediction=0.7, window_start=16000, window_end=32000),
            WindowPrediction(timestamp=0.0, prediction=0.5, window_start=0, window_end=16000),
            WindowPrediction(timestamp=0.5, prediction=0.6, window_start=8000, window_end=24000),
        ]

        result = _deduplicate_predictions(preds)

        assert result[0].window_start == 0
        assert result[1].window_start == 8000
        assert result[2].window_start == 16000

    def test_empty_list(self) -> None:
        """Empty list should return empty list."""
        result = _deduplicate_predictions([])
        assert result == []


class TestProcessLongAudio:
    """Tests for process_long_audio function."""

    def test_file_not_found_raises(self) -> None:
        """Non-existent file should raise FileNotFoundError."""
        mock_model = MagicMock(return_value=0.5)
        with pytest.raises(FileNotFoundError, match="not found"):
            process_long_audio(mock_model, "/nonexistent/audio.wav")

    @patch("wakeword_workbench.mining.long_audio.librosa")
    def test_short_audio_processing(self, mock_librosa: MagicMock, tmp_path: Path) -> None:
        """Short audio files should be processed in one go."""
        mock_model = MagicMock(return_value=0.75)
        audio_path = tmp_path / "short.wav"
        audio_path.write_bytes(b"fake_audio_data")

        # Mock duration check (10 seconds, under chunk_duration default)
        mock_librosa.get_duration.return_value = 10.0
        # Mock audio loading
        mock_audio = np.ones(160000, dtype=np.float32)  # 10s at 16kHz
        mock_librosa.load.return_value = (mock_audio, 16000)

        results = process_long_audio(
            mock_model,
            audio_path,
            window_size=16000,
            hop_size=8000,
            sample_rate=16000,
            chunk_duration=30.0,
        )

        # Should load full audio
        mock_librosa.load.assert_called_once()
        # Should have multiple windows
        assert len(results) > 0

    @patch("wakeword_workbench.mining.long_audio.librosa")
    def test_long_audio_chunked_processing(self, mock_librosa: MagicMock, tmp_path: Path) -> None:
        """Long audio files should be processed in chunks."""
        mock_model = MagicMock(return_value=0.6)
        audio_path = tmp_path / "long.wav"
        audio_path.write_bytes(b"fake_audio_data")

        # Mock duration check (100 seconds, over chunk_duration default)
        mock_librosa.get_duration.return_value = 100.0
        # Mock audio loading for chunks
        mock_audio = np.ones(480000, dtype=np.float32)  # 30s at 16kHz
        mock_librosa.load.return_value = (mock_audio, 16000)

        process_long_audio(
            mock_model,
            audio_path,
            window_size=16000,
            hop_size=8000,
            sample_rate=16000,
            chunk_duration=30.0,
        )

        # Should load multiple times (for each chunk)
        assert mock_librosa.load.call_count > 1

    @patch("wakeword_workbench.mining.long_audio.librosa")
    def test_correct_timestamps(self, mock_librosa: MagicMock, tmp_path: Path) -> None:
        """Timestamps should be correctly calculated."""
        mock_model = MagicMock(return_value=0.5)
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake_audio_data")

        mock_librosa.get_duration.return_value = 5.0
        mock_audio = np.ones(80000, dtype=np.float32)  # 5s at 16kHz
        mock_librosa.load.return_value = (mock_audio, 16000)

        results = process_long_audio(
            mock_model,
            audio_path,
            window_size=16000,
            hop_size=8000,
            sample_rate=16000,
        )

        # Check timestamps are in seconds and increasing
        for i, pred in enumerate(results):
            expected_timestamp = i * 0.5  # hop_size=8000 at 16000Hz = 0.5s
            assert abs(pred.timestamp - expected_timestamp) < 0.01

    def test_invalid_window_size(self, tmp_path: Path) -> None:
        """Zero or negative window_size should raise ValueError."""
        mock_model = MagicMock(return_value=0.5)
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake")
        with pytest.raises(ValueError, match="window_size must be positive"):
            process_long_audio(mock_model, audio_path, window_size=0)
        with pytest.raises(ValueError, match="window_size must be positive"):
            process_long_audio(mock_model, audio_path, window_size=-1)

    def test_invalid_hop_size(self, tmp_path: Path) -> None:
        """Zero or negative hop_size should raise ValueError."""
        mock_model = MagicMock(return_value=0.5)
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake")
        with pytest.raises(ValueError, match="hop_size must be positive"):
            process_long_audio(mock_model, audio_path, hop_size=0)
        with pytest.raises(ValueError, match="hop_size must be positive"):
            process_long_audio(mock_model, audio_path, hop_size=-1)

    def test_invalid_chunk_duration(self, tmp_path: Path) -> None:
        """Zero or negative chunk_duration should raise ValueError."""
        mock_model = MagicMock(return_value=0.5)
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake")
        with pytest.raises(ValueError, match="chunk_duration must be positive"):
            process_long_audio(mock_model, audio_path, chunk_duration=0)
        with pytest.raises(ValueError, match="chunk_duration must be positive"):
            process_long_audio(mock_model, audio_path, chunk_duration=-1)


class TestProcessInChunks:
    """Tests for _process_in_chunks helper function."""

    @patch("wakeword_workbench.mining.long_audio.librosa")
    def test_chunk_overlap_handling(self, mock_librosa: MagicMock, tmp_path: Path) -> None:
        """Chunks should overlap by window_size to ensure continuity."""
        mock_model = MagicMock(return_value=0.5)
        audio_path = tmp_path / "chunked.wav"
        audio_path.write_bytes(b"fake_audio_data")

        # Create 60 seconds of audio
        mock_audio = np.ones(960000, dtype=np.float32)  # 60s at 16kHz
        mock_librosa.load.return_value = (mock_audio, 16000)

        _process_in_chunks(
            mock_model,
            audio_path,
            window_size=16000,
            hop_size=8000,
            sample_rate=16000,
            chunk_duration=30.0,
            total_duration=60.0,
        )

        # Should have processed multiple chunks
        assert mock_librosa.load.call_count >= 2

    @patch("wakeword_workbench.mining.long_audio.librosa")
    def test_deduplication_across_chunks(self, mock_librosa: MagicMock, tmp_path: Path) -> None:
        """Predictions at chunk boundaries should be deduplicated."""
        mock_model = MagicMock(return_value=0.5)
        audio_path = tmp_path / "chunked.wav"
        audio_path.write_bytes(b"fake_audio_data")

        mock_audio = np.ones(480000, dtype=np.float32)  # 30s at 16kHz
        mock_librosa.load.return_value = (mock_audio, 16000)

        results = _process_in_chunks(
            mock_model,
            audio_path,
            window_size=16000,
            hop_size=8000,
            sample_rate=16000,
            chunk_duration=30.0,
            total_duration=60.0,
        )

        # Check no duplicate window_start values
        starts = [r.window_start for r in results]
        assert len(starts) == len(set(starts))


class TestSelectWithCooldown:
    """Tests for _select_with_cooldown helper function."""

    def test_select_above_threshold(self) -> None:
        """Predictions above threshold should be selected."""
        preds = [
            WindowPrediction(timestamp=0.0, prediction=0.3, window_start=0, window_end=16000),
            WindowPrediction(timestamp=0.5, prediction=0.9, window_start=8000, window_end=24000),
            WindowPrediction(timestamp=1.0, prediction=0.4, window_start=16000, window_end=32000),
        ]

        selected = _select_with_cooldown(preds, threshold=0.5, cooldown_seconds=0.5)

        assert len(selected) == 1
        assert selected[0].prediction == 0.9

    def test_cooldown_skips_close_predictions(self) -> None:
        """Predictions within cooldown period should be skipped."""
        preds = [
            WindowPrediction(timestamp=0.0, prediction=0.9, window_start=0, window_end=16000),
            WindowPrediction(timestamp=0.3, prediction=0.8, window_start=4800, window_end=20800),
            WindowPrediction(timestamp=2.0, prediction=0.85, window_start=32000, window_end=48000),
        ]

        selected = _select_with_cooldown(preds, threshold=0.5, cooldown_seconds=1.0)

        assert len(selected) == 2  # First and third
        assert selected[0].prediction == 0.9
        assert selected[1].prediction == 0.85

    def test_consecutive_detections_filtered(self) -> None:
        """Multiple consecutive detections should be filtered by cooldown."""
        preds = [
            WindowPrediction(timestamp=0.0, prediction=0.9, window_start=0, window_end=16000),
            WindowPrediction(timestamp=0.1, prediction=0.85, window_start=1600, window_end=17600),
            WindowPrediction(timestamp=0.2, prediction=0.88, window_start=3200, window_end=19200),
            WindowPrediction(timestamp=0.3, prediction=0.82, window_start=4800, window_end=20800),
        ]

        selected = _select_with_cooldown(preds, threshold=0.5, cooldown_seconds=0.5)

        assert len(selected) == 1  # Only first one

    def test_exactly_at_threshold_not_selected(self) -> None:
        """Predictions exactly at threshold should not be selected."""
        preds = [
            WindowPrediction(timestamp=0.0, prediction=0.5, window_start=0, window_end=16000),
        ]

        selected = _select_with_cooldown(preds, threshold=0.5, cooldown_seconds=0.5)

        assert len(selected) == 0

    def test_zero_cooldown_selects_all(self) -> None:
        """With zero cooldown, all predictions above threshold should be selected."""
        preds = [
            WindowPrediction(timestamp=0.0, prediction=0.9, window_start=0, window_end=16000),
            WindowPrediction(timestamp=0.1, prediction=0.8, window_start=1600, window_end=17600),
            WindowPrediction(timestamp=0.2, prediction=0.85, window_start=3200, window_end=19200),
        ]

        selected = _select_with_cooldown(preds, threshold=0.5, cooldown_seconds=0.0)

        assert len(selected) == 3

    def test_empty_predictions(self) -> None:
        """Empty predictions list should return empty list."""
        selected = _select_with_cooldown([], threshold=0.5, cooldown_seconds=1.0)
        assert selected == []

    def test_all_below_threshold(self) -> None:
        """All predictions below threshold should return empty list."""
        preds = [
            WindowPrediction(timestamp=0.0, prediction=0.1, window_start=0, window_end=16000),
            WindowPrediction(timestamp=0.5, prediction=0.2, window_start=8000, window_end=24000),
        ]

        selected = _select_with_cooldown(preds, threshold=0.5, cooldown_seconds=0.5)

        assert selected == []


class TestGenerateClipPath:
    """Tests for _generate_clip_path helper function."""

    def test_basic_filename_generation(self, tmp_path: Path) -> None:
        """Should generate filename with timestamp."""
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        result = _generate_clip_path(output_dir, "recording", 1.234)

        assert result.name == "recording_T1.234.wav"
        assert result.parent == output_dir

    def test_counter_suffix_for_duplicates(self, tmp_path: Path) -> None:
        """Should add counter suffix if file already exists."""
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        # Create existing file
        existing = output_dir / "recording_T1.234.wav"
        existing.write_bytes(b"existing")

        result = _generate_clip_path(output_dir, "recording", 1.234)

        assert result.name == "recording_T1.234_1.wav"

    def test_multiple_duplicates(self, tmp_path: Path) -> None:
        """Should increment counter for multiple duplicates."""
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        # Create multiple existing files
        (output_dir / "audio_T5.000.wav").write_bytes(b"1")
        (output_dir / "audio_T5.000_1.wav").write_bytes(b"2")
        (output_dir / "audio_T5.000_2.wav").write_bytes(b"3")

        result = _generate_clip_path(output_dir, "audio", 5.0)

        assert result.name == "audio_T5.000_3.wav"

    def test_timestamp_formatting(self, tmp_path: Path) -> None:
        """Should format timestamp with 3 decimal places."""
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        result = _generate_clip_path(output_dir, "test", 123.456789)

        assert "T123.457" in result.name  # Rounded to 3 decimal places


class TestExtractFalsePositives:
    """Tests for extract_false_positives function."""

    def test_audio_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent audio file should raise FileNotFoundError."""
        predictions = []
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="not found"):
            extract_false_positives(
                predictions,
                "/nonexistent/audio.wav",
                threshold=0.5,
                output_dir=output_dir,
            )

    def test_output_dir_not_exists(self, tmp_path: Path) -> None:
        """Non-existent output directory should raise NotADirectoryError."""
        predictions = []
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake")

        with pytest.raises(NotADirectoryError, match="does not exist"):
            extract_false_positives(
                predictions,
                audio_path,
                threshold=0.5,
                output_dir=tmp_path / "nonexistent",
            )

    def test_invalid_threshold(self, tmp_path: Path) -> None:
        """Threshold outside 0-1 range should raise ValueError."""
        predictions = []
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(ValueError, match="threshold must be between"):
            extract_false_positives(predictions, audio_path, threshold=-0.1, output_dir=output_dir)
        with pytest.raises(ValueError, match="threshold must be between"):
            extract_false_positives(predictions, audio_path, threshold=1.1, output_dir=output_dir)

    def test_invalid_cooldown(self, tmp_path: Path) -> None:
        """Negative cooldown should raise ValueError."""
        predictions = []
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(ValueError, match="cooldown_seconds must be non-negative"):
            extract_false_positives(
                predictions, audio_path, threshold=0.5, output_dir=output_dir, cooldown_seconds=-1
            )

    @patch("wakeword_workbench.mining.extractor.librosa")
    @patch("wakeword_workbench.mining.extractor.sf")
    def test_extracts_clips_above_threshold(
        self, mock_sf: MagicMock, mock_librosa: MagicMock, tmp_path: Path
    ) -> None:
        """Should extract audio clips for predictions above threshold."""
        # Setup
        audio_path = tmp_path / "recording.wav"
        audio_path.write_bytes(b"fake")
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        # Create mock audio data (5 seconds at 16kHz)
        mock_audio = np.random.randn(80000).astype(np.float32)
        mock_librosa.load.return_value = (mock_audio, 16000)

        # Create predictions
        predictions = [
            WindowPrediction(timestamp=0.5, prediction=0.9, window_start=8000, window_end=24000),
            WindowPrediction(timestamp=2.0, prediction=0.3, window_start=32000, window_end=48000),
            WindowPrediction(timestamp=3.5, prediction=0.85, window_start=56000, window_end=72000),
        ]

        results = extract_false_positives(
            predictions,
            audio_path,
            threshold=0.5,
            output_dir=output_dir,
            cooldown_seconds=0.5,
        )

        # Should extract 2 clips (above threshold, within cooldown)
        assert len(results) == 2
        assert all(isinstance(r, ExtractedClip) for r in results)
        assert mock_sf.write.call_count == 2

    @patch("wakeword_workbench.mining.extractor.librosa")
    @patch("wakeword_workbench.mining.extractor.sf")
    def test_clip_metadata(
        self, mock_sf: MagicMock, mock_librosa: MagicMock, tmp_path: Path
    ) -> None:
        """Extracted clips should have correct metadata."""
        audio_path = tmp_path / "source.wav"
        audio_path.write_bytes(b"fake")
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        mock_audio = np.ones(80000, dtype=np.float32)
        mock_librosa.load.return_value = (mock_audio, 16000)

        predictions = [
            WindowPrediction(timestamp=1.5, prediction=0.8, window_start=24000, window_end=40000),
        ]

        results = extract_false_positives(
            predictions,
            audio_path,
            threshold=0.5,
            output_dir=output_dir,
        )

        assert len(results) == 1
        clip = results[0]
        assert clip.original_path == audio_path
        assert clip.timestamp == 1.5
        assert clip.prediction == 0.8
        assert clip.threshold == 0.5
        assert clip.duration == 1.0  # 16000 samples at 16000Hz

    @patch("wakeword_workbench.mining.extractor.librosa")
    @patch("wakeword_workbench.mining.extractor.sf")
    def test_cooldown_deduplication(
        self, mock_sf: MagicMock, mock_librosa: MagicMock, tmp_path: Path
    ) -> None:
        """Cooldown should prevent extracting duplicate clips from same event."""
        audio_path = tmp_path / "recording.wav"
        audio_path.write_bytes(b"fake")
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        mock_audio = np.ones(160000, dtype=np.float32)
        mock_librosa.load.return_value = (mock_audio, 16000)

        # Multiple close predictions (same false positive event)
        predictions = [
            WindowPrediction(timestamp=0.5, prediction=0.9, window_start=8000, window_end=24000),
            WindowPrediction(timestamp=0.6, prediction=0.85, window_start=9600, window_end=25600),
            WindowPrediction(timestamp=0.7, prediction=0.88, window_start=11200, window_end=27200),
        ]

        results = extract_false_positives(
            predictions,
            audio_path,
            threshold=0.5,
            output_dir=output_dir,
            cooldown_seconds=1.0,  # 1 second cooldown
        )

        # Should only extract first clip due to cooldown
        assert len(results) == 1
        assert results[0].prediction == 0.9

    @patch("wakeword_workbench.mining.extractor.librosa")
    @patch("wakeword_workbench.mining.extractor.sf")
    def test_no_predictions_above_threshold(
        self, mock_sf: MagicMock, mock_librosa: MagicMock, tmp_path: Path
    ) -> None:
        """No predictions above threshold should return empty list."""
        audio_path = tmp_path / "recording.wav"
        audio_path.write_bytes(b"fake")
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        mock_audio = np.ones(80000, dtype=np.float32)
        mock_librosa.load.return_value = (mock_audio, 16000)

        predictions = [
            WindowPrediction(timestamp=0.5, prediction=0.2, window_start=8000, window_end=24000),
            WindowPrediction(timestamp=1.5, prediction=0.3, window_start=24000, window_end=40000),
        ]

        results = extract_false_positives(
            predictions,
            audio_path,
            threshold=0.5,
            output_dir=output_dir,
        )

        assert results == []
        mock_sf.write.assert_not_called()

    @patch("wakeword_workbench.mining.extractor.librosa")
    @patch("wakeword_workbench.mining.extractor.sf")
    def test_empty_predictions_list(
        self, mock_sf: MagicMock, mock_librosa: MagicMock, tmp_path: Path
    ) -> None:
        """Empty predictions list should return empty list without loading audio."""
        audio_path = tmp_path / "recording.wav"
        audio_path.write_bytes(b"fake")
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        results = extract_false_positives(
            [],
            audio_path,
            threshold=0.5,
            output_dir=output_dir,
        )

        assert results == []
        mock_librosa.load.assert_not_called()
        mock_sf.write.assert_not_called()


class TestExtractedClip:
    """Tests for ExtractedClip dataclass."""

    def test_clip_creation(self, tmp_path: Path) -> None:
        """Test creating an ExtractedClip instance."""
        clip = ExtractedClip(
            clip_path=tmp_path / "clip.wav",
            original_path=tmp_path / "original.wav",
            timestamp=2.5,
            prediction=0.9,
            threshold=0.5,
            duration=1.0,
        )
        assert clip.clip_path == tmp_path / "clip.wav"
        assert clip.original_path == tmp_path / "original.wav"
        assert clip.timestamp == 2.5
        assert clip.prediction == 0.9
        assert clip.threshold == 0.5
        assert clip.duration == 1.0


class TestAddToTraining:
    """Tests for add_to_training function."""

    def test_new_negatives_manifest_not_found(self, tmp_path: Path) -> None:
        """Non-existent new negatives manifest should raise MergeBackError."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text("")

        with pytest.raises(MergeBackError, match="New negatives manifest not found"):
            add_to_training(tmp_path / "nonexistent.jsonl", training_manifest)

    def test_training_manifest_not_found(self, tmp_path: Path) -> None:
        """Non-existent training manifest should raise MergeBackError."""
        new_negatives = tmp_path / "new.jsonl"
        new_negatives.write_text("")

        with pytest.raises(MergeBackError, match="Training manifest not found"):
            add_to_training(new_negatives, tmp_path / "nonexistent.jsonl")

    def test_valid_manifest_merge(self, tmp_path: Path) -> None:
        """Valid entries should be added to training manifest."""
        # Create training manifest with existing entry
        training_manifest = tmp_path / "training.jsonl"
        training_entry = {
            "path": "existing.wav",
            "label": 0,
            "text": "existing",
            "duration_ms": 1000,
        }
        training_manifest.write_text(json.dumps(training_entry) + "\n")
        # Create the audio file
        (tmp_path / "existing.wav").write_bytes(b"fake")

        # Create new negatives manifest
        new_negatives = tmp_path / "new_negatives.jsonl"
        new_entry = {
            "path": "new_negative.wav",
            "label": 0,
            "text": "new negative",
            "duration_ms": 1200,
        }
        new_negatives.write_text(json.dumps(new_entry) + "\n")
        # Create the audio file
        (tmp_path / "new_negative.wav").write_bytes(b"fake")

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert result.added_count == 1
        assert result.total_count == 2
        assert result.backup_path is None
        assert len(result.errors) == 0

    def test_duplicate_detection(self, tmp_path: Path) -> None:
        """Duplicate entries should be skipped with error logged."""
        # Create training manifest
        training_manifest = tmp_path / "training.jsonl"
        training_entry = {
            "path": "duplicate.wav",
            "label": 0,
            "text": "duplicate",
            "duration_ms": 1000,
        }
        training_manifest.write_text(json.dumps(training_entry) + "\n")
        (tmp_path / "duplicate.wav").write_bytes(b"fake")

        # Create new negatives with duplicate
        new_negatives = tmp_path / "new.jsonl"
        duplicate_entry = {
            "path": "duplicate.wav",
            "label": 0,
            "text": "duplicate again",
            "duration_ms": 1200,
        }
        new_negatives.write_text(json.dumps(duplicate_entry) + "\n")

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert result.added_count == 0
        assert result.total_count == 1
        assert len(result.errors) == 1
        assert "Duplicate entry skipped" in result.errors[0]

    def test_backup_creation(self, tmp_path: Path) -> None:
        """Backup should be created when backup=True."""
        # Create training manifest
        training_manifest = tmp_path / "training.jsonl"
        training_entry = {
            "path": "existing.wav",
            "label": 0,
            "text": "existing",
            "duration_ms": 1000,
        }
        training_manifest.write_text(json.dumps(training_entry) + "\n")
        (tmp_path / "existing.wav").write_bytes(b"fake")

        # Create new negatives
        new_negatives = tmp_path / "new.jsonl"
        new_entry = {
            "path": "new.wav",
            "label": 0,
            "text": "new",
            "duration_ms": 1000,
        }
        new_negatives.write_text(json.dumps(new_entry) + "\n")
        (tmp_path / "new.wav").write_bytes(b"fake")

        result = add_to_training(new_negatives, training_manifest, backup=True)

        assert result.backup_path is not None
        assert result.backup_path.exists()
        assert "training_manifest." in result.backup_path.name
        assert ".backup.jsonl" in result.backup_path.name

    def test_backup_not_created_when_false(self, tmp_path: Path) -> None:
        """No backup should be created when backup=False."""
        training_manifest = tmp_path / "training.jsonl"
        training_entry = {
            "path": "existing.wav",
            "label": 0,
            "text": "existing",
            "duration_ms": 1000,
        }
        training_manifest.write_text(json.dumps(training_entry) + "\n")
        (tmp_path / "existing.wav").write_bytes(b"fake")

        new_negatives = tmp_path / "new.jsonl"
        new_entry = {
            "path": "new.wav",
            "label": 0,
            "text": "new",
            "duration_ms": 1000,
        }
        new_negatives.write_text(json.dumps(new_entry) + "\n")
        (tmp_path / "new.wav").write_bytes(b"fake")

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert result.backup_path is None

    def test_invalid_json_handling(self, tmp_path: Path) -> None:
        """Invalid JSON lines should be skipped with error logged."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            json.dumps({"path": "existing.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )
        (tmp_path / "existing.wav").write_bytes(b"fake")

        new_negatives = tmp_path / "new.jsonl"
        new_negatives.write_text("invalid json\n")  # Invalid JSON

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert result.added_count == 0
        assert len(result.errors) == 1
        assert "Invalid JSON" in result.errors[0]

    def test_missing_path_field(self, tmp_path: Path) -> None:
        """Entries missing path field should be skipped."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            json.dumps({"path": "existing.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )
        (tmp_path / "existing.wav").write_bytes(b"fake")

        new_negatives = tmp_path / "new.jsonl"
        new_negatives.write_text(json.dumps({"label": 0}) + "\n")  # Missing path

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert result.added_count == 0
        assert len(result.errors) == 1
        assert "Missing 'path' field" in result.errors[0]

    def test_missing_audio_file(self, tmp_path: Path) -> None:
        """Missing audio files should be logged as errors but abort merge if too many."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            json.dumps({"path": "existing.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )
        (tmp_path / "existing.wav").write_bytes(b"fake")

        new_negatives = tmp_path / "new.jsonl"
        new_negatives.write_text(
            json.dumps({"path": "missing.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )
        # Don't create missing.wav

        # When all entries have missing files, merge should be aborted
        with pytest.raises(MergeBackError, match="Too many missing files"):
            add_to_training(new_negatives, training_manifest, backup=False)

    def test_invalid_label(self, tmp_path: Path) -> None:
        """Invalid label values should be skipped."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            json.dumps({"path": "existing.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )
        (tmp_path / "existing.wav").write_bytes(b"fake")

        new_negatives = tmp_path / "new.jsonl"
        (tmp_path / "invalid_label.wav").write_bytes(b"fake")
        new_negatives.write_text(
            json.dumps({"path": "invalid_label.wav", "label": 2, "text": "", "duration_ms": 1000})
            + "\n"
        )

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert result.added_count == 0
        assert len(result.errors) == 1
        assert "Invalid label" in result.errors[0]

    def test_empty_new_negatives(self, tmp_path: Path) -> None:
        """Empty new negatives file should return zero additions."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            json.dumps({"path": "existing.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )
        (tmp_path / "existing.wav").write_bytes(b"fake")

        new_negatives = tmp_path / "new.jsonl"
        new_negatives.write_text("")  # Empty file

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert result.added_count == 0
        assert result.total_count == 1

    def test_multiple_entries(self, tmp_path: Path) -> None:
        """Multiple valid entries should all be added."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            json.dumps({"path": "existing.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )
        (tmp_path / "existing.wav").write_bytes(b"fake")

        new_negatives = tmp_path / "new.jsonl"
        entries = [
            {"path": "new1.wav", "label": 0, "text": "new1", "duration_ms": 1000},
            {"path": "new2.wav", "label": 0, "text": "new2", "duration_ms": 1200},
            {"path": "new3.wav", "label": 0, "text": "new3", "duration_ms": 800},
        ]
        new_negatives.write_text("\n".join(json.dumps(e) for e in entries))
        for i in range(1, 4):
            (tmp_path / f"new{i}.wav").write_bytes(b"fake")

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert result.added_count == 3
        assert result.total_count == 4

    def test_relative_path_lookup(self, tmp_path: Path) -> None:
        """Should look up audio files relative to manifest directory."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            json.dumps({"path": "existing.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )
        (tmp_path / "existing.wav").write_bytes(b"fake")

        new_negatives = tmp_path / "new.jsonl"
        # Use relative path
        new_negatives.write_text(
            json.dumps({"path": "subdir/new.wav", "label": 0, "text": "new", "duration_ms": 1000})
            + "\n"
        )
        # Create file relative to manifest directory
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "new.wav").write_bytes(b"fake")

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert result.added_count == 1

    def test_corrupted_training_manifest(self, tmp_path: Path) -> None:
        """Corrupted training manifest should raise MergeBackError."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text("invalid json\n")

        new_negatives = tmp_path / "new.jsonl"
        new_negatives.write_text(
            json.dumps({"path": "new.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )

        with pytest.raises(MergeBackError, match="Failed to load training manifest"):
            add_to_training(new_negatives, training_manifest)

    def test_merge_result_attributes(self, tmp_path: Path) -> None:
        """MergeResult should have correct attributes."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            json.dumps({"path": "existing.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )
        (tmp_path / "existing.wav").write_bytes(b"fake")

        new_negatives = tmp_path / "new.jsonl"
        new_negatives.write_text(
            json.dumps({"path": "new.wav", "label": 0, "text": "new", "duration_ms": 1000}) + "\n"
        )
        (tmp_path / "new.wav").write_bytes(b"fake")

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert isinstance(result, MergeResult)
        assert result.added_count == 1
        assert result.total_count == 2
        assert isinstance(result.errors, list)
        assert isinstance(result.timestamp, str)


class TestEdgeCases:
    """Edge case tests for mining module."""

    def test_exact_chunk_boundary(self, tmp_path: Path) -> None:
        """Audio exactly matching chunk duration should process correctly."""
        with patch("wakeword_workbench.mining.long_audio.librosa") as mock_librosa:
            mock_model = MagicMock(return_value=0.5)
            audio_path = tmp_path / "exact.wav"
            audio_path.write_bytes(b"fake")

            # Exactly 30 seconds (default chunk size)
            mock_librosa.get_duration.return_value = 30.0
            mock_audio = np.ones(480000, dtype=np.float32)
            mock_librosa.load.return_value = (mock_audio, 16000)

            results = process_long_audio(mock_model, audio_path)

            assert len(results) > 0

    def test_very_short_audio(self, tmp_path: Path) -> None:
        """Very short audio (< 1 window) should be handled."""
        with patch("wakeword_workbench.mining.long_audio.librosa") as mock_librosa:
            mock_model = MagicMock(return_value=0.5)
            audio_path = tmp_path / "short.wav"
            audio_path.write_bytes(b"fake")

            mock_librosa.get_duration.return_value = 0.1  # 100ms
            mock_audio = np.ones(1600, dtype=np.float32)
            mock_librosa.load.return_value = (mock_audio, 16000)

            results = process_long_audio(mock_model, audio_path, window_size=16000)

            assert len(results) == 1
            assert results[0].prediction == 0.5

    def test_hop_size_equals_window_size(self, tmp_path: Path) -> None:
        """Non-overlapping windows (hop_size = window_size)."""
        with patch("wakeword_workbench.mining.long_audio.librosa") as mock_librosa:
            mock_model = MagicMock(return_value=0.6)
            audio_path = tmp_path / "no_overlap.wav"
            audio_path.write_bytes(b"fake")

            mock_librosa.get_duration.return_value = 5.0
            mock_audio = np.ones(80000, dtype=np.float32)
            mock_librosa.load.return_value = (mock_audio, 16000)

            results = process_long_audio(mock_model, audio_path, window_size=16000, hop_size=16000)

            # 5 seconds / 1 second per window = 5 windows
            assert len(results) == 5

    def test_threshold_zero(self, tmp_path: Path) -> None:
        """Threshold of zero should select all predictions."""
        with (
            patch("wakeword_workbench.mining.extractor.librosa") as mock_librosa,
            patch("wakeword_workbench.mining.extractor.sf"),
        ):
            audio_path = tmp_path / "test.wav"
            audio_path.write_bytes(b"fake")
            output_dir = tmp_path / "clips"
            output_dir.mkdir()

            mock_audio = np.ones(80000, dtype=np.float32)
            mock_librosa.load.return_value = (mock_audio, 16000)

            predictions = [
                WindowPrediction(
                    timestamp=0.5, prediction=0.1, window_start=8000, window_end=24000
                ),
                WindowPrediction(
                    timestamp=1.5, prediction=0.01, window_start=24000, window_end=40000
                ),
            ]

            results = extract_false_positives(
                predictions,
                audio_path,
                threshold=0.0,
                output_dir=output_dir,
                cooldown_seconds=0.0,
            )

            # Both should be selected (strictly greater than threshold)
            assert len(results) == 2

    def test_all_lines_empty_in_manifest(self, tmp_path: Path) -> None:
        """Manifest with all empty lines should be handled."""
        training_manifest = tmp_path / "training.jsonl"
        training_manifest.write_text(
            json.dumps({"path": "existing.wav", "label": 0, "text": "", "duration_ms": 1000}) + "\n"
        )
        (tmp_path / "existing.wav").write_bytes(b"fake")

        new_negatives = tmp_path / "new.jsonl"
        new_negatives.write_text("\n\n\n")  # Only empty lines

        result = add_to_training(new_negatives, training_manifest, backup=False)

        assert result.added_count == 0
        assert result.total_count == 1
