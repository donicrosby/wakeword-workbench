"""Tests for reverb augmentation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wakeword_workbench.augment.reverb import (
    AddReverb,
    ReverbError,
    RIRCache,
)


class TestRIRCache:
    """Tests for RIRCache."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = RIRCache(max_size=10)
        assert cache.get(Path("test.wav")) is None
        assert cache._max_size == 10

    def test_cache_put_and_get(self):
        """Test cache stores and retrieves RIRs."""
        cache = RIRCache()
        rir = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        path = Path("test.wav")

        cache.put(path, rir)
        retrieved = cache.get(path)

        np.testing.assert_array_equal(retrieved, rir)

    def test_cache_eviction(self):
        """Test cache evicts oldest entries when full."""
        cache = RIRCache(max_size=2)

        rir1 = np.array([1.0], dtype=np.float32)
        rir2 = np.array([2.0], dtype=np.float32)
        rir3 = np.array([3.0], dtype=np.float32)

        cache.put(Path("1.wav"), rir1)
        cache.put(Path("2.wav"), rir2)
        cache.put(Path("3.wav"), rir3)  # Should evict rir1

        assert cache.get(Path("1.wav")) is None
        assert cache.get(Path("2.wav")) is not None
        assert cache.get(Path("3.wav")) is not None

    def test_cache_lru_order(self):
        """Test cache respects LRU eviction order."""
        cache = RIRCache(max_size=3)

        rir = np.array([1.0], dtype=np.float32)
        cache.put(Path("1.wav"), rir)
        cache.put(Path("2.wav"), rir.copy())
        cache.put(Path("3.wav"), rir.copy())

        # Access 1.wav to make it recently used
        _ = cache.get(Path("1.wav"))

        # Add new item - should evict 2.wav (oldest after 1 was accessed)
        cache.put(Path("4.wav"), rir.copy())

        assert cache.get(Path("1.wav")) is not None  # Recently used
        assert cache.get(Path("2.wav")) is None  # Evicted


class TestAddReverb:
    """Tests for AddReverb class."""

    @pytest.fixture
    def rir_dir(self):
        """Create a temporary directory with mock RIR files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rir_path = Path(tmpdir)

            # Create simple RIR files (synthetic impulse responses)
            for i in range(3):
                # Simple synthetic RIR with exponential decay
                sr = 16000
                t = np.linspace(0, 0.5, int(0.5 * sr))
                rir = np.exp(-10 * t) * np.random.randn(len(t)).astype(np.float32)
                rir[0] = 1.0  # Strong initial impulse

                sf.write(rir_path / f"rir_{i}.wav", rir, sr)

            yield rir_path

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio for testing."""
        sr = 16000
        duration = 1.0
        # Simple sine wave
        t = np.linspace(0, duration, int(duration * sr))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio

    def test_initialization_success(self, rir_dir):
        """Test AddReverb initializes correctly with valid directory."""
        reverb = AddReverb(rir_dir, p=0.5)
        assert reverb._p == 0.5
        assert len(reverb._available_rirs) == 3

    def test_initialization_missing_dir(self):
        """Test AddReverb raises error for missing directory."""
        with pytest.raises(ReverbError, match="RIR directory not found"):
            AddReverb(Path("/nonexistent/path"), p=0.5)

    def test_initialization_empty_dir(self):
        """Test AddReverb raises error for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ReverbError, match="No RIR files"):
                AddReverb(Path(tmpdir), p=0.5)

    def test_apply_returns_same_length(self, rir_dir, sample_audio):
        """Test apply returns audio of same length as input."""
        reverb = AddReverb(rir_dir, p=1.0)
        result = reverb.apply(sample_audio, sr=16000)

        assert len(result) == len(sample_audio)

    def test_apply_respects_probability(self, rir_dir, sample_audio):
        """Test that reverb is applied based on probability."""
        # With p=0, should always return original
        reverb_never = AddReverb(rir_dir, p=0.0)
        for _ in range(10):
            result = reverb_never.apply(sample_audio, sr=16000)
            np.testing.assert_array_equal(result, sample_audio)

        # With p=1, should always apply reverb
        reverb_always = AddReverb(rir_dir, p=1.0)
        applied_count = 0
        for _ in range(10):
            result = reverb_always.apply(sample_audio, sr=16000)
            if not np.array_equal(result, sample_audio):
                applied_count += 1

        # Should be different (applied reverb)
        assert applied_count == 10, "Reverb should always be applied with p=1.0"

    def test_apply_adds_reverb_tail(self, rir_dir, sample_audio):
        """Test that reverb adds audible tail to audio."""
        reverb = AddReverb(rir_dir, p=1.0)

        # Apply reverb multiple times to ensure we get different audio
        # (since probability can cause skip)
        results = set()
        for _ in range(20):
            result = reverb.apply(sample_audio, sr=16000)
            # If result differs from original, reverb was applied
            if not np.array_equal(result, sample_audio):
                results.add(tuple(result[:100]))  # Sample first 100 samples

        # Should have at least some different results due to random RIR selection
        assert len(results) >= 1, "Reverb should produce different output from input"

    def test_apply_handles_stereo_rir(self, rir_dir, sample_audio):
        """Test that stereo RIRs are handled correctly."""
        # Add a stereo RIR file
        rir_path = rir_dir / "stereo_rir.wav"
        sr = 16000
        t = np.linspace(0, 0.5, int(0.5 * sr))
        rir = np.column_stack(
            [
                np.exp(-10 * t) * np.random.randn(len(t)),
                np.exp(-10 * t) * np.random.randn(len(t)),
            ]
        ).astype(np.float32)
        rir[0, 0] = 1.0
        rir[0, 1] = 1.0
        sf.write(rir_path, rir, sr)

        reverb = AddReverb(rir_dir, p=1.0)
        result = reverb.apply(sample_audio, sr=16000)

        assert len(result) == len(sample_audio)
        assert result.dtype == np.float32

    def test_apply_preserves_dtype(self, rir_dir, sample_audio):
        """Test that output dtype is float32."""
        reverb = AddReverb(rir_dir, p=1.0)
        result = reverb.apply(sample_audio, sr=16000)

        assert result.dtype == np.float32

    def test_apply_no_clipping(self, rir_dir, sample_audio):
        """Test that reverb output doesn't clip."""
        reverb = AddReverb(rir_dir, p=1.0)

        # Run multiple times to catch any potential clipping
        for _ in range(10):
            result = reverb.apply(sample_audio, sr=16000)
            assert np.abs(result).max() <= 1.0, "Audio should not clip"


class TestRT60Estimation:
    """Tests for RT60 estimation."""

    def test_estimate_rt60_short_rir(self):
        """Test RT60 estimation on short RIR."""
        # Short RIR
        rir = np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.0], dtype=np.float32)
        rt60 = AddReverb.estimate_rt60(rir)

        assert rt60 >= 0
        assert rt60 <= 1.0

    def test_estimate_rt60_empty_rir(self):
        """Test RT60 estimation on empty RIR."""
        rir = np.array([], dtype=np.float32)
        rt60 = AddReverb.estimate_rt60(rir)

        assert rt60 == 0.0

    def test_estimate_rt60_no_decay(self):
        """Test RT60 estimation on RIR with no decay."""
        rir = np.array([0.0], dtype=np.float32)
        rt60 = AddReverb.estimate_rt60(rir)

        assert rt60 == 0.0
