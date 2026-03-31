"""Reverberation augmentation via Room Impulse Response (RIR) convolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from wakeword_workbench.logging_config import get_logger

log = get_logger(__name__)


class ReverbError(Exception):
    """Raised when reverb augmentation fails."""

    pass


class RIRCache:
    """Cache for loaded Room Impulse Response files."""

    def __init__(self, max_size: int = 100):
        """Initialize RIR cache.

        Args:
            max_size: Maximum number of RIRs to cache.
        """
        self._cache: dict[Path, np.ndarray] = {}
        self._max_size = max_size
        self._access_order: list[Path] = []

    def get(self, path: Path) -> np.ndarray | None:
        """Get cached RIR or None if not cached."""
        result = self._cache.get(path)
        if result is not None and path in self._access_order:
            # Update LRU order - move to end
            self._access_order.remove(path)
            self._access_order.append(path)
        return result

    def put(self, path: Path, rir: np.ndarray) -> None:
        """Cache an RIR, evicting oldest if at capacity."""
        if path in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(path)
            self._access_order.append(path)
            return

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[path] = rir
        self._access_order.append(path)


class AddReverb:
    """Reverberation augmentation via RIR convolution.

    Simulates acoustic environments by convolving audio with Room Impulse Responses
    (RIRs), adding realistic reverb tails that mimic real room acoustics.

    Supports common RIR datasets (MIT, RVB, SARDINA) and loads from user-specified
    directories containing WAV files.

    Example:
        >>> reverb = AddReverb(Path("rir/"), p=0.5)
        >>> audio_reverb = reverb.apply(audio, sr=16000)
    """

    def __init__(self, rir_dir: Path | str, p: float = 0.5, max_cache: int = 100):
        """Initialize reverb augmentor.

        Args:
            rir_dir: Directory containing RIR WAV files.
            p: Probability of applying reverb (0.0 to 1.0). Default: 0.5.
            max_cache: Maximum number of RIRs to cache. Default: 100.

        Raises:
            ReverbError: If rir_dir doesn't exist or contains no WAV files.
        """
        self._rir_dir = Path(rir_dir)
        self._p = p
        self._cache = RIRCache(max_size=max_cache)
        self._available_rirs: list[Path] = []

        self._load_rir_index()

    def _load_rir_index(self) -> None:
        """Load all available RIR files from directory."""
        if not self._rir_dir.exists():
            raise ReverbError(f"RIR directory not found: {self._rir_dir}")

        rir_files = sorted(self._rir_dir.glob("*.wav"))
        rir_files.extend(sorted(self._rir_dir.glob("*.flac")))

        if not rir_files:
            raise ReverbError(f"No RIR files (.wav, .flac) found in: {self._rir_dir}")

        self._available_rirs = rir_files
        log.debug("rir_files_loaded", count=len(rir_files), rir_dir=str(self._rir_dir))

    def _load_rir(self, path: Path, target_sr: int) -> np.ndarray:
        """Load and optionally resample an RIR file.

        Args:
            path: Path to RIR file.
            target_sr: Target sample rate.

        Returns:
            RIR array (normalized, mono).
        """
        cached = self._cache.get(path)
        if cached is not None:
            rir = cached
        else:
            try:
                rir, rir_sr = sf.read(path, dtype="float32")
            except Exception as e:
                raise ReverbError(f"Failed to load RIR from '{path}': {e}") from e

            # Convert stereo to mono if needed
            if rir.ndim > 1:
                rir = np.mean(rir, axis=1)

            # Normalize RIR
            rir = rir / (np.abs(rir).max() + 1e-10)

            self._cache.put(path, rir)
            log.debug("rir_cached", path=str(path), sr=rir_sr)

        return rir

    def _resample_rir(self, rir: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample RIR to target sample rate.

        Args:
            rir: RIR array.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled RIR.
        """
        if orig_sr == target_sr:
            return rir

        # Use linear interpolation for efficient resampling
        duration = len(rir) / orig_sr
        new_length = int(duration * target_sr)
        indices = np.linspace(0, len(rir) - 1, new_length)
        return np.interp(indices, np.arange(len(rir)), rir)

    def _select_random_rir(self) -> Path:
        """Select a random RIR file.

        Returns:
            Path to selected RIR.
        """
        import random

        return random.choice(self._available_rirs)

    @staticmethod
    def estimate_rt60(rir: np.ndarray) -> float:
        """Estimate RT60 (reverb decay time) from RIR.

        RT60 is the time for reverberation to decay by 60dB.

        Args:
            rir: Room Impulse Response (normalized, mono).

        Returns:
            RT60 in seconds, or 0.0 if cannot be estimated.
        """
        if len(rir) == 0:
            return 0.0

        # Find the peak (direct sound)
        abs_rir = np.abs(rir)
        peak_idx = np.argmax(abs_rir)

        if peak_idx >= len(rir):
            return 0.0

        # Start from peak, find where energy falls to -60dB
        peak_val = abs_rir[peak_idx]
        if peak_val < 1e-10:
            return 0.0

        threshold = peak_val * 10 ** (-60 / 20)  # -60 dB threshold

        # Find last sample above threshold
        above_threshold = abs_rir[peak_idx:] > threshold
        if not np.any(above_threshold):
            return len(rir) / 16000  # Fallback: use full length

        tail_length = np.sum(above_threshold)
        # Approximate RT60 (assumes uniform decay)
        rt60_samples = tail_length * 2  # Rough estimate
        return rt60_samples / 16000

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply reverb augmentation to audio.

        Convolves the audio with a randomly selected RIR, simulating
        room acoustics. With probability (1-p), returns the original
        audio unchanged.

        Args:
            audio: Input audio as numpy array (float32, mono, values in [-1, 1]).
            sr: Sample rate in Hz.

        Returns:
            Audio with reverb applied (or original if probability check fails).
            Same shape and sample rate as input.

        Raises:
            ReverbError: If convolution fails.
        """
        import random

        # Probability check - skip augmentation
        if random.random() > self._p:
            return audio

        # Select and load RIR
        rir_path = self._select_random_rir()

        try:
            rir, rir_sr = sf.read(rir_path, dtype="float32")
        except Exception as e:
            log.warning("rir_load_failed", path=str(rir_path), error=str(e))
            return audio

        # Handle stereo RIRs (average to mono)
        if rir.ndim > 1:
            rir = np.mean(rir, axis=1)

        # Normalize RIR
        rir = rir / (np.abs(rir).max() + 1e-10)

        # Resample RIR if needed
        if rir_sr != sr:
            rir = self._resample_rir(rir, rir_sr, sr)

        # Compute reverb using overlap-add (fftconvolve is faster for long IRs)
        # For short audio and typical RIRs, direct convolution is fine
        try:
            # Use 'same' mode to keep output same length as input
            reverb_audio = scipy.signal.fftconvolve(audio, rir, mode="same")
        except Exception as e:
            raise ReverbError(f"Convolution failed: {e}") from e

        # Normalize output to prevent clipping
        max_val = np.abs(reverb_audio).max()
        if max_val > 0.99:
            reverb_audio = reverb_audio / (max_val * 1.1)

        # Log room characteristics
        rt60 = self.estimate_rt60(rir)
        log.debug("reverb_applied", rir_path=rir_path.name, rt60=round(rt60, 2), sr=sr)

        return reverb_audio.astype(np.float32)
