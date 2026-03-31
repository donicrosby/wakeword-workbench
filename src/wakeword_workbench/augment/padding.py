"""Silence padding and transforms for audio augmentation.

Provides FixedSizeClip to pad or crop audio to exact target length,
and TrimSilence to remove leading/trailing silence.
"""

from __future__ import annotations

import numpy as np


class FixedSizeClip:
    """Clip audio to exact length via padding or cropping.

    Modes:
        pad:    pad with silence to reach target length
        center: pad equally on both sides, or crop equally from both sides

    Jitter adds random temporal offset within the padding area,
    creating training variation for temporal robustness.
    """

    def __init__(self, target_samples: int, *, mode: str = "pad", jitter: bool = True):
        """Initialize FixedSizeClip.

        Args:
            target_samples: Desired output length in samples.
            mode: Padding mode — "pad" (add silence), "center" (pad/crop to center).
            jitter: If True, randomly offset audio within padding region.
        """
        if target_samples <= 0:
            raise ValueError("target_samples must be positive")
        if mode not in ("pad", "center"):
            raise ValueError(f"mode must be 'pad' or 'center', got {mode!r}")

        self.target_samples = target_samples
        self.mode = mode
        self.jitter = jitter

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Clip audio to target length.

        Args:
            audio: 1-D audio array (samples,).
            sr: Sample rate (unused, kept for API compatibility).

        Returns:
            Audio array of exactly self.target_samples length.
        """
        audio = np.asarray(audio, dtype=np.float32)
        current = audio.shape[0]

        if current == self.target_samples:
            return audio

        if current < self.target_samples:
            return self._pad(audio, current)
        else:
            return self._crop(audio, current)

    def _pad(self, audio: np.ndarray, current: int) -> np.ndarray:
        """Pad audio to target length."""
        delta = self.target_samples - current
        pad_total = delta

        if self.mode == "pad":
            # pad at end by default, jitter shifts left
            if self.jitter:
                # jitter: randomly reduce padding at end, add to beginning
                max_jitter = pad_total
                jitter_front = np.random.randint(0, max_jitter + 1)
            else:
                jitter_front = 0
            pad_front = jitter_front
            pad_back = pad_total - pad_front

        elif self.mode == "center":
            # distribute padding equally on both sides
            pad_front = pad_total // 2
            pad_back = pad_total - pad_front
            # jitter spreads remaining sample across both sides
            if self.jitter and pad_total % 2 == 0:
                # random 1-sample shift to either side
                if np.random.random() < 0.5:
                    pad_front += 1
                    pad_back -= 1

        else:
            pad_front = 0
            pad_back = pad_total

        return np.pad(audio, (pad_front, pad_back), mode="constant", constant_values=0.0)

    def _crop(self, audio: np.ndarray, current: int) -> np.ndarray:
        """Crop audio to target length."""
        delta = current - self.target_samples

        if self.mode == "center":
            # crop equally from both sides
            crop_front = delta // 2
            crop_back = delta - crop_front
            if self.jitter and delta % 2 == 0:
                # random 1-sample shift
                if np.random.random() < 0.5:
                    crop_front += 1
                    crop_back -= 1
            start = crop_front
        else:
            # "pad" mode — crop from end by default
            if self.jitter:
                # jitter: randomly crop from start instead of end
                max_jitter = delta
                crop_front = np.random.randint(0, max_jitter + 1)
                start = crop_front
            else:
                start = 0

        return audio[start : start + self.target_samples]


class TrimSilence:
    """Remove leading and trailing silence from audio.

    Uses librosa.effects.trim internally. Threshold is expressed
    as a proportion of the RMS energy of the signal.
    """

    def __init__(self, threshold_db: float = 20.0):
        """Initialize TrimSilence.

        Args:
            threshold_db: Energy threshold in dB below which
                          regions are considered silence.
        """
        self.threshold_db = threshold_db

    def apply(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trim silence from audio.

        Args:
            audio: 1-D audio array.
            sr: Sample rate.

        Returns:
            Trimmed audio with leading/trailing silence removed.
        """
        try:
            import librosa  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "librosa is required for TrimSilence. Install it with: pip install librosa"
            ) from exc

        import librosa

        trimmed, _ = librosa.effects.trim(audio, top_db=self.threshold_db)
        return np.asarray(trimmed, dtype=audio.dtype)
