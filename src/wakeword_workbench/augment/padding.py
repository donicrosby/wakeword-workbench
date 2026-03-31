"""Silence padding and transforms for audio augmentation.

Provides FixedSizeClip to pad or crop audio to exact target length,
and TrimSilence to remove leading/trailing silence.
"""

from __future__ import annotations

import numpy as np


class FixedSizeClip:
    """Clip audio to exact length via padding or cropping.

    Modes:
        pad_or_crop: pad short audio, crop long audio (default, backward compatible)
        pad:         pad with silence to reach target length (alias for pad_or_crop)
        center:      pad equally on both sides, or crop equally from both sides
        pad_only:    never crop — stretch long audio to fit target length

    Jitter adds random temporal offset within the padding area,
    creating training variation for temporal robustness.
    """

    def __init__(self, target_samples: int, *, mode: str = "pad_or_crop", jitter: bool = True):
        """Initialize FixedSizeClip.

        Args:
            target_samples: Desired output length in samples.
            mode: Padding mode:
                - "pad_or_crop": pad short audio, crop long audio (default)
                - "pad": alias for pad_or_crop
                - "center": pad/crop equally on both sides
                - "pad_only": never crop — stretches long audio to fit
            jitter: If True, randomly offset audio within padding region
                    (stretch audio slightly for pad_only mode when jitter=True).
        """
        if target_samples <= 0:
            raise ValueError("target_samples must be positive")
        valid_modes = ("pad_or_crop", "pad", "center", "pad_only")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

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

        if self.mode in ("pad", "pad_or_crop"):
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

        elif self.mode == "pad_only":
            # pad_only mode doesn't need padding when audio is short
            # (this branch should not be reached for short audio, but handle it)
            pad_front = 0
            pad_back = pad_total

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

        elif self.mode == "pad_only":
            # pad_only mode: stretch audio to fit instead of cropping
            return self._stretch(audio, current)

        else:
            # "pad" / "pad_or_crop" mode — crop from end by default
            if self.jitter:
                # jitter: randomly crop from start instead of end
                max_jitter = delta
                crop_front = np.random.randint(0, max_jitter + 1)
                start = crop_front
            else:
                start = 0

        return audio[start : start + self.target_samples]

    def _stretch(self, audio: np.ndarray, current: int) -> np.ndarray:
        """Stretch audio to target length using linear interpolation.

        When audio is longer than target and mode is pad_only, we stretch
        (time-domain resampling) to fit the target length without losing
        any audio content.

        Args:
            audio: The audio array to stretch.
            current: Current length in samples.

        Returns:
            Audio stretched to target_samples length.
        """
        if current == self.target_samples:
            return audio

        # Create evenly-spaced indices over the original audio
        original_indices = np.linspace(0, current - 1, current)

        # Create target-length indices for output positions
        # These sample evenly from 0 to (current - 1)
        output_indices = np.linspace(0, current - 1, self.target_samples)

        if self.jitter:
            # Jitter: slightly vary the output indices
            # ±2% jitter in sampling positions
            jitter_factor = 1.0 + (np.random.random() - 0.5) * 0.04
            output_indices = output_indices * jitter_factor
            # Clip to valid range to avoid extrapolation
            output_indices = np.clip(output_indices, 0, current - 1)

        # Interpolate to get the stretched audio
        stretched = np.interp(output_indices, original_indices, audio)
        return stretched.astype(np.float32)


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
