"""Gain/volume augmentation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class AdjustGain:
    """Apply random gain (volume adjustment) to audio.

    Gain is applied in dB and converted to linear scale for multiplication.

    Args:
        gain_range: Tuple of (min_dB, max_dB) for random gain selection.
            Typical range is (-45, 0) to simulate distance from microphone.
        p: Probability of applying gain. Default 1.0 (always apply).

    Example:
        >>> gain = AdjustGain(gain_range=(-45, 0), p=1.0)
        >>> audio_gain = gain.apply(audio, sr=16000)  # Random gain from -45 to 0 dB
    """

    def __init__(self, gain_range: tuple[float, float] = (-45, 0), p: float = 1.0) -> None:
        self.gain_range = gain_range
        self.p = p

    def apply(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Apply gain to audio.

        Args:
            audio: Input audio as float32 array, values in [-1, 1].
            sr: Sample rate (used for transition timing if applicable).

        Returns:
            Audio with gain applied.
        """
        if np.random.random() > self.p:
            return audio

        # Random gain in dB
        gain_db = np.random.uniform(self.gain_range[0], self.gain_range[1])

        # Convert dB to linear scale: output = input * 10^(gain_db/20)
        linear_gain = 10 ** (gain_db / 20.0)

        return (audio * linear_gain).astype(np.float32)


class SoftClip:
    """Apply soft clipping using tanh saturation.

    Soft clipping smoothly saturates peaks using tanh, producing less
    harsh artifacts than hard clipping.

    Args:
        threshold: Input level at which clipping begins (0 to 1).
            Default 0.8.

    Example:
        >>> clipper = SoftClip(threshold=0.8)
        >>> audio_clipped = clipper.apply(audio)
    """

    def __init__(self, threshold: float = 0.8) -> None:
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.threshold = threshold

    def apply(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply soft clipping to audio.

        Uses tanh for smooth saturation curve.

        Args:
            audio: Input audio as float32 array.

        Returns:
            Soft-clipped audio with peaks smoothly compressed.
        """
        # Scale input relative to threshold
        # When input = threshold, output = threshold (tanh(1) ≈ 0.76)
        # This creates gradual saturation above threshold
        scaled = audio / self.threshold
        clipped = np.tanh(scaled) * self.threshold
        return clipped.astype(np.float32)


class HardClip:
    """Apply hard clipping using np.clip.

    Hard limiting that cuts off peaks at specified bounds.

    Args:
        min_val: Minimum output value. Default -1.0.
        max_val: Maximum output value. Default 1.0.

    Example:
        >>> clipper = HardClip(min_val=-1.0, max_val=1.0)
        >>> audio_clipped = clipper.apply(audio)
    """

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0) -> None:
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply hard clipping to audio.

        Args:
            audio: Input audio as float32 array.

        Returns:
            Hard-clipped audio with values clamped to [min_val, max_val].
        """
        return np.clip(audio, self.min_val, self.max_val).astype(np.float32)


class GainTransition:
    """Apply gain with smooth fade in/out transitions.

    Simulates distance changes by smoothly ramping gain at start/end
    of audio or at specified points.

    Args:
        gain_range: Tuple of (min_dB, max_dB) for random gain selection.
        fade_samples: Number of samples for fade in/out. Default 1000.
        transition_prob: Probability of applying transition. Default 0.5.
            When not applied, uses constant gain.

    Example:
        >>> gt = GainTransition(gain_range=(-30, 0), fade_samples=1000)
        >>> audio_gain = gt.apply(audio, sr=16000)
    """

    def __init__(
        self,
        gain_range: tuple[float, float] = (-30, 0),
        fade_samples: int = 1000,
        transition_prob: float = 0.5,
    ) -> None:
        self.gain_range = gain_range
        self.fade_samples = max(1, fade_samples)
        self.transition_prob = transition_prob

    def apply(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Apply gain with smooth fade transitions.

        Args:
            audio: Input audio as float32 array.
            sr: Sample rate (used to scale fade duration).

        Returns:
            Audio with gain applied, including fade in/out.
        """
        # Random gain in dB
        gain_db = np.random.uniform(self.gain_range[0], self.gain_range[1])
        linear_gain = 10 ** (gain_db / 20.0)

        # Apply base gain
        gained = audio * linear_gain

        # Optionally add fade transition
        if np.random.random() < self.transition_prob:
            gained = self._apply_fade(gained)

        return gained.astype(np.float32)

    def _apply_fade(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply fade in/out to audio."""
        n = len(audio)
        if n <= 2 * self.fade_samples:
            # Audio too short for full fade, use what we can
            fade_len = max(1, n // 2)
        else:
            fade_len = self.fade_samples

        # Ensure we don't exceed array bounds
        fade_len = min(fade_len, n)

        # Create fade envelope
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)

        # Apply fade
        result = audio.copy()
        result[:fade_len] *= fade_in
        result[-fade_len:] *= fade_out

        return result
