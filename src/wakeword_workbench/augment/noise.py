"""Noise injection augmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from wakeword_workbench.logging_config import get_logger

log = get_logger(__name__)


class AddNoise:
    """Add random noise files at specified SNR levels.

    Loads noise files from a directory and mixes them with audio at
    a random SNR within the specified range. Applies with probability p.

    Args:
        noise_dir: Directory containing noise audio files (.wav, .mp3, .flac, etc.)
        snr_range: Min and max SNR in dB (default: (-10, 10)). Higher = less noise.
        p: Probability of applying augmentation (default: 0.75).

    Example:
        >>> noise = AddNoise(Path("noise/"), snr_range=(-10, 10), p=0.75)
        >>> audio_noisy = noise.apply(audio, sr=16000)
    """

    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aiff"}

    def __init__(
        self,
        noise_dir: Path,
        snr_range: tuple[float, float] = (-10, 10),
        p: float = 0.75,
    ) -> None:
        self.noise_dir = Path(noise_dir)
        self.snr_range = snr_range
        self.p = p

        self._noise_files: list[Path] = []
        self._load_noise_files()

    def _load_noise_files(self) -> None:
        """Discover noise files in the noise directory."""
        if not self.noise_dir.exists():
            raise FileNotFoundError(f"Noise directory not found: {self.noise_dir}")

        self._noise_files = [
            f
            for f in self.noise_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not self._noise_files:
            raise ValueError(f"No noise files found in {self.noise_dir}")

        log.debug(
            "noise_files_loaded",
            noise_dir=str(self.noise_dir),
            file_count=len(self._noise_files),
            snr_range=self.snr_range,
            probability=self.p,
        )

    @staticmethod
    def _compute_signal_power(audio: NDArray[np.floating]) -> float:
        """Compute mean squared power of signal."""
        return float(np.mean(np.asarray(audio, dtype=np.float64) ** 2))

    @staticmethod
    def _compute_noise_power(noise: NDArray[np.floating]) -> float:
        """Compute mean squared power of noise."""
        return float(np.mean(np.asarray(noise, dtype=np.float64) ** 2))

    @staticmethod
    def _calculate_scale_factor(
        signal_power: float,
        target_snr_db: float,
    ) -> float:
        """Calculate noise scale factor to achieve target SNR.

        SNR_db = 10 * log10(signal_power / noise_power)
        => noise_power = signal_power / 10^(SNR_db / 10)
        => scale = sqrt(noise_power / current_noise_power)
        """
        target_noise_power = signal_power / (10 ** (target_snr_db / 10))
        return np.sqrt(target_noise_power)

    def _load_noise_file(self, path: Path, target_length: int, sr: int) -> NDArray[np.float32]:
        """Load a noise file and adjust to match target length."""
        import librosa as _librosa

        noise, _noise_sr = _librosa.load(path, sr=sr, mono=True)

        if len(noise) == 0:
            raise ValueError(f"Noise file {path} is empty")

        # Handle length mismatch
        if len(noise) < target_length:
            # Tile/loop the noise to cover the audio
            repeats = int(np.ceil(target_length / len(noise)))
            noise = np.tile(noise, repeats)

        # Truncate to exact length
        noise = noise[:target_length]

        return noise.astype(np.float32)

    def _normalize_output(self, audio: NDArray[np.floating]) -> NDArray[np.float32]:
        """Normalize audio to [-1, 1] range after mixing."""
        audio_f = np.asarray(audio, dtype=np.float64)
        max_val = np.max(np.abs(audio_f))
        if max_val > 1.0:
            audio_f = audio_f / max_val
        return audio_f.astype(np.float32)

    def apply(self, audio: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        """Apply noise augmentation to audio.

        Args:
            audio: Audio samples as float32 array.
            sr: Sample rate in Hz.

        Returns:
            Audio with noise mixed in (if p < random roll), otherwise unchanged.
        """
        if random.random() > self.p:
            return audio.copy()

        audio = np.asarray(audio, dtype=np.float32)

        # Select random noise file
        noise_path = random.choice(self._noise_files)

        # Load and adjust noise to match audio length
        noise = self._load_noise_file(noise_path, len(audio), sr)

        # Compute powers
        signal_power = self._compute_signal_power(audio)
        noise_power = self._compute_noise_power(noise)

        if signal_power <= 0:
            # Audio is silent, can't compute meaningful SNR
            return audio.copy()

        if noise_power <= 0:
            # Noise is silent, skip augmentation
            return audio.copy()

        # Select random SNR within range
        target_snr_db = random.uniform(self.snr_range[0], self.snr_range[1])

        # Calculate scale factor for noise
        scale = self._calculate_scale_factor(signal_power, target_snr_db)
        scaled_noise = noise * scale

        # Mix: output = signal + scaled_noise
        output = audio + scaled_noise

        # Normalize to prevent clipping
        output = self._normalize_output(output)

        return output


class AddColoredNoise:
    """Generate synthetic colored noise (white, pink, brown).

    Does not require external noise files. Uses spectral shaping
    to generate different noise colors.

    Args:
        color: Noise color - "white", "pink", or "brown".
        snr_range: Min and max SNR in dB (default: (-10, 10)).
        p: Probability of applying augmentation (default: 0.5).

    Example:
        >>> pink_noise = AddColoredNoise("pink", snr_range=(-5, 5), p=0.5)
        >>> audio_noisy = pink_noise.apply(audio, sr=16000)
    """

    def __init__(
        self,
        color: Literal["white", "pink", "brown"] = "white",
        snr_range: tuple[float, float] = (-10, 10),
        p: float = 0.5,
    ) -> None:
        self.color = color
        self.snr_range = snr_range
        self.p = p

        log.debug(
            "colored_noise_init",
            color=color,
            snr_range=snr_range,
            probability=p,
        )

    def _generate_white(self, length: int) -> NDArray[np.float32]:
        """Generate white noise (flat spectrum)."""
        return np.random.randn(length).astype(np.float32) * 0.5

    def _generate_pink(self, length: int) -> NDArray[np.float32]:
        """Generate pink noise (1/f spectrum) using IIR filter."""
        white = np.random.randn(length).astype(np.float32)

        # Paul Kellet's refined IIR filter coefficients for pink noise
        b = np.array([0.021128, 0.07872, 0.15877, 0.2118, 0.15877, 0.07872, 0.021128])
        a = np.array([1.0, -1.747, 1.823, -1.129, 0.874, -0.514, 0.178])

        try:
            from scipy.signal import lfilter

            pink = lfilter(b, a, white)
            return pink.astype(np.float32)
        except ImportError:
            # Fallback: simple IIR approximation
            pink = np.zeros(length, dtype=np.float32)
            b_prev = np.zeros(7, dtype=np.float32)
            for i in range(length):
                white_sample = white[i]
                # Apply IIR directly in loop
                b_prev[0] = 0.021128 * white_sample + b_prev[0]
                b_prev[1] = 0.07872 * white_sample + b_prev[1]
                b_prev[2] = 0.15877 * white_sample + b_prev[2]
                b_prev[3] = 0.2118 * white_sample + b_prev[3]
                b_prev[4] = 0.15877 * white_sample + b_prev[4]
                b_prev[5] = 0.07872 * white_sample + b_prev[5]
                b_prev[6] = 0.021128 * white_sample + b_prev[6]
                pink[i] = (
                    b_prev[0]
                    - 1.747 * b_prev[1]
                    + 1.823 * b_prev[2]
                    - 1.129 * b_prev[3]
                    + 0.874 * b_prev[4]
                    - 0.514 * b_prev[5]
                    + 0.178 * b_prev[6]
                )
            return pink

    def _generate_brown(self, length: int) -> NDArray[np.float32]:
        """Generate brown noise (1/f^2 spectrum) using integration."""
        white = np.random.randn(length).astype(np.float64)
        # Integrate to get brown noise
        brown = np.cumsum(white)
        # Normalize and scale
        brown = brown - np.mean(brown)
        std = np.std(brown)
        if std > 0:
            brown = brown / std
        return brown.astype(np.float32) * 0.25

    def _generate_colored_noise(self, length: int) -> NDArray[np.float32]:
        """Generate colored noise of specified type."""
        if self.color == "white":
            return self._generate_white(length)
        elif self.color == "pink":
            return self._generate_pink(length)
        elif self.color == "brown":
            return self._generate_brown(length)
        else:
            raise ValueError(f"Unknown noise color: {self.color}")

    @staticmethod
    def _compute_signal_power(audio: NDArray[np.floating]) -> float:
        """Compute mean squared power of signal."""
        return float(np.mean(np.asarray(audio, dtype=np.float64) ** 2))

    @staticmethod
    def _compute_noise_power(noise: NDArray[np.floating]) -> float:
        """Compute mean squared power of noise."""
        return float(np.mean(np.asarray(noise, dtype=np.float64) ** 2))

    @staticmethod
    def _calculate_scale_factor(
        signal_power: float,
        target_snr_db: float,
    ) -> float:
        """Calculate noise scale factor to achieve target SNR."""
        target_noise_power = signal_power / (10 ** (target_snr_db / 10))
        return np.sqrt(target_noise_power)

    def _normalize_output(self, audio: NDArray[np.floating]) -> NDArray[np.float32]:
        """Normalize audio to [-1, 1] range after mixing."""
        audio_f = np.asarray(audio, dtype=np.float64)
        max_val = np.max(np.abs(audio_f))
        if max_val > 1.0:
            audio_f = audio_f / max_val
        return audio_f.astype(np.float32)

    def apply(self, audio: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        """Apply colored noise augmentation to audio.

        Args:
            audio: Audio samples as float32 array.
            sr: Sample rate in Hz.

        Returns:
            Audio with colored noise mixed in (if p < random roll), otherwise unchanged.
        """
        if random.random() > self.p:
            return audio.copy()

        audio = np.asarray(audio, dtype=np.float32)

        # Generate colored noise
        noise = self._generate_colored_noise(len(audio))

        # Compute powers
        signal_power = self._compute_signal_power(audio)
        noise_power = self._compute_noise_power(noise)

        if signal_power <= 0 or noise_power <= 0:
            return audio.copy()

        # Select random SNR within range
        target_snr_db = random.uniform(self.snr_range[0], self.snr_range[1])

        # Calculate scale factor
        scale = self._calculate_scale_factor(signal_power, target_snr_db)
        scaled_noise = noise * scale

        # Mix
        output = audio + scaled_noise

        # Normalize
        output = self._normalize_output(output)

        return output
