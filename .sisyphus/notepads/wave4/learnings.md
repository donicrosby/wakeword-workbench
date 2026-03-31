
## Task 18: Noise Augmentation (Wave 4)
- AddNoise class: SNR mixing with probability parameter p
- SNR formula: SNR_db = 10 * log10(signal_power / noise_power)
- Scale factor: sqrt(signal_power / 10^(SNR_db/10))
- Noise tiling for shorter files, truncation for longer
- Output normalized to [-1, 1] after mixing
- AddColoredNoise: white/pink/brown using scipy.signal.lfilter
- Pink noise fallback: manual IIR loop when scipy unavailable
