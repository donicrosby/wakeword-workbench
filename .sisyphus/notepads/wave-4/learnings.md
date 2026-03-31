# Wave 4 Learnings

## Task 17: Audio Loader/Resampler

### Key Implementation Details

1. **Audio Loading with Librosa**
   - `librosa.load(path, sr=None, mono=False)` handles most formats (WAV, FLAC, OGG, MP3)
   - `sr=None` preserves original sample rate for proper resampling later
   - Fallback to audioread when soundfile fails

2. **Resampling**
   - Use `librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)`
   - Default target_sr is 16kHz (wake word standard)

3. **Stereo to Mono Conversion**
   - Use `librosa.to_mono(audio)` for proper channel mixing
   - Important for wake word processing which expects mono

4. **Float32 Conversion**
   - librosa.load may return float64; convert to float32 for memory efficiency
   - soundfile.write expects float64; convert before saving

5. **16-bit PCM WAV Saving**
   - Use `soundfile.write(path, audio, sr, subtype="PCM_16")`
   - This is the standard format for wake word models

6. **Error Handling**
   - Wrap loading in try/except and raise `AudioLoadError`
   - Check file existence before attempting to load
   - Handle corrupted files gracefully

### Testing Notes

- Test with both matching and mismatched sample rates (e.g., 16kHz and 48kHz)
- audioread fallback may have slight numerical differences; use atol=1e-3 for tolerance
- Verify metadata is correctly populated
- Test corrupted file handling
- Test stereo to mono conversion

### Dependencies

- librosa>=0.10,<0.11
- soundfile>=0.12
- numpy>=1.24,<2.0

### Gotchas

- `np.random.randn()` requires integer arguments in numpy 1.24+
- `np.linspace()` num parameter must be int, not float
- audioread fallback (used by librosa) may have slight precision differences
- soundfile.info() may fail for some formats; fallback to librosa channel detection
