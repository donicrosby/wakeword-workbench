# Task 4 Evidence: WakeWord Workbench Feature Compatibility Matrix

## Executive Summary

This document provides a comprehensive feature compatibility matrix showing which WakeWord Workbench features are compatible with each training harness (microWakeWord and openWakeWord), along with recommendations, known limitations, and actionable guidance for users.

**Key Finding**: Neither harness can directly consume workbench exports without additional processing. Workbench produces intermediate artifacts that require harness-specific adapters to convert into training-ready formats.

---

## Compatibility Legend

| Symbol | Meaning |
|--------|---------|
| ✅ **Full** | Feature works without limitations |
| ⚠️ **Partial** | Feature works with caveats or requires additional steps |
| ❌ **Incompatible** | Feature cannot be used with this harness |
| 🔄 **Adapter Required** | Feature output needs conversion before harness can use it |
| 🔧 **Harness Native** | This feature is better handled by the harness itself |

---

## 1. Data Generation Features

### 1.1 TTS Backends

| Feature | microWakeWord | openWakeWord | Notes |
|---------|---------------|--------------|-------|
| Kokoro Backend | ✅ Full | ✅ Full | Both harnesses can use Kokoro-generated audio |
| Piper Backend | ✅ Full | ✅ Full | Both harnesses can use Piper-generated audio |
| Voice Variation | ✅ Full | ✅ Full | Multiple voices increase dataset diversity |
| Speed Control | ✅ Full | ✅ Full | Speed variation creates additional diversity |
| TTS Caching | ✅ Full | ✅ Full | Workbench caching speeds up re-generation |

**Recommendation**: Use multiple voices (3-5) and slight speed variations (0.9x-1.1x) for both harnesses. The TTS cache prevents redundant synthesis.

**Known Limitations**: None.

---

### 1.2 Positive Sample Generation

| Feature | microWakeWord | openWakeWord | Notes |
|---------|---------------|--------------|-------|
| Base Phrase Generation | ✅ Full | ✅ Full | Core TTS synthesis works for both |
| Phrase Variant Generation | ✅ Full | ✅ Full | Punctuation, casing variations work |
| Workbench Export | 🔄 Adapter Required | 🔄 Adapter Required | Neither can directly consume workbench output |

**Integration Paths**:

- microWakeWord: Workbench TTS → WAV files → SpectrogramGeneration → RaggedMmap
- openWakeWord: Workbench TTS → WAV files → AudioFeatures.embed_clips → .npy files

**Recommendation**: For both harnesses, use workbench TTS to generate WAV files, then let each harness handle its own feature extraction. Do NOT use workbench built-in mel/mmap exporters for training data.

---

### 1.3 Negative Sample Generation

| Feature | microWakeWord | openWakeWord | Notes |
|---------|---------------|--------------|-------|
| Phonetic Confusion Generator | ✅ Full | ✅ Full | Both benefit from phonetically similar negatives |
| Synthetic Negative Generator | ✅ Full | ✅ Full | Random phrase negatives work for both |
| Adversarial Negatives | 🔄 Adapter Required | 🔄 Adapter Required | Must be converted to harness-specific format |

**microWakeWord Negative Strategy**:
- Uses pre-generated negative RaggedMmap datasets (from Hugging Face)
- Adversarial negatives (phonetic confusions) should be generated and converted to RaggedMmap
- Custom negative phrases can be TTS-synthesized and added to training set

**openWakeWord Negative Strategy**:
- Uses large negative corpora (ACAV100M_sample) via feature_data_files
- Adversarial negatives mixed via batch_n_per_class configuration
- Phonetic confusions should be TTS-synthesized and embedded via AudioFeatures

**Recommendation**:
- microWakeWord: Download official negative datasets, supplement with workbench-generated adversarial negatives converted to RaggedMmap
- openWakeWord: Use workbench to generate adversarial negatives, embed with AudioFeatures.embed_clips(), add to feature_data_files config


---

## 2. Audio Augmentation Features

### 2.1 Augmentation Transforms

| Feature | microWakeWord | openWakeWord | Notes |
|---------|---------------|--------------|-------|
| AddNoise (file-based) | 🔧 Harness Native | ✅ Full | microWakeWord has own pipeline |
| AddColoredNoise | 🔧 Harness Native | ✅ Full | microWakeWord has own pipeline |
| AddReverb (RIR) | 🔧 Harness Native | ✅ Full | microWakeWord has own pipeline |
| AdjustGain | 🔧 Harness Native | ✅ Full | microWakeWord has own pipeline |
| FixedSizeClip | 🔧 Harness Native | ⚠️ Partial | Clip size matters differently |

**Critical Finding**: microWakeWord has its own comprehensive augmentation pipeline (Augmentation class with EQ, distortion, pitch, noise, reverb, gain, jitter). Using workbench augmentation before microWakeWord pipeline would result in double augmentation.

**Recommendation**:
- microWakeWord: DO NOT use workbench augmentation — use microWakeWord native augmentation instead
- openWakeWord: USE workbench augmentation — apply before exporting to openWakeWord format

---

### 2.2 Augmentation Configuration

| Configuration | microWakeWord | openWakeWord | Notes |
|---------------|---------------|--------------|-------|
| Noise SNR Range | 🔧 Harness Native | ✅ Full | Configure in workbench |
| Reverb Probability | 🔧 Harness Native | ✅ Full | Configure in workbench |
| Gain Range | 🔧 Harness Native | ✅ Full | Configure in workbench |
| Background Paths | 🔧 Harness Native | ✅ Full | openWakeWord uses background_paths config |
| RIR Paths | 🔧 Harness Native | ✅ Full | openWakeWord uses rir_paths config |

**microWakeWord Augmentation Settings** (from notebook):
- Background noise mixing
- Room impulse response convolution
- Gain variation
- EQ, distortion, pitch shifts
- Jitter (time shifting)

**openWakeWord Augmentation Settings** (via workbench):
- Configured in augmentation section of workbench config
- Applied during workbench pipeline execution
- Exported as augmented audio files

---

## 3. Dataset Management Features

### 3.1 Manifest System

| Feature | microWakeWord | openWakeWord | Notes |
|---------|---------------|--------------|-------|
| JSONL Manifests | ⚠️ Partial | ⚠️ Partial | Both need conversion to harness format |
| Train/Val/Test Split | ✅ Full | ✅ Full | Workbench splitter works for both |
| Speaker Leakage Prevention | ✅ Full | ✅ Full | Critical for both harnesses |
| Dataset Merger | ✅ Full | ✅ Full | Combine positives + negatives |

**Manifest Conversion Requirements**:

microWakeWord: Workbench manifest → microWakeWord Clips + SpectrogramGeneration → RaggedMmap

openWakeWord: Workbench manifest → openWakeWord feature extraction → class-specific .npy files

**Recommendation**: Use workbench dataset splitter and merger to create balanced datasets, then convert manifests to harness-specific formats.

---

### 3.2 Dataset Splits

| Split | microWakeWord | openWakeWord | Notes |
|-------|---------------|--------------|-------|
| training/ | ✅ Required | ✅ Required | Training data |
| validation/ | ✅ Required | ✅ Required | Validation data |
| testing/ | ✅ Required | ✅ Required | Test data |
| validation_ambient/ | ✅ Optional | ❌ N/A | microWakeWord-specific |
| testing_ambient/ | ✅ Optional | ❌ N/A | microWakeWord-specific |

**microWakeWord Directory Structure**:
```
output/
├── training/
│   ├── wakeword_mmap/      # RaggedMmap folder
│   └── negative_mmap/      # RaggedMmap folder
├── validation/
│   └── wakeword_mmap/
├── validation_ambient/     # Optional
└── testing/
    └── wakeword_mmap/
```

**openWakeWord Directory Structure**:
```
output/
├── positive_features_train.npy
├── positive_features_val.npy
├── adversarial_negative_train.npy
├── adversarial_negative_val.npy
└── custom_model.yml        # Config with feature_data_files
```


---

## 4. Hard Negative Mining Features

### 4.1 Mining Capabilities

| Feature | microWakeWord | openWakeWord | Notes |
|---------|---------------|--------------|-------|
| Long Audio Processing | ✅ Full | ✅ Full | Sliding window detection works for both |
| False Positive Extraction | ✅ Full | ✅ Full | Clip extraction is harness-agnostic |
| Cooldown Deduplication | ✅ Full | ✅ Full | Prevents duplicate extractions |
| Merging Back to Dataset | 🔄 Adapter Required | 🔄 Adapter Required | Must convert to harness format |

**Integration Path**:
1. Mine false positives using trained model (from either harness)
2. Extract audio clips with workbench mining tools
3. Convert clips to harness-specific format:
   - microWakeWord: Generate RaggedMmap from mined clips
   - openWakeWord: Run AudioFeatures.embed_clips() on mined clips

**Recommendation**: Hard negative mining works well with both harnesses. The key is converting mined clips to the appropriate format before adding to training data.

---

## 5. Export Features

### 5.1 Current Export Capabilities

| Export Format | microWakeWord Compatible | openWakeWord Compatible | Status |
|---------------|-------------------------|------------------------|---------|
| export_to_mmap (flat audio) | ❌ No | ❌ No | Neither uses flat audio mmap |
| export_with_features (mel) | ❌ No | ❌ No | Feature mismatch for both |
| export_to_numpy (raw/mel) | ❌ No | ❌ No | Format mismatch |
| Direct WAV output | ✅ Yes | ✅ Yes | Best approach for both |

**Feature Mismatches**:

microWakeWord expects:
- RaggedMmap folder structure with per-clip spectrogram arrays
- microfrontend 40-dim features (16kHz, 30ms window, 10ms step)
- Shape: (T, 40) per clip, stored in RaggedMmap format

openWakeWord expects:
- Per-class .npy files with embedded features
- Google speech embeddings (96-dim) via AudioFeatures
- Shape: (N, 16, 96) for 2-second clips at default settings

Workbench produces:
- librosa mel-spectrograms (40 or 96 mel bins, hop_length=480)
- Flat concatenated arrays with index sidecars
- Not directly compatible with either harness

**Recommendation**: DO NOT use workbench exporters for training data. Instead, export WAV files and let each harness handle feature extraction.

---

## 6. Evaluation Features

### 6.1 Evaluation Metrics

| Feature | microWakeWord | openWakeWord | Notes |
|---------|---------------|--------------|-------|
| FAR Calculator | ✅ Full | ✅ Full | Both use false accepts per hour |
| FRR Calculator | ✅ Full | ✅ Full | Both use false reject rate |
| ROC Curve Generation | ✅ Full | ✅ Full | Both benefit from ROC analysis |
| Threshold Optimization | ✅ Full | ✅ Full | EER calculation works for both |
| Evaluation Reports | ✅ Full | ✅ Full | Harness-agnostic reporting |

**Recommendation**: Evaluation tools are fully compatible with both harnesses. Use workbench eval tools for model assessment regardless of training harness.

---

## 7. Recommended Configurations

### 7.1 For microWakeWord

```yaml
# Workbench config for microWakeWord
wake_word: "hey assistant"

samples:
  positives: 500               # More positives for microWakeWord
  negatives_multiplier: 3      # Download official negatives separately

tts:
  backend: "kokoro"
  voices:
    - "af_sarah"
    - "am_adam"
    - "af_bella"
  speed: 1.0

augmentation:
  # ⚠️ DISABLE for microWakeWord - use native augmentation instead
  noise_snr: [0, 0]
  reverb_probability: 0.0
  gain_range: [0, 0]

output:
  path: "./output/microwakeword"
  format: []                   # Do not use workbench exporters
```

**Post-Workbench Steps**:
1. Workbench generates WAV files to output directory
2. Load audio into microWakeWord's Clips class
3. Use microWakeWord's SpectrogramGeneration with built-in augmentation
4. Save as RaggedMmap using RaggedMmap.from_generator()
5. Reference in training YAML config


### 7.2 For openWakeWord

```yaml
# Workbench config for openWakeWord
wake_word: "hey assistant"

samples:
  positives: 10000             # openWakeWord default
  negatives_multiplier: 2      # Will generate adversarial negatives

tts:
  backend: "kokoro"
  voices:
    - "af_sarah"
    - "am_adam"
  speed: 1.0

augmentation:
  # ✅ ENABLE for openWakeWord - harness does not augment internally
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]

output:
  path: "./output/openwakeword"
  format: []                   # Do not use workbench exporters
```

**Post-Workbench Steps**:
1. Workbench generates augmented WAV files
2. Load audio with openWakeWord's AudioFeatures.embed_clips()
3. Save as per-class .npy files (positive_features_train.npy, etc.)
4. Configure feature_data_files in custom_model.yml
5. Run openWakeWord auto_train

---

## 8. Known Limitations and Workarounds

### 8.1 Critical Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Workbench exporters incompatible with both harnesses | Cannot use workbench export functions | Export WAV files and use harness-native feature extraction |
| microWakeWord has native augmentation | Double augmentation if using workbench | Disable workbench augmentation for microWakeWord |
| openWakeWord expects specific embeddings | Workbench mel features are wrong type | Use openWakeWord's AudioFeatures.embed_clips() |
| Feature dimension mismatch | Shape errors in training | Always use harness-native feature extraction |
| RaggedMmap vs flat arrays | Format mismatch | Convert using harness-specific tools |

### 8.2 microWakeWord Specific Issues

**Issue**: RaggedMmap folder format required
- Workbench produces flat .mmap files
- microWakeWord expects RaggedMmap folder structure

**Workaround**:
```python
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.spectrograms import SpectrogramGeneration

# Use microWakeWord's pipeline to generate proper format
spectrograms = SpectrogramGeneration(clips=clips, augmenter=augmenter, step_ms=10)
RaggedMmap.from_generator(
    out_dir="training/wakeword_mmap",
    sample_generator=spectrograms.spectrogram_generator(split="train", repeat=2),
    batch_size=100,
)
```

### 8.3 openWakeWord Specific Issues

**Issue**: Embedding features required, not mel-spectrograms
- Workbench produces librosa mel features
- openWakeWord expects Google speech embeddings (96-dim)

**Workaround**:
```python
from openwakeword.utils import AudioFeatures

# Use openWakeWord's feature extraction
F = AudioFeatures(device="cpu")
X_emb = F.embed_clips(audio_array.astype(np.int16), batch_size=256)  # -> (N, 16, 96)
np.save("positive_features_train.npy", X_emb.astype(np.float32))
```

### 8.4 Common Integration Pitfalls

1. **Using workbench mel exports directly**
   - ❌ Don't: Load workbench mel features into either harness
   - ✅ Do: Export WAV files and use harness feature extraction

2. **Applying augmentation twice**
   - ❌ Don't: Use workbench augmentation + microWakeWord augmentation
   - ✅ Do: Disable workbench augmentation for microWakeWord

3. **Wrong negative sample format**
   - ❌ Don't: Mix workbench-generated negatives without format conversion
   - ✅ Do: Convert to harness-specific format (RaggedMmap or embeddings)

4. **Ignoring clip duration requirements**
   - ❌ Don't: Use variable-length clips without padding
   - ✅ Do: Ensure consistent clip sizes (e.g., 2.0s for openWakeWord default)


---

## 9. Actionable Recommendations for Users

### 9.1 Quick Start Decision Tree

**Choose microWakeWord if:**
- You need streaming inference on microcontrollers
- You want a compact model with lower memory footprint
- You have limited training data (hundreds of samples)
- You prefer TensorFlow Lite deployment

**Choose openWakeWord if:**
- You need robust false positive rejection
- You want automatic 3-sequence training with FP/hr optimization
- You have access to large negative corpora (e.g., ACAV100M)
- You prefer ONNX model deployment

### 9.2 Feature Usage Guide

**Always Use Workbench For:**
- ✅ TTS generation (both harnesses)
- ✅ Phonetic confusion generation (both harnesses)
- ✅ Dataset splitting and organization (both harnesses)
- ✅ Hard negative mining (both harnesses)
- ✅ Evaluation metrics and reporting (both harnesses)
- ✅ Augmentation (openWakeWord only)

**Never Use Workbench For:**
- ❌ Feature extraction/export (neither harness)
- ❌ Augmentation (microWakeWord)
- ❌ Direct training data generation (neither harness)

**Use Harness-Native Tools For:**
- 🔧 Feature extraction (both harnesses)
- 🔧 Augmentation (microWakeWord)
- 🔧 Model training (both harnesses)
- 🔧 Model export to deployment format (both harnesses)

### 9.3 Suggested Workflow

**For microWakeWord:**
1. Use workbench to generate positive samples (TTS, variants)
2. Use workbench to generate adversarial negatives
3. Use workbench to split train/val/test
4. Export as WAV files (not workbench mel/mmap)
5. Use microWakeWord's SpectrogramGeneration for features
6. Use microWakeWord's native augmentation
7. Download official negative datasets from Hugging Face
8. Train with microWakeWord model_train_eval
9. Evaluate with workbench eval tools

**For openWakeWord:**
1. Use workbench to generate positive samples (TTS, variants)
2. Use workbench to generate adversarial negatives
3. Use workbench augmentation during generation
4. Use workbench to split train/val/test
5. Export as WAV files (not workbench mel/mmap)
6. Use openWakeWord's AudioFeatures.embed_clips() for features
7. Configure feature_data_files in YAML
8. Train with openWakeWord auto_train
9. Evaluate with workbench eval tools

### 9.4 Integration Priority Matrix

**High Priority (Essential):**
- TTS generation pipeline
- Phonetic confusion generation
- Dataset splitting
- Evaluation metrics

**Medium Priority (Useful):**
- Hard negative mining
- Dataset merging
- Manifest management

**Low Priority (Optional):**
- Workbench augmentation (microWakeWord)
- Workbench exporters (both - do not use)
- Built-in mel features (both - do not use)

---

## 10. Summary Matrix

### 10.1 Complete Feature Compatibility

| Workbench Feature | microWakeWord | openWakeWord | Recommendation |
|-------------------|---------------|--------------|----------------|
| **TTS Generation** | | | |
| Kokoro Backend | ✅ Full | ✅ Full | Use for both |
| Piper Backend | ✅ Full | ✅ Full | Use for both |
| Voice Variation | ✅ Full | ✅ Full | Use for both |
| Speed Control | ✅ Full | ✅ Full | Use for both |
| TTS Caching | ✅ Full | ✅ Full | Use for both |
| **Positive Generation** | | | |
| Base Phrase | ✅ Full | ✅ Full | Use for both |
| Phrase Variants | ✅ Full | ✅ Full | Use for both |
| Export to Harness | 🔄 Adapter | 🔄 Adapter | Export WAV, not features |
| **Negative Generation** | | | |
| Phonetic Confusion | ✅ Full | ✅ Full | Use for both |
| Synthetic Negatives | ✅ Full | ✅ Full | Use for both |
| Export to Harness | 🔄 Adapter | 🔄 Adapter | Convert to harness format |
| **Augmentation** | | | |
| AddNoise | 🔧 Native | ✅ Full | Only for openWakeWord |
| AddColoredNoise | 🔧 Native | ✅ Full | Only for openWakeWord |
| AddReverb | 🔧 Native | ✅ Full | Only for openWakeWord |
| AdjustGain | 🔧 Native | ✅ Full | Only for openWakeWord |
| FixedSizeClip | 🔧 Native | ⚠️ Partial | Only for openWakeWord |
| **Dataset Management** | | | |
| JSONL Manifests | ⚠️ Partial | ⚠️ Partial | Convert to harness format |
| Train/Val/Test Split | ✅ Full | ✅ Full | Use for both |
| Speaker Leakage Prevention | ✅ Full | ✅ Full | Use for both |
| Dataset Merger | ✅ Full | ✅ Full | Use for both |
| **Mining** | | | |
| Long Audio Processing | ✅ Full | ✅ Full | Use for both |
| False Positive Extraction | ✅ Full | ✅ Full | Use for both |
| Cooldown Deduplication | ✅ Full | ✅ Full | Use for both |
| Merge to Dataset | 🔄 Adapter | 🔄 Adapter | Convert to harness format |
| **Export** | | | |
| export_to_mmap | ❌ No | ❌ No | Do not use |
| export_with_features | ❌ No | ❌ No | Do not use |
| export_to_numpy | ❌ No | ❌ No | Do not use |
| **Evaluation** | | | |
| FAR Calculator | ✅ Full | ✅ Full | Use for both |
| FRR Calculator | ✅ Full | ✅ Full | Use for both |
| ROC Generation | ✅ Full | ✅ Full | Use for both |
| Threshold Optimization | ✅ Full | ✅ Full | Use for both |
| Evaluation Reports | ✅ Full | ✅ Full | Use for both |


---

## 11. Appendix: Technical Details

### 11.1 Feature Extraction Comparison

| Aspect | microWakeWord | openWakeWord | Workbench |
|--------|---------------|--------------|-----------|
| Frontend | microfrontend | Google Speech Embed | librosa mel |
| Sample Rate | 16 kHz | 16 kHz | 16 kHz |
| Window Size | 30 ms | 76 frames | 2048 samples |
| Hop Length | 10 ms | 8 frames | 480 samples |
| Feature Dimensions | 40 | 96 | 40 or 96 |
| Output Shape | (T, 40) | (N, 16, 96) | (T, n_mels) |
| Format | RaggedMmap | .npy per class | flat + indices |

### 11.2 Training Data Requirements

| Requirement | microWakeWord | openWakeWord |
|-------------|---------------|--------------|
| Positive samples | 100-1000 | 10000 (default) |
| Negative samples | Pre-generated + custom | Large corpora + adversarial |
| Clip duration | Variable (typically 1-2s) | 2.0s (default) |
| Augmentation | Built-in | External (use workbench) |
| Feature storage | RaggedMmap folders | Per-class .npy files |

### 11.3 Configuration Files

**microWakeWord training_parameters.yaml**:
```yaml
train_dir: "./training_data"
features:
  - features_dir: "./training_data"
    truth: True
    sampling_weight: 2.0
    penalty_weight: 1.0
    truncation_strategy: "truncate_start"
    type: "mmap"
clip_duration_ms: 1000
batch_size: 64
window_step_ms: 10
training_steps: [20000]
learning_rates: [0.001]
```

**openWakeWord custom_model.yml**:
```yaml
model_name: "my_model"
target_phrase: ["hey assistant"]
n_samples: 10000
feature_data_files:
  ACAV100M_sample: "./features/ACAV100M.npy"
  positive: "./features/positive_train.npy"
  adversarial_negative: "./features/adversarial_train.npy"
batch_n_per_class:
  ACAV100M_sample: 1024
  adversarial_negative: 50
  positive: 50
steps: 50000
target_false_positives_per_hour: 0.2
```

---

## 12. Conclusion

### 12.1 Key Takeaways

1. **Workbench is NOT a one-stop solution** - It excels at data generation and evaluation but cannot produce training-ready exports for either harness.

2. **Use harness-native feature extraction** - Both microWakeWord and openWakeWord have specific feature requirements that workbench cannot satisfy.

3. **Augmentation strategy differs by harness**:
   - microWakeWord: Use native augmentation, disable workbench augmentation
   - openWakeWord: Use workbench augmentation before feature extraction

4. **Hard negative mining is universal** - The mining tools work with both harnesses, but mined clips must be converted to the appropriate format.

5. **Evaluation is harness-agnostic** - Workbench's FAR/FRR/ROC tools work well with predictions from either harness.

### 12.2 Recommended Next Steps

**For Workbench Development**:
- Add harness-specific export adapters that convert workbench manifests to:
  - microWakeWord: RaggedMmap generation via SpectrogramGeneration
  - openWakeWord: Per-class .npy files via AudioFeatures.embed_clips()

**For Users**:
- Use workbench for data generation, splitting, and evaluation
- Use harness-native tools for feature extraction and training
- Do not use workbench augmentation with microWakeWord
- Do not use workbench exporters for training data

### 12.3 Final Verdict

| Harness | Workbench Compatibility | Recommendation |
|---------|------------------------|----------------|
| microWakeWord | ⚠️ Partial | Use for data generation only; use native augmentation |
| openWakeWord | ⚠️ Partial | Use for data generation + augmentation; use native feature extraction |

**Workbench is best used as a data preparation and evaluation toolkit**, not as a direct training data producer for either harness.

---

*Evidence document generated from Task 1 (microWakeWord analysis) and Task 2 (openWakeWord analysis) findings.*

*Date: 2026-04-06*
*Version: 1.0*
