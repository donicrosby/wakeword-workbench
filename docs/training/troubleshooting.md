# Troubleshooting Guide

**WakeWord Workbench Integration Issues and Solutions**

This guide covers common problems when integrating WakeWord Workbench with microWakeWord and openWakeWord training harnesses, with debugging steps and solutions.

## Setup checkpoint

Before troubleshooting deeper issues, verify the baseline environment:

```bash
uv sync --group dev
source .venv/bin/activate
uv run pre-commit install
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If backend initialization fails, install `uv sync --extra kokoro` or `uv sync --extra piper` and retry.

---

## Table of Contents

1. [Workbench Issues](#1-workbench-issues)
2. [microWakeWord Issues](#2-microwakeword-issues)
3. [openWakeWord Issues](#3-openwakeword-issues)
4. [Integration Issues](#4-integration-issues)
5. [Performance Issues](#5-performance-issues)
6. [FAQ](#6-faq)

---

## 1. Workbench Issues

### 1.1 TTS Backend Not Found

**Error:**
```
TTSError: Backend 'kokoro' not available
```

**Cause:** The TTS backend package is not installed.

**Solution:**
```bash
# Install Kokoro backend
uv sync --extra kokoro

# Or install Piper backend
uv sync --extra piper

# Or install all backends
uv sync --all-extras
```

**Prevention:** Always run `uv sync --extra <backend>` after adding a TTS backend to your config.

---

### 1.2 Config Validation Failed

**Error:**
```
ConfigError: Missing required field: wake_word
```

**Cause:** Required configuration fields are missing or invalid.

**Solution:**

1. Check your config against the schema:
```bash
uv run wakeword-workbench validate config.yaml
```

2. Ensure all required fields are present:
```yaml
wake_word: "hey vera"  # Required

samples:
  positives: 100        # Required
  negatives_multiplier: 5  # Required

tts:
  providers:            # Required
    - backend: "kokoro"
      voices:
        - "af_sarah"
      speed: 1.0        # Optional, defaults to 1.0

augmentation:
  noise_snr: [-10, 10]  # Required
  reverb_probability: 0.5  # Required
  gain_range: [-45, 0]    # Required

output:
  path: "./output"      # Required
  format: ["microwakeword"]  # Required
```

**Prevention:** Use the `validate` command before running the pipeline.

---

### 1.3 Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'wakeword_workbench'
```

**Cause:** Virtual environment not activated or package not installed.

**Solution:**
```bash
# Activate virtual environment
source .venv/bin/activate

# If still failing, reinstall
uv sync
```

---

### 1.4 TTS Synthesis Cache Issues

**Error:**
```
FileNotFoundError: Cache directory not accessible
```

**Cause:** Cache directory permissions or disk space issues.

**Solution:**

1. Check cache location:
```bash
ls -la ~/.cache/wakeword_workbench/tts/
```

2. Clear cache if corrupted:
```bash
uv run wakeword-workbench cache-clear
```

3. Set custom cache directory (if needed):
```bash
export WAKEWORD_CACHE_DIR=/path/to/cache
```

---

### 1.5 Dataset Generation Failures

**Error:**
```
RuntimeError: Failed to generate positive samples
```

**Cause:** TTS synthesis failure, audio file corruption, or path issues.

**Debugging Steps:**

1. Check TTS backend status:
```python
from wakeword_workbench.tts.registry import get_backend

backend = get_backend("kokoro")
backend.set_voice("af_sarah")
result = backend.synthesize("test")
print(f"Success: duration={result.duration:.2f}s, sample_rate={result.sample_rate}")
```

2. Verify audio files:
```bash
# Check audio file integrity
ffprobe -v error -show_format -show_streams output/audio/*.wav
```

3. Check disk space:
```bash
df -h output/
```

**Solution:**

- Ensure TTS backend is properly installed
- Verify output directory has write permissions
- Check available disk space (recommend 10GB+ for large datasets)

---

## 2. microWakeWord Issues

### 2.1 RaggedMmap Format Errors

**Error:**
```
FileNotFoundError: training/wakeword_mmap/ not found
```

**Cause:** microWakeWord expects RaggedMmap folder structure, not flat `.mmap` files.

**Expected Structure:**
```
training_features/
├── training/
│   ├── wakeword_mmap/
│   │   ├── data.npy
│   │   └── metadata.json
│   └── negative_mmap/
│       └── ...
├── validation/
│   └── wakeword_mmap/
└── testing/
    └── wakeword_mmap/
```

**Solution:**

Workbench exports flat arrays. Convert to RaggedMmap:

```python
from pathlib import Path
import numpy as np
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.audio_utils import generate_features_for_clip

def convert_to_ragged_mmap(
    audio_clips: list[np.ndarray],
    output_dir: Path,
    split: str = "train",
    class_name: str = "wakeword",
):
    """Convert audio clips to microWakeWord RaggedMmap format."""

    output_path = output_dir / split / f"{class_name}_mmap"
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate microfrontend features for each clip
    spectrograms = []
    for audio in audio_clips:
        spec = generate_features_for_clip(
            audio.astype(np.float32),
            sample_rate=16000,
            window_size_ms=30,
            window_step_ms=10,
            num_channels=40,
        )
        spectrograms.append(spec)  # Shape: (time_frames, 40)

    # Write as RaggedMmap
    RaggedMmap.from_generator(
        out_dir=str(output_path),
        sample_generator=iter(spectrograms),
        batch_size=100,
        verbose=True,
    )
```

---

### 2.2 Feature Shape Mismatches

**Error:**
```
ValueError: Expected shape (T, 40), got (T, 96)
```

**Cause:** Using wrong feature extraction method. microWakeWord expects 40 mel bins from microfrontend, not librosa mel.

**Solution:**

Do NOT use workbench mel features. Use microWakeWord's feature extraction:

```python
# ❌ WRONG: Workbench mel features
from wakeword_workbench.export.microwakeword import export_with_features
# This produces (T, 40) but with wrong parameters

# ✅ CORRECT: microWakeWord microfrontend
from microwakeword.audio.audio_utils import generate_features_for_clip

spec = generate_features_for_clip(
    audio.astype(np.float32),
    sample_rate=16000,
    window_size_ms=30,    # NOT librosa's 2048 samples
    window_step_ms=10,     # NOT librosa's 480 samples
    num_channels=40,       # Fixed at 40
)
```

**Key Differences:**

| Parameter | microWakeWord | Workbench (librosa) |
|-----------|---------------|---------------------|
| Window Size | 30 ms (480 samples) | ~128 ms (2048 samples) |
| Hop Length | 10 ms (160 samples) | 30 ms (480 samples) |
| Mel Bands | 40 | 40 or 96 |
| PCAN | Yes | No |

---

### 2.3 Training Convergence Problems

**Symptom:** Model loss doesn't decrease or validation accuracy stays flat.

**Causes:**

1. **Class imbalance:** Too few positive samples
2. **Wrong learning rate:** Too high or too low
3. **Insufficient data:** Not enough training samples
4. **Feature mismatch:** Wrong feature extraction

**Debugging Steps:**

1. Check class balance:
```python
import numpy as np
from pathlib import Path

# Count samples in each split
for split in ["training", "validation", "testing"]:
    pos_count = len(list(Path(f"features/{split}/wakeword_mmap").glob("*")))
    neg_count = len(list(Path(f"features/{split}/negative_mmap").glob("*")))
    print(f"{split}: {pos_count} positive, {neg_count} negative")
```

2. Check feature statistics:
```python
from mmap_ninja.ragged import RaggedMmap

mmap = RaggedMmap("features/training/wakeword_mmap")
sample = mmap[0]
print(f"Shape: {sample.shape}")
print(f"Dtype: {sample.dtype}")
print(f"Range: [{sample.min()}, {sample.max()}]")
```

**Solutions:**

1. **Increase positive samples:**
```yaml
samples:
  positives: 500  # Increase from 100
  negatives_multiplier: 3
```

2. **Adjust learning rate:**
```yaml
training_steps: [20000]
learning_rates: [0.001]  # Try 0.0001 if unstable
```

3. **Use official negative datasets:**
```bash
# Download from Hugging Face
huggingface-cli download kahara/microWakeWord-negatives
```

---

### 2.4 TFLite Export Failures

**Error:**
```
RuntimeError: Failed to convert to TFLite
```

**Cause:** Model architecture not compatible with TFLite or missing SavedModel.

**Solution:**

1. Ensure model was trained with streaming architecture:
```bash
python -m microwakeword.model_train_eval \
  --training_config config.yaml \
  --test_tflite_streaming_quantized 1
```

2. Check for unsupported operations:
```python
import tensorflow as tf

# Load SavedModel
model = tf.saved_model.load("trained_model/streaming")

# Check for TFLite compatibility
converter = tf.lite.TFLiteConverter.from_saved_model("trained_model/streaming")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
```

---

## 3. openWakeWord Issues

### 3.1 Embedding Generation Errors

**Error:**
```
RuntimeError: Failed to load embedding model
```

**Cause:** Missing ONNX models or wrong device configuration.

**Solution:**

1. Download required models:
```python
from openwakeword.utils import AudioFeatures

# This downloads models on first use
features = AudioFeatures(device="cpu")
```

2. Check model files:
```bash
ls -la ~/.cache/openwakeword/
```

3. Specify device explicitly:
```python
# CPU
features = AudioFeatures(device="cpu")

# GPU (if available)
features = AudioFeatures(device="cuda")
```

---

### 3.2 Numpy Format Issues

**Error:**
```
ValueError: Shape mismatch: expected (N, 16, 96), got (N, 20, 96)
```

**Cause:** Variable clip lengths or wrong embedding extraction.

**Solution:**

Ensure consistent clip duration:

```python
import numpy as np
from openwakeword.utils import AudioFeatures

def extract_embeddings_fixed_duration(
    audio_clips: list[np.ndarray],
    target_samples: int = 32000,  # 2 seconds at 16 kHz
):
    """Extract embeddings with fixed duration."""

    features = AudioFeatures(device="cpu")

    # Pad/crop to fixed length
    fixed_clips = []
    for audio in audio_clips:
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]
        fixed_clips.append(audio.astype(np.int16))

    # Stack and extract
    audio_batch = np.stack(fixed_clips)
    embeddings = features.embed_clips(audio_batch, batch_size=256)

    # Shape: (N, 16, 96) for 2-second clips
    return embeddings
```

---

### 3.3 Training Sequence Failures

**Error:**
```
RuntimeError: Training sequence 2 failed: FP/hr too high
```

**Cause:** Model not converging or insufficient negative samples.

**Solution:**

1. Increase negative weight:
```yaml
max_negative_weight: 2000  # Increase from default 1500
```

2. Add more negative samples:
```yaml
feature_data_files:
  ACAV100M_sample: ./features/ACAV100M.npy
  adversarial_negative: ./features/adversarial_train.npy
  positive: ./features/positive_train.npy

batch_n_per_class:
  ACAV100M_sample: 1024  # Large negative corpus
  adversarial_negative: 100  # Increase from 50
  positive: 50
```

3. Lower FP/hr target:
```yaml
target_false_positives_per_hour: 0.5  # Increase from 0.2
```

---

### 3.4 ONNX Export Problems

**Error:**
```
RuntimeError: ONNX export failed: unsupported operation
```

**Cause:** Model architecture not ONNX-compatible.

**Solution:**

1. Use supported model type:
```yaml
model_type: "dnn"  # Use DNN instead of RNN for better compatibility
layer_size: 32
```

2. Check ONNX export:
```python
import onnx

# Load and validate ONNX model
model = onnx.load("trained_model.onnx")
onnx.checker.check_model(model)
```

---

## 4. Integration Issues

### 4.1 Path Resolution Problems

**Error:**
```
FileNotFoundError: Cannot find features at ./features/training
```

**Cause:** Relative paths in config or wrong working directory.

**Solution:**

1. Use absolute paths:
```yaml
features:
  - features_dir: "/absolute/path/to/features"
```

2. Or run from correct directory:
```bash
cd /path/to/project
python -m microwakeword.model_train_eval --training_config config.yaml
```

3. Check path resolution:
```python
from pathlib import Path

config_path = Path("config.yaml")
features_dir = Path(config["features"][0]["features_dir"])

if not features_dir.is_absolute():
    features_dir = config_path.parent / features_dir

print(f"Resolved path: {features_dir.resolve()}")
```

---

### 4.2 Data Format Conversions

**Error:**
```
ValueError: Cannot load workbench export directly
```

**Cause:** Workbench exports are not directly compatible with either harness.

**Solution:**

Workbench exports require conversion. See the [Data Format Reference](data-format-reference.md) for complete conversion code.

**Quick Reference:**

| From | To | Method |
|------|-----|--------|
| Workbench WAV | microWakeWord | Use `microwakeword.audio.audio_utils.generate_features_for_clip()` |
| Workbench WAV | openWakeWord | Use `openwakeword.utils.AudioFeatures.embed_clips()` |
| Workbench mel | Either | ❌ Do NOT use - wrong parameters |

---

### 4.3 Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'mmap_ninja'
```

**Cause:** Harness dependencies not installed.

**Solution:**

```bash
# For microWakeWord
pip install microwakeword
pip install mmap-ninja

# For openWakeWord
pip install openwakeword
```

---

### 4.4 Version Incompatibilities

**Error:**
```
AttributeError: 'RaggedMmap' object has no attribute 'from_generator'
```

**Cause:** Outdated `mmap_ninja` version.

**Solution:**

```bash
# Update to latest version
pip install --upgrade mmap-ninja

# Or specify minimum version
pip install "mmap-ninja>=0.5.0"
```

---

## 5. Performance Issues

### 5.1 Slow Training

**Symptom:** Training takes hours for small datasets.

**Causes:**

1. **CPU-only training:** No GPU acceleration
2. **Large batch sizes:** Memory thrashing
3. **Inefficient data loading:** Not using memmap properly
4. **Too many augmentation rounds:** Excessive data generation

**Solutions:**

1. **Enable GPU:**
```python
# For TensorFlow (microWakeWord)
import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))

# For PyTorch (openWakeWord)
import torch
print("GPU available:", torch.cuda.is_available())
```

2. **Optimize batch size:**
```yaml
# microWakeWord
batch_size: 64  # Increase if GPU memory allows

# openWakeWord
batch_n_per_class:
  ACAV100M_sample: 1024
  adversarial_negative: 50
  positive: 50
```

3. **Use memmap efficiently:**
```python
# Pre-load into memory if dataset is small
# Or use memory-mapped files for large datasets
from mmap_ninja.ragged import RaggedMmap

mmap = RaggedMmap("features/training/wakeword_mmap")
# Access is lazy - only loads what's needed
```

4. **Reduce augmentation:**
```yaml
# openWakeWord
augmentation_rounds: 1  # Reduce from default

# microWakeWord - use built-in augmentation sparingly
```

---

### 5.2 Memory Exhaustion

**Error:**
```
MemoryError: Unable to allocate array
```

**Causes:**

1. **Loading entire dataset into memory**
2. **Large feature files without memmap**
3. **Too many concurrent processes**

**Solutions:**

1. **Use memory-mapped files:**
```python
# ❌ WRONG: Load entire dataset
X = np.load("features.npy")  # Loads all into memory

# ✅ CORRECT: Use memmap
X = np.load("features.npy", mmap_mode="r")  # Memory-mapped
```

2. **Process in batches:**
```python
from openwakeword.utils import AudioFeatures

features = AudioFeatures(device="cpu")

# Process in batches instead of all at once
batch_size = 256
for i in range(0, len(audio_clips), batch_size):
    batch = audio_clips[i:i+batch_size]
    embeddings = features.embed_clips(batch, batch_size=batch_size)
    # Save incrementally
```

3. **Reduce concurrent processes:**
```python
# openWakeWord
embeddings = features.embed_clips(audio_batch, batch_size=256, ncpu=4)  # Reduce ncpu
```

---

### 5.3 GPU/CPU Utilization

**Symptom:** GPU utilization is low (< 50%).

**Causes:**

1. **Data loading bottleneck:** CPU can't feed GPU fast enough
2. **Small batch sizes:** Not enough parallelism
3. **Synchronous data loading:** Not using prefetch

**Solutions:**

1. **Increase batch size:**
```yaml
batch_size: 128  # Increase until GPU memory fills
```

2. **Use data prefetching:**
```python
# TensorFlow (microWakeWord)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# PyTorch (openWakeWord)
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```

3. **Monitor utilization:**
```bash
# GPU utilization
watch -n 1 nvidia-smi

# CPU utilization
htop
```

---

### 5.4 Disk Space Issues

**Error:**
```
OSError: [Errno 28] No space left on device
```

**Causes:**

1. **Large feature files:** Embeddings and spectrograms take space
2. **TTS cache:** Cached audio files accumulate
3. **Augmented audio:** Multiple augmentation rounds

**Solutions:**

1. **Check disk usage:**
```bash
du -sh ~/.cache/wakeword_workbench/
du -sh output/
```

2. **Clear TTS cache:**
```bash
uv run wakeword-workbench cache-clear
```

3. **Use compressed storage:**
```python
import numpy as np

# Save with compression
np.savez_compressed("features.npz", X=X, y=y)

# Load compressed
data = np.load("features.npz")
X = data["X"]
```

4. **Clean up intermediate files:**
```bash
# Remove intermediate audio after feature extraction
find output/ -name "*.wav" -delete
```

---

## 6. FAQ

### Q1: Which harness should I choose?

**Answer:** It depends on your deployment target and data availability.

| Factor | Choose microWakeWord | Choose openWakeWord |
|--------|---------------------|---------------------|
| Deployment | Microcontrollers, embedded | Server, edge devices |
| Model size | Small (< 100KB) | Medium (~1MB) |
| Training data | Limited (100-1000 samples) | Large (10K+ samples) |
| FP/hr target | Moderate | Very low (< 0.5) |
| Export format | TFLite | ONNX |

**Recommendation:**
- **Microcontrollers:** microWakeWord (streaming TFLite)
- **Server/edge:** openWakeWord (ONNX with FP/hr optimization)

---

### Q2: How much data do I need?

**Answer:**

| Harness | Minimum | Recommended | Optimal |
|---------|---------|-------------|---------|
| microWakeWord | 100 positives | 500 positives | 1000+ positives |
| openWakeWord | 1000 positives | 5000 positives | 10000+ positives |

**Negative samples:**
- microWakeWord: Use official negative datasets + custom adversarial negatives (3-5x positives)
- openWakeWord: Use large negative corpora (ACAV100M) + adversarial negatives (2-3x positives)

**Key insight:** Data quality matters more than quantity. Diverse negatives and phonetic confusions are critical.

---

### Q3: Can I use workbench augmentation with both harnesses?

**Answer:** No. Use workbench augmentation only with openWakeWord.

| Harness | Workbench Augmentation | Native Augmentation |
|---------|------------------------|-------------------|
| microWakeWord | ❌ Do NOT use | ✅ Use built-in |
| openWakeWord | ✅ Use workbench | N/A |

**Reason:** microWakeWord has its own comprehensive augmentation pipeline. Using workbench augmentation would result in double augmentation.

**For microWakeWord:**
```yaml
# Disable workbench augmentation
augmentation:
  noise_snr: [0, 0]
  reverb_probability: 0.0
  gain_range: [0, 0]
```

**For openWakeWord:**
```yaml
# Enable workbench augmentation
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
```

---

### Q4: Why can't the harness read my workbench export?

**Answer:** Workbench exports are intermediate formats, not training-ready.

**Workbench produces:**
- Flat `.mmap` files with index sidecars
- librosa mel-spectrograms (wrong parameters)
- Single `X/y` arrays (not class-separated)

**Harnesses expect:**

| Harness | Format | Features |
|---------|--------|----------|
| microWakeWord | RaggedMmap folders | microfrontend (40 mel, 30ms window) |
| openWakeWord | Per-class `.npy` files | Google speech embeddings (96-d) |

**Solution:** Export WAV files from workbench, then use harness-native feature extraction.

```python
# ❌ WRONG: Use workbench exporters
from wakeword_workbench.export.microwakeword import export_to_mmap

# ✅ CORRECT: Export WAV, use harness extraction
# 1. Workbench generates WAV files
# 2. Use microwakeword.audio.audio_utils.generate_features_for_clip()
# 3. Or use openwakeword.utils.AudioFeatures.embed_clips()
```

---

### Q5: How do I convert workbench output to harness format?

**Answer:** See the [Data Format Reference](data-format-reference.md) for complete code examples.

**Quick summary:**

**For microWakeWord:**
```python
from microwakeword.audio.audio_utils import generate_features_for_clip
from mmap_ninja.ragged import RaggedMmap

# 1. Load audio from workbench manifest
# 2. Generate microfrontend features
spec = generate_features_for_clip(audio, sample_rate=16000,
                                   window_size_ms=30, window_step_ms=10,
                                   num_channels=40)
# 3. Save as RaggedMmap
RaggedMmap.from_generator(out_dir="training/wakeword_mmap",
                          sample_generator=iter(spectrograms))
```

**For openWakeWord:**
```python
from openwakeword.utils import AudioFeatures

# 1. Load audio from workbench manifest
# 2. Extract embeddings
features = AudioFeatures(device="cpu")
embeddings = features.embed_clips(audio_batch, batch_size=256)
# 3. Save per-class files
np.save("positive_features_train.npy", embeddings[y==1])
np.save("adversarial_negative_features_train.npy", embeddings[y==0])
```

---

### Q6: Training is very slow — how can I speed it up?

**Answer:** Optimize data loading, batch size, and hardware utilization.

**Checklist:**

1. **Enable GPU:**
```bash
# Verify GPU is available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import torch; print(torch.cuda.is_available())"
```

2. **Increase batch size:**
```yaml
# microWakeWord
batch_size: 64  # Increase until GPU memory fills

# openWakeWord
batch_n_per_class:
  ACAV100M_sample: 1024
  adversarial_negative: 50
  positive: 50
```

3. **Use memory-mapped files:**
```python
# Don't load entire dataset
X = np.load("features.npy", mmap_mode="r")
```

4. **Reduce augmentation:**
```yaml
# openWakeWord
augmentation_rounds: 1  # Reduce from default
```

5. **Monitor utilization:**
```bash
# GPU should be > 80% utilized during training
watch -n 1 nvidia-smi
```

---

### Q7: I'm getting out of memory errors — what should I do?

**Answer:** Reduce memory footprint with batch processing and memmap.

**Solutions:**

1. **Use memory-mapped files:**
```python
# ❌ WRONG: Load everything
X = np.load("features.npy")

# ✅ CORRECT: Memory-map
X = np.load("features.npy", mmap_mode="r")
```

2. **Process in batches:**
```python
batch_size = 256
for i in range(0, len(audio_clips), batch_size):
    batch = audio_clips[i:i+batch_size]
    embeddings = features.embed_clips(batch, batch_size=batch_size)
```

3. **Reduce concurrent workers:**
```python
# openWakeWord
embeddings = features.embed_clips(audio_batch, ncpu=4)  # Reduce from 8
```

4. **Clear GPU memory:**
```python
import torch
torch.cuda.empty_cache()

import tensorflow as tf
tf.keras.backend.clear_session()
```

---

### Q8: How do I know if my model is good enough?

**Answer:** Evaluate with FAR/FRR metrics and test on real-world audio.

**Metrics to check:**

| Metric | Target | Interpretation |
|--------|--------|----------------|
| FAR (False Accept Rate) | < 1 per hour | Wake word triggers on non-wake word |
| FRR (False Reject Rate) | < 5% | Wake word not detected when spoken |
| Accuracy | > 95% | Overall classification accuracy |
| ROC AUC | > 0.95 | Model discrimination ability |

**Evaluation workflow:**

```python
from wakeword_workbench.eval.metrics import calculate_far_frr
from wakeword_workbench.eval.roc import generate_roc_curve

# Calculate FAR/FRR
far, frr, threshold = calculate_far_frr(predictions, labels)

# Generate ROC curve
generate_roc_curve(predictions, labels, output_path="roc.png")

# Find optimal threshold
optimal_threshold = find_eer_threshold(far, frr)
print(f"Optimal threshold: {optimal_threshold}")
print(f"FAR at threshold: {far[optimal_threshold]}")
print(f"FRR at threshold: {frr[optimal_threshold]}")
```

**Real-world testing:**

1. Test on held-out data (not used in training)
2. Test with background noise
3. Test with different speakers
4. Test with phonetic confusions
5. Test in target deployment environment

---

### Q9: Can I fine-tune an existing model?

**Answer:** Yes, but approach differs by harness.

**microWakeWord:**

```bash
# Restore from checkpoint
python -m microwakeword.model_train_eval \
  --training_config config.yaml \
  --restore_checkpoint 1 \
  --use_weights path/to/best_weights.weights.h5
```

**openWakeWord:**

```python
from openwakeword.train import Model

# Load existing model
model = Model(model_type="dnn")
model.load("existing_model.onnx")

# Continue training with new data
model.auto_train(
    feature_data_files=new_features,
    steps=10000,  # Fewer steps for fine-tuning
    learning_rates=[0.0001],  # Lower learning rate
)
```

**Best practices:**
- Use lower learning rate (10x smaller)
- Use fewer training steps
- Ensure new data is similar to original data
- Validate on original test set to prevent catastrophic forgetting

---

### Q10: How do I deploy my trained model?

**Answer:** Export to deployment format and integrate with inference engine.

**microWakeWord deployment:**

```bash
# Train and export TFLite
python -m microwakeword.model_train_eval \
  --training_config config.yaml \
  --test_tflite_streaming_quantized 1

# Output: trained_model/streaming_quantized.tflite
```

**Inference code:**
```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter("streaming_quantized.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
interpreter.set_tensor(input_details[0]['index'], audio_features)
interpreter.invoke()
probability = interpreter.get_tensor(output_details[0]['index'])
```

**openWakeWord deployment:**

```python
from openwakeword import Model

# Load ONNX model
model = Model("custom_model.onnx")

# Run inference
predictions = model.predict(audio_clip)
```

**Deployment checklist:**

- [ ] Model quantized (if needed for size)
- [ ] Latency tested on target hardware
- [ ] Memory footprint acceptable
- [ ] FAR/FRR validated on test set
- [ ] Real-world testing completed
- [ ] Fallback behavior defined

---

### Q11: What's the difference between validation and testing splits?

**Answer:** They serve different purposes in the training pipeline.

| Split | Purpose | When Used |
|-------|---------|-----------|
| Training | Learn model weights | During gradient descent |
| Validation | Tune hyperparameters | After each epoch, for early stopping |
| Testing | Final evaluation | After training complete |

**Best practices:**

1. **No speaker overlap:** Same speaker should not appear in multiple splits
2. **Stratified sampling:** Maintain class balance across splits
3. **Temporal separation:** If data is time-series, split by time
4. **Typical ratio:** 70% train, 15% validation, 15% test

**Workbench split:**
```yaml
# Workbench automatically splits with speaker leakage prevention
samples:
  positives: 1000
  negatives_multiplier: 5
  # Results in ~700 train, ~150 val, ~150 test
```

---

### Q12: How do I handle class imbalance?

**Answer:** Use class weighting and balanced sampling.

**microWakeWord:**
```yaml
# Increase weight for positive class
positive_class_weight: [2.0]  # Weight positive samples 2x
negative_class_weight: [1.0]

# Or increase sampling weight for positives
features:
  - features_dir: "training"
    truth: True
    sampling_weight: 2.0  # Sample positives 2x more often
```

**openWakeWord:**
```yaml
# Adjust batch composition
batch_n_per_class:
  ACAV100M_sample: 1024  # Large negative corpus
  adversarial_negative: 50
  positive: 50  # Balance with negatives

# Increase negative weight during training
max_negative_weight: 1500  # Penalize false positives more
```

**General strategies:**

1. **More negative samples:** Generate more adversarial negatives
2. **Class weighting:** Penalize false positives more
3. **Balanced batches:** Equal positive/negative in each batch
4. **Focal loss:** Focus on hard examples
5. **Data augmentation:** Augment minority class more

---

### Q13: Why are my features the wrong shape?

**Answer:** Feature shape depends on clip duration and extraction method.

**Expected shapes:**

| Harness | Clip Duration | Feature Shape |
|---------|--------------|---------------|
| microWakeWord | Variable | `(time_frames, 40)` |
| openWakeWord | 2.0 seconds | `(N, 16, 96)` |

**Shape calculation:**

**microWakeWord:**
```python
time_frames = ceil(audio_samples / 160)  # 10 ms hop
# For 1 second: time_frames ≈ 100
```

**openWakeWord:**
```python
mel_frames = ceil(audio_samples / 160 - 3)
embedding_frames = (mel_frames - 76) // 8 + 1
# For 2 seconds: embedding_frames = 16
```

**Common issues:**

1. **Variable clip lengths:** Pad/crop to fixed duration
```python
target_samples = 32000  # 2 seconds
if len(audio) < target_samples:
    audio = np.pad(audio, (0, target_samples - len(audio)))
else:
    audio = audio[:target_samples]
```

2. **Wrong feature extraction:** Use harness-native extraction
```python
# ❌ WRONG: Workbench mel
from wakeword_workbench.export.openwakeword import export_to_numpy

# ✅ CORRECT: Harness-native
from openwakeword.utils import AudioFeatures
features = AudioFeatures(device="cpu")
embeddings = features.embed_clips(audio_batch)
```

---

### Q14: How do I add custom negative phrases?

**Answer:** Generate with TTS and convert to harness format.

**Workbench generation:**
```yaml
wake_word: "hey vera"

samples:
  positives: 500
  negatives_multiplier: 5

# Custom negative phrases
custom_negatives:
  - "hey sarah"
  - "hey vera's"
  - "hey very"
  - "hey era"
```

**Convert to harness format:**

**For microWakeWord:**
```python
# Generate negative clips with workbench TTS
# Then convert to RaggedMmap
convert_to_ragged_mmap(negative_clips, output_dir, split="train", class_name="negative")
```

**For openWakeWord:**
```python
# Generate negative clips with workbench TTS
# Then extract embeddings
embeddings = features.embed_clips(negative_clips)
np.save("adversarial_negative_features_train.npy", embeddings)
```

**Add to config:**

**microWakeWord:**
```yaml
features:
  - features_dir: "training"
    truth: False
    sampling_weight: 1.0
    penalty_weight: 1.0
    truncation_strategy: "truncate_start"
    type: "mmap"
```

**openWakeWord:**
```yaml
feature_data_files:
  adversarial_negative: ./features/adversarial_negative_train.npy

batch_n_per_class:
  adversarial_negative: 50
```

---

### Q15: What audio format should I use?

**Answer:** Both harnesses expect 16 kHz mono audio.

**Required format:**

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16000 Hz |
| Channels | Mono |
| Bit Depth | 16-bit PCM (int16) or float32 |
| Duration | Variable (typically 1-3 seconds) |

**Conversion:**
```python
import librosa
import soundfile as sf

# Load and convert
audio, sr = librosa.load("input.wav", sr=16000, mono=True)

# Save as 16-bit PCM
sf.write("output.wav", audio, 16000, subtype="PCM_16")

# Or as float32
sf.write("output.wav", audio, 16000, subtype="FLOAT")
```

**Batch conversion:**
```bash
# Using ffmpeg
for f in *.wav; do
  ffmpeg -i "$f" -ar 16000 -ac 1 -acodec pcm_s16le "converted/$f"
done
```

**Validation:**
```python
import soundfile as sf

info = sf.info("audio.wav")
assert info.samplerate == 16000, f"Wrong sample rate: {info.samplerate}"
assert info.channels == 1, f"Wrong channels: {info.channels}"
```

---

## Prevention Tips

### General Best Practices

1. **Always validate config before running:**
```bash
uv run wakeword-workbench validate config.yaml
```

2. **Use version control for configs:**
```bash
git add config.yaml
git commit -m "Add training config"
```

3. **Test with small dataset first:**
```yaml
samples:
  positives: 10  # Start small
  negatives_multiplier: 2
```

4. **Monitor training progress:**
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check logs
tail -f training.log
```

5. **Keep intermediate outputs:**
```bash
# Don't delete until training succeeds
ls output/
```

### microWakeWord-Specific

1. **Use official negative datasets:**
```bash
huggingface-cli download kahara/microWakeWord-negatives
```

2. **Use native augmentation:**
```yaml
# Disable workbench augmentation
augmentation:
  noise_snr: [0, 0]
  reverb_probability: 0.0
  gain_range: [0, 0]
```

3. **Validate RaggedMmap structure:**
```python
from mmap_ninja.ragged import RaggedMmap
mmap = RaggedMmap("training/wakeword_mmap")
print(f"Samples: {len(mmap)}")
print(f"First shape: {mmap[0].shape}")
```

### openWakeWord-Specific

1. **Use large negative corpora:**
```yaml
feature_data_files:
  ACAV100M_sample: ./features/ACAV100M.npy
```

2. **Use workbench augmentation:**
```yaml
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
```

3. **Validate embedding shape:**
```python
import numpy as np
embeddings = np.load("positive_features_train.npy")
print(f"Shape: {embeddings.shape}")  # Should be (N, 16, 96) for 2s clips
```

---

## Additional Resources

- **Data Format Reference:** [data-format-reference.md](data-format-reference.md)
- **microWakeWord Repository:** https://github.com/OHF-Voice/micro-wake-word
- **openWakeWord Repository:** https://github.com/dscripka/openWakeWord
- **Workbench Documentation:** [../README.md](../README.md)

---

*Last updated: 2026-04-06*
*Version: 1.0*
