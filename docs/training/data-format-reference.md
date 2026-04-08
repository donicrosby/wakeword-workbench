# Data Format Conversion Reference

**WakeWord Workbench → Training Harness Integration Guide**

This document maps WakeWord Workbench outputs to both microWakeWord and openWakeWord training harnesses, providing exact format requirements, conversion steps, and code snippets.

## Setup checkpoint

Before following conversion steps, initialize and verify the repo:

```bash
uv sync --group dev
source .venv/bin/activate
uv run pre-commit install
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If validation reports a missing backend, install `uv sync --extra kokoro` or `uv sync --extra piper`.

---

## Quick Reference

| Aspect | microWakeWord | openWakeWord |
|--------|---------------|--------------|
| **Audio Format** | 16 kHz, mono, float32 or int16 | 16 kHz, mono, 16-bit PCM (int16) |
| **Feature Type** | Spectrogram (40 mel bins) | Google speech embedding (96-d) |
| **Feature Shape** | `(time_frames, 40)` | `(N, 16, 96)` for 2s clips |
| **Storage Format** | RaggedMmap folder per split | Per-class `.npy` files |
| **Sample Rate** | 16000 Hz | 16000 Hz |
| **Window** | 30 ms | 76 mel frames (~80 ms stride) |
| **Step** | 10 ms | 8 mel frames |

---

## 1. Audio Format Requirements

### 1.1 Common Requirements

Both harnesses expect:

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16000 Hz |
| Channels | Mono |
| Bit Depth | 16-bit PCM (int16) or float32 |
| Duration | Variable (typically 1-3 seconds) |

### 1.2 microWakeWord Audio Pipeline

```
Raw Audio (16 kHz)
    ↓
microfrontend feature extraction
    ↓
Spectrogram: (time_frames, 40)
    ↓
RaggedMmap storage
```

**Parameters:**
- Sample rate: 16000 Hz
- Window size: 30 ms (480 samples)
- Window step: 10 ms (160 samples)
- Mel bands: 40
- Frequency range: 125 Hz - 7500 Hz
- PCAN noise reduction: Enabled (default)

### 1.3 openWakeWord Audio Pipeline

```
Raw Audio (16 kHz, int16)
    ↓
Melspectrogram (32 mel bins, stride 160)
    ↓
Sliding window (76 frames, stride 8)
    ↓
Google speech embedding model
    ↓
Embedding: (N, embedding_frames, 96)
```

**Parameters:**
- Sample rate: 16000 Hz
- Mel bands: 32
- Mel stride: 160 samples (~10 ms)
- Embedding window: 76 mel frames
- Embedding stride: 8 mel frames
- Embedding dimension: 96

**For 2-second clips (32000 samples):**
- Mel frames: `ceil(32000/160 - 3) = 197`
- Embedding frames: `(197 - 76) // 8 + 1 = 16`
- Final shape: `(N, 16, 96)`

---

## 2. Data Format Mappings

### 2.1 microWakeWord Format

**Expected Directory Structure:**
```
training_features/
├── training/
│   ├── wakeword_mmap/
│   │   ├── data.npy
│   │   └── metadata.json
│   └── negative_mmap/
│       ├── data.npy
│       └── metadata.json
├── validation/
│   └── wakeword_mmap/
│       └── ...
└── testing/
    └── wakeword_mmap/
        └── ...
```

**Per-sample format:**
- Shape: `(time_frames, 40)`
- Dtype: `uint16` (rescaled by `* 0.0390625`) or `float32`
- Storage: RaggedMmap (one array per clip)

**Configuration entry:**
```yaml
features:
  - features_dir: "training_features"
    truth: true
    sampling_weight: 2.0
    penalty_weight: 1.0
    truncation_strategy: "truncate_start"
    type: "mmap"
```

### 2.2 openWakeWord Format

**Expected File Structure:**
```
features/
├── positive_features_train.npy      # Shape: (N, 16, 96)
├── positive_features_val.npy
├── adversarial_negative_features_train.npy
├── ACAV100M_sample.npy              # Large negative corpus
└── validation_set_features.npy      # For FP/hr tuning
```

**Per-sample format:**
- Shape: `(N, T, 96)` where T depends on clip duration
- Dtype: `float32`
- Storage: Standard `.npy` files grouped by class

**Configuration entry:**
```yaml
feature_data_files:
  positive: ./features/positive_features_train.npy
  adversarial_negative: ./features/adversarial_negative_features_train.npy
  ACAV100M_sample: ./features/ACAV100M_sample.npy

batch_n_per_class:
  positive: 50
  adversarial_negative: 50
  ACAV100M_sample: 1024
```

---

## 3. Workbench Export Outputs

### 3.1 microWakeWord Exporter

**Function:** `export_to_mmap()`

**Output files:**
```
{split}_data.mmap     # Concatenated raw audio (float32)
{split}_indices.npy   # Start/end indices (N x 2, int64)
{split}_labels.npy    # Labels (N, int32)
```

**Function:** `export_with_features()`

**Output files:**
```
{split}_data.npy      # Concatenated mel features (n_frames, 40)
{split}_indices.npy   # Start/end indices (N x 2, int64)
{split}_labels.npy    # Labels (N, int32)
{split}_durations.npy # Frames per clip (N, int64)
{split}_shape.npy     # Shape metadata (2, int64)
```

**Gap:** Neither output is directly consumable by microWakeWord `FeatureHandler`.

**Reasons:**
1. microWakeWord expects RaggedMmap folders (`*_mmap/`), not flat `.mmap` files
2. Workbench mel features use librosa settings (`n_fft=2048`, `hop_length=480`) that differ from microfrontend
3. Labels are stored separately, but microWakeWord infers class from directory structure

### 3.2 openWakeWord Exporter

**Function:** `export_to_numpy()`

**Output files:**
```
X_{split}.npy    # Features (N, ...) - raw audio or mel
y_{split}.npy    # Labels (N, int64)
```

**Format options:**
- `format="raw"`: Raw audio samples, shape `(N, samples)`
- `format="mel"`: Mel spectrogram, shape `(N, time, 96)`

**Gap:** Not directly consumable by openWakeWord trainer.

**Reasons:**
1. openWakeWord expects **Google speech embeddings** (96-d), not raw mel
2. Workbench mel uses 96 bins, but openWakeWord expects post-embedding features
3. openWakeWord uses per-class files, not single `X/y` pairs

---

## 4. Conversion Code Snippets

### 4.1 Workbench → microWakeWord

**Step 1: Generate microfrontend-compatible features**

```python
from pathlib import Path
import numpy as np
from mmap_ninja.ragged import RaggedMmap

# Assuming you have audio clips from workbench manifest
def generate_microwakeword_features(
    audio_clips: list[np.ndarray],  # List of 16 kHz audio arrays
    output_dir: Path,
    split: str = "train",
    step_ms: int = 10,
    slide_frames: int = 10,
) -> None:
    """Generate microWakeWord-compatible RaggedMmap features.

    Args:
        audio_clips: List of audio arrays (float32 or int16, 16 kHz)
        output_dir: Output directory for RaggedMmap
        split: Dataset split name
        step_ms: Feature step in milliseconds (default: 10)
        slide_frames: Slide frames for augmentation (default: 10 for train)
    """
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from microwakeword.audio.audio_utils import generate_features_for_clip

    output_path = output_dir / split / "wakeword_mmap"
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate spectrograms for each clip
    spectrograms = []
    for audio in audio_clips:
        # generate_features_for_clip uses microfrontend settings:
        # - sample_rate: 16000
        # - window_size_ms: 30
        # - window_step_ms: step_ms
        # - num_channels: 40
        spec = generate_features_for_clip(
            audio.astype(np.float32),
            sample_rate=16000,
            window_size_ms=30,
            window_step_ms=step_ms,
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

**Step 2: Create training configuration**

```python
import yaml

def create_microwakeword_config(
    features_dir: Path,
    output_path: Path,
) -> None:
    """Create microWakeWord training configuration."""

    config = {
        "train_dir": str(features_dir),
        "features": [
            {
                "features_dir": str(features_dir),
                "truth": True,
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            }
        ],
        "clip_duration_ms": 1000,  # Adjust based on your clips
        "batch_size": 32,
        "eval_step_interval": 1000,
        "window_step_ms": 10,
        "training_steps": [20000],
        "learning_rates": [0.001],
        "positive_class_weight": [1.0],
        "negative_class_weight": [1.0],
        "target_minimization": 0.05,
        "maximization_metric": "accuracy",
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f)
```

### 4.2 Workbench → openWakeWord

**Step 1: Convert to Google speech embeddings**

```python
from pathlib import Path
import numpy as np
from openwakeword.utils import AudioFeatures

def convert_to_openwakeword_embeddings(
    audio_clips: list[np.ndarray],  # List of 16 kHz int16 audio
    output_dir: Path,
    class_name: str = "positive",
    split: str = "train",
    batch_size: int = 256,
) -> Path:
    """Convert audio clips to openWakeWord embedding format.

    Args:
        audio_clips: List of audio arrays (int16, 16 kHz)
        output_dir: Output directory
        class_name: Class name for file naming
        split: Dataset split
        batch_size: Batch size for embedding extraction

    Returns:
        Path to saved embedding file
    """
    # Initialize feature extractor
    features = AudioFeatures(device="cpu")

    # Stack audio clips
    # Ensure clips are int16 PCM
    audio_batch = np.stack([
        clip.astype(np.int16) if clip.dtype != np.int16 else clip
        for clip in audio_clips
    ])

    # Extract embeddings
    # Output shape: (N, embedding_frames, 96)
    embeddings = features.embed_clips(
        audio_batch,
        batch_size=batch_size,
        ncpu=8,
    )

    # Save as float32
    output_path = output_dir / f"{class_name}_features_{split}.npy"
    np.save(output_path, embeddings.astype(np.float32))

    return output_path
```

**Step 2: Create openWakeWord configuration**

```python
import yaml

def create_openwakeword_config(
    features_dir: Path,
    output_path: Path,
    positive_train: str = "positive_features_train.npy",
    negative_train: str = "adversarial_negative_features_train.npy",
    validation_corpus: str | None = None,
) -> None:
    """Create openWakeWord training configuration."""

    feature_files = {
        "positive": str(features_dir / positive_train),
        "adversarial_negative": str(features_dir / negative_train),
    }

    if validation_corpus:
        feature_files["validation_corpus"] = str(features_dir / validation_corpus)

    config = {
        "model_name": "custom_wakeword",
        "target_phrase": ["hey vera"],
        "n_samples": 10000,
        "n_samples_val": 2000,
        "output_dir": "./trained_model",
        "feature_data_files": feature_files,
        "batch_n_per_class": {
            "positive": 50,
            "adversarial_negative": 50,
        },
        "model_type": "dnn",
        "layer_size": 32,
        "steps": 50000,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f)
```

### 4.3 Split Workbench X/y by Class

```python
import numpy as np
from pathlib import Path

def split_by_class(
    X_path: Path,  # X_train.npy from workbench
    y_path: Path,  # y_train.npy from workbench
    output_dir: Path,
    split: str = "train",
) -> dict[str, Path]:
    """Split workbench export into class-specific files.

    Args:
        X_path: Path to features array
        y_path: Path to labels array
        output_dir: Output directory
        split: Split name for file naming

    Returns:
        Dict mapping class name to output path
    """
    X = np.load(X_path)
    y = np.load(y_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Split by class
    positive_mask = y == 1
    negative_mask = y == 0

    paths = {}

    if positive_mask.any():
        pos_path = output_dir / f"positive_features_{split}.npy"
        np.save(pos_path, X[positive_mask])
        paths["positive"] = pos_path

    if negative_mask.any():
        neg_path = output_dir / f"adversarial_negative_features_{split}.npy"
        np.save(neg_path, X[negative_mask])
        paths["adversarial_negative"] = neg_path

    return paths
```

---

## 5. Feature Extraction Parameters Comparison

### 5.1 Spectrogram Parameters

| Parameter | microWakeWord | Workbench (mel) | openWakeWord (mel stage) |
|-----------|---------------|-----------------|--------------------------|
| Sample Rate | 16000 Hz | 16000 Hz | 16000 Hz |
| Window Size | 30 ms (480 samples) | ~128 ms (2048 samples) | 512 samples |
| Hop Length | 10 ms (160 samples) | 30 ms (480 samples) | 10 ms (160 samples) |
| Mel Bands | 40 | 40 | 32 |
| FFT Size | 512 | 2048 | 512 |
| Freq Range | 125-7500 Hz | 0-8000 Hz | 0-8000 Hz |
| PCAN | Yes | No | No |

### 5.2 Embedding Parameters (openWakeWord only)

| Parameter | Value |
|-----------|-------|
| Mel Window | 76 frames |
| Mel Stride | 8 frames |
| Embedding Dim | 96 |
| Embedding Model | Google speech embedding |
| Input to Embedding | Mel spectrogram (76, 32, 1) |
| Output | 96-d vector per window |

### 5.3 Temporal Resolution

| Harness | Time per Feature Frame |
|---------|----------------------|
| microWakeWord | 10 ms |
| openWakeWord (mel) | 10 ms |
| openWakeWord (embedding) | ~80 ms (8 mel frames × 10 ms) |

---

## 6. Integration Workflows

### 6.1 microWakeWord Integration

```
Workbench Manifest (.jsonl)
    ↓
Load audio clips (16 kHz)
    ↓
Generate microfrontend features (40 mel, 30ms window, 10ms step)
    ↓
Write RaggedMmap to {split}/{class}_mmap/
    ↓
Create training YAML with features_dir pointing to parent
    ↓
Run: python -m microwakeword.model_train_eval --training_config config.yaml
```

### 6.2 openWakeWord Integration

```
Workbench Manifest (.jsonl)
    ↓
Load audio clips (16 kHz, int16)
    ↓
Run AudioFeatures.embed_clips() → (N, T, 96)
    ↓
Split by class → positive_features_{split}.npy, negative_features_{split}.npy
    ↓
Create custom_model.yml with feature_data_files
    ↓
Run: openwakeword.train.Model.auto_train()
```

---

## 7. Common Pitfalls

### 7.1 microWakeWord

| Issue | Symptom | Solution |
|-------|---------|----------|
| Wrong directory structure | `FeatureHandler` can't find data | Use `{split}/{class}_mmap/` folders |
| Wrong feature extraction | Poor model performance | Use microfrontend, not librosa mel |
| Missing RaggedMmap | `KeyError` or `FileNotFoundError` | Use `mmap_ninja.ragged.RaggedMmap` |
| Wrong sample rate | Distorted audio | Resample to 16000 Hz before feature extraction |

### 7.2 openWakeWord

| Issue | Symptom | Solution |
|-------|---------|----------|
| Wrong embedding shape | Shape mismatch error | Ensure 16 kHz int16 input, use `AudioFeatures.embed_clips()` |
| Missing class files | Training fails to start | Create per-class `.npy` files |
| Using raw mel | Poor performance | Must use Google speech embeddings, not raw mel |
| Variable clip lengths | Shape inconsistency | Pad/crop to fixed duration (e.g., 2s = 32000 samples) |

---

## 8. Validation Checklist

### 8.1 Before Conversion

- [ ] All audio files are 16 kHz mono
- [ ] Audio is int16 PCM or float32 in [-1, 1]
- [ ] Manifest contains valid file paths
- [ ] Labels are 0 (negative) or 1 (positive)

### 8.2 After Conversion (microWakeWord)

- [ ] Directory structure: `{split}/{class}_mmap/`
- [ ] Each sample shape: `(time_frames, 40)`
- [ ] Dtype: `uint16` or `float32`
- [ ] Config `features_dir` points to parent of split dirs

### 8.3 After Conversion (openWakeWord)

- [ ] Files named: `{class}_features_{split}.npy`
- [ ] Shape: `(N, T, 96)` where T matches model input
- [ ] Dtype: `float32`
- [ ] Config `feature_data_files` maps class names to paths

---

## 9. Quick Reference Tables

### 9.1 File Naming Conventions

| Harness | Pattern | Example |
|---------|---------|---------|
| Workbench (mmap) | `{split}_data.mmap` | `train_data.mmap` |
| Workbench (features) | `{split}_data.npy` | `train_data.npy` |
| microWakeWord | `{split}/{class}_mmap/` | `training/wakeword_mmap/` |
| openWakeWord | `{class}_features_{split}.npy` | `positive_features_train.npy` |

### 9.2 Shape Conventions

| Format | Shape | Notes |
|--------|-------|-------|
| Workbench raw audio | `(N, samples)` | Variable length, padded |
| Workbench mel | `(N, time, 40)` | librosa mel |
| microWakeWord spectrogram | `(time, 40)` | Per clip, ragged |
| openWakeWord embedding | `(N, T, 96)` | T = 16 for 2s clips |

### 9.3 Key Functions

| Task | Function | Module |
|------|----------|--------|
| Load workbench manifest | `Manifest.load()` | `wakeword_workbench.dataset.metadata` |
| Export to mmap | `export_to_mmap()` | `wakeword_workbench.export.microwakeword` |
| Export to numpy | `export_to_numpy()` | `wakeword_workbench.export.openwakeword` |
| Generate microfrontend | `generate_features_for_clip()` | `microwakeword.audio.audio_utils` |
| Extract embeddings | `AudioFeatures.embed_clips()` | `openwakeword.utils` |
| Create RaggedMmap | `RaggedMmap.from_generator()` | `mmap_ninja.ragged` |

---

## 10. Additional Resources

- **microWakeWord Repository:** https://github.com/OHF-Voice/micro-wake-word
- **openWakeWord Repository:** https://github.com/dscripka/openWakeWord
- **Workbench Exporters:** `src/wakeword_workbench/export/`
- **Task 1 Evidence:** `.sisyphus/evidence/task-1-microwakeword-analysis.md`
- **Task 2 Evidence:** `.sisyphus/evidence/task-2-openwakeword-analysis.md`

---

## Appendix A: Complete Conversion Example

### A.1 Full Pipeline: Workbench → microWakeWord

```python
from pathlib import Path
import numpy as np
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.audio_utils import generate_features_for_clip
from wakeword_workbench.dataset.metadata import Manifest
from wakeword_workbench.augment.audio_loader import load_audio

def convert_workbench_to_microwakeword(
    manifest_path: Path,
    audio_dir: Path,
    output_dir: Path,
    split: str = "train",
) -> None:
    """Complete conversion from workbench manifest to microWakeWord format."""

    # Load manifest
    manifest = Manifest.load(manifest_path)

    # Separate by class
    positive_clips = []
    negative_clips = []

    for entry in manifest:
        audio, _ = load_audio(audio_dir / entry.path, target_sr=16000)
        audio = audio.astype(np.float32)

        if entry.label == 1:
            positive_clips.append(audio)
        else:
            negative_clips.append(audio)

    # Generate features for each class
    for clips, class_name in [(positive_clips, "wakeword"), (negative_clips, "negative")]:
        if not clips:
            continue

        spectrograms = []
        for audio in clips:
            spec = generate_features_for_clip(
                audio,
                sample_rate=16000,
                window_size_ms=30,
                window_step_ms=10,
                num_channels=40,
            )
            spectrograms.append(spec)

        # Write RaggedMmap
        class_dir = output_dir / split / f"{class_name}_mmap"
        class_dir.mkdir(parents=True, exist_ok=True)

        RaggedMmap.from_generator(
            out_dir=str(class_dir),
            sample_generator=iter(spectrograms),
            batch_size=100,
            verbose=True,
        )
```

### A.2 Full Pipeline: Workbench → openWakeWord

```python
from pathlib import Path
import numpy as np
from openwakeword.utils import AudioFeatures
from wakeword_workbench.dataset.metadata import Manifest
from wakeword_workbench.augment.audio_loader import load_audio

def convert_workbench_to_openwakeword(
    manifest_path: Path,
    audio_dir: Path,
    output_dir: Path,
    split: str = "train",
    target_samples: int = 32000,  # 2 seconds
) -> None:
    """Complete conversion from workbench manifest to openWakeWord format."""

    # Load manifest
    manifest = Manifest.load(manifest_path)

    # Separate by class
    positive_clips = []
    negative_clips = []

    for entry in manifest:
        audio, _ = load_audio(audio_dir / entry.path, target_sr=16000)

        # Pad/crop to target length
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]

        # Convert to int16
        audio = (audio * 32767).astype(np.int16)

        if entry.label == 1:
            positive_clips.append(audio)
        else:
            negative_clips.append(audio)

    # Initialize feature extractor
    features = AudioFeatures(device="cpu")

    # Extract embeddings for each class
    for clips, class_name in [(positive_clips, "positive"), (negative_clips, "adversarial_negative")]:
        if not clips:
            continue

        # Stack and extract
        audio_batch = np.stack(clips)
        embeddings = features.embed_clips(audio_batch, batch_size=256, ncpu=8)

        # Save
        output_path = output_dir / f"{class_name}_features_{split}.npy"
        np.save(output_path, embeddings.astype(np.float32))
```

---

*Document generated from Task 1 and Task 2 evidence files.*
*Last updated: 2026-04-06*
