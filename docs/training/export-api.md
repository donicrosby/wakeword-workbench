# Export API Reference

Complete API reference for WakeWord Workbench export modules. These modules convert dataset manifests into intermediate artifacts used by microWakeWord and openWakeWord workflows.

## Setup checkpoint

Before running export examples, bootstrap and verify the repo environment:

```bash
uv sync --group dev
source .venv/bin/activate
uv run pre-commit install
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If backend initialization fails, install `uv sync --extra kokoro` or `uv sync --extra piper`.

## Overview

The export modules provide two distinct export formats:

| Format | Module | Output | Use Case |
|--------|--------|--------|----------|
| **MicroWakeWord Mmap** | `microwakeword.py` | Memory-mapped raw audio | Large datasets, streaming training |
| **MicroWakeWord Features** | `microwakeword.py` | Pre-computed mel-spectrograms | Faster training startup |
| **OpenWakeWord** | `openwakeword.py` | NumPy arrays (X, y) | Standard ML workflows |

All export functions accept a `Manifest` object and produce intermediate files that may require additional harness-specific conversion before model training.

---

## MicroWakeWord Export Module

Import the module:

```python
from wakeword_workbench.export.microwakeword import (
    export_to_mmap,
    export_with_features,
    compute_mel_spectrogram,
    load_mmap,
    validate_mmap,
    MicroWakeWordExportError,
)
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
```

### compute_mel_spectrogram()

Compute mel-spectrogram features from audio samples.

**Signature:**

```python
def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 40,
    hop_length: int = 480,
    n_fft: int = 2048,
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | `np.ndarray` | Required | Audio samples as float32 array in [-1, 1] range |
| `sample_rate` | `int` | `16000` | Sample rate in Hz |
| `n_mels` | `int` | `40` | Number of mel filterbanks |
| `hop_length` | `int` | `480` | Number of samples between frames (30ms at 16kHz) |
| `n_fft` | `int` | `2048` | FFT window size |

**Returns:**

`np.ndarray` — Mel-spectrogram features as float32 array with shape `(n_mels, n_frames)`.

**Example:**

```python
import numpy as np
from wakeword_workbench.export.microwakeword import compute_mel_spectrogram

# Generate 1 second of random audio at 16kHz
sample_rate = 16000
duration = 1.0
num_samples = int(sample_rate * duration)
audio = np.random.randn(num_samples).astype(np.float32) * 0.1

# Compute mel-spectrogram with default parameters
features = compute_mel_spectrogram(audio, sample_rate=sample_rate)

print(f"Feature shape: {features.shape}")  # (40, ~33) for 1 second
print(f"Feature dtype: {features.dtype}")  # float32

# Use different parameters for higher resolution
features_high_res = compute_mel_spectrogram(
    audio,
    sample_rate=sample_rate,
    n_mels=80,      # More mel bands
    hop_length=160, # Finer time resolution (~10ms)
    n_fft=512       # Smaller FFT window
)
print(f"High-res shape: {features_high_res.shape}")  # (80, ~100)
```

**Notes:**

- Output is in log scale (dB) using `librosa.power_to_db()`
- The default `hop_length=480` gives ~30ms frame spacing at 16kHz
- For openWakeWord compatibility, use `n_mels=96`, `n_fft=512`, `hop_length=160`

---

### export_to_mmap()

Export a manifest to microWakeWord Ragged Mmap format for raw audio data.

**Signature:**

```python
def export_to_mmap(
    manifest: Manifest,
    output_dir: Path,
    split: str = "train",
    audio_dir: Path | None = None,
) -> dict[str, Path]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `manifest` | `Manifest` | Required | Manifest containing audio file entries |
| `output_dir` | `Path` | Required | Directory to write output files |
| `split` | `str` | `"train"` | Split name for output files |
| `audio_dir` | `Path \| None` | `None` | Base directory for audio files (optional) |

**Returns:**

`dict[str, Path]` — Dictionary mapping file types to output paths:

```python
{
    "data": Path("train_data.mmap"),      # Raw audio bytes (float32)
    "indices": Path("train_indices.npy"), # Start/end indices (N x 2)
    "labels": Path("train_labels.npy"),   # Labels (N,)
}
```

**Raises:**

- `MicroWakeWordExportError` — If export fails (missing files, I/O errors)

**Example:**

```python
from pathlib import Path
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.export.microwakeword import export_to_mmap

# Create a manifest with positive and negative samples
manifest = Manifest([
    ManifestEntry(path="audio/positive1.wav", label=1, text="hey vera", duration_ms=1000),
    ManifestEntry(path="audio/positive2.wav", label=1, text="hey vera", duration_ms=1200),
    ManifestEntry(path="audio/negative1.wav", label=0, text="hey there", duration_ms=800),
    ManifestEntry(path="audio/negative2.wav", label=0, text="hi vera", duration_ms=900),
])

# Export to mmap format
output_dir = Path("output/microwakeword")
audio_dir = Path("datasets/audio")

result = export_to_mmap(
    manifest,
    output_dir=output_dir,
    split="train",
    audio_dir=audio_dir
)

print(f"Data file: {result['data']}")
print(f"Indices file: {result['indices']}")
print(f"Labels file: {result['labels']}")
```

**Output Files:**

| File | Format | Description |
|------|--------|-------------|
| `{split}_data.mmap` | Binary float32 | Concatenated raw audio samples |
| `{split}_indices.npy` | NumPy int64 (N×2) | Start and end indices for each clip |
| `{split}_labels.npy` | NumPy int32 (N,) | Binary labels (0=negative, 1=positive) |

**Memory Efficiency:**

The mmap format uses memory-mapped files, allowing efficient random access to individual clips without loading the entire dataset into memory:

```python
# Load the mmap dataset
data, indices, labels = load_mmap(
    result["data"],
    result["indices"],
    result["labels"]
)

# Access a specific clip without loading everything
clip_idx = 42
start, end = indices[clip_idx]
clip_audio = data[start:end]  # Memory-mapped access
```

---

### export_with_features()

Export a manifest to microWakeWord format with pre-computed mel-spectrogram features.

**Signature:**

```python
def export_with_features(
    manifest: Manifest,
    output_dir: Path,
    split: str = "train",
    audio_dir: Path | None = None,
    n_mels: int = 40,
    hop_length: int = 480,
) -> dict[str, Path]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `manifest` | `Manifest` | Required | Manifest containing audio file entries |
| `output_dir` | `Path` | Required | Directory to write output files |
| `split` | `str` | `"train"` | Split name for output files |
| `audio_dir` | `Path \| None` | `None` | Base directory for audio files (optional) |
| `n_mels` | `int` | `40` | Number of mel filterbanks |
| `hop_length` | `int` | `480` | Hop length in samples (30ms at 16kHz) |

**Returns:**

`dict[str, Path]` — Dictionary mapping file types to output paths:

```python
{
    "data": Path("train_data.npy"),       # Mel-spectrogram features (n_frames, n_mels)
    "indices": Path("train_indices.npy"), # Start/end indices (N x 2)
    "labels": Path("train_labels.npy"),   # Labels (N,)
    "durations": Path("train_durations.npy"), # Frames per clip (N,)
    "shape": Path("train_shape.npy"),     # Shape metadata (2,)
}
```

**Raises:**

- `MicroWakeWordExportError` — If export fails

**Example:**

```python
from pathlib import Path
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.export.microwakeword import export_with_features
import numpy as np

# Create manifest
manifest = Manifest([
    ManifestEntry(path="audio/sample1.wav", label=1, text="wake word", duration_ms=1000),
    ManifestEntry(path="audio/sample2.wav", label=0, text="other", duration_ms=800),
])

# Export with mel features
output_dir = Path("output/features")
result = export_with_features(
    manifest,
    output_dir=output_dir,
    split="train",
    audio_dir=Path("datasets/audio"),
    n_mels=40,
    hop_length=480
)

# Load and inspect the features
data = np.load(result["data"])
indices = np.load(result["indices"])
labels = np.load(result["labels"])
durations = np.load(result["durations"])
shape = np.load(result["shape"])

print(f"Feature array shape: {data.shape}")  # (total_frames, n_mels)
print(f"Number of clips: {len(labels)}")
print(f"Frames per clip: {durations}")
print(f"Stored shape: {shape}")

# Extract features for a specific clip
clip_idx = 0
start, end = indices[clip_idx]
clip_features = data[start:end]  # Shape: (n_frames, n_mels)
print(f"Clip {clip_idx} features shape: {clip_features.shape}")
```

**When to Use:**

- **Use `export_to_mmap()`** when you need raw audio for on-the-fly feature extraction during training
- **Use `export_with_features()`** when you want faster training startup by pre-computing features

---

### load_mmap()

Load microWakeWord mmap format files into memory.

**Signature:**

```python
def load_mmap(
    data_path: Path,
    indices_path: Path,
    labels_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_path` | `Path` | Path to `*_data.mmap` file |
| `indices_path` | `Path` | Path to `*_indices.npy` file |
| `labels_path` | `Path` | Path to `*_labels.npy` file |

**Returns:**

`tuple[np.ndarray, np.ndarray, np.ndarray]` — Tuple of:
1. `data` — Memory-mapped audio array (float32)
2. `indices` — Start/end indices array (int64, N×2)
3. `labels` — Labels array (int32, N)

**Raises:**

- `MicroWakeWordExportError` — If files cannot be loaded

**Example:**

```python
from pathlib import Path
from wakeword_workbench.export.microwakeword import load_mmap
import numpy as np

# Load exported mmap dataset
data, indices, labels = load_mmap(
    Path("output/train_data.mmap"),
    Path("output/train_indices.npy"),
    Path("output/train_labels.npy")
)

print(f"Total audio samples: {len(data):,}")
print(f"Number of clips: {len(labels)}")
print(f"Labels distribution: {np.bincount(labels)}")

# Extract a specific clip
clip_idx = 10
start, end = indices[clip_idx]
clip_audio = np.array(data[start:end])  # Convert from memmap

print(f"Clip {clip_idx}:")
print(f"  Label: {labels[clip_idx]} ({'positive' if labels[clip_idx] == 1 else 'negative'})")
print(f"  Duration: {len(clip_audio) / 16000:.3f}s")
print(f"  Samples: {len(clip_audio)}")
```

**Memory Efficiency:**

The `data` array is memory-mapped, not loaded into memory. This allows working with datasets larger than available RAM:

```python
# This doesn't load the entire dataset into memory
data, indices, labels = load_mmap(...)

# Only the accessed portion is loaded
for i in range(len(labels)):
    start, end = indices[i]
    clip = data[start:end]  # Only this clip is loaded
    # Process clip...
```

---

### validate_mmap()

Validate microWakeWord mmap format files for integrity and correctness.

**Signature:**

```python
def validate_mmap(
    data_path: Path,
    indices_path: Path,
    labels_path: Path,
    expected_count: int | None = None,
) -> dict[str, bool]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | `Path` | Required | Path to `*_data.mmap` file |
| `indices_path` | `Path` | Required | Path to `*_indices.npy` file |
| `labels_path` | `Path` | Required | Path to `*_labels.npy` file |
| `expected_count` | `int \| None` | `None` | Expected number of clips (optional) |

**Returns:**

`dict[str, bool]` — Dictionary with validation results:

| Key | Description |
|-----|-------------|
| `files_exist` | All required files exist |
| `indices_shape` | Indices array has correct shape (N×2) |
| `labels_shape` | Labels array is 1D |
| `counts_match` | Number of indices matches number of labels |
| `expected_count` | (Optional) Count matches expected value |
| `valid_ranges` | All index ranges are within data bounds |
| `valid_labels` | All labels are 0 or 1 |

**Example:**

```python
from pathlib import Path
from wakeword_workbench.export.microwakeword import validate_mmap

# Validate exported dataset
results = validate_mmap(
    Path("output/train_data.mmap"),
    Path("output/train_indices.npy"),
    Path("output/train_labels.npy"),
    expected_count=1000  # Optional: verify clip count
)

# Check validation results
if all(results.values()):
    print("✓ Validation passed")
else:
    print("✗ Validation failed:")
    for key, passed in results.items():
        if not passed:
            print(f"  - {key}: FAILED")

# Example output for valid dataset:
# {'files_exist': True, 'indices_shape': True, 'labels_shape': True,
#  'counts_match': True, 'expected_count': True, 'valid_ranges': True, 'valid_labels': True}
```

**Common Validation Failures:**

```python
# Missing files
results = validate_mmap(
    Path("nonexistent.mmap"),
    Path("nonexistent.npy"),
    Path("nonexistent.npy")
)
# {'files_exist': False}

# Corrupted indices (start > end)
# Would return: {'valid_ranges': False, ...}

# Invalid labels (not 0 or 1)
# Would return: {'valid_labels': False, ...}
```

---

## OpenWakeWord Export Module

Import the module:

```python
from wakeword_workbench.export.openwakeword import (
    export_to_numpy,
    validate_export,
    ExportConfig,
    OpenWakeWordExportError,
)
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
```

### ExportConfig Dataclass

Configuration for openWakeWord export.

```python
@dataclass
class ExportConfig:
    sample_rate: int = 16000
    format: Literal["raw", "mel"] = "raw"
    n_mels: int = 96
    n_fft: int = 512
    hop_length: int = 160  # ~10ms at 16kHz
    fixed_length: int | None = None
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sample_rate` | `int` | `16000` | Target sample rate for audio |
| `format` | `Literal["raw", "mel"]` | `"raw"` | Export format: raw audio or mel-spectrogram |
| `n_mels` | `int` | `96` | Number of mel bands (for mel format) |
| `n_fft` | `int` | `512` | FFT window size (for mel format) |
| `hop_length` | `int` | `160` | Hop length in samples, ~10ms at 16kHz |
| `fixed_length` | `int \| None` | `None` | Fixed length for padding/cropping |

---

### export_to_numpy()

Export a manifest to openWakeWord numpy format.

**Signature:**

```python
def export_to_numpy(
    manifest: Manifest,
    output_dir: Path | str,
    split: str = "train",
    fixed_length: int | None = None,
    format: Literal["raw", "mel"] = "raw",
    sample_rate: int = 16000,
) -> tuple[Path, Path]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `manifest` | `Manifest` | Required | Manifest containing audio entries |
| `output_dir` | `Path \| str` | Required | Directory to save numpy files |
| `split` | `str` | `"train"` | Dataset split name |
| `fixed_length` | `int \| None` | `None` | Fixed length for padding/cropping |
| `format` | `Literal["raw", "mel"]` | `"raw"` | Export format: raw audio or mel-spectrogram |
| `sample_rate` | `int` | `16000` | Target sample rate |

**Returns:**

`tuple[Path, Path]` — Tuple of (X_path, y_path) pointing to saved files.

**Raises:**

- `OpenWakeWordExportError` — If export fails (empty manifest, missing files, I/O errors)

**Example — Raw Audio Export:**

```python
from pathlib import Path
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.export.openwakeword import export_to_numpy
import numpy as np

# Create manifest with variable-length audio
manifest = Manifest([
    ManifestEntry(path="audio/pos1.wav", label=1, text="hey vera", duration_ms=1000),
    ManifestEntry(path="audio/pos2.wav", label=1, text="hey vera", duration_ms=1500),
    ManifestEntry(path="audio/neg1.wav", label=0, text="hey there", duration_ms=800),
])

# Export raw audio (variable length, padded to max)
output_dir = Path("output/openwakeword")
X_path, y_path = export_to_numpy(
    manifest,
    output_dir=output_dir,
    split="train",
    format="raw"
)

# Load the exported data
X = np.load(X_path)
y = np.load(y_path)

print(f"X shape: {X.shape}")  # (3, max_length)
print(f"y shape: {y.shape}")  # (3,)
print(f"Labels: {y}")  # [1, 1, 0]
```

**Example — Fixed-Length Raw Audio:**

```python
# Export with fixed length (1 second = 16000 samples)
X_path, y_path = export_to_numpy(
    manifest,
    output_dir=output_dir,
    split="train",
    fixed_length=16000,  # Pad shorter, crop longer
    format="raw"
)

X = np.load(X_path)
print(f"X shape: {X.shape}")  # (3, 16000) - all clips same length
```

**Example — Mel-Spectrogram Export:**

```python
# Export mel-spectrogram features
X_path, y_path = export_to_numpy(
    manifest,
    output_dir=output_dir,
    split="train",
    format="mel"  # Pre-computed mel features
)

X = np.load(X_path)
y = np.load(y_path)

# Mel format: (n_samples, time_steps, n_mels)
print(f"X shape: {X.shape}")  # (3, time_steps, 96)
print(f"y shape: {y.shape}")  # (3,)
```

**Example — Fixed-Length Mel Features:**

```python
# Export mel features with fixed time dimension
fixed_time_steps = 100  # 100 frames
X_path, y_path = export_to_numpy(
    manifest,
    output_dir=output_dir,
    split="train",
    fixed_length=fixed_time_steps,
    format="mel"
)

X = np.load(X_path)
print(f"X shape: {X.shape}")  # (n_samples, 100, 96)
```

**Output Files:**

| File | Format | Description |
|------|--------|-------------|
| `X_{split}.npy` | NumPy float32 | Features array (raw audio or mel-spectrogram) |
| `y_{split}.npy` | NumPy int64 | Labels array (0=negative, 1=positive) |

**Format Comparison:**

| Format | X Shape | Use Case |
|--------|---------|----------|
| `raw` (variable) | `(N, max_length)` | Standard ML, variable-length sequences |
| `raw` (fixed) | `(N, fixed_length)` | Fixed-size models, CNNs |
| `mel` (variable) | `(N, time_steps, 96)` | Pre-computed features, faster training |
| `mel` (fixed) | `(N, fixed_frames, 96)` | Fixed-size feature models |

---

### validate_export()

Load and validate exported openWakeWord files.

**Signature:**

```python
def validate_export(
    output_dir: Path | str,
    split: str = "train"
) -> dict[str, np.ndarray]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `Path \| str` | Required | Directory containing export files |
| `split` | `str` | `"train"` | Split name used during export |

**Returns:**

`dict[str, np.ndarray]` — Dictionary with `'X'` and `'y'` arrays.

**Raises:**

- `OpenWakeWordExportError` — If files don't exist or are invalid

**Example:**

```python
from pathlib import Path
from wakeword_workbench.export.openwakeword import export_to_numpy, validate_export

# Export dataset
manifest = Manifest([...])
export_to_numpy(manifest, Path("output/"), split="train")

# Validate and load
try:
    data = validate_export(Path("output/"), split="train")
    X = data["X"]
    y = data["y"]

    print(f"Loaded {len(X)} samples")
    print(f"X shape: {X.shape}")
    print(f"Positive samples: {sum(y == 1)}")
    print(f"Negative samples: {sum(y == 0)}")

except OpenWakeWordExportError as e:
    print(f"Validation failed: {e}")
```

**Validation Checks:**

1. Files exist and are not empty
2. X is at least 2D
3. y is 1D
4. X and y have matching lengths
5. Labels are only 0 or 1
6. No NaN or Inf values

---

## Validator Module

Import the module:

```python
from wakeword_workbench.export.validator import (
    ValidationReport,
    validate_microwakeword,
    validate_openwakeword,
)
```

### ValidationReport Dataclass

Report from validating an export directory.

```python
@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `valid` | `bool` | True if all validation checks passed |
| `errors` | `list[str]` | Critical errors that make the export invalid |
| `warnings` | `list[str]` | Non-critical issues found during validation |
| `stats` | `dict` | Statistics about the validated data |

**Example:**

```python
from wakeword_workbench.export.validator import validate_microwakeword

report = validate_microwakeword(Path("output/"), split="train")

if report.valid:
    print("✓ Export is valid")
    print(f"  Samples: {report.stats['n_samples']}")
    print(f"  Positives: {report.stats['n_positives']}")
    print(f"  Negatives: {report.stats['n_negatives']}")
else:
    print("✗ Export is invalid")
    for error in report.errors:
        print(f"  ERROR: {error}")
    for warning in report.warnings:
        print(f"  WARNING: {warning}")
```

---

### validate_microwakeword()

Validate a MicroWakeWord format export directory.

**Signature:**

```python
def validate_microwakeword(
    export_dir: Path,
    split: str = "train"
) -> ValidationReport
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `export_dir` | `Path` | Required | Path to export directory |
| `split` | `str` | `"train"` | Split name used during export |

**Returns:**

`ValidationReport` — Validation report with errors, warnings, and stats.

**Validation Checks:**

1. **File existence** — Required files present and not empty
2. **Shape consistency** — Labels and indices have matching counts
3. **Data integrity** — No NaN or Inf values
4. **Valid dtypes** — Correct array data types
5. **Index bounds** — All indices within data bounds
6. **Label validity** — Labels are 0 or 1

**Example:**

```python
from pathlib import Path
from wakeword_workbench.export.validator import validate_microwakeword

# Validate exported dataset
report = validate_microwakeword(Path("output/microwakeword/"), split="train")

print(f"Valid: {report.valid}")
print(f"Errors: {len(report.errors)}")
print(f"Warnings: {len(report.warnings)}")

# Access statistics
if report.valid:
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {report.stats['n_samples']}")
    print(f"  Positives: {report.stats['n_positives']}")
    print(f"  Negatives: {report.stats['n_negatives']}")
    print(f"  Feature shape: {report.stats['feature_shape']}")
    print(f"  Is mmap format: {report.stats['is_mmap_format']}")
```

**Stats Dictionary:**

| Key | Type | Description |
|-----|------|-------------|
| `n_samples` | `int` | Number of samples |
| `feature_shape` | `list[int]` | Shape of feature array |
| `feature_dtype` | `str` | Feature array dtype |
| `labels_shape` | `list[int]` | Shape of labels array |
| `labels_dtype` | `str` | Labels array dtype |
| `indices_shape` | `list[int]` | Shape of indices array |
| `indices_dtype` | `str` | Indices array dtype |
| `is_mmap_format` | `bool` | True if mmap format, False if feature format |
| `n_positives` | `int` | Number of positive samples |
| `n_negatives` | `int` | Number of negative samples |
| `features_bytes` | `int` | Size of features in bytes |
| `labels_bytes` | `int` | Size of labels in bytes |
| `indices_bytes` | `int` | Size of indices in bytes |

---

### validate_openwakeword()

Validate an OpenWakeWord format export directory.

**Signature:**

```python
def validate_openwakeword(
    export_dir: Path,
    split: str = "train"
) -> ValidationReport
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `export_dir` | `Path` | Required | Path to export directory |
| `split` | `str` | `"train"` | Split name used during export |

**Returns:**

`ValidationReport` — Validation report with errors, warnings, and stats.

**Validation Checks:**

1. **File existence** — `X_{split}.npy` and `y_{split}.npy` present
2. **Shape matching** — X and y have matching first dimension
3. **Data integrity** — No NaN or Inf values
4. **Label validity** — Labels are 0 or 1
5. **Valid dtypes** — Correct array data types

**Example:**

```python
from pathlib import Path
from wakeword_workbench.export.validator import validate_openwakeword

# Validate exported dataset
report = validate_openwakeword(Path("output/openwakeword/"), split="train")

if report.valid:
    print("✓ Export is valid")
    print(f"  Samples: {report.stats['n_samples']}")
    print(f"  X shape: {report.stats['X_shape']}")
    print(f"  X dtype: {report.stats['X_dtype']}")
    print(f"  Positives: {report.stats['n_positives']}")
    print(f"  Negatives: {report.stats['n_negatives']}")
else:
    print("✗ Validation failed:")
    for error in report.errors:
        print(f"  ERROR: {error}")
```

**Stats Dictionary:**

| Key | Type | Description |
|-----|------|-------------|
| `n_samples` | `int` | Number of samples |
| `X_shape` | `list[int]` | Shape of X array |
| `X_dtype` | `str` | X array dtype |
| `y_shape` | `list[int]` | Shape of y array |
| `y_dtype` | `str` | y array dtype |
| `n_positives` | `int` | Number of positive samples |
| `n_negatives` | `int` | Number of negative samples |
| `X_bytes` | `int` | Size of X in bytes |
| `y_bytes` | `int` | Size of y in bytes |

---

## Integration Examples

### Complete Pipeline: Generate and Export

```python
from pathlib import Path
from wakeword_workbench.config import load_config
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.export.microwakeword import export_to_mmap, validate_mmap
from wakeword_workbench.export.openwakeword import export_to_numpy, validate_export
from wakeword_workbench.export.validator import validate_microwakeword, validate_openwakeword

# Step 1: Load or create manifest
# (In practice, this would come from dataset generation)
manifest = Manifest([
    ManifestEntry(path="audio/pos1.wav", label=1, text="hey vera", duration_ms=1000),
    ManifestEntry(path="audio/pos2.wav", label=1, text="hey vera", duration_ms=1200),
    ManifestEntry(path="audio/neg1.wav", label=0, text="hey there", duration_ms=800),
    ManifestEntry(path="audio/neg2.wav", label=0, text="hi vera", duration_ms=900),
])

# Step 2: Export to MicroWakeWord mmap format
output_dir = Path("output/microwakeword")
audio_dir = Path("datasets/audio")

result = export_to_mmap(
    manifest,
    output_dir=output_dir,
    split="train",
    audio_dir=audio_dir
)

# Step 3: Validate the export
report = validate_microwakeword(output_dir, split="train")

if not report.valid:
    print("Validation failed:")
    for error in report.errors:
        print(f"  ERROR: {error}")
    raise SystemExit(1)

print(f"✓ MicroWakeWord export valid")
print(f"  Samples: {report.stats['n_samples']}")
print(f"  Positives: {report.stats['n_positives']}")
print(f"  Negatives: {report.stats['n_negatives']}")

# Step 4: Export to OpenWakeWord format
output_dir_ow = Path("output/openwakeword")

X_path, y_path = export_to_numpy(
    manifest,
    output_dir=output_dir_ow,
    split="train",
    fixed_length=16000,  # 1 second
    format="raw"
)

# Step 5: Validate OpenWakeWord export
report_ow = validate_openwakeword(output_dir_ow, split="train")

if not report_ow.valid:
    print("OpenWakeWord validation failed:")
    for error in report_ow.errors:
        print(f"  ERROR: {error}")
    raise SystemExit(1)

print(f"✓ OpenWakeWord export valid")
print(f"  X shape: {report_ow.stats['X_shape']}")
print(f"  y shape: {report_ow.stats['y_shape']}")
```

### Multi-Split Export

```python
from pathlib import Path
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.export.microwakeword import export_to_mmap
from wakeword_workbench.export.openwakeword import export_to_numpy

# Create train/val/test splits
train_manifest = Manifest([...])  # Your training data
val_manifest = Manifest([...])    # Your validation data
test_manifest = Manifest([...])    # Your test data

audio_dir = Path("datasets/audio")
output_base = Path("output")

# Export all splits
for split_name, manifest in [
    ("train", train_manifest),
    ("val", val_manifest),
    ("test", test_manifest),
]:
    # MicroWakeWord format
    export_to_mmap(
        manifest,
        output_dir=output_base / "microwakeword",
        split=split_name,
        audio_dir=audio_dir
    )

    # OpenWakeWord format
    export_to_numpy(
        manifest,
        output_dir=output_base / "openwakeword",
        split=split_name,
        fixed_length=16000,
        format="raw"
    )

print("✓ All splits exported")
```

### Feature Export with Custom Parameters

```python
from pathlib import Path
from wakeword_workbench.dataset.metadata import Manifest
from wakeword_workbench.export.microwakeword import export_with_features
import numpy as np

manifest = Manifest([...])
output_dir = Path("output/features")
audio_dir = Path("datasets/audio")

# Export with custom mel-spectrogram parameters
result = export_with_features(
    manifest,
    output_dir=output_dir,
    split="train",
    audio_dir=audio_dir,
    n_mels=80,       # More mel bands for higher resolution
    hop_length=160,  # Finer time resolution (~10ms)
)

# Load and inspect
data = np.load(result["data"])
indices = np.load(result["indices"])
labels = np.load(result["labels"])
durations = np.load(result["durations"])

print(f"Feature shape: {data.shape}")  # (total_frames, 80)
print(f"Number of clips: {len(labels)}")
print(f"Frames per clip: {durations}")

# Extract features for a specific clip
clip_idx = 0
start, end = indices[clip_idx]
clip_features = data[start:end]  # Shape: (n_frames, 80)
print(f"Clip {clip_idx} features: {clip_features.shape}")
```

### Error Handling

```python
from pathlib import Path
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.export.microwakeword import (
    export_to_mmap,
    MicroWakeWordExportError
)
from wakeword_workbench.export.openwakeword import (
    export_to_numpy,
    OpenWakeWordExportError
)

manifest = Manifest([...])
output_dir = Path("output")

try:
    result = export_to_mmap(manifest, output_dir, split="train")
    print(f"✓ Export successful: {result}")

except MicroWakeWordExportError as e:
    print(f"✗ Export failed: {e}")
    # Handle specific errors:
    # - "Audio file not found: ..." — Missing audio file
    # - "Failed to load audio: ..." — Corrupted audio file
    # - "Failed to write data file: ..." — I/O error

except Exception as e:
    print(f"✗ Unexpected error: {e}")
    raise
```

---

## Error Reference

### MicroWakeWordExportError

Raised when microWakeWord export operations fail.

**Common Causes:**

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `Audio file not found: {path}` | Audio file doesn't exist | Check `audio_dir` parameter and file paths |
| `Failed to load audio '{path}': {error}` | Corrupted or unsupported audio file | Verify audio file format (WAV recommended) |
| `Failed to write data file: {error}` | I/O error writing output | Check disk space and permissions |
| `Data file not found: {path}` | Missing mmap file during load | Verify export completed successfully |
| `Failed to load data: {error}` | Corrupted mmap file | Re-run export |

### OpenWakeWordExportError

Raised when openWakeWord export operations fail.

**Common Causes:**

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `Manifest is empty - nothing to export` | Empty manifest | Add entries to manifest before exporting |
| `Failed to export entry {i} ({path}): {error}` | Audio loading failed | Check audio file exists and is valid |
| `X file not found: {path}` | Missing X file during validation | Run export first |
| `y file not found: {path}` | Missing y file during validation | Run export first |
| `Labels must be 0 or 1` | Invalid label values | Ensure labels are binary (0 or 1) |
| `X must be at least 2D, got shape {shape}` | Invalid X array shape | Check export parameters |
| `y length ({len_y}) doesn't match X batch size ({len_x})` | Mismatched arrays | Re-run export |

---

## Best Practices

### Choosing Export Format

| Use Case | Recommended Format | Reason |
|----------|-------------------|--------|
| Large datasets (>10GB) | `export_to_mmap()` | Memory-mapped, efficient random access |
| Fast training startup | `export_with_features()` | Pre-computed features skip feature extraction |
| Standard ML workflow | `export_to_numpy()` | Compatible with sklearn, PyTorch, TensorFlow |
| Fixed-size models | `export_to_numpy(fixed_length=...)` | Uniform input size for CNNs |

### Memory Management

```python
# For large datasets, use mmap format
result = export_to_mmap(manifest, output_dir, split="train")

# Load without loading entire dataset into memory
data, indices, labels = load_mmap(result["data"], result["indices"], result["labels"])

# Process clips one at a time
for i in range(len(labels)):
    start, end = indices[i]
    clip = data[start:end]  # Only this clip is loaded
    # Process clip...
```

### Validation Workflow

```python
from wakeword_workbench.export.validator import validate_microwakeword, validate_openwakeword

def export_and_validate(manifest, output_dir, format="microwakeword"):
    """Export and validate in one step."""
    if format == "microwakeword":
        result = export_to_mmap(manifest, output_dir, split="train")
        report = validate_microwakeword(output_dir, split="train")
    else:
        export_to_numpy(manifest, output_dir, split="train")
        report = validate_openwakeword(output_dir, split="train")

    if not report.valid:
        print("Validation errors:")
        for error in report.errors:
            print(f"  - {error}")
        raise ValueError("Export validation failed")

    if report.warnings:
        print("Validation warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")

    return report
```

### Split Management

```python
from pathlib import Path
from wakeword_workbench.dataset.metadata import Manifest
from wakeword_workbench.export.microwakeword import export_to_mmap

def export_splits(splits: dict[str, Manifest], output_dir: Path):
    """Export multiple splits to the same directory."""
    for split_name, manifest in splits.items():
        export_to_mmap(
            manifest,
            output_dir=output_dir,
            split=split_name,
        )

    # Resulting files:
    # - train_data.mmap, train_indices.npy, train_labels.npy
    # - val_data.mmap, val_indices.npy, val_labels.npy
    # - test_data.mmap, test_indices.npy, test_labels.npy
```
