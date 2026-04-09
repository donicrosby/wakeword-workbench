# WakeWord Workbench API Usage Guide

This document provides comprehensive API documentation for the WakeWord Workbench Python library, covering all major modules with working code examples.

## Setup checkpoint

Before running API examples, initialize the repo and verify CLI/config wiring:

```bash
uv sync --group dev
source .venv/bin/activate
uv run pre-commit install
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If validation reports a missing TTS backend, install `uv sync --extra kokoro` or `uv sync --extra piper`.

## Table of Contents

1. [Configuration Module](#1-configuration-module)
2. [TTS Module](#2-tts-module)
3. [Augmentation Module](#3-augmentation-module)
4. [Dataset Module](#4-dataset-module)
5. [Evaluation Module](#5-evaluation-module)
6. [Export Module](#6-export-module)
7. [Complete Example](#7-complete-example)

---

## 1. Configuration Module

The configuration module provides YAML-based configuration loading with validation.

### Imports

```python
from wakeword_workbench.config import (
    Config,
    ConfigError,
    SamplesConfig,
    TTSConfig,
    AugmentationConfig,
    OutputConfig,
    load_config,
)
```

### Loading Configuration

```python
from pathlib import Path
from wakeword_workbench.config import load_config, ConfigError

# Load configuration from YAML file
try:
    config = load_config("config.yaml")
    print(f"Wake word: {config.wake_word}")
    print(f"Positives: {config.samples.positives}")
    print(f"TTS providers: {len(config.tts.providers)}")
except ConfigError as e:
    print(f"Configuration error: {e}")
```

### Configuration Structure

```python
from wakeword_workbench.config import Config, SamplesConfig, TTSConfig

# Access nested configuration
config = load_config("config.yaml")

# Wake word
wake_word = config.wake_word  # str

# Samples configuration
samples = config.samples
samples.positives  # int (must be > 0)
samples.negatives_multiplier  # int (must be > 0)

# Negative generation configuration
negatives = config.negatives
negatives.custom_phrases  # list[str] | None (explicit phrases like ["archer"])
negatives.confusion.weight  # float (default 0.6)
negatives.confusion.min_similarity  # float (0.0-1.0)
negatives.synthetic.strategy  # "random", "sentence", or "topic"
negatives.synthetic.min_word_count  # int >= 1

# TTS configuration
tts = config.tts
tts.providers  # list[TTSProviderConfig] (must not be empty)
tts.providers[0].backend  # str (e.g., "kokoro", "piper")
tts.providers[0].voices  # list[str] (must not be empty)
tts.providers[0].speed  # float (0 < speed <= 3.0, default 1.0)

# Augmentation configuration
aug = config.augmentation
aug.noise_snr  # list[float] (exactly 2 values, [min, max])
aug.reverb_probability  # float (0.0 to 1.0)
aug.gain_range  # list[float] (exactly 2 values, [min, max])

# Output configuration
output = config.output
output.path  # str (output directory)
output.format  # list[str] (e.g., ["microwakeword"], ["openwakeword"])
```

### Error Handling

```python
from wakeword_workbench.config import load_config, ConfigError

try:
    config = load_config("missing_config.yaml")
except ConfigError as e:
    # Handle missing file
    print(f"Error: {e}")

try:
    config = load_config("invalid_config.yaml")
except ConfigError as e:
    # Handle validation errors (missing fields, invalid values)
    print(f"Validation failed: {e}")
```

### Sample YAML Configuration

```yaml
wake_word: "hey vera"

samples:
  positives: 100
  negatives_multiplier: 5

tts:
  providers:
    - backend: "kokoro"
      voices:
        - "af_sarah"
        - "am_adam"
      speed: 1.0

augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]

output:
  path: "./output/dataset"
  format: ["microwakeword"]
```

---

## 2. TTS Module

The TTS module provides text-to-speech synthesis with caching and multiple backend support.

### Imports

```python
from wakeword_workbench.tts.base import TTSBackend, TTSResult, TTSError, BackendNotAvailableError
from wakeword_workbench.tts.registry import (
    get_backend,
    list_available_backends,
    list_all_backends,
    register_backend,
    UnknownBackendError,
)
from wakeword_workbench.tts.cache import TTSCache, get_default_cache
```

### Getting a TTS Backend

```python
from wakeword_workbench.tts.registry import get_backend, list_available_backends, UnknownBackendError

# List available backends
available = list_available_backends()
print(f"Available backends: {available}")  # e.g., ['kokoro', 'piper']

# Get a specific backend
try:
    tts = get_backend("kokoro")
except UnknownBackendError as e:
    print(f"Unknown backend: {e}")
```

### Synthesizing Speech

```python
from wakeword_workbench.tts.registry import get_backend
from wakeword_workbench.tts.base import TTSError

try:
    tts = get_backend("kokoro")

    # List available voices
    voices = tts.list_voices()
    print(f"Available voices: {voices}")

    # Set a voice
    tts.set_voice(voices[0])

    # Synthesize text
    result = tts.synthesize("hello world")

    # Access results
    print(f"Sample rate: {result.sample_rate} Hz")
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"Audio shape: {result.audio.shape}")

except TTSError as e:
    print(f"TTS synthesis failed: {e}")
```

### Using the TTS Cache

```python
from wakeword_workbench.tts.cache import TTSCache, get_default_cache
from wakeword_workbench.tts.registry import get_backend

# Get the default cache (stored at ~/.cache/wakeword_workbench/tts/)
cache = get_default_cache()

# Or create a custom cache
cache = TTSCache(cache_dir="./tts_cache", max_size_mb=500)

# Check cache before synthesizing
text = "hey vera"
voice = "af_sarah"
backend_name = "kokoro"

result = cache.get(text, voice, backend_name)
if result is None:
    # Cache miss - synthesize and cache
    tts = get_backend(backend_name)
    tts.set_voice(voice)
    result = tts.synthesize(text)
    cache.put(text, voice, backend_name, result)

# Get cache statistics
stats = cache.get_stats()
print(f"Cache entries: {stats['num_entries']}")
print(f"Cache size: {stats['size_mb']:.2f} MB")

# Clear cache when needed
cache.clear()
```

### Error Handling

```python
from wakeword_workbench.tts.base import TTSError, BackendNotAvailableError
from wakeword_workbench.tts.registry import get_backend, UnknownBackendError

try:
    tts = get_backend("nonexistent")
except UnknownBackendError as e:
    print(f"Backend not found: {e}")

try:
    tts = get_backend("kokoro")
except BackendNotAvailableError as e:
    print(f"Backend not available (install dependencies): {e}")

try:
    result = tts.synthesize("test")
except TTSError as e:
    print(f"Synthesis failed: {e}")
```

---

## 3. Augmentation Module

The augmentation module provides audio transformation pipelines for dataset augmentation.

### Imports

```python
from wakeword_workbench.augment.pipeline import (
    Compose,
    Transform,
    from_config,
    minimal_pipeline,
    default_pipeline,
    heavy_pipeline,
)
from wakeword_workbench.augment.gain import AdjustGain, GainTransition
from wakeword_workbench.augment.noise import AddNoise, AddColoredNoise
from wakeword_workbench.augment.reverb import AddReverb
from wakeword_workbench.augment.padding import FixedSizeClip, TrimSilence
```

### Creating a Pipeline

```python
import numpy as np
from wakeword_workbench.augment.pipeline import Compose
from wakeword_workbench.augment.gain import AdjustGain
from wakeword_workbench.augment.noise import AddColoredNoise

# Create a custom pipeline
pipeline = Compose([
    AdjustGain(gain_range=(-45, 0), p=1.0),
    AddColoredNoise(color="pink", snr_range=(-5, 15), p=0.5),
])

# Apply to audio
audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
sample_rate = 16000

augmented = pipeline.apply(audio, sample_rate)
print(f"Input shape: {audio.shape}")
print(f"Output shape: {augmented.shape}")
```

### Using Preset Pipelines

```python
from wakeword_workbench.augment.pipeline import minimal_pipeline, default_pipeline, heavy_pipeline
import numpy as np

# Minimal: gain adjustment only
pipeline = minimal_pipeline()
audio = np.random.randn(16000).astype(np.float32)
augmented = pipeline.apply(audio, 16000)

# Default: gain + colored noise
pipeline = default_pipeline()
augmented = pipeline.apply(audio, 16000)

# Heavy: gain + multiple noise types
pipeline = heavy_pipeline()
augmented = pipeline.apply(audio, 16000)
```

### Loading Pipeline from Config

```python
from wakeword_workbench.augment.pipeline import from_config
from pathlib import Path

# From YAML file
pipeline = from_config(Path("augmentation_config.yaml"))

# From JSON file
pipeline = from_config(Path("augmentation_config.json"))

# From dictionary
config = {
    "transforms": [
        {"type": "AdjustGain", "kwargs": {"gain_range": [-45, 0], "p": 1.0}},
        {"type": "AddColoredNoise", "kwargs": {"color": "pink", "snr_range": (-5, 15), "p": 0.5}},
        {"type": "AddReverb", "kwargs": {"rir_dir": "path/to/rir/", "p": 0.3}},
    ]
}
pipeline = from_config(config)
```

### Individual Transforms

```python
import numpy as np
from wakeword_workbench.augment.gain import AdjustGain
from wakeword_workbench.augment.noise import AddNoise, AddColoredNoise
from wakeword_workbench.augment.reverb import AddReverb
from wakeword_workbench.augment.padding import FixedSizeClip

audio = np.random.randn(16000).astype(np.float32)
sr = 16000

# Adjust gain (random in range)
gain = AdjustGain(gain_range=(-20, 0), p=1.0)
augmented = gain.apply(audio, sr)

# Add colored noise (pink, brown, white)
noise = AddColoredNoise(color="pink", snr_range=(-10, 10), p=0.5)
augmented = noise.apply(audio, sr)

# Add reverb (requires a directory with RIR WAV files)
reverb = AddReverb(rir_dir="path/to/rir/", p=0.3)
augmented = reverb.apply(audio, sr)

# Fixed-size clipping
clipper = FixedSizeClip(target_samples=16000, mode="pad", jitter=False)
augmented = clipper.apply(audio, sr)
```

### Error Handling

```python
from wakeword_workbench.augment.pipeline import Compose, from_config

try:
    # Invalid transform type
    pipeline = from_config({"transforms": [{"type": "NonexistentTransform"}]})
except ValueError as e:
    print(f"Invalid transform: {e}")

try:
    # Transform that doesn't implement protocol
    pipeline = Compose([object()])  # Will raise TypeError
except TypeError as e:
    print(f"Invalid transform: {e}")
```

---

## 4. Dataset Module

The dataset module provides manifest management for training datasets.

### Imports

```python
from wakeword_workbench.dataset.metadata import (
    Manifest,
    ManifestEntry,
    ManifestError,
    ManifestValidationError,
)
```

### Creating Manifests

```python
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry

# Create individual entries
entry1 = ManifestEntry(
    path="audio/positive_001.wav",
    label=1,  # 1 = wake word present
    text="hey vera",
    voice="af_sarah",
    duration_ms=1500,
    sample_rate=16000,
)

entry2 = ManifestEntry(
    path="audio/negative_001.wav",
    label=0,  # 0 = wake word absent
    text="hello world",
    voice="am_adam",
    duration_ms=1200,
)

# Create manifest and add entries
manifest = Manifest()
manifest.add(entry1)
manifest.add(entry2)

# Or create with initial entries
manifest = Manifest([entry1, entry2])

# Add multiple entries at once
manifest.extend([entry1, entry2])
```

### Saving and Loading

```python
from pathlib import Path
from wakeword_workbench.dataset.metadata import Manifest, ManifestError

# Save manifest to JSONL
manifest.save(Path("output/train.jsonl"))

# Load manifest from JSONL
try:
    loaded = Manifest.load(Path("output/train.jsonl"))
    print(f"Loaded {len(loaded)} entries")
except ManifestError as e:
    print(f"Failed to load manifest: {e}")

# Iterate over entries
for entry in loaded:
    print(f"Path: {entry.path}, Label: {entry.label}")
```

### Validation

```python
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry, ManifestValidationError

# Create manifest with invalid entry (absolute path is invalid)
entry = ManifestEntry(
    path="/absolute/path.wav",  # Invalid: must be relative
    label=0,
    text="test",
)

manifest = Manifest([entry])

# Validate and get errors
errors = manifest.validate()
print(f"Validation errors: {errors}")
# Output: ["Entry 0: path must be relative, got absolute path '/absolute/path.wav'"]

# Save will raise on validation failure
try:
    manifest.save("output.jsonl")
except ManifestValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Errors: {e.errors}")
```

### Merging Manifests

```python
from wakeword_workbench.dataset.metadata import Manifest

# Load two manifests
train = Manifest.load("train.jsonl")
val = Manifest.load("val.jsonl")

# Merge into new manifest
combined = train.merge(val)
print(f"Combined entries: {len(combined)}")
```

---

## 5. Evaluation Module

The evaluation module provides metrics calculation and report generation.

### Imports

```python
from wakeword_workbench.eval.report import (
    EvaluationReport,
    generate_report,
)
from wakeword_workbench.eval.far import calculate_far, count_false_positives
from wakeword_workbench.eval.frr import calculate_frr, count_false_negatives
```

### Calculating FAR (False Acceptance Rate)

```python
import numpy as np
from wakeword_workbench.eval.far import calculate_far, count_false_positives

# Model confidence scores for each frame
predictions = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6])

# Calculate false accepts per hour
audio_duration_hours = 2.0
far = calculate_far(
    predictions=predictions,
    ground_truth=None,  # Not needed for FAR
    audio_duration_hours=audio_duration_hours,
    threshold=0.5,
)
print(f"FAR: {far:.4f} false accepts/hour")

# Count raw false positives
fp_count = count_false_positives(predictions, threshold=0.5)
print(f"False positives: {fp_count}")
```

### Calculating FRR (False Rejection Rate)

```python
import numpy as np
from wakeword_workbench.eval.frr import calculate_frr, count_false_negatives

# Model confidence scores
predictions = np.array([0.9, 0.3, 0.8, 0.2, 0.6])

# Ground truth: 1 = wake word present, 0 = absent
ground_truth = np.array([1, 1, 1, 0, 0])

# Calculate false rejection rate
frr = calculate_frr(
    predictions=predictions,
    ground_truth=ground_truth,
    threshold=0.5,
)
print(f"FRR: {frr:.4f}")  # 0.333 (1 false negative out of 3 positives)

# Count raw false negatives
fn_count = count_false_negatives(predictions, ground_truth, threshold=0.5)
print(f"False negatives: {fn_count}")
```

### Generating Evaluation Reports

```python
import numpy as np
from wakeword_workbench.eval.report import generate_report

# Sample evaluation data
results = {
    "predictions": np.random.rand(1000),  # Model confidence scores
    "ground_truth": np.random.randint(0, 2, 1000),  # Binary labels
    "audio_duration_hours": 2.5,
    "total_samples": 1000,
    "metadata": {
        "model": "my_wake_word_model",
        "wake_word": "hey vera",
    },
}

# Generate JSON report
json_report = generate_report(results, format="json")
print(json_report)

# Generate CSV report
csv_report = generate_report(results, format="csv")

# Generate Markdown report
md_report = generate_report(results, format="markdown")

# Save to file
from pathlib import Path
output_path = generate_report(results, format="json", output_path=Path("eval_report.json"))
```

### Report Structure

```python
from wakeword_workbench.eval.report import EvaluationReport

# The EvaluationReport dataclass contains:
# - far: False Acceptance Rate (false accepts per hour)
# - frr: False Rejection Rate (0.0 to 1.0)
# - eer: Equal Error Rate (where FAR equals FRR)
# - optimal_threshold: Threshold where FAR and FRR are balanced
# - audio_duration_hours: Total audio duration
# - total_samples: Number of samples evaluated
# - timestamp: ISO timestamp of report generation
# - metadata: Dict with additional context

# Convert to dictionary
report_dict = report.to_dict()
```

---

## 6. Export Module

The export module provides dataset export for model training.

### Imports

```python
from wakeword_workbench.export.openwakeword import (
    export_to_numpy,
    validate_export,
    ExportConfig,
    OpenWakeWordExportError,
)
```

### Exporting to NumPy Format

```python
from pathlib import Path
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.export.openwakeword import export_to_numpy, OpenWakeWordExportError

# Create or load a manifest
manifest = Manifest([
    ManifestEntry(path="audio/positive_001.wav", label=1, text="hey vera", duration_ms=1500),
    ManifestEntry(path="audio/negative_001.wav", label=0, text="hello world", duration_ms=1200),
])

# Export to numpy arrays
try:
    X_path, y_path = export_to_numpy(
        manifest=manifest,
        output_dir=Path("output/numpy"),
        split="train",
        fixed_length=16000,  # Pad/crop to 1 second
        format="raw",  # or "mel" for mel-spectrogram
        sample_rate=16000,
    )
    print(f"Features saved to: {X_path}")
    print(f"Labels saved to: {y_path}")
except OpenWakeWordExportError as e:
    print(f"Export failed: {e}")
```

### Exporting Mel Spectrograms

```python
from pathlib import Path
from wakeword_workbench.export.openwakeword import export_to_numpy

# Export with mel-spectrogram features
X_path, y_path = export_to_numpy(
    manifest=manifest,
    output_dir=Path("output/numpy"),
    split="train",
    format="mel",  # Mel-spectrogram features
    n_mels=96,
    n_fft=512,
    hop_length=160,
)
```

### Validating Exports

```python
from pathlib import Path
from wakeword_workbench.export.openwakeword import validate_export, OpenWakeWordExportError

try:
    data = validate_export(Path("output/numpy"), split="train")
    print(f"X shape: {data['X'].shape}")
    print(f"y shape: {data['y'].shape}")
    print(f"Unique labels: {set(data['y'])}")
except OpenWakeWordExportError as e:
    print(f"Validation failed: {e}")
```

### Export Configuration

```python
from wakeword_workbench.export.openwakeword import ExportConfig

# Create export configuration
config = ExportConfig(
    sample_rate=16000,
    format="mel",  # or "raw"
    n_mels=96,
    n_fft=512,
    hop_length=160,  # ~10ms at 16kHz
    fixed_length=None,  # Variable length
)
```

---

## 7. Complete Example

This example demonstrates a complete workflow combining multiple modules.

```python
from pathlib import Path
import numpy as np

from wakeword_workbench.config import load_config, ConfigError
from wakeword_workbench.tts.registry import get_backend, list_available_backends
from wakeword_workbench.tts.cache import TTSCache
from wakeword_workbench.augment.pipeline import default_pipeline
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.eval.report import generate_report
from wakeword_workbench.export.openwakeword import export_to_numpy


def main():
    # 1. Load configuration
    try:
        config = load_config("config.yaml")
        print(f"Wake word: {config.wake_word}")
        print(f"Positives: {config.samples.positives}")
    except ConfigError as e:
        print(f"Configuration error: {e}")
        return

    # 2. Check TTS backends
    available = list_available_backends()
    print(f"Available TTS backends: {available}")

    if not available:
        print("No TTS backends available. Install kokoro or piper extras.")
        return

    # 3. Initialize first provider with caching
    provider = config.tts.providers[0]
    tts = get_backend(provider.backend)
    cache = TTSCache(max_size_mb=500)

    # 4. Generate positive samples
    manifest = Manifest()

    for voice in provider.voices[:2]:  # Use first 2 voices
        tts.set_voice(voice)

        # Check cache first
        result = cache.get(config.wake_word, voice, provider.backend, speed=provider.speed)
        if result is None:
            result = tts.synthesize(config.wake_word)
            cache.put(config.wake_word, voice, provider.backend, result, speed=provider.speed)

        # Create manifest entry
        entry = ManifestEntry(
            path=f"audio/{voice}_positive.wav",
            label=1,
            text=config.wake_word,
            voice=voice,
            duration_ms=int(result.duration * 1000),
            sample_rate=result.sample_rate,
        )
        manifest.add(entry)

    # 5. Apply augmentation
    pipeline = default_pipeline()

    # Simulate audio processing
    audio = np.random.randn(16000).astype(np.float32)
    augmented = pipeline.apply(audio, 16000)
    print(f"Augmented audio shape: {augmented.shape}")

    # 6. Save manifest
    output_dir = Path(config.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest.save(output_dir / "train.jsonl")
    print(f"Saved manifest with {len(manifest)} entries")

    # 7. Generate evaluation report (simulated)
    results = {
        "predictions": np.random.rand(100),
        "ground_truth": np.random.randint(0, 2, 100),
        "audio_duration_hours": 1.0,
        "metadata": {"wake_word": config.wake_word},
    }

    report = generate_report(results, format="markdown")
    print("\nEvaluation Report:")
    print(report)

    # 8. Export for training
    try:
        X_path, y_path = export_to_numpy(
            manifest=manifest,
            output_dir=output_dir / "numpy",
            split="train",
            format="raw",
        )
        print(f"\nExported to: {X_path}")
    except Exception as e:
        print(f"Export error: {e}")


if __name__ == "__main__":
    main()
```

---

## Error Handling Summary

All modules use domain-specific exceptions for clear error handling:

```python
# Configuration errors
from wakeword_workbench.config import ConfigError

# TTS errors
from wakeword_workbench.tts.base import TTSError, BackendNotAvailableError
from wakeword_workbench.tts.registry import UnknownBackendError

# Dataset errors
from wakeword_workbench.dataset.metadata import ManifestError, ManifestValidationError

# Export errors
from wakeword_workbench.export.openwakeword import OpenWakeWordExportError

# Augmentation errors (ValueError for invalid configs)
from wakeword_workbench.augment.pipeline import from_config
```

---

## Logging

The project uses structured logging with `structlog`:

```python
from wakeword_workbench.logging_config import get_logger

log = get_logger()
log.info("processing_started", count=100, file="train.jsonl")
log.debug("cache_hit", key="abc123")
log.error("synthesis_failed", error="backend unavailable")
```
