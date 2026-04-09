# WakeWord Workbench Tutorial

A complete end-to-end walkthrough for training micro wake word detection models. This tutorial uses a concrete example throughout: training a model for the wake word "Hey Helper".

## Setup checkpoint

Before starting the tutorial, bootstrap and smoke-test the repo:

```bash
uv sync --group dev
source .venv/bin/activate
uv run pre-commit install
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If your chosen config uses a missing TTS backend, install `uv sync --extra kokoro` or `uv sync --extra piper` first.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Create Your First Config](#create-your-first-config)
4. [Validate Config](#validate-config)
5. [Generate Dataset](#generate-dataset)
6. [Hard Negative Mining Workflow](#hard-negative-mining-workflow)
7. [Augmentation](#augmentation)
8. [Evaluation](#evaluation)
9. [Export](#export)
10. [Next Steps](#next-steps)

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.11 or higher** — Required for the toolkit
- **uv package manager** — Fast Python package installer ([install uv](https://docs.astral.sh/uv/))
- **Wake word phrase chosen** — We'll use "Hey Helper" throughout this tutorial

### Verify Prerequisites

```bash
# Check Python version (must be 3.11+)
python --version

# Check uv is installed
uv --version
```

### What You'll Build

By the end of this tutorial, you'll have:

1. A validated configuration for "Hey Helper"
2. A dataset of positive and negative samples
3. Hard negative samples mined from real audio
4. An exported model ready for deployment

---

## Installation

### Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd wakeword-workbench

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Install TTS Backend

You need at least one TTS backend to generate positive samples:

```bash
# Option 1: Kokoro TTS (recommended)
uv sync --extra kokoro

# Option 2: Piper TTS
uv sync --extra piper

# Option 3: All backends
uv sync --all-extras
```

### Verify Installation

```bash
# Check version
uv run wakeword-workbench --version

# Expected output:
# WakeWord Workbench v0.1.0-alpha
```

---

## Create Your First Config

Create a configuration file for "Hey Helper" wake word.

### Create `config.yaml`

```bash
# Create config file
cat > config.yaml << 'EOF'
# Wake Word Dataset Configuration
# Target phrase: "Hey Helper"

wake_word: "Hey Helper"

samples:
  positives: 1000
  negatives_multiplier: 5

tts:
  providers:
    - backend: kokoro
      voices:
        - af_bella
        - af_nicole
        - af_sarah
        - am_adam
        - am_michael
      speed: 1.0

augmentation:
  noise_snr: [-5, 15]
  reverb_probability: 0.3
  gain_range: [-45, 0]

output:
  path: ./output/hey_helper
  format: [microwakeword]
EOF
```

### Understanding the Config

| Section | Purpose |
|---------|---------|
| `wake_word` | The phrase to train for: "Hey Helper" |
| `samples.positives` | Generate 1000 positive samples |
| `samples.negatives_multiplier` | Generate 5000 negatives (1000 × 5) |
| `negatives.confusion` / `negatives.synthetic` | Control which negative generators run and how strongly each contributes |
| `negatives.custom_phrases` | Force known false-trigger phrases like similar names into the negative set |
| `tts.providers[].backend` | Use Kokoro TTS engine |
| `tts.providers[].voices` | 5 different voices for diversity |
| `tts.providers[].speed` | Normal speech rate |
| `augmentation.noise_snr` | Add noise from -5dB to 15dB SNR |
| `augmentation.reverb_probability` | 30% of samples get reverb |
| `augmentation.gain_range` | Volume from -45dB to 0dB |
| `output.path` | Save to `./output/hey_helper` |
| `output.format` | Export for microWakeWord trainer |

### Why These Values?

- **1000 positives**: Enough for initial training, scalable later
- **5x negatives**: Wake word models need many more negatives than positives
- **Multiple voices**: Improves model generalization across speakers
- **Moderate augmentation**: Balances diversity and audio quality

For detailed configuration options, see [Config Reference](CONFIG_REFERENCE.md).

---

## Validate Config

Before running the pipeline, validate your configuration.

### Run Validation

```bash
uv run wakeword-workbench validate config.yaml
```

### Expected Output

```
╭─────────────────────────────────────────────╮
│ WakeWord Workbench - Validating config      │
╰─────────────────────────────────────────────╯

Wake word: Hey Helper
Samples: 1000 positives
TTS providers:
  - kokoro (5 voices) speed=1.0
✓ Config validation passed
```

### Common Validation Errors

#### Missing TTS Backend

**Error:**
```
Error: Backend 'kokoro' not available
```

**Solution:**
```bash
uv sync --extra kokoro
```

#### Invalid Config Field

**Error:**
```
Error: ConfigError: positives must be positive
```

**Solution:** Fix the value in `config.yaml`:
```yaml
samples:
  positives: 1000  # Must be > 0
```

#### Missing Required Field

**Error:**
```
Error: ConfigError: Missing required field: wake_word
```

**Solution:** Add the missing field to `config.yaml`.

For complete error reference, see [Config Reference - Common Errors](CONFIG_REFERENCE.md#common-errors).

---

## Generate Dataset

Run the full training pipeline to generate your dataset.

### Run Pipeline

```bash
uv run wakeword-workbench run config.yaml
```

### Expected Output

```
╭─────────────────────────────────────────────╮
│ WakeWord Workbench - Starting training      │
│ pipeline                                    │
╰─────────────────────────────────────────────╯

⠋ Initializing generator...
Dataset generation complete

✓ Pipeline completed successfully
```

**Note:** `run` executes dataset generation from your validated config. Use smaller sample counts first to verify your setup quickly.

### What the Pipeline Does

The pipeline currently:

1. **Generate positives** — Synthesize "Hey Helper" with TTS
2. **Generate negatives** — Create confusion phrases and synthetic samples
3. **Apply augmentation** — Add noise, reverb, gain variations
4. **Split dataset** — Create train/val/test splits
5. **Export** — Save in microWakeWord format

### Pipeline Status

| Component | Status |
|-----------|--------|
| TTS synthesis | ✅ Working |
| Augmentation | ✅ Working |
| Hard negative mining | ✅ Working |
| Dataset generation pipeline | ✅ Working |

---

## Hard Negative Mining Workflow

**This is the core working feature.** Hard negative mining extracts false positives from real-world audio where your model incorrectly predicts "Hey Helper". These are the most valuable training samples for improving accuracy.

### Why Hard Negatives Matter

Hard negatives are audio segments that:
- Do NOT contain "Hey Helper"
- But the model predicts "Hey Helper" with high confidence
- Represent real-world false positives

Training on these samples dramatically improves model accuracy because they're the exact cases where your model fails.

### Step 1: Obtain Long Audio Recordings

You need audio files that:
- Do NOT contain your wake word
- Represent real-world environments (TV, podcasts, conversations)
- Are long enough to contain potential false positives

**Good sources:**
- Podcast recordings
- TV show audio
- Meeting recordings
- Background noise from smart speakers

**Example directory structure:**
```
recordings/
├── podcast_episode_1.wav
├── podcast_episode_2.wav
├── tv_show_audio.wav
└── meeting_recording.wav
```

### Step 2: Run Mining Command

Use your trained model to find false positives:

```bash
uv run wakeword-workbench mine \
  --model output/hey_helper/model.onnx \
  --audio "recordings/*.wav" \
  --output ./mined_negatives \
  --threshold 0.7
```

### Mining Command Breakdown

| Option | Value | Purpose |
|--------|-------|---------|
| `--model` | `model.onnx` | Your trained "Hey Helper" model |
| `--audio` | `"recordings/*.wav"` | Wildcard pattern for audio files |
| `--output` | `./mined_negatives` | Directory for extracted clips |
| `--threshold` | `0.7` | Confidence threshold (0.0-1.0) |

### Threshold Selection Guide

| Threshold | Clips Extracted | Quality | Use Case |
|-----------|-----------------|---------|----------|
| 0.5-0.6 | Many | Lower | Aggressive mining, initial pass |
| 0.7-0.8 | Moderate | High | **Recommended** |
| 0.9-1.0 | Few | Highest | Conservative, final refinement |

### Expected Output

```
╭─────────────────────────────────────────────╮
│ WakeWord Workbench - Mining hard negatives  │
╰─────────────────────────────────────────────╯

⠋ Processing audio files...
✓ Extracted 47 clips
✓ Saved manifest to ./mined_negatives/manifest.jsonl
```

### Step 3: Review Extracted Clips

Check what was extracted:

```bash
# List extracted clips
ls ./mined_negatives/

# Expected output:
# clip_001.wav
# clip_002.wav
# ...
# clip_047.wav
# manifest.jsonl
```

### Directory Structure After Mining

```
mined_negatives/
├── clip_001.wav          # Extracted false positive
├── clip_002.wav          # Extracted false positive
├── ...
├── clip_047.wav          # Extracted false positive
└── manifest.jsonl        # Metadata for all clips
```

### Manifest Format

Each entry in `manifest.jsonl` contains:

```json
{
  "path": "clip_001.wav",
  "label": 0,
  "text": "",
  "voice": null,
  "duration_ms": 1500,
  "sample_rate": 16000,
  "metadata": {
    "source": "recordings/podcast_episode_1.wav",
    "timestamp": 45.2,
    "prediction": 0.82,
    "threshold": 0.7
  }
}
```

| Field | Value | Meaning |
|-------|-------|---------|
| `label` | `0` | Negative sample (not "Hey Helper") |
| `metadata.source` | Original file | Where the clip came from |
| `metadata.timestamp` | 45.2 seconds | Position in original audio |
| `metadata.prediction` | 0.82 | Model confidence (82%) |
| `metadata.threshold` | 0.7 | Threshold used for extraction |

### Step 4: Merge into Training Data

Add mined negatives to your training dataset:

```bash
uv run wakeword-workbench merge \
  --source ./mined_negatives/manifest.jsonl \
  --target ./output/hey_helper/train.jsonl \
  --backup
```

### Expected Merge Output

```
╭─────────────────────────────────────────────╮
│ WakeWord Workbench - Merging hard negatives │
╰─────────────────────────────────────────────╯

⠋ Merging manifests...

Merge Results:
  Source entries   : 47
  Total in target  : 1047
  Backup created   : ./output/hey_helper/train.jsonl.backup.20260404_143022

✓ Merge completed successfully
```

### Backup Behavior

The `--backup` flag creates a timestamped backup:
- Original: `train.jsonl`
- Backup: `train.jsonl.backup.20260404_143022`

This allows you to revert if needed.

### Iterative Mining Workflow

For best results, iterate:

```bash
# Iteration 1: Initial training
uv run wakeword-workbench run config.yaml

# Iteration 1: Mine hard negatives
uv run wakeword-workbench mine \
  --model output/hey_helper/model.onnx \
  --audio "recordings/*.wav" \
  --output ./mined_1 \
  --threshold 0.7

# Iteration 1: Merge into training set
uv run wakeword-workbench merge \
  --source ./mined_1/manifest.jsonl \
  --target ./output/hey_helper/train.jsonl \
  --backup

# Iteration 2: Retrain with augmented dataset
uv run wakeword-workbench run config.yaml

# Iteration 2: Mine again (model improved, find new false positives)
uv run wakeword-workbench mine \
  --model output/hey_helper/model.onnx \
  --audio "recordings/*.wav" \
  --output ./mined_2 \
  --threshold 0.75  # Higher threshold for harder negatives

# Continue iterating...
```

### Mining Best Practices

1. **Start with threshold 0.7** — Balanced quality/quantity
2. **Use diverse audio sources** — Podcasts, TV, meetings, music
3. **Iterate 3-5 times** — Each iteration improves the model
4. **Increase threshold gradually** — 0.7 → 0.75 → 0.8
5. **Always backup** — Use `--backup` flag when merging

For command details, see [CLI Guide - mine](CLI_GUIDE.md#mine).

---

## Augmentation

Audio augmentation increases dataset diversity by applying transformations to your samples.

### Available Transforms

| Transform | Purpose | Config Field |
|-----------|---------|--------------|
| **Noise injection** | Add background noise | `augmentation.noise_snr` |
| **Reverb** | Simulate room acoustics | `augmentation.reverb_probability` |
| **Gain variation** | Adjust volume levels | `augmentation.gain_range` |

### How Augmentation Works

For each positive sample, the pipeline:
1. Synthesizes "Hey Helper" with TTS
2. Randomly applies transforms based on config
3. Creates multiple augmented variants

### Example Augmentation

**Config:**
```yaml
augmentation:
  noise_snr: [-5, 15]      # Add noise at -5dB to 15dB SNR
  reverb_probability: 0.3   # 30% chance of reverb
  gain_range: [-45, 0]     # Volume from -45dB to 0dB
```

**Result:**
- Sample 1: Clean audio (no augmentation)
- Sample 2: Added noise at 10dB SNR
- Sample 3: Added reverb + noise at -2dB SNR
- Sample 4: Volume reduced by 20dB
- ...

### Augmentation Strategy

| Scenario | Config |
|----------|--------|
| **Clean dataset** | `noise_snr: [5, 20]`, `reverb_probability: 0.1` |
| **Robust model** | `noise_snr: [-10, 10]`, `reverb_probability: 0.5` |
| **Noisy environments** | `noise_snr: [-20, 5]`, `reverb_probability: 0.7` |

### Python API for Augmentation

You can also apply augmentation programmatically:

```python
from pathlib import Path
from wakeword_workbench.augment.pipeline import Compose
from wakeword_workbench.augment.noise import AddNoise
from wakeword_workbench.augment.reverb import AddReverb
from wakeword_workbench.augment.gain import AdjustGain
import numpy as np

# Create augmentation pipeline
pipeline = Compose([
    AddNoise(snr_db=10.0),
    AddReverb(room_scale=0.8, damping=0.5),
    AdjustGain(gain_db=-10.0),
])

# Load audio
audio_path = Path("sample.wav")
audio, sr = load_audio(audio_path)  # Returns np.array and sample rate

# Apply augmentation
augmented = pipeline.apply(audio, sr)
```

---

## Evaluation

Evaluate your model's performance using FAR/FRR metrics.

### Key Metrics

| Metric | Full Name | Meaning |
|--------|-----------|---------|
| **FAR** | False Acceptance Rate | % of negatives incorrectly classified as "Hey Helper" |
| **FRR** | False Rejection Rate | % of positives incorrectly rejected |
| **Threshold** | Decision boundary | Confidence above = wake word detected |

### Trade-off

There's always a trade-off:
- **Lower threshold** → Lower FRR, higher FAR (more false positives)
- **Higher threshold** → Lower FAR, higher FRR (more false negatives)

### Running Evaluation

The evaluation module is available via the Python API:

```python
import numpy as np
from wakeword_workbench.eval.report import generate_report

# Run your model on test samples to get predictions and ground truth
# predictions: array of confidence scores (0.0-1.0) from your model
# ground_truth: binary array (1 = wake word present, 0 = absent)
# audio_duration_hours: total hours of audio evaluated

results = {
    "predictions": np.array([0.2, 0.85, 0.1, 0.92, ...]),  # model confidence scores
    "ground_truth": np.array([0, 1, 0, 1, ...]),             # 1 = wake word present
    "audio_duration_hours": 4.5,                              # total audio hours
    "metadata": {"model": "hey_helper_v1", "test_set": "test.jsonl"},
}

# Generate evaluation report
report = generate_report(results, format="markdown")
print(report)

# Or save to file
report_path = generate_report(results, format="json", output_path="eval_report.json")
```

The report includes FAR, FRR, EER (Equal Error Rate), and the optimal threshold where FAR and FRR are best balanced.

### Interpreting Results

**Good model:**
- FAR < 1% (less than 1 false positive per 100 negatives)
- FRR < 5% (less than 5% of "Hey Helper" missed)

**Needs improvement:**
- FAR > 5% (too many false positives)
- FRR > 10% (too many missed detections)

### Improving Results

If FAR is too high (too many false triggers):
1. Mine more hard negatives
2. **Raise the threshold** — be more conservative about what counts as the wake word
3. Add more negative samples

If FRR is too high (too many missed detections):
1. Add more positive samples
2. Increase augmentation diversity
3. **Lower the threshold** — be more sensitive to catch more true detections

---

## Export

Export your dataset for training frameworks.

### Supported Formats

| Format | Framework | Use Case |
|--------|-----------|----------|
| `microwakeword` | microWakeWord | ESP32, embedded devices |
| `openwakeword` | openWakeWord | Python, larger models |

### Export Configuration

```yaml
output:
  path: ./output/hey_helper
  format: [microwakeword]  # or [openwakeword] or both
```

### Export for Both Frameworks

```yaml
output:
  path: ./output/hey_helper
  format: [microwakeword, openwakeword]
```

### Exported Structure

```
output/hey_helper/
├── train/
│   ├── positive/
│   │   ├── sample_001.wav
│   │   └── ...
│   └── negative/
│       ├── sample_001.wav
│       └── ...
├── val/
│   └── ...
├── test/
│   └── ...
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── config.yaml  # Copy of original config
```

---

## Next Steps

### Iterate on Data Quality

The most important factor in wake word model performance is **data quality**, not model architecture.

1. **Mine more hard negatives** — Use diverse audio sources
2. **Increase positive diversity** — Add more voices, accents
3. **Refine augmentation** — Match your deployment environment

### Continue Mining

```bash
# Find new audio sources
# Podcasts, TV shows, music, conversations

# Mine with current model
uv run wakeword-workbench mine \
  --model output/hey_helper/model.onnx \
  --audio "new_recordings/*.wav" \
  --output ./mined_new \
  --threshold 0.75

# Merge into training set
uv run wakeword-workbench merge \
  --source ./mined_new/manifest.jsonl \
  --target ./output/hey_helper/train.jsonl \
  --backup

# Retrain
uv run wakeword-workbench run config.yaml
```

### Evaluate and Refine

1. **Test on real audio** — Record yourself saying similar phrases
2. **Measure FAR/FRR** — Use held-out test set
3. **Adjust threshold** — Find optimal balance for your use case

### Additional Resources

| Resource | Description |
|----------|-------------|
| [CLI Guide](CLI_GUIDE.md) | Complete command reference |
| [Config Reference](CONFIG_REFERENCE.md) | All configuration options |
| [Background](BACKGROUND.md) | Wake word concepts and theory |
| [AGENTS.md](../AGENTS.md) | Project structure and conventions |

### Troubleshooting

#### TTS Backend Not Available

**Error:**
```
Error: Backend 'kokoro' not available
```

**Solution:**
```bash
# Install Kokoro TTS
uv sync --extra kokoro

# Or install Piper TTS
uv sync --extra piper
```

#### Config Validation Failed

**Error:**
```
Error: ConfigError: Missing required field: wake_word
```

**Solution:** Ensure all required fields are present in `config.yaml`:
- `wake_word`
- `samples.positives`
- `samples.negatives_multiplier`
- `tts.providers`
- `tts.providers[].backend`
- `tts.providers[].voices`
- `augmentation.noise_snr`
- `augmentation.reverb_probability`
- `augmentation.gain_range`
- `output.path`
- `output.format`

See [Config Reference](CONFIG_REFERENCE.md) for complete field documentation.

#### No Audio Files Found in Mining

**Error:**
```
Error: No audio files found matching: recordings/*.wav
```

**Solution:**
1. Check the path pattern is correct
2. Verify files exist: `ls recordings/*.wav`
3. Use absolute paths if needed: `--audio "/full/path/recordings/*.wav"`

#### Out of Memory During Mining

**Error:**
```
Error: Failed to process recording.wav: Out of memory
```

**Solution:**
1. Process fewer files at once
2. Use shorter audio files
3. Split long recordings into smaller chunks

#### Cache Issues

**Problem:** TTS cache taking too much disk space

**Solution:**
```bash
# Check cache size
du -sh ~/.cache/wakeword_workbench/tts/

# Clear cache
uv run wakeword-workbench cache-clear --yes
```

#### Merge Target Not Found

**Error:**
```
Error: Target manifest not found: ./dataset/train.jsonl
```

**Solution:**
1. Create the target manifest first
2. Or check the path is correct
3. Use absolute paths if needed

### Getting Help

1. Check [CLI Guide](CLI_GUIDE.md) for command details
2. Check [Config Reference](CONFIG_REFERENCE.md) for configuration errors
3. Check [Background](BACKGROUND.md) for conceptual questions
4. Review [AGENTS.md](../AGENTS.md) for project conventions

---

## Summary

You've learned:

1. **Installation** — Set up WakeWord Workbench with TTS backends
2. **Configuration** — Create and validate config files
3. **Hard Negative Mining** — Extract false positives from real audio
4. **Augmentation** — Increase dataset diversity
5. **Evaluation** — Measure FAR/FRR metrics
6. **Export** — Prepare datasets for training frameworks

The key insight: **Data quality matters more than model architecture**. Focus on:

- Mining diverse hard negatives
- Using multiple voices for positives
- Matching augmentation to deployment environment
- Iterating until FAR/FRR meet your requirements

For command details, see [CLI Guide](CLI_GUIDE.md). For configuration options, see [Config Reference](CONFIG_REFERENCE.md).
