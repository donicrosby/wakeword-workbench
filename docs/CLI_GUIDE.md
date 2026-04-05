# WakeWord Workbench CLI Guide

Complete reference for all command-line interface commands, arguments, options, and exit codes.

## Table of Contents

- [Overview](#overview)
- [Global Options](#global-options)
- [Commands](#commands)
  - [run](#run)
  - [validate](#validate)
  - [mine](#mine)
  - [merge](#merge)
  - [cache-clear](#cache-clear)
- [Exit Codes](#exit-codes)
- [Common Workflows](#common-workflows)

---

## Overview

WakeWord Workbench provides a CLI for training and evaluating micro wake word detection models. The CLI emphasizes data quality and negative coverage over model architecture.

**Installation:**

```bash
# Install with uv
uv sync

# Activate virtual environment
source .venv/bin/activate

# Verify installation
uv run wakeword-workbench --version
```

**Basic Usage:**

```bash
uv run wakeword-workbench [OPTIONS] COMMAND [ARGS]...
```

---

## Global Options

These options apply to all commands and must be placed before the command name.

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--version` | `-V` | bool | False | Show version and exit |
| `--verbose` | `-v` | bool | False | Enable DEBUG logging |
| `--quiet` | `-q` | bool | False | Only show ERROR logs |

### Examples

```bash
# Show version
uv run wakeword-workbench --version

# Run with verbose logging
uv run wakeword-workbench -v run config.yaml

# Run with minimal output
uv run wakeword-workbench -q validate config.yaml
```

### Logging Behavior

| Flag | Log Level | Use Case |
|------|-----------|-----------|
| (default) | INFO | Normal operation |
| `--verbose` | DEBUG | Debugging, detailed diagnostics |
| `--quiet` | ERROR | Scripts, CI/CD pipelines |

---

## Commands

### run

Run the full training pipeline from a configuration file.

**Purpose:** Execute the complete wake word training workflow including sample generation, augmentation, and dataset export.

**Usage:**

```bash
uv run wakeword-workbench run [OPTIONS] CONFIG_PATH
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `CONFIG_PATH` | Path | Yes | Path to YAML configuration file |

**Options:** None (all configuration is in the YAML file)

**Exit Codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error |
| 2 | Configuration error |

**Examples:**

```bash
# Basic usage
uv run wakeword-workbench run config.yaml

# With verbose logging
uv run wakeword-workbench -v run config.yaml

# Using absolute path
uv run wakeword-workbench run /path/to/config.yaml
```

**Error Scenarios:**

| Error | Cause | Solution |
|-------|-------|----------|
| `Config file not found` | `CONFIG_PATH` does not exist | Verify file path |
| `Config validation failed` | Missing or invalid fields | Run `validate` command first |
| `TTS backend not available` | Backend not installed | Install with `uv sync --extra kokoro` or `uv sync --extra piper` |

**What It Does:**

1. Loads and validates the configuration file
2. Generates positive samples using TTS
3. Creates negative samples (confusion phrases, synthetic)
4. Applies audio augmentation (noise, reverb, gain)
5. Splits dataset into train/val/test
6. Exports to specified format(s)

---

### validate

Validate a configuration file without running the pipeline.

**Purpose:** Check configuration syntax and semantics before starting a potentially long training run.

**Usage:**

```bash
uv run wakeword-workbench validate [OPTIONS] CONFIG_PATH
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `CONFIG_PATH` | Path | Yes | Path to YAML configuration file |

**Options:** None

**Exit Codes:**

| Code | Meaning |
|------|---------|
| 0 | Validation passed |
| 2 | Configuration error |

**Examples:**

```bash
# Basic validation
uv run wakeword-workbench validate config.yaml

# Check before running
uv run wakeword-workbench validate config.yaml && \
uv run wakeword-workbench run config.yaml
```

**Output:**

On success, displays:
- Wake word phrase
- Number of positive samples
- TTS backend name

```text
Wake word: hey vera
Samples: 100 positives
TTS backend: kokoro
✓ Config validation passed
```

**Error Scenarios:**

| Error | Cause | Solution |
|-------|-------|----------|
| `Config file not found` | File does not exist | Check file path |
| `Missing required field` | YAML missing required key | Add missing field to config |
| `Invalid value` | Field has wrong type/range | Fix field value |

---

### mine

Mine hard negatives from long audio recordings.

**Purpose:** Extract false positives from real-world audio where the model incorrectly predicts the wake word. These are the most valuable training samples for improving model accuracy.

**Usage:**

```bash
uv run wakeword-workbench mine [OPTIONS]
```

**Options:**

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--model` | `-m` | Path | Yes | — | Path to ONNX model file |
| `--audio` | `-a` | str | Yes | — | Path to audio file(s). Supports wildcards |
| `--output` | `-o` | Path | Yes | — | Output directory for extracted clips |
| `--threshold` | `-t` | float | No | 0.7 | Confidence threshold for extraction (0.0-1.0) |

**Arguments:** None (all inputs are options)

**Exit Codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error |
| 2 | Configuration error |

**Examples:**

```bash
# Basic usage
uv run wakeword-workbench mine \
  --model model.onnx \
  --audio recording.wav \
  --output ./mined_negatives

# With custom threshold
uv run wakeword-workbench mine \
  -m model.onnx \
  -a recording.wav \
  -o ./mined_negatives \
  -t 0.8

# Process multiple files with wildcard
uv run wakeword-workbench mine \
  --model model.onnx \
  --audio "recordings/*.wav" \
  --output ./mined_negatives \
  --threshold 0.75

# Absolute paths
uv run wakeword-workbench mine \
  --model /path/to/model.onnx \
  --audio "/data/audio/*.wav" \
  --output /data/mined_negatives
```

**How It Works:**

1. Loads the ONNX model
2. Processes each audio file with sliding window
3. Identifies segments where model confidence exceeds threshold
4. Extracts clips around false positive predictions
5. Saves clips and creates `manifest.jsonl`

**Output:**

- Extracted audio clips in `OUTPUT` directory
- `manifest.jsonl` with metadata for each clip:
  - Source file path
  - Timestamp in original audio
  - Prediction confidence
  - Threshold used

**Error Scenarios:**

| Error | Cause | Solution |
|-------|-------|----------|
| `Model file not found` | `--model` path invalid | Verify model path |
| `No audio files found` | `--audio` pattern matches nothing | Check file pattern |
| `Failed to load model` | Invalid ONNX file | Re-export model |
| `Failed to create output directory` | Permission denied | Check write permissions |
| `Threshold must be between 0.0 and 1.0` | Invalid threshold | Use value in range |

**Threshold Selection:**

| Threshold | Use Case |
|-----------|----------|
| 0.5-0.6 | Aggressive mining (more false positives, lower quality) |
| 0.7-0.8 | Balanced (recommended) |
| 0.9-1.0 | Conservative (fewer clips, higher quality) |

---

### merge

Merge mined hard negatives into training dataset manifests.

**Purpose:** Append newly mined hard negatives to an existing training manifest, optionally creating a backup.

**Usage:**

```bash
uv run wakeword-workbench merge [OPTIONS]
```

**Options:**

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--source` | `-s` | Path | Yes | — | Path to source manifest (new negatives) |
| `--target` | `-t` | Path | Yes | — | Path to target training manifest |
| `--backup` | `-b` | bool | No | False | Create timestamped backup before merging |

**Arguments:** None (all inputs are options)

**Exit Codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error |
| 2 | Configuration error |

**Examples:**

```bash
# Basic merge
uv run wakeword-workbench merge \
  --source ./mined_negatives/manifest.jsonl \
  --target ./dataset/train.jsonl

# With backup
uv run wakeword-workbench merge \
  -s ./mined_negatives/manifest.jsonl \
  -t ./dataset/train.jsonl \
  --backup

# Short form
uv run wakeword-workbench merge -s source.jsonl -t target.jsonl -b
```

**Output:**

```text
Merge Results:
  Source entries   : 150
  Total in target  : 1150
  Backup created   : ./dataset/train.jsonl.backup.20240315_143022

✓ Merge completed successfully
```

**Backup Behavior:**

When `--backup` is specified:
- Creates copy of target before modification
- Backup filename format: `{target}.backup.{timestamp}`
- Timestamp format: `YYYYMMDD_HHMMSS`

**Error Scenarios:**

| Error | Cause | Solution |
|-------|-------|----------|
| `Source must be a JSONL file` | Source doesn't end in `.jsonl` | Use `.jsonl` extension |
| `Target must be a JSONL file` | Target doesn't end in `.jsonl` | Use `.jsonl` extension |
| `Target manifest not found` | Target file doesn't exist | Create target first or check path |
| `Failed to save manifest` | Write permission denied | Check directory permissions |

**Workflow:**

```bash
# 1. Mine hard negatives
uv run wakeword-workbench mine \
  --model model.onnx \
  --audio "recordings/*.wav" \
  --output ./mined_negatives

# 2. Merge into training set
uv run wakeword-workbench merge \
  --source ./mined_negatives/manifest.jsonl \
  --target ./dataset/train.jsonl \
  --backup

# 3. Re-train model with augmented dataset
uv run wakeword-workbench run config.yaml
```

---

### cache-clear

Clear the TTS synthesis cache.

**Purpose:** Remove cached TTS audio files to free disk space or force regeneration.

**Usage:**

```bash
uv run wakeword-workbench cache-clear [OPTIONS]
```

**Options:**

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--cache-dir` | — | Path | No | `~/.cache/wakeword_workbench/tts/` | TTS cache directory |
| `--yes` | `-y` | bool | No | False | Skip confirmation prompt |

**Arguments:** None

**Exit Codes:**

| Code | Meaning |
|------|---------|
| 0 | Success (including empty cache) |

**Examples:**

```bash
# Interactive clear (prompts for confirmation)
uv run wakeword-workbench cache-clear

# Skip confirmation
uv run wakeword-workbench cache-clear --yes

# Clear custom cache location
uv run wakeword-workbench cache-clear --cache-dir /custom/cache/path

# Short form
uv run wakeword-workbench cache-clear -y
```

**Output:**

```text
TTS Cache
  Entries : 234
  Size    : 45.67 MB
  Location: /home/user/.cache/wakeword_workbench/tts

Clear all cached entries? [y/N]: y
✓ Cache cleared successfully
```

**Cache Location:**

Default: `~/.cache/wakeword_workbench/tts/`

Contains:
- Synthesized audio files (WAV format)
- SHA256 hash-based filenames
- Metadata for cache lookup

**When to Clear:**

- Free disk space
- Force regeneration after TTS backend update
- Remove corrupted cache entries
- Before running benchmarks

---

## Exit Codes

All commands use consistent exit codes:

| Code | Constant | Meaning | When Used |
|------|----------|---------|-----------|
| 0 | `EXIT_SUCCESS` | Success | Command completed without errors |
| 1 | `EXIT_ERROR` | Runtime error | File I/O errors, model loading failures, processing errors |
| 2 | `EXIT_CONFIG_ERROR` | Configuration error | Invalid config, missing files, validation failures |

### Exit Code Reference

```python
# From cli.py
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_CONFIG_ERROR = 2
```

### Using Exit Codes in Scripts

```bash
#!/bin/bash

uv run wakeword-workbench validate config.yaml
case $? in
  0) echo "Validation passed" ;;
  2) echo "Configuration error"; exit 1 ;;
  *) echo "Unknown error"; exit 1 ;;
esac

uv run wakeword-workbench run config.yaml
if [ $? -eq 0 ]; then
  echo "Training completed"
else
  echo "Training failed"
  exit 1
fi
```

### Exit Code Mapping by Command

| Command | Exit 0 | Exit 1 | Exit 2 |
|---------|--------|--------|--------|
| `run` | Pipeline completed | Processing error | Config not found, validation failed |
| `validate` | Config valid | — | Config not found, validation failed |
| `mine` | Clips extracted | Model/audio error | Invalid paths, threshold out of range |
| `merge` | Manifests merged | I/O error | Invalid paths, file not found |
| `cache-clear` | Cache cleared (or empty) | — | — |

---

## Common Workflows

### Initial Setup and Validation

```bash
# 1. Create config file
cat > config.yaml << 'EOF'
wake_word: "hey vera"
samples:
  positives: 100
  negatives_multiplier: 5
tts:
  backend: "kokoro"
  voices: ["af_sarah"]
  speed: 1.0
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: "./output/dataset"
  format: ["microwakeword"]
EOF

# 2. Validate configuration
uv run wakeword-workbench validate config.yaml

# 3. Run pipeline
uv run wakeword-workbench run config.yaml
```

### Hard Negative Mining Workflow

```bash
# 1. Mine hard negatives from recordings
uv run wakeword-workbench mine \
  --model model.onnx \
  --audio "recordings/*.wav" \
  --output ./mined_negatives \
  --threshold 0.75

# 2. Review extracted clips
ls ./mined_negatives/

# 3. Merge into training set
uv run wakeword-workbench merge \
  --source ./mined_negatives/manifest.jsonl \
  --target ./dataset/train.jsonl \
  --backup

# 4. Re-train with augmented dataset
uv run wakeword-workbench run config.yaml
```

### Iterative Improvement

```bash
# Loop: train -> mine -> merge -> retrain
for i in {1..3}; do
  echo "=== Iteration $i ==="
  
  # Train model
  uv run wakeword-workbench run config.yaml
  
  # Mine hard negatives
  uv run wakeword-workbench mine \
    -m output/model.onnx \
    -a "test_audio/*.wav" \
    -o "./mined_$i" \
    -t 0.7
  
  # Merge into training set
  uv run wakeword-workbench merge \
    -s "./mined_$i/manifest.jsonl" \
    -t ./dataset/train.jsonl \
    --backup
done
```

### Cache Management

```bash
# Check cache size
du -sh ~/.cache/wakeword_workbench/tts/

# Clear cache before benchmarking
uv run wakeword-workbench cache-clear --yes

# Run training with fresh TTS
uv run wakeword-workbench run config.yaml
```

### CI/CD Integration

```bash
#!/bin/bash
set -e

# Validate config
uv run wakeword-workbench validate config.yaml

# Run with quiet logging for CI
uv run wakeword-workbench --quiet run config.yaml

# Exit code 0 = success
# Exit code 1 = error (fail the build)
# Exit code 2 = config error (fail the build)
```

---

## See Also

- [Configuration Guide](../examples/) — Sample configuration files
- [AGENTS.md](../AGENTS.md) — Project structure and conventions
- [README.md](../README.md) — Project overview and installation