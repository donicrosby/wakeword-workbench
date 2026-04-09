# Configuration Reference

Complete documentation for WakeWord Workbench configuration files.

## Setup checkpoint

Before validating or editing configs, bootstrap and smoke-test the repo:

```bash
uv sync --group dev
source .venv/bin/activate
uv run pre-commit install
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If your config backend is missing, install `uv sync --extra kokoro`, `uv sync --extra kokoro-cuda`, `uv sync --extra kokoro-openvino`, `uv sync --extra piper`, or `uv sync --extra piper-cuda`.

## Introduction

WakeWord Workbench uses YAML configuration files to define dataset generation parameters. All configuration is validated at load time with clear error messages for invalid values.

Configuration files are loaded via:

```python
from wakeword_workbench.config import load_config

config = load_config("config.yaml")
```

## Quick Example

```yaml
wake_word: "Hey Assistant"

samples:
  positives: 1000
  negatives_multiplier: 5

tts:
  providers:
    - backend: kokoro
      voices:
        - af_bella
        - af_nicole
      speed: 1.0
      acceleration: cuda
    - backend: piper
      voices:
        - en_US-lessac-medium
      speed: 1.0
      acceleration: cuda
      model_path: ./models/en_US-lessac-medium.onnx

augmentation:
  noise_snr: [-5, 15]
  reverb_probability: 0.3
  gain_range: [-45, 0]

output:
  path: ./output
  format: [microwakeword, openwakeword]
```

---

## Configuration Sections

### `wake_word`

**Type:** `string`
**Required:** Yes

The wake word phrase to train the model for.

#### Validation Rules

- Cannot be empty
- Cannot be missing from config

#### Error Messages

| Condition | Error |
|-----------|-------|
| Field missing | `ConfigError: Missing required field: wake_word` |
| Empty string | `ConfigError: wake_word cannot be empty` |

#### Examples

```yaml
wake_word: "hey vera"
wake_word: "Hey Assistant"
wake_word: "okay google"
```

By default, positive sample generation uses only this exact wake word.

---

### `wake_word_variants`

**Type:** `list[string]`
**Required:** No

Optional explicit list of positive phrases to synthesize. If provided, the pipeline uses only this list and does not invent additional text variants automatically.

#### Validation Rules

- Cannot be an empty list
- Cannot contain blank values
- Duplicate entries are removed while preserving order

#### Examples

```yaml
# Exact wake word only (default when omitted)
wake_word: "hey vera"

# Explicit curated positive phrase list
wake_word: "hey vera"
wake_word_variants:
  - "hey vera"
  - "hey, vera"
  - "hey vera please"
```

---

### `samples`

**Type:** `SamplesConfig` (nested object)
**Required:** Yes

Controls the number of positive and negative samples to generate.

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `positives` | `int` | Yes | Number of positive samples to generate |
| `negatives_multiplier` | `int` | Yes | Multiplier for negative samples (total negatives = positives × multiplier) |

#### Validation Rules

- `positives` must be greater than 0
- `negatives_multiplier` must be greater than 0

#### Error Messages

| Condition | Error |
|-----------|-------|
| Field missing | `ConfigError: Missing required field: samples` |
| `positives <= 0` | `ConfigError: positives must be positive` |
| `negatives_multiplier <= 0` | `ConfigError: negatives_multiplier must be positive` |

#### Examples

```yaml
# Generate 1000 positives and 5000 negatives (1000 × 5)
samples:
  positives: 1000
  negatives_multiplier: 5

# Generate 500 positives and 2500 negatives (500 × 5)
samples:
  positives: 500
  negatives_multiplier: 5

# Generate 2000 positives and 2000 negatives (2000 × 1)
samples:
  positives: 2000
  negatives_multiplier: 1
```

---

### `negatives`

**Type:** `NegativeGenerationConfig` (nested object)
**Required:** No

Controls which negative generators run and how the requested negative total is divided between them.

#### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `custom_phrases` | `list[string]` | No | `null` | Explicit negative phrases to seed into the dataset before generated negatives |
| `confusion.enabled` | `bool` | No | `true` | Enable phonetic confusion negatives |
| `confusion.weight` | `float` | No | `0.6` | Relative share of negatives assigned to confusion generation |
| `confusion.min_similarity` | `float` | No | `0.6` | Similarity threshold passed to `generate_confusions()` |
| `synthetic.enabled` | `bool` | No | `true` | Enable synthetic phrase negatives |
| `synthetic.weight` | `float` | No | `0.4` | Relative share of negatives assigned to synthetic generation |
| `synthetic.strategy` | `string` | No | `random` | One of `random`, `sentence`, `topic` |
| `synthetic.min_word_count` | `int` | No | `2` | Minimum words per synthetic phrase |
| `synthetic.max_word_count` | `int` | No | `4` | Maximum words per synthetic phrase |
| `synthetic.topics` | `list[string]` | No | `null` | Optional topic allowlist for `topic` strategy |
| `synthetic.word_list` | `list[string]` | No | `null` | Optional custom vocabulary for synthetic phrases |

#### Validation Rules

- At least one negative source must be enabled
- Enabled negative sources must have a positive total weight
- `custom_phrases` cannot be empty when provided
- `custom_phrases` cannot contain blank values
- `confusion.weight` and `synthetic.weight` must be non-negative
- `confusion.min_similarity` must be in range `[0, 1]`
- `synthetic.strategy` must be one of `random`, `sentence`, `topic`
- `synthetic.min_word_count` must be at least 1
- `synthetic.max_word_count` must be greater than or equal to `synthetic.min_word_count`

#### Examples

```yaml
# Default behavior (same as previous hardcoded pipeline)
negatives:
  custom_phrases: [archer, richard]
  confusion:
    enabled: true
    weight: 0.6
    min_similarity: 0.6
  synthetic:
    enabled: true
    weight: 0.4
    strategy: random
    min_word_count: 2
    max_word_count: 4

# Confusion-heavy run
negatives:
  custom_phrases: [archer, archer assistant]
  confusion:
    enabled: true
    weight: 0.85
    min_similarity: 0.75
  synthetic:
    enabled: true
    weight: 0.15
    strategy: sentence

# Synthetic-only run with custom vocabulary
negatives:
  confusion:
    enabled: false
    weight: 0.0
  synthetic:
    enabled: true
    weight: 1.0
    strategy: topic
    topics: [technology, weather]
    word_list: [status, report, lights, timer, weather, kitchen]

# Explicitly include known false-trigger phrases like similar names
negatives:
  custom_phrases: [archer, rj, hey archer]
  confusion:
    enabled: true
    weight: 0.7
  synthetic:
    enabled: true
    weight: 0.3
```

---

### `tts`

**Type:** `TTSConfig` (nested object)
**Required:** Yes

Configures one or more text-to-speech providers for generating samples.

#### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `providers` | `list[TTSProviderConfig]` | Yes | — | List of provider configurations |

#### Validation Rules

- `providers` cannot be empty
- Each provider `backend` cannot be empty
- Each provider `voices` list cannot be empty
- Each provider `speed` must be in range `(0, 3]` (exclusive of 0, inclusive of 3)

#### Valid Backend Values

| Backend | Description | Installation |
|---------|-------------|--------------|
| `kokoro` | Kokoro TTS engine (CPU) | `uv sync --extra kokoro` |
| `kokoro` | Kokoro TTS engine (CUDA) | `uv sync --extra kokoro-cuda` |
| `kokoro` | Kokoro TTS engine (OpenVINO) | `uv sync --extra kokoro-openvino` |
| `piper` | Piper TTS engine (CPU) | `uv sync --extra piper` |
| `piper` | Piper TTS engine (CUDA) | `uv sync --extra piper-cuda` |

#### Error Messages

| Condition | Error |
|-----------|-------|
| Field missing | `ConfigError: Missing required field: tts` |
| `providers` missing | `ConfigError: Missing required field: tts.providers` |
| `providers` empty | `ConfigError: providers cannot be empty` |
| Provider `backend` empty | `ConfigError: backend cannot be empty` |
| Provider `voices` empty | `ConfigError: voices cannot be empty` |
| Provider `speed <= 0` or `speed > 3` | `ConfigError: speed must be between 0 and 3` |

#### Examples

```yaml
# Single-provider config
tts:
  providers:
    - backend: kokoro
      voices:
        - af_bella
        - af_nicole
        - am_adam
      speed: 1.0

# Multi-provider config
tts:
  providers:
    - backend: kokoro
      voices:
        - af_sarah
      speed: 0.8
    - backend: piper
      voices:
        - en_US-lessac-medium
      speed: 1.5
```

---

### `augmentation`

**Type:** `AugmentationConfig` (nested object)
**Required:** Yes

Configures audio augmentation parameters for increasing dataset diversity.

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `noise_snr` | `list[float]` | Yes | Signal-to-noise ratio range for noise injection (dB) |
| `reverb_probability` | `float` | Yes | Probability of applying reverb effect |
| `gain_range` | `list[float]` | Yes | Gain adjustment range (dB) |

#### Validation Rules

- `noise_snr` must have exactly 2 values
- `noise_snr[0]` must be less than or equal to `noise_snr[1]`
- `reverb_probability` must be in range `[0, 1]` (inclusive)
- `gain_range` must have exactly 2 values
- `gain_range[0]` must be less than or equal to `gain_range[1]`

#### Error Messages

| Condition | Error |
|-----------|-------|
| Field missing | `ConfigError: Missing required field: augmentation` |
| `noise_snr` not 2 values | `ConfigError: noise_snr must have exactly 2 values` |
| `noise_snr[0] > noise_snr[1]` | `ConfigError: noise_snr[0] must be <= noise_snr[1]` |
| `reverb_probability < 0` or `> 1` | `ConfigError: reverb_probability must be between 0 and 1` |
| `gain_range` not 2 values | `ConfigError: gain_range must have exactly 2 values` |
| `gain_range[0] > gain_range[1]` | `ConfigError: gain_range[0] must be <= gain_range[1]` |

#### Examples

```yaml
# Moderate augmentation
augmentation:
  noise_snr: [-10, 10]      # SNR from -10dB to 10dB
  reverb_probability: 0.5    # 50% chance of reverb
  gain_range: [-45, 0]       # Gain from -45dB to 0dB

# Heavy augmentation (more noise, always reverb)
augmentation:
  noise_snr: [-20, 5]
  reverb_probability: 1.0
  gain_range: [-50, -10]

# Light augmentation (minimal noise, rare reverb)
augmentation:
  noise_snr: [5, 20]
  reverb_probability: 0.1
  gain_range: [-30, -5]

# No reverb
augmentation:
  noise_snr: [-5, 15]
  reverb_probability: 0.0
  gain_range: [-45, 0]
```

---

### `output`

**Type:** `OutputConfig` (nested object)
**Required:** Yes

Configures output directory and export formats.

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | `string` | Yes | Output directory path for generated dataset |
| `format` | `list[string]` | Yes | List of export formats |

#### Validation Rules

- `path` cannot be empty
- `format` cannot be empty
- Each format must be one of the supported values

#### Valid Format Values

| Format | Description |
|--------|-------------|
| `microwakeword` | Export for microWakeWord trainer |
| `openwakeword` | Export for openWakeWord trainer |

#### Error Messages

| Condition | Error |
|-----------|-------|
| Field missing | `ConfigError: Missing required field: output` |
| `path` empty | `ConfigError: path cannot be empty` |
| `format` empty | `ConfigError: format cannot be empty` |
| Invalid format | `ConfigError: output.format contains invalid format: {fmt}` |

#### Examples

```yaml
# Single format
output:
  path: ./output/dataset
  format: [microwakeword]

# Multiple formats
output:
  path: ./output/dataset
  format: [microwakeword, openwakeword]

# Absolute path
output:
  path: /home/user/wakeword_datasets/hey_vera
  format: [openwakeword]

# Relative path
output:
  path: ../datasets/hey_assistant
  format: [microwakeword]
```

---

## Complete Example

Production-ready configuration for training a "Hey Assistant" wake word model:

```yaml
# Wake Word Configuration
# Target phrase: "Hey Assistant"
# Dataset: 1000 positives, 5000 negatives

wake_word: "Hey Assistant"

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
    - backend: piper
      voices:
        - en_US-lessac-medium
      speed: 1.0

augmentation:
  noise_snr: [-5, 15]        # Moderate noise variation
  reverb_probability: 0.3     # 30% of samples get reverb
  gain_range: [-45, 0]        # Full gain range

output:
  path: ./output/hey_assistant
  format: [microwakeword, openwakeword]
```

---

## Common Errors

### Missing Required Field

**Error:**
```
ConfigError: Missing required field: wake_word
```

**Cause:** A required top-level field is missing from the YAML file.

**Solution:** Add the missing field. All top-level fields (`wake_word`, `samples`, `tts`, `augmentation`, `output`) are required.

---

### Empty Wake Word

**Error:**
```
ConfigError: wake_word cannot be empty
```

**Cause:** `wake_word` field is present but empty.

**Solution:** Provide a non-empty wake word phrase.

```yaml
# Wrong
wake_word: ""

# Correct
wake_word: "hey vera"
```

---

### Invalid Positive Count

**Error:**
```
ConfigError: positives must be positive
```

**Cause:** `samples.positives` is zero or negative.

**Solution:** Use a positive integer.

```yaml
# Wrong
samples:
  positives: 0
  negatives_multiplier: 5

# Correct
samples:
  positives: 100
  negatives_multiplier: 5
```

---

### Empty Voices List

**Error:**
```
ConfigError: voices cannot be empty
```

**Cause:** A provider in `tts.providers` has an empty `voices` list.

**Solution:** Provide at least one voice identifier.

```yaml
# Wrong
tts:
  providers:
    - backend: kokoro
      voices: []

# Correct
tts:
  providers:
    - backend: kokoro
      voices:
        - af_bella
```

---

### Invalid Speed Range

**Error:**
```
ConfigError: speed must be between 0 and 3
```

**Cause:** A provider speed in `tts.providers` is outside the valid range `(0, 3]`.

**Solution:** Use a speed value between 0 (exclusive) and 3 (inclusive).

```yaml
# Wrong
tts:
  providers:
    - backend: kokoro
      voices: [af_bella]
      speed: 0

# Wrong
tts:
  providers:
    - backend: kokoro
      voices: [af_bella]
      speed: 5

# Correct
tts:
  providers:
    - backend: kokoro
      voices: [af_bella]
      speed: 1.0
```

---

### Invalid Noise SNR Range

**Error:**
```
ConfigError: noise_snr must have exactly 2 values
```

**Cause:** `augmentation.noise_snr` does not have exactly 2 elements.

**Solution:** Provide exactly 2 values representing the min and max SNR.

```yaml
# Wrong
augmentation:
  noise_snr: [-10]           # Only 1 value
  reverb_probability: 0.5
  gain_range: [-45, 0]

# Wrong
augmentation:
  noise_snr: [-10, 0, 10]    # 3 values
  reverb_probability: 0.5
  gain_range: [-45, 0]

# Correct
augmentation:
  noise_snr: [-10, 10]       # Exactly 2 values
  reverb_probability: 0.5
  gain_range: [-45, 0]
```

---

### Inverted Range Values

**Error:**
```
ConfigError: noise_snr[0] must be <= noise_snr[1]
```

**Cause:** The first value in a range is greater than the second.

**Solution:** Ensure range values are ordered from smallest to largest.

```yaml
# Wrong
augmentation:
  noise_snr: [10, -10]       # First > second
  reverb_probability: 0.5
  gain_range: [-45, 0]

# Correct
augmentation:
  noise_snr: [-10, 10]       # First <= second
  reverb_probability: 0.5
  gain_range: [-45, 0]
```

---

### Invalid Reverb Probability

**Error:**
```
ConfigError: reverb_probability must be between 0 and 1
```

**Cause:** `augmentation.reverb_probability` is outside `[0, 1]`.

**Solution:** Use a probability value between 0 and 1 (inclusive).

```yaml
# Wrong
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 1.5     # > 1
  gain_range: [-45, 0]

# Wrong
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: -0.1    # < 0
  gain_range: [-45, 0]

# Correct
augmentation:
  noise_snr: [-10, 10]
  reverb_probability: 0.5     # 0 <= value <= 1
  gain_range: [-45, 0]
```

---

### Invalid Output Format

**Error:**
```
ConfigError: output.format contains invalid format: invalid_format
```

**Cause:** `output.format` contains a value not in the supported formats list.

**Solution:** Use only supported formats: `microwakeword` or `openwakeword`.

```yaml
# Wrong
output:
  path: ./output
  format: [invalid_format]

# Correct
output:
  path: ./output
  format: [microwakeword]

# Correct (multiple formats)
output:
  path: ./output
  format: [microwakeword, openwakeword]
```

---

### Empty Output Path

**Error:**
```
ConfigError: path cannot be empty
```

**Cause:** `output.path` is an empty string.

**Solution:** Provide a valid directory path.

```yaml
# Wrong
output:
  path: ""
  format: [microwakeword]

# Correct
output:
  path: ./output
  format: [microwakeword]
```

---

### YAML Parse Error

**Error:**
```
ConfigError: Failed to parse YAML: ...
```

**Cause:** The configuration file contains invalid YAML syntax.

**Solution:** Validate YAML syntax. Common issues:
- Missing quotes around strings with special characters
- Incorrect indentation
- Missing colons after keys

```yaml
# Wrong (missing colon)
wake_word "Hey Assistant"

# Correct
wake_word: "Hey Assistant"

# Wrong (bad indentation)
samples:
positives: 100
  negatives_multiplier: 5

# Correct
samples:
  positives: 100
  negatives_multiplier: 5
```

---

### Empty Config File

**Error:**
```
ConfigError: Config file is empty
```

**Cause:** The configuration file exists but contains no content.

**Solution:** Add configuration content to the file.

---

### Config File Not Found

**Error:**
```
ConfigError: Config file not found: /path/to/config.yaml
```

**Cause:** The specified configuration file does not exist.

**Solution:** Verify the file path is correct and the file exists.

---

## Validation Flow

Configuration validation follows this order:

1. **File existence** — Check if config file exists
2. **YAML parsing** — Parse YAML syntax
3. **Top-level fields** — Verify all required sections present
4. **Field validation** — Validate each field's value
5. **Cross-field validation** — Check range ordering and constraints

Validation happens in `load_config()` and each dataclass's `__post_init__()` method.

---

## Type Reference

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text value | `"hey vera"` |
| `int` | Integer number | `1000` |
| `float` | Floating-point number | `0.5` |
| `list[string]` | List of strings | `[microwakeword, openwakeword]` |
| `list[float]` | List of floats | `[-10.0, 10.0]` |

---

## See Also

- [README.md](../README.md) — Project overview and quick start
- [examples/](../examples/) — Sample configuration files
- [AGENTS.md](../AGENTS.md) — Project structure and conventions
