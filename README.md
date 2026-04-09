# Wake Word Workbench

A Python toolkit for training and evaluating micro wake word detection models. Emphasizes data quality and negative coverage over model architecture.

## Agent Setup (Do This First)

Run this sequence before using any workflow in this repo:

```bash
uv sync --group dev
source .venv/bin/activate
uv run pre-commit install
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If your config uses Kokoro or Piper and validation fails, install the missing backend:

```bash
uv sync --extra kokoro
uv sync --extra piper

# accelerator-aware variants
uv sync --extra kokoro-cuda
uv sync --extra kokoro-openvino
uv sync --extra piper-cuda
```

Agent-specific conventions and anti-drift rules live in [AGENTS.md](AGENTS.md).

## Quick Links

| Resource | Description |
|----------|-------------|
| [AGENTS.md](AGENTS.md) | Project structure and conventions |
| [specs/](specs/) | Implementation specifications |
| [examples/](examples/) | Sample configuration files |

**New to wake words?** Start with the project philosophy below, then explore the examples.

## Documentation

| Document | Description |
|----------|-------------|
| [BACKGROUND.md](docs/BACKGROUND.md) | Foundational concepts and background on wake word detection |
| [TUTORIAL.md](docs/TUTORIAL.md) | Step-by-step guide for common workflows |
| [CLI_GUIDE.md](docs/CLI_GUIDE.md) | Detailed CLI commands and usage |
| [CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | Complete configuration schema reference |
| [API_USAGE.md](docs/API_USAGE.md) | Using the toolkit as a Python library |

> **New to the project?** Start with [BACKGROUND.md](docs/BACKGROUND.md) for essential concepts before diving into other guides.

## Features

- **TTS Integration** — Generate positive samples with Kokoro or Piper text-to-speech engines
- **Audio Augmentation** — Apply noise, reverb, gain variations, and silence padding
- **Negative Generation** — Create confusion phrases and synthetic negatives
- **Hard Negative Mining** — Extract false positives from long audio recordings
- **Dataset Management** — JSONL manifests with train/val/test splits
- **Model Export** — Export datasets for microWakeWord and openWakeWord trainers
- **Evaluation Metrics** — Calculate FAR/FRR, generate ROC curves, find optimal thresholds

## Philosophy

> **Data quality and negative coverage matter more than model architecture.**

This toolkit focuses on generating high-quality training data with comprehensive negative coverage. A well-curated dataset with diverse negatives outperforms architectural tweaks.

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd wakeword-workbench

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Optional Dependencies

For TTS support:

```bash
# Kokoro TTS backend
uv sync --extra kokoro

# Kokoro with CUDA
uv sync --extra kokoro-cuda

# Kokoro with OpenVINO
uv sync --extra kokoro-openvino

# Piper TTS backend
uv sync --extra piper

# Piper with CUDA
uv sync --extra piper-cuda

# All TTS backends
uv sync --all-extras
```

### Development Setup

```bash
# Install with dev dependencies
uv sync --group dev

# Install git hooks
uv run pre-commit install

# Run hooks across the repo
uv run pre-commit run --all-files

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=wakeword_workbench --cov-report=term-missing

# Type checking
uv run mypy src/wakeword_workbench

# Linting
uv run ruff check src/
```

## Quick Start

### 1. Create a Configuration File

Create `config.yaml`:

```yaml
# Wake Word Dataset Configuration
wake_word: "hey vera"

# Optional: explicit list of positive phrases to synthesize.
# If omitted, the pipeline uses only the exact wake_word above.
# wake_word_variants:
#   - "hey vera"
#   - "hey, vera"

samples:
  positives: 100
  negatives_multiplier: 5  # Generate 500 negative samples

negatives:
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

tts:
  providers:
    - backend: "kokoro"
      voices:
        - "af_sarah"
        - "am_adam"
      speed: 1.0
      acceleration: "cuda"  # cpu, cuda, or openvino

    - backend: "piper"
      voices:
        - "en_US-amy-low"
      acceleration: "cuda"  # Piper supports cpu/cuda
      model_path: "./models/en_US-amy-low.onnx"  # optional

augmentation:
  noise_snr: [-10, 10]      # dB range for noise injection
  reverb_probability: 0.5    # Probability of applying reverb
  gain_range: [-45, 0]       # dB range for gain adjustment

output:
  path: "./output/dataset"
  format: ["microwakeword"]  # or ["openwakeword"]
```

### 2. Validate Configuration

```bash
uv run wakeword-workbench validate config.yaml
```

### 3. Run the Pipeline

```bash
uv run wakeword-workbench run config.yaml
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `run <config>` | Run the full training pipeline from a config file |
| `validate <config>` | Validate a config file without running the pipeline |
| `mine` | Mine hard negatives from long audio recordings |
| `merge` | Merge mined hard negatives into training dataset manifests |
| `cache-clear` | Clear the TTS synthesis cache |

### Global Options

| Option | Description |
|--------|-------------|
| `--version`, `-V` | Show version and exit |
| `--verbose`, `-v` | Enable DEBUG logging |
| `--quiet`, `-q` | Only show ERROR logs |

### Mining Hard Negatives

Extract false positives from long audio recordings:

```bash
uv run wakeword-workbench mine \
  --model path/to/model.onnx \
  --audio "recordings/*.wav" \
  --output ./mined_negatives \
  --threshold 0.7
```

### Merging Negatives

Add mined negatives to your training dataset:

```bash
uv run wakeword-workbench merge \
  --source ./mined_negatives/manifest.jsonl \
  --target ./dataset/train.jsonl \
  --backup
```

## Project Structure

```
wakeword-workbench/
├── src/wakeword_workbench/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # YAML config schema
│   ├── logging_config.py   # structlog setup
│   ├── augment/            # Audio augmentation pipeline
│   ├── dataset/            # Dataset generation/management
│   ├── eval/               # FAR/FRR metrics, ROC, thresholds
│   ├── export/             # microWakeWord/openWakeWord exporters
│   ├── mining/             # Hard negative mining
│   ├── negatives/          # Negative sample generation
│   └── tts/                # TTS backends (Kokoro, Piper)
├── tests/                  # pytest test suite
├── examples/               # Sample configuration files
├── specs/                  # Implementation specifications
└── pyproject.toml          # Project configuration
```

## Configuration

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `wake_word` | string | The wake word phrase to train for |
| `samples.positives` | int | Number of positive samples to generate |
| `samples.negatives_multiplier` | int | Multiplier for negative samples |
| `tts.providers` | list | Non-empty list of TTS provider configs |
| `tts.providers[].backend` | string | TTS engine: `"kokoro"` or `"piper"` |
| `tts.providers[].voices` | list | Non-empty list of voice identifiers |
| `tts.providers[].acceleration` | string | Runtime acceleration: `cpu`, `cuda`, or `openvino` |
| `augmentation.noise_snr` | [float, float] | SNR range for noise injection (dB) |
| `augmentation.reverb_probability` | float | Probability of reverb (0.0-1.0) |
| `augmentation.gain_range` | [float, float] | Gain range (dB) |
| `output.path` | string | Output directory path |
| `output.format` | list | Export formats: `["microwakeword"]` or `["openwakeword"]` |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wake_word_variants` | list[string] | — | Explicit positive phrases to synthesize; if omitted, only `wake_word` is used |
| `tts.providers[].speed` | float | 1.0 | Speech speed multiplier (0.0-3.0) |
| `tts.providers[].acceleration` | string | `cpu` | Runtime acceleration mode |
| `tts.providers[].device` | string | — | Optional device selector such as `GPU` or `cuda:0` |
| `tts.providers[].model_path` | string | — | Optional explicit model path for compatible backends |
| `negatives.confusion.enabled` | bool | true | Enable phonetic confusion negatives |
| `negatives.custom_phrases` | list[string] | — | Explicit negative phrases to always include first |
| `negatives.confusion.weight` | float | 0.6 | Relative share of generated negatives |
| `negatives.confusion.min_similarity` | float | 0.6 | Phonetic similarity threshold |
| `negatives.synthetic.enabled` | bool | true | Enable synthetic phrase negatives |
| `negatives.synthetic.weight` | float | 0.4 | Relative share of generated negatives |
| `negatives.synthetic.strategy` | string | random | One of `random`, `sentence`, `topic` |

See `examples/` directory for sample configurations.

## Python API

Use the workbench as a library:

```python
from pathlib import Path
from wakeword_workbench.config import load_config
from wakeword_workbench.tts.registry import get_backend

# Load configuration
config = load_config("config.yaml")

# Get first configured provider
provider = config.tts.providers[0]
tts = get_backend(provider.backend)
tts.set_voice(provider.voices[0])

# Generate samples
result = tts.synthesize(config.wake_word)

print(f"Samples: {len(result.audio)}")
print(f"Duration: {result.duration:.2f}s")
```

## Key Concepts

### Hard Negatives

False positives extracted from real-world audio where the model incorrectly predicts the wake word. These are the most valuable training samples for improving model accuracy.

### Augmentation

Audio transformations applied to increase dataset diversity:
- **Noise injection** — Add background noise at varying SNR levels
- **Reverb** — Simulate room acoustics
- **Gain variation** — Adjust volume levels
- **Silence padding** — Add leading/trailing silence

### Metrics

- **FAR (False Acceptance Rate)** — Percentage of negatives incorrectly classified as wake word
- **FRR (False Rejection Rate)** — Percentage of positives incorrectly rejected
- **ROC Curve** — Trade-off between FAR and FRR across thresholds

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=wakeword_workbench --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_config.py

# Run with verbose output
uv run pytest -v
```

### Code Style

- Line length: 100 characters
- Type hints required (mypy `disallow_untyped_defs = true`)
- Docstrings: Google-style
- `from __future__ import annotations` required in every file

### Logging

The project uses `structlog` for structured logging:

```python
from wakeword_workbench.logging_config import get_logger

log = get_logger()
log.info("processing", count=100, file=str(path))
```

## Troubleshooting

### TTS Backend Not Found

**Error:** `TTSError: Backend 'kokoro' not available`

**Solution:** Install the TTS backend:
```bash
uv sync --extra kokoro
# or
uv sync --extra piper

# accelerator-aware variants
uv sync --extra kokoro-cuda
uv sync --extra kokoro-openvino
uv sync --extra piper-cuda
```

### Config Validation Failed

**Error:** `ConfigError: Missing required field: wake_word`

**Solution:** Ensure your YAML config has all required fields. See the Configuration section or `examples/basic_config.yaml`.

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'wakeword_workbench'`

**Solution:** Ensure you've activated the virtual environment:
```bash
source .venv/bin/activate
```

### Cache Location

TTS synthesis cache is stored at `~/.cache/wakeword_workbench/tts/`. Clear it with:

```bash
uv run wakeword-workbench cache-clear
```

## License

MIT License - see [LICENSE](LICENSE) for details.
