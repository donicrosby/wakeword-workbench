# Wake Word Workbench

A training workbench for micro wake word detection models.

## Features

- Audio preprocessing and augmentation
- Wake word dataset management
- Model training utilities
- TTS integration (Kokoro, Piper)

## Installation

### Prerequisites

- Python 3.11+
- uv package manager

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
# Kokoro TTS
uv sync --extra kokoro

# Piper TTS
uv sync --extra piper

# All extras
uv sync --all-extras
```

### Development Setup

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=wakeword_workbench
```

## Quick Start

```bash
# Show help
wakeword-workbench --help

# Or via uv run
uv run wakeword-workbench --help
```

## Project Structure

```
wakeword-workbench/
├── src/wakeword_workbench/  # Source code
├── tests/                   # Test files
├── pyproject.toml          # Project configuration
└── README.md
```

## License

MIT
