# PROJECT KNOWLEDGE BASE — WakeWord Workbench

**Generated:** 2026-03-30  
**Commit:** be48811  
**Branch:** opencode/playful-cactus  
**Version:** 0.1.0-alpha

## OVERVIEW
Python 3.11+ toolkit for training and evaluating micro wake word detection models. Emphasizes data quality over model architecture.

## AGENT BOOTSTRAP (DO THIS FIRST)

Use this exact startup sequence before making changes so you do not get stuck on environment or command issues.

```bash
# 1) Install dependencies for development work
uv sync --group dev

# 2) Optional: install TTS backends used by config/examples
uv sync --extra kokoro
uv sync --extra piper

# 3) Smoke-check CLI and config pathing
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If validation fails because a backend is missing, install the required backend extra and rerun validation.

## STRUCTURE

```
.
├── src/wakeword_workbench/  # Main source (41 Python files)
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
├── tests/                  # 26 pytest files + conftest.py
├── specs/                  # Implementation specification
└── pyproject.toml          # Project config
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add CLI command | `src/wakeword_workbench/cli.py` | Uses Typer, add `@app.command()` |
| Update config schema | `src/wakeword_workbench/config.py` | Dataclass + `__post_init__` validation |
| Add TTS backend | `src/wakeword_workbench/tts/` | Extend `TTSBackend` ABC, register via `register_backend()` |
| Add audio transform | `src/wakeword_workbench/augment/` | Implement `Transform` Protocol, auto-registered |
| Update test fixtures | `tests/conftest.py` | Shared fixtures: `mock_audio`, `sample_config`, `valid_config_file` |
| Check spec | `specs/wakeword_workbench_opencode_spec.md` | Phase-based implementation plan |

## COMMANDS

```bash
# Development
uv sync                    # Install dependencies
uv sync --group dev        # With dev dependencies
uv sync --extra kokoro     # With Kokoro TTS
uv sync --extra piper      # With Piper TTS

# Testing
uv run pytest              # Run all tests
uv run pytest --cov=wakeword_workbench  # With coverage

# CLI
uv run wakeword-workbench --help
uv run wakeword-workbench validate config.yaml
uv run wakeword-workbench run config.yaml
```

## CONVENTIONS

**Code Style:**
- `from __future__ import annotations` — Required in every file
- Type hints — Required (mypy `disallow_untyped_defs = true`)
- Line length — 100 chars (ruff)
- Docstrings — Google-style

**Patterns:**
- **Registry pattern** — TTS backends and augment transforms use module-level registries
- **Protocol-based** — `Transform` Protocol with `@runtime_checkable` for duck typing
- **Dataclass validation** — Config classes use `__post_init__` for validation
- **Custom exceptions** — All modules define domain-specific exceptions (e.g., `ConfigError`, `TTSError`)

**Testing:**
- pytest with fixtures in `conftest.py`
- Class-based test organization: `class TestComponent:`
- CLI tests use `typer.testing.CliRunner`
- Mock external services with `unittest.mock`

## ANTI-PATTERNS (THIS PROJECT)

- **Don't use** `requirements.txt` — Use `pyproject.toml` + uv
- **Don't hardcode** paths — Use `pathlib.Path` everywhere
- **Don't mutate** input audio arrays — Transforms must return new arrays
- **Don't skip** `__post_init__` validation — All config dataclasses validate
- **Don't use** `print()` — Use `structlog` via `get_logger()`

## UNIQUE STYLES

**Structured Logging:**
```python
from wakeword_workbench.logging_config import get_logger
log = get_logger()
log.info("processing", count=100, file=str(path))
```

**Config Loading:**
```python
from wakeword_workbench.config import load_config
config = load_config("config.yaml")  # Returns validated Config dataclass
```

**TTS Backend:**
```python
from wakeword_workbench.tts.registry import get_backend
backend = get_backend("kokoro")  # Auto-discovers available backends
```

**Transform Registration:**
Transforms in `augment/` are auto-registered on import via `_auto_register()` in `pipeline.py`.

## NOTES

- **No CI/CD** — No `.github/workflows/` directory yet
- **Early stage** — Dataset pipeline exists and is evolving rapidly; verify behavior with `validate` + `run` on sample configs
- **Key principle** — "Data quality and negative coverage matter more than model architecture" (from spec)
- **Cache location** — `~/.cache/wakeword_workbench/tts/` for TTS synthesis
- **Exit codes** — 0=success, 1=error, 2=config error

## DOCUMENTATION SYNC TASK (ANTI-DRIFT)

When any setup command, CLI flag/command, config schema, or workflow step changes, update all user-facing docs in the same change:

- `README.md`
- `AGENTS.md`
- `src/wakeword_workbench/AGENTS.md`
- `docs/*.md`
- `docs/training/*.md`
- `examples/training/README.md`

This sync task is mandatory to keep agent instructions current and prevent onboarding drift.
