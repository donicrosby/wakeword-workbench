# MODULE KNOWLEDGE BASE — wakeword_workbench

**Generated:** 2026-03-30
**Parent:** ../AGENTS.md

## OVERVIEW
Core wake word workbench package with 8 submodules covering TTS, augmentation, dataset generation, evaluation, export, and mining.

## MODULE BOOTSTRAP (DO THIS FIRST)

Before touching this module, bootstrap and smoke-check from repo root:

```bash
uv sync --group dev
source .venv/bin/activate
uv run pre-commit install
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If the selected config backend is unavailable, install the matching extra (`kokoro` or `piper`) and re-run validation.

## STRUCTURE

```
wakeword_workbench/
├── __init__.py            # Version info only
├── cli.py                 # Typer CLI commands
├── config.py              # YAML config dataclasses
├── logging_config.py      # structlog setup
├── augment/               # Audio augmentation (6 files)
│   ├── pipeline.py        # Compose + Transform Protocol
│   ├── noise.py           # AddNoise, AddColoredNoise
│   ├── reverb.py          # AddReverb
│   ├── gain.py            # AdjustGain, GainTransition, clipping
│   ├── padding.py         # FixedSizeClip, TrimSilence
│   └── audio_loader.py    # WAV file loading
├── dataset/               # Dataset generation (6 files)
│   ├── generator.py       # Main dataset builder
│   ├── positive_generator.py
│   ├── phrase_variants.py
│   ├── metadata.py        # Manifest management
│   ├── merger.py          # Positives + negatives
│   └── splitter.py        # Train/val/test splits
├── eval/                  # Evaluation metrics (5 files)
│   ├── far.py             # False Acceptance Rate
│   ├── frr.py             # False Rejection Rate
│   ├── threshold_sweep.py
│   ├── roc.py             # ROC curve data
│   └── report.py          # EvaluationReport
├── export/                # Model exporters (3 files)
│   ├── openwakeword.py
│   ├── microwakeword.py
│   └── validator.py
├── mining/                # Hard negative mining (3 files)
│   ├── extractor.py       # Clip extraction
│   ├── long_audio.py      # Sliding window processing
│   └── merge_back.py      # Add mined samples to dataset
├── negatives/             # Negative generation (2 files)
│   ├── phrase_generator.py
│   └── synthetic_generator.py
└── tts/                   # Text-to-speech (5 files)
    ├── base.py            # TTSBackend ABC, TTSResult
    ├── registry.py        # Backend discovery
    ├── cache.py           # TTSCache with SHA256 keys
    ├── kokoro_backend.py
    └── piper_backend.py
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add new TTS engine | `tts/base.py` + `tts/registry.py` | Extend `TTSBackend`, call `register_backend()` |
| Add augment transform | `augment/pipeline.py` + new module | Implement `Transform` Protocol, add to `_auto_register()` |
| Update config fields | `config.py` | Add dataclass field + `__post_init__` validation |
| Tune negative generation | `config.py` + `dataset/generator.py` | Optional `negatives` section controls confusion/synthetic mix |
| Change logging format | `logging_config.py` | Modify `_console_renderer` or `_json_renderer` |
| Add CLI subcommand | `cli.py` | Add `@app.command()` function |

## CONVENTIONS (MODULE-SPECIFIC)

**Registry Pattern:**
```python
# tts/registry.py or augment/pipeline.py
_BACKENDS: dict[str, type[TTSBackend]] = {}

def register_backend(name: str, cls: type[TTSBackend]) -> None:
    _BACKENDS[name] = cls
```

**Protocol Definition:**
```python
@runtime_checkable
class Transform(Protocol):
    def apply(self, audio: NDArray[np.float32], sr: int) -> NDArray[np.float32]: ...
```

**Exception Hierarchy:**
```python
class ModuleError(Exception): pass
class SpecificError(ModuleError): pass
```

**Dataclass with Validation:**
```python
@dataclass
class SomeConfig:
    field: int
    def __post_init__(self) -> None:
        if self.field <= 0:
            raise ConfigError("field must be positive")
```

## ANTI-PATTERNS (THIS MODULE)

- **Don't mutate input arrays** — Audio transforms must return new arrays
- **Don't skip `__post_init__`** — Always validate dataclass fields
- **Don't import at module level** — Use lazy imports in `_auto_register()` to avoid circular deps
- **Don't forget TYPE_CHECKING** — Use for imports only needed for type hints
- **Don't raise generic Exception** — Always use domain-specific exceptions

## PATTERNS SUMMARY

| Pattern | Location | Purpose |
|---------|----------|---------|
| Registry | `tts/registry.py`, `augment/pipeline.py` | Plugin architecture for backends/transforms |
| Protocol | `augment/pipeline.py` | Duck typing without inheritance |
| ABC | `tts/base.py` | Enforce interface for TTS backends |
| Dataclass + validation | `config.py` | Type-safe config with runtime checks |
| Lazy discovery | `tts/registry.py` | Auto-detect available backends |
| Custom exceptions | Every module | Domain-specific error handling |
