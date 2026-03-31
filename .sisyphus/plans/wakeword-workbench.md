# Wake Word Workbench — Implementation Plan

## TL;DR

> **Build a modular Python CLI tool for generating wake word datasets and evaluating models for microWakeWord and openWakeWord.**
>
> **Deliverables**:
> - CLI application with YAML config support
> - Pluggable TTS backends (Kokoro primary, Piper secondary)
> - Positive dataset generator (1000+ samples)
> - Negative dataset generator (5x positives, confusion phrases)
> - Audio augmentation pipeline (noise, reverb, gain, clipping)
> - Dataset builder with leak-proof train/val/test splits
> - Export modules for both microWakeWord and openWakeWord
> - Evaluation pipeline with FAR/FRR metrics
> - Hard negative mining for iterative improvement
>
> **Estimated Effort**: Large (9 phases, 35+ tasks)
> **Parallel Execution**: YES — 7 waves
> **Critical Path**: Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 6 → Phase 7 → Phase 8

---

## Context

### Original Request
Build a modular, reproducible pipeline for generating datasets and evaluating wake word models (microWakeWord, openWakeWord).

### Interview Summary
**Decisions Made**:
- Package manager: **uv** (modern, fast Python tooling)
- TTS backends: **Optional installation** with graceful fallback
- Testing strategy: **pytest from start** (TDD approach)
- Phase approach: All 9 phases in one comprehensive plan

### Research Findings

**TTS Backends**:
- **Kokoro (pykokoro)**: Apache 2.0, 24000 Hz output, 54+ voices, voice blending, phoneme support
- **Piper (piper-tts)**: GPL-3.0, 22050 Hz output, 50+ voices, streaming API
- **Abstraction**: Custom TTSBackend ABC with factory pattern

**Wake Word Formats**:
- **microWakeWord**: Ragged Mmap format, 16kHz/40 mel bins, truncation strategies
- **openWakeWord**: Memory-mapped numpy arrays, 16kHz/96 mel features via SpeechBrain
- **Metrics**: FAR (FP/hour), FRR (1-Recall), ROC curve with cooldown windows

**Audio Augmentation** (from production projects):
- Background noise: SNR -10 to 10 dB, p=0.75
- Reverb (RIR): p=0.5 with room-specific RT60
- Gain: -45 to 0 dB, p=1.0
- Pitch shift: ±3 semitones, p=0.25
- Library: audiomentations (CPU), torch-audiomentations (GPU batch)

### Metis Review
**Identified Gaps** (all addressed in this plan):
- Audio specs: 16kHz, 16-bit PCM, mono (added to Phase 1)
- Storage estimation formulas (added)
- Explicit guardrails per phase
- TDD acceptance criteria for every task
- Edge case handling (disk full, corrupted audio, Unicode)
- TTS caching strategy (~/.cache/wakeword_workbench/)
- Resume/interruption handling

---

## Work Objectives

### Core Objective
Build a production-ready CLI tool that can generate reproducible wake word datasets from config and evaluate model performance with FAR/FRR metrics.

### Concrete Deliverables
1. wakeword-workbench CLI with run command
2. TTS backends: kokoro, piper with optional imports
3. Dataset generators: positive, negative, augment
4. Dataset builder: merge, split (leak-proof)
5. Export modules: microwakeword, openwakeword
6. Evaluation: far, frr, roc with threshold sweep
7. Hard negative mining: extract, merge

### Definition of Done
- [ ] Full pipeline runs end-to-end from example config
- [ ] All tests pass with >80% coverage
- [ ] Generated datasets pass format validation for both targets
- [ ] Evaluation produces actionable FAR/FRR report
- [ ] Hard negatives improve model performance (demonstrable)

### Must Have
- YAML config-driven reproducibility
- Optional TTS backends (graceful degradation)
- 16kHz/16-bit/mono audio throughout
- JSONL manifest format
- Leak-proof dataset splitting
- Agent-executable QA for every task

### Must NOT Have (Guardrails)
- NO GUI (out of scope per spec)
- NO cloud deployment (out of scope per spec)
- NO real-time inference engine (out of scope per spec)
- NO model training code (dataset generation only)
- NO distributed processing (single-machine)
- NO model zoo/management
- NO audio capture from microphone

---

## Verification Strategy

### Test Decision
- **Infrastructure**: pytest with uv
- **Strategy**: TDD (tests first, then implementation)
- **Coverage target**: >80%
- **CI**: GitHub Actions on push/PR

### QA Policy
Every task MUST include agent-executed QA scenarios:
- **CLI**: Bash commands with exit code verification
- **TTS**: Audio file validation (sox, ffprobe)
- **Datasets**: JSONL validation with jq
- **Augmentation**: Audio property checks (duration, sample rate)
- **Export**: Format-specific validation scripts
- **Evaluation**: Metric calculation verification

Evidence saved to .sisyphus/evidence/task-{N}-{scenario}.{ext}

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - ALL can start immediately):
- Task 1: Project scaffolding (pyproject.toml, uv setup)
- Task 2: Directory structure (src/wakeword_workbench/)
- Task 3: Config system (YAML loader, validation)
- Task 4: CLI entry point (typer, --version, --help)
- Task 5: Test infrastructure (pytest, conftest.py)
- Task 6: Logging system (structlog, verbosity levels)

Wave 2 (TTS Backends - parallel after Wave 1):
- Task 7: TTSBackend ABC and factory
- Task 8: KokoroBackend implementation
- Task 9: PiperBackend implementation
- Task 10: TTS registry and discovery
- Task 11: TTS caching system

Wave 3 (Dataset Generation - parallel after Wave 2):
- Task 12: Phrase variant generator
- Task 13: Positive dataset generator
- Task 14: Negative phrase generator (confusions)
- Task 15: Synthetic negative generator
- Task 16: Metadata JSONL system

Wave 4 (Augmentation - parallel after Wave 3):
- Task 17: Audio loader/resampler
- Task 18: Noise augmentation (background, colored)
- Task 19: Reverb augmentation (RIR)
- Task 20: Gain/clipping augmentation
- Task 21: Silence padding/transforms
- Task 22: Augmentation pipeline composer

Wave 5 (Dataset Builder and Export - parallel after Wave 4):
- Task 23: Dataset merger
- Task 24: Leak-proof splitter
- Task 25: microWakeWord exporter
- Task 26: openWakeWord exporter
- Task 27: Export validation

Wave 6 (Evaluation - parallel after Wave 5):
- Task 28: FAR calculator
- Task 29: FRR calculator
- Task 30: Threshold sweep
- Task 31: ROC curve generator
- Task 32: Evaluation report formatter

Wave 7 (Hard Negative Mining - after Wave 6):
- Task 33: Long audio processor
- Task 34: False positive extractor
- Task 35: Dataset merge-back

Wave FINAL (Verification - after ALL):
- Task F1: Plan compliance audit (oracle)
- Task F2: Code quality review (unspecified-high)
- Task F3: End-to-end pipeline test (unspecified-high)
- Task F4: Scope fidelity check (deep)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1-6 | — | 7-11 |
| 7-11 | 1-6 | 12-16 |
| 12-16 | 7-11 | 17-22 |
| 17-22 | 12-16 | 23-27 |
| 23-27 | 17-22 | 28-32 |
| 28-32 | 23-27 | 33-35 |
| 33-35 | 28-32 | F1-F4 |

---

## TODOs

### Wave 1: Foundation (ALL can start immediately)

- [ ] **1. Project Scaffolding and UV Setup**

  **What to do**:
  - Initialize project with `uv init --lib wakeword-workbench`
  - Create `pyproject.toml` with dependencies:
    - Required: typer, pyyaml, numpy, scipy, librosa, soundfile, audiomentations, structlog
    - Optional: pykokoro, piper-tts (with warnings if missing)
    - Dev: pytest, pytest-cov, ruff, mypy
  - Set up `.gitignore` for Python/uv projects
  - Create `README.md` with installation and quickstart

  **Must NOT do**:
  - Do not pin exact versions (use `>=x,<y` syntax)
  - Do not include GUI dependencies (out of scope)
  - Do not include cloud/storage dependencies

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: File creation and dependency specification only

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 1
  - **Blocks**: Task 2-6 (depends on pyproject.toml structure)

  **Acceptance Criteria**:
  - [ ] `uv sync` completes without errors
  - [ ] `uv run python -c "import wakeword_workbench"` succeeds
  - [ ] `cat pyproject.toml | grep -q "typer"` passes

  **QA Scenarios**:
  ```
  Scenario: Project initializes correctly
    Tool: Bash
    Steps:
      1. cd /home/doni/projects/micro_wake_word_trainer/micro-wakerword-workbench && uv sync
      2. uv run python --version
    Expected Result: Python 3.11+ and no import errors
    Evidence: .sisyphus/evidence/task-1-init.log
  ```

  **Commit**: YES
  - Message: `chore: initial project setup with uv`
  - Files: `pyproject.toml`, `README.md`, `.gitignore`

- [ ] **2. Directory Structure Setup**

  **What to do**:
  - Create `src/wakeword_workbench/` package structure:
    ```
    src/wakeword_workbench/
    ├── __init__.py
    ├── cli.py
    ├── config.py
    ├── tts/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── kokoro_backend.py
    │   └── piper_backend.py
    ├── dataset/
    │   ├── __init__.py
    │   ├── generator.py
    │   └── metadata.py
    ├── augment/
    │   ├── __init__.py
    │   └── transforms.py
    ├── negatives/
    │   ├── __init__.py
    │   └── generator.py
    ├── export/
    │   ├── __init__.py
    │   ├── microwakeword.py
    │   └── openwakeword.py
    └── eval/
        ├── __init__.py
        └── metrics.py
    ```
  - Create `tests/` directory with `__init__.py` and `conftest.py`
  - Create `examples/` directory for sample configs

  **Must NOT do**:
  - Do not implement any logic yet (skeleton files only)
  - Do not add imports that would fail

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Directory and empty file creation

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 1 (with Task 1)
  - **Blocks**: All other tasks

  **Acceptance Criteria**:
  - [ ] All directories exist
  - [ ] All `__init__.py` files exist and have docstrings
  - [ ] `python -c "from wakeword_workbench import cli"` succeeds (empty import)

  **QA Scenarios**:
  ```
  Scenario: Directory structure exists
    Tool: Bash
    Steps:
      1. find src/wakeword_workbench -type d | wc -l
      2. test -f src/wakeword_workbench/__init__.py
    Expected Result: 10+ directories and key files exist
    Evidence: .sisyphus/evidence/task-2-structure.txt
  ```

  **Commit**: YES (grouped with Task 1)

- [ ] **3. Config System (YAML Loader and Validation)**

  **What to do**:
  - Implement `config.py` with:
    - `Config` dataclass with all config fields
    - `load_config(path: Path) -> Config` function
    - Schema validation using dataclasses
  - Define config structure:
    ```yaml
    wake_word: "hey assistant"
    samples:
      positives: 1000
      negatives_multiplier: 5
    tts:
      backend: kokoro  # or piper
      voices:
        - af_sarah
        - af_nicole
      speed: 1.0
    augmentation:
      noise_snr: [-10, 10]
      reverb_probability: 0.5
      gain_range: [-45, 0]
    output:
      path: ./datasets
      format: [microwakeword, openwakeword]
    ```
  - Add validation for required fields
  - Add helpful error messages for missing/invalid config

  **Must NOT do**:
  - Do not use complex validation libraries (keep it simple)
  - Do not support multiple config formats (YAML only)

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Dataclass-based config with standard YAML loader

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 1
  - **Blocks**: Tasks 12-16 (dataset generation uses config)

  **References**:
  - Pattern: Standard Python dataclasses with `__post_init__` validation
  - Library: PyYAML with `safe_load`

  **Acceptance Criteria**:
  - [ ] `load_config("examples/basic.yaml")` returns Config object
  - [ ] Missing required field raises `ConfigError` with helpful message
  - [ ] Invalid YAML raises `ConfigError` with line number

  **QA Scenarios**:
  ```
  Scenario: Config loads and validates
    Tool: Bash
    Steps:
      1. Create test config with required fields
      2. python -c "from wakeword_workbench.config import load_config; print(load_config('test.yaml').wake_word)"
    Expected Result: Config loads and returns expected value
    Evidence: .sisyphus/evidence/task-3-config.log
  
  Scenario: Invalid config raises error
    Tool: Bash
    Steps:
      1. Create config with missing wake_word field
      2. python -c "from wakeword_workbench.config import load_config; load_config('invalid.yaml')"
    Expected Result: ConfigError raised with clear message
    Evidence: .sisyphus/evidence/task-3-config-error.log
  ```

  **Commit**: YES
  - Message: `feat(config): YAML config loader with validation`

- [ ] **4. CLI Entry Point (Typer)**

  **What to do**:
  - Implement `cli.py` with:
    - `wakeword-workbench` command using typer
    - `--version` flag showing version from package
    - `--verbose` / `--quiet` flags for logging control
    - `run` command: `wakeword-workbench run config.yaml`
    - `validate` command: `wakeword-workbench validate config.yaml`
  - Add console output with rich formatting (progress bars, tables)
  - Implement proper exit codes (0=success, 1=error, 2=config error)

  **Must NOT do**:
  - Do not implement actual logic yet (just CLI structure)
  - Do not add GUI elements

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Typer CLI setup with proper structure

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 1 (with Task 3)
  - **Blocks**: All later tasks

  **Acceptance Criteria**:
  - [ ] `wakeword-workbench --version` prints version
  - [ ] `wakeword-workbench --help` shows all commands
  - [ ] `wakeword-workbench validate missing.yaml` exits with code 2

  **QA Scenarios**:
  ```
  Scenario: CLI responds to basic commands
    Tool: Bash
    Steps:
      1. uv run wakeword-workbench --version
      2. uv run wakeword-workbench --help
    Expected Result: Version printed, help shows commands
    Evidence: .sisyphus/evidence/task-4-cli.txt
  ```

  **Commit**: YES
  - Message: `feat(cli): typer CLI with run and validate commands`

- [ ] **5. Test Infrastructure (pytest + TDD setup)**

  **What to do**:
  - Create `tests/conftest.py` with:
    - pytest fixtures for temporary directories
    - Fixtures for sample configs
    - Fixtures for mock audio data
  - Create `tests/test_config.py` with tests for config loading
  - Create `tests/test_cli.py` with tests for CLI commands
  - Set up `pytest.ini` or `pyproject.toml` pytest section:
    - Coverage reporting
    - Test discovery patterns
    - Markers for slow tests
  - Add test utilities in `tests/utils.py`

  **Must NOT do**:
  - Do not write integration tests yet (unit tests only)
  - Do not test external services (mock TTS)

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Standard pytest setup with fixtures

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 1
  - **Blocks**: All tasks (TDD approach - tests first)

  **Acceptance Criteria**:
  - [ ] `uv run pytest` discovers and runs tests
  - [ ] `uv run pytest --cov` shows coverage report
  - [ ] At least 5 test cases pass

  **QA Scenarios**:
  ```
  Scenario: Test suite runs
    Tool: Bash
    Steps:
      1. uv run pytest -v
    Expected Result: All tests pass (at least 5)
    Evidence: .sisyphus/evidence/task-5-tests.log
  ```

  **Commit**: YES
  - Message: `test(infra): pytest setup with fixtures and coverage`

- [ ] **6. Logging System (structlog)**

  **What to do**:
  - Set up `structlog` configuration in `config.py` or new `logging.py`:
    - JSON format for machine parsing
    - Human-readable format for CLI
    - Log level control via CLI flags
    - Separate log files for different stages
  - Add logging to all modules
  - Create log rotation for long-running operations

  **Must NOT do**:
  - Do not use standard library logging directly
  - Do not print() for diagnostics

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Standard structlog configuration

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 1
  - **Blocks**: Tasks 12+ (all operations use logging)

  **Acceptance Criteria**:
  - [ ] `wakeword-workbench --verbose` shows DEBUG logs
  - [ ] `wakeword-workbench --quiet` shows only ERROR logs
  - [ ] Logs are structured (JSON when redirected)

  **QA Scenarios**:
  ```
  Scenario: Logging respects verbosity flags
    Tool: Bash
    Steps:
      1. uv run wakeword-workbench --version --verbose 2>&1 | grep -i debug
      2. uv run wakeword-workbench --version --quiet 2>&1 | wc -l
    Expected Result: Verbose shows debug, quiet shows minimal
    Evidence: .sisyphus/evidence/task-6-logging.log
  ```

  **Commit**: YES (grouped with Tasks 4-5)
  - Message: `feat(logging): structlog setup with verbosity control`

### Wave 2: TTS Backends (parallel after Wave 1)

- [ ] **7. TTSBackend Abstract Base Class**

  **What to do**:
  - Create `tts/base.py` with:
    - `TTSResult` dataclass (audio: np.ndarray, sample_rate: int, duration: float)
    - `TTSBackend` ABC with methods:
      - `synthesize(text: str) -> TTSResult`
      - `set_voice(voice: str) -> None`
      - `list_voices() -> list[str]`
      - `is_available() -> bool` (class method for checking installation)
  - Define `TTSError` exception class
  - Add factory function `create_backend(name: str) -> TTSBackend`

  **Must NOT do**:
  - Do not implement platform-specific code in base class
  - Do not import optional dependencies at module level

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Abstract class design, no external dependencies

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 2
  - **Blocks**: Tasks 8-9 (implementations)
  - **Blocked By**: Task 1 (project structure)

  **References**:
  - Pattern: Abstract Base Class with factory pattern
  - Similar to: Connection backends in database libraries

  **Acceptance Criteria**:
  - [ ] `TTSBackend` is abstract (cannot instantiate directly)
  - [ ] Factory raises `TTSError` for unknown backend name
  - [ ] All abstract methods are documented

  **QA Scenarios**:
  ```
  Scenario: ABC prevents direct instantiation
    Tool: Bash (python -c)
    Steps:
      1. from wakeword_workbench.tts.base import TTSBackend
      2. TTSBackend()
    Expected Result: TypeError (cannot instantiate abstract)
    Evidence: .sisyphus/evidence/task-7-abc.txt
  ```

  **Commit**: YES
  - Message: `feat(tts): TTSBackend abstract base class`
  - Pre-commit: `uv run pytest tests/test_tts_base.py -v`

- [ ] **8. KokoroBackend Implementation**

  **What to do**:
  - Create `tts/kokoro_backend.py`:
    - Import `pykokoro` optionally (graceful fallback if missing)
    - Implement `KokoroBackend` class extending `TTSBackend`
    - Resample output from 24000 Hz to 16000 Hz using librosa
    - Normalize audio to [-1, 1] float32
    - Support voice selection and speed control
  - Add `is_available()` check for pykokoro installation
  - Add error handling for model download failures

  **Must NOT do**:
  - Do not fail import if pykokoro not installed (optional dependency)
  - Do not cache models in repo directory

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Simple adapter pattern with optional import

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 2 (with Task 9)
  - **Blocks**: Task 13 (positive dataset generation)
  - **Blocked By**: Task 7 (base class)

  **References**:
  - API: `pykokoro.KokoroPipeline`, `pykokoro.PipelineConfig`
  - Research: Output is numpy array at 24000 Hz

  **Acceptance Criteria**:
  - [ ] `KokoroBackend.is_available()` returns True if pykokoro installed
  - [ ] `synthesize()` returns 16000 Hz audio
  - [ ] Missing pykokoro gives graceful warning, not crash

  **QA Scenarios**:
  ```
  Scenario: KokoroBackend synthesizes audio
    Tool: Bash (python -c)
    Steps:
      1. from wakeword_workbench.tts.kokoro_backend import KokoroBackend
      2. backend = KokoroBackend()
      3. result = backend.synthesize("hello")
      4. print(result.sample_rate, len(result.audio))
    Expected Result: sample_rate=16000, audio is numpy array
    Evidence: .sisyphus/evidence/task-8-kokoro.wav (save audio)
  
  Scenario: Missing dependency handled gracefully
    Tool: Bash
    Steps:
      1. pip uninstall pykokoro -y
      2. python -c "from wakeword_workbench.tts.kokoro_backend import KokoroBackend; print(KokoroBackend.is_available())"
    Expected Result: Returns False, no ImportError
    Evidence: .sisyphus/evidence/task-8-graceful.log
  ```

  **Commit**: YES
  - Message: `feat(tts): KokoroBackend with resampling`
  - Pre-commit: `uv run pytest tests/test_kokoro_backend.py -v`

- [ ] **9. PiperBackend Implementation**

  **What to do**:
  - Create `tts/piper_backend.py`:
    - Import `piper` optionally (graceful fallback if missing)
    - Implement `PiperBackend` class extending `TTSBackend`
    - Resample output from 22050 Hz to 16000 Hz
    - Support model path configuration
    - Handle Piper's WAV output format
  - Add `is_available()` check for piper-tts installation
  - Add model auto-download/cache logic

  **Must NOT do**:
  - Do not fail import if piper not installed
  - Do not store model files in git

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Similar to Kokoro, adapter pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 2 (with Task 8)
  - **Blocks**: Task 13
  - **Blocked By**: Task 7

  **References**:
  - API: `piper.PiperVoice.load()`, `synthesize_wav()`
  - Research: Output is 22050 Hz WAV

  **Acceptance Criteria**:
  - [ ] `PiperBackend.is_available()` returns True if piper installed
  - [ ] `synthesize()` returns 16000 Hz audio
  - [ ] Missing piper gives graceful warning

  **QA Scenarios**:
  ```
  Scenario: PiperBackend synthesizes audio
    Tool: Bash (python -c)
    Steps:
      1. from wakeword_workbench.tts.piper_backend import PiperBackend
      2. backend = PiperBackend(model_path="path/to/model.onnx")
      3. result = backend.synthesize("hello")
    Expected Result: 16000 Hz numpy array
    Evidence: .sisyphus/evidence/task-9-piper.wav
  ```

  **Commit**: YES
  - Message: `feat(tts): PiperBackend with model loading`

- [ ] **10. TTS Registry and Discovery**

  **What to do**:
  - Create `tts/__init__.py` with:
    - `list_available_backends() -> list[str]`
    - `get_backend(name: str) -> TTSBackend`
    - Auto-discovery of installed backends
  - Add `tts/registry.py` with backend registration decorator
  - Support for user-defined backends via entry points

  **Must NOT do**:
  - Do not hardcode backend list
  - Do not import all backends at startup

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Plugin registry pattern

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 2
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 8-9

  **Acceptance Criteria**:
  - [ ] `list_available_backends()` shows installed backends
  - [ ] `get_backend("kokoro")` returns KokoroBackend instance
  - [ ] Unknown backend raises clear error

  **QA Scenarios**:
  ```
  Scenario: Registry lists available backends
    Tool: Bash (python -c)
    Steps:
      1. from wakeword_workbench.tts import list_available_backends
      2. print(list_available_backends())
    Expected Result: List contains installed backends
    Evidence: .sisyphus/evidence/task-10-registry.txt
  ```

  **Commit**: YES
  - Message: `feat(tts): backend registry with auto-discovery`

- [ ] **11. TTS Caching System**

  **What to do**:
  - Create `tts/cache.py`:
    - Cache directory: `~/.cache/wakeword_workbench/tts/`
    - Cache key: hash of (text, voice, speed, backend)
    - Cache format: numpy `.npy` files with metadata JSON
    - Cache size limit with LRU eviction
    - Cache cleanup command
  - Integrate caching into backends
  - Add cache hit logging

  **Must NOT do**:
  - Do not cache in project directory
  - Do not cache indefinitely (implement size limits)

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: File-based caching with size management

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 2
  - **Blocks**: Task 13
  - **Blocked By**: Tasks 8-10

  **Acceptance Criteria**:
  - [ ] Second synthesis of same text uses cache
  - [ ] Cache respects size limits
  - [ ] Cache cleanup removes old files

  **QA Scenarios**:
  ```
  Scenario: Caching reduces synthesis time
    Tool: Bash (python script)
    Steps:
      1. Time first synthesis of "hello"
      2. Time second synthesis of "hello"
    Expected Result: Second call is significantly faster
    Evidence: .sisyphus/evidence/task-11-cache-timing.txt
  ```

  **Commit**: YES
  - Message: `feat(tts): synthesis result caching with LRU`

### Wave 3: Dataset Generation (parallel after Wave 2)

- [ ] **12. Phrase Variant Generator**

  **What to do**:
  - Implement `dataset/phrase_variants.py` with `generate_variants(wake_word: str, count: int) -> list[str]`
  - Generate natural variations: punctuation ("hey, assistant"), casing ("HEY Assistant"), spacing ("hey  assistant")
  - Use deterministic seeding for reproducibility

  **Must NOT do**:
  - Do NOT use regex-only approach that misses linguistic variations
  - Do NOT hardcode variant rules; make them configurable

  **Recommended Agent Profile**:
  - **Category**: unspecified-high
  - **Reason**: Linguistic variation generation with phonetic awareness

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 3
  - **Blocks**: Task 13 (uses phrase variants)
  - **Blocked By**: Task 2 (directory structure)

  **Acceptance Criteria**:
  - [ ] `generate_variants("hey assistant", 50)` returns 50 unique strings
  - [ ] Output includes variations in casing, punctuation, spacing
  - [ ] Same seed produces identical output

  **QA Scenarios**:
  ```
  Scenario: Generate variants
    Tool: Bash (python -c)
    Steps:
      1. from wakeword_workbench.dataset.phrase_variants import generate_variants
      2. variants = generate_variants("hey assistant", 50)
      3. print(len(set(variants)))
    Expected Result: 50 (all unique)
    Evidence: .sisyphus/evidence/task-12-variants.txt
  ```

  **Commit**: YES
  - Message: `feat(dataset): phrase variant generator`

- [ ] **13. Positive Dataset Generator**

  **What to do**:
  - Implement `dataset/positive_generator.py` with TTS integration
  - Generate 1000+ samples: iterate phrase variants through multiple voices
  - Output: audio files + JSONL manifest with path, transcript, speaker_id, duration

  **Must NOT do**:
  - Do NOT generate without progress logging
  - Do NOT fail on single TTS failure (continue with next)

  **Recommended Agent Profile**:
  - **Category**: unspecified-high
  - **Reason**: Complex orchestration of TTS and file I/O

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 3
  - **Blocks**: Tasks 14-16
  - **Blocked By**: Tasks 10-12 (TTS registry, phrase variants)

  **Acceptance Criteria**:
  - [ ] Generates specified count of samples
  - [ ] All samples are 16000 Hz mono WAV
  - [ ] Manifest contains all required fields

  **QA Scenarios**:
  ```
  Scenario: Generate positive samples
    Tool: Bash
    Steps:
      1. Generate 10 samples
      2. ls output/positive/*.wav | wc -l
      3. cat output/positive/manifest.jsonl | wc -l
    Expected Result: 10 wav files, 10 manifest lines
    Evidence: .sisyphus/evidence/task-13-positive.log
  ```

  **Commit**: YES
  - Message: `feat(dataset): positive sample generator with TTS`

- [ ] **14. Negative Phrase Generator (Confusions)**

  **What to do**:
  - Implement `negatives/phrase_generator.py`
  - Generate phonetically similar phrases: "hay assistant", "hey assistance"
  - Use phoneme distance (metaphone/double metaphone) to find confusable phrases

  **Must NOT do**:
  - Do NOT use random unrelated phrases
  - Do NOT include actual wake word in negatives

  **Recommended Agent Profile**:
  - **Category**: unspecified-high
  - **Reason**: Phonetic similarity algorithms

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 3
  - **Blocks**: Task 15
  - **Blocked By**: Task 12 (phrase structure understanding)

  **Acceptance Criteria**:
  - [ ] Generated phrases are phonetically similar (metaphone match > 0.6)
  - [ ] No exact wake word matches
  - [ ] Sufficient diversity in confusions

  **QA Scenarios**:
  ```
  Scenario: Generate confusions
    Tool: Bash (python -c)
    Steps:
      1. from wakeword_workbench.negatives.phrase_generator import generate_confusions
      2. confusions = generate_confusions("hey marvin", 20)
      3. Check phonetic similarity
    Expected Result: All confusions phonetically similar, none exact match
    Evidence: .sisyphus/evidence/task-14-confusions.txt
  ```

  **Commit**: YES
  - Message: `feat(negatives): confusion phrase generator`

- [ ] **15. Synthetic Negative Generator**

  **What to do**:
  - Implement `negatives/synthetic_generator.py`
  - Concatenate random words not forming wake word
  - Match length distribution to positive samples

  **Must NOT do**:
  - Do NOT use Markov chains that might generate wake word
  - Do NOT generate shorter than shortest positive

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Simple random concatenation

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 3
  - **Blocks**: Task 16
  - **Blocked By**: Task 13 (length distribution reference)

  **Acceptance Criteria**:
  - [ ] Generated phrases don't contain wake word
  - [ ] Length distribution matches positives
  - [ ] Configurable randomness

  **QA Scenarios**:
  ```
  Scenario: Generate synthetic negatives
    Tool: Bash (python -c)
    Steps:
      1. Generate 100 synthetic negatives
      2. Check for wake word substring
    Expected Result: 100 unique phrases, none contain wake word
    Evidence: .sisyphus/evidence/task-15-synthetic.log
  ```

  **Commit**: YES
  - Message: `feat(negatives): synthetic negative generator`

- [ ] **16. Metadata JSONL System**

  **What to do**:
  - Implement `dataset/metadata.py` with `ManifestEntry` dataclass
  - JSONL format: `{"path": "...", "label": 0/1, "duration_ms": ..., "speaker_id": ...}`
  - Support append, merge, validate operations

  **Must NOT do**:
  - Do NOT use CSV (JSONL is standard)
  - Do NOT omit required fields

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Simple dataclass and file I/O

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 3
  - **Blocks**: Tasks 23-27
  - **Blocked By**: Tasks 13, 15 (need samples to track)

  **Acceptance Criteria**:
  - [ ] Manifest validates schema correctly
  - [ ] Merge operation handles duplicates
  - [ ] JSONL lines are valid JSON

  **QA Scenarios**:
  ```
  Scenario: Create and validate manifest
    Tool: Bash (python -c)
    Steps:
      1. Create manifest with 10 entries
      2. Validate manifest
    Expected Result: All entries valid, no schema errors
    Evidence: .sisyphus/evidence/task-16-manifest.jsonl
  ```

  **Commit**: YES
  - Message: `feat(dataset): JSONL metadata system`

### Wave 4: Augmentation (parallel after Wave 3)

- [ ] **17. Audio Loader/Resampler**

  **What to do**:
  - Implement `augment/audio_loader.py` with `load_audio(path, target_sr=16000)`
  - Use librosa for loading with automatic resampling
  - Return numpy array + metadata (original_sr, duration, channels)

  **Must NOT do**:
  - Do NOT assume all audio is 16kHz
  - Do NOT silently fail on corrupted files

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Standard librosa usage

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 4
  - **Blocks**: Tasks 18-22
  - **Blocked By**: Task 2 (directory structure)

  **Acceptance Criteria**:
  - [ ] Loads any supported audio format
  - [ ] Resamples to target sample rate
  - [ ] Raises descriptive errors on corruption

  **QA Scenarios**:
  ```
  Scenario: Load and resample audio
    Tool: Bash (python -c)
    Steps:
      1. Load 48kHz audio
      2. Check output sample rate
    Expected Result: Output is 16000 Hz
    Evidence: .sisyphus/evidence/task-17-loader.log
  ```

  **Commit**: YES
  - Message: `feat(augment): audio loader with resampling`

- [ ] **18. Noise Augmentation**

  **What to do**:
  - Implement `augment/noise.py` with `AddNoise(noise_dir, snr_range=(-10, 10), p=0.75)`
  - Support background noise dataset (MUSAN or custom)
  - SNR mixing with configurable probability

  **Must NOT do**:
  - Do NOT apply to all samples (respect p=0.75)
  - Do NOT use noise shorter than audio

  **Recommended Agent Profile**:
  - **Category**: unspecified-high
  - **Reason**: Signal processing with SNR calculations

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 4
  - **Blocks**: Task 22
  - **Blocked By**: Task 17 (audio loading)

  **Acceptance Criteria**:
  - [ ] Correct SNR mixing
  - [ ] Respects probability parameter
  - [ ] Handles various noise lengths

  **QA Scenarios**:
  ```
  Scenario: Add noise at 0dB SNR
    Tool: Bash (python -c)
    Steps:
      1. Augment clean speech with 0dB SNR
      2. Measure RMS of output
    Expected Result: Output RMS ≈ 2x clean (signal + noise)
    Evidence: .sisyphus/evidence/task-18-noise.wav
  ```

  **Commit**: YES
  - Message: `feat(augment): background noise augmentation`

- [ ] **19. Reverb Augmentation**

  **What to do**:
  - Implement `augment/reverb.py` with `AddReverb(rir_dir, p=0.5)`
  - Convolve with room impulse responses
  - Support configurable RT60 ranges

  **Must NOT do**:
  - Do NOT apply to all samples (p=0.5)
  - Do NOT use FFT that creates artifacts

  **Recommended Agent Profile**:
  - **Category**: unspecified-high
  - **Reason**: Convolution for audio effects

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 4
  - **Blocks**: Task 22
  - **Blocked By**: Task 17

  **Acceptance Criteria**:
  - [ ] Convolution produces reverb tail
  - [ ] Respects probability parameter
  - [ ] No artifacts at boundaries

  **QA Scenarios**:
  ```
  Scenario: Apply reverb
    Tool: Bash (python -c)
    Steps:
      1. Augment dry speech with reverb
      2. Check spectrogram for smearing
    Expected Result: Longer tail, frequency smearing visible
    Evidence: .sisyphus/evidence/task-19-reverb.wav
  ```

  **Commit**: YES
  - Message: `feat(augment): reverb augmentation with RIR`

- [ ] **20. Gain/Clipping Augmentation**

  **What to do**:
  - Implement `augment/gain.py` with `AdjustGain(gain_range=(-45, 0), p=1.0)`
  - Apply gain in dB, handle clipping gracefully
  - Optional soft clipping mode

  **Must NOT do**:
  - Do NOT clip by default
  - Do NOT push all samples to silence

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Simple numpy operations

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 4
  - **Blocks**: Task 22
  - **Blocked By**: Task 17

  **Acceptance Criteria**:
  - [ ] Correct dB gain application
  - [ ] Optional clipping handled
  - [ ] No overflow/underflow

  **QA Scenarios**:
  ```
  Scenario: Apply -20dB gain
    Tool: Bash (python -c)
    Steps:
      1. Input RMS=0.1
      2. Apply -20dB gain
      3. Measure output RMS
    Expected Result: Output RMS ≈ 0.01
    Evidence: .sisyphus/evidence/task-20-gain.log
  ```

  **Commit**: YES
  - Message: `feat(augment): gain and clipping transforms`

- [ ] **21. Silence Padding/Transforms**

  **What to do**:
  - Implement `augment/padding.py` with `FixedSizeClip(target_samples=16000, jitter=True)`
  - Center or random pad audio to fixed length
  - Jitter: random offset for temporal variation

  **Must NOT do**:
  - Do NOT crop long audio (pad only)
  - Do NOT introduce DC offset

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Simple padding operations

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 4
  - **Blocks**: Task 22
  - **Blocked By**: Task 17

  **Acceptance Criteria**:
  - [ ] Output is exactly target length
  - [ ] Jitter produces variation
  - [ ] No DC offset introduced

  **QA Scenarios**:
  ```
  Scenario: Pad short audio
    Tool: Bash (python -c)
    Steps:
      1. Input 8000 samples
      2. Pad to 16000
      3. Check output length
    Expected Result: Exactly 16000 samples
    Evidence: .sisyphus/evidence/task-21-padding.log
  ```

  **Commit**: YES
  - Message: `feat(augment): silence padding and fixed-size clips`

- [ ] **22. Augmentation Pipeline Composer**

  **What to do**:
  - Implement `augment/pipeline.py` with `Compose([transform1, ...])`
  - Configurable YAML/JSON pipeline definition
  - Support probability chains

  **Must NOT do**:
  - Do NOT mutate input array
  - Do NOT skip transform validation

  **Recommended Agent Profile**:
  - **Category**: unspecified-high
  - **Reason**: Pipeline orchestration with config

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 4
  - **Blocks**: Tasks 23-27
  - **Blocked By**: Tasks 18-21

  **Acceptance Criteria**:
  - [ ] Applies transforms in order
  - [ ] Config loading works
  - [ ] Returns new array (no mutation)

  **QA Scenarios**:
  ```
  Scenario: Compose pipeline
    Tool: Bash (python -c)
    Steps:
      1. Create pipeline with Gain + Noise
      2. Apply to audio
      3. Verify both applied
    Expected Result: Both transforms effects visible
    Evidence: .sisyphus/evidence/task-22-pipeline.log
  ```

  **Commit**: YES
  - Message: `feat(augment): configurable augmentation pipeline`

### Wave 5: Dataset Builder and Export (parallel after Wave 4)

- [ ] **23. Dataset Merger**

  **What to do**:
  - Implement `dataset/merger.py` with `merge(pos_manifest, neg_manifest, output_manifest)`
  - Balance classes with configurable ratio
  - Shuffle deterministically

  **Must NOT do**:
  - Do NOT merge without checking path collisions
  - Do NOT lose metadata

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: File operations and validation

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 5
  - **Blocks**: Task 24
  - **Blocked By**: Tasks 16, 22 (manifests and augmented data)

  **Acceptance Criteria**:
  - [ ] Correct class balance
  - [ ] No duplicate paths
  - [ ] Deterministic shuffle

  **QA Scenarios**:
  ```
  Scenario: Merge datasets
    Tool: Bash (python -c)
    Steps:
      1. Merge 1000 pos + 3000 neg with 1:1 ratio
      2. Check output count
    Expected Result: 2000 samples (1000 each)
    Evidence: .sisyphus/evidence/task-23-merge.log
  ```

  **Commit**: YES
  - Message: `feat(dataset): positive/negative merger`

- [ ] **24. Leak-Proof Splitter**

  **What to do**:
  - Implement `dataset/splitter.py` with `split(manifest, train=0.7, val=0.15, test=0.15, by='speaker')`
  - Ensure no speaker overlap between splits
  - Stratify by label

  **Must NOT do**:
  - Do NOT allow same speaker in multiple splits
  - Do NOT shuffle before stratification

  **Recommended Agent Profile**:
  - **Category**: unspecified-high
  - **Reason**: Data leakage prevention is critical

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 5
  - **Blocks**: Tasks 25-27
  - **Blocked By**: Task 23

  **Acceptance Criteria**:
  - [ ] No speaker overlap between splits
  - [ ] Class ratios preserved
  - [ ] Correct split proportions

  **QA Scenarios**:
  ```
  Scenario: Split by speaker
    Tool: Bash (python -c)
    Steps:
      1. Split manifest with 10 speakers
      2. Check for overlap
    Expected Result: Each speaker in exactly one split
    Evidence: .sisyphus/evidence/task-24-split.log
  ```

  **Commit**: YES
  - Message: `feat(dataset): leak-proof train/val/test splitter`

- [ ] **25. microWakeWord Exporter**

  **What to do**:
  - Implement `export/microwakeword.py` with `export_to_mmap(manifest, output_dir)`
  - Ragged Mmap format for variable-length clips
  - Generate indices + data + labels files

  **Must NOT do**:
  - Do NOT pad all clips (defeats ragged purpose)
  - Do NOT forget label array

  **Recommended Agent Profile**:
  - **Category**: unspecified-high
  - **Reason**: Binary format expertise needed

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 5
  - **Blocks**: Task 27
  - **Blocked By**: Task 24

  **Acceptance Criteria**:
  - [ ] Creates valid mmap files
  - [ ] Indices match manifest
  - [ ] Loadable by microWakeWord

  **QA Scenarios**:
  ```
  Scenario: Export microWakeWord format
    Tool: Bash
    Steps:
      1. Export 100 samples
      2. Check files: data.mmap, indices.npy, labels.npy
    Expected Result: All files present, correct shapes
    Evidence: .sisyphus/evidence/task-25-microwakeword/
  ```

  **Commit**: YES
  - Message: `feat(export): microWakeWord Ragged Mmap exporter`

- [ ] **26. openWakeWord Exporter**

  **What to do**:
  - Implement `export/openwakeword.py` with `export_to_numpy(manifest, output_dir)`
  - Export as X_train.npy, y_train.npy, etc.
  - Option for fixed-length padding

  **Must NOT do**:
  - Do NOT export without label encoding
  - Do NOT use pickle

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Simple numpy array export

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 5
  - **Blocks**: Task 27
  - **Blocked By**: Task 24

  **Acceptance Criteria**:
  - [ ] Creates valid numpy arrays
  - [ ] Correct shapes
  - [ ] Loadable by openWakeWord

  **QA Scenarios**:
  ```
  Scenario: Export openWakeWord format
    Tool: Bash
    Steps:
      1. Export dataset
      2. Load X_train.npy, y_train.npy
    Expected Result: Arrays loadable, consistent dimensions
    Evidence: .sisyphus/evidence/task-26-openwakeword/
  ```

  **Commit**: YES
  - Message: `feat(export): openWakeWord numpy exporter`

- [ ] **27. Export Validation**

  **What to do**:
  - Implement `export/validator.py` with format-specific validators
  - Check file existence, format integrity
  - Verify data matches manifest

  **Must NOT do**:
  - Do NOT just check file existence
  - Do NOT skip corruption detection

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Validation logic

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 5
  - **Blocks**: Tasks 28-32
  - **Blocked By**: Tasks 25-26

  **Acceptance Criteria**:
  - [ ] Detects format errors
  - [ ] Validates data integrity
  - [ ] Returns detailed report

  **QA Scenarios**:
  ```
  Scenario: Validate export
    Tool: Bash
    Steps:
      1. Run validator on export
      2. Check output
    Expected Result: Validation report with pass/fail status
    Evidence: .sisyphus/evidence/task-27-validation.log
  ```

  **Commit**: YES
  - Message: `feat(export): export format validator`

### Wave 6: Evaluation (parallel after Wave 5)

- [ ] **28. FAR Calculator**

  **What to do**:
  - Implement `eval/far.py` with `calculate_far(predictions, audio_hours)`
  - FAR = false_positives / audio_hours
  - Support sliding window aggregation

  **Must NOT do**:
  - Do NOT confuse with FRR
  - Do NOT forget cooldown window

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Simple calculation

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 6
  - **Blocks**: Tasks 30-32
  - **Blocked By**: Task 27

  **Acceptance Criteria**:
  - [ ] Correct FAR calculation
  - [ ] Handles edge cases
  - [ ] Cooldown window respected

  **QA Scenarios**:
  ```
  Scenario: Calculate FAR
    Tool: Bash (python -c)
    Steps:
      1. 10 FPs in 2 hours
      2. Calculate FAR
    Expected Result: FAR = 5.0 per hour
    Evidence: .sisyphus/evidence/task-28-far.log
  ```

  **Commit**: YES
  - Message: `feat(eval): FAR calculator with cooldown`

- [ ] **29. FRR Calculator**

  **What to do**:
  - Implement `eval/frr.py` with `calculate_frr(predictions, ground_truth)`
  - FRR = false_negatives / total_positives

  **Must NOT do**:
  - Do NOT include true negatives in denominator
  - Do NOT use unthresholded probabilities

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Simple calculation

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 6
  - **Blocks**: Tasks 30-32
  - **Blocked By**: Task 27

  **Acceptance Criteria**:
  - [ ] Correct FRR calculation
  - [ ] Handles edge cases

  **QA Scenarios**:
  ```
  Scenario: Calculate FRR
    Tool: Bash (python -c)
    Steps:
      1. 5 FNs from 100 positives
      2. Calculate FRR
    Expected Result: FRR = 0.05 (5%)
    Evidence: .sisyphus/evidence/task-29-frr.log
  ```

  **Commit**: YES
  - Message: `feat(eval): FRR calculator`

- [ ] **30. Threshold Sweep**

  **What to do**:
  - Implement `eval/threshold_sweep.py` with `sweep(predictions, ground_truth, thresholds=100)`
  - Generate (threshold, FAR, FRR) tuples from 0.0 to 1.0

  **Must NOT do**:
  - Do NOT hardcode to 100 points
  - Do NOT miss endpoints

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Simple iteration

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 6
  - **Blocks**: Tasks 31-32
  - **Blocked By**: Tasks 28-29

  **Acceptance Criteria**:
  - [ ] Correct threshold range
  - [ ] Returns all metrics per threshold

  **QA Scenarios**:
  ```
  Scenario: Sweep thresholds
    Tool: Bash (python -c)
    Steps:
      1. Sweep 10 thresholds
      2. Check range
    Expected Result: 10 points from ~0.0 to ~1.0
    Evidence: .sisyphus/evidence/task-30-sweep.log
  ```

  **Commit**: YES
  - Message: `feat(eval): threshold sweep analysis`

- [ ] **31. ROC Curve Generator**

  **What to do**:
  - Implement `eval/roc.py` with `generate_roc(far_list, frr_list)`
  - Generate ROC curve data (FAR vs FRR)
  - Optional plotting

  **Must NOT do**:
  - Do NOT confuse axes
  - Do NOT skip raw data return

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Data transformation

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 6
  - **Blocks**: Task 32
  - **Blocked By**: Task 30

  **Acceptance Criteria**:
  - [ ] Correct curve data
  - [ ] Plot generates correctly

  **QA Scenarios**:
  ```
  Scenario: Generate ROC
    Tool: Bash (python -c)
    Steps:
      1. Generate ROC data
      2. Create plot
    Expected Result: Valid curve data, plot file created
    Evidence: .sisyphus/evidence/task-31-roc.png
  ```

  **Commit**: YES
  - Message: `feat(eval): ROC curve generator`

- [ ] **32. Evaluation Report Formatter**

  **What to do**:
  - Implement `eval/report.py` with `generate_report(results, format='json')`
  - Output: JSON, CSV, Markdown
  - Include EER, optimal threshold

  **Must NOT do**:
  - Do NOT omit EER
  - Do NOT skip format validation

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Report formatting

  **Parallelization**:
  - **Can Run In Parallel**: YES - Wave 6
  - **Blocks**: Tasks 33-35
  - **Blocked By**: Task 31

  **Acceptance Criteria**:
  - [ ] All formats work
  - [ ] EER calculated correctly
  - [ ] All metrics included

  **QA Scenarios**:
  ```
  Scenario: Generate report
    Tool: Bash
    Steps:
      1. Generate JSON report
      2. Check fields
    Expected Result: Valid JSON with far, frr, eer, optimal_threshold
    Evidence: .sisyphus/evidence/task-32-report.json
  ```

  **Commit**: YES
  - Message: `feat(eval): evaluation report formatter`

### Wave 7: Hard Negative Mining (after Wave 6)

- [ ] **33. Long Audio Processor**

  **What to do**:
  - Implement `mining/long_audio.py` with `process_long_audio(model, audio_path, window_size, hop_size)`
  - Sliding window inference
  - Chunked loading for memory efficiency

  **Must NOT do**:
  - Do NOT load entire hour-long audio to memory
  - Do NOT miss windows at end

  **Recommended Agent Profile**:
  - **Category**: unspecified-high
  - **Reason**: Memory-efficient streaming

  **Parallelization**:
  - **Can Run In Parallel**: NO - Sequential on Wave 7
  - **Blocks**: Task 34
  - **Blocked By**: Task 32

  **Acceptance Criteria**:
  - [ ] Processes long audio without OOM
  - [ ] All windows covered
  - [ ] Returns predictions with timestamps

  **QA Scenarios**:
  ```
  Scenario: Process long audio
    Tool: Bash (python -c)
    Steps:
      1. Process 10-min audio
      2. Check predictions count
    Expected Result: ~1200 predictions (1s window, 0.5s hop)
    Evidence: .sisyphus/evidence/task-33-longaudio.log
  ```

  **Commit**: YES
  - Message: `feat(mining): sliding window long audio processor`

- [ ] **34. False Positive Extractor**

  **What to do**:
  - Implement `mining/extractor.py` with `extract_false_positives(predictions, audio_path, threshold, output_dir)`
  - Extract clips where prediction > threshold
  - Save with metadata

  **Must NOT do**:
  - Do NOT extract true positives
  - Do NOT overwrite without warning

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Audio slicing and saving

  **Parallelization**:
  - **Can Run In Parallel**: NO - Sequential on Wave 7
  - **Blocks**: Task 35
  - **Blocked By**: Task 33

  **Acceptance Criteria**:
  - [ ] Extracts only false positives
  - [ ] Saves with metadata
  - [ ] No duplicate clips

  **QA Scenarios**:
  ```
  Scenario: Extract false positives
    Tool: Bash
    Steps:
      1. Extract at threshold 0.5
      2. Check output clips
    Expected Result: Clips above threshold saved
    Evidence: .sisyphus/evidence/task-34-extracted/
  ```

  **Commit**: YES
  - Message: `feat(mining): false positive clip extractor`

- [ ] **35. Dataset Merge-Back**

  **What to do**:
  - Implement `mining/merge_back.py` with `add_to_training(new_negatives_manifest, training_manifest)`
  - Validate new negatives
  - Update training manifest with backup

  **Must NOT do**:
  - Do NOT merge without backup
  - Do NOT add without verification

  **Recommended Agent Profile**:
  - **Category**: quick
  - **Reason**: Manifest manipulation

  **Parallelization**:
  - **Can Run In Parallel**: NO - Sequential on Wave 7
  - **Blocks**: Tasks F1-F4
  - **Blocked By**: Task 34

  **Acceptance Criteria**:
  - [ ] Manifest updated correctly
  - [ ] Backup created
  - [ ] Count increased correctly

  **QA Scenarios**:
  ```
  Scenario: Merge new negatives
    Tool: Bash
    Steps:
      1. Add 50 new negatives
      2. Check manifest count
    Expected Result: Count increased by 50
    Evidence: .sisyphus/evidence/task-35-mergeback.log
  ```

  **Commit**: YES
  - Message: `feat(mining): hard negative merge-back`

---

## Final Verification Wave

- [ ] **F1. Plan Compliance Audit (oracle)**

  **What to do**:
  - Librarian agent reviews all features against plan
  - Check: All waves completed, dependencies respected
  - Generate compliance report

  **Acceptance Criteria**:
  - [ ] All tasks implemented
  - [ ] No missing features
  - [ ] Dependencies respected

  **QA Scenarios**:
  ```
  Scenario: Compliance check
    Tool: oracle agent
    Steps:
      1. Review all implemented code
      2. Compare against plan
    Expected Result: Compliance report showing 100% or issues flagged
    Evidence: .sisyphus/evidence/f1-compliance.md
  ```

- [ ] **F2. Code Quality Review (unspecified-high)**

  **What to do**:
  - Run ruff check
  - Run mypy type checking
  - Run pytest with coverage

  **Acceptance Criteria**:
  - [ ] ruff: zero errors
  - [ ] mypy: zero type errors
  - [ ] pytest: >80% coverage, all tests pass

  **QA Scenarios**:
  ```
  Scenario: Quality gates
    Tool: Bash
    Steps:
      1. uv run ruff check .
      2. uv run mypy src/
      3. uv run pytest --cov
    Expected Result: All pass with coverage >= 80%
    Evidence: .sisyphus/evidence/f2-quality.log
  ```

- [ ] **F3. End-to-End Pipeline Test (unspecified-high)**

  **What to do**:
  - Run complete pipeline with small dataset
  - Verify all outputs
  - Test both export formats

  **Acceptance Criteria**:
  - [ ] Pipeline completes without errors
  - [ ] All artifacts created
  - [ ] Exports are valid

  **QA Scenarios**:
  ```
  Scenario: Full pipeline
    Tool: Bash
    Steps:
      1. uv run wakeword-workbench run examples/test-config.yaml
      2. Check outputs
    Expected Result: All stages complete, exports valid
    Evidence: .sisyphus/evidence/f3-e2e/
  ```

- [ ] **F4. Scope Fidelity Check (deep)**

  **What to do**:
  - Review against original scope
  - Document any deviations
  - Verify no scope creep

  **Acceptance Criteria**:
  - [ ] All planned features present
  - [ ] No unplanned additions
  - [ ] Guardrails respected

  **QA Scenarios**:
  ```
  Scenario: Scope review
    Tool: Manual review
    Steps:
      1. Compare to original spec
      2. Check Must NOT Have list
    Expected Result: All in scope present, out of scope absent
    Evidence: .sisyphus/evidence/f4-scope.md
  ```

---

## Commit Strategy

- **1**: Project setup and scaffolding
- **2**: TTS backends and abstractions
- **3**: Dataset generation infrastructure
- **4**: Augmentation pipeline
- **5**: Dataset builder and export
- **6**: Evaluation metrics
- **7**: Hard negative mining
- **8**: Final verification and polish

---

## Success Criteria

### Verification Commands
```bash
# Test suite
uv run pytest --cov=wakeword_workbench --cov-report=term-missing

# Type checking
uv run mypy src/wakeword_workbench

# Linting
uv run ruff check src/wakeword_workbench

# CLI smoke test
uv run wakeword-workbench --version

# Config validation
uv run wakeword-workbench validate examples/basic-config.yaml

# Full pipeline (end-to-end)
uv run wakeword-workbench run examples/full-pipeline.yaml
```

### Final Checklist
- [ ] All Must Have items present
- [ ] All Must NOT Have items absent
- [ ] All tests pass with >80% coverage
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)
- [ ] CLI runs without errors
- [ ] Example configs execute successfully
