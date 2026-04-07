# Fix microWakeWord Notebook - Complete Training Pipeline

## TL;DR

> Fix the microWakeWord notebook by implementing the missing DatasetGenerator orchestrator, wiring up the CLI run command, and adding required dependencies. Enable end-to-end wake word training from "hey vera" dataset generation through TensorFlow model training.
> 
> **Deliverables:**
> - DatasetGenerator class in `dataset/generator.py`
> - Working CLI `run` command
> - Updated pyproject.toml with dev dependencies
> - Fixed notebook cells
> - End-to-end test with "hey vera"
> 
> **Estimated Effort:** Medium (3-4 hours)
> **Parallel Execution:** YES - Dependencies + Generator + CLI can be parallel waves
> **Critical Path:** Dependencies → Generator → CLI → Integration Test

---

## Context

### Original Request
User wants to fix the microWakeWord notebook so it actually functions and can train wake word models.

### Current State
| Component | Status | Notes |
|-----------|--------|-------|
| `dataset/generator.py` | **EMPTY** - Only docstring | Main orchestrator missing |
| `cli.py run command` | **STUBBED** - Line 106 | Just prints "Pipeline ready (stub)" |
| Dependencies | **MISSING** | tensorflow, pymicro-features, mmap-ninja, etc. |
| All other components | **WORKING** ✅ | TTS, positives, negatives, augmentation, export |

### Interview Summary
**Key Decisions:**
- Scope: Fix EVERYTHING (notebook + generator + CLI + dependencies)
- Testing: Full pipeline (dataset generation through TensorFlow training)
- Dependencies: Add to pyproject.toml dev group AND notebook installs them
- Wake word: "hey vera" for testing

**Design Decisions (Confirmed):**
- Augmentation: Auto-disable for microWakeWord output
- Split ratio: Configurable with defaults `[0.7, 0.15, 0.15]`
- Output collision: Raise error if directory exists
- Partial failure: Fail fast + cleanup
- Voice distribution: Cycle evenly
- Negative sources: Both confusion phrases + synthetic

### Metis Review
**Identified Gaps (addressed):**
- DatasetGenerator scope clarified: Output WAV files + JSONL (NOT spectrograms)
- Architecture: Thin orchestrator using existing components
- CLI wiring: Replace stub with actual DatasetGenerator call

---

## Work Objectives

### Core Objective
Implement a working DatasetGenerator that orchestrates the full pipeline from config to exported dataset, enabling the microWakeWord notebook to run end-to-end.

### Concrete Deliverables
1. `DatasetGenerator` class in `src/wakeword_workbench/dataset/generator.py`
2. Wired-up `run` command in `src/wakeword_workbench/cli.py`
3. Updated `pyproject.toml` with dev dependencies
4. Fixed notebook cells in `notebooks/training-with-microwakeword.ipynb`
5. End-to-end test configuration and verification

### Definition of Done
- [ ] `uv run wakeword-workbench run test_config.yaml` completes successfully
- [ ] Notebook Section 2 executes without errors
- [ ] Generated dataset has correct structure (audio/ + manifests)
- [ ] All dependencies install correctly

### Must Have
- DatasetGenerator orchestrating: positives → negatives → merge → split → export
- CLI run command using DatasetGenerator
- Dependencies in pyproject.toml
- Notebook fixed and functional
- End-to-end test passes

### Must NOT Have (Guardrails)
- Do NOT implement spectrogram generation (notebook handles this)
- Do NOT use pymicro-features or mmap-ninja in workbench (notebook handles this)
- Do NOT modify other CLI commands
- Do NOT add new TTS/augmentation logic (use existing)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES - pytest is set up
- **Automated tests**: Tests-after (add tests after implementation)
- **Framework**: pytest with fixtures in `conftest.py`

### QA Policy
Every task MUST include agent-executed QA scenarios.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation):
├── Task 1: Add dependencies to pyproject.toml
├── Task 2: Read existing components (positive_generator, merger, splitter)
└── Task 3: Design DatasetGenerator interface

Wave 2 (After Wave 1 - core implementation, MAX PARALLEL):
├── Task 4: Implement DatasetGenerator class
├── Task 5: Add GeneratorError exception
├── Task 6: Create GenerationResult dataclass
└── Task 7: Write unit tests for DatasetGenerator

Wave 3 (After Wave 2 - CLI integration):
├── Task 8: Wire CLI run command to DatasetGenerator
├── Task 9: Add progress bars for CLI feedback
├── Task 10: Write CLI integration tests

Wave 4 (After Wave 3 - notebook & integration):
├── Task 11: Fix notebook dependency cells
├── Task 12: Update notebook Section 2 for new workflow
├── Task 13: Create test config for "hey vera"
└── Task 14: Run end-to-end integration test

Wave FINAL (After ALL tasks - verification):
├── Task F1: Full pipeline verification
└── Task F2: Code quality check
```

### Dependency Matrix

| Task | Blocks | Blocked By |
|------|--------|------------|
| 1 | 4, 8 | — |
| 2 | 4 | — |
| 3 | 4 | 2 |
| 4 | 5, 6, 7, 8 | 1, 2, 3 |
| 5 | — | 4 |
| 6 | — | 4 |
| 7 | — | 4 |
| 8 | 9, 10 | 4 |
| 9 | — | 8 |
| 10 | — | 8 |
| 11 | 14 | — |
| 12 | 14 | 4 |
| 13 | 14 | — |
| 14 | F1, F2 | 7, 10, 12, 13 |

### Agent Dispatch Summary

- **Wave 1**: quick (3 tasks)
- **Wave 2**: deep (4 tasks)  
- **Wave 3**: quick (3 tasks)
- **Wave 4**: unspecified-high (4 tasks)
- **FINAL**: unspecified-high (2 tasks)

---

## TODOs

- [x] 1. Add notebook dependencies to pyproject.toml

  **What to do:**
  - Add tensorflow, pymicro-features, mmap-ninja, datasets, tqdm, matplotlib to dev dependency group
  - Handle platform-specific pymicro-features (macOS fork)
  - Verify dependencies resolve correctly

  **Must NOT do:**
  - Don't add to main dependencies (keep them dev-only)
  - Don't add version constraints that conflict with existing packages

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: []
  - **Justification**: Simple pyproject.toml edit, no complex logic

  **Parallelization:**
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 4, Task 8
  - **Blocked By**: None

  **References:**
  - `pyproject.toml` - Current dependency structure
  - `notebooks/training-with-microwakeword.ipynb` - Cell showing required packages

  **Acceptance Criteria:**
  - [ ] `uv sync --group dev` completes without errors
  - [ ] `python -c "import tensorflow, pymicro_features, mmap_ninja, datasets, tqdm, matplotlib"` succeeds

  **QA Scenarios:**
  ```
  Scenario: Dependencies install correctly
    Tool: Bash
    Steps:
      1. uv sync --group dev
      2. python -c "import tensorflow; print('TF:', tensorflow.__version__)"
    Expected: Exit 0, TensorFlow version printed
  ```

  **Commit**: YES
  - Message: `chore(deps): add notebook training dependencies`
  - Files: `pyproject.toml`

---

- [x] 2. Read and understand existing component interfaces

  **What to do:**
  - Read `dataset/positive_generator.py` - understand PositiveGenerator API
  - Read `dataset/merger.py` - understand merge_manifests function
  - Read `dataset/splitter.py` - understand split_manifest function
  - Read `negatives/phrase_generator.py` - understand generate_confusions
  - Read `negatives/synthetic_generator.py` - understand generate_synthetic_negatives
  - Document function signatures, return types, and usage patterns

  **Must NOT do:**
  - Don't modify any existing files
  - Don't implement anything yet

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: []
  - **Justification**: Read-only analysis task

  **Parallelization:**
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3, Task 4
  - **Blocked By**: None

  **References:**
  - `src/wakeword_workbench/dataset/positive_generator.py`
  - `src/wakeword_workbench/dataset/merger.py`
  - `src/wakeword_workbench/dataset/splitter.py`
  - `src/wakeword_workbench/negatives/phrase_generator.py`
  - `src/wakeword_workbench/negatives/synthetic_generator.py`

  **Acceptance Criteria:**
  - [ ] Documented function signatures for all key functions
  - [ ] Understanding of how components connect

  **QA Scenarios:**
  ```
  Scenario: Components documented
    Tool: N/A (read-only)
    Expected: Design doc with component interfaces
  ```

  **Commit**: NO (research only)

---

- [x] 3. Design DatasetGenerator interface

  **What to do:**
  - Define DatasetGenerator class signature
  - Define GeneratorError exception
  - Define GenerationResult dataclass
  - Document the orchestration flow
  - Get design review (if needed)

  **Must NOT do:**
  - Don't implement yet
  - Don't modify existing code

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: []
  - **Justification**: Design task, no implementation

  **Parallelization:**
  - **Can Run In Parallel**: YES (after Task 2)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References:**
  - Results from Task 2
  - `src/wakeword_workbench/config.py` - Config dataclass structure

  **Acceptance Criteria:**
  - [ ] Class design documented
  - [ ] Method signatures defined
  - [ ] Flow diagram created

  **QA Scenarios:**
  ```
  Scenario: Design documented
    Tool: N/A
    Expected: Design doc in .sisyphus/drafts/
  ```

  **Commit**: NO (design doc)

---

- [x] 4. Implement DatasetGenerator class

  **What to do:**
  - Create `GeneratorError` exception class
  - Create `GenerationResult` dataclass
  - Implement `DatasetGenerator.__init__(self, config: Config)`
  - Implement `DatasetGenerator.generate(self) -> GenerationResult`:
    1. Create output directories
    2. Generate positive samples via PositiveGenerator
    3. Generate negative samples (confusion phrases + synthetic)
    4. Skip augmentation for microWakeWord output
    5. Merge positives + negatives via merger
    6. Split into train/val/test via splitter
    7. Export manifests and audio files
    8. Return GenerationResult
  - Add comprehensive logging with structlog

  **Must NOT do:**
  - Don't implement spectrogram generation
  - Don't call pymicro-features or mmap-ninja
  - Don't add augmentation logic (skip for microWakeWord)

  **Recommended Agent Profile:**
  - **Category**: `deep`
  - **Skills**: []
  - **Justification**: Core orchestrator logic, requires careful integration of multiple components

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on Wave 1)
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 5, 6, 7, 8
  - **Blocked By**: Tasks 1, 2, 3

  **References:**
  - `src/wakeword_workbench/dataset/positive_generator.py` - PositiveGenerator usage
  - `src/wakeword_workbench/dataset/merger.py` - merge_manifests function
  - `src/wakeword_workbench/dataset/splitter.py` - split_manifest function
  - `src/wakeword_workbench/negatives/phrase_generator.py` - generate_confusions
  - `src/wakeword_workbench/negatives/synthetic_generator.py` - generate_synthetic_negatives
  - `src/wakeword_workbench/logging_config.py` - get_logger usage

  **Acceptance Criteria:**
  - [ ] DatasetGenerator class implemented
  - [ ] All methods have type hints
  - [ ] Logging added throughout
  - [ ] Docstrings follow Google style

  **QA Scenarios:**
  ```
  Scenario: DatasetGenerator can be instantiated
    Tool: Bash
    Steps:
      1. python -c "from wakeword_workbench.dataset.generator import DatasetGenerator; print('OK')"
    Expected: "OK"
  ```

  **Commit**: YES
  - Message: `feat(dataset): implement DatasetGenerator orchestrator`
  - Files: `src/wakeword_workbench/dataset/generator.py`

---

- [x] 5. Write unit tests for DatasetGenerator

  **What to do:**
  - Create `tests/test_dataset_generator.py`
  - Test DatasetGenerator instantiation
  - Test generate() with mock config
  - Test error handling
  - Test GenerationResult structure

  **Must NOT do:**
  - Don't test external components (mock them)
  - Don't require actual TTS synthesis in tests

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: []
  - **Justification**: Standard pytest unit tests

  **Parallelization:**
  - **Can Run In Parallel**: YES (after Task 4)
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: Task 4

  **References:**
  - `tests/conftest.py` - Existing fixtures
  - `tests/test_config.py` - Example test patterns

  **Acceptance Criteria:**
  - [ ] Test file created
  - [ ] `uv run pytest tests/test_dataset_generator.py -v` passes

  **QA Scenarios:**
  ```
  Scenario: Unit tests pass
    Tool: Bash
    Steps:
      1. uv run pytest tests/test_dataset_generator.py -v
    Expected: All tests pass
  ```

  **Commit**: YES
  - Message: `test(dataset): add unit tests for DatasetGenerator`
  - Files: `tests/test_dataset_generator.py`

---

- [x] 6. Wire CLI run command to DatasetGenerator

  **What to do:**
  - Replace stub in `cli.py` line 106-110
  - Instantiate DatasetGenerator with loaded config
  - Call generator.generate()
  - Handle GeneratorError with user-friendly messages
  - Add progress indication

  **Must NOT do:**
  - Don't modify other CLI commands
  - Don't change CLI interface/arguments

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: []
  - **Justification**: Simple wiring task

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on Task 4)
  - **Parallel Group**: Wave 3
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Task 4

  **References:**
  - `src/wakeword_workbench/cli.py` - Current run_command implementation
  - `src/wakeword_workbench/dataset/generator.py` - DatasetGenerator (from Task 4)

  **Acceptance Criteria:**
  - [ ] CLI run command uses DatasetGenerator
  - [ ] Error handling implemented
  - [ ] Progress indication added

  **QA Scenarios:**
  ```
  Scenario: CLI imports successfully
    Tool: Bash
    Steps:
      1. uv run wakeword-workbench --help
    Expected: Help displays without import errors
  ```

  **Commit**: YES
  - Message: `feat(cli): wire run command to DatasetGenerator`
  - Files: `src/wakeword_workbench/cli.py`

---

- [ ] 7. Add progress bars and user feedback

  **What to do:**
  - Add Rich progress bars for each pipeline stage
  - Show counts: positives generated, negatives generated, splits
  - Add success/failure messages
  - Use Rich Panels for final summary

  **Must NOT do:**
  - Don't clutter output with debug info
  - Don't use plain print() (use structlog)

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: []
  - **Justification**: UI/UX enhancement

  **Parallelization:**
  - **Can Run In Parallel**: YES (after Task 6)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 6

  **References:**
  - `src/wakeword_workbench/cli.py` - Existing Rich usage
  - Rich documentation for Progress and Panel

  **Acceptance Criteria:**
  - [ ] Progress bars visible during generation
  - [ ] Final summary displayed

  **QA Scenarios:**
  ```
  Scenario: Progress displays
    Tool: Bash
    Steps:
      1. Run CLI command and verify output format
    Expected: Progress bars and final summary visible
  ```

  **Commit**: YES
  - Message: `feat(cli): add progress bars and user feedback`
  - Files: `src/wakeword_workbench/cli.py`

---

- [x] 8. Fix notebook dependency cells

  **What to do:**
  - Update Section 1 cells to handle uv environment
  - Fix pip install to work with system-managed Python
  - Add detection for already-installed packages
  - Update verification cell

  **Must NOT do:**
  - Don't require manual pip installs if using uv
  - Don't break existing notebook structure

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: []
  - **Justification**: Notebook cell updates

  **Parallelization:**
  - **Can Run In Parallel**: YES (with Wave 1)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References:**
  - `notebooks/training-with-microwakeword.ipynb` - Section 1

  **Acceptance Criteria:**
  - [ ] Notebook cells execute without "pip install" errors
  - [ ] Packages detected if already installed

  **QA Scenarios:**
  ```
  Scenario: Notebook Section 1 runs
    Tool: Bash
    Steps:
      1. jupyter nbconvert --to notebook --execute notebooks/training-with-microwakeword.ipynb --stdout | head -100
    Expected: Section 1 completes without pip errors
  ```

  **Commit**: YES
  - Message: `fix(notebook): update dependency installation cells`
  - Files: `notebooks/training-with-microwakeword.ipynb`

---

- [x] 9. Update notebook Section 2 for new workflow

  **What to do:**
  - Update config creation cell to match new config schema
  - Update validation cell
  - Update dataset generation cell to use CLI or API
  - Update verification cell to check correct output structure

  **Must NOT do:**
  - Don't change Sections 3-9 (feature extraction onwards)
  - Don't break existing flow

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: []
  - **Justification**: Notebook updates

  **Parallelization:**
  - **Can Run In Parallel**: YES (after Task 4)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 14
  - **Blocked By**: Task 4

  **References:**
  - `notebooks/training-with-microwakeword.ipynb` - Section 2
  - `src/wakeword_workbench/config.py` - Config schema

  **Acceptance Criteria:**
  - [ ] Section 2 cells execute successfully
  - [ ] Output structure matches expectations

  **QA Scenarios:**
  ```
  Scenario: Notebook generates dataset
    Tool: Bash
    Steps:
      1. Run notebook Section 2
      2. Verify output directory exists
    Expected: ./output/microwakeword/ created with audio/ and manifests
  ```

  **Commit**: YES
  - Message: `fix(notebook): update Section 2 for new DatasetGenerator`
  - Files: `notebooks/training-with-microwakeword.ipynb`

---

- [x] 10. Create test config for "hey vera"

  **What to do:**
  - Create `test_hey_vera_config.yaml`
  - Use small sample counts for quick testing (50 positives, 100 negatives)
  - Configure for kokoro TTS backend
  - Set output to ./output/test_microwakeword

  **Must NOT do:**
  - Don't use large sample counts (slows testing)
  - Don't use piper if kokoro is available

  **Recommended Agent Profile:**
  - **Category**: `quick`
  - **Skills**: []
  - **Justification**: Config file creation

  **Parallelization:**
  - **Can Run In Parallel**: YES (with Wave 1)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 14
  - **Blocked By**: None

  **References:**
  - `examples/` - Existing config examples
  - `src/wakeword_workbench/config.py` - Config schema

  **Acceptance Criteria:**
  - [ ] Config file created
  - [ ] Config validates: `uv run wakeword-workbench validate test_hey_vera_config.yaml`

  **QA Scenarios:**
  ```
  Scenario: Test config validates
    Tool: Bash
    Steps:
      1. uv run wakeword-workbench validate test_hey_vera_config.yaml
    Expected: "Config validation passed"
  ```

  **Commit**: YES
  - Message: `test: add test config for hey vera wake word`
  - Files: `test_hey_vera_config.yaml`

---

- [ ] 11. Run end-to-end integration test

  **What to do:**
  - Run: `uv run wakeword-workbench run test_hey_vera_config.yaml`
  - Verify output directory structure
  - Verify manifests have correct entries
  - Verify audio files exist
  - Clean up test output

  **Must NOT do:**
  - Don't skip verification steps
  - Don't leave test output in repo

  **Recommended Agent Profile:**
  - **Category**: `unspecified-high`
  - **Skills**: []
  - **Justification**: Integration testing, may need debugging

  **Parallelization:**
  - **Can Run In Parallel**: NO (depends on Waves 2, 3, 4)
  - **Parallel Group**: Wave 4
  - **Blocks**: Tasks F1, F2
  - **Blocked By**: Tasks 7, 10, 12, 13

  **References:**
  - `test_hey_vera_config.yaml` - Test config
  - Expected output structure from design doc

  **Acceptance Criteria:**
  - [ ] Pipeline completes with exit code 0
  - [ ] Output directory has audio/ subdirectory
  - [ ] train.jsonl, val.jsonl, test.jsonl exist
  - [ ] Audio files count matches expected (50 pos + 100 neg = 150)

  **QA Scenarios:**
  ```
  Scenario: Full pipeline succeeds
    Tool: Bash
    Steps:
      1. uv run wakeword-workbench run test_hey_vera_config.yaml
      2. ls -la ./output/test_microwakeword/
      3. wc -l ./output/test_microwakeword/train.jsonl
    Expected: Exit 0, audio/ exists, manifests have entries
  ```

  **Commit**: NO (test artifacts)

---

## Final Verification Wave

- [ ] F1. Full pipeline verification

  Run complete end-to-end test:
  1. Clean environment: `rm -rf ./output/`
  2. Run pipeline: `uv run wakeword-workbench run test_hey_vera_config.yaml`
  3. Verify: `./output/test_microwakeword/` has correct structure
  4. Run notebook Section 2: `jupyter nbconvert --execute notebooks/training-with-microwakeword.ipynb`
  5. Document any issues

  **Expected**: All steps pass, no errors

- [ ] F2. Code quality check

  1. Run: `uv run ruff check src/`
  2. Run: `uv run mypy src/wakeword_workbench`
  3. Run: `uv run pytest tests/`
  4. Check all new files follow project conventions

  **Expected**: No lint errors, no type errors, all tests pass

---

## Commit Strategy

| Commit | Files | Message |
|--------|-------|---------|
| 1 | `pyproject.toml` | `chore(deps): add notebook training dependencies` |
| 2 | `src/wakeword_workbench/dataset/generator.py` | `feat(dataset): implement DatasetGenerator orchestrator` |
| 3 | `tests/test_dataset_generator.py` | `test(dataset): add unit tests for DatasetGenerator` |
| 4 | `src/wakeword_workbench/cli.py` | `feat(cli): wire run command to DatasetGenerator` |
| 5 | `src/wakeword_workbench/cli.py` | `feat(cli): add progress bars and user feedback` |
| 6 | `notebooks/training-with-microwakeword.ipynb` | `fix(notebook): update dependency installation cells` |
| 7 | `notebooks/training-with-microwakeword.ipynb` | `fix(notebook): update Section 2 for new DatasetGenerator` |
| 8 | `test_hey_vera_config.yaml` | `test: add test config for hey vera wake word` |

---

## Success Criteria

### Verification Commands

```bash
# 1. Dependencies install
uv sync --group dev
python -c "import tensorflow, pymicro_features, mmap_ninja, datasets, tqdm, matplotlib; print('OK')"
# Expected: "OK"

# 2. CLI works
uv run wakeword-workbench run test_hey_vera_config.yaml
# Expected: Exit 0, progress bars, success message

# 3. Output structure correct
ls ./output/test_microwakeword/
# Expected: audio/, train.jsonl, val.jsonl, test.jsonl

# 4. Notebook Section 2 runs
jupyter nbconvert --to notebook --execute notebooks/training-with-microwakeword.ipynb 2>&1 | grep -E "(Error|Exception|FAILED)" || echo "OK"
# Expected: "OK" (no errors)

# 5. Tests pass
uv run pytest tests/ -v
# Expected: All tests pass
```

### Final Checklist

- [ ] All "Must Have" deliverables present
- [ ] All "Must NOT Have" guardrails respected
- [ ] Code follows project conventions (type hints, structlog, docstrings)
- [ ] All tests pass
- [ ] Notebook runs end-to-end
- [ ] No lint or type errors
