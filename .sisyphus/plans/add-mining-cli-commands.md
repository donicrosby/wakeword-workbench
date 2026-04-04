# Add Mining CLI Commands

## TL;DR

> **Quick Summary**: Add two missing CLI subcommands (`mine` and `merge`) to expose the existing hard negative mining functionality. The backend code exists in `mining/` directory but isn't wired to the CLI.
> 
> **Deliverables**: 
> - `wakeword-workbench mine` command with ONNX model loading
> - `wakeword-workbench merge` command for manifest merging
> - Model loader utility for ONNX runtime
> - Comprehensive tests for both commands
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - Sequential dependency (model loader → mine command → merge command)
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5

---

## Context

### Original Request
Add missing CLI subcommands referenced in `examples/mining_config.yaml`:
- `wakeword-workbench mine` - Process long audio for false positives
- `wakeword-workbench merge` - Merge mined negatives back to training set

### Metis Review Findings (Addressed)
**Major Gap Identified**: Model loading - CLI passes a path but backend expects a callable (`Callable[[np.ndarray], float]`). Added Task 5 for ONNX model loader utility.

**Scope Decisions Made**:
- ✅ Wildcard support: Yes, use shell glob expansion (`*.wav`)
- ✅ Error handling: Fail-fast on first error (matches existing CLI pattern)
- ✅ Manifest format: Reuse existing `Manifest` dataclass from `dataset/metadata.py`
- ✅ Model format: ONNX only for MVP (can extend later)
- ✅ Threshold handling: Validate 0.0-1.0 range, reject edge cases

### Key Principle
> Backend functionality exists. This is CLI integration work, not feature development.

---

## Work Objectives

### Core Objective
Expose the existing hard negative mining functionality through the CLI by adding `mine` and `merge` subcommands that follow the established CLI patterns.

### Concrete Deliverables
1. **Model Loader Utility** (`src/wakeword_workbench/mining/model_loader.py`)
   - Load ONNX models and return callable for inference
   - Handle missing onnxruntime gracefully
   - Validate model format

2. **Mine Command** (`src/wakeword_workbench/cli.py` - add `@app.command(name="mine")`)
   - Process long audio files for false positives
   - Extract clips above threshold
   - Save to output directory with manifest

3. **Merge Command** (`src/wakeword_workbench/cli.py` - add `@app.command(name="merge")`)
   - Merge mined negatives into training manifest
   - Create backup of original
   - Validate merged result

4. **Tests** (`tests/test_cli_mining.py`)
   - Test mine command with mock model
   - Test merge command with test manifests
   - Test error handling

### Definition of Done
- [ ] `wakeword-workbench mine --help` shows usage
- [ ] `wakeword-workbench merge --help` shows usage
- [ ] Mine command successfully extracts false positives from test audio
- [ ] Merge command successfully merges manifests
- [ ] All tests pass (`uv run pytest tests/test_cli_mining.py -v`)

### Must Have
- ONNX model loading support
- Progress indication during processing
- Proper error messages for common failures
- JSONL manifest output
- Backup creation on merge

### Must NOT Have (Guardrails)
- NO GUI or interactive visualization
- NO real-time audio streaming support
- NO model training during mining
- NO automatic threshold optimization
- NO cloud storage integration
- NO config-file driven mining (keep CLI args only for MVP)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest configured)
- **Automated tests**: YES (Tests after implementation)
- **Framework**: pytest
- **If TDD**: Tests written after implementation for integration commands

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **CLI Commands**: Use Bash - Run commands, capture output, verify exit codes
- **Model Loading**: Use Bash (bun/node REPL) - Import, test loading
- **Manifest Merging**: Use Bash - Verify JSONL files, check backups

---

## Execution Strategy

### Sequential Execution (Dependencies Required)

This work has inherent dependencies - must be sequential:

```
Wave 1 (Foundation):
├── Task 1: Model loader utility [quick]
│   └── Creates: mining/model_loader.py
│   └── Tests: test_model_loader.py
│   └── Blocks: Task 2
│
Wave 2 (Mine Command):
├── Task 2: Mine CLI command [unspecified-high]
│   └── Modifies: cli.py (adds @app.command(name="mine"))
│   └── Tests: test_cli_mine.py
│   └── Depends: Task 1
│   └── Blocks: Task 3
│
Wave 3 (Merge Command):
├── Task 3: Merge CLI command [unspecified-high]
│   └── Modifies: cli.py (adds @app.command(name="merge"))
│   └── Tests: test_cli_merge.py
│   └── Depends: Task 2 (follows same patterns)
│   └── Blocks: Task 4
│
Wave 4 (Integration):
├── Task 4: Integration tests [quick]
│   └── Creates: tests/test_cli_mining.py
│   └── Tests: End-to-end mine → merge workflow
│   └── Depends: Task 2, Task 3
│   └── Blocks: Task 5
│
Wave FINAL (Verification):
├── Task F1: Code quality review (unspecified-high)
├── Task F2: Manual QA (unspecified-high)
└── Task F3: Scope fidelity check (deep)
```

**Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → F1-F3
**Parallel Speedup**: 0% (purely sequential due to dependencies)
**Max Concurrent**: 1

### Agent Dispatch Summary

- **Wave 1**: **1** — Task 1 → `quick`
- **Wave 2**: **1** — Task 2 → `unspecified-high`
- **Wave 3**: **1** — Task 3 → `unspecified-high`
- **Wave 4**: **1** — Task 4 → `quick`
- **Wave FINAL**: **3** — F1 → `unspecified-high`, F2 → `unspecified-high`, F3 → `deep`

---

## TODOs

- [x] 1. Model Loader Utility

  **What to do**:
  Create `src/wakeword_workbench/mining/model_loader.py` with:
  - `load_onnx_model(path: Path) -> Callable[[np.ndarray], float]` function
  - Input validation (file exists, readable)
  - Graceful handling of missing onnxruntime (informative error)
  - Returns callable that takes audio array and returns confidence score

  **Must NOT do**:
  - Don't support other model formats (TensorFlow, PyTorch) - ONNX only for MVP
  - Don't implement model caching (add later if needed)
  - Don't add model validation beyond basic ONNX loading

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single utility function, well-defined scope, clear requirements
  - **Skills**: []
    - No special skills needed - standard Python development

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2 (Mine command needs this utility)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `src/wakeword_workbench/mining/long_audio.py:36` - `process_long_audio()` expects `model_fn: Callable[[np.ndarray], float]`
  - `src/wakeword_workbench/mining/extractor.py:41` - `extract_false_positives()` uses the model callable
  - `src/wakeword_workbench/cli.py:77-106` - Pattern for CLI command structure
  - Example ONNX loading: Standard onnxruntime `InferenceSession` pattern

  **WHY Each Reference Matters**:
  - `long_audio.py:36`: Shows the expected callable signature - takes numpy array, returns float confidence
  - `extractor.py:41`: Shows how the model callable is used in the extraction pipeline
  - `cli.py:77-106`: Shows the CLI pattern to follow (typer decorator, progress spinner, error handling)

  **Acceptance Criteria**:
  - [ ] `load_onnx_model()` function exists and is importable
  - [ ] Returns callable that accepts `np.ndarray` and returns `float`
  - [ ] Raises clear error if onnxruntime not installed
  - [ ] Raises clear error if model file not found
  - [ ] Tests pass: `uv run pytest tests/test_model_loader.py -v`

  **QA Scenarios**:

  ```
  Scenario: Successfully load ONNX model
    Tool: Bash
    Preconditions: Test ONNX model exists at tests/fixtures/dummy_model.onnx
    Steps:
      1. Run: uv run python -c "from wakeword_workbench.mining.model_loader import load_onnx_model; fn = load_onnx_model('tests/fixtures/dummy_model.onnx'); print(fn.__call__)"
      2. Assert: Exit code 0, output shows callable object
    Expected Result: Model loads without error, returns callable function
    Failure Indicators: ImportError, FileNotFoundError, or non-callable returned
    Evidence: .sisyphus/evidence/task-1-load-model.txt

  Scenario: Handle missing onnxruntime gracefully
    Tool: Bash
    Preconditions: onnxruntime not installed (or mock unavailable)
    Steps:
      1. Run: uv run python -c "from wakeword_workbench.mining.model_loader import load_onnx_model" in environment without onnxruntime
      2. Assert: Clear error message about missing dependency
    Expected Result: Informative error: "onnxruntime not installed. Install with: uv sync --extra onnx"
    Evidence: .sisyphus/evidence/task-1-missing-runtime.txt
  ```

  **Evidence to Capture**:
  - [ ] Screenshot or text capture of successful model load
  - [ ] Error message for missing onnxruntime

  **Commit**: YES
  - Message: `feat(mining): add ONNX model loader utility`
  - Files: `src/wakeword_workbench/mining/model_loader.py`, `tests/test_model_loader.py`
  - Pre-commit: `uv run pytest tests/test_model_loader.py -v`

---

- [x] 2. Mine CLI Command

  **What to do**:
  Add `@app.command(name="mine")` to `cli.py` with:
  - Arguments: `--model`, `--audio`, `--threshold`, `--output`
  - Support wildcard expansion for `--audio` (`*.wav`)
  - Use model_loader to load ONNX model
  - Call `process_long_audio()` from mining module
  - Call `extract_false_positives()` to save clips
  - Progress indication with rich Progress spinner
  - Exit codes: 0=success, 1=error, 2=config error

  **Must NOT do**:
  - Don't add config file support (keep CLI args only)
  - Don't support real-time streaming
  - Don't add visualization/plotting
  - Don't validate model architecture beyond loading

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: CLI command with multiple arguments, error handling, progress indication, and integration with existing mining modules
  - **Skills**: []
    - No special skills needed - follows established patterns

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 3, Task 4
  - **Blocked By**: Task 1 (needs model loader)

  **References**:
  - `src/wakeword_workbench/cli.py:77-106` - Pattern: `run_command()` - args, validation, progress, logging, exit
  - `src/wakeword_workbench/cli.py:109-138` - Pattern: `validate_command()` - file validation, error handling
  - `src/wakeword_workbench/mining/long_audio.py:36` - `process_long_audio()` signature and usage
  - `src/wakeword_workbench/mining/extractor.py:41` - `extract_false_positives()` signature
  - `src/wakeword_workbench/dataset/metadata.py:50-192` - `Manifest` class for output

  **WHY Each Reference Matters**:
  - `cli.py:77-106`: Shows complete CLI command pattern - args, Progress spinner, logging, exit codes
  - `cli.py:109-138`: Shows file validation and ConfigError handling pattern
  - `long_audio.py:36`: Shows expected parameters: model_fn, audio_path, threshold, etc.
  - `extractor.py:41`: Shows how to call extraction with predictions and save clips
  - `metadata.py:50-192`: Shows Manifest format for saving results

  **Acceptance Criteria**:
  - [ ] `wakeword-workbench mine --help` shows usage
  - [ ] Command accepts `--model`, `--audio`, `--threshold`, `--output` arguments
  - [ ] Wildcards expand properly (e.g., `--audio "*.wav"`)
  - [ ] Progress spinner shows during processing
  - [ ] Creates output directory if not exists
  - [ ] Generates manifest.jsonl in output directory
  - [ ] Exit code 0 on success, 1 on error, 2 on config error
  - [ ] Tests pass: `uv run pytest tests/test_cli_mine.py -v`

  **QA Scenarios**:

  ```
  Scenario: Mine command help shows correctly
    Tool: Bash
    Preconditions: CLI is importable
    Steps:
      1. Run: uv run wakeword-workbench mine --help
      2. Assert: Help text shows with --model, --audio, --threshold, --output options
    Expected Result: Clear help output with all expected arguments
    Failure Indicators: Command not found, missing arguments in help
    Evidence: .sisyphus/evidence/task-2-mine-help.txt

  Scenario: Successfully mine single audio file
    Tool: Bash
    Preconditions: Test ONNX model exists, test audio file exists
    Steps:
      1. Run: uv run wakeword-workbench mine --model tests/fixtures/dummy_model.onnx --audio tests/fixtures/test_audio.wav --threshold 0.7 --output ./test_mined/
      2. Assert: Exit code 0, progress shown, output directory created
      3. Verify: ls ./test_mined/manifest.jsonl exists
    Expected Result: Clips extracted and manifest created
    Evidence: .sisyphus/evidence/task-2-mine-single.txt

  Scenario: Handle wildcard expansion
    Tool: Bash
    Preconditions: Multiple test audio files exist
    Steps:
      1. Run: uv run wakeword-workbench mine --model model.onnx --audio "tests/fixtures/*.wav" --threshold 0.7 --output ./test_wildcard/
      2. Assert: All matching files processed
    Expected Result: Multiple files processed, single manifest with all results
    Evidence: .sisyphus/evidence/task-2-wildcard.txt

  Scenario: Invalid threshold rejected
    Tool: Bash
    Preconditions: Valid model and audio
    Steps:
      1. Run: uv run wakeword-workbench mine --model model.onnx --audio audio.wav --threshold 1.5 --output ./out/
      2. Assert: Exit code 2 (config error), error message about threshold range
    Expected Result: Clear error: "threshold must be between 0.0 and 1.0"
    Evidence: .sisyphus/evidence/task-2-invalid-threshold.txt
  ```

  **Evidence to Capture**:
  - [ ] Help output screenshot
  - [ ] Successful mining output with progress
  - [ ] Manifest file content
  - [ ] Error output for invalid threshold

  **Commit**: YES
  - Message: `feat(cli): add mine command for hard negative mining`
  - Files: `src/wakeword_workbench/cli.py` (add mine command)
  - Pre-commit: `uv run pytest tests/test_cli_mine.py -v`

---

- [x] 3. Merge CLI Command

  **What to do**:
  Add `@app.command(name="merge")` to `cli.py` with:
  - Arguments: `--source`, `--target`, `--backup` (optional)
  - Use `add_to_training()` from `merge_back.py`
  - Create backup of target before merge (if --backup flag)
  - Validate both manifests exist and are valid JSONL
  - Show statistics: source entries, target entries, merged total
  - Progress indication
  - Exit codes: 0=success, 1=error, 2=config error

  **Must NOT do**:
  - Don't modify source manifest (read-only)
  - Don't deduplicate entries (keep all)
  - Don't validate audio file existence (assume caller did)
  - Don't support config file driven merging

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: CLI command with file operations, backup logic, manifest manipulation
  - **Skills**: []
    - No special skills needed - standard file I/O and CLI patterns

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 4
  - **Blocked By**: Task 2 (follows same CLI patterns)

  **References**:
  - `src/wakeword_workbench/cli.py:77-106` - Pattern: `run_command()` - args, validation, progress
  - `src/wakeword_workbench/mining/merge_back.py:34` - `add_to_training()` function signature
  - `src/wakeword_workbench/dataset/metadata.py:134` - `Manifest.load()` for reading manifests
  - `src/wakeword_workbench/dataset/metadata.py:101` - `Manifest.save()` for writing manifests

  **WHY Each Reference Matters**:
  - `cli.py:77-106`: CLI command pattern to follow
  - `merge_back.py:34`: Shows the core merge logic already implemented
  - `metadata.py:134`: Shows how to load existing manifests
  - `metadata.py:101`: Shows how to save merged manifests

  **Acceptance Criteria**:
  - [ ] `wakeword-workbench merge --help` shows usage
  - [ ] Command accepts `--source`, `--target`, `--backup` arguments
  - [ ] Shows statistics before/after merge
  - [ ] Creates backup if `--backup` flag provided
  - [ ] Validates manifests are valid JSONL
  - [ ] Exit code 0 on success, 1 on error, 2 on config error
  - [ ] Tests pass: `uv run pytest tests/test_cli_merge.py -v`

  **QA Scenarios**:

  ```
  Scenario: Merge command help shows correctly
    Tool: Bash
    Preconditions: CLI is importable
    Steps:
      1. Run: uv run wakeword-workbench merge --help
      2. Assert: Help text shows with --source, --target, --backup options
    Expected Result: Clear help output with all expected arguments
    Evidence: .sisyphus/evidence/task-3-merge-help.txt

  Scenario: Successfully merge manifests
    Tool: Bash
    Preconditions: Source and target manifest files exist
    Steps:
      1. Run: uv run wakeword-workbench merge --source source.jsonl --target target.jsonl
      2. Assert: Exit code 0, statistics shown, target updated
      3. Verify: Target manifest now contains source entries
    Expected Result: Manifests merged, statistics displayed
    Evidence: .sisyphus/evidence/task-3-merge-success.txt

  Scenario: Merge with backup creation
    Tool: Bash
    Preconditions: Target manifest exists
    Steps:
      1. Run: uv run wakeword-workbench merge --source source.jsonl --target target.jsonl --backup
      2. Assert: Backup file created (target.jsonl.backup or similar)
      3. Verify: Original target preserved in backup
    Expected Result: Backup created before modification
    Evidence: .sisyphus/evidence/task-3-merge-backup.txt

  Scenario: Handle missing source file
    Tool: Bash
    Preconditions: Source file doesn't exist
    Steps:
      1. Run: uv run wakeword-workbench merge --source nonexistent.jsonl --target target.jsonl
      2. Assert: Exit code 2, clear error about missing source
    Expected Result: Informative error message
    Evidence: .sisyphus/evidence/task-3-merge-missing.txt
  ```

  **Evidence to Capture**:
  - [ ] Help output screenshot
  - [ ] Merge output with statistics
  - [ ] Backup file listing
  - [ ] Error output for missing file

  **Commit**: YES
  - Message: `feat(cli): add merge command for hard negative integration`
  - Files: `src/wakeword_workbench/cli.py` (add merge command)
  - Pre-commit: `uv run pytest tests/test_cli_merge.py -v`

---

- [ ] 4. Integration Tests

  **What to do**:
  Create `tests/test_cli_mining.py` with end-to-end tests:
  - Test complete workflow: mine → merge
  - Test with real test fixtures (create minimal ONNX model and test audio)
  - Test error conditions (invalid model, missing files, invalid threshold)
  - Verify manifest integrity after merge
  - Test backup functionality

  **Must NOT do**:
  - Don't test model inference accuracy (out of scope)
  - Don't test audio processing quality (unit tests cover that)
  - Don't require large test files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Writing tests, straightforward assertions
  - **Skills**: []
    - Standard pytest patterns

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None (final implementation task)
  - **Blocked By**: Task 2, Task 3 (tests both commands)

  **References**:
  - `tests/conftest.py` - Existing fixtures pattern
  - `tests/test_cli.py` - CLI testing pattern with `typer.testing.CliRunner`
  - `src/wakeword_workbench/cli.py` - Commands being tested

  **Acceptance Criteria**:
  - [ ] Test file created: `tests/test_cli_mining.py`
  - [ ] All tests pass: `uv run pytest tests/test_cli_mining.py -v`
  - [ ] Coverage for mine command
  - [ ] Coverage for merge command
  - [ ] Coverage for end-to-end workflow
  - [ ] Coverage for error conditions

  **QA Scenarios**:

  ```
  Scenario: End-to-end mine → merge workflow
    Tool: Bash
    Preconditions: Test fixtures exist
    Steps:
      1. Run mine command on test audio
      2. Run merge command to integrate results
      3. Verify: Training manifest contains mined entries
    Expected Result: Complete workflow succeeds
    Evidence: .sisyphus/evidence/task-4-e2e.txt

  Scenario: All tests pass
    Tool: Bash
    Preconditions: Implementation complete
    Steps:
      1. Run: uv run pytest tests/test_cli_mining.py -v
      2. Assert: All tests pass, no failures
    Expected Result: 100% pass rate
    Evidence: .sisyphus/evidence/task-4-tests.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing all tests passing
  - [ ] Coverage report

  **Commit**: YES
  - Message: `test(cli): add integration tests for mining commands`
  - Files: `tests/test_cli_mining.py`, `tests/fixtures/` (minimal test files)
  - Pre-commit: `uv run pytest tests/test_cli_mining.py -v`

---

## Final Verification Wave

> 3 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Code Quality Review** — `unspecified-high`
  Run `tsc --noEmit` + linter + `bun test`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names (data/result/item/temp).
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F2. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (mine → merge workflow). Test edge cases: empty manifest, invalid JSONL, very long audio file.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F3. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

1. `feat(mining): add ONNX model loader utility`
2. `feat(cli): add mine command for hard negative mining`
3. `feat(cli): add merge command for hard negative integration`
4. `test(cli): add integration tests for mining commands`

---

## Success Criteria

### Verification Commands
```bash
# Verify commands exist
uv run wakeword-workbench mine --help
uv run wakeword-workbench merge --help

# Run all tests
uv run pytest tests/test_cli_mining.py -v

# Run full test suite
uv run pytest

# Check code quality
uv run ruff check src/wakeword_workbench/cli.py
uv run mypy src/wakeword_workbench/mining/
```

### Final Checklist
- [ ] All "Must Have" present (model loader, mine command, merge command, tests)
- [ ] All "Must NOT Have" absent (no GUI, no streaming, no config-driven mining)
- [ ] All tests pass
- [ ] Code quality checks pass (ruff, mypy)
- [ ] Example from mining_config.yaml works: `wakeword-workbench mine --model ./models/wakeword_v1.onnx --audio ./data/long_recordings/*.wav --threshold 0.7 --output ./mined_negatives/`
- [ ] Example from mining_config.yaml works: `wakeword-workbench merge --source ./mined_negatives/manifest.jsonl --target ./output/mining_example/train_manifest.jsonl --backup`
