# Fix Plan: wakeword-workbench Verification Issues

**Date:** 2026-03-30  
**Source:** Verification Reports F1, F2, F3, F4  
**Priority:** P0 (Critical), P1 (High), P2 (Medium), P3 (Low)

---

## Executive Summary

The verification identified **4 critical bugs**, **39 code style issues**, **30 type errors**, and **71% test coverage** (9% below target). This plan prioritizes fixes by impact and effort.

### Quick Wins (Do First)
1. Run `ruff check src/ --fix` → Auto-fixes 27 issues
2. Fix FRR argument order bug → Affects all evaluation metrics
3. Fix padding crop bug → Violates plan requirements

---

## P0 - Critical Bugs (Fix Immediately)

These bugs affect correctness or violate plan requirements.

### P0.1: FRR Argument Order Bug
**Issue:** `calculate_frr()` called with reversed arguments  
**Files:** 
- `src/wakeword_workbench/eval/threshold_sweep.py:66`
- `src/wakeword_workbench/eval/report.py:243`

**Current Code:**
```python
# WRONG - threshold_sweep.py:66
frr = calculate_frr(ground_truth, predictions, threshold=float(t))

# WRONG - report.py:243
frr = calculate_frr(ground_truth, predictions)
```

**Fix:**
```python
# CORRECT - must be (predictions, ground_truth, threshold)
frr = calculate_frr(predictions, ground_truth, threshold=float(t))
frr = calculate_frr(predictions, ground_truth)
```

**Impact:** All FRR calculations are currently wrong, affecting:
- Threshold sweep results
- Evaluation reports
- EER calculations
- ROC curves

**Verification:** After fix, run tests in `tests/test_far.py`

---

### P0.2: Padding Cropping Instead of Padding
**Issue:** `FixedSizeClip` crops long audio, violating Task 21 "pad only" requirement  
**File:** `src/wakeword_workbench/augment/padding.py`

**Fix Strategy:**
1. Add `mode="pad_only"` parameter to `FixedSizeClip`
2. When mode is "pad_only", truncate/pad logic should only pad, never crop
3. Consider audio stretching or multiple windows for very long audio

**Implementation:**
```python
class FixedSizeClip:
    def __init__(self, target_samples: int, mode: str = "pad_or_crop"):
        """
        Args:
            target_samples: Target length in samples
            mode: "pad_or_crop" (default) or "pad_only" (never crop)
        """
        self.target_samples = target_samples
        self.mode = mode
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < self.target_samples:
            # Pad short audio
            return self._pad(audio)
        elif len(audio) > self.target_samples:
            if self.mode == "pad_only":
                # Don't crop - could stretch or return multiple windows
                return self._stretch_or_window(audio)
            else:
                # Original behavior: crop
                return audio[:self.target_samples]
        return audio
```

---

### P0.3: Export Validator Filename Mismatch
**Issue:** Validator expects wrong filenames vs actual exporter outputs  
**File:** `src/wakeword_workbench/export/validator.py`

**Current:** Validator expects `features.npy`, `X.npy`  
**Actual:** Exporters create `train_data.mmap`, `X_train.npy`, `y_train.npy`

**Fix:** Update validator to match actual exporter outputs:
- microWakeWord: `train_data.mmap`, `train_indices.npy`, `train_labels.npy`
- openWakeWord: `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`

---

### P0.4: PiperBackend Constructor Mismatch
**Issue:** `PiperBackend` requires `model_path` but registry calls zero-arg constructors  
**File:** `src/wakeword_workbench/tts/piper_backend.py`

**Fix Options:**
1. Make `model_path` optional with auto-download logic
2. Update registry to pass config to constructors
3. Add factory method that handles model downloading

**Recommended:** Option 1 - Add auto-download in `__init__` if model_path not provided

---

## P1 - High Priority (Code Quality)

### P1.1: Fix Auto-Correctable Ruff Errors
**Command:** `uv run ruff check src/ --fix`

**Fixes 27 issues:**
- Unused imports (F401): 7 errors
- Import sorting (I001): 4 errors
- Python upgrade rules (UP*): 10 errors
- Bugbear rules (B*): 5 errors
- F-string without placeholders (F541): 1 error

**Manual fixes needed after auto-fix:**
- F811 (redefinition): 1 error in `dataset/merger.py`
- F841 (unused variable): 2 errors
- E741 (ambiguous variable name): 1 error

---

### P1.2: Install Type Stubs
**Missing stubs causing 11 errors:**
```bash
uv add --dev types-PyYAML scipy-stubs
```

**Remaining mypy errors (19) are genuine type issues:**
- `no-any-return`: 6 errors - add explicit return types
- `assignment`: 2 errors - fix type mismatches
- `attr-defined`: 1 error - check attribute exists
- `arg-type`: 1 error - fix argument type

---

### P1.3: Fix Remaining Ruff Errors (Manual)

**F811 - Redefinition in merger.py:**
```python
# Line ~45-50, check for duplicate imports or variable assignments
```

**F841 - Unused variables:**
- `dataset/splitter.py`: Check for assigned but unused variables
- `mining/long_audio.py`: Same

**E741 - Ambiguous variable name:**
- `export/openwakeword.py`: Rename single-char variable like `l` or `I`

---

## P2 - Medium Priority (Test Coverage)

### P2.1: Add Mining Module Tests (0% → 80%)
**Files to test:**
- `src/wakeword_workbench/mining/long_audio.py` (97 lines, 0% coverage)
- `src/wakeword_workbench/mining/extractor.py` (58 lines, 0% coverage)
- `src/wakeword_workbench/mining/merge_back.py` (93 lines, 0% coverage)

**Test scenarios:**
1. **long_audio.py:**
   - Process short audio (< chunk size)
   - Process long audio with chunks
   - Window predictions accuracy
   - Edge cases: empty audio, exact chunk boundaries

2. **extractor.py:**
   - Extract above threshold
   - Cooldown deduplication
   - Overlapping detection handling
   - Output file naming

3. **merge_back.py:**
   - Valid manifest merge
   - Duplicate detection
   - Backup creation
   - Error handling (missing files)

---

### P2.2: Fix Positive Generator Tests (19% → 80%)
**9 failing tests in `tests/test_positive_generator.py`**

**Likely causes:**
- TTS backend mocking issues
- File system mocking problems
- Test isolation issues

**Fix approach:**
1. Check if tests need mocking of TTS backends
2. Verify temp directory handling
3. Check for shared state between tests

---

### P2.3: Fix Synthetic Generator Tests (23 failures)
**File:** `tests/test_synthetic_generator.py`

**Likely causes:**
- Missing word list data files
- Random seed handling
- File system operations not mocked

**Fix approach:**
1. Add mock word lists for testing
2. Ensure deterministic random seed usage
3. Use tmp_path fixture for file operations

---

### P2.4: Fix Gain Test Failure
**1 failure in `tests/test_gain.py::TestGainTransition::test_gain_applied`**

**Investigate:**
- Check gain calculation logic
- Verify transition curve implementation
- Check audio array shape handling

---

## P3 - Low Priority (Improvements)

### P3.1: Add Example Config Files
**Location:** `examples/` directory

**Create:**
- `examples/basic_config.yaml` - Simple wake word dataset
- `examples/augmented_config.yaml` - With noise/reverb
- `examples/mining_config.yaml` - Hard negative mining pipeline

---

### P3.2: Integrate TTS Cache
**Issue:** Cache module exists but not wired into backends

**Files to update:**
- `src/wakeword_workbench/tts/kokoro_backend.py`
- `src/wakeword_workbench/tts/piper_backend.py`

**Implementation:**
```python
# In synthesize() method:
from .cache import TTSCache

cache = TTSCache()
cached = cache.get(text, voice_id)
if cached:
    return cached

# ... generate audio ...

cache.set(text, voice_id, audio)
return audio
```

---

### P3.3: Add Logging Throughout
**Issue:** Logging config exists but not consistently used

**Add to:**
- All augmentation modules
- Dataset generators
- Export modules
- Mining modules

**Pattern:**
```python
import structlog

logger = structlog.get_logger(__name__)

# In functions:
logger.debug("Processing audio", samples=len(audio))
logger.info("Export complete", path=output_path)
```

---

### P3.4: Add Documentation
**Files to create:**
- `docs/usage.md` - Basic usage examples
- `docs/config.md` - Configuration reference
- `docs/api.md` - API documentation

---

## Implementation Order

### Phase 1: Critical Fixes (1-2 days)
1. [ ] P0.1 - Fix FRR argument order
2. [ ] P0.2 - Fix padding crop behavior
3. [ ] P0.3 - Fix validator filename mismatch
4. [ ] P0.4 - Fix PiperBackend constructor

### Phase 2: Code Quality (1 day)
5. [ ] P1.1 - Run ruff auto-fix
6. [ ] P1.2 - Install type stubs
7. [ ] P1.3 - Fix remaining ruff errors

### Phase 3: Test Coverage (2-3 days)
8. [ ] P2.1 - Add mining module tests
9. [ ] P2.2 - Fix positive generator tests
10. [ ] P2.3 - Fix synthetic generator tests
11. [ ] P2.4 - Fix gain test

### Phase 4: Polish (1 day)
12. [ ] P3.1 - Add example configs
13. [ ] P3.2 - Integrate TTS cache
14. [ ] P3.3 - Add logging
15. [ ] P3.4 - Add documentation

---

## Verification Checklist

After fixes, verify:
- [ ] `uv run ruff check src/` → 0 errors
- [ ] `uv run mypy src/wakeword_workbench` → < 10 errors (mostly stubs)
- [ ] `uv run pytest --cov=wakeword_workbench` → Coverage >= 80%
- [ ] `uv run pytest` → All tests pass
- [ ] FRR calculations verified correct
- [ ] Padding never crops (only pads)
- [ ] Export validation matches actual outputs

---

## Effort Estimate

| Phase | Duration | Tasks |
|-------|----------|-------|
| Phase 1 | 1-2 days | 4 critical bugs |
| Phase 2 | 1 day | Code quality |
| Phase 3 | 2-3 days | Test coverage |
| Phase 4 | 1 day | Polish |
| **Total** | **5-7 days** | **15 tasks** |

---

## Success Criteria

**Definition of Done for this Fix Plan:**
1. All P0 bugs fixed and verified
2. Ruff errors = 0
3. MyPy errors < 10 (genuine issues only)
4. Test coverage >= 80%
5. All tests passing
6. No plan compliance violations
7. Example configs available

**Target Quality Metrics:**
- Code style: 100% clean (ruff)
- Type safety: 95%+ (mypy)
- Test coverage: 80%+
- Test pass rate: 100%
- Plan compliance: 100%
