# F1 Plan Compliance Audit - wakeword-workbench

Date: 2026-03-30
Auditor: OpenCode
Scope: `.sisyphus/plans/wakeword-workbench.md` Waves 1-7 against implementation in `src/wakeword_workbench/` and `tests/`.

## Summary

- Total planned implementation tasks (Waves 1-7): **35**
- Fully compliant tasks: **17**
- Partially compliant tasks: **16**
- Non-compliant tasks: **2**
- Plan checkbox status (Waves 1-7): **13/35 checked**
  - Wave 1: 6/6 checked
  - Wave 2: 0/5 checked
  - Wave 3: 0/5 checked
  - Wave 4: 0/6 checked
  - Wave 5: 0/5 checked
  - Wave 6: 4/5 checked
  - Wave 7: 3/3 checked

Assessment basis:
- Existence of required modules/files
- Presence of required APIs/functions/classes
- Conformance to key plan constraints ("Must NOT do", expected output formats, interface compatibility)
- Repository-wide `TODO|FIXME` scan: **no matches**

## Wave-by-Wave Compliance

## Wave 1 - Foundation (Tasks 1-6)

1. Task 1 (Scaffolding): **Compliant**
   - Evidence: `pyproject.toml`, `.gitignore`, `README.md`
   - Dependencies and optional extras present.

2. Task 2 (Directory structure): **Partially compliant**
   - Expected skeleton files missing from original plan names:
     - Missing: `src/wakeword_workbench/augment/transforms.py`
     - Missing: `src/wakeword_workbench/negatives/generator.py`
     - Missing: `src/wakeword_workbench/eval/metrics.py`
   - Alternative module layout exists and is richer.

3. Task 3 (Config system): **Compliant**
   - Evidence: `src/wakeword_workbench/config.py`
   - Dataclass-based config and YAML loading present with validation.

4. Task 4 (CLI entry point): **Compliant**
   - Evidence: `src/wakeword_workbench/cli.py`
   - `run`, `validate`, `--version`, verbosity flags, exit codes implemented.

5. Task 5 (Test infrastructure): **Partially compliant**
   - Evidence: `tests/conftest.py`, `tests/test_config.py`, `tests/test_cli.py`, pytest config in `pyproject.toml`
   - Missing expected utility file: `tests/utils.py`.

6. Task 6 (Logging system): **Partially compliant**
   - Evidence: `src/wakeword_workbench/logging_config.py`
   - Structlog exists, but plan-level expectations like broad module usage + explicit rotation behavior are not consistently wired through all modules.

Wave 1 result: **3 compliant, 3 partial, 0 non-compliant**

## Wave 2 - TTS Backends (Tasks 7-11)

7. Task 7 (TTS ABC/factory): **Compliant**
   - Evidence: `src/wakeword_workbench/tts/base.py`

8. Task 8 (Kokoro backend): **Compliant**
   - Evidence: `src/wakeword_workbench/tts/kokoro_backend.py`
   - Optional import, resampling to 16kHz, normalization.

9. Task 9 (Piper backend): **Partially compliant**
   - Evidence: `src/wakeword_workbench/tts/piper_backend.py`
   - Implemented, but no model auto-download/cache logic; constructor requires `model_path`, which conflicts with registry instantiation pattern.

10. Task 10 (Registry/discovery): **Partially compliant**
   - Evidence: `src/wakeword_workbench/tts/registry.py`, `src/wakeword_workbench/tts/__init__.py`
   - Registry exists; however backend list is still static (`_BACKEND_MODULES`), and entry-point based user backend discovery is not implemented.

11. Task 11 (TTS caching): **Partially compliant**
   - Evidence: `src/wakeword_workbench/tts/cache.py`
   - Cache module exists with LRU behavior and clear command in CLI, but caching is not integrated into synthesis flow in backend implementations.

Wave 2 result: **2 compliant, 3 partial, 0 non-compliant**

## Wave 3 - Dataset Generation (Tasks 12-16)

12. Task 12 (Phrase variants): **Partially compliant**
   - Evidence: `src/wakeword_workbench/dataset/phrase_variants.py`
   - Strong variant generation exists, but rules are hardcoded (plan requested configurable rule system).

13. Task 13 (Positive generator): **Partially compliant**
   - Evidence: `src/wakeword_workbench/dataset/positive_generator.py`
   - Works and emits JSONL + WAV, but manifest field names differ from plan wording (`text`/`voice` vs `transcript`/`speaker_id`).

14. Task 14 (Negative confusion phrases): **Compliant**
   - Evidence: `src/wakeword_workbench/negatives/phrase_generator.py`

15. Task 15 (Synthetic negatives): **Partially compliant**
   - Evidence: `src/wakeword_workbench/negatives/synthetic_generator.py`
   - Generator exists, but explicit positive-length-distribution matching is not implemented as described.

16. Task 16 (Metadata JSONL): **Partially compliant**
   - Evidence: `src/wakeword_workbench/dataset/metadata.py`
   - Append/load/validate are present; merge exists but does not perform dedup handling per plan expectation.

Wave 3 result: **1 compliant, 4 partial, 0 non-compliant**

## Wave 4 - Augmentation (Tasks 17-22)

17. Task 17 (Audio loader/resampler): **Compliant**
   - Evidence: `src/wakeword_workbench/augment/audio_loader.py`

18. Task 18 (Noise augmentation): **Compliant**
   - Evidence: `src/wakeword_workbench/augment/noise.py`

19. Task 19 (Reverb): **Partially compliant**
   - Evidence: `src/wakeword_workbench/augment/reverb.py`
   - Reverb implementation exists; RT60 range configurability from plan is not clearly implemented.

20. Task 20 (Gain/clipping): **Compliant**
   - Evidence: `src/wakeword_workbench/augment/gain.py`

21. Task 21 (Silence padding/transforms): **Non-compliant**
   - Evidence: `src/wakeword_workbench/augment/padding.py`
   - Plan says "Do NOT crop long audio (pad only)", but `FixedSizeClip` currently crops when audio is longer than target.

22. Task 22 (Pipeline composer): **Compliant**
   - Evidence: `src/wakeword_workbench/augment/pipeline.py`

Wave 4 result: **4 compliant, 1 partial, 1 non-compliant**

## Wave 5 - Dataset Builder and Export (Tasks 23-27)

23. Task 23 (Dataset merger): **Partially compliant**
   - Evidence: `src/wakeword_workbench/dataset/merger.py`
   - Merge + collision checks exist; deterministic behavior is incomplete in ratio sampling path.

24. Task 24 (Leak-proof splitter): **Partially compliant**
   - Evidence: `src/wakeword_workbench/dataset/splitter.py`
   - Splitter exists and does speaker grouping; `by` parameter is effectively ignored (always uses voice field behavior).

25. Task 25 (microWakeWord exporter): **Compliant**
   - Evidence: `src/wakeword_workbench/export/microwakeword.py`

26. Task 26 (openWakeWord exporter): **Compliant**
   - Evidence: `src/wakeword_workbench/export/openwakeword.py`

27. Task 27 (Export validation): **Non-compliant**
   - Evidence: `src/wakeword_workbench/export/validator.py`
   - Validator expects file naming/shape conventions (`features.npy`, `X.npy`, etc.) that do not match exporter outputs (`train_data.mmap`, `X_train.npy`, `y_train.npy`).

Wave 5 result: **2 compliant, 2 partial, 1 non-compliant**

## Wave 6 - Evaluation (Tasks 28-32)

28. Task 28 (FAR): **Compliant**
   - Evidence: `src/wakeword_workbench/eval/far.py`

29. Task 29 (FRR): **Compliant**
   - Evidence: `src/wakeword_workbench/eval/frr.py`

30. Task 30 (Threshold sweep): **Partially compliant**
   - Evidence: `src/wakeword_workbench/eval/threshold_sweep.py`
   - Implementation exists, but FRR call argument order is reversed (`calculate_frr(ground_truth, predictions)`), which invalidates metric correctness.

31. Task 31 (ROC): **Compliant**
   - Evidence: `src/wakeword_workbench/eval/roc.py`

32. Task 32 (Evaluation report): **Partially compliant**
   - Evidence: `src/wakeword_workbench/eval/report.py`
   - Multi-format reporting and EER are implemented, but FRR call argument order is also reversed in report calculations.

Wave 6 result: **3 compliant, 2 partial, 0 non-compliant**

## Wave 7 - Hard Negative Mining (Tasks 33-35)

33. Task 33 (Long audio processor): **Compliant**
   - Evidence: `src/wakeword_workbench/mining/long_audio.py`

34. Task 34 (False positive extractor): **Partially compliant**
   - Evidence: `src/wakeword_workbench/mining/extractor.py`
   - Extraction pipeline is implemented, but does not include explicit true-positive exclusion logic from labels/transcripts (plan says do not extract true positives).

35. Task 35 (Merge-back): **Compliant**
   - Evidence: `src/wakeword_workbench/mining/merge_back.py`

Wave 7 result: **2 compliant, 1 partial, 0 non-compliant**

## Completed Plan Tasks Verification

Tasks marked complete in the plan file: 1-6, 29-32, 33-35.

- Implementations exist for all checked tasks: **13/13 present**.
- However, checked tasks with compliance concerns:
  - Task 30 (metric argument order)
  - Task 32 (metric argument order)

## Missing Features

- Missing expected files from original Wave 1 skeleton naming:
  - `src/wakeword_workbench/augment/transforms.py`
  - `src/wakeword_workbench/negatives/generator.py`
  - `src/wakeword_workbench/eval/metrics.py`
  - `tests/utils.py`
- `examples/` exists but no sample YAML configs currently present.
- TTS cache not wired into Kokoro/Piper synthesis execution path.
- Export validation module not aligned to actual exporter file naming conventions.

## Issues Found

1. Evaluation metric integration bug:
   - `calculate_frr` called with reversed arguments in:
     - `src/wakeword_workbench/eval/threshold_sweep.py`
     - `src/wakeword_workbench/eval/report.py`

2. Plan constraint violation in padding:
   - `src/wakeword_workbench/augment/padding.py` crops long audio despite Wave 4 Task 21 "pad only" requirement.

3. Registry/backends mismatch risk:
   - `PiperBackend` requires `model_path` constructor arg while generic registry/factory calls zero-arg constructors.

4. Validator/export mismatch:
   - `src/wakeword_workbench/export/validator.py` expects filenames/shapes inconsistent with exporters.

## Final Verdict

**NON-COMPLIANT**

Rationale: substantial implementation coverage exists, but multiple plan-critical mismatches remain (including two non-compliant tasks, several partial tasks, and API/format inconsistencies that break strict plan adherence).
