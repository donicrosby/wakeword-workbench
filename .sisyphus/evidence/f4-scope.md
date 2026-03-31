# F4 Scope Fidelity Report

Date: 2026-03-30
Plan reviewed: `.sisyphus/plans/wakeword-workbench.md`
Implementation reviewed: `src/wakeword_workbench/**/*.py`, `tests/**/*.py`, `pyproject.toml`, `README.md`, `.sisyphus/evidence/*`

## Must Have

1) YAML config-driven reproducibility
- Status: PRESENT
- Evidence: `src/wakeword_workbench/config.py` uses YAML loading/validation (`yaml.safe_load`) and strongly-typed config sections.

2) Optional TTS backends (graceful degradation)
- Status: PRESENT
- Evidence: `src/wakeword_workbench/tts/kokoro_backend.py` and `src/wakeword_workbench/tts/piper_backend.py` use optional imports and availability checks; `src/wakeword_workbench/tts/registry.py` filters available backends.

3) 16kHz/16-bit/mono audio throughout
- Status: PRESENT (minor caveat)
- Evidence:
  - 16kHz normalization/resampling in `src/wakeword_workbench/augment/audio_loader.py`, `src/wakeword_workbench/dataset/positive_generator.py`, `src/wakeword_workbench/tts/kokoro_backend.py`, and `src/wakeword_workbench/tts/piper_backend.py`.
  - Mono conversion in `src/wakeword_workbench/augment/audio_loader.py` and `src/wakeword_workbench/dataset/positive_generator.py`.
  - Explicit 16-bit PCM writing is implemented in `src/wakeword_workbench/augment/audio_loader.py` (`subtype="PCM_16"`).
- Caveat: `src/wakeword_workbench/dataset/positive_generator.py` writes with `sf.write(file_path, audio, 16000)` without explicit subtype, so bit depth may depend on SoundFile default.

4) JSONL manifest format
- Status: PRESENT
- Evidence: `src/wakeword_workbench/dataset/metadata.py` implements JSONL manifest read/write/validation; positive generation emits JSONL manifest.

5) Leak-proof dataset splitting
- Status: PRESENT
- Evidence: `src/wakeword_workbench/dataset/splitter.py` enforces train/val/test overlap checks and speaker leakage checks.

6) Agent-executable QA for every task
- Status: ABSENT
- Evidence: Evidence directory contains aggregate reports, but per-task evidence artifacts described by plan QA policy are not broadly present.

## Must NOT Have (Guardrails)

1) NO GUI
- Status: ABSENT (COMPLIANT)
- Evidence: No GUI/frontend implementation files; no GUI framework dependencies.

2) NO cloud deployment
- Status: ABSENT (COMPLIANT)
- Evidence: No cloud deployment/runtime code detected in source/dependencies.

3) NO real-time inference engine
- Status: ABSENT (COMPLIANT)
- Evidence: Inference-related code is offline file/chunk processing (`src/wakeword_workbench/mining/long_audio.py`), not a live serving engine.

4) NO model training code (dataset generation only)
- Status: ABSENT (COMPLIANT)
- Evidence: No trainer/optimizer/epoch/fit implementation detected.

5) NO distributed processing (single-machine)
- Status: ABSENT (COMPLIANT)
- Evidence: No multiprocessing/distributed framework usage (Ray/Dask/Celery/etc.).

6) NO model zoo/management
- Status: ABSENT (COMPLIANT)
- Evidence: No model registry/zoo management subsystem detected.

7) NO audio capture from microphone
- Status: ABSENT (COMPLIANT)
- Evidence: No microphone capture APIs/libraries used by implementation.

## Definition of Done

1) Full pipeline runs end-to-end from example config
- Status: ABSENT / NOT VERIFIED
- Evidence: `src/wakeword_workbench/cli.py` `run` command still contains placeholder/stub behavior; `examples/` currently has `.gitkeep` only.

2) All tests pass with >80% coverage
- Status: ABSENT
- Evidence: `.sisyphus/evidence/f2-quality.log` reports 33 failing tests and 71% coverage.

3) Generated datasets pass format validation
- Status: PARTIAL
- Evidence: Validation modules exist (`src/wakeword_workbench/export/validator.py` and exporter-level validators), but no conclusive end-to-end generated-artifact validation evidence meeting final criteria.

4) Evaluation produces actionable FAR/FRR report
- Status: PRESENT (implementation), PARTIAL (acceptance not fully verified)
- Evidence: FAR/FRR/ROC/report modules implemented in `src/wakeword_workbench/eval/`.

5) Hard negatives improve model performance
- Status: ABSENT / NOT VERIFIED
- Evidence: Hard-negative modules exist in `src/wakeword_workbench/mining/`, but no measured improvement evidence is present.

## Deviations

1) QA evidence coverage is incomplete relative to the per-task QA policy in plan.
2) End-to-end execution acceptance is incomplete due to missing runnable example config and stubbed CLI run path.
3) Documentation wording drift: `README.md` and package text mention "training" broadly, while delivered scope is primarily dataset/eval tooling.

## Final Verdict

IN SCOPE

- No material scope creep detected against Must NOT Have guardrails.
- Primary issues are delivery/verification gaps against Definition of Done and QA evidence expectations, not out-of-scope feature additions.
