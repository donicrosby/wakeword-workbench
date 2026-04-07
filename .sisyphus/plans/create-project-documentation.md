# Project Documentation Plan

Create comprehensive user documentation for the WakeWord Workbench project.

**IMPORTANT**: This plan includes a RESEARCH PHASE with authoritative sources. The BACKGROUND.md task MUST use these findings — no speculation allowed. See "Research Phase" section below.

## TL;DR

Generate 5 documentation files that enable users to understand, install, configure, and use the WakeWord Workbench toolkit effectively.

**Deliverables**:
1. `README.md` — Updated main documentation with quick start
2. `docs/BACKGROUND.md` — Concepts for newcomers to wake word systems
3. `docs/CLI_GUIDE.md` — Complete CLI command reference
4. `docs/CONFIG_REFERENCE.md` — All configuration options
5. `docs/API_USAGE.md` — Python API documentation
6. `docs/TUTORIAL.md` — Step-by-step tutorial

**Estimated Effort**: Medium (single session, multiple files)
**Parallel Execution**: YES — 6 independent tasks

---

## Context

The WakeWord Workbench is a Python 3.11+ toolkit for training micro wake word detection models. Key features:
- TTS integration (Kokoro, Piper)
- Audio augmentation pipeline
- Hard negative mining from ONNX models
- Dataset management with JSONL manifests
- FAR/FRR evaluation metrics
- Export to openWakeWord/microWakeWord formats

The project currently has minimal documentation. Users need comprehensive guides to:
- Install and configure the toolkit
- Understand CLI commands and options
- Write configuration files
- Use the Python API programmatically
- Follow a complete end-to-end workflow

---

## Research Phase (REQUIRED before writing BACKGROUND.md)

The BACKGROUND.md documentation must be based on authoritative sources, not speculation.

### Research Findings

**Wake Word Detection Fundamentals:**
- Wake word detection uses keyword spotting with sub-50ms latency and sub-1MB footprint (Arun Baby, 2026)
- Production systems use cascaded architecture: tiny detector → larger verifier (Arun Baby, 2026)
- Models run 24/7 under extreme power constraints (sub-1mW) (Arun Baby, 2026)
- MFCCs (Mel-Frequency Cepstral Coefficients) are standard features extracted via Librosa (DEV Community, 2025)
- Architecture: Convolutional Recurrent Neural Networks (CRNN) for small-footprint keyword spotting (Arik et al., Baidu, 2017)

**Sources:**
- https://www.arunbaby.com/speech-tech/0040-wake-word-detection/ (Arun Baby, 2026)
- https://www.arunbaby.com/speech-tech/0009-keyword-spotting/ (Arun Baby, 2026)
- https://dev.to/m-a-h-b-u-b/95-accurate-wake-word-detection-low-power-cnn-mfcc-guide-3377 (DEV Community, 2025)
- https://www.isca-archive.org/interspeech_2017/ark17_interspeech.pdf (Arik et al., Baidu, 2017)

**Hard Negative Mining:**
- "Mining Effective Negative Training Samples for Keyword Spotting" (Hou et al., IEEE ICASSP 2020)
- Hard negatives are examples that are similar to the target but not the target
- Random audio isn't sufficient — need confusing examples
- GraphemeAug: Systematic approach to synthesized hard negatives (Google DeepMind, 2025)
- Reduces false positives significantly

**Sources:**
- https://ieeexplore.ieee.org/document/9053009 (Hou et al., 2020)
- https://arxiv.org/html/2505.14814v2 (Google DeepMind, 2025)
- https://www.futurebeeai.com/knowledge-hub/false-positives-wake-word (FutureBeeAI, 2025)

**Evaluation Metrics (FAR/FRR/EER):**
- FAR (False Acceptance Rate): False alarms per hour — measures accidental triggers
- FRR (False Rejection Rate): Miss rate — measures when wake word isn't detected
- EER (Equal Error Rate): Point where FAR = FRR, used for system comparison
- Trade-off: Lower FAR increases FRR and vice versa (Recogtech, 2023)
- In wake word detection: FAR is typically more critical than FRR (FutureBeeAI, 2025)

**Sources:**
- https://www.futurebeeai.com/knowledge-hub/false-acceptance-rate-wake-word (FutureBeeAI, 2025)
- https://www.futurebeeai.com/knowledge-hub/false-rejection-wake-word (FutureBeeAI, 2025)
- https://recogtech.com/en/insights-en/far-and-frr-security-level-versus-ease-of-use/ (Recogtech, 2023)
- https://www.innovatrics.com/glossary/equal-error-rate-eer/ (Innovatrics, 2020)

**TTS for Wake Words:**
- "Utilizing TTS Synthesized Data for Efficient Development of Keyword Spotting Model" (Google, 2024)
- Pros: Unlimited data, fast iteration, privacy (no human recordings needed)
- Cons: Domain gap between synthetic and real speech
- Best practice: Mix TTS with real recordings
- Amazon research confirms synthetic audio can be effective (Amazon Alexa Speech, 2020)

**Sources:**
- https://arxiv.org/pdf/2407.18879 (Google, 2024)
- https://arxiv.org/pdf/2407.16840 (Google, 2024)
- https://assets.amazon.science/20/8b/696fd11d4d99a69e2adfc4602ccd/exploring-the-application-of-synthetic-audio-in-training-keyword-spotters.pdf (Amazon, 2020)

**Audio Augmentation:**
- Bridges gap between clean training and real-world conditions (Arun Baby, 2026)
- Key techniques: noise injection, reverb, time stretching, pitch shifting
- Reverb and noise simulate real-world acoustic environments (MDPI Applied Sciences, 2024)
- SpecAugment: Time- and frequency-domain masking for ASR (Park et al., 2019)
- Makes models robust to different environments, accents, channels

**Sources:**
- https://arunbaby.com/speech-tech/0018-audio-augmentation-techniques/ (Arun Baby, 2026)
- https://www.mdpi.com/2076-3417/14/23/11446 (MDPI, 2024)
- https://arxiv.gg/abs/1904.08779 (Park et al., 2019)

### Research Task

Before writing BACKGROUND.md, the agent must:
1. Review the above research findings
2. Cite sources appropriately in the document
3. Use analogies based on these authoritative sources
4. Ensure technical accuracy matches published research

---

## Work Objectives

### Core Objective
Create documentation that enables a new user to install, configure, and use the WakeWord Workbench without reading source code.

### Concrete Deliverables
1. **README.md** — Project overview, features, installation, quick start
2. **docs/BACKGROUND.md** — Background concepts for beginners to wake word ML
3. **docs/CLI_GUIDE.md** — All CLI commands with examples
4. **docs/CONFIG_REFERENCE.md** — Complete config file reference
5. **docs/API_USAGE.md** — Python API with code examples
6. **docs/TUTORIAL.md** — End-to-end walkthrough

### Definition of Done
- All files created in correct locations
- Documentation is accurate based on source code
- Examples are copy-paste ready
- Cross-references between docs work
- All CLI commands documented
- All config options explained

### Must Have
- Installation instructions (with uv)
- All 5 CLI commands documented
- Config file schema documented
- Code examples for Python API
- Troubleshooting section

### Must NOT Have
- Outdated or incorrect information
- Placeholder text like "TODO" or "coming soon"
- Assumptions about user knowledge
- Docs that require reading source to understand

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (project uses pytest)
- **Automated tests**: NO (documentation, not code)
- **Agent-Executed QA**: YES — Verify documentation accuracy

### QA Policy
Each task includes verification to ensure documentation accuracy:
- Cross-reference with source code
- Test examples work as written
- Verify file paths and commands exist

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — 6 independent tasks):
├── Task 1: Create comprehensive README.md
├── Task 2: Create BACKGROUND.md with ML concepts
├── Task 3: Create CLI_GUIDE.md with all commands
├── Task 4: Create CONFIG_REFERENCE.md
├── Task 5: Create API_USAGE.md with Python examples
└── Task 6: Create TUTORIAL.md with full workflow

No dependencies — all 6 tasks can run in parallel
```

### Agent Dispatch Summary
- **1**: **6** — T1-T6 → `writing` category
  - All tasks are documentation writing
  - Independent tasks, maximum parallelism

---

## TODOs

- [x] 1. Create README.md with comprehensive overview

  **What to do**:
  Write an updated README.md at the project root that includes:
  - Project overview and philosophy
  - Feature list with descriptions
  - Prerequisites (Python 3.11+, uv)
  - Installation instructions (quick and with extras)
  - Quick start guide (create config, validate, run)
  - Project structure overview
  - Development setup (tests, linting)
  - Key concepts (hard negatives, augmentation, metrics)
  - Troubleshooting section
  - Links to other documentation files
  - Note: New to wake words? Start with BACKGROUND.md

  **Must NOT do**:
  - Don't include outdated information
  - Don't leave TODOs or placeholders
  - Don't assume prior knowledge of wake word systems

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: None needed
  - This is pure technical writing based on existing code

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `pyproject.toml` — Dependencies, scripts, metadata
  - `src/wakeword_workbench/cli.py` — Available commands
  - `src/wakeword_workbench/config.py` — Config schema
  - `specs/wakeword_workbench_opencode_spec.md` — Project spec
  - `AGENTS.md` — Project conventions and structure

  **Acceptance Criteria**:
  - [ ] File created at `/home/doni/projects/micro_wake_word_trainer/micro-wakerword-workbench/README.md`
  - [ ] All sections from "What to do" are present
  - [ ] Installation instructions mention `uv sync`
  - [ ] Quick start includes config example
  - [ ] Links to docs/ files work correctly
  - [ ] Reference to BACKGROUND.md for newcomers included
  - [ ] No TODOs or placeholder text remains

  **QA Scenarios**:

  ```
  Scenario: README is comprehensive
    Tool: Read tool
    Steps:
      1. Read /home/doni/projects/micro_wake_word_trainer/micro-wakerword-workbench/README.md
      2. Verify all required sections exist
      3. Verify installation command is correct
      4. Verify quick start config is valid YAML
      5. Verify BACKGROUND.md is referenced for newcomers
    Expected Result: README contains all sections, no errors
    Evidence: .sisyphus/evidence/task-1-readme-check.md
  ```

  **Commit**: YES
  - Message: `docs: Add comprehensive README with quick start guide`
  - Files: `README.md`

---

- [x] 2. Create BACKGROUND.md with ML concepts for beginners

  **What to do**:
  Create `docs/BACKGROUND.md` explaining wake word ML concepts for newcomers:
  
  Sections to include:
  
  1. **What is a Wake Word?**
     - Definition and examples ("Hey Siri", "Alexa", "OK Google")
     - Why wake words matter (privacy, always-on, low power)
     - How wake word detection works at a high level
  
  2. **The Machine Learning Problem**
     - Binary classification: wake word vs. not wake word
     - The challenge: similar sounding phrases
     - Trade-offs: accuracy vs. false triggers vs. missed detections
  
  3. **Training Data Basics**
     - Why you need positive examples (the wake word itself)
     - Why you need negative examples (everything else)
     - The importance of variety: voices, accents, environments
  
  4. **Understanding Hard Negatives**
     - What are hard negatives?
     - Why random audio isn't enough
     - Examples: "Hey Davy" triggering "Hey Daisy" model
     - How hard negatives improve robustness
  
  5. **Data Augmentation Explained**
     - Why augmentation matters
     - Types of transforms:
       - Noise injection (real-world environments)
       - Reverb (room acoustics)
       - Gain/volume changes
       - Clipping/distortion
     - How augmentation multiplies your dataset
  
  6. **TTS for Wake Words**
     - Text-to-speech as a data source
     - Pros: unlimited variety, consistent quality
     - Cons: synthetic sound, limited speaker diversity
     - Best practices: mix TTS with real recordings
  
  7. **Evaluation Metrics**
     - **FAR (False Acceptance Rate)**: false alarms per hour
     - **FRR (False Rejection Rate)**: miss rate
     - **EER (Equal Error Rate)**: balanced threshold
     - Why these metrics matter for user experience
  
  8. **The Iterative Process**
     - 1. Generate initial dataset
     - 2. Train model
     - 3. Evaluate and find failure modes
     - 4. Mine hard negatives from failures
     - 5. Add to training data
     - 6. Retrain and repeat
  
  9. **Key Principles**
     - Data quality > model architecture
     - Negative coverage is critical
     - Real-world audio > synthetic only
     - Iteration beats perfection
  
  Use analogies and examples. Avoid heavy math. Focus on intuition.

  **Must NOT do**:
  - Don't assume ML background
  - Don't use jargon without explanation
  - Don't dive into neural network architectures
  - Don't skip the "why" behind each concept
  - **CRITICAL**: Don't make up facts — use the research findings in this plan
  - Don't cite sources improperly or fabricate citations

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: None needed
  - Focus on educational clarity for beginners
  - Must use research findings from "Research Phase" section above

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `specs/wakeword_workbench_opencode_spec.md` — Project goals and philosophy
  - **Research Phase findings** (REQUIRED USE):
    - Arun Baby (2026) on wake word detection and audio augmentation
    - Hou et al. (IEEE ICASSP 2020) on hard negative mining
    - Google (2024) papers on TTS for keyword spotting
    - FutureBeeAI (2025) on FAR/FRR evaluation metrics
    - Recogtech (2023) on FAR/FRR trade-offs
    - Arik et al. (Baidu, 2017) on CRNN architectures
    - Amazon (2020) on synthetic audio training
    - Park et al. (2019) on SpecAugment

  **Acceptance Criteria**:
  - [ ] File created at `docs/BACKGROUND.md`
  - [ ] All 9 sections present
  - [ ] Analogies used to explain concepts
  - [ ] Jargon defined on first use
  - [ ] Examples provided for key concepts
  - [ ] No advanced math or architecture details
  - [ ] Readable by someone new to ML
  - [ ] **Sources cited**: At least 5 of the research sources listed in "Research Phase" must be cited
  - [ ] **Technical accuracy**: Facts must match the research findings (e.g., FAR definition from FutureBeeAI)

  **QA Scenarios**:

  ```
  Scenario: Background guide is beginner-friendly and research-backed
    Tool: Read tool + Research verification
    Steps:
      1. Read docs/BACKGROUND.md
      2. Verify sections cover all core concepts
      3. Check that jargon is explained
      4. Verify examples are concrete and clear
      5. Verify at least 5 research sources are cited with URLs
      6. Cross-check key facts against research phase findings:
         - FAR definition matches FutureBeeAI
         - Hard negatives explanation matches Hou et al.
         - TTS pros/cons match Google papers
         - Augmentation techniques match Arun Baby
    Expected Result: Document is educational, accessible, and factually accurate
    Evidence: .sisyphus/evidence/task-2-background-check.md
  ```

  **Commit**: YES
  - Message: `docs: Add background guide for wake word ML concepts`
  - Files: `docs/BACKGROUND.md`

---

- [x] 3. Create CLI_GUIDE.md with full command reference

  **What to do**:
  Create `docs/CLI_GUIDE.md` documenting all CLI commands:
  
  Commands to document:
  - `run` — Run training pipeline
  - `validate` — Validate config file
  - `mine` — Mine hard negatives from audio
  - `merge` — Merge negatives into dataset
  - `cache-clear` — Clear TTS cache
  
  For each command include:
  - Purpose and use case
  - All arguments with types and defaults
  - Required vs optional arguments
  - Example usage for common scenarios
  - Exit codes and error handling
  
  Additional sections:
  - Global options (--version, --verbose, --quiet)
  - Environment variables
  - Exit code reference (0=success, 1=error, 2=config error)

  **Must NOT do**:
  - Don't miss any command or argument
  - Don't skip error scenarios
  - Don't use vague descriptions

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/wakeword_workbench/cli.py` — Complete CLI implementation
  - Look for `@app.command()` decorators
  - Check `typer.Option()` and `typer.Argument()` definitions
  - Note exit codes (EXIT_SUCCESS, EXIT_ERROR, EXIT_CONFIG_ERROR)

  **Acceptance Criteria**:
  - [ ] File created at `docs/CLI_GUIDE.md`
  - [ ] All 5 commands documented
  - [ ] Every argument has type, default, and description
  - [ ] At least one example per command
  - [ ] Exit codes explained
  - [ ] Global options documented

  **QA Scenarios**:

  ```
  Scenario: CLI docs are complete
    Tool: Read + Grep
    Steps:
      1. Read docs/CLI_GUIDE.md
      2. Grep for each command name (run, validate, mine, merge, cache-clear)
      3. Verify all arguments from cli.py are documented
      4. Check that examples are present
    Expected Result: All commands found, all arguments covered
    Evidence: .sisyphus/evidence/task-2-cli-check.md
  ```

  **Commit**: YES
  - Message: `docs: Add CLI command reference guide`
  - Files: `docs/CLI_GUIDE.md`

---

- [x] 3. Create CONFIG_REFERENCE.md with all options

  **What to do**:
  Create `docs/CONFIG_REFERENCE.md` with complete config file documentation:
  
  Config sections to document (from `config.py`):
  - `wake_word` — String, required
  - `samples` — Object with:
    - `positives` — Integer, >0
    - `negatives_multiplier` — Integer, >0
  - `tts` — Object with:
    - `backend` — String (kokoro, piper)
    - `voices` — List of strings
    - `speed` — Float, 0-3, default 1.0
  - `augmentation` — Object with:
    - `noise_snr` — List of 2 floats [min, max]
    - `reverb_probability` — Float, 0-1
    - `gain_range` — List of 2 floats [min, max]
  - `output` — Object with:
    - `path` — String (directory path)
    - `format` — List of strings (microwakeword, openwakeword)
  
  Include:
  - Complete example config
  - Field descriptions and constraints
  - Validation rules
  - Error messages for invalid values

  **Must NOT do**:
  - Don't omit any config field
  - Don't skip validation constraints
  - Don't use placeholder descriptions

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/wakeword_workbench/config.py` — All config classes and validation
  - `SamplesConfig.__post_init__` — Validation rules
  - `TTSConfig.__post_init__` — TTS validation
  - `AugmentationConfig.__post_init__` — Aug validation
  - `OutputConfig.__post_init__` — Output validation

  **Acceptance Criteria**:
  - [ ] File created at `docs/CONFIG_REFERENCE.md`
  - [ ] All 5 top-level sections documented
  - [ ] All nested fields have descriptions
  - [ ] Validation constraints listed
  - [ ] Complete example provided
  - [ ] Valid values enumerated where applicable

  **QA Scenarios**:

  ```
  Scenario: Config reference is complete
    Tool: Read + Compare
    Steps:
      1. Read docs/CONFIG_REFERENCE.md
      2. Read src/wakeword_workbench/config.py
      3. Compare: every field in config.py must be in reference
      4. Verify validation rules match __post_init__ methods
    Expected Result: All config fields documented with correct constraints
    Evidence: .sisyphus/evidence/task-3-config-check.md
  ```

  **Commit**: YES
  - Message: `docs: Add configuration reference guide`
  - Files: `docs/CONFIG_REFERENCE.md`

---

- [x] 4. Create API_USAGE.md with Python examples

  **What to do**:
  Create `docs/API_USAGE.md` documenting the Python API:
  
  Modules to cover:
  
  1. **Config Module** (`wakeword_workbench.config`)
     - `load_config()` — Load and validate YAML
     - `Config` dataclass structure
     - Error handling with `ConfigError`
  
  2. **TTS Module** (`wakeword_workbench.tts`)
     - `get_backend()` — Get TTS instance
     - `list_available_backends()` — List available
     - `TTSBackend.synthesize()` — Generate audio
     - `TTSCache` — Cache management
  
  3. **Augmentation Module** (`wakeword_workbench.augment`)
     - `Compose` — Pipeline composition
     - `Transform` protocol
     - Available transforms (AdjustGain, AddNoise, etc.)
     - Preset pipelines (minimal_pipeline, default_pipeline, heavy_pipeline)
     - `from_config()` — Load from YAML/JSON
  
  4. **Dataset Module** (`wakeword_workbench.dataset`)
     - `Manifest` — JSONL manifest handling
     - `ManifestEntry` — Single entry structure
  
  5. **Evaluation Module** (`wakeword_workbench.eval`)
     - `generate_report()` — Create evaluation report
     - FAR/FRR calculation functions
     - Report formats (json, csv, markdown)
  
  6. **Export Module** (`wakeword_workbench.export`)
     - `export_to_numpy()` — openWakeWord format
     - Export configuration options
  
  Include code examples for each API section.

  **Must NOT do**:
  - Don't skip any major module
  - Don't provide broken code examples
  - Don't omit error handling patterns

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/wakeword_workbench/config.py` — Config loading
  - `src/wakeword_workbench/tts/registry.py` — TTS registry
  - `src/wakeword_workbench/tts/base.py` — TTSBackend ABC
  - `src/wakeword_workbench/augment/pipeline.py` — Augmentation
  - `src/wakeword_workbench/dataset/metadata.py` — Manifest
  - `src/wakeword_workbench/eval/report.py` — Evaluation
  - `src/wakeword_workbench/export/openwakeword.py` — Export

  **Acceptance Criteria**:
  - [ ] File created at `docs/API_USAGE.md`
  - [ ] All 6 modules documented
  - [ ] Code examples for major functions
  - [ ] Import statements shown
  - [ ] Error handling demonstrated
  - [ ] Examples are syntactically valid Python

  **QA Scenarios**:

  ```
  Scenario: API examples are valid
    Tool: Python syntax check
    Steps:
      1. Extract all Python code blocks from docs/API_USAGE.md
      2. Run python -m py_compile on each
      3. Verify no syntax errors
      4. Check imports match actual module structure
    Expected Result: All code examples are valid Python
    Evidence: .sisyphus/evidence/task-4-api-check.md
  ```

  **Commit**: YES
  - Message: `docs: Add Python API usage guide`
  - Files: `docs/API_USAGE.md`

---

- [x] 5. Create TUTORIAL.md with step-by-step guide

  **What to do**:
  Create `docs/TUTORIAL.md` with a complete end-to-end workflow:
  
  Tutorial sections:
  
  1. **Prerequisites**
     - Python 3.11+ installed
     - uv installed
     - Wake word phrase chosen
  
  2. **Installation**
     - Clone repo
     - uv sync
     - Install TTS extras
  
  3. **Create Your First Config**
     - Write config.yaml
     - Choose wake word
     - Set sample counts
     - Select TTS voices
  
  4. **Validate Config**
     - Run validate command
     - Fix any errors
  
  5. **Generate Dataset (when ready)**
     - Run pipeline (stub for now)
     - Check output
  
  6. **Hard Negative Mining Workflow**
     - Record long audio
     - Mine false positives
     - Review extracted clips
     - Merge into training data
  
  7. **Augmentation**
     - Understand transforms
     - Configure augmentation
     - Apply to samples
  
  8. **Evaluation**
     - Run model on test audio
     - Generate FAR/FRR report
     - Interpret results
  
  9. **Export**
     - Export to openWakeWord format
     - Export to microWakeWord format
  
  10. **Next Steps**
      - Iterate on data quality
      - Mine more hard negatives
      - Retrain and re-evaluate

  Use a concrete example throughout (e.g., wake word "Hey Helper").

  **Must NOT do**:
  - Don't skip the mining workflow (this is the working feature)
  - Don't use abstract examples
  - Don't omit troubleshooting for common issues

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `specs/wakeword_workbench_opencode_spec.md` — Full project spec
  - `src/wakeword_workbench/cli.py` — Working commands (mine, merge)
  - `README.md` (Task 1 output) — For cross-references
  - `docs/CLI_GUIDE.md` (Task 2 output) — Command details

  **Acceptance Criteria**:
  - [ ] File created at `docs/TUTORIAL.md`
  - [ ] All 10 sections present
  - [ ] Concrete example used throughout
  - [ ] Hard negative mining covered in detail
  - [ ] Troubleshooting tips included
  - [ ] Next steps guide provided

  **QA Scenarios**:

  ```
  Scenario: Tutorial is complete and usable
    Tool: Read tool
    Steps:
      1. Read docs/TUTORIAL.md
      2. Verify all 10 sections exist
      3. Check that commands are copy-paste ready
      4. Verify hard negative mining has detailed steps
    Expected Result: Tutorial covers full workflow, commands are valid
    Evidence: .sisyphus/evidence/task-5-tutorial-check.md
  ```

  **Commit**: YES
  - Message: `docs: Add step-by-step tutorial`
  - Files: `docs/TUTORIAL.md`

---

## Final Verification Wave (after ALL tasks)

- [x] F1. **Documentation Completeness Check** — `deep`
  Read all 6 documentation files. Verify:
  - No TODOs or placeholder text
  - All cross-references between docs work
  - No conflicting information
  - All code examples are syntactically valid
  Output: `Files [6/6] | Issues [N] | VERDICT`

- [x] F2. **Cross-Reference Validation** — `quick`
  Check that docs reference each other correctly:
  - README links to docs/ files
  - TUTORIAL references CLI_GUIDE for command details
  - TUTORIAL references CONFIG_REFERENCE for options
  - All relative paths are correct
  Output: `Links [N/N valid] | VERDICT`

- [x] F3. **Accuracy Check Against Source and Research** — `deep`
  Verify documentation matches source code AND research findings:
  - CLI args match cli.py
  - Config fields match config.py
  - API signatures match actual code
  - Examples use correct import paths
  - **BACKGROUND.md cites at least 5 research sources**
  - **BACKGROUND.md facts match research phase findings**
  Output: `Accuracy [N/N checks passed] | Research Sources [N cited] | VERDICT`

---

## Commit Strategy

- **1**: `docs: Add comprehensive README with quick start guide` — README.md
- **2**: `docs: Add background guide for wake word ML concepts` — docs/BACKGROUND.md
- **3**: `docs: Add CLI command reference guide` — docs/CLI_GUIDE.md
- **4**: `docs: Add configuration reference guide` — docs/CONFIG_REFERENCE.md
- **5**: `docs: Add Python API usage guide` — docs/API_USAGE.md
- **6**: `docs: Add step-by-step tutorial` — docs/TUTORIAL.md

---

## Success Criteria

### Verification Commands
```bash
# Check all docs exist
ls -la docs/
cat README.md | head -50

# Verify no TODOs
grep -i "todo\|coming soon\|placeholder" docs/*.md README.md || echo "No TODOs found"

# Check cross-references work
grep -r "\[.*\](docs/" README.md
grep -r "\.md)" docs/*.md

# Verify BACKGROUND.md has research citations
grep -c "http" docs/BACKGROUND.md  # Should be >= 5
grep -E "Arun Baby|Hou et al|Google|FutureBeeAI|Amazon|Recogtech" docs/BACKGROUND.md | wc -l
```

### Final Checklist
- [ ] README.md updated with comprehensive overview
- [ ] docs/BACKGROUND.md explains concepts for beginners
- [ ] docs/CLI_GUIDE.md has all commands
- [ ] docs/CONFIG_REFERENCE.md has all options
- [ ] docs/API_USAGE.md has Python examples
- [ ] docs/TUTORIAL.md has step-by-step guide
- [ ] No TODOs or placeholders remain
- [ ] All cross-references are valid
- [ ] Examples are syntactically correct
