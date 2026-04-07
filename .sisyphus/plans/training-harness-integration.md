# Training Harness Integration Research & Documentation Plan

## Executive Summary

This plan outlines research and documentation tasks to help users integrate the WakeWord Workbench with microWakeWord and openWakeWord training harnesses. The workbench generates high-quality datasets; this plan bridges the gap to actual model training.

**Key Finding**: The workbench is a sophisticated data generation toolkit (Phases 1-7 implemented), but the main pipeline orchestrator (`dataset/generator.py`) and CLI `run` command are stubbed. Training happens in external repositories (microWakeWord/openWakeWord).

---

## Research Findings Summary

### Current Workbench State

| Component | Status | Notes |
|-----------|--------|-------|
| **TTS Generation** | ✅ Complete | Kokoro + Piper backends |
| **Augmentation** | ✅ Complete | Noise, reverb, gain, padding |
| **Negative Generation** | ✅ Complete | Phonetic confusion + synthetic |
| **Dataset Export** | ✅ Complete | microWakeWord & openWakeWord formats |
| **Evaluation** | ✅ Complete | FAR/FRR, ROC, threshold sweep |
| **Hard Negative Mining** | ✅ Complete | ONNX inference + extraction |
| **Dataset Generator** | ❌ **STUBBED** | Main orchestrator not implemented |
| **CLI Run Command** | ❌ **STUBBED** | Pipeline execution not implemented |

### microWakeWord Integration Points

**Repository**: [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word) (~800 stars)

**Data Flow**:
```
Workbench Output (.wav, 16kHz mono)
  → SpectrogramGeneration (40 mel features, 10ms stride)
  → RaggedMmap storage
  → FeatureHandler (data loading)
  → Training pipeline
```

**Key Training Resource**: [basic_training_notebook.ipynb](https://github.com/OHF-Voice/micro-wake-word/blob/main/notebooks/basic_training_notebook.ipynb)

### openWakeWord Integration Points

**Repository**: [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord) (2.1K stars)

**Data Flow**:
```
Workbench Output (.wav, 16kHz mono)
  → AudioFeatures (Google speech embedding model)
  → Features shape: (N, 16, 96)
  → Model.auto_train()
  → ONNX/TFLite export
```

**Key Training Resources**:
- [automatic_model_training.ipynb](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)
- [training_models.ipynb](https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb)

---

## Work Objectives

### Core Objective
Create comprehensive documentation and example notebooks that guide users through the complete workflow: WakeWord Workbench → Training Harness → Trained Model.

### Concrete Deliverables
1. **Integration Guide** — Compare microWakeWord vs openWakeWord, help users choose
2. **End-to-End Tutorial Notebooks** (2 notebooks)
3. **API Reference** — Workbench export modules for programmatic use
4. **Training Configuration Templates** — YAML configs for both harnesses
5. **Troubleshooting Guide** — Common issues and solutions

### Definition of Done
- User can follow a notebook from "I have a wake word phrase" to "I have a trained .tflite model"
- All export formats are documented with code examples
- Configuration templates are ready-to-use

### Must Have
- Step-by-step notebooks for both microWakeWord and openWakeWord
- Clear explanation of data format conversions
- Working code examples that can be copy-pasted

### Must NOT Have (Guardrails)
- NO implementation of actual training code in workbench (out of scope)
- NO modification of external training repositories
- NO "quick start" that skips important configuration steps
- NO documentation without tested, working examples

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest, pytest-cov)
- **Automated tests for notebooks**: NO — notebooks will be verified via manual execution during documentation phase
- **Manual QA**: Required for notebooks — they must run end-to-end without errors

### QA Policy
- Each notebook will be executed end-to-end before finalization
- Screenshots/logs captured as evidence
- External dependencies (microWakeWord/openWakeWord repos) pinned to specific commits

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Research & Planning — Start Immediately):
├── Task 1: Deep-dive microWakeWord training pipeline
├── Task 2: Deep-dive openWakeWord training pipeline  
├── Task 3: Create data format conversion reference
└── Task 4: Map workbench features → harness requirements

Wave 2 (Documentation — After Wave 1):
├── Task 5: Write Integration Guide (markdown)
├── Task 6: Create Training Configuration Templates
├── Task 7: Write API Reference for export modules
└── Task 8: Write Troubleshooting Guide

Wave 3 (Notebooks — After Wave 2):
├── Task 9: Create microWakeWord training notebook
├── Task 10: Create openWakeWord training notebook
└── Task 11: Create "Choosing Your Training Harness" comparison notebook

Wave 4 (Review & Polish — After Wave 3):
├── Task F1: Execute all notebooks, capture evidence
├── Task F2: Review documentation for accuracy
├── Task F3: Check all external links and references
└── Task F4: Final user validation
```

### Critical Path
Task 1/2 → Task 3 → Task 5 → Task 9/10 → F1 → F4 → User okay

---

## TODOs

### Wave 1: Deep Research & Data Format Mapping

- [x] 1. **Deep-dive microWakeWord Training Pipeline**

  **What to do**:
  - Study the complete microWakeWord training notebook end-to-end
  - Document the exact data format conversions required (workbench .wav → RaggedMmap)
  - Map workbench export format to microWakeWord FeatureHandler expectations
  - Identify all configuration parameters and their defaults
  - Document the 2-stage training process and hyperparameter tuning

  **Must NOT do**:
  - Don't copy microWakeWord code into workbench
  - Don't modify the microWakeWord repository
  - Don't assume users have GPU access

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Need thorough understanding of training pipeline, data flow, and integration points
  - **Skills**: None needed — pure research and documentation

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1 (Tasks 1-4)
  - **Blocks**: Task 9 (microWakeWord notebook)
  - **Blocked By**: None

  **References**:
  - Repository: `https://github.com/OHF-Voice/micro-wake-word`
  - Notebook: `notebooks/basic_training_notebook.ipynb`
  - Data loading: `microwakeword/data.py` — FeatureHandler class
  - Training: `microwakeword/train.py` — Core training loop
  - Features: `microwakeword/audio/spectrograms.py` — SpectrogramGeneration

  **Acceptance Criteria**:
  - [ ] Complete data format mapping documented (what workbench outputs → what microWakeWord expects)
  - [ ] Configuration parameters table created with defaults and recommendations
  - [ ] Training workflow diagram created
  - [ ] Evidence file: `.sisyphus/evidence/task-1-microwakeword-analysis.md`

  **QA Scenarios**:
  ```
  Scenario: Validate data format understanding
    Tool: Bash (file inspection)
    Steps:
      1. Clone microWakeWord repo to /tmp/
      2. Inspect FeatureHandler class signature
      3. Verify expected input shapes and dtypes
      4. Compare with workbench export format
    Expected Result: Clear mapping document showing compatibility
    Evidence: .sisyphus/evidence/task-1-microwakeword-analysis.md
  ```

  **Commit**: NO (documentation phase)

---

- [x] 2. **Deep-dive openWakeWord Training Pipeline**

  **What to do**:
  - Study both openWakeWord notebooks (automatic and manual training)
  - Document the exact feature extraction process (audio → 16×96 features)
  - Map workbench export to openWakeWord's expected numpy format
  - Document the 3-sequence training process and hyperparameter tuning
  - Identify key differences from microWakeWord approach

  **Must NOT do**:
  - Don't copy openWakeWord code into workbench
  - Don't modify the openWakeWord repository
  - Don't assume users have specific GPU/CPU configurations

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex training pipeline with multiple stages and feature extraction
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1)
  - **Parallel Group**: Wave 1 (Tasks 1-4)
  - **Blocks**: Task 10 (openWakeWord notebook)
  - **Blocked By**: None

  **References**:
  - Repository: `https://github.com/dscripka/openWakeWord`
  - Notebook 1: `notebooks/automatic_model_training.ipynb`
  - Notebook 2: `notebooks/training_models.ipynb`
  - Training: `openwakeword/train.py` — Model class with auto_train()
  - Features: `openwakeword/utils.py` — AudioFeatures class
  - Config: `examples/custom_model.yml`

  **Acceptance Criteria**:
  - [ ] Complete data format mapping documented (workbench .wav → 16×96 features)
  - [ ] Configuration parameters table created with defaults and recommendations
  - [ ] Training workflow diagram created
  - [ ] Comparison with microWakeWord documented
  - [ ] Evidence file: `.sisyphus/evidence/task-2-openwakeword-analysis.md`

  **QA Scenarios**:
  ```
  Scenario: Validate feature extraction understanding
    Tool: Bash (file inspection)
    Steps:
      1. Clone openWakeWord repo to /tmp/
      2. Inspect AudioFeatures class and feature shape
      3. Verify Model.auto_train() signature
      4. Compare with workbench export format
    Expected Result: Clear mapping showing feature conversion requirements
    Evidence: .sisyphus/evidence/task-2-openwakeword-analysis.md
  ```

  **Commit**: NO

---

- [x] 3. **Create Data Format Conversion Reference**

  **What to do**:
  - Create a comprehensive reference document mapping workbench outputs to both training harnesses
  - Document audio format requirements (sample rate, channels, bit depth)
  - Create conversion code snippets for both directions
  - Document feature extraction parameters (mel spectrogram settings)
  - Create quick reference tables for developers

  **Must NOT do**:
  - Don't implement conversion code in workbench yet (out of scope)
  - Don't assume specific hardware capabilities

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Creating reference documentation and tables
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Tasks 1 & 2 start)
  - **Parallel Group**: Wave 1 (Tasks 1-4)
  - **Blocks**: Task 5 (Integration Guide), Task 6 (Config Templates)
  - **Blocked By**: Task 1, Task 2

  **References**:
  - Workbench export: `src/wakeword_workbench/export/microwakeword.py`
  - Workbench export: `src/wakeword_workbench/export/openwakeword.py`
  - microWakeWord spectrograms: `microwakeword/audio/spectrograms.py`
  - openWakeWord features: `openwakeword/utils.py`

  **Acceptance Criteria**:
  - [ ] Data format reference document created: `docs/training/data-format-reference.md`
  - [ ] Conversion code snippets for both harnesses
  - [ ] Audio requirements table (sample rate, channels, etc.)
  - [ ] Feature extraction parameters table

  **QA Scenarios**:
  ```
  Scenario: Validate reference document completeness
    Tool: Read (file inspection)
    Steps:
      1. Read docs/training/data-format-reference.md
      2. Verify all required formats are documented
      3. Check code snippets are syntactically correct
      4. Verify tables are complete
    Expected Result: Document contains all required information
    Evidence: .sisyphus/evidence/task-3-format-reference-review.md
  ```

  **Commit**: YES
  - Message: `docs(training): Add data format conversion reference`
  - Files: `docs/training/data-format-reference.md`

---

- [x] 4. **Map Workbench Features → Harness Requirements**

  **What to do**:
  - Create a matrix showing which workbench features are used by each training harness
  - Identify any workbench features that need special handling
  - Document recommended workbench configurations for each harness
  - Note any compatibility issues or limitations

  **Must NOT do**:
  - Don't recommend features that don't work well with either harness

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Analysis task requiring cross-referencing multiple systems
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Tasks 1 & 2 start)
  - **Parallel Group**: Wave 1 (Tasks 1-4)
  - **Blocks**: Task 5 (Integration Guide)
  - **Blocked By**: Task 1, Task 2

  **References**:
  - Workbench features: Review all modules in `src/wakeword_workbench/`
  - microWakeWord requirements: From Task 1 research
  - openWakeWord requirements: From Task 2 research

  **Acceptance Criteria**:
  - [ ] Feature compatibility matrix created
  - [ ] Recommended configurations documented
  - [ ] Known limitations and workarounds documented
  - [ ] Evidence file: `.sisyphus/evidence/task-4-feature-mapping.md`

  **QA Scenarios**:
  ```
  Scenario: Validate feature mapping
    Tool: Read (file inspection)
    Steps:
      1. Read feature mapping document
      2. Verify all major workbench features are covered
      3. Check recommendations are consistent with harness docs
    Expected Result: Complete mapping with actionable recommendations
    Evidence: .sisyphus/evidence/task-4-feature-mapping.md
  ```

  **Commit**: NO (intermediate research artifact)

---

### Wave 2: Documentation

- [x] 5. **Write Integration Guide**

  **What to do**:
  - Create comprehensive guide comparing microWakeWord vs openWakeWord
  - Help users choose the right harness for their use case
  - Document installation requirements for both
  - Provide quick-start instructions for each harness
  - Include architecture comparison and trade-offs

  **Must NOT do**:
  - Don't duplicate external documentation — link to it instead
  - Don't provide outdated installation instructions

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Creating comprehensive documentation with comparisons
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Wave 1 completes)
  - **Parallel Group**: Wave 2 (Tasks 5-8)
  - **Blocks**: Task 11 (Choosing Harness notebook)
  - **Blocked By**: Tasks 1-4

  **References**:
  - microWakeWord README and docs
  - openWakeWord README and DeepWiki
  - Data format reference from Task 3
  - Feature mapping from Task 4

  **Acceptance Criteria**:
  - [ ] Guide created: `docs/training/integration-guide.md`
  - [ ] Comparison table: microWakeWord vs openWakeWord
  - [ ] Decision flowchart for choosing harness
  - [ ] Installation instructions for both (verified working)
  - [ ] Architecture comparison section

  **QA Scenarios**:
  ```
  Scenario: Validate guide completeness
    Tool: Read (file inspection)
    Steps:
      1. Read integration-guide.md
      2. Verify comparison table covers key factors
      3. Check decision flowchart is clear
      4. Verify all external links work
    Expected Result: Guide helps users make informed choice
    Evidence: .sisyphus/evidence/task-5-integration-guide-review.md
  ```

  **Commit**: YES
  - Message: `docs(training): Add integration guide comparing harnesses`
  - Files: `docs/training/integration-guide.md`

---

- [x] 6. **Create Training Configuration Templates**

  **What to do**:
  - Create ready-to-use YAML configs for microWakeWord
  - Create ready-to-use YAML configs for openWakeWord
  - Include comments explaining each parameter
  - Provide templates for different scenarios (quick test, production, fine-tuning)
  - Document how to adapt configs for workbench-generated datasets

  **Must NOT do**:
  - Don't create configs that won't work without modification
  - Don't omit important parameters

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Creating configuration files based on research
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Wave 1 completes)
  - **Parallel Group**: Wave 2 (Tasks 5-8)
  - **Blocks**: Tasks 9-10 (Notebooks)
  - **Blocked By**: Tasks 1-4

  **References**:
  - microWakeWord config structure from Task 1
  - openWakeWord `examples/custom_model.yml`
  - Workbench config examples: `examples/*.yaml`

  **Acceptance Criteria**:
  - [ ] microWakeWord configs: `examples/training/microwakeword-*.yaml`
  - [ ] openWakeWord configs: `examples/training/openwakeword-*.yaml`
  - [ ] Each config has inline comments
  - [ ] README explaining config selection

  **QA Scenarios**:
  ```
  Scenario: Validate config templates
    Tool: Bash (yamllint)
    Steps:
      1. Validate YAML syntax of all config files
      2. Check configs match harness requirements
      3. Verify paths are reasonable defaults
    Expected Result: All configs are valid YAML and match harness specs
    Evidence: .sisyphus/evidence/task-6-config-validation.log
  ```

  **Commit**: YES
  - Message: `examples(training): Add configuration templates for both harnesses`
  - Files: `examples/training/*`

---

- [x] 7. **Write API Reference for Export Modules**

  **What to do**:
  - Document the workbench export API (`microwakeword.py`, `openwakeword.py`)
  - Provide code examples for programmatic use
  - Document all parameters with types and descriptions
  - Show how to integrate exports into custom pipelines
  - Include validation examples

  **Must NOT do**:
  - Don't duplicate docstrings — enhance them with examples

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Technical API documentation
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Wave 1 completes)
  - **Parallel Group**: Wave 2 (Tasks 5-8)
  - **Blocks**: None (reference material)
  - **Blocked By**: None

  **References**:
  - Source: `src/wakeword_workbench/export/microwakeword.py`
  - Source: `src/wakeword_workbench/export/openwakeword.py`
  - Tests: `tests/test_*_export.py` for usage examples

  **Acceptance Criteria**:
  - [ ] API reference created: `docs/training/export-api.md`
  - [ ] All public functions documented
  - [ ] Code examples for each function
  - [ ] Integration example showing full pipeline

  **QA Scenarios**:
  ```
  Scenario: Validate API examples work
    Tool: Bash (python execution)
    Steps:
      1. Run code examples from API docs
      2. Verify they execute without errors
      3. Check outputs match expectations
    Expected Result: All code examples run successfully
    Evidence: .sisyphus/evidence/task-7-api-examples.log
  ```

  **Commit**: YES
  - Message: `docs(training): Add export API reference with examples`
  - Files: `docs/training/export-api.md`

---

- [x] 8. **Write Troubleshooting Guide**

  **What to do**:
  - Document common issues when integrating with training harnesses
  - Include workbench-specific issues (dataset generation, export)
  - Include harness-specific issues (training failures, model export)
  - Provide debugging steps and solutions
  - Include FAQ section

  **Must NOT do**:
  - Don't include issues that are already well-documented in harness repos
  - Don't provide solutions that could damage user data

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Creating troubleshooting documentation
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Wave 1 completes)
  - **Parallel Group**: Wave 2 (Tasks 5-8)
  - **Blocks**: None (reference material)
  - **Blocked By**: Tasks 1-2 (for understanding common issues)

  **References**:
  - Common issues from Task 1 and 2 research
  - GitHub issues from harness repositories
  - Workbench test failures and edge cases

  **Acceptance Criteria**:
  - [ ] Troubleshooting guide: `docs/training/troubleshooting.md`
  - [ ] Common issues organized by category
  - [ ] Solutions include example commands/code
  - [ ] FAQ section with at least 10 questions

  **QA Scenarios**:
  ```
  Scenario: Validate troubleshooting coverage
    Tool: Read (file inspection)
    Steps:
      1. Read troubleshooting.md
      2. Verify all major issue categories covered
      3. Check solutions are actionable
      4. Verify FAQ addresses common questions
    Expected Result: Guide helps users resolve common issues
    Evidence: .sisyphus/evidence/task-8-troubleshooting-review.md
  ```

  **Commit**: YES
  - Message: `docs(training): Add troubleshooting guide`
  - Files: `docs/training/troubleshooting.md`

---

### Wave 3: Interactive Notebooks

- [x] 9. **Create microWakeWord Training Notebook**

  **What to do**:
  - Create Jupyter notebook showing complete workflow: workbench → microWakeWord → trained model
  - Include dataset generation with workbench
  - Show data conversion and feature extraction
  - Walk through training configuration
  - Execute training (or show how to)
  - Include evaluation and model export
  - Add explanatory text and visualizations

  **Must NOT do**:
  - Don't assume GPU availability — provide CPU instructions
  - Don't skip error handling

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex notebook requiring multiple integrated steps
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Wave 2 completes)
  - **Parallel Group**: Wave 3 (Tasks 9-11)
  - **Blocks**: Task F1 (Notebook execution)
  - **Blocked By**: Tasks 1, 3, 5, 6

  **References**:
  - microWakeWord notebook: `notebooks/basic_training_notebook.ipynb`
  - Workbench examples: `examples/*.yaml`
  - Config templates from Task 6

  **Acceptance Criteria**:
  - [ ] Notebook created: `notebooks/training-with-microwakeword.ipynb`
  - [ ] All cells have clear explanations
  - [ ] Code is tested and executable
  - [ ] Includes visualization of results
  - [ ] Estimated runtime documented per section

  **QA Scenarios**:
  ```
  Scenario: Validate notebook runs end-to-end
    Tool: Bash (jupyter execution)
    Steps:
      1. Execute notebook in clean environment
      2. Verify all cells complete without errors
      3. Check outputs are reasonable
      4. Validate trained model file is created
    Expected Result: Notebook completes successfully
    Evidence: .sisyphus/evidence/task-9-notebook-execution.log
  ```

  **Commit**: YES
  - Message: `notebooks: Add microWakeWord training tutorial`
  - Files: `notebooks/training-with-microwakeword.ipynb`

---

- [x] 10. **Create openWakeWord Training Notebook**

  **What to do**:
  - Create Jupyter notebook showing complete workflow: workbench → openWakeWord → trained model
  - Include dataset generation with workbench
  - Show feature extraction with AudioFeatures
  - Walk through training configuration
  - Execute training (or show how to)
  - Include evaluation and model export
  - Add explanatory text and visualizations

  **Must NOT do**:
  - Don't assume specific hardware
  - Don't skip installation steps

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex multi-step notebook with external dependencies
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Wave 2 completes)
  - **Parallel Group**: Wave 3 (Tasks 9-11)
  - **Blocks**: Task F1 (Notebook execution)
  - **Blocked By**: Tasks 2, 3, 5, 6

  **References**:
  - openWakeWord notebooks: `notebooks/automatic_model_training.ipynb`, `notebooks/training_models.ipynb`
  - Workbench examples: `examples/*.yaml`
  - Config templates from Task 6

  **Acceptance Criteria**:
  - [ ] Notebook created: `notebooks/training-with-openwakeword.ipynb`
  - [ ] All cells have clear explanations
  - [ ] Code is tested and executable
  - [ ] Includes visualization of results
  - [ ] Estimated runtime documented per section

  **QA Scenarios**:
  ```
  Scenario: Validate notebook runs end-to-end
    Tool: Bash (jupyter execution)
    Steps:
      1. Execute notebook in clean environment
      2. Verify all cells complete without errors
      3. Check outputs are reasonable
      4. Validate trained model file is created
    Expected Result: Notebook completes successfully
    Evidence: .sisyphus/evidence/task-10-notebook-execution.log
  ```

  **Commit**: YES
  - Message: `notebooks: Add openWakeWord training tutorial`
  - Files: `notebooks/training-with-openwakeword.ipynb`

---

- [x] 11. **Create "Choosing Your Training Harness" Comparison Notebook**

  **What to do**:
  - Create interactive notebook helping users choose between microWakeWord and openWakeWord
  - Include side-by-side comparisons
  - Provide decision matrix/quiz
  - Show code snippets for both harnesses
  - Include performance benchmarks
  - Add real-world use case examples

  **Must NOT do**:
  - Don't bias toward one harness — present objective comparison

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Creating comparison content with interactive elements
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Wave 2 completes)
  - **Parallel Group**: Wave 3 (Tasks 9-11)
  - **Blocks**: None
  - **Blocked By**: Task 5 (Integration Guide)

  **References**:
  - Integration Guide from Task 5
  - Research from Tasks 1 and 2

  **Acceptance Criteria**:
  - [ ] Notebook created: `notebooks/choosing-your-training-harness.ipynb`
  - [ ] Interactive decision elements included
  - [ ] Code snippets for both harnesses
  - [ ] Performance comparison table
  - [ ] Use case recommendations

  **QA Scenarios**:
  ```
  Scenario: Validate notebook content
    Tool: Read (file inspection)
    Steps:
      1. Read notebook content
      2. Verify comparisons are balanced
      3. Check decision matrix is helpful
      4. Verify code snippets are correct
    Expected Result: Notebook helps users make informed decision
    Evidence: .sisyphus/evidence/task-11-comparison-review.md
  ```

  **Commit**: YES
  - Message: `notebooks: Add training harness comparison guide`
  - Files: `notebooks/choosing-your-training-harness.ipynb`

---

### Wave 4: Final Verification

- [x] F1. **Execute All Notebooks, Capture Evidence**

  **What to do**:
  - Run all three notebooks end-to-end in clean environment
  - Capture execution logs and outputs
  - Verify all code cells execute without errors
  - Document any environment-specific issues
  - Create execution summary report

  **Must NOT do**:
  - Don't skip cells or assume they work
  - Don't ignore warnings that might indicate problems

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Comprehensive testing across multiple notebooks
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (sequential notebook execution)
  - **Blocks**: Tasks F2-F4
  - **Blocked By**: Tasks 9-11

  **References**:
  - Notebooks: `notebooks/*.ipynb`

  **Acceptance Criteria**:
  - [ ] All three notebooks execute successfully
  - [ ] Execution logs saved: `.sisyphus/evidence/notebook-execution-*.log`
  - [ ] Summary report created
  - [ ] Any issues documented with workarounds

  **QA Scenarios**:
  ```
  Scenario: Validate all notebooks run successfully
    Tool: Bash (jupyter nbconvert)
    Steps:
      1. Execute each notebook with --execute flag
      2. Capture all output to log files
      3. Check for any ERROR or Exception in logs
      4. Verify expected output files are created
    Expected Result: All notebooks complete with zero errors
    Evidence: .sisyphus/evidence/notebook-execution-summary.md
  ```

  **Commit**: NO (evidence files only)

---

- [x] F2. **Review Documentation for Accuracy**

  **What to do**:
  - Review all documentation created in Wave 2
  - Verify technical accuracy against harness documentation
  - Check for outdated information
  - Verify all code examples work
  - Check for broken links

  **Must NOT do**:
  - Don't assume docs are correct without verification

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Comprehensive documentation review
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (with F3, F4)
  - **Parallel Group**: Wave 4 (Tasks F1-F4)
  - **Blocks**: None
  - **Blocked By**: Tasks 5-8, F1

  **References**:
  - All documentation files in `docs/training/`
  - External harness documentation

  **Acceptance Criteria**:
  - [ ] All docs reviewed and verified
  - [ ] Inaccuracies corrected
  - [ ] Broken links fixed
  - [ ] Review report created

  **QA Scenarios**:
  ```
  Scenario: Validate documentation accuracy
    Tool: Read (file inspection) + webfetch
    Steps:
      1. Read each documentation file
      2. Verify facts against external sources
      3. Test all external links
      4. Check code examples for correctness
    Expected Result: All documentation is accurate and up-to-date
    Evidence: .sisyphus/evidence/documentation-review-report.md
  ```

  **Commit**: YES (if any fixes needed)

---

- [x] F3. **Check All External Links and References**

  **What to do**:
  - Verify all external links work
  - Check that repository references point to correct commits/versions
  - Verify HuggingFace dataset links
  - Check that API documentation links are current
  - Document any version pinning requirements

  **Must NOT do**:
  - Don't leave broken links in final documentation

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Link verification task
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (with F2, F4)
  - **Parallel Group**: Wave 4 (Tasks F1-F4)
  - **Blocks**: None
  - **Blocked By**: Tasks 5-11

  **References**:
  - All markdown and notebook files

  **Acceptance Criteria**:
  - [ ] All external links verified working
  - [ ] Version pins documented
  - [ ] Link check report created

  **QA Scenarios**:
  ```
  Scenario: Validate all external links
    Tool: Bash (curl) + webfetch
    Steps:
      1. Extract all URLs from documentation
      2. Test each URL with curl HEAD request
      3. Document any broken links
      4. Update or remove broken links
    Expected Result: Zero broken external links
    Evidence: .sisyphus/evidence/link-check-report.md
  ```

  **Commit**: YES (if any link updates needed)

---

- [x] F4. **Final User Validation**

  **What to do**:
  - Present completed documentation to user
  - Get explicit approval on scope and quality
  - Address any final feedback
  - Create handoff summary

  **Must NOT do**:
  - Don't declare complete without user approval

  **Recommended Agent Profile**:
  - N/A — This is a user review task

  **Parallelization**:
  - **Can Run In Parallel**: NO (user review)
  - **Blocks**: Final completion
  - **Blocked By**: Tasks F1-F3

  **Acceptance Criteria**:
  - [ ] User has reviewed all deliverables
  - [ ] User provides explicit "okay" to proceed
  - [ ] Any feedback documented

  **QA Scenarios**:
  ```
  Scenario: Obtain user approval
    Tool: Question (user interaction)
    Steps:
      1. Present completed work summary
      2. Ask user to review deliverables
      3. Get explicit approval or feedback
    Expected Result: User approves or provides actionable feedback
    Evidence: User response captured in conversation
  ```

  **Commit**: NO (user review phase)

---

## Final Verification Wave

After ALL implementation tasks complete:

- [x] **FV1. Execute All Notebooks** — Run end-to-end, capture logs to `.sisyphus/evidence/`
- [x] **FV2. Verify Documentation** — Check accuracy, fix any issues
- [x] **FV3. Validate Links** — All external links must work
- [x] **FV4. User Approval** — Get explicit user "okay" before completing

**All verification tasks COMPLETED. Plan approved.**

---

## Commit Strategy

| Wave | Commit Message Pattern | Files |
|------|------------------------|-------|
| Wave 2 | `docs(training): [description]` | `docs/training/*.md`, `examples/training/*.yaml` |
| Wave 3 | `notebooks: [description]` | `notebooks/*.ipynb` |
| Wave 4 | `docs(training): Fix [issue]` | Any corrections |

**Pre-commit verification**:
- All code examples tested
- All YAML configs validated
- All external links checked

---

## Success Criteria

### Verification Commands

```bash
# Verify all notebooks execute
jupyter nbconvert --to notebook --execute notebooks/training-with-microwakeword.ipynb
jupyter nbconvert --to notebook --execute notebooks/training-with-openwakeword.ipynb
jupyter nbconvert --to notebook --execute notebooks/choosing-your-training-harness.ipynb

# Verify YAML configs
yamllint examples/training/*.yaml

# Check documentation exists
ls -la docs/training/
```

### Final Checklist

- [ ] All 3 notebooks created and tested
- [ ] All documentation complete and accurate
- [ ] Configuration templates ready to use
- [ ] API reference with working examples
- [ ] Troubleshooting guide with FAQ
- [ ] Integration guide with comparison
- [ ] All external links verified working
- [ ] User has provided explicit approval

---

## Appendix: Key Resources

### microWakeWord
- Repository: https://github.com/OHF-Voice/micro-wake-word
- Training Notebook: https://github.com/OHF-Voice/micro-wake-word/blob/main/notebooks/basic_training_notebook.ipynb
- Pre-trained Models: https://github.com/esphome/micro-wake-word-models
- Pre-computed Features: https://huggingface.co/datasets/kahrendt/microwakeword

### openWakeWord
- Repository: https://github.com/dscripka/openWakeWord
- Auto Training Notebook: https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
- Manual Training Notebook: https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb
- Pre-computed Features: https://huggingface.co/datasets/davidscripka/openwakeword_features

### Related Tools
- **openwakeword-trainer**: https://github.com/lgpearson1771/openwakeword-trainer
- **easy-oww**: https://github.com/pjdoland/easy-oww
