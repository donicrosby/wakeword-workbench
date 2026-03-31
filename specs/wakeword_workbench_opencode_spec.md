# Wake Word Workbench — Implementation Spec (oh-my-opencode Ready)

## 🎯 Project Goal
Build a modular, reproducible pipeline for generating datasets and evaluating wake word models (microWakeWord, openWakeWord).

---

## ⚙️ Tech Constraints
- Language: Python 3.11+
- CLI: typer
- Audio: numpy, scipy, librosa, soundfile
- Config: pyyaml
- Data format: JSONL manifests

---

## 📁 Target Repo Structure

src/wakeword_workbench/
- cli.py
- config.py
- tts/
- augment/
- dataset/
- negatives/
- eval/
- export/

---

## 🚀 Phase 1 — CLI + Config

### Tasks
- Create CLI using typer
- Implement config loader (YAML)
- Add command: `wakeword run config.yaml`

### Acceptance Criteria
- CLI runs without crashing
- Config loads correctly
- Prints parsed config

---

## 🚀 Phase 2 — TTS System

### Tasks
- Create base TTS interface
- Implement Kokoro backend
- Implement Piper backend
- Add registry for engines

### Acceptance Criteria
- Can generate WAV files from both engines
- Outputs metadata per file

---

## 🚀 Phase 3 — Positive Dataset Generation

### Tasks
- Generate phrase variants
- Loop through TTS engines
- Save WAV + metadata

### Acceptance Criteria
- 1000+ samples generated
- JSONL manifest created

---

## 🚀 Phase 4 — Negative Dataset

### Tasks
- Implement phrase confusion generator
- Add synthetic negative generation
- Add config-driven phrases

### Acceptance Criteria
- Confusion phrases generated
- Negatives >= 5x positives

---

## 🚀 Phase 5 — Augmentation Pipeline

### Tasks
- Add noise transform
- Add reverb
- Add gain + clipping
- Add silence padding

### Acceptance Criteria
- Each file produces augmented variants
- Audio still valid

---

## 🚀 Phase 6 — Dataset Builder

### Tasks
- Merge positives + negatives
- Split into train/val/test
- Prevent leakage

### Acceptance Criteria
- Splits created correctly
- No duplicate entries

---

## 🚀 Phase 7 — Export

### Tasks
- Export dataset for:
  - microWakeWord
  - openWakeWord

### Acceptance Criteria
- Output folder usable by trainers

---

## 🚀 Phase 8 — Evaluation

### Tasks
- Implement FAR calculation
- Implement FRR calculation
- Add threshold sweep

### Acceptance Criteria
- Report generated with metrics

---

## 🚀 Phase 9 — Hard Negative Mining

### Tasks
- Run model on long audio
- Extract false positives
- Save clips
- Add back into dataset

### Acceptance Criteria
- False positives saved
- Can retrain with new negatives

---

## 📊 Definition of Done

- CLI can run full pipeline end-to-end
- Dataset reproducible from config
- Evaluation produces FAR/FRR
- Hard negatives improve results

---

## ❌ Out of Scope

- GUI
- Cloud deployment
- Real-time inference engine

---

## 🧠 Key Principle

> Data quality and negative coverage matter more than model architecture.

---

## 📌 Final Note

This is a **dataset + evaluation system**, not just a training script.
