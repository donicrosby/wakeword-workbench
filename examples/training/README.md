# Training Configuration Templates

This directory contains ready-to-use training configurations for both microWakeWord and openWakeWord training harnesses.

## Setup checkpoint

Before using these templates, run this from the repository root:

```bash
uv sync --group dev
source .venv/bin/activate
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If your selected configuration uses a missing backend, install `uv sync --extra kokoro` or `uv sync --extra piper`.

## Quick Start

Choose the configuration that matches your goal:

| Scenario | microWakeWord | openWakeWord |
|----------|--------------|--------------|
| Test pipeline (5-15 min) | `microwakeword-quick-test.yaml` | `openwakeword-quick-test.yaml` |
| Production model (30+ min) | `microwakeword-production.yaml` | `openwakeword-production.yaml` |
| Adapt to new data (15-20 min) | `microwakeword-finetune.yaml` | `openwakeword-finetune.yaml` |

## Configuration Selection Guide

### When to Use Quick Test

- **Validating the training pipeline** before committing to long training
- **Testing new augmentation strategies** on small scale
- **Debugging** configuration or data issues
- **CI/CD pipelines** that need fast feedback

**Expected runtime:** 5-15 minutes  
**Dataset size:** 100-500 positive samples

### When to Use Production

- **Deploying a model** for real-world use
- **Final training** after iterating with quick-test
- **Comprehensive evaluation** requiring robust model

**Expected runtime:** 30+ minutes (openWakeWord: 1+ hour)  
**Dataset size:** 10,000+ positive samples, diverse negatives

### When to Use Fine-tuning

- **Adding new TTS voices** to existing model
- **Domain adaptation** (e.g., specific accent, noise environment)
- **Iterative improvement** based on observed failures
- **Quick updates** without full retraining

**Expected runtime:** 15-20 minutes  
**Dataset size:** 1,000-5,000 new samples + original data

## Configuration Parameters

### microWakeWord Key Parameters

| Parameter | Purpose | Quick Test | Production | Fine-tune |
|-----------|---------|------------|------------|-----------|
| `training_steps` | Total optimization steps | 3,000 | 50,000 | 15,000 |
| `learning_rate` | Optimization rate | 0.001 | 0.001 | **0.0001** |
| `batch_size` | Samples per batch | 64 | 128 | 64 |
| `negative_class_weight` | FP rejection strength | 10 | 20 | 15 |
| `pointwise_filters` | Model capacity | 32 | 64 | 64 |

### openWakeWord Key Parameters

| Parameter | Purpose | Quick Test | Production | Fine-tune |
|-----------|---------|------------|------------|-----------|
| `steps` | Base training steps | 5,000 | 50,000 | 10,000 |
| `max_negative_weight` | FP penalty cap | 500 | 1500 | 800 |
| `target_fp_per_hour` | FP goal | 0.5 | 0.2 | 0.3 |
| `layer_size` | Hidden layer width | 32 | 64 | 64 |

## Before Training

### 1. Generate Training Data

Use wakeword-workbench to generate your dataset:

```bash
# Create dataset configuration
cat > config.yaml << 'EOF'
wake_word: "hey vera"
samples:
  positives: 1000
  negatives_multiplier: 10
tts:
  providers:
    - backend: "kokoro"
      voices: ["af_sarah", "am_adam"]
      speed: 1.0
augmentation:
  noise_snr: [-15, 5]
  reverb_probability: 0.5
  gain_range: [-45, 0]
output:
  path: "./output/my_dataset"
  format: ["microwakeword", "openwakeword"]
EOF

# Generate dataset
uv run wakeword-workbench run config.yaml
```

### 2. Convert to Harness Format

**For microWakeWord:**
- Features must be in RaggedMmap format (`*_mmap/` directories)
- Each split (training/, validation/, testing/) contains feature directories

**For openWakeWord:**
- Convert audio clips to embedded features (16x96):

```python
from openwakeword.utils import AudioFeatures
import numpy as np

# Load your audio clips
audio = np.load("clips.npy")  # Shape: (N, 32000) at 16kHz

# Generate embeddings
fe = AudioFeatures()
features = fe.embed_clips(audio, batch_size=256)  # Shape: (N, 16, 96)

# Save per-class
np.save("positive_features.npy", features[labels == 1])
np.save("negative_features.npy", features[labels == 0])
```

### 3. Update Config Paths

Edit the config files to point to your data:

```yaml
# microWakeWord
train_dir: "/path/to/your/data"

features:
  - features_dir: "/path/to/your/data/wakeword_mmap"
    truth: true
    ...

# openWakeWord
feature_data_files:
  positive: "/path/to/your/positive_features.npy"
  adversarial_negative: "/path/to/your/negative_features.npy"
```

### 4. Update Model Names

```yaml
# openWakeWord
model_name: "my_wake_word_model"
target_phrase: ["my phrase"]
```

## Training Commands

### microWakeWord

```bash
python -m microwakeword.model_train_eval \
  --training_config examples/training/microwakeword-production.yaml \
  --train 1 \
  --test_tflite_streaming_quantized 1
```

### openWakeWord

```bash
python -m openwakeword.train \
  --config examples/training/openwakeword-production.yaml \
  --generate_clips \
  --augment_clips \
  --train_model
```

## Key Differences Between Harnesses

| Aspect | microWakeWord | openWakeWord |
|--------|--------------|--------------|
| **Feature format** | Microfrontend 40-dim | Google embeddings 96-d |
| **Model output** | TFLite (streaming) | ONNX |
| **Training approach** | Single-stage with metrics | 3-stage with FP/hr targeting |
| **Best for** | Embedded/microcontroller | General deployment |
| **Memory footprint** | Smaller | Larger |

## Troubleshooting

### microWakeWord

**Error: "No mmap directories found"**
- Ensure feature directories are named `*_mmap/`
- Check that `train_dir` contains `training/`, `validation/`, `testing/` subdirs

**Low recall on validation**
- Increase `positive_class_weight`
- Add more diverse positive samples
- Reduce `negative_class_weight`

### openWakeWord

**High false positive rate**
- Increase `max_negative_weight`
- Add more adversarial negatives
- Include external negative corpus (ACAV100M)

**Training diverges**
- Reduce `max_negative_weight`
- Lower learning rate
- Increase `batch_n_per_class` for positives

## Further Reading

- [microWakeWord Documentation](https://github.com/kahrendt/microWakeWord)
- [openWakeWord Documentation](https://github.com/dscripka/openWakeWord)
- [WakeWord Workbench README](../README.md)
