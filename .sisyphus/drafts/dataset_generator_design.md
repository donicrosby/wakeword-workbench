# DatasetGenerator Interface Design

**Task:** 3 — Design DatasetGenerator interface  
**Date:** 2026-04-06  
**Based on:** components.md (Task 2 output)

---

## 1. Design Goals

1. **Orchestration hub** — Coordinates all generation components
2. **Handles negative TTS synthesis** — Closes the gap identified in component analysis
3. **Unified interface** — Single `generate()` entry point
4. **Rich feedback** — Detailed `GenerationResult` with stats
5. **Error clarity** — Specific `GeneratorError` hierarchy

---

## 2. GeneratorError Exception Hierarchy

```python
class GeneratorError(Exception):
    """Base exception for DatasetGenerator errors."""
    pass


class GeneratorConfigError(GeneratorError):
    """Configuration validation failed."""
    pass


class GeneratorTTSError(GeneratorError):
    """TTS synthesis failed during negative generation."""
    pass


class GeneratorMergeError(GeneratorError):
    """Manifest merge failed."""
    pass


class GeneratorSplitError(GeneratorError):
    """Manifest split failed."""
    pass


class GeneratorIOError(GeneratorError):
    """File I/O operation failed."""
    pass
```

---

## 3. GenerationResult Dataclass

```python
@dataclass
class GenerationResult:
    """Result of a dataset generation run."""
    
    # Output manifests
    train_manifest: Manifest
    val_manifest: Manifest
    test_manifest: Manifest
    
    # Counts
    total_positives: int
    total_negatives: int
    total_entries: int
    
    # Split breakdown
    train_count: int
    val_count: int
    test_count: int
    
    # Ratio info
    target_ratio: float | None
    actual_ratio: float
    
    # Metadata
    output_dir: Path
    generation_time_seconds: float
    
    # Warnings (non-fatal issues)
    warnings: list[str]
    
    def has_warnings(self) -> bool:
        """Check if any warnings were generated."""
        return len(self.warnings) > 0
    
    def summary(self) -> dict[str, Any]:
        """Return a summary dict for logging/reporting."""
        return {
            "total_entries": self.total_entries,
            "positives": self.total_positives,
            "negatives": self.total_negatives,
            "ratio": self.actual_ratio,
            "splits": {
                "train": self.train_count,
                "val": self.val_count,
                "test": self.test_count,
            },
            "output_dir": str(self.output_dir),
            "warnings": len(self.warnings),
        }
```

---

## 4. DatasetGenerator Class

### 4.1 Public Interface

```python
class DatasetGenerator:
    """Orchestrates the full dataset generation pipeline.
    
    Coordinates PositiveGenerator, negative phrase generation,
    TTS synthesis, merging, and splitting into a unified workflow.
    
    Example:
        >>> config = load_config("config.yaml")
        >>> generator = DatasetGenerator(config)
        >>> result = generator.generate()
        >>> print(f"Generated {result.total_entries} samples")
    """
    
    def __init__(
        self,
        config: Config,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize DatasetGenerator.
        
        Args:
            config: Validated Config object from load_config().
            output_dir: Override output directory. Defaults to config.output.path.
            
        Raises:
            GeneratorConfigError: If config is invalid for generation.
        """
```

### 4.2 Main Generation Method

```python
    def generate(
        self,
        positive_count: int | None = None,
        negatives_multiplier: int | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        ratio: float | None = None,
        validate_files: bool = False,
    ) -> GenerationResult:
        """Generate a complete dataset with train/val/test splits.
        
        This is the main entry point. It orchestrates:
        1. Positive sample generation via PositiveGenerator
        2. Negative phrase generation (confusions + synthetic)
        3. TTS synthesis for negative phrases
        4. Manifest merging
        5. Stratified train/val/test split
        
        Args:
            positive_count: Override positive sample count. 
                           Defaults to config.samples.positives.
            negatives_multiplier: Override negatives multiplier.
                                 Defaults to config.samples.negatives_multiplier.
            train_ratio: Fraction for training split (default: 0.7).
            val_ratio: Fraction for validation split (default: 0.15).
            test_ratio: Fraction for test split (default: 0.15).
            ratio: Target pos:neg ratio for merge. 
                   None = use all negatives.
            validate_files: Whether to validate audio files exist.
                           Slower but catches missing files.
        
        Returns:
            GenerationResult with all manifests and statistics.
            
        Raises:
            GeneratorConfigError: Invalid parameters.
            GeneratorTTSError: TTS synthesis failed.
            GeneratorMergeError: Manifest merge failed.
            GeneratorSplitError: Split validation failed.
            GeneratorIOError: File write/read failed.
        """
```

### 4.3 Private Helper Methods

```python
    # --- Phase 1: Positive Generation ---
    
    def _generate_positives(self, count: int) -> Manifest:
        """Generate positive samples via PositiveGenerator.
        
        Args:
            count: Number of positive samples.
            
        Returns:
            Manifest with positive entries.
            
        Raises:
            GeneratorIOError: PositiveGenerator failed.
        """
    
    # --- Phase 2: Negative Phrase Generation ---
    
    def _generate_negative_phrases(self, total_count: int) -> list[str]:
        """Generate negative phrase strings.
        
        Combines confusion phrases and synthetic negatives.
        Does NOT synthesize audio - that's in _synthesize_negatives().
        
        Args:
            total_count: Total number of negative phrases needed.
            
        Returns:
            List of unique negative phrase strings.
        """
    
    def _generate_confusion_phrases(self, count: int) -> list[str]:
        """Generate confusion phrases.
        
        Args:
            count: Number of confusion phrases to generate.
            
        Returns:
            List of confusion phrase strings.
            
        Raises:
            GeneratorError: If jellyfish not installed.
        """
    
    def _generate_synthetic_phrases(self, count: int) -> list[str]:
        """Generate synthetic negative phrases.
        
        Args:
            count: Number of synthetic phrases to generate.
            
        Returns:
            List of synthetic phrase strings.
        """
    
    # --- Phase 3: Negative TTS Synthesis ---
    
    def _synthesize_negatives(self, phrases: list[str]) -> Manifest:
        """Synthesize audio for negative phrases and create Manifest.
        
        This is the key bridging method that closes the gap identified
        in the component analysis. Negative generators return strings,
        this method synthesizes audio and builds ManifestEntries.
        
        Args:
            phrases: List of negative phrase strings to synthesize.
            
        Returns:
            Manifest with negative entries (label=0).
            
        Raises:
            GeneratorTTSError: TTS synthesis failed.
        """
    
    def _get_tts_backend(self) -> TTSBackend:
        """Get configured TTS backend.
        
        Returns:
            TTSBackend instance from registry.
            
        Raises:
            GeneratorConfigError: Backend not available.
        """
    
    def _select_voice_for_phrase(self, phrase: str) -> str:
        """Select a voice for synthesizing a phrase.
        
        Uses round-robin across configured voices for diversity.
        
        Args:
            phrase: The phrase to be synthesized.
            
        Returns:
            Voice identifier string.
        """
    
    # --- Phase 4: Merge ---
    
    def _merge_manifests(
        self,
        pos_manifest: Manifest,
        neg_manifest: Manifest,
        ratio: float | None,
    ) -> Manifest:
        """Merge positive and negative manifests.
        
        Args:
            pos_manifest: Positive samples manifest.
            neg_manifest: Negative samples manifest.
            ratio: Target ratio (pos:neg). None = use all.
            
        Returns:
            Combined manifest.
            
        Raises:
            GeneratorMergeError: Merge failed.
        """
    
    # --- Phase 5: Split ---
    
    def _split_manifest(
        self,
        manifest: Manifest,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple[Manifest, Manifest, Manifest]:
        """Split manifest into train/val/test.
        
        Args:
            manifest: Combined manifest to split.
            train_ratio: Training fraction.
            val_ratio: Validation fraction.
            test_ratio: Test fraction.
            
        Returns:
            Tuple of (train, val, test) manifests.
            
        Raises:
            GeneratorSplitError: Split validation failed.
        """
    
    # --- Phase 6: Save ---
    
    def _save_splits(
        self,
        train: Manifest,
        val: Manifest,
        test: Manifest,
    ) -> None:
        """Save split manifests to output directory.
        
        Files:
            <output_dir>/train.jsonl
            <output_dir>/val.jsonl
            <output_dir>/test.jsonl
        
        Args:
            train: Training manifest.
            val: Validation manifest.
            test: Test manifest.
            
        Raises:
            GeneratorIOError: File write failed.
        """
    
    # --- Utility ---
    
    def _ensure_output_dir(self) -> Path:
        """Ensure output directory exists.
        
        Returns:
            Path to output directory.
        """
    
    def _log_progress(self, phase: str, message: str) -> None:
        """Log progress with structured logger."""
```

---

## 5. Orchestration Flow

```
generate()
│
├── _ensure_output_dir()
│
├── Phase 1: POSITIVE GENERATION
│   │
│   └── _generate_positives(count)
│       │
│       └── PositiveGenerator(config, output_dir).generate(count)
│           │
│           └── Returns: Manifest (from positive_manifest.jsonl)
│
├── Phase 2: NEGATIVE PHRASE GENERATION
│   │
│   └── _generate_negative_phrases(total_count)
│       │
│       ├── _generate_confusion_phrases(~60%)
│       │   └── generate_confusions(wake_word, count)
│       │       └── Returns: list[str]
│       │
│       └── _generate_synthetic_phrases(~40%)
│           └── generate_synthetic_negatives(count, ...)
│               └── Returns: list[str]
│
├── Phase 3: NEGATIVE TTS SYNTHESIS
│   │
│   └── _synthesize_negatives(phrases)
│       │
│       ├── _get_tts_backend()
│       │   └── get_backend(config.tts.backend)
│       │
│       ├── _select_voice_for_phrase(phrase)  [round-robin]
│       │
│       └── For each phrase:
│           ├── backend.set_voice(voice)
│           ├── backend.synthesize(phrase)
│           ├── Resample to 16000 Hz mono
│           └── Save WAV → Create ManifestEntry(label=0)
│       │
│       └── Returns: Manifest (negative entries)
│
├── Phase 4: MERGE
│   │
│   └── _merge_manifests(pos_manifest, neg_manifest, ratio)
│       │
│       └── merge_with_result(pos, neg, ratio, validate_files)
│           │
│           └── Returns: MergeResult → Manifest
│
├── Phase 5: SPLIT
│   │
│   └── _split_manifest(manifest, train, val, test)
│       │
│       └── split(manifest, train, val, test, by="speaker")
│           │
│           └── Returns: (train, val, test) Manifests
│
├── Phase 6: SAVE
│   │
│   └── _save_splits(train, val, test)
│       │
│       ├── train.save(output_dir / "train.jsonl")
│       ├── val.save(output_dir / "val.jsonl")
│       └── test.save(output_dir / "test.jsonl")
│
└── Build GenerationResult and return
```

---

## 6. Configuration Integration

The `Config` dataclass should provide:

```python
@dataclass
class Config:
    wake_word: str
    samples: SamplesConfig
    tts: TTSConfig
    output: OutputConfig
    # ... augmentation, etc.

@dataclass
class SamplesConfig:
    positives: int
    negatives_multiplier: int

@dataclass  
class TTSConfig:
    backend: str  # "kokoro" | "piper"
    voices: list[str]
    speed: float = 1.0

@dataclass
class OutputConfig:
    path: Path
    # format, etc.
```

---

## 7. Usage Examples

### Basic Usage

```python
from pathlib import Path
from wakeword_workbench.config import load_config
from wakeword_workbench.dataset.generator import DatasetGenerator

# Load config
config = load_config("config.yaml")

# Generate dataset
generator = DatasetGenerator(config)
result = generator.generate()

# Access results
print(f"Train: {result.train_count} samples")
print(f"Val: {result.val_count} samples")
print(f"Test: {result.test_count} samples")
```

### With Custom Parameters

```python
# Override counts and ratios
result = generator.generate(
    positive_count=500,
    negatives_multiplier=10,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    ratio=3.0,  # 3 negatives per positive
    validate_files=True,
)
```

### Handle Errors

```python
from wakeword_workbench.dataset.generator import (
    GeneratorError,
    GeneratorTTSError,
    GeneratorConfigError,
)

try:
    result = generator.generate()
except GeneratorConfigError as e:
    print(f"Invalid configuration: {e}")
except GeneratorTTSError as e:
    print(f"TTS synthesis failed: {e}")
except GeneratorError as e:
    print(f"Generation failed: {e}")
```

---

## 8. File Location

```
src/wakeword_workbench/dataset/
├── __init__.py           # Exports: DatasetGenerator, GenerationResult, GeneratorError
├── positive_generator.py # Existing
├── metadata.py          # Existing: Manifest, ManifestEntry
├── merger.py             # Existing: merge, merge_with_result
├── splitter.py           # Existing: split
├── generator.py          # NEW: DatasetGenerator (this design)
└── generator_errors.py   # NEW: GeneratorError hierarchy
```

---

## 9. Dependencies

- `Config` from `wakeword_workbench.config`
- `Manifest`, `ManifestEntry` from `wakeword_workbench.dataset.metadata`
- `PositiveGenerator` from `wakeword_workbench.dataset.positive_generator`
- `merge_with_result` from `wakeword_workbench.dataset.merger`
- `split` from `wakeword_workbench.dataset.splitter`
- `generate_confusions` from `wakeword_workbench.negatives.phrase_generator`
- `generate_synthetic_negatives` from `wakeword_workbench.negatives.synthetic_generator`
- `get_backend` from `wakeword_workbench.tts.registry`
- `TTSBackend` from `wakeword_workbench.tts.base`
- `get_logger` from `wakeword_workbench.logging_config`

---

## 10. Design Decisions

1. **Separate error types** — Easier to catch specific failures
2. **GenerationResult as return value** — Rich feedback without side effects
3. **Private helper methods** — Clear separation of concerns
4. **Config passthrough** — Don't duplicate validation
5. **TTS synthesis in DatasetGenerator** — Closes the orchestration gap
6. **Round-robin voice selection** — Simple, ensures voice diversity
7. **Progress logging** — Rich integration for observability
