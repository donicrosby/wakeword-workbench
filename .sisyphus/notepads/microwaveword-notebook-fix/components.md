# Component Interface Analysis — microwaveword-notebook-fix

**Task:** 2 — Read and understand existing component interfaces  
**Date:** 2026-04-06  
**Files analyzed:** 5 core + 2 supporting

---

## 1. PositiveGenerator (`dataset/positive_generator.py`)

### Class: `PositiveGenerator`

```python
class PositiveGenerator:
    def __init__(self, config: Config, output_dir: Path) -> None:
        """Args:
            config: Config object with wake_word, tts, samples settings.
            output_dir: Directory path for generated WAV files.
        Raises:
            PositiveGeneratorError: On critical init failure.
        """
```

```python
    def generate(self, count: int) -> Path:
        """Returns: Path to generated JSONL manifest file.
        Raises:
            PositiveGeneratorError: On critical generation failure.
        """
```

**Key internals:**
- Gets TTS backend via `get_backend(name)` from registry
- Calls `backend.set_voice(voice)` then `backend.synthesize(phrase)`
- Uses `generate_variants(wake_word)` from `phrase_variants.py`
- Resamples to 16000 Hz mono, saves as WAV via `soundfile`
- Writes JSONL manifest at `<output_dir>/positive_manifest.jsonl`
- Progress via `rich.Progress`

**Manifest entry shape (from generate):**
```python
{
    "path": str(file_path),
    "label": 1,
    "text": phrase,
    "voice": voice,
    "duration_ms": int,
}
```

**Exception:** `PositiveGeneratorError` (custom, extends Exception)

---

## 2. Manifest & ManifestEntry (`dataset/metadata.py`)

### `ManifestEntry` dataclass
```python
@dataclass
class ManifestEntry:
    path: str
    label: int  # 0 or 1 only
    text: str
    voice: str | None = None
    duration_ms: int = 0
    sample_rate: int = 16000
    metadata: dict = field(default_factory=dict)
```

### `Manifest` class
```python
class Manifest:
    def __init__(self, entries: list[ManifestEntry] | None = None) -> None
    def add(self, entry: ManifestEntry) -> None
    def extend(self, entries: list[ManifestEntry]) -> None
    def validate(self) -> list[str]
    def save(self, path: Path | str) -> None  # JSONL
    @classmethod
    def load(cls, path: Path | str) -> Manifest  # from JSONL
    def merge(self, other: Manifest) -> Manifest
    def __len__(self) -> int
    def __iter__(self) -> Iterator[ManifestEntry]
```

**Core data model:** All dataset components produce/consume `Manifest[ManifestEntry]`.

---

## 3. merge_manifests (`dataset/merger.py`)

Two functions:

### `merge()`
```python
def merge(
    pos_manifest: Manifest,
    neg_manifest: Manifest,
    ratio: float | None = None,
    seed: int | None = None,
    validate_files: bool = False,
) -> Manifest:
    """Returns: Combined Manifest with optional ratio balancing."""
    # Raises:
    #   MergerError: Both manifests empty
    #   PathCollisionError: Same path in pos+neg
    #   MergerValidationError: File validation fails
```

### `merge_with_result()`
```python
def merge_with_result(
    pos_manifest: Manifest,
    neg_manifest: Manifest,
    ratio: float | None = None,
    seed: int | None = None,
    validate_files: bool = False,
) -> MergeResult:
    """Returns: MergeResult dataclass with stats."""
```

### `MergeResult` dataclass
```python
@dataclass
class MergeResult:
    manifest: Manifest
    total_entries: int
    positive_count: int
    negative_count: int
    target_ratio: float | None
    actual_ratio: float
    warnings: list[str]
    def has_warnings(self) -> bool: ...
```

**Usage pattern:** `merge()` is the simpler variant; `merge_with_result()` returns detailed stats. Both take two `Manifest` objects, not paths.

**Ratio behavior:**
- `ratio=1.0` → equal pos/neg
- `ratio=2.0` → 2x negatives per positive
- `ratio=None` → use all entries from both

---

## 4. split_manifest (`dataset/splitter.py`)

### `split()`
```python
def split(
    manifest: Manifest,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    by: str = "speaker",
    seed: int | None = None,
) -> tuple[Manifest, Manifest, Manifest]:
    """Returns: (train_manifest, val_manifest, test_manifest)
    Raises:
        SplitValidationError: Ratios don't sum to 1.0 or validation fails.
    """
```

**Key design:**
- Groups by `voice` field (or "unknown" if None) as `SpeakerGroup`
- Stratified split: pure-positive, pure-negative, and mixed groups kept separate
- **Speaker leakage prevention:** Same voice never appears in multiple splits
- Validates no overlapping entries between splits

**Internal helpers (not for external use):**
- `_group_by_speaker(manifest, by)` → `list[SpeakerGroup]`
- `_stratified_split(groups, train, val, test)` → tuple of group lists
- `_build_manifest(groups)` → `Manifest`
- `_validate_split(...)` → raises `SplitValidationError`

---

## 5. generate_confusions (`negatives/phrase_generator.py`)

### `generate_confusions()`
```python
def generate_confusions(
    wake_word: str,
    count: int = 50,
    seed: int | None = None,
    min_similarity: float = 0.6,
) -> list[str]:
    """Returns: List of unique confusion phrase strings.
    Raises:
        ValueError: count < 1 or wake_word empty
        ImportError: jellyfish not installed
    """
```

**Strategies used:**
1. Homophone substitution
2. Vowel substitution
3. Consonant substitution
4. Rhyming word endings
5. Prefix replacement (WAKE_WORD_PREFIXES dict)
6. Consonant cluster variation

**Output:** Returns **phrase strings only** — NOT audio, NOT Manifest entries. Caller must synthesize + create entries.

**Dependency:** `jellyfish` (optional, raises `ImportError` if missing)

### `phonetic_similarity()`
```python
def phonetic_similarity(phrase1: str, phrase2: str) -> float:
    """Returns: Similarity score 0.0–1.0 using metaphone + Jaro-Winkler."""
```

### `get_confusion_count()`
```python
def get_confusion_count(phrase: str) -> int:
    """Returns: Estimated number of unique confusions possible."""
```

---

## 6. generate_synthetic_negatives (`negatives/synthetic_generator.py`)

### `generate_synthetic_negatives()`
```python
def generate_synthetic_negatives(
    count: int,
    word_list: list[str] | None = None,
    seed: int | None = None,
    wake_word: str = "hey assistant",
    min_word_count: int = 2,
    max_word_count: int = 4,
    strategy: str = "random",  # "random" | "sentence" | "topic"
    topics: list[str] | None = None,
) -> list[str]:
    """Returns: List of unique synthetic negative phrase strings.
    Raises:
        SyntheticGeneratorError: Invalid params or generation failure.
    """
```

**Strategies:**
- `"random"` → random word concatenation (default)
- `"sentence"` → subject + verb + object structure
- `"topic"` → draws from `TOPIC_CLUSTERS` (kitchen, office, outdoor, home, technology, weather, emotion, time)

**Output:** Returns **phrase strings only** — NOT audio, NOT Manifest entries. Caller must synthesize + create entries.

**Validation:** Excludes any phrase containing a word from the wake word phrase.

---

## Data Flow / Component Connections

```
Config (YAML)
  │
  ├──► PositiveGenerator.generate(count)
  │         │
  │         ├── generate_variants(wake_word)  [phrase_variants.py]
  │         │         → returns list[str]
  │         │
  │         ├── get_backend(name)  [tts/registry.py]
  │         │         → TTSBackend (synthesize, set_voice)
  │         │
  │         └── writes: <output>/positive_manifest.jsonl
  │                   (list[dict] with path/label/text/voice/duration_ms)
  │
  ├──► generate_confusions(wake_word, count)  [phrase_generator.py]
  │         → returns list[str] (phrase strings only)
  │         (must be synthesized separately)
  │
  ├──► generate_synthetic_negatives(count)  [synthetic_generator.py]
  │         → returns list[str] (phrase strings only)
  │         (must be synthesized separately)
  │
  ├──► TTS Backend (caller's responsibility)
  │         → audio files + ManifestEntry creation
  │
  ├──► Manifest.load(path) / Manifest.save(path)  [metadata.py]
  │
  ├──► merge(pos_manifest, neg_manifest, ratio)  [merger.py]
  │         → Manifest (combined)
  │
  └──► split(manifest, train/val/test)  [splitter.py]
            → (train_manifest, val_manifest, test_manifest)
```

---

## Key Design Observations

1. **Phrase generators return STRINGS, not Manifests.** Caller must synthesize audio and build ManifestEntries.

2. **PositiveGenerator does it all:** TTS synthesis, audio processing, file writing, manifest creation — all in one class.

3. **Negative generators are text-only:** `generate_confusions` and `generate_synthetic_negatives` only produce phrase strings. A separate orchestration step must:
   - Get TTS backend
   - Synthesize each phrase
   - Create ManifestEntry objects
   - Write the negative manifest

4. **This is the orchestration gap.** The `DatasetGenerator` (mentioned in the spec as "Task 7+") needs to bridge:
   - PositiveGenerator (self-contained)
   - Negative phrase generators (text-only → needs synthesis pipeline)
   - Merger
   - Splitter

5. **Manifest is the universal currency:** All components work with `Manifest[ManifestEntry]` objects.

6. **No shared base class for generators.** PositiveGenerator is a class with `generate()`; negative generators are standalone functions. This is an asymmetry to consider in DatasetGenerator design.

7. **TTS is pluggable via registry:** `get_backend(name)` returns a `TTSBackend` instance.

8. **Config drives everything:** All parameters come from `Config` dataclass (YAML-validated).

9. **Splitter is sophisticated:** Groups by speaker/voice to prevent leakage. This is important for training quality.

---

## What DatasetGenerator Needs to Orchestrate

Given the above, `DatasetGenerator` (or equivalent orchestrator) must:

1. Accept `Config` object
2. Call `PositiveGenerator(config, output_dir).generate(count)` → get `Manifest`
3. Call `generate_confusions()` → list[str]
4. Call `generate_synthetic_negatives()` → list[str]
5. **Synthesize negative phrases** using TTS backend (step missing from current components)
6. Build negative `Manifest` from synthesized audio
7. Call `merge()` to combine pos + neg manifests
8. Call `split()` to create train/val/test splits
9. Save splits to disk (JSONL)

Steps 5-6 are the gap — negative generators give strings, someone needs to TTS them.
