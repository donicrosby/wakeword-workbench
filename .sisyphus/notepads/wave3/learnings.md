# Wave 3 Learnings

## Task 13: Positive Dataset Generator

### Key Patterns Discovered

1. **TTS Integration**: Use `wakeword_workbench.tts.registry.get_backend()` to get TTS backend instance
2. **Backend Interface**: TTS backends implement `synthesize(text)`, `set_voice(voice)`, `list_voices()` methods
3. **TTSResult**: Returns audio (numpy float32), sample_rate, duration
4. **Audio Format**: Ensure 16000 Hz mono WAV using `_ensure_format()` and `_ensure_mono()` helpers
5. **Rich Progress**: Use `Progress` with `SpinnerColumn`, `TextColumn`, `BarColumn`, `TaskProgressColumn` for progress bars

### Error Handling Strategy
- Continue on TTS synthesis failures (log warning)
- Stop on critical errors (backend init failure)
- Log all errors with context (phrase, voice)

### Manifest Fields
- `path`: relative path to audio file
- `label`: 1 (positive)
- `text`: the phrase spoken
- `voice`: voice/speaker ID
- `duration_ms`: audio duration in milliseconds

### File Naming Convention
- `{wake_word}_{voice}_{index:04d}.wav`
- Example: `hey_vera_af_sarah_0000.wav`

### Testing Approach
- Mock `get_backend()` to avoid real TTS calls
- Mock `generate_variants()` for deterministic tests
- Use `TTSResult` fixture for valid audio data
- Verify WAV files are 16kHz mono with soundfile

### Dependencies
- `soundfile` for WAV writing
- `librosa` for resampling
- `rich` for progress bars
- `numpy` for audio processing

---

## Task 14: Negative Phrase Generator (Confusions)

### Key Patterns Discovered

1. **Phonetic Similarity**: Use `jellyfish` library with metaphone for phonetic matching
2. **jellyfish API**: `jellyfish.metaphone(word)` for phoneme encoding, `jellyfish.jaro_winkler_similarity(str1, str2)` for similarity scoring
3. **RandomProvider Protocol**: Define interface for deterministic random with `choice`, `sample`, `randint`, `shuffle` methods
4. **Strategy Pattern**: Multiple substitution strategies (homophone, vowel, consonant, rhyming, prefix, cluster)
5. **Validation**: Ensure no exact matches, phonetic similarity > 0.6, all results unique

### Confusion Strategies Implemented
- **Homophone substitution**: Replace words with phonetic equivalents ("hey" → "hay")
- **Vowel substitution**: Change vowels to similar-sounding alternatives
- **Consonant substitution**: Replace consonants with similar-sounding ones
- **Rhyming substitution**: Replace word endings with rhyming patterns
- **Prefix substitution**: Replace wake word prefixes with alternatives
- **Consonant cluster variation**: Substitute consonant clusters

### API Design
```python
generate_confusions(wake_word, count=50, seed=None, min_similarity=0.6) -> list[str]
phonetic_similarity(phrase1, phrase2) -> float  # 0.0-1.0
get_confusion_count(phrase) -> int
```

### Dependencies
- `jellyfish>=1.0` for metaphone and Jaro-Winkler similarity

### Testing Coverage (31 tests)
- Phonetic similarity: identical, case-insensitive, homophones, different phrases, empty
- DeterministicRandom: choice, sample, randint
- generate_confusions: basic, no exact match, unique, min_similarity, deterministic, edge cases
- Import error handling for missing jellyfish
