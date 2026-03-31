# Wave 3 - Task 15: Synthetic Negative Generator

## Task Summary
Created `src/wakeword_workbench/negatives/synthetic_generator.py` with `generate_synthetic_negatives()` function.

## Implementation Details

### Core Function
- `generate_synthetic_negatives()` - Main function that generates random word concatenations
- Parameters: count, word_list, seed, wake_word, min_word_count, max_word_count, strategy, topics

### Generation Strategies
1. **random** (default) - Random word selection with configurable word count
2. **sentence** - Article + noun + verb structure
3. **topic** - Uses topic clusters (kitchen, office, outdoor, etc.)

### Key Features
- **Wake word exclusion**: Words from wake word phrase are excluded from generation
- **Uniqueness**: All generated phrases are unique
- **Determinism**: Seed parameter ensures reproducible results
- **Custom word lists**: Users can provide their own word lists
- **Topic clusters**: 8 predefined topic clusters for variety

### Default Word Lists
- Articles: the, a, an, some, any
- Pronouns: I, you, he, she, it, we, they, etc.
- Prepositions: in, on, at, by, for, with, etc.
- Verbs: is, are, was, were, have, has, do, does, etc.
- Nouns: Common English nouns

### Topic Clusters
- kitchen: table, chair, plate, cooking, dinner, etc.
- office: desk, computer, monitor, meeting, report, etc.
- outdoor: park, street, tree, mountain, river, etc.
- home: bedroom, bathroom, window, couch, lamp, etc.
- technology: phone, tablet, laptop, network, app, etc.
- weather: sunny, cloudy, rainy, snowy, windy, etc.
- emotion: happy, sad, angry, scared, excited, etc.
- time: second, minute, hour, day, week, month, etc.

## Testing
Created `tests/test_synthetic_generator.py` with 31 tests covering:
- Correct count generation
- Uniqueness
- Wake word exclusion
- Seed determinism
- Custom word lists
- Error handling
- All strategies (random, sentence, topic)
- Edge cases

## Key Patterns Learned
1. Use `random.Random()` for seeded randomness
2. Build default words from multiple categories for variety
3. Exclude wake word words using set intersection
4. Track seen phrases in lowercase for case-insensitive uniqueness

## Files Created
- `src/wakeword_workbench/negatives/synthetic_generator.py`
- `tests/test_synthetic_generator.py`

## Verification
```bash
uv run pytest tests/test_synthetic_generator.py -v
# Result: 31 passed
```
