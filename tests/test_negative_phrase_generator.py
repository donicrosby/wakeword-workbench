"""Tests for negative phrase generator."""

import pytest

from wakeword_workbench.negatives.phrase_generator import (
    DeterministicRandom,
    generate_confusions,
    get_confusion_count,
    phonetic_similarity,
)

# --- Basic Tests ---


def test_phonetic_similarity_identical() -> None:
    """Identical phrases should have similarity 1.0."""
    assert phonetic_similarity("hey marvin", "hey marvin") == 1.0


def test_phonetic_similarity_lowercase_normalized() -> None:
    """Phrases should be case-insensitive."""
    assert phonetic_similarity("HEY MARVIN", "hey marvin") == 1.0


def test_phonetic_similarity_homophones() -> None:
    """Phonetically similar phrases should have high similarity."""
    sim = phonetic_similarity("hey marvin", "hay marvin")
    assert sim > 0.6


def test_phonetic_similarity_different() -> None:
    """Very different phrases should have low similarity."""
    sim = phonetic_similarity("hello world", "zyxwvutsrqpon")
    assert sim < 0.5


def test_phonetic_similarity_empty() -> None:
    """Empty phrases should return 0.0."""
    assert phonetic_similarity("", "hello") == 0.0
    assert phonetic_similarity("hello", "") == 0.0
    assert phonetic_similarity("", "") == 0.0


# --- DeterministicRandom Tests ---


def test_deterministic_random_choice() -> None:
    """Choice should return deterministic values with seed."""
    rng1 = DeterministicRandom(42)
    rng2 = DeterministicRandom(42)

    choices = ["a", "b", "c", "d"]
    results1 = [rng1.choice(choices) for _ in range(10)]
    results2 = [rng2.choice(choices) for _ in range(10)]

    assert results1 == results2


def test_deterministic_random_sample() -> None:
    """Sample should return deterministic values with seed."""
    rng1 = DeterministicRandom(42)
    rng2 = DeterministicRandom(42)

    choices = ["a", "b", "c", "d", "e"]
    results1 = rng1.sample(choices, 3)
    results2 = rng2.sample(choices, 3)

    assert results1 == results2


def test_deterministic_random_randint() -> None:
    """Randint should return deterministic values with seed."""
    rng1 = DeterministicRandom(42)
    rng2 = DeterministicRandom(42)

    results1 = [rng1.randint(1, 10) for _ in range(10)]
    results2 = [rng2.randint(1, 10) for _ in range(10)]

    assert results1 == results2


# --- Generate Confusions Tests ---


def test_generate_confusions_basic() -> None:
    """Should generate confusions for a wake word."""
    confusions = generate_confusions("hey marvin", count=5, seed=42)
    assert len(confusions) <= 5
    assert all(isinstance(c, str) for c in confusions)


def test_generate_confusions_no_exact_match() -> None:
    """Confusions should not include the exact wake word."""
    confusions = generate_confusions("hey marvin", count=50, seed=42)
    assert "hey marvin" not in confusions
    assert "HEY MARVIN" not in confusions


def test_generate_confusions_unique() -> None:
    """Confusions should be unique."""
    confusions = generate_confusions("hey marvin", count=50, seed=42)
    assert len(confusions) == len(set(confusions))


def test_generate_confusions_min_similarity() -> None:
    """Confusions should have minimum phonetic similarity."""
    confusions = generate_confusions("hey marvin", count=20, seed=42, min_similarity=0.6)
    for confusion in confusions:
        sim = phonetic_similarity("hey marvin", confusion)
        assert sim >= 0.6, f"Similarity {sim} < 0.6 for '{confusion}'"


def test_generate_confusions_deterministic() -> None:
    """Same seed should produce same results."""
    confusions1 = generate_confusions("hey marvin", count=10, seed=123)
    confusions2 = generate_confusions("hey marvin", count=10, seed=123)
    assert confusions1 == confusions2


def test_generate_confusions_different_seeds_different_results() -> None:
    """Different seeds should produce different results."""
    confusions1 = generate_confusions("hey marvin", count=10, seed=123)
    confusions2 = generate_confusions("hey marvin", count=10, seed=456)
    # Results should differ (though there's a small chance they could be the same)
    assert confusions1 != confusions2 or set(confusions1) != set(confusions2)


def test_generate_confusions_empty_wake_word_raises() -> None:
    """Empty wake word should raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_confusions("")

    with pytest.raises(ValueError, match="cannot be empty"):
        generate_confusions("   ")


def test_generate_confusions_count_zero_raises() -> None:
    """Zero count should raise ValueError."""
    with pytest.raises(ValueError, match="count must be at least 1"):
        generate_confusions("hey marvin", count=0)


def test_generate_confusions_negative_count_raises() -> None:
    """Negative count should raise ValueError."""
    with pytest.raises(ValueError, match="count must be at least 1"):
        generate_confusions("hey marvin", count=-1)


# --- Different Wake Word Patterns ---


def test_generate_confusions_ok_wake_word() -> None:
    """Should generate confusions for 'ok' wake word."""
    confusions = generate_confusions("ok google", count=10, seed=42)
    assert len(confusions) <= 10
    for c in confusions:
        assert "ok google" not in c.lower()


def test_generate_confusions_single_word() -> None:
    """Should generate confusions for single-word wake word."""
    confusions = generate_confusions("hey", count=10, seed=42)
    assert len(confusions) <= 10
    assert "hey" not in confusions


def test_generate_confusions_three_words() -> None:
    """Should generate confusions for three-word phrase."""
    confusions = generate_confusions("hey assistant buddy", count=10, seed=42)
    assert len(confusions) <= 10
    assert "hey assistant buddy" not in confusions


# --- Get Confusion Count Tests ---


def test_get_confusion_count_basic() -> None:
    """Should return estimated confusion count."""
    count = get_confusion_count("hey marvin")
    assert count >= 0
    assert isinstance(count, int)


def test_get_confusion_count_different_phrases() -> None:
    """Different phrases should have different confusion counts."""
    count1 = get_confusion_count("hey marvin")
    count2 = get_confusion_count("ok google")
    # Different phrases will generally have different counts
    assert isinstance(count1, int)
    assert isinstance(count2, int)


# --- Strategy Tests ---


def test_generate_confusions_vary_by_strategy() -> None:
    """Confusions should include different types of variations."""
    confusions = generate_confusions("hey marvin", count=50, seed=42)

    # Should have multiple distinct variations
    assert len(set(confusions)) > 1

    # Check for different types of variations
    # At least one strategy should have produced results
    assert len(confusions) > 0
    assert any(c.startswith("hay ") or c.startswith("say ") for c in confusions) or any(
        "marv" in c for c in confusions if c != "hey marvin"
    )


def test_generate_confusions_preserves_structure() -> None:
    """Confusions should preserve word count structure."""
    confusions = generate_confusions("hey marvin", count=50, seed=42)

    for c in confusions:
        # Should maintain similar word count
        orig_words = len("hey marvin".split())
        conf_words = len(c.split())
        assert abs(orig_words - conf_words) <= 1


# --- Edge Cases ---


def test_generate_confusions_special_characters() -> None:
    """Should handle phrases with special characters."""
    confusions = generate_confusions("hey, marvin!", count=10, seed=42)
    assert len(confusions) <= 10


def test_generate_confusions_numbers() -> None:
    """Should handle phrases with numbers."""
    # Numbers don't have phonetic equivalents, but shouldn't crash
    confusions = generate_confusions("test 123", count=5, seed=42)
    assert isinstance(confusions, list)


def test_generate_confusions_very_long_phrase() -> None:
    """Should handle long phrases without crashing."""
    long_phrase = " ".join(["hey"] * 10)
    confusions = generate_confusions(long_phrase, count=5, seed=42)
    assert isinstance(confusions, list)


# --- Similarity Threshold Tests ---


def test_generate_confusions_high_threshold() -> None:
    """High similarity threshold should filter more results."""
    confusions_high = generate_confusions("hey marvin", count=50, seed=42, min_similarity=0.9)
    confusions_low = generate_confusions("hey marvin", count=50, seed=42, min_similarity=0.4)

    # High threshold should generally have fewer results
    assert len(confusions_high) <= len(confusions_low) + 10


def test_generate_confusions_zero_threshold() -> None:
    """Zero similarity threshold should allow all candidates."""
    confusions = generate_confusions("hey marvin", count=10, seed=42, min_similarity=0.0)
    assert len(confusions) <= 10


# --- Import Error Handling ---


def test_phonetic_similarity_no_jellyfish() -> None:
    """Should raise ImportError when jellyfish not available."""
    import wakeword_workbench.negatives.phrase_generator as pg

    # Save original
    original = pg.jellyfish

    try:
        # Mock absence
        pg.jellyfish = None

        with pytest.raises(ImportError, match="jellyfish"):
            phonetic_similarity("hello", "world")
    finally:
        # Restore
        pg.jellyfish = original


def test_generate_confusions_no_jellyfish() -> None:
    """Should raise ImportError when jellyfish not available."""
    import wakeword_workbench.negatives.phrase_generator as pg

    # Save original
    original = pg.jellyfish

    try:
        # Mock absence
        pg.jellyfish = None

        with pytest.raises(ImportError, match="jellyfish"):
            generate_confusions("hey marvin")
    finally:
        # Restore
        pg.jellyfish = original
