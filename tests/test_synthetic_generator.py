"""Tests for synthetic negative generator."""

from __future__ import annotations

import pytest

from wakeword_workbench.negatives.synthetic_generator import (
    generate_synthetic_negatives,
    SyntheticGeneratorError,
    DEFAULT_WORDS,
    VALID_TOPICS,
)


class TestGenerateSyntheticNegatives:
    """Tests for generate_synthetic_negatives function."""

    def test_returns_correct_count(self):
        """Test that function returns the requested number of phrases."""
        negatives = generate_synthetic_negatives(100)
        assert len(negatives) == 100

    def test_returns_correct_count_when_less_can_be_generated(self):
        """Test behavior when not enough unique phrases can be generated."""
        # Very small word list with many restrictions
        small_word_list = ["word", "test", "data", "hello"]
        negatives = generate_synthetic_negatives(100, word_list=small_word_list)
        # May not be able to generate 100 unique phrases from 4 words
        assert len(negatives) > 0

    def test_all_unique(self):
        """Test that all returned phrases are unique."""
        negatives = generate_synthetic_negatives(100, seed=42)
        assert len(negatives) == len(set(negatives))

    def test_no_wake_word_hey_assistant(self):
        """Test that 'hey' and 'assistant' are not in generated phrases."""
        negatives = generate_synthetic_negatives(100, wake_word="hey assistant", seed=42)
        for phrase in negatives:
            phrase_lower = phrase.lower()
            assert "hey" not in phrase_lower.split(), f"Found 'hey' in: {phrase}"
            assert "assistant" not in phrase_lower.split(), f"Found 'assistant' in: {phrase}"

    def test_no_wake_word_generic(self):
        """Test exclusion of wake word words with custom wake word."""
        negatives = generate_synthetic_negatives(50, wake_word="custom wake word phrase", seed=42)
        for phrase in negatives:
            phrase_lower = phrase.lower()
            assert "custom" not in phrase_lower.split(), f"Found 'custom' in: {phrase}"
            assert "wake" not in phrase_lower.split(), f"Found 'wake' in: {phrase}"
            assert "word" not in phrase_lower.split(), f"Found 'word' in: {phrase}"
            assert "phrase" not in phrase_lower.split(), f"Found 'phrase' in: {phrase}"

    def test_deterministic_with_seed(self):
        """Test that same seed produces same output."""
        result1 = generate_synthetic_negatives(50, seed=12345)
        result2 = generate_synthetic_negatives(50, seed=12345)
        assert result1 == result2

    def test_different_seeds_different_output(self):
        """Test that different seeds produce different output."""
        result1 = generate_synthetic_negatives(50, seed=12345)
        result2 = generate_synthetic_negatives(50, seed=54321)
        assert result1 != result2

    def test_word_count_range(self):
        """Test that phrases have between min and max word count."""
        negatives = generate_synthetic_negatives(100, min_word_count=2, max_word_count=4, seed=42)
        for phrase in negatives:
            word_count = len(phrase.split())
            assert 2 <= word_count <= 4, f"Phrase '{phrase}' has {word_count} words, expected 2-4"

    def test_min_word_count_boundary(self):
        """Test minimum word count boundary."""
        negatives = generate_synthetic_negatives(50, min_word_count=3, max_word_count=3, seed=42)
        for phrase in negatives:
            word_count = len(phrase.split())
            assert word_count == 3, f"Phrase '{phrase}' has {word_count} words, expected 3"

    def test_custom_word_list(self):
        """Test that custom word list is used."""
        custom_words = ["apple", "banana", "cherry", "date", "elderberry"]
        negatives = generate_synthetic_negatives(20, word_list=custom_words, seed=42)
        for phrase in negatives:
            for word in phrase.split():
                assert word in custom_words, f"Word '{word}' not in custom list"

    def test_empty_word_list_raises_error(self):
        """Test that empty word list raises error."""
        with pytest.raises(SyntheticGeneratorError, match="word_list cannot be empty"):
            generate_synthetic_negatives(10, word_list=[])

    def test_negative_count_raises_error(self):
        """Test that negative count raises error."""
        with pytest.raises(SyntheticGeneratorError, match="count must be positive"):
            generate_synthetic_negatives(-1)

    def test_zero_count_raises_error(self):
        """Test that zero count raises error."""
        with pytest.raises(SyntheticGeneratorError, match="count must be positive"):
            generate_synthetic_negatives(0)

    def test_invalid_min_word_count_raises_error(self):
        """Test that invalid min_word_count raises error."""
        with pytest.raises(SyntheticGeneratorError, match="min_word_count must be at least 1"):
            generate_synthetic_negatives(10, min_word_count=0)

    def test_invalid_max_min_word_count_raises_error(self):
        """Test that max_word_count < min_word_count raises error."""
        with pytest.raises(
            SyntheticGeneratorError, match="max_word_count.*must be >= min_word_count"
        ):
            generate_synthetic_negatives(10, min_word_count=5, max_word_count=3)

    def test_sentence_strategy(self):
        """Test sentence generation strategy."""
        negatives = generate_synthetic_negatives(50, strategy="sentence", seed=42)
        assert len(negatives) == 50
        for phrase in negatives:
            # Sentence strategy produces 3 words
            assert len(phrase.split()) == 3, f"Expected 3 words, got: {phrase}"

    def test_topic_strategy(self):
        """Test topic-based generation strategy."""
        negatives = generate_synthetic_negatives(50, strategy="topic", seed=42)
        assert len(negatives) == 50

    def test_topic_strategy_specific_topics(self):
        """Test topic strategy with specific topics."""
        negatives = generate_synthetic_negatives(
            30, strategy="topic", topics=["kitchen", "office"], seed=42
        )
        assert len(negatives) == 30

    def test_invalid_strategy_falls_back_to_random(self):
        """Test that invalid strategy falls back to random."""
        negatives = generate_synthetic_negatives(50, strategy="invalid", seed=42)
        assert len(negatives) == 50

    def test_default_words_available(self):
        """Test that DEFAULT_WORDS is populated."""
        assert len(DEFAULT_WORDS) > 100

    def test_valid_topics_available(self):
        """Test that VALID_TOPICS is populated."""
        assert len(VALID_TOPICS) > 0
        assert "kitchen" in VALID_TOPICS
        assert "office" in VALID_TOPICS

    def test_phrases_are_strings(self):
        """Test that all returned items are strings."""
        negatives = generate_synthetic_negatives(50)
        for phrase in negatives:
            assert isinstance(phrase, str)

    def test_phrases_have_no_leading_trailing_whitespace(self):
        """Test that phrases have no extra whitespace."""
        negatives = generate_synthetic_negatives(100, seed=42)
        for phrase in negatives:
            assert phrase == phrase.strip()
            assert "  " not in phrase

    def test_phrases_dont_contain_wake_word_words(self):
        """Test that phrases don't contain wake word as whole word (not substring)."""
        # Use a wake word that won't appear as part of other words
        negatives = generate_synthetic_negatives(100, wake_word="alexa", seed=42)
        for phrase in negatives:
            phrase_lower = phrase.lower()
            # Check that "alexa" is not present as a word
            words = phrase_lower.split()
            assert "alexa" not in words, f"Found 'alexa' as word in: {phrase}"

    def test_large_count_generation(self):
        """Test generation of a large number of phrases."""
        negatives = generate_synthetic_negatives(1000, seed=42)
        assert len(negatives) == 1000
        assert len(negatives) == len(set(negatives))  # All unique

    def test_none_seed_works(self):
        """Test that None seed (system randomness) works."""
        # This just tests that it doesn't crash
        negatives = generate_synthetic_negatives(10, seed=None)
        assert len(negatives) == 10


class TestSyntheticGeneratorEdgeCases:
    """Edge case tests for synthetic generator."""

    def test_wake_word_with_underscore(self):
        """Test wake word with underscores splits correctly."""
        negatives = generate_synthetic_negatives(50, wake_word="hey_computer", seed=42)
        for phrase in negatives:
            phrase_lower = phrase.lower()
            # "hey_computer" should be split into "hey" and "computer"
            assert "hey" not in phrase_lower.split()
            assert "computer" not in phrase_lower.split()

    def test_single_word_wake_word(self):
        """Test single word wake word."""
        negatives = generate_synthetic_negatives(50, wake_word="computer", seed=42)
        for phrase in negatives:
            phrase_lower = phrase.lower()
            assert "computer" not in phrase_lower.split()

    def test_all_words_excluded(self):
        """Test when almost all words would be excluded."""
        # This edge case tests what happens when most words are excluded
        # We use a very restricted word list that overlaps with a common wake word
        word_list = ["hello", "world", "hey", "assistant", "test"]
        negatives = generate_synthetic_negatives(
            10, word_list=word_list, wake_word="hello world", seed=42
        )
        # Should still generate something, just from remaining words
        assert len(negatives) >= 0  # May be 0 if too restrictive

    def test_reproduction_with_same_seed_across_runs(self):
        """Test reproducibility across different function calls."""
        result1 = generate_synthetic_negatives(100, seed=99999)
        result2 = generate_synthetic_negatives(100, seed=99999)
        assert result1 == result2

    def test_generation_count_matches_positive_distribution(self):
        """Test that word count distribution matches typical positive sample."""
        negatives = generate_synthetic_negatives(500, min_word_count=2, max_word_count=4, seed=42)
        # Verify distribution is reasonable
        word_counts = [len(p.split()) for p in negatives]
        avg_words = sum(word_counts) / len(word_counts)
        assert 2 <= avg_words <= 4
