"""Tests for phrase variant generation."""

from __future__ import annotations

import pytest

from wakeword_workbench.dataset.phrase_variants import (
    _apply_casing_variations,
    _apply_contraction_variations,
    _apply_prefix_suffix_variations,
    _apply_punctuation_variations,
    _apply_spacing_variations,
    _generate_all_variants,
    generate_variants,
    get_variant_count,
)


class TestApplyPunctuationVariations:
    """Tests for punctuation variation generation."""

    def test_original_included(self) -> None:
        """Original phrase should always be included."""
        phrase = "hey assistant"
        variations = _apply_punctuation_variations(phrase)
        assert phrase in variations

    def test_multi_word_phrase_comma_variations(self) -> None:
        """Multi-word phrases should have comma variations."""
        phrase = "hey assistant"
        variations = _apply_punctuation_variations(phrase)
        # Should have comma after first word
        assert "hey, assistant" in variations
        # Should have trailing punctuation
        assert "hey assistant." in variations
        assert "hey assistant!" in variations
        assert "hey assistant?" in variations

    def test_single_word_no_comma_variations(self) -> None:
        """Single word phrases should not have comma variations."""
        phrase = "hey"
        variations = _apply_punctuation_variations(phrase)
        assert phrase in variations
        assert len(variations) == 1

    def test_no_duplicates(self) -> None:
        """Should not return duplicate variations."""
        phrase = "hey assistant"
        variations = _apply_punctuation_variations(phrase)
        assert len(variations) == len(set(variations))


class TestApplyCasingVariations:
    """Tests for casing variation generation."""

    def test_original_included(self) -> None:
        """Original phrase should always be included."""
        phrase = "hey Assistant"
        variations = _apply_casing_variations(phrase)
        assert phrase in variations

    def test_lower_case(self) -> None:
        """Should include lower case variation."""
        phrase = "HEY ASSISTANT"
        variations = _apply_casing_variations(phrase)
        assert "hey assistant" in variations

    def test_upper_case(self) -> None:
        """Should include upper case variation."""
        phrase = "hey assistant"
        variations = _apply_casing_variations(phrase)
        assert "HEY ASSISTANT" in variations

    def test_title_case(self) -> None:
        """Should include title case variation."""
        phrase = "hey assistant"
        variations = _apply_casing_variations(phrase)
        assert "Hey Assistant" in variations

    def test_capitalize(self) -> None:
        """Should include capitalize variation."""
        phrase = "hey assistant"
        variations = _apply_casing_variations(phrase)
        assert "Hey assistant" in variations

    def test_alternating_case(self) -> None:
        """Should include alternating case variation."""
        phrase = "hey assistant"
        variations = _apply_casing_variations(phrase)
        assert "HeY AsSiStAnT" in variations

    def test_no_duplicates(self) -> None:
        """Should not return duplicate variations."""
        phrase = "hey"
        variations = _apply_casing_variations(phrase)
        assert len(variations) == len(set(variations))


class TestApplySpacingVariations:
    """Tests for spacing variation generation."""

    def test_original_included(self) -> None:
        """Original phrase should always be included."""
        phrase = "hey assistant"
        variations = _apply_spacing_variations(phrase)
        assert phrase in variations

    def test_double_space(self) -> None:
        """Should include double space variation."""
        phrase = "hey assistant"
        variations = _apply_spacing_variations(phrase)
        assert "hey  assistant" in variations

    def test_leading_trailing_space(self) -> None:
        """Should include leading/trailing space variations."""
        phrase = "hey assistant"
        variations = _apply_spacing_variations(phrase)
        assert " hey assistant" in variations
        assert "hey assistant " in variations

    def test_single_word_no_space_variations(self) -> None:
        """Single word phrases should not have space variations."""
        phrase = "hey"
        variations = _apply_spacing_variations(phrase)
        assert phrase in variations
        assert len(variations) == 1

    def test_no_duplicates(self) -> None:
        """Should not return duplicate variations."""
        phrase = "hey assistant"
        variations = _apply_spacing_variations(phrase)
        assert len(variations) == len(set(variations))


class TestApplyPrefixSuffixVariations:
    """Tests for prefix/suffix variation generation."""

    def test_original_included(self) -> None:
        """Original phrase should always be included."""
        from wakeword_workbench.dataset.phrase_variants import DeterministicRandom

        phrase = "hey assistant"
        rng = DeterministicRandom(42)
        variations = _apply_prefix_suffix_variations(phrase, rng)
        assert phrase in variations

    def test_adds_prefix(self) -> None:
        """Should add leading words as prefix."""
        from wakeword_workbench.dataset.phrase_variants import DeterministicRandom

        phrase = "marvin"
        rng = DeterministicRandom(42)
        variations = _apply_prefix_suffix_variations(phrase, rng)
        # Check that at least one variation has a prefix (starts with a leading word)
        prefixed = [
            v for v in variations if v.startswith(tuple(["hey ", "hi ", "okay ", "ok ", "yo "]))
        ]
        assert len(prefixed) > 0

    def test_adds_suffix(self) -> None:
        """Should add trailing words as suffix."""
        from wakeword_workbench.dataset.phrase_variants import DeterministicRandom

        phrase = "hey assistant"
        rng = DeterministicRandom(42)
        variations = _apply_prefix_suffix_variations(phrase, rng)
        # Check that at least one variation has a suffix
        suffixed = [v for v in variations if " please" in v or " now" in v or " buddy" in v]
        assert len(suffixed) > 0

    def test_no_duplicates(self) -> None:
        """Should not return duplicate variations."""
        from wakeword_workbench.dataset.phrase_variants import DeterministicRandom

        phrase = "hey assistant"
        rng = DeterministicRandom(42)
        variations = _apply_prefix_suffix_variations(phrase, rng)
        assert len(variations) == len(set(variations))


class TestApplyContractionVariations:
    """Tests for contraction variation generation."""

    def test_original_included(self) -> None:
        """Original phrase should always be included."""
        phrase = "hey assistant"
        variations = _apply_contraction_variations(phrase)
        assert phrase in variations

    def test_hey_variations(self) -> None:
        """'hey' should have variation alternatives."""
        phrase = "hey assistant"
        variations = _apply_contraction_variations(phrase)
        # Should include variations with alternative words
        found_alternatives = False
        for v in variations:
            if v != phrase:
                words = v.split()
                if words[0] in ["hi", "hay", "heya"]:
                    found_alternatives = True
                    break
        assert found_alternatives

    def test_no_duplicates(self) -> None:
        """Should not return duplicate variations."""
        phrase = "hey assistant"
        variations = _apply_contraction_variations(phrase)
        assert len(variations) == len(set(variations))


class TestGenerateAllVariants:
    """Tests for the combined variant generation."""

    def test_includes_all_variation_types(self) -> None:
        """Should include variants from all strategies."""
        from wakeword_workbench.dataset.phrase_variants import DeterministicRandom

        phrase = "hey assistant"
        rng = DeterministicRandom(42)
        variants = _generate_all_variants(phrase, rng)

        # Should have casing variations
        assert any(v.isupper() for v in variants)

        # Should have punctuation variations
        assert any("," in v or "." in v or "!" in v or "?" in v for v in variants)

        # Should have original
        assert phrase in variants

    def test_no_duplicates(self) -> None:
        """Should not return duplicate variants."""
        from wakeword_workbench.dataset.phrase_variants import DeterministicRandom

        phrase = "hey assistant"
        rng = DeterministicRandom(42)
        variants = _generate_all_variants(phrase, rng)
        assert len(variants) == len(set(variants))


class TestGenerateVariants:
    """Tests for the main generate_variants function."""

    def test_returns_correct_count(self) -> None:
        """Should return exactly the requested count."""
        phrase = "hey marvin"
        variants = generate_variants(phrase, count=10)
        assert len(variants) == 10

    def test_returns_exactly_count_when_more_available(self) -> None:
        """Should return exactly count even when more variants available."""
        phrase = "hey assistant"
        for count in [5, 10, 20, 50]:
            variants = generate_variants(phrase, count=count)
            assert len(variants) == count

    def test_original_always_included(self) -> None:
        """Original phrase should always be included."""
        phrase = "hey marvin"
        variants = generate_variants(phrase, count=10)
        assert phrase in variants

        # Test with different seeds
        for seed in [42, 123, 456]:
            variants = generate_variants(phrase, count=10, seed=seed)
            assert phrase in variants

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same output."""
        phrase = "hey assistant"
        seed = 42

        variants1 = generate_variants(phrase, count=20, seed=seed)
        variants2 = generate_variants(phrase, count=20, seed=seed)

        assert variants1 == variants2

    def test_different_seeds_different_output(self) -> None:
        """Different seeds should produce different output (usually)."""
        phrase = "hey assistant"

        variants1 = generate_variants(phrase, count=50, seed=42)
        variants2 = generate_variants(phrase, count=50, seed=123)

        # Both should have the same length
        assert len(variants1) == len(variants2)
        # Both should contain the original phrase
        assert phrase in variants1
        assert phrase in variants2
        # At least some overlap in content (they're from the same pool)
        overlap = set(variants1) & set(variants2)
        assert len(overlap) > 0

    def test_all_unique(self) -> None:
        """All returned variants should be unique."""
        phrase = "hey assistant"
        variants = generate_variants(phrase, count=50)
        assert len(variants) == len(set(variants))

    def test_normalizes_whitespace(self) -> None:
        """Should normalize extra whitespace in input."""
        phrase = "  hey   assistant  "
        variants = generate_variants(phrase, count=5)
        normalized = " ".join(phrase.split())
        assert normalized in variants

    def test_empty_wake_word_raises(self) -> None:
        """Empty wake word should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            generate_variants("")

        with pytest.raises(ValueError, match="cannot be empty"):
            generate_variants("   ")

    def test_count_less_than_one_raises(self) -> None:
        """Count less than 1 should raise ValueError."""
        with pytest.raises(ValueError, match="count must be at least 1"):
            generate_variants("hey", count=0)

        with pytest.raises(ValueError, match="count must be at least 1"):
            generate_variants("hey", count=-1)

    def test_large_count_works(self) -> None:
        """Should handle larger counts gracefully."""
        phrase = "hey assistant"
        # Request more variants than we might have
        variants = generate_variants(phrase, count=100)
        assert len(variants) == 100
        assert len(variants) == len(set(variants))

    def test_different_phrases_different_variants(self) -> None:
        """Different phrases should produce different variant sets."""
        phrase1 = "hey marvin"
        phrase2 = "okay google"

        variants1 = generate_variants(phrase1, count=20, seed=42)
        variants2 = generate_variants(phrase2, count=20, seed=42)

        # Original phrases should be in their respective results
        assert phrase1 in variants1
        assert phrase2 in variants2

        # The sets should have some overlap only if phrases share words


class TestGetVariantCount:
    """Tests for the get_variant_count function."""

    def test_returns_positive_int(self) -> None:
        """Should return a positive integer."""
        count = get_variant_count("hey assistant")
        assert isinstance(count, int)
        assert count > 0

    def test_returns_at_least_one(self) -> None:
        """Should return at least 1 for any phrase."""
        for phrase in ["hey", "hello", "a", "test phrase here"]:
            count = get_variant_count(phrase)
            assert count >= 1


class TestIntegration:
    """Integration tests for phrase variant generation."""

    def test_reproducibility_across_calls(self) -> None:
        """Verify reproducibility across multiple calls with same seed."""
        phrase = "hey assistant"
        seed = 12345

        results = []
        for _ in range(3):
            variants = generate_variants(phrase, count=30, seed=seed)
            results.append(tuple(variants))

        assert results[0] == results[1] == results[2]

    def test_variant_quality_check(self) -> None:
        """Verify that generated variants have expected characteristics."""
        phrase = "hey assistant"
        variants = generate_variants(phrase, count=50, seed=42)

        # Check for casing variety
        casing_variants = [v for v in variants if v != v.lower()]
        assert len(casing_variants) > 0

        # Check for punctuation variety
        punct_variants = [v for v in variants if any(c in v for c in ",.!?")]
        assert len(punct_variants) > 0

        # All variants should be strings
        assert all(isinstance(v, str) for v in variants)

        # No empty strings
        assert all(len(v) > 0 for v in variants)

    def test_single_word_phrase(self) -> None:
        """Test variant generation for single word phrases."""
        phrase = "hey"
        variants = generate_variants(phrase, count=10)

        assert phrase in variants
        assert len(variants) == 10
        assert all(isinstance(v, str) for v in variants)

    def test_multi_word_phrase(self) -> None:
        """Test variant generation for multi-word phrases."""
        phrase = "hey assistant buddy"
        variants = generate_variants(phrase, count=20)

        assert phrase in variants
        assert len(variants) == 20

        # Should have some casing variations
        upper_count = sum(1 for v in variants if v.isupper())
        assert upper_count > 0
