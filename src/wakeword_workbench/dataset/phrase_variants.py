"""Phrase variant generation utilities."""

from __future__ import annotations

import random
import string
from typing import Protocol

# Leading/trailing words that can be added as prefixes or suffixes
LEADING_WORDS = ["hey", "okay", "ok", "yo", "hi", "hey"]
TRAILING_WORDS = ["please", "now", "thanks", "buddy", "friend"]

# Contractions/variations for common wake words
CONTRACTIONS: dict[str, list[str]] = {
    "hey": ["hey", "hi", "hay", "heya"],
    "okay": ["okay", "ok", "okee", "k"],
    "ok": ["ok", "okay", "okee", "k"],
    "alexa": ["alexa"],
    "siri": ["siri"],
    "cortana": ["cortana"],
}


class RandomProvider(Protocol):
    """Protocol for random number generator."""

    def random(self) -> float:
        """Return random float in [0.0, 1.0)."""
        ...

    def choice(self, seq: list[str]) -> str:
        """Return random element from sequence."""
        ...


class DeterministicRandom:
    """Deterministic random provider using a seeded random instance."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def choice(self, seq: list[str]) -> str:
        return self._rng.choice(seq)

    def sample(self, seq: list[str], k: int) -> list[str]:
        return self._rng.sample(seq, k)

    def shuffle(self, seq: list[str]) -> None:
        self._rng.shuffle(seq)


def _apply_punctuation_variations(phrase: str) -> list[str]:
    """Generate punctuation variations of a phrase.

    Args:
        phrase: Original phrase (e.g., "hey assistant")

    Returns:
        List of variations with different punctuation
    """
    variations: list[str] = []

    # Original (no punctuation change)
    variations.append(phrase)

    # Add comma after first word if phrase has multiple words
    words = phrase.split()
    if len(words) >= 2:
        # Comma after first word
        variations.append(f"{words[0]}, {' '.join(words[1:])}")
        # Comma after second word
        if len(words) >= 3:
            variations.append(f"{words[0]} {words[1]}, {' '.join(words[2:])}")
        # Trailing comma
        variations.append(f"{phrase},")
        # Period at end
        variations.append(f"{phrase}.")
        # Exclamation
        variations.append(f"{phrase}!")
        # Question
        variations.append(f"{phrase}?")

    return list(dict.fromkeys(variations))  # Remove duplicates while preserving order


def _apply_casing_variations(phrase: str) -> list[str]:
    """Generate casing variations of a phrase.

    Args:
        phrase: Original phrase (e.g., "hey assistant")

    Returns:
        List of variations with different casing
    """
    variations: list[str] = []

    # Original
    variations.append(phrase)

    # Lower case
    variations.append(phrase.lower())

    # Upper case
    variations.append(phrase.upper())

    # Title case
    variations.append(phrase.title())

    # First letter capital
    variations.append(phrase.capitalize())

    # Alternating case (HeY AsSiStAnT)
    alt = []
    for i, char in enumerate(phrase):
        if char.isalpha():
            alt.append(char.upper() if i % 2 == 0 else char.lower())
        else:
            alt.append(char)
    variations.append("".join(alt))

    return list(dict.fromkeys(variations))


def _apply_spacing_variations(phrase: str) -> list[str]:
    """Generate spacing variations of a phrase.

    Args:
        phrase: Original phrase (e.g., "hey assistant")

    Returns:
        List of variations with different spacing
    """
    variations: list[str] = []

    # Original
    variations.append(phrase)

    # Normalize whitespace first
    normalized = " ".join(phrase.split())

    # Single space (original normalized)
    variations.append(normalized)

    # Double space between words
    if " " in phrase:
        words = phrase.split()
        variations.append("  ".join(words))
        variations.append(f" {normalized} ")  # Leading and trailing space
        variations.append(f" {normalized}")  # Leading space
        variations.append(f"{normalized} ")  # Trailing space

    # Tab character (represented as double space)
    words_list = phrase.split()
    tab_repr = "\t".join(words_list) if " " in phrase else phrase
    variations.append(tab_repr)

    return list(dict.fromkeys(variations))


def _apply_prefix_suffix_variations(phrase: str, rng: RandomProvider) -> list[str]:
    """Generate prefix/suffix variations by adding leading/trailing words.

    Args:
        phrase: Original phrase
        rng: Random provider for deterministic selection

    Returns:
        List of variations with prefixes/suffixes
    """
    variations: list[str] = []

    # Original
    variations.append(phrase)

    words = phrase.split()
    first_word = words[0] if words else ""

    # Add leading words (avoiding duplicates with existing first word)
    available_leading = [w for w in LEADING_WORDS if w.lower() != first_word.lower()]
    for prefix in available_leading:
        variations.append(f"{prefix} {phrase}")

    # Add trailing words
    for suffix in TRAILING_WORDS:
        variations.append(f"{phrase} {suffix}")

    # Add prefix + suffix combinations
    for prefix in available_leading[:3]:
        for suffix in TRAILING_WORDS[:3]:
            variations.append(f"{prefix} {phrase} {suffix}")

    return list(dict.fromkeys(variations))


def _apply_contraction_variations(phrase: str) -> list[str]:
    """Generate contraction variations by replacing words with alternatives.

    Args:
        phrase: Original phrase

    Returns:
        List of variations with word substitutions
    """
    variations: list[str] = []

    words = phrase.split()
    variations.append(phrase)

    for i, word in enumerate(words):
        word_lower = word.lower()
        word_clean = word_lower.rstrip(string.punctuation)
        trailing_punct = word[len(word_clean) :]

        if word_clean in CONTRACTIONS:
            alternatives = CONTRACTIONS[word_clean]
            for alt in alternatives[:3]:  # Limit to first 3 alternatives
                if alt != word_clean:
                    new_words = words.copy()
                    new_words[i] = alt + trailing_punct
                    variations.append(" ".join(new_words))

    return list(dict.fromkeys(variations))


def _generate_all_variants(phrase: str, rng: RandomProvider) -> list[str]:
    """Generate all possible variants from a phrase.

    Args:
        phrase: Original phrase
        rng: Random provider for deterministic selection

    Returns:
        List of all unique variants
    """
    all_variants: set[str] = set()

    # Apply all variation strategies
    for v in _apply_punctuation_variations(phrase):
        all_variants.add(v)
    for v in _apply_casing_variations(phrase):
        all_variants.add(v)
    for v in _apply_spacing_variations(phrase):
        all_variants.add(v)
    for v in _apply_prefix_suffix_variations(phrase, rng):
        all_variants.add(v)
    for v in _apply_contraction_variations(phrase):
        all_variants.add(v)

    # Apply combination variations (casing + punctuation, spacing + casing, etc.)
    punct_vars = _apply_punctuation_variations(phrase)
    casing_vars = _apply_casing_variations(phrase)
    spacing_vars = _apply_spacing_variations(phrase)
    prefix_vars = _apply_prefix_suffix_variations(phrase, rng)
    contract_vars = _apply_contraction_variations(phrase)

    # Casing + punctuation combinations
    for pv in punct_vars[1:]:  # Skip original (already included)
        for _ in casing_vars[1:]:  # Skip original
            all_variants.add(pv.upper())
            all_variants.add(pv.lower())
            all_variants.add(pv.title())

    # Spacing + casing combinations
    for sv in spacing_vars[1:]:
        all_variants.add(sv.upper())
        all_variants.add(sv.lower())

    # Prefix/suffix + casing combinations
    for pv in prefix_vars:
        all_variants.add(pv.upper())
        all_variants.add(pv.lower())
        all_variants.add(pv.title())

    # Contractions + casing combinations
    for cv in contract_vars[1:]:
        all_variants.add(cv.upper())
        all_variants.add(cv.lower())

    # Contractions + spacing combinations
    for cv in contract_vars:
        all_variants.add(cv + " ")  # trailing space
        all_variants.add(" " + cv)  # leading space

    # Contractions + punctuation combinations
    for cv in contract_vars:
        all_variants.add(cv + ".")
        all_variants.add(cv + "!")
        all_variants.add(cv + "?")

    # Prefix/suffix + punctuation combinations
    for pv in prefix_vars:
        all_variants.add(pv + ".")
        all_variants.add(pv + "!")
        all_variants.add(pv + "?")
        all_variants.add(pv + ",")

    # Triple spacing variations
    words = phrase.split()
    if len(words) >= 2:
        all_variants.add("   ".join(words))
        all_variants.add("   ".join(words).upper())

    # Always include original
    all_variants.add(phrase)

    return list(all_variants)


def generate_variants(
    wake_word: str,
    count: int = 50,
    seed: int | None = None,
) -> list[str]:
    """Generate natural variations of a wake word phrase.

    Args:
        wake_word: The wake word phrase to generate variants for
        count: Number of unique variants to return
        seed: Random seed for deterministic output (None for random)

    Returns:
        List of unique variant strings, always including the original phrase

    Example:
        >>> variants = generate_variants("hey marvin", count=10)
        >>> len(variants) == 10
        True
        >>> "hey marvin" in variants
        True
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    if not wake_word or not wake_word.strip():
        raise ValueError("wake_word cannot be empty")

    # Normalize the phrase
    normalized = " ".join(wake_word.split())

    # Create deterministic random provider
    rng = DeterministicRandom(seed)

    # Generate all possible variants
    all_variants_set = set(_generate_all_variants(normalized, rng))

    # If we need more variants than available, regenerate with different seeds
    base_seed = seed if seed is not None else 42
    attempt = 0

    while len(all_variants_set) < count:
        attempt += 1
        # Use different seed for each attempt to get different random selections
        new_rng = DeterministicRandom(base_seed + attempt * 1000)
        new_variants = _generate_all_variants(normalized, new_rng)
        all_variants_set.update(new_variants)

        # Safety limit to prevent infinite loops
        if attempt > 100:
            break

    # Shuffle using the original random provider
    all_variants_list = list(all_variants_set)
    rng.shuffle(all_variants_list)

    # Take the requested count, ensuring the original and representative
    # variation categories are included when available.
    result: list[str] = []

    preferred_variants: list[str] = []
    for candidate in [
        normalized,
        normalized.upper(),
        normalized.title(),
        f"{normalized}!",
    ]:
        if candidate in all_variants_set and candidate not in preferred_variants:
            preferred_variants.append(candidate)

    for candidate in preferred_variants[:count]:
        result.append(candidate)
        all_variants_set.discard(candidate)

    # Add remaining variants
    remaining = list(all_variants_set)
    if len(remaining) >= count - len(result):
        result.extend(rng.sample(remaining, count - len(result)))
    else:
        result.extend(remaining)

    return result


def get_variant_count(phrase: str) -> int:
    """Get the total number of possible variants for a phrase.

    This is useful for determining how many variants can be generated
    before cycling.

    Args:
        phrase: The phrase to analyze

    Returns:
        Estimated number of unique variants possible
    """
    normalized = " ".join(phrase.split())
    rng = DeterministicRandom(None)
    variants = _generate_all_variants(normalized, rng)
    return len(variants)
