"""Negative phrase generator for phonetic confusions."""

from __future__ import annotations

import random
import string
from collections.abc import Sequence
from typing import Protocol

try:
    import jellyfish
except ImportError:
    jellyfish = None  # type: ignore[assignment, unused-ignore]

# --- Phonetic Dictionaries for Wake Word Confusion Generation ---

# Common wake word prefixes and their phonetic alternatives
WAKE_WORD_PREFIXES: dict[str, list[str]] = {
    "hey": ["hay", "say", "pay", "way", "day", "ray", "may", "he"],
    "ok": ["ok", "okay", "kay", "k", "ohk", "ohcay"],
    "okay": ["ok", "kay", "k", "ohk"],
    "hi": ["hi", "hai", "hay", "high"],
    "yo": ["yo", "yoh"],
}

# Phonetic homophones for common words
HOMOPHONES: dict[str, list[str]] = {
    "hey": ["hay"],
    "hi": ["hai"],
    "to": ["too", "two"],
    "for": ["fore", "four"],
    "marvin": ["marven"],
    "assistant": ["assistant"],
    "alexa": [],
    "siri": [],
    "google": [],
    "cortana": [],
}

# Vowel substitutions that maintain phonetic similarity
VOWEL_SUBSTITUTIONS: dict[str, list[str]] = {
    "a": ["e", "i", "o", "u"],
    "e": ["a", "i", "o", "u"],
    "i": ["a", "e", "o", "u"],
    "o": ["a", "e", "i", "u"],
    "u": ["a", "e", "i", "o"],
}

# Consonant substitutions that maintain phonetic similarity
CONSONANT_SUBSTITUTIONS: dict[str, list[str]] = {
    "b": ["p", "d", "g"],
    "p": ["b", "t", "k"],
    "d": ["t", "b", "g"],
    "t": ["d", "p", "k"],
    "g": ["k", "j", "d"],
    "k": ["g", "c", "t"],
    "v": ["f", "b"],
    "f": ["v", "ph"],
    "s": ["z", "c", "ss"],
    "z": ["s"],
    "m": ["n"],
    "n": ["m"],
    "l": ["r"],
    "r": ["l"],
}

# Word endings for rhyming patterns
RHYMING_ENDINGS: list[str] = [
    "in",
    "en",
    "an",
    "on",
    "un",
    "ing",
    "eng",
    "ang",
    "ong",
    "ung",
    "ine",
    "ene",
    "ane",
    "one",
    "une",
    "er",
    "ar",
    "or",
    "ur",
]

# Consonant cluster variations
CLUSTER_VARIATIONS: dict[str, list[str]] = {
    "rv": ["rb", "rp"],
    "rb": ["rv", "rp"],
    "mp": ["mb", "np"],
    "nd": ["nt", "ng"],
    "st": ["sk", "sp"],
    "sk": ["st", "sp"],
    "sp": ["st", "sk"],
    "th": ["t", "z"],
    "sh": ["ch", "s"],
    "ch": ["sh", "t"],
}


class RandomProvider(Protocol):
    """Protocol for random number generator."""

    def random(self) -> float:
        """Return random float in [0.0, 1.0)."""
        ...

    def choice(self, seq: Sequence[str]) -> str:
        """Return random element from sequence."""
        ...

    def sample(self, seq: Sequence[str], k: int) -> list[str]:
        """Return random sample of k elements from sequence."""
        ...

    def randint(self, a: int, b: int) -> int:
        """Return random integer in range [a, b]."""
        ...

    def shuffle(self, seq: list[str]) -> None:
        """Shuffle sequence in place."""
        ...


class DeterministicRandom:
    """Deterministic random provider using a seeded random instance."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def choice(self, seq: Sequence[str]) -> str:
        return self._rng.choice(list(seq))

    def sample(self, seq: Sequence[str], k: int) -> list[str]:
        return self._rng.sample(list(seq), k)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def shuffle(self, seq: list[str]) -> None:
        self._rng.shuffle(seq)


def phonetic_similarity(phrase1: str, phrase2: str) -> float:
    """Calculate phonetic similarity between two phrases using metaphone.

    Args:
        phrase1: First phrase
        phrase2: Second phrase

    Returns:
        Similarity score between 0.0 (no similarity) and 1.0 (identical)
    """
    if jellyfish is None:
        raise ImportError(
            "jellyfish is required for phonetic similarity. Install with: pip install jellyfish"
        )

    # Normalize phrases
    words1 = phrase1.lower().split()
    words2 = phrase2.lower().split()

    # Handle empty or different-length phrases
    if not words1 or not words2:
        return 0.0

    # Calculate metaphone codes
    try:
        codes1 = [jellyfish.metaphone(w) for w in words1]
        codes2 = [jellyfish.metaphone(w) for w in words2]
    except Exception:
        return 0.0

    # Handle empty codes
    codes1 = [c for c in codes1 if c]
    codes2 = [c for c in codes2 if c]

    if not codes1 or not codes2:
        return 0.0

    # Exact match
    if codes1 == codes2:
        return 1.0

    # Calculate similarity using Levenshtein distance on metaphone codes
    # Compare as sequences using Jaro-Winkler for better phoneme matching
    if len(codes1) == 1 and len(codes2) == 1:
        # Single word comparison
        try:
            # Normalize by max length
            max_len = max(len(codes1[0]), len(codes2[0]))
            if max_len == 0:
                return 0.0
            # Use Jaro-Winkler for phoneme comparison
            sim = jellyfish.jaro_winkler_similarity(codes1[0], codes2[0])
            return round(sim, 2)
        except Exception:
            return 0.0
    else:
        # Multi-word phrase comparison
        # Use sequence alignment approach
        total_sim = 0.0
        pairs = min(len(codes1), len(codes2))

        for i in range(pairs):
            try:
                sim = jellyfish.jaro_winkler_similarity(codes1[i], codes2[i])
                total_sim += sim
            except Exception:
                continue

        # Average similarity weighted by matched pairs
        if codes1 == codes2:
            return 1.0

        # Penalize length differences
        len_penalty = 1.0 - abs(len(codes1) - len(codes2)) * 0.1
        len_penalty = max(len_penalty, 0.5)

        if pairs > 0:
            avg_sim = total_sim / max(len(codes1), len(codes2))
            return round(avg_sim * len_penalty, 2)

        return 0.0


def _apply_homophone_substitution(phrase: str, rng: RandomProvider) -> list[str]:
    """Generate confusions by replacing words with homophones.

    Args:
        phrase: Original phrase
        rng: Random provider

    Returns:
        List of homophone variations
    """
    results: list[str] = []
    words = phrase.split()

    for i, word in enumerate(words):
        word_lower = word.lower().strip(string.punctuation)
        if word_lower in HOMOPHONES and HOMOPHONES[word_lower]:
            for homophone in HOMOPHONES[word_lower][:3]:
                new_words = words.copy()
                # Preserve punctuation
                trailing_punct = word[len(word_lower) :] if len(word) > len(word_lower) else ""
                new_words[i] = homophone + trailing_punct
                results.append(" ".join(new_words))

    return results


def _apply_vowel_substitution(phrase: str, rng: RandomProvider) -> list[str]:
    """Generate confusions by substituting vowels in words.

    Args:
        phrase: Original phrase
        rng: Random provider

    Returns:
        List of vowel-substituted variations
    """
    results: list[str] = []
    words = phrase.split()

    for i, word in enumerate(words):
        word_clean = word.lower().strip(string.punctuation)
        # Skip short words and words without vowels
        if len(word_clean) < 2:
            continue

        trailing_punct = word[len(word_clean) :] if len(word) > len(word_clean) else ""

        # Find positions of vowels
        vowels = "aeiou"
        vowel_positions = [j for j, c in enumerate(word_clean) if c in vowels]

        if not vowel_positions:
            continue

        # Try substituting 1-2 vowels
        for _ in range(min(3, len(vowel_positions))):
            pos = vowel_positions[rng.randint(0, len(vowel_positions) - 1)]
            original_vowel = word_clean[pos]
            if original_vowel in VOWEL_SUBSTITUTIONS:
                new_vowel = rng.choice(VOWEL_SUBSTITUTIONS[original_vowel])
                if new_vowel != original_vowel:
                    new_word_chars = list(word_clean)
                    new_word_chars[pos] = new_vowel
                    new_words = words.copy()
                    new_words[i] = "".join(new_word_chars) + trailing_punct
                    results.append(" ".join(new_words))

    return results


def _apply_consonant_substitution(phrase: str, rng: RandomProvider) -> list[str]:
    """Generate confusions by substituting consonants in words.

    Args:
        phrase: Original phrase
        rng: Random provider

    Returns:
        List of consonant-substituted variations
    """
    results: list[str] = []
    words = phrase.split()

    for i, word in enumerate(words):
        word_clean = word.lower().strip(string.punctuation)
        # Skip short words
        if len(word_clean) < 2:
            continue

        trailing_punct = word[len(word_clean) :] if len(word) > len(word_clean) else ""

        # Find positions of consonants that can be substituted
        sub_positions = [j for j, c in enumerate(word_clean) if c in CONSONANT_SUBSTITUTIONS]

        if not sub_positions:
            continue

        for _ in range(min(2, len(sub_positions))):
            pos = sub_positions[rng.randint(0, len(sub_positions) - 1)]
            original_consonant = word_clean[pos]
            if original_consonant in CONSONANT_SUBSTITUTIONS:
                alternatives = [
                    c
                    for c in CONSONANT_SUBSTITUTIONS[original_consonant]
                    if c != original_consonant
                ]
                if alternatives:
                    new_consonant = rng.choice(alternatives)
                    new_word_chars = list(word_clean)
                    new_word_chars[pos] = new_consonant
                    new_words = words.copy()
                    new_words[i] = "".join(new_word_chars) + trailing_punct
                    results.append(" ".join(new_words))

    return results


def _apply_rhyming_substitution(phrase: str, rng: RandomProvider) -> list[str]:
    """Generate confusions by replacing word endings with rhyming alternatives.

    Args:
        phrase: Original phrase
        rng: Random provider

    Returns:
        List of rhyming variations
    """
    results: list[str] = []
    words = phrase.split()

    for i, word in enumerate(words):
        word_clean = word.lower().strip(string.punctuation)
        # Skip short words
        if len(word_clean) < 3:
            continue

        trailing_punct = word[len(word_clean) :] if len(word) > len(word_clean) else ""

        # Find last consonant cluster and replace ending
        for ending in rng.sample(RHYMING_ENDINGS, min(3, len(RHYMING_ENDINGS))):
            # Only replace if the word ends with a similar pattern
            if len(word_clean) > len(ending) + 1:
                # Replace the ending
                new_word = word_clean[: -(len(ending))] + ending
                if new_word != word_clean:
                    new_words = words.copy()
                    new_words[i] = new_word + trailing_punct
                    results.append(" ".join(new_words))

    return results


def _apply_prefix_substitution(phrase: str, rng: RandomProvider) -> list[str]:
    """Generate confusions by replacing wake word prefixes.

    Args:
        phrase: Original phrase
        rng: Random provider

    Returns:
        List of prefix variations
    """
    results: list[str] = []
    words = phrase.split()

    if not words:
        return results

    first_word = words[0].lower().strip(string.punctuation)

    # Check for wake word prefix matches
    for prefix_key, alternatives in WAKE_WORD_PREFIXES.items():
        if first_word.startswith(prefix_key) or first_word == prefix_key:
            trailing_punct = words[0][len(first_word) :]
            for alt in alternatives[:4]:
                if alt != first_word:
                    new_words = words.copy()
                    new_words[0] = alt + trailing_punct
                    results.append(" ".join(new_words))

    return results


def _apply_cluster_substitution(phrase: str, rng: RandomProvider) -> list[str]:
    """Generate confusions by substituting consonant clusters.

    Args:
        phrase: Original phrase
        rng: Random provider

    Returns:
        List of cluster-substituted variations
    """
    results: list[str] = []
    words = phrase.split()

    for i, word in enumerate(words):
        word_clean = word.lower().strip(string.punctuation)

        # Look for consonant clusters
        for cluster, alternatives in CLUSTER_VARIATIONS.items():
            if cluster in word_clean and alternatives:
                for alt in alternatives[:2]:
                    new_word = word_clean.replace(cluster, alt, 1)
                    if new_word != word_clean:
                        trailing_punct = (
                            word[len(word_clean) :] if len(word) > len(word_clean) else ""
                        )
                        new_words = words.copy()
                        new_words[i] = new_word + trailing_punct
                        results.append(" ".join(new_words))

    return results


def _generate_confusion_candidates(
    wake_word: str,
    rng: RandomProvider,
) -> list[str]:
    """Generate all confusion candidates using various strategies.

    Args:
        wake_word: The wake word phrase
        rng: Random provider

    Returns:
        List of generated confusion phrases
    """
    candidates: set[str] = set()

    strategies = [
        _apply_homophone_substitution,
        _apply_vowel_substitution,
        _apply_consonant_substitution,
        _apply_rhyming_substitution,
        _apply_prefix_substitution,
        _apply_cluster_substitution,
    ]

    for strategy in strategies:
        try:
            results = strategy(wake_word, rng)
            for result in results:
                if result != wake_word.lower() and result.strip():
                    candidates.add(result.lower())
        except Exception:
            # Skip failed strategies
            continue

    return list(candidates)


def generate_confusions(
    wake_word: str,
    count: int = 50,
    seed: int | None = None,
    min_similarity: float = 0.6,
) -> list[str]:
    """Generate phonetically similar phrases to confuse wake word detection.

    Uses multiple strategies to generate confusions:
    - Homophone substitution
    - Vowel substitution
    - Consonant substitution
    - Rhyming word endings
    - Prefix replacement
    - Consonant cluster variation

    Args:
        wake_word: The wake word phrase to generate confusions for
        count: Number of unique confusions to return
        seed: Random seed for deterministic output (None for random)
        min_similarity: Minimum phonetic similarity score (0.0-1.0)

    Returns:
        List of unique confusion phrases with phonetic similarity > min_similarity

    Raises:
        ValueError: If count < 1 or wake_word is empty

    Example:
        >>> confusions = generate_confusions("hey marvin", count=10)
        >>> len(confusions) <= 10
        True
        >>> "hey marvin" not in confusions
        True
        >>> all(phonetic_similarity("hey marvin", c) >= 0.6 for c in confusions)
        True
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    if not wake_word or not wake_word.strip():
        raise ValueError("wake_word cannot be empty")

    if jellyfish is None:
        raise ImportError("jellyfish is required. Install with: pip install jellyfish")

    # Normalize the wake word
    normalized = " ".join(wake_word.lower().split())

    # Create deterministic random provider
    rng = DeterministicRandom(seed)

    # Generate candidates
    candidates = _generate_confusion_candidates(normalized, rng)

    # Calculate similarity scores and filter
    scored_confusions: list[tuple[str, float]] = []
    for candidate in candidates:
        if candidate.strip() == normalized.strip():
            continue  # Skip exact match

        similarity = phonetic_similarity(normalized, candidate)
        if similarity >= min_similarity:
            scored_confusions.append((candidate, similarity))

    # Sort by similarity (descending) and then shuffle within similarity tiers
    scored_confusions.sort(key=lambda x: -x[1])

    # Take the best candidates up to count
    result: list[str] = []
    seen: set[str] = set()

    for phrase, _score in scored_confusions:
        if phrase not in seen:
            result.append(phrase)
            seen.add(phrase)
            if len(result) >= count:
                break

    # If we need more candidates, try with additional random variations
    if len(result) < count:
        # Try generating more with additional random variations
        base_seed = seed if seed is not None else 42
        for attempt in range(10):
            extra_rng = DeterministicRandom(base_seed + attempt * 1000)
            extra_candidates = _generate_confusion_candidates(normalized, extra_rng)

            for candidate in extra_candidates:
                if candidate.strip() == normalized.strip():
                    continue
                if candidate in seen:
                    continue

                similarity = phonetic_similarity(normalized, candidate)
                if similarity >= min_similarity:  # Use same threshold
                    result.append(candidate)
                    seen.add(candidate)
                    if len(result) >= count:
                        break

            if len(result) >= count:
                break

    # Final shuffle to avoid ordering bias
    final_rng = DeterministicRandom(seed)
    final_rng.shuffle(result)

    return result[:count]


def get_confusion_count(phrase: str) -> int:
    """Estimate the total number of possible confusion phrases for a phrase.

    Args:
        phrase: The phrase to analyze

    Returns:
        Estimated number of unique confusions possible
    """
    normalized = " ".join(phrase.lower().split())
    rng = DeterministicRandom(None)
    candidates = _generate_confusion_candidates(normalized, rng)

    # Filter by minimum similarity
    valid = [
        c
        for c in candidates
        if c.strip() != normalized.strip() and phonetic_similarity(normalized, c) >= 0.6
    ]

    return len(valid)
