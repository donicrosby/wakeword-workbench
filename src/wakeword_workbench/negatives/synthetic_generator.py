"""Synthetic negative sample generator.

Generates random word concatenations that don't contain the wake word,
matching the length distribution of positive samples.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from wakeword_workbench.logging_config import get_logger

if TYPE_CHECKING:
    from typing import Callable


log = get_logger(__name__)


# Default word lists for generating synthetic negatives
DEFAULT_ARTICLES = ["the", "a", "an", "some", "any"]
DEFAULT_PRONOUNS = [
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "this",
    "that",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
]
DEFAULT_PREPOSITIONS = [
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
]
DEFAULT_VERBS = [
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "go",
    "come",
    "see",
    "make",
    "take",
    "get",
    "know",
    "think",
    "say",
    "tell",
    "want",
    "use",
    "find",
    "give",
    "try",
    "call",
    "keep",
    "let",
    "begin",
    "seem",
    "help",
    "show",
    "hear",
    "play",
    "run",
    "move",
    "live",
    "believe",
    "hold",
    "bring",
    "happen",
    "write",
    "provide",
    "sit",
    "stand",
    "lose",
    "pay",
    "meet",
    "include",
    "continue",
    "set",
    "learn",
    "change",
    "lead",
    "understand",
    "watch",
    "follow",
    "stop",
    "create",
    "speak",
    "read",
    "spend",
    "grow",
    "open",
    "walk",
    "win",
    "offer",
    "remember",
    "love",
    "consider",
    "appear",
    "buy",
    "wait",
    "serve",
    "die",
    "send",
    "expect",
    "build",
    "stay",
    "fall",
    "cut",
    "reach",
    "kill",
    "remain",
]
DEFAULT_NOUNS = [
    "time",
    "year",
    "people",
    "way",
    "day",
    "man",
    "thing",
    "woman",
    "life",
    "child",
    "world",
    "school",
    "state",
    "family",
    "student",
    "group",
    "country",
    "problem",
    "hand",
    "part",
    "place",
    "case",
    "week",
    "company",
    "system",
    "program",
    "question",
    "work",
    "government",
    "number",
    "night",
    "point",
    "home",
    "water",
    "room",
    "mother",
    "area",
    "money",
    "story",
    "fact",
    "month",
    "lot",
    "right",
    "study",
    "book",
    "eye",
    "job",
    "word",
    "business",
    "issue",
    "side",
    "kind",
    "head",
    "house",
    "service",
    "friend",
    "father",
    "power",
    "hour",
    "game",
    "line",
    "end",
    "member",
    "law",
    "car",
    "city",
    "name",
    "president",
    "team",
    "minute",
    "idea",
    "kid",
    "body",
    "back",
    "parent",
    "face",
    "others",
    "level",
    "office",
    "door",
    "health",
    "person",
    "art",
    "war",
    "history",
    "party",
    "result",
    "change",
    "morning",
    "reason",
    "research",
    "girl",
    "guy",
    "moment",
    "air",
    "teacher",
    "force",
    "education",
    "foot",
    "boy",
    "age",
    "policy",
    "process",
    "music",
    "market",
    "sense",
    "nation",
    "plan",
    "college",
    "interest",
    "death",
    "experience",
    "effect",
    "use",
    "class",
    "control",
    "care",
    "field",
    "development",
    "role",
    "effort",
    "rate",
    "heart",
    "drug",
    "show",
    "leader",
    "light",
    "voice",
    "wife",
    "police",
    "mind",
    "difference",
    "period",
    "building",
    "action",
    "industry",
    "food",
    "theory",
    "away",
    "never",
    "today",
    "turn",
    "everything",
    "against",
    "between",
    "while",
    "another",
    "around",
    "however",
    "through",
    "next",
    "sound",
    "computer",
    "hope",
    "able",
    "model",
    "traditional",
    "project",
    "single",
    "paper",
    "record",
    "piece",
    "often",
    "done",
    "well",
    "big",
    "small",
    "large",
    "little",
    "long",
    "short",
    "high",
    "low",
    "new",
    "old",
    "first",
    "last",
    "good",
    "bad",
    "great",
    "hard",
    "fast",
    "slow",
    "easy",
    "soft",
    "hot",
    "cold",
    "warm",
    "cool",
    "full",
    "empty",
    "heavy",
    "light",
    "dark",
    "bright",
    "quiet",
    "loud",
    "strong",
    "weak",
    "clean",
    "dirty",
    "dry",
    "wet",
    "rich",
    "poor",
    "happy",
    "sad",
    "young",
    "early",
    "late",
    "close",
    "far",
    "deep",
    "wide",
    "thick",
    "thin",
    "square",
    "round",
]

# Topic clusters for variety
TOPIC_KITCHEN = [
    "table",
    "chair",
    "plate",
    "cup",
    "glass",
    "spoon",
    "fork",
    "knife",
    "cooking",
    "eating",
    "dinner",
    "breakfast",
    "lunch",
    "food",
    "drink",
    "water",
    "coffee",
    "tea",
    "juice",
    "bread",
    "cheese",
    "fruit",
    "vegetable",
    "meat",
    "fish",
    "rice",
    "soup",
    "salad",
    "dessert",
    "snack",
    "kitchen",
    "stove",
    "oven",
    "fridge",
    "freezer",
    "microwave",
    "sink",
    "dishwasher",
    "counter",
]
TOPIC_OFFICE = [
    "desk",
    "computer",
    "monitor",
    "keyboard",
    "mouse",
    "printer",
    "phone",
    "email",
    "meeting",
    "report",
    "document",
    "file",
    "folder",
    "paperwork",
    "deadline",
    "project",
    "client",
    "manager",
    "colleague",
    "schedule",
    "calendar",
    "appointment",
    "conference",
    "presentation",
    "note",
    "memo",
    "signature",
    "stapler",
    "pencil",
    "marker",
    "whiteboard",
]
TOPIC_OUTDOOR = [
    "park",
    "street",
    "road",
    "tree",
    "flower",
    "garden",
    "grass",
    "sky",
    "cloud",
    "sun",
    "rain",
    "snow",
    "wind",
    "mountain",
    "river",
    "lake",
    "forest",
    "beach",
    "ocean",
    "field",
    "path",
    "walk",
    "run",
    "bike",
    "drive",
    "car",
    "bus",
    "train",
    "plane",
    "ship",
    "boat",
]
TOPIC_HOME = [
    "bedroom",
    "bathroom",
    "living",
    "window",
    "curtain",
    "carpet",
    "couch",
    "sofa",
    "lamp",
    "rug",
    "closet",
    "drawer",
    "shelf",
    "picture",
    "frame",
    "clock",
    "mirror",
    "towel",
    "soap",
    "shampoo",
    "brush",
    "comb",
    "razor",
    "toothbrush",
    "pillow",
    "blanket",
    "sheet",
    "mattress",
    "dresser",
    "nightstand",
]
TOPIC_TECHNOLOGY = [
    "phone",
    "tablet",
    "laptop",
    "desktop",
    "server",
    "network",
    "internet",
    "website",
    "app",
    "software",
    "download",
    "upload",
    "update",
    "install",
    "delete",
    "backup",
    "storage",
    "memory",
    "battery",
    "charger",
    "cable",
    "screen",
    "display",
    "camera",
    "speaker",
    "microphone",
    "headphone",
    "device",
    "gadget",
    "smart",
    "digital",
    "online",
    "offline",
]
TOPIC_WEATHER = [
    "sunny",
    "cloudy",
    "rainy",
    "snowy",
    "windy",
    "stormy",
    "foggy",
    "clear",
    "humid",
    "dry",
    "wet",
    "cold",
    "hot",
    "warm",
    "cool",
    "freezing",
    "scorching",
    "mild",
    "breezy",
    "gusty",
    "rain",
    "hail",
    "sleet",
    "thunder",
    "lightning",
    "shower",
    "downpour",
    "drizzle",
    "mist",
    "fog",
]
TOPIC_EMOTION = [
    "happy",
    "sad",
    "angry",
    "scared",
    "excited",
    "bored",
    "tired",
    "stressed",
    "calm",
    "nervous",
    "confident",
    "proud",
    "jealous",
    "grateful",
    "hopeful",
    "worried",
    "relieved",
    "surprised",
    "disappointed",
    "frustrated",
    "content",
    "joyful",
    "lonely",
    "confused",
    "curious",
    "interested",
    "indifferent",
    "enthusiastic",
    "pessimistic",
    "optimistic",
]
TOPIC_TIME = [
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "decade",
    "century",
    "morning",
    "afternoon",
    "evening",
    "night",
    "midnight",
    "noon",
    "dawn",
    "dusk",
    "yesterday",
    "today",
    "tomorrow",
    "weekday",
    "weekend",
    "spring",
    "summer",
    "autumn",
    "winter",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

TOPIC_CLUSTERS = {
    "kitchen": TOPIC_KITCHEN,
    "office": TOPIC_OFFICE,
    "outdoor": TOPIC_OUTDOOR,
    "home": TOPIC_HOME,
    "technology": TOPIC_TECHNOLOGY,
    "weather": TOPIC_WEATHER,
    "emotion": TOPIC_EMOTION,
    "time": TOPIC_TIME,
}

# Valid topic names for selection
VALID_TOPICS = ["kitchen", "office", "outdoor", "home", "technology", "weather", "emotion", "time"]


class SyntheticGeneratorError(Exception):
    """Raised when synthetic negative generation fails."""

    pass


def _build_default_words() -> list[str]:
    """Build the default word list from all categories.

    Returns:
        List of unique default words.
    """
    all_words: set[str] = set()
    all_words.update(DEFAULT_ARTICLES)
    all_words.update(DEFAULT_PRONOUNS)
    all_words.update(DEFAULT_PREPOSITIONS)
    all_words.update(DEFAULT_VERBS)
    all_words.update(DEFAULT_NOUNS)
    return sorted(all_words)


# Build default word list once
DEFAULT_WORDS = _build_default_words()


def _extract_wake_word_words(wake_word: str) -> set[str]:
    """Extract individual words from wake word phrase.

    Args:
        wake_word: The wake word phrase (e.g., "hey assistant").

    Returns:
        Set of lowercase words from the wake word.
    """
    return {w.lower() for w in wake_word.replace("_", " ").split()}


def _contains_wake_word(phrase: str, wake_word_words: set[str]) -> bool:
    """Check if phrase contains any word from wake word.

    Args:
        phrase: The phrase to check.
        wake_word_words: Set of wake word words.

    Returns:
        True if phrase contains any wake word word.
    """
    phrase_words = set(phrase.lower().split())
    return bool(phrase_words & wake_word_words)


def _get_random_word(word_list: list[str], rng: random.Random, exclude: set[str]) -> str:
    """Get a random word from list, excluding given words.

    Args:
        word_list: List of words to choose from.
        rng: Random number generator.
        exclude: Words to exclude.

    Returns:
        A random word not in exclude set.
    """
    available = [w for w in word_list if w not in exclude]
    if not available:
        available = word_list
    return rng.choice(available)


def _generate_random_phrase(
    word_list: list[str],
    wake_word_words: set[str],
    rng: random.Random,
    min_words: int = 2,
    max_words: int = 4,
) -> str:
    """Generate a random word phrase.

    Args:
        word_list: List of words to use.
        wake_word_words: Set of wake word words to exclude.
        rng: Random number generator.
        min_words: Minimum number of words.
        max_words: Maximum number of words.

    Returns:
        A random phrase that doesn't contain wake word words.
    """
    num_words = rng.randint(min_words, max_words)
    phrase_words: list[str] = []

    for _ in range(num_words):
        word = _get_random_word(word_list, rng, wake_word_words)
        phrase_words.append(word)

    return " ".join(phrase_words)


def _generate_sentence_like(
    word_list: list[str],
    wake_word_words: set[str],
    rng: random.Random,
) -> str:
    """Generate a sentence-like phrase with article + noun + verb structure.

    Args:
        word_list: List of words to use.
        wake_word_words: Set of wake word words to exclude.
        rng: Random number generator.

    Returns:
        A sentence-like phrase.
    """
    # Select words from different categories if possible
    subject = _get_random_word(word_list, rng, wake_word_words)
    verb = _get_random_word(word_list, rng, wake_word_words)
    obj = _get_random_word(word_list, rng, wake_word_words)

    return f"{subject} {verb} {obj}"


def _generate_topic_phrase(
    topic: str,
    rng: random.Random,
) -> str:
    """Generate a phrase from a topic cluster.

    Args:
        topic: Topic name (e.g., "kitchen", "office").
        rng: Random number generator.

    Returns:
        A phrase from the topic cluster.
    """
    if topic not in TOPIC_CLUSTERS:
        topic = "kitchen"  # Default fallback

    cluster = TOPIC_CLUSTERS[topic]
    num_words = rng.randint(2, 4)
    words = rng.sample(cluster, min(num_words, len(cluster)))
    return " ".join(words)


def generate_synthetic_negatives(
    count: int,
    word_list: list[str] | None = None,
    seed: int | None = None,
    wake_word: str = "hey assistant",
    min_word_count: int = 2,
    max_word_count: int = 4,
    strategy: str = "random",
    topics: list[str] | None = None,
) -> list[str]:
    """Generate random word concatenations that don't contain the wake word.

    Generates synthetic negative phrases by randomly concatenating words from a word list.
    All generated phrases are validated to not contain any words from the wake word phrase.

    Args:
        count: Number of synthetic negatives to generate.
        word_list: Custom word list to use. If None, uses default English words.
        seed: Random seed for deterministic generation. If None, uses system randomness.
        wake_word: The wake word phrase to avoid (default: "hey assistant").
        min_word_count: Minimum number of words per phrase (default: 2).
        max_word_count: Maximum number of words per phrase (default: 4).
        strategy: Generation strategy - "random", "sentence", or "topic" (default: "random").
        topics: List of topics for topic strategy. If None, uses all topics.

    Returns:
        List of unique synthetic negative phrases that don't contain wake word words.

    Raises:
        SyntheticGeneratorError: If generation fails or invalid parameters provided.

    Examples:
        >>> negatives = generate_synthetic_negatives(100, seed=42)
        >>> len(negatives)
        100
        >>> all(len(n.split()) >= 2 for n in negatives)
        True
    """
    if count <= 0:
        raise SyntheticGeneratorError(f"count must be positive, got {count}")

    if min_word_count < 1:
        raise SyntheticGeneratorError(f"min_word_count must be at least 1, got {min_word_count}")

    if max_word_count < min_word_count:
        raise SyntheticGeneratorError(
            f"max_word_count ({max_word_count}) must be >= min_word_count ({min_word_count})"
        )

    # Set up RNG
    rng = random.Random(seed)

    # Use default word list if none provided
    words = word_list if word_list is not None else DEFAULT_WORDS

    if not words:
        raise SyntheticGeneratorError("word_list cannot be empty")

    # Extract wake word words to exclude
    wake_word_words = _extract_wake_word_words(wake_word)

    log.info(
        "generating_synthetic_negatives",
        count=count,
        strategy=strategy,
        wake_word=wake_word,
        excluded_words=list(wake_word_words),
        word_list_size=len(words),
        seed=seed,
    )

    # Strategy selection
    if strategy not in ("random", "sentence", "topic"):
        log.warning("unknown_strategy", strategy=strategy, default="random")
        strategy = "random"

    # Topic list for topic strategy
    available_topics = topics if topics is not None else VALID_TOPICS

    # Generate phrases
    generated: list[str] = []
    seen: set[str] = set()
    max_attempts = count * 100  # Prevent infinite loops
    attempts = 0

    while len(generated) < count and attempts < max_attempts:
        attempts += 1

        try:
            if strategy == "sentence":
                phrase = _generate_sentence_like(words, wake_word_words, rng)
            elif strategy == "topic":
                topic = rng.choice(available_topics)
                phrase = _generate_topic_phrase(topic, rng)
            else:  # random
                phrase = _generate_random_phrase(
                    words,
                    wake_word_words,
                    rng,
                    min_words=min_word_count,
                    max_words=max_word_count,
                )

            # Validate phrase
            phrase_lower = phrase.lower()

            # Check for wake word words
            if _contains_wake_word(phrase, wake_word_words):
                continue

            # Check for uniqueness
            if phrase_lower in seen:
                continue

            seen.add(phrase_lower)
            generated.append(phrase)

        except Exception as e:
            log.warning("phrase_generation_error", error=str(e))
            continue

    # Final validation
    final_count = len(generated)
    log.info(
        "synthetic_negatives_complete",
        generated=final_count,
        requested=count,
        attempts=attempts,
    )

    if final_count < count:
        log.warning(
            "could_not_generate_requested_count",
            generated=final_count,
            requested=count,
            reason="word_list_too_small_or_too_restrictive",
        )

    return generated


__all__ = [
    "generate_synthetic_negatives",
    "SyntheticGeneratorError",
    "DEFAULT_WORDS",
    "VALID_TOPICS",
]
