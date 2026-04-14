"""Dataset generator for wake word training samples."""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

from wakeword_workbench.config import Config, TTSProviderConfig
from wakeword_workbench.logging_config import get_logger
from wakeword_workbench.negatives.phrase_generator import generate_confusions
from wakeword_workbench.negatives.synthetic_generator import generate_synthetic_negatives
from wakeword_workbench.tts.base import TTSError
from wakeword_workbench.tts.pool import BackendPool

from .merger import MergerError, merge
from .metadata import Manifest, ManifestEntry, ManifestError
from .positive_generator import (
    PositiveGenerator,
    PositiveGeneratorError,
)
from .splitter import SplitValidationError, split

log = get_logger(__name__)


class GeneratorError(Exception):
    """Base exception for DatasetGenerator errors."""


class GeneratorConfigError(GeneratorError):
    """Configuration validation failed."""


class GeneratorTTSError(GeneratorError):
    """TTS synthesis failed during negative generation."""


class GeneratorMergeError(GeneratorError):
    """Manifest merge failed."""


class GeneratorSplitError(GeneratorError):
    """Manifest split failed."""


class GeneratorIOError(GeneratorError):
    """File I/O operation failed."""


@dataclass
class GenerationResult:
    """Result of a dataset generation run."""

    train_manifest: Manifest
    val_manifest: Manifest
    test_manifest: Manifest
    total_positives: int
    total_negatives: int
    total_entries: int
    train_count: int
    val_count: int
    test_count: int
    target_ratio: float | None
    actual_ratio: float
    output_dir: Path
    generation_time_seconds: float
    warnings: list[str] = field(default_factory=list)

    def has_warnings(self) -> bool:
        """Check if any warnings were generated."""
        return len(self.warnings) > 0

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary for logging/reporting."""
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


class DatasetGenerator:
    """Orchestrates the full dataset generation pipeline.

    Coordinates PositiveGenerator, negative phrase generation, TTS synthesis,
    merging, and splitting into a unified workflow.
    """

    def __init__(self, config: Config, output_dir: Path | None = None) -> None:
        """Initialize DatasetGenerator.

        Args:
            config: Validated Config object from load_config().
            output_dir: Override output directory. Defaults to config.output.path.

        Raises:
            GeneratorConfigError: If config is invalid for generation.
        """
        if not config.wake_word.strip():
            raise GeneratorConfigError("wake_word cannot be empty")
        if config.samples.positives <= 0:
            raise GeneratorConfigError("samples.positives must be positive")
        if config.samples.negatives_multiplier <= 0:
            raise GeneratorConfigError("samples.negatives_multiplier must be positive")
        if not config.tts.providers:
            raise GeneratorConfigError("tts.providers cannot be empty")

        self.config = config
        self.output_dir = Path(output_dir) if output_dir is not None else Path(config.output.path)
        self._voice_index = 0
        self._validate_files = False
        self._parallelism = 1
        self._providers = config.tts.providers
        self._backend_pool = BackendPool()

        log.info(
            "dataset_generator_init",
            wake_word=self.config.wake_word,
            providers=[provider.backend for provider in self._providers],
            voices=len(self.config.tts.get_all_voices()),
            output_dir=str(self.output_dir),
        )

    def generate(
        self,
        positive_count: int | None = None,
        negatives_multiplier: int | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        ratio: float | None = None,
        validate_files: bool = False,
        parallelism: int = 1,
    ) -> GenerationResult:
        """Generate a complete dataset with train/val/test splits.

        Args:
            positive_count: Override positive sample count.
            negatives_multiplier: Override negatives multiplier.
            train_ratio: Fraction for training split.
            val_ratio: Fraction for validation split.
            test_ratio: Fraction for test split.
            ratio: Target neg:pos ratio for merge; None uses all negatives.
            validate_files: Whether to validate referenced files exist.
            parallelism: Number of I/O worker threads (1-32). TTS remains sequential,
                only file I/O is parallelized. Defaults to 1.

        Returns:
            GenerationResult with manifests and generation statistics.

        Raises:
            GeneratorConfigError: Invalid parameters.
            GeneratorTTSError: TTS synthesis failed.
            GeneratorMergeError: Manifest merge failed.
            GeneratorSplitError: Split validation failed.
            GeneratorIOError: File write/read failed.
        """
        start = time.perf_counter()
        warnings: list[str] = []

        resolved_positive_count = (
            positive_count if positive_count is not None else self.config.samples.positives
        )
        resolved_neg_multiplier = (
            negatives_multiplier
            if negatives_multiplier is not None
            else self.config.samples.negatives_multiplier
        )

        if resolved_positive_count <= 0:
            raise GeneratorConfigError(
                f"positive_count must be positive, got {resolved_positive_count}"
            )
        if resolved_neg_multiplier <= 0:
            raise GeneratorConfigError(
                f"negatives_multiplier must be positive, got {resolved_neg_multiplier}"
            )
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
            raise GeneratorConfigError("train_ratio + val_ratio + test_ratio must sum to 1.0")
        if ratio is not None and ratio <= 0:
            raise GeneratorConfigError(f"ratio must be positive when provided, got {ratio}")
        if parallelism < 1 or parallelism > 32:
            raise GeneratorConfigError(f"parallelism must be between 1 and 32, got {parallelism}")

        total_negatives_target = resolved_positive_count * resolved_neg_multiplier
        output_dir = self._ensure_output_dir()
        self._validate_files = validate_files
        self._parallelism = parallelism

        self._log_progress("start", "beginning dataset generation")
        log.info(
            "generation_parameters",
            positive_count=resolved_positive_count,
            negatives_multiplier=resolved_neg_multiplier,
            target_negatives=total_negatives_target,
            split_train=train_ratio,
            split_val=val_ratio,
            split_test=test_ratio,
            merge_ratio=ratio,
            validate_files=validate_files,
            parallelism=parallelism,
        )

        self._log_progress("positives", "generating positive samples")
        pos_manifest = self._generate_positives(resolved_positive_count, parallelism)

        self._log_progress("negative_phrases", "generating negative phrases")
        negative_phrases = self._generate_negative_phrases(total_negatives_target)
        if len(negative_phrases) < total_negatives_target:
            warnings.append(
                "Generated fewer negative phrases than requested "
                f"({len(negative_phrases)}/{total_negatives_target})"
            )

        self._log_progress("negative_tts", "synthesizing negative phrases")
        neg_manifest = self._synthesize_negatives(negative_phrases)

        self._log_progress("merge", "merging positive and negative manifests")
        combined_manifest = self._merge_manifests(pos_manifest, neg_manifest, ratio)

        self._log_progress("split", "splitting merged manifest")
        train_manifest, val_manifest, test_manifest = self._split_manifest(
            combined_manifest,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        self._log_progress("save", "saving split manifests")
        self._save_splits(train_manifest, val_manifest, test_manifest)

        total_positives = sum(1 for entry in combined_manifest if entry.label == 1)
        total_negatives = sum(1 for entry in combined_manifest if entry.label == 0)
        actual_ratio = total_negatives / total_positives if total_positives > 0 else 0.0

        result = GenerationResult(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            total_positives=total_positives,
            total_negatives=total_negatives,
            total_entries=len(combined_manifest),
            train_count=len(train_manifest),
            val_count=len(val_manifest),
            test_count=len(test_manifest),
            target_ratio=ratio,
            actual_ratio=actual_ratio,
            output_dir=output_dir,
            generation_time_seconds=time.perf_counter() - start,
            warnings=warnings,
        )

        log.info("dataset_generation_complete", **result.summary())
        return result

    def _generate_positives(self, count: int, parallelism: int = 1) -> Manifest:
        """Generate positive samples via PositiveGenerator.

        Args:
            count: Number of positive samples.
            parallelism: Number of I/O worker threads for parallel file writing.

        Returns:
            Manifest with positive entries.

        Raises:
            GeneratorIOError: Positive generation failed.
        """
        try:
            generator = PositiveGenerator(self.config, self.output_dir, parallelism=parallelism)
            manifest_path = generator.generate(count)
            manifest = Manifest.load(manifest_path)
        except (PositiveGeneratorError, ManifestError, OSError) as exc:
            raise GeneratorIOError(f"Failed generating positives: {exc}") from exc

        log.info("positive_generation_complete", count=len(manifest), manifest=str(manifest_path))
        return manifest

    def _generate_negative_phrases(self, total_count: int) -> list[str]:
        """Generate negative phrase strings.

        Args:
            total_count: Total number of negative phrases needed.

        Returns:
            List of unique negative phrase strings.
        """
        if total_count <= 0:
            return []

        custom_phrases = self._get_custom_negative_phrases(limit=total_count)
        remaining_count = max(0, total_count - len(custom_phrases))

        source_counts = self._calculate_negative_source_counts(remaining_count)

        confusion_phrases = (
            self._generate_confusion_phrases(source_counts["confusion"])
            if source_counts["confusion"] > 0
            else []
        )
        synthetic_phrases = (
            self._generate_synthetic_phrases(source_counts["synthetic"])
            if source_counts["synthetic"] > 0
            else []
        )

        seen: set[str] = set()
        combined: list[str] = []
        for phrase in custom_phrases + confusion_phrases + synthetic_phrases:
            normalized = phrase.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            combined.append(phrase.strip())
            if len(combined) >= total_count:
                break

        if len(combined) < total_count:
            shortfall = total_count - len(combined)
            log.warning("negative_phrase_shortfall", shortfall=shortfall)
            for source_name in self._get_fill_source_order():
                fill_phrases = self._generate_phrases_for_source(source_name, shortfall)
                for phrase in fill_phrases:
                    normalized = phrase.strip().lower()
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    combined.append(phrase.strip())
                    if len(combined) >= total_count:
                        break

                shortfall = total_count - len(combined)
                if shortfall <= 0:
                    break

        log.info(
            "negative_phrases_generated",
            requested=total_count,
            custom_generated=len(custom_phrases),
            confusion_generated=len(confusion_phrases),
            synthetic_generated=len(synthetic_phrases),
            total_unique=len(combined),
        )
        return combined

    def _get_custom_negative_phrases(self, limit: int | None = None) -> list[str]:
        """Return user-supplied custom negative phrases in stable order."""
        phrases = self.config.negatives.custom_phrases or []
        if limit is None:
            return phrases.copy()
        return phrases[:limit]

    def _calculate_negative_source_counts(self, total_count: int) -> dict[str, int]:
        """Calculate exact negative counts per enabled source."""
        negatives = self.config.negatives
        weighted_sources: list[tuple[str, float]] = []
        if negatives.confusion.enabled:
            weighted_sources.append(("confusion", negatives.confusion.weight))
        if negatives.synthetic.enabled:
            weighted_sources.append(("synthetic", negatives.synthetic.weight))

        total_weight = sum(weight for _name, weight in weighted_sources)
        counts = {"confusion": 0, "synthetic": 0}
        raw_allocations: list[tuple[str, float]] = []

        for name, weight in weighted_sources:
            raw_count = total_count * (weight / total_weight)
            raw_allocations.append((name, raw_count))
            counts[name] = int(raw_count)

        remaining = total_count - sum(counts.values())
        if remaining > 0:
            remainders = sorted(
                raw_allocations,
                key=lambda item: (item[1] - int(item[1]), item[1]),
                reverse=True,
            )
            for index in range(remaining):
                counts[remainders[index % len(remainders)][0]] += 1

        return counts

    def _get_fill_source_order(self) -> list[str]:
        """Return enabled negative sources ordered by fill priority."""
        negatives = self.config.negatives
        weighted_sources: list[tuple[str, float]] = []
        if negatives.confusion.enabled:
            weighted_sources.append(("confusion", negatives.confusion.weight))
        if negatives.synthetic.enabled:
            weighted_sources.append(("synthetic", negatives.synthetic.weight))

        weighted_sources.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _weight in weighted_sources]

    def _generate_phrases_for_source(self, source_name: str, count: int) -> list[str]:
        """Dispatch phrase generation by configured negative source."""
        if source_name == "confusion":
            return self._generate_confusion_phrases(count)
        if source_name == "synthetic":
            return self._generate_synthetic_phrases(count)
        raise GeneratorConfigError(f"Unknown negative source: {source_name}")

    def _generate_confusion_phrases(self, count: int) -> list[str]:
        """Generate confusion phrases.

        Args:
            count: Number of confusion phrases to generate.

        Returns:
            List of confusion phrase strings.

        Raises:
            GeneratorError: If jellyfish is not installed.
        """
        if count <= 0:
            return []

        try:
            return generate_confusions(
                self.config.wake_word,
                count=count,
                min_similarity=self.config.negatives.confusion.min_similarity,
            )
        except ImportError as exc:
            raise GeneratorError(f"Failed generating confusion phrases: {exc}") from exc
        except ValueError as exc:
            raise GeneratorConfigError(f"Invalid confusion generation parameters: {exc}") from exc

    def _generate_synthetic_phrases(self, count: int) -> list[str]:
        """Generate synthetic negative phrases.

        Args:
            count: Number of synthetic phrases to generate.

        Returns:
            List of synthetic phrase strings.
        """
        if count <= 0:
            return []

        synthetic = self.config.negatives.synthetic
        return generate_synthetic_negatives(
            count=count,
            wake_word=self.config.wake_word,
            word_list=synthetic.word_list,
            min_word_count=synthetic.min_word_count,
            max_word_count=synthetic.max_word_count,
            strategy=synthetic.strategy,
            topics=synthetic.topics,
        )

    def _synthesize_negatives(self, phrases: list[str]) -> Manifest:
        """Synthesize audio for negative phrases and create a Manifest.

        Args:
            phrases: Negative phrase strings to synthesize.

        Returns:
            Manifest with negative entries.

        Raises:
            GeneratorTTSError: TTS synthesis failed.
        """
        negatives_dir = self.output_dir / "negatives"
        negatives_dir.mkdir(parents=True, exist_ok=True)

        entries: list[ManifestEntry] = []
        failed = 0

        provider_voices = [
            (provider, voice) for provider in self._providers for voice in provider.voices
        ]

        # Phase 1: preserve the existing round-robin distribution by pre-assigning
        # a provider/voice to each phrase in original phrase order.
        assigned_provider_voices = [
            provider_voices[(self._voice_index + index) % len(provider_voices)]
            for index in range(len(phrases))
        ]
        self._voice_index += len(phrases)

        # Phase 2: sort execution by backend/runtime/voice so synthesis runs in
        # contiguous groups and minimizes expensive model/voice switching.
        phrase_voice_assignments = [
            (index, phrase, *self._select_voice_for_phrase(phrase, assigned_provider_voices, index))
            for index, phrase in enumerate(phrases)
        ]
        grouped_assignments = sorted(
            phrase_voice_assignments,
            key=lambda assignment: (
                assignment[2].backend,
                assignment[2].acceleration,
                assignment[3],
            ),
        )

        def provider_instance_key(
            provider: TTSProviderConfig,
        ) -> tuple[str, float, str, str | None, str | None]:
            key_getter = cast(Any, getattr(provider, "backend_instance_key", None))
            if callable(key_getter):
                return cast(tuple[str, float, str, str | None, str | None], key_getter())

            backend, _voices, speed, acceleration, device, model_path = provider.backend_cache_key()
            return (backend, speed, acceleration, device, model_path)

        backends_by_key = {
            provider_instance_key(provider): self._backend_pool.get(provider)
            for provider in self._providers
        }

        # Collect write operations for parallel I/O
        write_operations: list[tuple[Path, np.ndarray, ManifestEntry]] = []

        for index, phrase, provider, voice in grouped_assignments:
            filename = f"negative_{index:06d}.wav"
            file_path = negatives_dir / filename
            backend_key = provider_instance_key(provider)

            try:
                backend = backends_by_key[backend_key]

                try:
                    backend.set_voice(voice)
                except NotImplementedError as exc:
                    log.warning(
                        "negative_voice_set_not_supported",
                        backend=provider.backend,
                        voice=voice,
                        error=str(exc),
                    )

                result = backend.synthesize(phrase)
                audio = self._ensure_format(result.audio, result.sample_rate)
                audio = self._ensure_mono(audio)

                duration_ms = int(len(audio) / 16000 * 1000)
                entry = ManifestEntry(
                    path=f"negatives/{filename}",
                    label=0,
                    text=phrase,
                    voice=voice,
                    backend=provider.backend,
                    duration_ms=duration_ms,
                    sample_rate=16000,
                )

                # Store for parallel write
                write_operations.append((file_path, audio, entry))
            except (TTSError, OSError, ValueError) as exc:
                failed += 1
                log.warning(
                    "negative_tts_failed",
                    backend=provider.backend,
                    phrase=phrase,
                    voice=voice,
                    error=str(exc),
                )

        # Perform parallel file I/O for WAV writes
        if write_operations:
            if self._parallelism > 1:
                soundfile = import_module("soundfile")
                with concurrent.futures.ThreadPoolExecutor(max_workers=self._parallelism) as pool:
                    futures = {
                        pool.submit(soundfile.write, file_path, audio, 16000): entry
                        for file_path, audio, entry in write_operations
                    }
                    done, not_done = concurrent.futures.wait(futures.keys())

                    for future in done:
                        entry = futures[future]
                        try:
                            future.result()
                            entries.append(entry)
                        except (OSError, ValueError) as exc:
                            failed += 1
                            log.error(
                                "negative_file_write_failed",
                                path=entry.path,
                                error=str(exc),
                            )
            else:
                # Sequential I/O when parallelism is 1
                soundfile = import_module("soundfile")
                for file_path, audio, entry in write_operations:
                    try:
                        soundfile.write(file_path, audio, 16000)
                        entries.append(entry)
                    except (OSError, ValueError) as exc:
                        failed += 1
                        log.error(
                            "negative_file_write_failed",
                            path=entry.path,
                            error=str(exc),
                        )

        if not entries:
            raise GeneratorTTSError(
                "Failed to synthesize any negative samples. "
                f"attempted={len(phrases)} failed={failed}"
            )

        log.info(
            "negative_tts_complete",
            requested=len(phrases),
            generated=len(entries),
            failed=failed,
            output_dir=str(negatives_dir),
        )
        return Manifest(entries)

    def _select_voice_for_phrase(
        self,
        phrase: str,
        assigned_provider_voices: list[tuple[TTSProviderConfig, str]] | None = None,
        assigned_index: int | None = None,
    ) -> tuple[TTSProviderConfig, str]:
        """Select a provider and voice for synthesizing a phrase.

        Uses round-robin across configured voices.

        Args:
            phrase: Phrase to be synthesized.
            assigned_provider_voices: Optional pre-assigned provider/voice list.
            assigned_index: Index within pre-assigned provider/voice list.

        Returns:
            Tuple of (provider_config, voice_identifier).
        """
        if assigned_provider_voices is not None and assigned_index is not None:
            del phrase  # phrase reserved for future voice selection strategies
            return assigned_provider_voices[assigned_index]

        del phrase  # phrase reserved for future voice selection strategies
        provider_voices = [
            (provider, voice) for provider in self._providers for voice in provider.voices
        ]
        provider, voice = provider_voices[self._voice_index % len(provider_voices)]
        self._voice_index += 1
        return provider, voice

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
            ratio: Target ratio (neg/pos). None uses all entries.
        Returns:
            Combined manifest.

        Raises:
            GeneratorMergeError: Merge failed.
        """
        try:
            merged = merge(
                pos_manifest,
                neg_manifest,
                ratio=ratio,
                validate_files=self._validate_files,
            )
        except MergerError as exc:
            raise GeneratorMergeError(f"Failed merging manifests: {exc}") from exc

        log.info(
            "manifest_merge_complete",
            positive_count=len(pos_manifest),
            negative_count=len(neg_manifest),
            merged_count=len(merged),
            target_ratio=ratio,
        )
        return merged

    def _split_manifest(
        self,
        manifest: Manifest,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple[Manifest, Manifest, Manifest]:
        """Split manifest into train/val/test sets.

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
        try:
            train, val, test = split(
                manifest,
                train=train_ratio,
                val=val_ratio,
                test=test_ratio,
                by="speaker",
            )
        except SplitValidationError as exc:
            raise GeneratorSplitError(f"Failed splitting manifest: {exc}") from exc

        log.info(
            "manifest_split_complete",
            total=len(manifest),
            train=len(train),
            val=len(val),
            test=len(test),
        )
        return train, val, test

    def _save_splits(self, train: Manifest, val: Manifest, test: Manifest) -> None:
        """Save split manifests to output directory.

        Args:
            train: Training manifest.
            val: Validation manifest.
            test: Test manifest.

        Raises:
            GeneratorIOError: File write failed.
        """
        train_path = self.output_dir / "train.jsonl"
        val_path = self.output_dir / "val.jsonl"
        test_path = self.output_dir / "test.jsonl"

        try:
            train.save(train_path)
            val.save(val_path)
            test.save(test_path)
        except (ManifestError, OSError) as exc:
            raise GeneratorIOError(f"Failed saving split manifests: {exc}") from exc

        log.info(
            "split_manifests_saved",
            train_path=str(train_path),
            val_path=str(val_path),
            test_path=str(test_path),
        )

    def _ensure_output_dir(self) -> Path:
        """Ensure output directory exists.

        Returns:
            Output directory path.

        Raises:
            GeneratorIOError: Directory creation failed.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise GeneratorIOError(f"Failed creating output directory: {exc}") from exc
        return self.output_dir

    def _log_progress(self, phase: str, message: str) -> None:
        """Log generation progress with structured context.

        Args:
            phase: Current phase identifier.
            message: Human-readable progress message.
        """
        log.info("dataset_generation_progress", phase=phase, message=message)

    def _ensure_format(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Ensure audio is 16 kHz.

        Args:
            audio: Audio samples.
            sample_rate: Original sample rate.

        Returns:
            Audio resampled to 16 kHz.
        """
        if sample_rate == 16000:
            return audio

        librosa = import_module("librosa")
        resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        return np.asarray(resampled, dtype=np.float32)

    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono.

        Args:
            audio: Input audio samples.

        Returns:
            Mono audio samples.
        """
        if audio.ndim > 1:
            return np.asarray(audio.mean(axis=1), dtype=np.float32)
        return np.asarray(audio, dtype=np.float32)


__all__ = [
    "DatasetGenerator",
    "GenerationResult",
    "GeneratorError",
    "GeneratorConfigError",
    "GeneratorTTSError",
    "GeneratorMergeError",
    "GeneratorSplitError",
    "GeneratorIOError",
]
