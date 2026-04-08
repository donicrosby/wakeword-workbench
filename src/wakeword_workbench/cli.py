"""Command-line interface for WakeWord Workbench."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from wakeword_workbench import __version__
from wakeword_workbench.config import ConfigError, load_config
from wakeword_workbench.dataset.generator import DatasetGenerator, GeneratorError
from wakeword_workbench.dataset.metadata import Manifest, ManifestEntry
from wakeword_workbench.logging_config import configure_logging, get_logger
from wakeword_workbench.mining.extractor import ExtractedClip, extract_false_positives
from wakeword_workbench.mining.long_audio import process_long_audio
from wakeword_workbench.mining.merge_back import MergeBackError, add_to_training
from wakeword_workbench.mining.model_loader import ModelLoadError, load_onnx_model
from wakeword_workbench.tts.cache import TTSCache

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_CONFIG_ERROR = 2

# Console for rich output
console = Console()

# Logger configuration (initialized in cli_main callback)
log = get_logger()


def version_callback(version: bool) -> None:
    """Print version and exit."""
    if version:
        console.print(f"[bold cyan]WakeWord Workbench[/bold cyan] v{__version__}")
        raise typer.Exit(code=EXIT_SUCCESS)


app = typer.Typer(
    name="wakeword-workbench",
    help="WakeWord Workbench - Training toolkit for micro wake word detection",
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)


@app.callback()
def cli_main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Only show ERROR logs",
    ),
) -> None:
    """Global options for WakeWord Workbench CLI."""
    configure_logging(verbose=verbose, quiet=quiet)
    ctx.meta["verbose"] = verbose
    ctx.meta["quiet"] = quiet


@app.command(name="run")
def run_command(
    config_path: Annotated[Path, typer.Argument(help="Path to config file")],
) -> None:
    """Run the full training pipeline from a config file."""
    console.print(
        Panel.fit(
            "[bold]WakeWord Workbench[/bold] - Starting training pipeline",
            border_style="cyan",
        )
    )

    # Validate config file exists
    if not config_path.exists():
        console.print(f"[bold red]Error:[/bold red] Config file not found: {config_path}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    log.info("starting-pipeline", config=str(config_path))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Loading config...", total=None)

        # Load configuration
        try:
            config = load_config(config_path)
        except ConfigError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            log.error("config-load-failed", error=str(e))
            raise typer.Exit(code=EXIT_CONFIG_ERROR) from None

        progress.update(task, description="[cyan]Initializing generator...")

        # Instantiate and run dataset generator
        try:
            generator = DatasetGenerator(config)
            result = generator.generate()
        except GeneratorError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            log.error("generation-failed", error=str(e))
            raise typer.Exit(code=EXIT_ERROR) from None

        progress.update(task, description="[green]Dataset generation complete")

    # Display success summary
    console.print("\n[bold]Generation Results:[/bold]")
    console.print(f"  Positives: {result.total_positives}")
    console.print(f"  Negatives: {result.total_negatives}")
    console.print(f"  Total:     {result.total_entries}")
    console.print(f"  Ratio:     {result.actual_ratio:.2f}")
    console.print(f"  Output:    {result.output_dir}")
    console.print(f"  Time:      {result.generation_time_seconds:.1f}s")

    if result.has_warnings():
        console.print(f"\n[yellow]Warnings ({len(result.warnings)}):[/yellow]")
        for warning in result.warnings[:5]:
            console.print(f"  • {warning}")
        if len(result.warnings) > 5:
            console.print(f"  ... and {len(result.warnings) - 5} more")

    log.info("pipeline-complete", **result.summary())
    console.print("\n[bold green]✓[/bold green] Pipeline completed successfully")
    raise typer.Exit(code=EXIT_SUCCESS)


@app.command(name="validate")
def validate_command(
    config_path: Annotated[Path, typer.Argument(help="Path to config file")],
) -> None:
    """Validate a config file without running the pipeline."""
    console.print(
        Panel.fit(
            "[bold]WakeWord Workbench[/bold] - Validating config",
            border_style="cyan",
        )
    )

    # Validate config file exists
    if not config_path.exists():
        console.print(f"[bold red]Error:[/bold red] Config file not found: {config_path}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    # Load and validate config
    try:
        config = load_config(config_path)
        log.info("validating-config", config=str(config_path))
        console.print(f"[bold]Wake word:[/bold] {config.wake_word}")
        console.print(f"[bold]Samples:[/bold] {config.samples.positives} positives")
        console.print("[bold]TTS providers:[/bold]")
        for provider in config.tts.providers:
            voice_label = "voice" if len(provider.voices) == 1 else "voices"
            console.print(
                f"  - {provider.backend} ({len(provider.voices)} {voice_label})"
                f" speed={provider.speed}"
            )
    except ConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from None

    console.print("[bold green]✓[/bold green] Config validation passed")
    raise typer.Exit(code=EXIT_SUCCESS)


@app.command(name="cache-clear")
def cache_clear_command(
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            help="TTS cache directory (default: ~/.cache/wakeword_workbench/tts/)",
        ),
    ] = None,
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Clear the TTS synthesis cache."""
    cache = TTSCache(cache_dir=cache_dir) if cache_dir else TTSCache()
    stats = cache.get_stats()

    if stats["num_entries"] == 0:
        console.print("[yellow]Cache is already empty.[/yellow]")
        raise typer.Exit(code=EXIT_SUCCESS)

    console.print("[bold]TTS Cache[/bold]")
    console.print(f"  Entries : {stats['num_entries']}")
    console.print(f"  Size    : {stats['size_mb']:.2f} MB")
    console.print(f"  Location: {cache._cache_dir}")

    if not yes:
        confirm = typer.prompt(
            "\nClear all cached entries?",
            default="n",
            show_default=True,
        )
        if confirm.lower() not in ("y", "yes"):
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(code=EXIT_SUCCESS)

    cache.clear()
    console.print("[bold green]✓[/bold green] Cache cleared successfully")
    raise typer.Exit(code=EXIT_SUCCESS)


@app.command(name="mine")
def mine_command(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            help="Path to ONNX model file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    audio: Annotated[
        str,
        typer.Option(
            "--audio",
            "-a",
            help="Path to audio file(s). Supports wildcards (e.g., '*.wav')",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for extracted clips",
            file_okay=False,
            dir_okay=True,
        ),
    ],
    threshold: float = typer.Option(
        0.7,
        "--threshold",
        "-t",
        help="Confidence threshold for extraction (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
) -> None:
    """Mine hard negatives from long audio recordings.

    Process audio files to find false positives (segments where the model
    predicts the wake word with high confidence but no wake word is present).
    Extracted clips are saved for use as hard negative training samples.
    """
    console.print(
        Panel.fit(
            "[bold]WakeWord Workbench[/bold] - Mining hard negatives",
            border_style="cyan",
        )
    )

    # Validate threshold range (typer handles basic min/max but we ensure ConfigError)
    if not 0.0 <= threshold <= 1.0:
        console.print(
            f"[bold red]Error:[/bold red] Threshold must be between 0.0 and 1.0, got {threshold}"
        )
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    # Expand wildcards in audio path
    audio_path_pattern = Path(audio)
    if audio_path_pattern.is_absolute():
        # For absolute paths, glob from the parent directory
        audio_files = list(audio_path_pattern.parent.glob(audio_path_pattern.name))
    else:
        # For relative paths, use current directory
        audio_files = list(Path().glob(audio))
    if not audio_files:
        console.print(f"[bold red]Error:[/bold red] No audio files found matching: {audio}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    # Create output directory if needed
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(f"[bold red]Error:[/bold red] Failed to create output directory: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    log.info(
        "starting-mining",
        model=str(model),
        audio_pattern=audio,
        audio_files=len(audio_files),
        output=str(output),
        threshold=threshold,
    )

    # Load ONNX model
    try:
        model_fn = load_onnx_model(model)
    except ModelLoadError as e:
        console.print(f"[bold red]Error:[/bold red] Failed to load model: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    all_clips: list[ExtractedClip] = []

    # Process each audio file with progress indication
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing audio files...", total=len(audio_files))

        for audio_file in audio_files:
            progress.update(task, description=f"[cyan]Processing {audio_file.name}...")

            try:
                # Process long audio to get predictions
                predictions = process_long_audio(
                    model=model_fn,
                    audio_path=audio_file,
                )

                # Extract false positives above threshold
                clips = extract_false_positives(
                    predictions=predictions,
                    audio_path=audio_file,
                    threshold=threshold,
                    output_dir=output,
                )

                all_clips.extend(clips)
                log.info(
                    "audio-processing-complete",
                    file=str(audio_file),
                    predictions=len(predictions),
                    clips_extracted=len(clips),
                )

            except (FileNotFoundError, NotADirectoryError, ValueError) as e:
                console.print(f"[bold red]Error:[/bold red] Failed to process {audio_file}: {e}")
                raise typer.Exit(code=EXIT_ERROR) from None

            progress.advance(task)

    # Create manifest with all extracted clips
    manifest = Manifest()
    for clip in all_clips:
        entry = ManifestEntry(
            path=str(clip.clip_path.relative_to(output)),
            label=0,  # Hard negatives
            text="",  # Empty text for negatives
            voice=None,
            duration_ms=int(clip.duration * 1000),
            sample_rate=16000,
            metadata={
                "source": str(clip.original_path),
                "timestamp": clip.timestamp,
                "prediction": clip.prediction,
                "threshold": clip.threshold,
            },
        )
        manifest.add(entry)

    # Save manifest
    manifest_path = output / "manifest.jsonl"
    try:
        manifest.save(manifest_path)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to save manifest: {e}")
        raise typer.Exit(code=EXIT_ERROR) from None

    log.info(
        "mining-complete",
        total_clips=len(all_clips),
        manifest=str(manifest_path),
    )

    console.print(f"[bold green]✓[/bold green] Extracted {len(all_clips)} clips")
    console.print(f"[bold green]✓[/bold green] Saved manifest to {manifest_path}")
    raise typer.Exit(code=EXIT_SUCCESS)


@app.command(name="merge")
def merge_command(
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            "-s",
            help="Path to source manifest (new negatives to merge)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    target: Annotated[
        Path,
        typer.Option(
            "--target",
            "-t",
            help="Path to target training manifest",
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    backup: bool = typer.Option(
        False,
        "--backup",
        "-b",
        help="Create timestamped backup of target before merging",
    ),
) -> None:
    """Merge mined hard negatives into training dataset manifests.

    Takes a manifest of newly mined hard negatives and appends them to an
    existing training manifest. Creates a backup of the target manifest
    before modification if --backup is specified.
    """
    console.print(
        Panel.fit(
            "[bold]WakeWord Workbench[/bold] - Merging hard negatives",
            border_style="cyan",
        )
    )

    if not str(source).endswith(".jsonl"):
        console.print(f"[bold red]Error:[/bold red] Source must be a JSONL file: {source}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    if not str(target).endswith(".jsonl"):
        console.print(f"[bold red]Error:[/bold red] Target must be a JSONL file: {target}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    if not target.exists():
        console.print(f"[bold red]Error:[/bold red] Target manifest not found: {target}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    log.info(
        "starting-merge",
        source=str(source),
        target=str(target),
        backup=backup,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Merging manifests...", total=None)

        try:
            result = add_to_training(
                new_negatives_manifest=source,
                training_manifest=target,
                backup=backup,
            )
            progress.update(task, description="[green]Merge completed")
        except MergeBackError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            log.error("merge-failed", error=str(e))
            raise typer.Exit(code=EXIT_ERROR) from None

    console.print("\n[bold]Merge Results:[/bold]")
    console.print(f"  Source entries   : {result.added_count}")
    console.print(f"  Total in target  : {result.total_count}")

    if result.backup_path:
        console.print(f"  Backup created   : {result.backup_path}")

    if result.errors:
        console.print(f"\n[yellow]Warnings ({len(result.errors)}):[/yellow]")
        for error in result.errors[:5]:
            console.print(f"  • {error}")
        if len(result.errors) > 5:
            console.print(f"  ... and {len(result.errors) - 5} more")

    log.info(
        "merge-complete",
        added=result.added_count,
        total=result.total_count,
        backup=str(result.backup_path) if result.backup_path else None,
        warnings=len(result.errors),
    )

    console.print("\n[bold green]✓[/bold green] Merge completed successfully")
    raise typer.Exit(code=EXIT_SUCCESS)


def main() -> None:
    """Entry point for the CLI."""
    try:
        app()
    except typer.Exit as e:
        sys.exit(e.exit_code)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        log.exception("unhandled-error")
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
