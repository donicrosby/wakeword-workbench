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
from wakeword_workbench.logging_config import configure_logging, get_logger

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
        # Placeholder for actual pipeline logic (Task 7+)
        task = progress.add_task("[cyan]Initializing pipeline...", total=None)
        progress.update(task, description="[green]Pipeline ready (stub)")

    console.print("[bold green]✓[/bold green] Pipeline completed successfully")
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
        console.print(f"[bold]TTS backend:[/bold] {config.tts.backend}")
    except ConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    console.print("[bold green]✓[/bold green] Config validation passed")
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
