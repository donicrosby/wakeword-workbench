"""Structured logging configuration for WakeWord Workbench.

This module provides centralized logging configuration using structlog for:
- Structured JSON output for log files (machine-parseable)
- Human-readable colored output for console
- Configurable log levels via CLI flags
- Optional file output with rotation support

Example usage:
    from wakeword_workbench.logging_config import configure_logging, get_logger

    configure_logging(verbose=True, log_file="app.log")
    log = get_logger()
    log.info("starting", count=100)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from structlog.types import EventDict, Processor

if TYPE_CHECKING:
    from collections.abc import Callable

# Log level constants
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"

# Color codes for console output (ANSI)
COLOR_RESET = "\x1b[0m"
COLOR_BOLD = "\x1b[1m"
COLOR_DIM = "\x1b[2m"
COLOR_RED = "\x1b[31m"
COLOR_GREEN = "\x1b[32m"
COLOR_YELLOW = "\x1b[33m"
COLOR_BLUE = "\x1b[34m"
COLOR_CYAN = "\x1b[36m"
COLOR_GRAY = "\x1b[90m"

# Log level colors
LEVEL_COLORS = {
    "DEBUG": COLOR_GRAY,
    "INFO": COLOR_BLUE,
    "WARNING": COLOR_YELLOW,
    "ERROR": COLOR_RED,
    "CRITICAL": f"{COLOR_RED}{COLOR_BOLD}",
}

# Current log level (set by configure_logging)
_current_level: str = LOG_LEVEL_INFO

# File logger instance (if configured)
_file_logger: Any = None
_log_file: Path | None = None


def _add_timestamp(logger: object, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to log entries."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return event_dict


def _add_log_level(logger: object, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def _add_caller_info(logger: object, method_name: str, event_dict: EventDict) -> EventDict:
    """Add caller information (module, function, line) to log entries."""
    # Get caller's frame
    frame = sys._getframe(3)  # noqa: SLF001
    if frame is not None:
        event_dict["module"] = frame.f_globals.get("__name__", "<unknown>")
    return event_dict


def _console_renderer(logger: object, method_name: str, event_dict: EventDict) -> str:
    """Render log entry for console with colors and formatting."""
    # Extract fields
    level = event_dict.pop("level", method_name.upper())
    timestamp = event_dict.pop("timestamp", "")
    module = event_dict.pop("module", "")
    event = event_dict.pop("event", "")

    # Get color for level
    level_color = LEVEL_COLORS.get(level, "")

    # Format timestamp (show only time part for readability)
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_str = dt.strftime("%H:%M:%S")
        except ValueError:
            time_str = timestamp[11:19] if len(timestamp) > 19 else timestamp
    else:
        time_str = ""

    # Build colored level badge
    level_str = f"{level_color}[{level:<7}]{COLOR_RESET}"

    # Build message
    if module:
        msg = f"{COLOR_DIM}{module}:{COLOR_RESET} {event}"
    else:
        msg = event

    # Add key-value pairs if present
    extras = []
    for key, value in sorted(event_dict.items()):
        extras.append(f"{COLOR_CYAN}{key}={COLOR_RESET}{value}")

    extra_str = " ".join(extras)

    # Build final line
    parts = [f"{COLOR_DIM}{time_str}{COLOR_RESET}", level_str]
    if extra_str:
        parts.extend([msg, extra_str])
    else:
        parts.append(msg)

    return " ".join(parts)


def _json_renderer(logger: object, method_name: str, event_dict: EventDict) -> str:
    """Render log entry as JSON for file output."""
    import json

    # Build dict for JSON output
    output = {
        "timestamp": event_dict.pop("timestamp", ""),
        "level": event_dict.pop("level", method_name.upper()),
        "message": event_dict.pop("event", ""),
    }

    # Add module if present
    if "module" in event_dict:
        output["module"] = event_dict.pop("module")

    # Add remaining fields as context
    if event_dict:
        output["context"] = event_dict

    return json.dumps(output, separators=(",", ":"))


def _isatty() -> bool:
    """Check if stdout is a TTY (for determining output format)."""
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _get_processors(
    console_only: bool = True,
) -> list[Processor]:
    """Build processor chain for structlog.

    Args:
        console_only: If True, use console renderer; otherwise JSON for files.

    Returns:
        List of processors for the logging pipeline.
    """
    processors: list[Processor | Callable[..., Any]] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_timestamp,
        _add_log_level,
        _add_caller_info,
    ]

    # Add renderer based on output mode
    if console_only and _isatty():
        processors.append(_console_renderer)
    else:
        processors.append(_json_renderer)

    return processors


def configure_logging(
    verbose: bool = False,
    quiet: bool = False,
    log_file: Path | str | None = None,
    log_level: str | None = None,
) -> None:
    """Configure structured logging for the application.

    Sets up structlog with appropriate processors based on:
    - Verbosity level (--verbose for DEBUG, --quiet for ERROR, default INFO)
    - Output destination (console with colors, JSON for files)
    - Optional log file with rotation support

    Args:
        verbose: Enable DEBUG level logging (overrides quiet).
        quiet: Enable ERROR-only logging (overridden by verbose).
        log_file: Optional path to write logs to (JSON format).
        log_level: Explicit log level override (DEBUG, INFO, WARNING, ERROR).

    Example:
        >>> from wakeword_workbench.logging_config import configure_logging
        >>> configure_logging(verbose=True, log_file="app.log")
    """
    global _current_level, _log_file  # noqa: PLW0603

    # Determine log level
    if log_level:
        _current_level = log_level.upper()
    elif quiet:
        _current_level = LOG_LEVEL_ERROR
    elif verbose:
        _current_level = LOG_LEVEL_DEBUG
    else:
        _current_level = LOG_LEVEL_INFO

    # Get the numeric level for filtering
    level_mapping = {
        LOG_LEVEL_DEBUG: 10,
        LOG_LEVEL_INFO: 20,
        LOG_LEVEL_WARNING: 30,
        LOG_LEVEL_ERROR: 40,
    }
    numeric_level = level_mapping.get(_current_level, 20)

    # Determine if we should use console or JSON renderer
    console_only = log_file is None and _isatty()

    # Build processor chain
    processors = _get_processors(console_only=console_only)

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(
            file=sys.stderr if console_only else sys.stdout
        ),
        cache_logger_on_first_use=False,
    )

    # Set up file logging if requested
    if log_file:
        _log_file = Path(log_file)
        _setup_file_logging(_log_file)


def _setup_file_logging(log_path: Path) -> None:
    """Set up file logging with JSON format.

    Args:
        log_path: Path to the log file.
    """
    global _file_logger  # noqa: PLW0603

    # Create parent directories if needed
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Open file in append mode for file rotation compatibility
    try:
        _file_logger = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    except OSError as e:
        # Fall back to console-only logging if file can't be opened
        import warnings

        warnings.warn(f"Could not open log file {log_path}: {e}. Using console only.", UserWarning)
        _file_logger = None


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Optional logger name for context.

    Returns:
        Configured structlog logger instance.

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("message", key="value")
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


def get_log_level() -> str:
    """Get the current configured log level.

    Returns:
        Current log level as string (DEBUG, INFO, WARNING, ERROR).
    """
    return _current_level


def is_debug_enabled() -> bool:
    """Check if DEBUG logging is currently enabled.

    Returns:
        True if debug logging is enabled.
    """
    return _current_level == LOG_LEVEL_DEBUG


def shutdown_logging() -> None:
    """Clean up logging resources (close file handles, etc).

    Call this at application shutdown if file logging was used.
    """
    global _file_logger  # noqa: PLW0603

    if _file_logger is not None:
        try:
            _file_logger.close()
        except Exception:
            pass
        _file_logger = None
