# Wave 1 - Task 6: Logging System

## Task Summary
Created `src/wakeword_workbench/logging_config.py` with structlog-based logging.

## Key Implementation Details

### Log Levels
- DEBUG: Enabled with `--verbose`
- INFO: Default level
- WARNING: Warnings that don't stop execution
- ERROR: Problems that may affect results
- Quiet mode (`--quiet`) shows only ERROR

### Output Formats
- **Console (TTY)**: Human-readable colored output with timestamps
- **Non-TTY/JSON**: Machine-parseable JSON format for log files

### Processors Chain
1. `merge_contextvars` - Merge context variables
2. `add_log_level` - Add log level to entries
3. `PositionalArgumentsFormatter` - Format positional args
4. `StackInfoRenderer` - Add stack info
5. `UnicodeDecoder` - Handle unicode
6. `_add_timestamp` - Custom ISO timestamp
7. `_add_log_level` - Custom level field
8. `_add_caller_info` - Add module info
9. Console or JSON renderer

### Color Codes (ANSI)
- DEBUG: Gray
- INFO: Blue
- WARNING: Yellow
- ERROR: Red
- CRITICAL: Bold Red

## Gotchas Encountered
- `structlog.stdlib.add_logger_name` doesn't work with PrintLogger - removed
- PrintLogger has no `name` attribute - handled by removing the processor
- Must use `get_logger()` after `configure_logging()` for proper initialization

## Files Created
- `src/wakeword_workbench/logging_config.py`

## Files Modified
- `src/wakeword_workbench/cli.py` - Now imports from logging_config module

## Verification
```bash
# Test log levels
uv run python -c "
from wakeword_workbench.logging_config import configure_logging, get_logger, get_log_level

configure_logging(verbose=True)
log = get_logger()
log.debug('debug msg')
log.info('info msg')
log.warning('warning msg')
log.error('error msg')

configure_logging(quiet=True)
log = get_logger()
log.info('should not appear')
log.error('should appear')
"
```
