

## Task: Add mine command for hard negative mining

### Key Findings

1. **Pathlib.glob() with Absolute Paths**
   - `Path().glob()` doesn't support absolute paths in the pattern
   - Must split the path: use `path.parent.glob(path.name)` for absolute paths
   - For relative paths, use `Path().glob(pattern)` directly

2. **Typer Built-in Validation**
   - Using `exists=True` in `typer.Option()` causes Typer to validate before our code runs
   - This means we can't show custom error messages for missing files on those options
   - The exit code is still correct (2 = config error)

3. **Wildcard Support Implementation**
   ```python
   audio_path_pattern = Path(audio)
   if audio_path_pattern.is_absolute():
       audio_files = list(audio_path_pattern.parent.glob(audio_path_pattern.name))
   else:
       audio_files = list(Path().glob(audio))
   ```

4. **Mocking Strategy for Tests**
   - Mock `load_onnx_model` to return a callable mock
   - Mock `process_long_audio` to return list of `WindowPrediction`-like mocks
   - Mock `extract_false_positives` to return list of `ExtractedClip`-like mocks
   - Need `MockExtractedClip` class since `ExtractedClip` is a dataclass with `relative_to()` method

5. **Manifest Entry Structure**
   - label=0 for hard negatives
   - text="" (empty) for negatives
   - duration_ms calculated from clip.duration * 1000
   - metadata includes source, timestamp, prediction, threshold

6. **Exit Code Consistency**
   - 0 = success (extracted clips, saved manifest)
   - 1 = error (model load failed, processing error, manifest save failed)
   - 2 = config error (threshold out of range, no files match, missing model)

### Code Pattern for CLI Commands

```python
@app.command(name="mine")
def mine_command(
    model: Annotated[Path, typer.Option(...)],
    audio: Annotated[str, typer.Option(...)],
    output: Annotated[Path, typer.Option(...)],
    threshold: float = typer.Option(0.7, ...),
) -> None:
    """Docstring appears in --help output."""
    # 1. Print panel header
    # 2. Validate inputs
    # 3. Expand wildcards
    # 4. Create output directory
    # 5. Use Progress spinner during processing
    # 6. Collect results
    # 7. Save manifest
    # 8. Print summary and exit with appropriate code
```

### Testing Patterns

- Use `CliRunner` from `typer.testing`
- Mock external dependencies at the module level (`@patch("wakeword_workbench.cli.xxx")`)
- Create mock classes for dataclasses that need methods like `relative_to()`
- Test both success and error paths
- Verify manifest content separately from CLI output
