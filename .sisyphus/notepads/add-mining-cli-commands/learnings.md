

## Task: Add merge command for merging hard negatives

### Key Findings

1. **Typer File Validation with exists=True**
   - When using `exists=True` in `typer.Option()`, Typer validates the file exists before our code runs
   - For the target file (which we don't want to require to exist), don't use `exists=True`
   - This allows us to show custom error messages

2. **Exit Code Strategy**
   - 0 = success (entries merged, manifest saved)
   - 1 = error (merge operation failed, backup failed, save failed)
   - 2 = config error (invalid file extension, target not found)

3. **MergeResult Structure**
   - `added_count`: number of entries added from source
   - `total_count`: total entries in target after merge
   - `backup_path`: path to backup file if --backup was specified
   - `errors`: list of warnings (duplicates, missing files, etc.)

4. **Testing Pattern for Merge Command**
   - Mock `add_to_training()` from mining.merge_back module
   - Create `MagicMock()` for `MergeResult` with required attributes
   - Test warning truncation (show first 5, then "... and N more")
   - Test both with and without --backup flag

5. **CLI Command Pattern**
   ```python
   @app.command(name="merge")
   def merge_command(
       source: Annotated[Path, typer.Option(..., exists=True)],
       target: Annotated[Path, typer.Option(...)],  # No exists=True
       backup: bool = typer.Option(False, "--backup"),
   ) -> None:
   ```

6. **Statistics Display**
   - Source entries: result.added_count
   - Total in target: result.total_count
   - Backup path (if created): result.backup_path
   - Warnings count: len(result.errors)

### Code Pattern for merge command

```python
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task("[cyan]Merging manifests...", total=None)
    result = add_to_training(...)
    progress.update(task, description="[green]Merge completed")
```

### Testing Lessons

- Mocking `add_to_training` is simpler than mocking the underlying Manifest operations
- Test warning truncation to verify UX behavior
- Test edge cases: no entries added, many warnings, backup creation
- The manifest extensions (.jsonl) validation is done before calling add_to_training
