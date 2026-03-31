# Issues and Gotchas

## Task 9: PiperBackend Implementation

### Issue 1: Optional Dependencies in Type Hints
**Problem**: When `piper` is not installed, `PiperVoice` is `None`, causing type annotation issues with `PiperVoice | None`.

**Solution**: Use `TYPE_CHECKING` block to import types only for static analysis:
```python
if TYPE_CHECKING:
    from piper import PiperVoice
```

### Issue 2: Mocking Optional Imports in Tests
**Problem**: When `piper` is not installed, patching `PiperVoice.load` fails because `PiperVoice` is `None`.

**Solution**: Patch the entire `PiperVoice` object in the module, not just the `load` method:
```python
with patch.object(piper_module, 'PiperVoice', mock_piper_voice):
    # ...
```

### Issue 3: Resampling Numerical Errors
**Problem**: After resampling, audio can slightly exceed [-1, 1] range due to numerical errors.

**Solution**: Clip the audio after resampling:
```python
audio_data = np.clip(audio_data, -1.0, 1.0)
```

### Issue 4: Class Methods vs Static Methods
**Problem**: Instance methods that don't use `self` cause issues when called via class (e.g., `PiperBackend.list_voices_in_directory()`).

**Solution**: Make such methods `@staticmethod`:
```python
@staticmethod
def list_voices_in_directory(directory: str | Path) -> list[str]:
```

### Issue 5: soundfile and BytesIO
**Problem**: soundfile.read() may not properly handle BytesIO from mocks.

**Solution**: Mock `soundfile.read` directly instead of trying to create valid WAV bytes in tests.
