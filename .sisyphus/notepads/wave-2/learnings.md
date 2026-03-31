# Wave 2 Learnings

## Task 7: TTSBackend ABC Implementation

### Successful Patterns

1. **ABC Pattern**: Used `abc.ABC` and `@abstractmethod` to create a proper abstract base class
2. **Factory Registry**: Used a class-level `_backend_registry` dict on `TTSBackend` itself (not instance-level) for the factory pattern
3. **Dataclass Validation**: Used `@dataclass` with `__post_init__` for validating TTSResult fields
4. **Exception Hierarchy**: `BackendNotAvailableError` extends `TTSError` extends `Exception`

### Key Implementation Details

- `TTSBackend._backend_registry` is a `ClassVar` so it's shared across all subclasses
- `register_backend()` and `unregister_backend()` class methods for factory registration
- `is_available()` class method defaults to `True` but can be overridden
- `create_backend()` checks both registration AND availability

### Testing Approach

- Tested ABC cannot be instantiated directly (TypeError)
- Tested incomplete subclasses cannot be instantiated
- Tested factory raises appropriate errors for unknown/unavailable backends
- Tested exception inheritance hierarchy

### Validation Gotchas

- `np.random.randn()` can produce values > 1, use `np.random.uniform()` instead for bounded random samples
- Duration validation checks that `duration ≈ len(audio)/sample_rate` with tolerance
