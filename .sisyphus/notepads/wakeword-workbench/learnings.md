# Wake Word Workbench - Learnings

## Task 28: FAR Calculator Implementation

### Key Insights
- **FAR = False Positives / audio_duration_hours**
- Cooldown logic: After detecting a frame above threshold, skip `cooldown_frames` to prevent double-counting the same trigger
- Default cooldown of 75 frames = 1.5 seconds at 20ms per frame
- Consecutive frames above threshold count as ONE false positive (grouping)
- Ground truth is ignored for FAR (unlike FRR) - only predictions matter

### Implementation Notes
- `calculate_far()` - main function returning FAR value
- `count_false_positives()` - helper returning raw count (useful for testing/debugging)
- Empty predictions array returns FAR of 0.0
- Zero/negative audio_duration_hours raises ValueError

### Testing Approach
- Tests use smaller cooldown values (2-3) than default (75) for short arrays
- Tests verify: empty arrays, all below threshold, consecutive frames, cooldown skip logic, audio duration scaling

### Files Created
- `src/wakeword_workbench/eval/far.py` - FAR calculator implementation
- `tests/test_far.py` - 27 comprehensive tests
