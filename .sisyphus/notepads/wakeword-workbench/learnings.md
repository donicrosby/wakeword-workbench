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

## Task F1: Plan Compliance Audit

### Key Findings
- Plan checkbox completion does not reflect implementation coverage; many unchecked tasks have code and tests.
- Critical compliance mismatches are mostly interface/contract drift rather than missing modules.
- Export contract drift is the highest risk area (`export/validator.py` expectations do not match exporter outputs).
- Evaluation metric wiring has a correctness defect: FRR argument order is reversed in threshold/report paths.

### Audit Patterns
- Verify both "file exists" and "API semantics match plan"; existence-only checks overestimate compliance.
- Cross-check planned filenames versus evolved module layout to identify intentional refactors vs missing artifacts.
- Include explicit counts for compliant/partial/non-compliant tasks to avoid ambiguous verdicts.

## Task F4: Scope Fidelity Check

### Key Findings
- Guardrail scope remained intact: no GUI, cloud deployment, realtime inference service, distributed processing, model zoo, or microphone capture implementation was detected.
- Most risk was delivery maturity, not scope creep: several Definition of Done criteria remain unverified or unmet.
- QA evidence policy should be tracked separately from feature scope; missing per-task evidence can coexist with in-scope implementation.

### Audit Pattern
- Separate three dimensions explicitly in final report: (1) scope boundaries, (2) must-have presence, (3) Definition of Done readiness.
