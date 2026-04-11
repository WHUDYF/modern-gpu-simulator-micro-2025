# Baseline Diagnosis Evaluation

**Workload**: _______________
**Date**: _______________

## Findings Evaluation

| # | AI Finding Summary | Category | Notes |
|---|-------------------|----------|-------|
| 1 |                   |          |       |
| 2 |                   |          |       |
| 3 |                   |          |       |
| 4 |                   |          |       |
| 5 |                   |          |       |

**Category definitions**:
- `correct-nontrivial`: Finding is accurate and reveals a non-obvious insight that requires cross-feature reasoning or architectural knowledge.
- `correct-trivial`: Finding is accurate but obvious from a single feature (e.g., "high L1 miss rate means poor cache behavior").
- `wrong`: Finding is factually incorrect or the causal reasoning is flawed.
- `blind-spot`: A known issue that the AI failed to identify (record in the table below).

## Blind Spots

| # | Known Issue | Why AI Missed It | What Feature Would Help |
|---|-------------|------------------|------------------------|
| 1 |             |                  |                        |
| 2 |             |                  |                        |
| 3 |             |                  |                        |

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total AI findings | |
| Correct-nontrivial | |
| Correct-trivial | |
| Wrong | |
| Blind spots | |
| **Diagnostic value score (1-5)** | |

**Scoring guide**:
- **5**: Multiple nontrivial correct findings, no wrong findings, no critical blind spots.
- **4**: At least one nontrivial correct finding, at most one wrong finding, no critical blind spots.
- **3**: Mix of trivial and nontrivial findings, minor errors present, one blind spot.
- **2**: Mostly trivial findings, multiple errors or blind spots.
- **1**: Predominantly wrong or irrelevant findings.

## Conclusion

- [ ] Proceed to next exploration step
- [ ] Revise feature set and re-run diagnosis
- [ ] Add more profiling data before proceeding

**Rationale**: _______________
