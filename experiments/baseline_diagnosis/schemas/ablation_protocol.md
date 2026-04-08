# Ablation Protocol for Squash/Batch/Delta Mechanisms

This document defines the standard protocol for running mechanism ablation
experiments. Every experiment across every dwarf follows the same structure.

## Experiment Matrix

For each dwarf, we run 5 experiments:

| ID | Name | Squash | Batch | Delta |
|----|------|--------|-------|-------|
| E0 | baseline | ❌ | ❌ | ❌ |
| E1 | squash | ✅ | ❌ | ❌ |
| E2 | batch | ❌ | ✅ | ❌ |
| E3 | delta | ❌ | ❌ | ✅ |
| E4 | full | ✅ | ✅ | ✅ |

## Execution Mode

- E0 (baseline) and E4 (full): manual diagnosis in conversation with Claude
- E1, E2, E3 (single-mechanism): automated via `/superpowers:diagnose-workload` skill in batch mode

## Input Files

Per dwarf `<D>`:
- `<D>_full.json`: base features (existing feature extraction)
- `<D>_squash.json`: Squash mechanism output
- `<D>_batch.json`: Batch mechanism output
- `<D>_delta.json`: Delta mechanism output

## Output Files

Per experiment `<E>` on dwarf `<D>`:
- `<D>_ablation/E<N>_<name>.md`: prescription report

Per dwarf:
- `<D>_ablation/_summary.md`: cross-experiment summary

Across all dwarfs:
- `ablation/_cross_dwarf_summary.md`: the final Phase 5 deliverable

## Evaluation Criteria

Each prescription report is evaluated by human on four dimensions:

1. **Correctness**: does the diagnosis match known ground truth (for backprop, the v2 report)?
2. **Non-triviality**: does it go beyond restating raw metrics?
3. **Actionability**: does each prescription have { change, reason, expected, verify, confidence }?
4. **New findings vs baseline**: does adding a mechanism surface prescriptions not in E0?

## Simulator Closed-Loop Validation

For each dwarf, at least one high-confidence prescription from E4 is tested
on the simulator:

1. Modify `gpgpusim.config` according to prescription
2. Rerun simulator with same trace
3. Compare stats with baseline
4. Record whether direction matches prediction (success criterion)

## Forgetting Protocol (for skill batch mode)

When running multiple experiments via skill, the skill must:
1. Load only the current experiment's feature files
2. Produce the diagnosis report
3. Write to disk immediately
4. Clear working memory of the previous report content
5. Keep only a one-line completion marker for each completed experiment
