---
name: diagnose-workload
description: Run a prescriptive GPU architecture diagnosis on a workload given feature JSONs. Supports single mode (one experiment) or batch mode (multi-experiment plan).
argument-hint: "[--features <path>] [--enable-mechanism none|squash|batch|delta|all] [--output <path>] OR [--plan <plan.md>]"
---

# Diagnose Workload

This skill runs a deterministic prescriptive diagnosis protocol on GPU
workload feature data. It produces a prescription report following the
standard diagnosis template.

## Modes

### Single Mode

```
/superpowers:diagnose-workload --features <path> --enable-mechanism <name> --output <path>
```

Runs one diagnosis for one experiment configuration.

### Batch Mode

```
/superpowers:diagnose-workload --plan <plan.md>
```

Runs multiple diagnoses as defined in plan.md. See `schemas/ablation_protocol.md`
for plan format.

## Protocol (single mode)

When called in single mode, follow these steps exactly:

1. **Parse arguments:** extract `--features`, `--enable-mechanism`, `--output`.

2. **Load feature data:**
   - Read the file at `--features`
   - If `--enable-mechanism` is not `none`, find and read the corresponding
     mechanism file in the same directory
     (e.g. `backprop_squash.json` next to `backprop_full.json`)
   - If `--enable-mechanism all`, read all three mechanism files

3. **Load the prompt template** from
   `~/.claude/skills/diagnose-workload/diagnosis_prompt_template.md`

4. **Execute the diagnosis protocol** as specified in the prompt template:
   - Stage A: software utilization check (waves_per_sm, occupancy)
   - Stage B: architecture bottleneck analysis (distance-to-roof,
     cross-source reasoning, mechanism-informed insights)
   - Prescription generation (each with change/reason/expected/verify/confidence)

5. **Write output** to the `--output` path using the format from
   `experiments/baseline_diagnosis/schemas/diagnosis_template.md`

6. **Respond in chat with only a one-line summary**:
   `Diagnosis complete. N prescriptions written to <output>. Highest confidence: <level>.`
   **Do not** repeat the full report content in chat.

## Protocol (batch mode)

When called in batch mode, follow these steps:

1. **Parse plan.md** to extract:
   - List of dwarfs with their feature file paths
   - List of experiments (E0-E4 or similar)
   - Output directory

2. **For each (dwarf, experiment) pair, in sequence:**

   a. Load only the current experiment's feature files (do not prefetch all)

   b. Execute the single-mode protocol for this experiment

   c. Write the output to `<output_dir>/<dwarf>/<experiment_id>.md`

   d. Respond in chat with only `<dwarf>/<experiment_id> done`

   e. **Discard the full diagnosis content from working memory.** Keep only
      a metadata line (completion marker) to track progress across
      iterations.

3. **After all experiments complete:**

   - Re-read each output file from disk (not from memory)
   - Extract key findings (prescription count, confidence distribution,
     new findings unique to each experiment)
   - Write `<output_dir>/_summary.md` with the cross-experiment comparison

4. **Final chat response:** `Batch complete. N experiments finished. Summary at <summary path>.`

## Forgetting Protocol Details

The critical constraint is context management. Specifically:

- After writing a diagnosis to disk, do not include the diagnosis content
  in any subsequent reasoning step
- Only keep: (a) the plan, (b) a list of completed experiment IDs, (c) the
  current experiment's features
- Before starting the next experiment, explicitly move to a fresh reasoning
  state with only the required inputs

## Output Format

All diagnoses MUST use `experiments/baseline_diagnosis/schemas/diagnosis_template.md`
as the template. Replace `{placeholders}` with actual values.

## Human Review Gate

Skill diagnoses are NOT automatically promoted to closed-loop validation.
The user must manually review each diagnosis report before modifying
`gpgpusim.config`. This protects against hallucinated prescriptions.
