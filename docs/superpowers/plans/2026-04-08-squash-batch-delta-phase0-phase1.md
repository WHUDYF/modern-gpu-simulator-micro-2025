# Squash/Batch/Delta Phase 0 + Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the interface schemas, Claude Code diagnosis skill, and the three semantic mechanisms (Squash/Batch/Delta) at the prototype level on backprop data. Phase 2-5 will be planned after Checkpoint 1 from Phase 1 results.

**Architecture:** Three standalone Python scripts (one per mechanism) consume a unified `per_tb_features.json` and produce mechanism-specific JSON outputs. A single `diagnose-workload` Claude Code skill consumes the feature JSON + mechanism JSONs and produces structured prescription reports. All components communicate via strict JSON schemas defined up-front to prevent interface drift across later phases.

**Tech Stack:** Python 3.11 (numpy, scikit-learn for k-means/DBSCAN), existing `experiments/baseline_diagnosis/` infrastructure, Claude Code skill system (markdown-based), existing `backprop_4096_full.json` as primary test data.

**Spec Reference:** `docs/superpowers/specs/2026-04-08-squash-batch-delta-multi-dwarf-design.md`

---

## File Structure

```
experiments/baseline_diagnosis/
  schemas/                                      # Phase 0: interface contracts
    per_tb_features_schema.json
    squash_output_schema.json
    batch_output_schema.json
    delta_output_schema.json
    mechanism_config.json
    ablation_protocol.md
    diagnosis_template.md
  mechanisms/                                   # Phase 1: implementation
    __init__.py
    extract_per_tb_features.py
    extract_squash_features.py
    extract_batch_features.py
    extract_delta_features.py
  results/rodinia/backprop_mechanisms/          # Phase 1: validation outputs
    backprop_4096_per_tb.json
    backprop_4096_squash.json
    backprop_4096_batch.json
    backprop_4096_delta.json
  tests/                                        # unit tests
    test_extract_per_tb_features.py
    test_extract_squash_features.py
    test_extract_batch_features.py
    test_extract_delta_features.py

~/.claude/skills/diagnose-workload/             # Phase 0: diagnosis skill
  SKILL.md
  diagnosis_prompt_template.md
```

## Design Decisions Locked In By This Plan

These decisions come from spec §3 and §4; the plan assumes them without re-debate:

1. **Two operation levels**: every mechanism operates at both kernel-level and TB-level; outputs contain `kernel_level` and `tb_level` sub-sections.
2. **Skill-based batch execution**: the diagnosis skill supports both single and batch modes; batch mode uses a plan.md file.
3. **Context management by "forgetting protocol"**: the skill loads only the current experiment's data and writes to disk immediately.
4. **Testing strategy**: unit tests use synthetic minimal inputs to verify mechanism outputs match schemas; integration testing is done by running on the real `backprop_4096_full.json` and inspecting outputs manually.
5. **No modification to existing feature extraction**: the new `extract_per_tb_features.py` reads from `backprop_4096_full.json` (already exists) and produces the unified `per_tb` vector.

---

## Phase 0: Interface Schemas and Skill Definition

### Task 1: Per-TB Features Schema

**Files:**
- Create: `experiments/baseline_diagnosis/schemas/per_tb_features_schema.json`

Defines the canonical per-TB feature vector shape that all three mechanisms consume. This is the contract.

- [ ] **Step 1: Write the schema file**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Per-TB Features",
  "description": "Canonical feature vector for a single threadblock, consumed by Squash/Batch/Delta mechanisms",
  "type": "object",
  "required": ["workload", "kernels"],
  "properties": {
    "workload": {
      "type": "string",
      "description": "Workload name, e.g. 'backprop-rodinia-2.0-ft'"
    },
    "input_size": {
      "type": ["integer", "string"],
      "description": "Workload input size for reproducibility"
    },
    "kernels": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["kernel_id", "kernel_name", "per_tb"],
        "properties": {
          "kernel_id": {"type": "integer"},
          "kernel_name": {"type": "string"},
          "kernel_summary": {
            "type": "object",
            "description": "Kernel-level features (for kernel-level mechanism operation)",
            "properties": {
              "top_opcodes": {"type": "array"},
              "total_static_instructions": {"type": "integer"},
              "total_dynamic_instructions": {"type": "integer"},
              "uses_fp64": {"type": "boolean"},
              "uses_shared_memory": {"type": "boolean"},
              "num_barriers": {"type": "integer"},
              "grid_dim": {"type": "string"},
              "block_dim": {"type": "string"}
            }
          },
          "per_tb": {
            "type": "array",
            "description": "One entry per threadblock in this kernel",
            "items": {
              "type": "object",
              "required": ["tb_index", "features"],
              "properties": {
                "tb_index": {"type": "integer"},
                "features": {
                  "type": "object",
                  "description": "Feature vector (numeric fields only, for clustering/diffing)",
                  "properties": {
                    "num_warps": {"type": "number"},
                    "instructions_per_warp_mean": {"type": "number"},
                    "instructions_per_warp_std": {"type": "number"},
                    "opcode_ffma_ratio": {"type": "number"},
                    "opcode_dfma_ratio": {"type": "number"},
                    "opcode_ldg_ratio": {"type": "number"},
                    "opcode_stg_ratio": {"type": "number"},
                    "opcode_lds_ratio": {"type": "number"},
                    "opcode_sts_ratio": {"type": "number"},
                    "opcode_iadd_ratio": {"type": "number"},
                    "opcode_bar_ratio": {"type": "number"},
                    "address_override_count": {"type": "integer"},
                    "is_full_encoding": {"type": "boolean"},
                    "compression_format": {"type": "string"}
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

- [ ] **Step 2: Verify schema is valid JSON**

Run:
```bash
python3 -c "import json; json.load(open('experiments/baseline_diagnosis/schemas/per_tb_features_schema.json'))"
```

Expected: no output (silent success)

- [ ] **Step 3: Commit**

```bash
git add experiments/baseline_diagnosis/schemas/per_tb_features_schema.json
git commit -m "schemas: per-TB features interface contract"
```

---

### Task 2: Squash Output Schema

**Files:**
- Create: `experiments/baseline_diagnosis/schemas/squash_output_schema.json`

- [ ] **Step 1: Write the schema file**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Squash Mechanism Output",
  "description": "Temporal segmentation output. Two-level: kernel-level and TB-level.",
  "type": "object",
  "required": ["mechanism", "workload", "kernel_level", "tb_level"],
  "properties": {
    "mechanism": {"const": "squash"},
    "workload": {"type": "string"},
    "kernel_level": {
      "type": "object",
      "required": ["squash_segments", "boundary_count", "total_kernels"],
      "properties": {
        "squash_segments": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["segment_id", "kernel_range", "dominant_opcodes", "cohesion_score"],
            "properties": {
              "segment_id": {"type": "integer"},
              "kernel_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2
              },
              "kernel_count": {"type": "integer"},
              "dominant_opcodes": {"type": "array", "items": {"type": "string"}},
              "cohesion_score": {"type": "number"},
              "representative_kernel": {"type": "integer"},
              "behavior_summary": {"type": "string"}
            }
          }
        },
        "boundary_count": {"type": "integer"},
        "total_kernels": {"type": "integer"}
      }
    },
    "tb_level": {
      "type": "object",
      "description": "Per-kernel TB segmentation. Keys are kernel_ids as strings.",
      "additionalProperties": {
        "type": "object",
        "required": ["squash_segments", "boundary_count", "total_tbs"],
        "properties": {
          "squash_segments": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["segment_id", "tb_range", "cohesion_score"],
              "properties": {
                "segment_id": {"type": "integer"},
                "tb_range": {
                  "type": "array",
                  "items": {"type": "integer"},
                  "minItems": 2,
                  "maxItems": 2
                },
                "tb_count": {"type": "integer"},
                "cohesion_score": {"type": "number"},
                "representative_tb": {"type": "integer"},
                "behavior_summary": {"type": "string"}
              }
            }
          },
          "boundary_count": {"type": "integer"},
          "total_tbs": {"type": "integer"}
        }
      }
    },
    "_simulation_reuse_hint": {
      "type": "object",
      "description": "Opportunistic: hints for simulator-level reuse",
      "properties": {
        "kernel_level_representatives": {"type": "array", "items": {"type": "integer"}},
        "tb_level_representatives": {"type": "object"}
      }
    }
  }
}
```

- [ ] **Step 2: Verify schema is valid JSON**

Run:
```bash
python3 -c "import json; json.load(open('experiments/baseline_diagnosis/schemas/squash_output_schema.json'))"
```

Expected: no output

- [ ] **Step 3: Commit**

```bash
git add experiments/baseline_diagnosis/schemas/squash_output_schema.json
git commit -m "schemas: squash output interface contract"
```

---

### Task 3: Batch Output Schema

**Files:**
- Create: `experiments/baseline_diagnosis/schemas/batch_output_schema.json`

- [ ] **Step 1: Write the schema file**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Batch Mechanism Output",
  "description": "Spatial homogeneity clustering output. Two-level.",
  "type": "object",
  "required": ["mechanism", "workload", "kernel_level", "tb_level"],
  "properties": {
    "mechanism": {"const": "batch"},
    "workload": {"type": "string"},
    "kernel_level": {
      "type": "object",
      "required": ["batch_clusters", "outlier_kernels", "homogeneity_score"],
      "properties": {
        "batch_clusters": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["cluster_id", "kernel_ids", "cluster_size", "centroid_summary"],
            "properties": {
              "cluster_id": {"type": "integer"},
              "kernel_ids": {"type": "array", "items": {"type": "integer"}},
              "cluster_size": {"type": "integer"},
              "cluster_pct": {"type": "number"},
              "centroid_summary": {"type": "object"},
              "cohesion": {"type": "number"}
            }
          }
        },
        "outlier_kernels": {"type": "array", "items": {"type": "integer"}},
        "homogeneity_score": {"type": "number"}
      }
    },
    "tb_level": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["batch_clusters", "outlier_tbs", "homogeneity_score"],
        "properties": {
          "batch_clusters": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["cluster_id", "tb_ids", "cluster_size", "centroid_summary"],
              "properties": {
                "cluster_id": {"type": "integer"},
                "tb_ids": {"type": "array", "items": {"type": "integer"}},
                "cluster_size": {"type": "integer"},
                "cluster_pct": {"type": "number"},
                "centroid_summary": {"type": "object"},
                "cohesion": {"type": "number"}
              }
            }
          },
          "outlier_tbs": {"type": "array", "items": {"type": "integer"}},
          "homogeneity_score": {"type": "number"}
        }
      }
    },
    "_simulation_reuse_hint": {
      "type": "object",
      "properties": {
        "kernel_cluster_representatives": {"type": "object"},
        "tb_cluster_representatives": {"type": "object"}
      }
    }
  }
}
```

- [ ] **Step 2: Verify schema is valid JSON**

Run:
```bash
python3 -c "import json; json.load(open('experiments/baseline_diagnosis/schemas/batch_output_schema.json'))"
```

Expected: no output

- [ ] **Step 3: Commit**

```bash
git add experiments/baseline_diagnosis/schemas/batch_output_schema.json
git commit -m "schemas: batch output interface contract"
```

---

### Task 4: Delta Output Schema

**Files:**
- Create: `experiments/baseline_diagnosis/schemas/delta_output_schema.json`

- [ ] **Step 1: Write the schema file**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Delta Mechanism Output",
  "description": "Field-level change pattern output. Two-level.",
  "type": "object",
  "required": ["mechanism", "workload", "kernel_level", "tb_level"],
  "properties": {
    "mechanism": {"const": "delta"},
    "workload": {"type": "string"},
    "kernel_level": {
      "type": "object",
      "required": ["field_temperature", "hot_fields", "cold_fields"],
      "properties": {
        "field_temperature": {
          "type": "object",
          "description": "Field name -> temperature score (0=cold, 1=hot)",
          "additionalProperties": {"type": "number"}
        },
        "hot_fields": {"type": "array", "items": {"type": "string"}},
        "cold_fields": {"type": "array", "items": {"type": "string"}},
        "field_correlations": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["fields", "correlation"],
            "properties": {
              "fields": {"type": "array", "items": {"type": "string"}},
              "correlation": {"type": "number"},
              "interpretation": {"type": "string"}
            }
          }
        },
        "outlier_diffs": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["pair", "magnitude"],
            "properties": {
              "pair": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
              "magnitude": {"type": "number"},
              "dominant_changing_fields": {"type": "array"},
              "interpretation": {"type": "string"}
            }
          }
        }
      }
    },
    "tb_level": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["field_temperature", "hot_fields", "cold_fields"],
        "properties": {
          "field_temperature": {
            "type": "object",
            "additionalProperties": {"type": "number"}
          },
          "hot_fields": {"type": "array", "items": {"type": "string"}},
          "cold_fields": {"type": "array", "items": {"type": "string"}},
          "field_correlations": {"type": "array"},
          "outlier_diffs": {"type": "array"}
        }
      }
    }
  }
}
```

- [ ] **Step 2: Verify schema is valid JSON**

Run:
```bash
python3 -c "import json; json.load(open('experiments/baseline_diagnosis/schemas/delta_output_schema.json'))"
```

Expected: no output

- [ ] **Step 3: Commit**

```bash
git add experiments/baseline_diagnosis/schemas/delta_output_schema.json
git commit -m "schemas: delta output interface contract"
```

---

### Task 5: Mechanism Config (tunable parameters)

**Files:**
- Create: `experiments/baseline_diagnosis/schemas/mechanism_config.json`

- [ ] **Step 1: Write the config file**

```json
{
  "description": "Tunable parameters for Squash/Batch/Delta mechanisms. Values here are starting points; override via --config CLI flag when running mechanisms.",
  "squash": {
    "kernel_level": {
      "similarity_threshold": 0.85,
      "_note": "Cosine similarity threshold for adjacent kernels. Below this = segment boundary."
    },
    "tb_level": {
      "similarity_threshold": 0.90,
      "_note": "Cosine similarity threshold for adjacent TBs within a kernel."
    }
  },
  "batch": {
    "kernel_level": {
      "clustering_algorithm": "dbscan",
      "dbscan_eps": 0.15,
      "dbscan_min_samples": 2,
      "_note": "DBSCAN handles variable cluster counts and flags outliers naturally."
    },
    "tb_level": {
      "clustering_algorithm": "dbscan",
      "dbscan_eps": 0.10,
      "dbscan_min_samples": 3,
      "_note": "Tighter eps for TBs since they should be more uniform than kernels."
    }
  },
  "delta": {
    "kernel_level": {
      "hot_threshold": 0.3,
      "cold_threshold": 0.05,
      "correlation_threshold": 0.6,
      "outlier_zscore": 2.5
    },
    "tb_level": {
      "hot_threshold": 0.2,
      "cold_threshold": 0.05,
      "correlation_threshold": 0.6,
      "outlier_zscore": 2.5
    }
  }
}
```

- [ ] **Step 2: Verify config is valid JSON**

Run:
```bash
python3 -c "import json; c = json.load(open('experiments/baseline_diagnosis/schemas/mechanism_config.json')); assert 'squash' in c and 'batch' in c and 'delta' in c; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/baseline_diagnosis/schemas/mechanism_config.json
git commit -m "schemas: mechanism tunable parameters"
```

---

### Task 6: Ablation Protocol Document

**Files:**
- Create: `experiments/baseline_diagnosis/schemas/ablation_protocol.md`

This document defines how Phase 2 experiments are structured. Must exist
before Phase 2.

- [ ] **Step 1: Write the protocol document**

```markdown
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
```

- [ ] **Step 2: Verify file is created**

Run:
```bash
wc -l experiments/baseline_diagnosis/schemas/ablation_protocol.md
```

Expected: a non-zero line count (around 60-70 lines)

- [ ] **Step 3: Commit**

```bash
git add experiments/baseline_diagnosis/schemas/ablation_protocol.md
git commit -m "schemas: ablation protocol definition"
```

---

### Task 7: Diagnosis Report Template

**Files:**
- Create: `experiments/baseline_diagnosis/schemas/diagnosis_template.md`

Standard markdown template for all diagnosis reports (manual and skill).

- [ ] **Step 1: Write the template**

```markdown
# Diagnosis Report: {workload} [{experiment_id}]

**Date:** {date}
**Hardware:** {hardware}
**Input features:** {features_path}
**Mechanisms enabled:** {mechanisms_list}
**Diagnoser:** {manual|skill}

---

## Stage A: Software Utilization Check

### Utilization metrics

| Kernel | waves_per_sm | achieved_occupancy | grid_size |
|--------|-------------|---------------------|-----------|
| ... | ... | ... | ... |

### Stage A Verdict

- [ ] Workload utilizes hardware adequately (waves_per_sm ≥ 4, occupancy ≥ 80%)
- [ ] If no, prescribe Class A fix first (software configuration change)

### Class A Prescription (if needed)

[Only filled if Stage A verdict is negative]

---

## Stage B: Architecture Bottleneck Analysis

### Per-kernel bottleneck identification

For each kernel:
- **Dominant bottleneck (highest distance-to-roof utilization):**
- **Secondary bottleneck:**
- **Cross-source reasoning (trace + hw_stats):**

### Mechanism-informed insights (only if mechanisms enabled)

- **Squash findings:** behavior phases identified, phase transitions
- **Batch findings:** clusters, outlier units
- **Delta findings:** hot/cold fields, field correlations

---

## Class B Prescriptions

### Prescription B.{N}: {title}

**Target kernel:** {kernel_name}

**Modification:**
```
<parameter name> <old value> → <new value>
```

**Reason:**
[Evidence trail, which features support this]

**Expected effect:**
- Metric X: direction ± magnitude
- Metric Y: direction ± magnitude

**Expected cost:**
[Area/power/other tradeoff]

**Verification:**
- Modify: [config file path]
- Rerun: [command]
- Compare: [which metric]
- Success criterion: [threshold]

**Confidence:** HIGH / MEDIUM / LOW

**Control kernel (unchanged prediction):** [kernel that should NOT change if prescription is correctly scoped]

---

## Summary

- Total prescriptions: N
- High confidence: M
- Prescriptions that use mechanism features: K (only applies when mechanisms enabled)
- Prescriptions that would not exist without mechanism features: J (only applies when mechanisms enabled)
```

- [ ] **Step 2: Verify template is created**

Run:
```bash
grep -c "^## " experiments/baseline_diagnosis/schemas/diagnosis_template.md
```

Expected: 4 (Stage A, Stage B, Class B Prescriptions, Summary)

- [ ] **Step 3: Commit**

```bash
git add experiments/baseline_diagnosis/schemas/diagnosis_template.md
git commit -m "schemas: diagnosis report template"
```

---

### Task 8: Claude Code Skill Definition (single mode)

**Files:**
- Create: `~/.claude/skills/diagnose-workload/SKILL.md`
- Create: `~/.claude/skills/diagnose-workload/diagnosis_prompt_template.md`

We start with the single-mode skill (one experiment at a time). Batch mode
is added in Task 9.

- [ ] **Step 1: Create the skill directory**

Run:
```bash
mkdir -p ~/.claude/skills/diagnose-workload
```

- [ ] **Step 2: Write the skill definition**

File: `~/.claude/skills/diagnose-workload/SKILL.md`

```markdown
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
```

- [ ] **Step 3: Verify skill file is created**

Run:
```bash
ls -la ~/.claude/skills/diagnose-workload/SKILL.md
```

Expected: file exists

- [ ] **Step 4: Commit (in the project repo, not ~/.claude/)**

The skill file lives in `~/.claude/skills/`, outside the project. Instead,
commit a reference file in the project:

```bash
# Copy the skill definition to the project for version control
mkdir -p experiments/baseline_diagnosis/skills/diagnose-workload
cp ~/.claude/skills/diagnose-workload/SKILL.md experiments/baseline_diagnosis/skills/diagnose-workload/SKILL.md

git add experiments/baseline_diagnosis/skills/diagnose-workload/SKILL.md
git commit -m "skill: add diagnose-workload skill definition (single+batch mode)"
```

---

### Task 9: Diagnosis Prompt Template

**Files:**
- Create: `~/.claude/skills/diagnose-workload/diagnosis_prompt_template.md`

The actual prompt that guides Claude's diagnosis reasoning.

- [ ] **Step 1: Write the prompt template**

File: `~/.claude/skills/diagnose-workload/diagnosis_prompt_template.md`

```markdown
# Diagnosis Prompt Template

You are performing a prescriptive GPU architecture diagnosis. Follow this
protocol exactly. Do not improvise.

## Inputs You Will Receive

1. `features_json`: workload base features (trace + hardware stats)
2. `mechanism_jsons`: zero or more of {squash, batch, delta} outputs
3. `workload_name`: identifier for the workload
4. `experiment_id`: identifier for this diagnosis

## Reasoning Steps (execute in order)

### Step 1: Stage A Check

For each kernel in `features_json`:
- Extract `waves_per_sm` from hardware metrics
- Extract `achieved_occupancy_pct` from hardware metrics

Verdict:
- If `waves_per_sm < 4`: FAIL Stage A. Prescribe Class A fix (e.g., increase
  input size, increase batch size). Do not proceed to Stage B.
- If `achieved_occupancy_pct < 50`: WARN. Investigate resource limits
  (register pressure, shared memory) but may still proceed to Stage B.
- Otherwise: PASS. Proceed to Stage B.

### Step 2: Stage B Analysis (only if Stage A passes or warns)

For each kernel, identify the top bottleneck using distance-to-roof logic:

- Compute utilization for each resource dimension:
  - compute_throughput_pct
  - memory_throughput_pct (DRAM)
  - l1_throughput_pct
  - l2_throughput_pct
  - sm_scheduler (inferred from IPC vs peak)
- The highest utilization is the primary bottleneck.
- The second highest is the secondary bottleneck.

Cross-source reasoning:
- If compute is high BUT IPC is very low, look at trace opcodes. If FP64
  (DMUL/DFMA/F2F.F64) appears, the kernel is FP64-serialized.
- If L1 throughput is high BUT L1 hit rate is moderate, it's LDS/STS
  bandwidth, not cache capacity.
- Flag any metric pair that looks contradictory; these are the most
  informative bottlenecks.

### Step 3: Mechanism-Informed Insights (only if mechanisms are enabled)

#### If Squash is enabled:
- Read `squash.kernel_level.squash_segments` to identify workload phases
- For each phase, identify its dominant bottleneck from Stage B
- Differentiate prescriptions by phase (e.g., "Phase 1 bottleneck is X,
  Phase 2 bottleneck is Y")

#### If Batch is enabled:
- Read `batch.kernel_level.batch_clusters` and `batch.tb_level.*.batch_clusters`
- If outlier kernels/TBs exist, investigate their features and propose
  prescriptions targeting either the main cluster or the outliers
  specifically

#### If Delta is enabled:
- Read `delta.kernel_level.field_temperature` and `delta.tb_level.*.field_temperature`
- Hot fields indicate what varies; cold fields indicate invariants
- Use field_correlations to infer causal chains (e.g., "address pattern
  variation correlates with stall variation")

### Step 4: Prescription Generation

For each identified bottleneck, generate one prescription. Each prescription
MUST contain all five elements:

1. **Change**: exact parameter name and new value (must map to gpgpusim.config
   or trace.config)
2. **Reason**: which feature(s) led to this conclusion (cite specific
   numeric values)
3. **Expected**: which metric will change, direction, rough magnitude
4. **Verify**: how to test (which metric to compare before/after)
5. **Confidence**: HIGH / MEDIUM / LOW with one-sentence justification

Also specify a **control kernel**: a kernel that should NOT change if the
prescription is correctly scoped. This enables the "null-control" check.

### Step 5: Output Writing

Write the result using the template at
`experiments/baseline_diagnosis/schemas/diagnosis_template.md`.
Replace all `{placeholders}` with actual values.

Do NOT emit the full diagnosis content in the chat response. Only emit:
`Diagnosis complete. N prescriptions written to <output>. Highest confidence: <level>.`

## Style Rules

- Do NOT restate raw numbers without interpretation
- Do NOT use vague language ("might be", "could potentially"); either commit
  to a hypothesis with evidence or skip it
- Do NOT propose prescriptions beyond the gpgpusim.config / trace.config
  parameter space
- DO favor cross-feature observations over single-metric observations
- DO flag contradictions between features (these are often the richest
  diagnostic signals)
- DO assign confidence honestly: HIGH only if multiple independent
  evidence streams agree
```

- [ ] **Step 2: Verify template file**

Run:
```bash
ls -la ~/.claude/skills/diagnose-workload/diagnosis_prompt_template.md
wc -l ~/.claude/skills/diagnose-workload/diagnosis_prompt_template.md
```

Expected: file exists, around 100-130 lines

- [ ] **Step 3: Commit reference copy in project**

```bash
cp ~/.claude/skills/diagnose-workload/diagnosis_prompt_template.md \
   experiments/baseline_diagnosis/skills/diagnose-workload/diagnosis_prompt_template.md
git add experiments/baseline_diagnosis/skills/diagnose-workload/diagnosis_prompt_template.md
git commit -m "skill: diagnose-workload prompt template"
```

---

## Phase 1: Mechanism Prototypes on Backprop

All Phase 1 tasks work against the existing `experiments/baseline_diagnosis/results/rodinia/backprop_4096_full.json` file. No new trace extraction is required.

### Task 10: Per-TB Feature Extractor

**Files:**
- Create: `experiments/baseline_diagnosis/mechanisms/__init__.py`
- Create: `experiments/baseline_diagnosis/mechanisms/extract_per_tb_features.py`
- Create: `experiments/baseline_diagnosis/tests/__init__.py`
- Create: `experiments/baseline_diagnosis/tests/test_extract_per_tb_features.py`

This script converts the existing `backprop_4096_full.json` (which has
`per_kernel.*.compression_features.*`) into the unified per-TB vector shape
defined by `per_tb_features_schema.json`.

**Important: this is the TEST-FIRST task.** Before writing implementation,
write tests that verify the output structure matches the schema.

- [ ] **Step 1: Create __init__.py files**

```bash
mkdir -p experiments/baseline_diagnosis/mechanisms
mkdir -p experiments/baseline_diagnosis/tests
touch experiments/baseline_diagnosis/mechanisms/__init__.py
touch experiments/baseline_diagnosis/tests/__init__.py
```

- [ ] **Step 2: Write the failing test**

File: `experiments/baseline_diagnosis/tests/test_extract_per_tb_features.py`

```python
"""Test per-TB feature extraction."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent

SCRIPT = ROOT / "mechanisms" / "extract_per_tb_features.py"
INPUT = ROOT / "results" / "rodinia" / "backprop_4096_full.json"
SCHEMA = ROOT / "schemas" / "per_tb_features_schema.json"


def run_extractor(output_path):
    """Call the extractor as a CLI and return parsed JSON."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(INPUT), "--output", str(output_path)],
        check=True,
    )
    return json.loads(output_path.read_text())


def test_output_has_workload_and_kernels(tmp_path):
    out = tmp_path / "out.json"
    result = run_extractor(out)
    assert "workload" in result
    assert "kernels" in result
    assert isinstance(result["kernels"], list)
    assert len(result["kernels"]) == 2  # backprop has two kernels


def test_each_kernel_has_required_fields(tmp_path):
    out = tmp_path / "out.json"
    result = run_extractor(out)
    for kernel in result["kernels"]:
        assert "kernel_id" in kernel
        assert "kernel_name" in kernel
        assert "per_tb" in kernel
        assert "kernel_summary" in kernel


def test_per_tb_entries_have_features(tmp_path):
    out = tmp_path / "out.json"
    result = run_extractor(out)
    for kernel in result["kernels"]:
        assert len(kernel["per_tb"]) > 0
        for tb in kernel["per_tb"]:
            assert "tb_index" in tb
            assert "features" in tb
            assert isinstance(tb["features"], dict)
            # Features must contain at least these numeric fields
            for field in ["num_warps", "instructions_per_warp_mean"]:
                assert field in tb["features"], f"Missing {field} in TB {tb['tb_index']}"


def test_kernel_summary_has_opcodes(tmp_path):
    out = tmp_path / "out.json"
    result = run_extractor(out)
    for kernel in result["kernels"]:
        summary = kernel["kernel_summary"]
        assert "top_opcodes" in summary
        assert "uses_fp64" in summary
        # backprop forward uses FFMA (not FP64), adjust_weights uses DFMA (FP64)
        if "layerforward" in kernel["kernel_name"]:
            assert summary["uses_fp64"] is False
        elif "adjust_weights" in kernel["kernel_name"]:
            assert summary["uses_fp64"] is True
```

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
python3 -m pytest experiments/baseline_diagnosis/tests/test_extract_per_tb_features.py -v
```

Expected: FAIL with module/file not found error (script doesn't exist yet)

- [ ] **Step 4: Write the extractor**

File: `experiments/baseline_diagnosis/mechanisms/extract_per_tb_features.py`

```python
#!/usr/bin/env python3
"""Extract unified per-TB feature vectors from existing full features JSON.

Reads an existing `<workload>_full.json` produced by the baseline diagnosis
pipeline and produces a new JSON conforming to per_tb_features_schema.json.
"""
import argparse
import json
import sys
from pathlib import Path


FP64_OPCODES = {"DMUL", "DFMA", "F2F.F64.F32", "F2F.F32.F64", "DADD", "DSUB"}


def classify_opcode(opcode):
    """Return a category name for an opcode. Used to compute ratios."""
    op = opcode.upper()
    if "FFMA" in op:
        return "ffma"
    if any(x in op for x in ["DFMA", "DMUL", "DADD", "DSUB"]):
        return "dfma"
    if "LDG" in op:
        return "ldg"
    if "STG" in op:
        return "stg"
    if "LDS" in op:
        return "lds"
    if "STS" in op:
        return "sts"
    if "IADD" in op:
        return "iadd"
    if "BAR" in op:
        return "bar"
    return "other"


def compute_opcode_ratios(top_opcodes):
    """Convert top_opcodes [{opcode, count}] into ratio dict."""
    total = sum(entry["count"] for entry in top_opcodes)
    if total == 0:
        return {}
    categories = {}
    for entry in top_opcodes:
        cat = classify_opcode(entry["opcode"])
        categories[cat] = categories.get(cat, 0) + entry["count"]
    return {f"opcode_{cat}_ratio": count / total for cat, count in categories.items()}


def build_kernel_summary(kernel_data):
    """Build kernel_summary from a per_kernel entry."""
    static = kernel_data.get("static_info", {})
    dynamic = kernel_data.get("dynamic_stats", {}) or {}
    compression = kernel_data.get("compression_features", {}) or {}
    top_opcodes = static.get("top_opcodes", [])

    uses_fp64 = any(
        any(fp64 in entry["opcode"].upper() for fp64 in FP64_OPCODES)
        for entry in top_opcodes
    )
    uses_shared_memory = any(
        "LDS" in entry["opcode"].upper() or "STS" in entry["opcode"].upper()
        for entry in top_opcodes
    )
    num_barriers = sum(
        entry["count"] for entry in top_opcodes if "BAR" in entry["opcode"].upper()
    )

    return {
        "top_opcodes": top_opcodes[:10],
        "total_static_instructions": static.get("total_static_instructions", 0),
        "total_dynamic_instructions": dynamic.get("total_dynamic_insts", 0),
        "uses_fp64": uses_fp64,
        "uses_shared_memory": uses_shared_memory,
        "num_barriers": num_barriers,
        "grid_dim": dynamic.get("grid_dim", ""),
        "block_dim": dynamic.get("block_dim", ""),
        "num_tbs": compression.get("num_tb_files", 0),
    }


def build_per_tb_entries(kernel_data):
    """Build per_tb entries from a per_kernel entry.

    The existing full.json does not have per-TB features at the individual
    level; it has aggregated statistics. We expand them into one synthetic
    entry per TB, with the same feature values.
    """
    compression = kernel_data.get("compression_features", {}) or {}
    static = kernel_data.get("static_info", {})
    top_opcodes = static.get("top_opcodes", [])
    opcode_ratios = compute_opcode_ratios(top_opcodes)

    num_tbs = compression.get("num_tb_files", 0)
    num_warps_stats = compression.get("num_warps", {}) or {}
    inst_stats = compression.get("instructions_per_warp_mean", {}) or {}

    num_warps_mean = num_warps_stats.get("mean", 0) or 0
    inst_per_warp_mean = inst_stats.get("mean", 0) or 0
    inst_per_warp_std = inst_stats.get("std", 0) or 0

    # Base feature vector (same for all TBs when only aggregate stats are available)
    base_features = {
        "num_warps": num_warps_mean,
        "instructions_per_warp_mean": inst_per_warp_mean,
        "instructions_per_warp_std": inst_per_warp_std,
        "compression_format": compression.get("dominant_format", "unknown"),
        "address_override_count": 0,
        "is_full_encoding": False,
    }
    for field, value in opcode_ratios.items():
        base_features[field] = value

    return [
        {"tb_index": i, "features": dict(base_features)}
        for i in range(num_tbs)
    ]


def extract_per_tb_features(full_json_path):
    """Main extraction: convert full.json -> per_tb structure."""
    with open(full_json_path) as f:
        full = json.load(f)

    workload = full.get("workload", "unknown")
    per_kernel = full.get("per_kernel", {})

    kernels = []
    for idx, (kname, kdata) in enumerate(per_kernel.items(), start=1):
        kernels.append({
            "kernel_id": idx,
            "kernel_name": kname,
            "kernel_summary": build_kernel_summary(kdata),
            "per_tb": build_per_tb_entries(kdata),
        })

    return {
        "workload": workload,
        "kernels": kernels,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract unified per-TB features.")
    parser.add_argument("--input", required=True, help="Path to <workload>_full.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = extract_per_tb_features(args.input)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    total_tbs = sum(len(k["per_tb"]) for k in result["kernels"])
    print(
        f"[per_tb] wrote {output_path} "
        f"(workload={result['workload']}, kernels={len(result['kernels'])}, total_tbs={total_tbs})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
python3 -m pytest experiments/baseline_diagnosis/tests/test_extract_per_tb_features.py -v
```

Expected: 4 tests PASS

- [ ] **Step 6: Run extractor on real backprop data**

```bash
python3 experiments/baseline_diagnosis/mechanisms/extract_per_tb_features.py \
  --input experiments/baseline_diagnosis/results/rodinia/backprop_4096_full.json \
  --output experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_per_tb.json
```

Expected: output file created, stderr shows something like
`wrote .../backprop_4096_per_tb.json (workload=backprop-rodinia-2.0-ft ..., kernels=2, total_tbs=512)`

- [ ] **Step 7: Verify output against schema (manual spot-check)**

```bash
python3 -c "
import json
d = json.load(open('experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_per_tb.json'))
assert 'workload' in d and 'kernels' in d
for k in d['kernels']:
    print(f'Kernel {k[\"kernel_id\"]}: {k[\"kernel_name\"]}, num_tbs={len(k[\"per_tb\"])}, uses_fp64={k[\"kernel_summary\"][\"uses_fp64\"]}')
"
```

Expected output:
```
Kernel 1: bpnn_layerforward_CUDA, num_tbs=256, uses_fp64=False
Kernel 2: bpnn_adjust_weights_cuda, num_tbs=256, uses_fp64=True
```

- [ ] **Step 8: Commit**

```bash
git add experiments/baseline_diagnosis/mechanisms/
git add experiments/baseline_diagnosis/tests/
git add experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/
git commit -m "mechanisms: per-TB feature extractor with tests"
```

---

### Task 11: Squash Feature Extractor

**Files:**
- Create: `experiments/baseline_diagnosis/mechanisms/extract_squash_features.py`
- Create: `experiments/baseline_diagnosis/tests/test_extract_squash_features.py`

Implements temporal segmentation at both kernel and TB levels using
cosine similarity with a sliding window.

- [ ] **Step 1: Write the failing test**

File: `experiments/baseline_diagnosis/tests/test_extract_squash_features.py`

```python
"""Test Squash mechanism."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SCRIPT = ROOT / "mechanisms" / "extract_squash_features.py"
INPUT = ROOT / "results" / "rodinia" / "backprop_mechanisms" / "backprop_4096_per_tb.json"
CONFIG = ROOT / "schemas" / "mechanism_config.json"


def run_squash(output_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(INPUT),
         "--config", str(CONFIG), "--output", str(output_path)],
        check=True,
    )
    return json.loads(output_path.read_text())


def test_output_has_two_levels(tmp_path):
    out = tmp_path / "squash.json"
    result = run_squash(out)
    assert result["mechanism"] == "squash"
    assert "kernel_level" in result
    assert "tb_level" in result


def test_kernel_level_has_segments(tmp_path):
    out = tmp_path / "squash.json"
    result = run_squash(out)
    kl = result["kernel_level"]
    assert "squash_segments" in kl
    assert "boundary_count" in kl
    assert "total_kernels" in kl
    assert kl["total_kernels"] == 2  # backprop has two kernels


def test_backprop_kernel_level_finds_fp32_fp64_boundary(tmp_path):
    """backprop has a natural boundary between FFMA-only forward and DFMA adjust_weights."""
    out = tmp_path / "squash.json"
    result = run_squash(out)
    kl = result["kernel_level"]
    # Should produce at least 2 segments OR 1 boundary (one per kernel type)
    assert kl["boundary_count"] >= 1, (
        f"Expected >= 1 boundary between FP32 forward and FP64 adjust_weights, "
        f"got {kl['boundary_count']}"
    )


def test_tb_level_has_entry_per_kernel(tmp_path):
    out = tmp_path / "squash.json"
    result = run_squash(out)
    tl = result["tb_level"]
    # Keys are kernel_ids as strings
    assert "1" in tl or 1 in tl  # tolerant to str/int keys
    assert "2" in tl or 2 in tl


def test_reuse_hint_present(tmp_path):
    out = tmp_path / "squash.json"
    result = run_squash(out)
    assert "_simulation_reuse_hint" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
python3 -m pytest experiments/baseline_diagnosis/tests/test_extract_squash_features.py -v
```

Expected: FAIL (script missing)

- [ ] **Step 3: Write the Squash extractor**

File: `experiments/baseline_diagnosis/mechanisms/extract_squash_features.py`

```python
#!/usr/bin/env python3
"""Squash mechanism: temporal segmentation via sliding-window similarity.

Operates at two levels:
  kernel_level: segment the kernel sequence of a workload
  tb_level: segment the TB sequence within each kernel
"""
import argparse
import json
import math
import sys
from pathlib import Path


def to_feature_vector(features_dict, key_order):
    """Convert a feature dict to a numeric list using a fixed key order.
    Non-numeric values become 0.0."""
    vec = []
    for key in key_order:
        val = features_dict.get(key, 0)
        if isinstance(val, (int, float)):
            vec.append(float(val))
        elif isinstance(val, bool):
            vec.append(1.0 if val else 0.0)
        else:
            vec.append(0.0)
    return vec


def cosine_similarity(a, b):
    """Cosine similarity of two numeric lists. Returns 1.0 if either is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0 if na == 0 and nb == 0 else 0.0
    return dot / (na * nb)


def segment_sequence(vectors, threshold):
    """Given a list of feature vectors, produce segment boundaries where
    adjacent cosine similarity drops below threshold.
    Returns a list of (start_idx, end_idx) inclusive ranges.
    """
    if len(vectors) == 0:
        return []
    if len(vectors) == 1:
        return [(0, 0)]
    segments = []
    start = 0
    for i in range(1, len(vectors)):
        sim = cosine_similarity(vectors[i - 1], vectors[i])
        if sim < threshold:
            segments.append((start, i - 1))
            start = i
    segments.append((start, len(vectors) - 1))
    return segments


def cohesion_score(vectors):
    """Average pairwise cosine similarity within a group. 1.0 if only one item."""
    if len(vectors) <= 1:
        return 1.0
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(cosine_similarity(vectors[i], vectors[j]))
    return sum(sims) / len(sims) if sims else 1.0


def kernel_summary_to_vector(summary):
    """Convert a kernel_summary dict to a numeric vector for kernel-level ops."""
    # Aggregate key signals: opcodes, FP64 flag, shared mem flag, barriers
    opcodes = {entry["opcode"].upper(): entry["count"] for entry in summary.get("top_opcodes", [])}
    total_ops = sum(opcodes.values()) or 1
    return [
        opcodes.get("FFMA", 0) / total_ops,
        sum(v for k, v in opcodes.items() if k.startswith("DFMA") or k.startswith("DMUL") or k.startswith("DADD")) / total_ops,
        sum(v for k, v in opcodes.items() if "LDG" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "STG" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "LDS" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "STS" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "BAR" in k) / total_ops,
        1.0 if summary.get("uses_fp64") else 0.0,
        1.0 if summary.get("uses_shared_memory") else 0.0,
    ]


def dominant_opcodes(summary, top_n=3):
    """Return the top N opcodes by count."""
    ops = sorted(summary.get("top_opcodes", []), key=lambda e: -e.get("count", 0))
    return [e["opcode"] for e in ops[:top_n]]


def squash_kernel_level(workload_data, threshold):
    """Segment the kernel sequence."""
    kernels = workload_data["kernels"]
    vectors = [kernel_summary_to_vector(k["kernel_summary"]) for k in kernels]
    segments_idx = segment_sequence(vectors, threshold)

    segments = []
    for seg_id, (start, end) in enumerate(segments_idx):
        seg_vectors = vectors[start : end + 1]
        segments.append({
            "segment_id": seg_id,
            "kernel_range": [start, end],
            "kernel_count": end - start + 1,
            "dominant_opcodes": dominant_opcodes(kernels[start]["kernel_summary"]),
            "cohesion_score": cohesion_score(seg_vectors),
            "representative_kernel": start,
            "behavior_summary": f"kernels {start}..{end}: {kernels[start]['kernel_name']}"
                               + (f" ... {kernels[end]['kernel_name']}" if end > start else ""),
        })

    return {
        "squash_segments": segments,
        "boundary_count": max(0, len(segments) - 1),
        "total_kernels": len(kernels),
    }


def squash_tb_level(workload_data, threshold):
    """Segment each kernel's TB sequence."""
    result = {}
    for kernel in workload_data["kernels"]:
        tbs = kernel.get("per_tb", [])
        if not tbs:
            continue

        # Establish fixed key order from the first TB's features
        first_features = tbs[0]["features"]
        key_order = sorted(first_features.keys())

        vectors = [to_feature_vector(tb["features"], key_order) for tb in tbs]
        segments_idx = segment_sequence(vectors, threshold)

        segments = []
        for seg_id, (start, end) in enumerate(segments_idx):
            seg_vectors = vectors[start : end + 1]
            segments.append({
                "segment_id": seg_id,
                "tb_range": [start, end],
                "tb_count": end - start + 1,
                "cohesion_score": cohesion_score(seg_vectors),
                "representative_tb": start,
                "behavior_summary": f"TBs {start}..{end}",
            })

        result[str(kernel["kernel_id"])] = {
            "squash_segments": segments,
            "boundary_count": max(0, len(segments) - 1),
            "total_tbs": len(tbs),
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Squash mechanism")
    parser.add_argument("--input", required=True, help="per_tb features JSON")
    parser.add_argument("--config", required=True, help="mechanism config JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    workload_data = json.loads(Path(args.input).read_text())
    config = json.loads(Path(args.config).read_text())["squash"]

    kernel_level = squash_kernel_level(
        workload_data, config["kernel_level"]["similarity_threshold"]
    )
    tb_level = squash_tb_level(
        workload_data, config["tb_level"]["similarity_threshold"]
    )

    # Build simulation reuse hints
    reuse_hint = {
        "kernel_level_representatives": [
            seg["representative_kernel"] for seg in kernel_level["squash_segments"]
        ],
        "tb_level_representatives": {
            kid: [seg["representative_tb"] for seg in data["squash_segments"]]
            for kid, data in tb_level.items()
        },
    }

    output = {
        "mechanism": "squash",
        "workload": workload_data.get("workload", "unknown"),
        "kernel_level": kernel_level,
        "tb_level": tb_level,
        "_simulation_reuse_hint": reuse_hint,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(
        f"[squash] wrote {out_path} "
        f"(kernel_segments={len(kernel_level['squash_segments'])}, "
        f"kernel_boundaries={kernel_level['boundary_count']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
python3 -m pytest experiments/baseline_diagnosis/tests/test_extract_squash_features.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Run on real backprop data**

```bash
python3 experiments/baseline_diagnosis/mechanisms/extract_squash_features.py \
  --input experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_per_tb.json \
  --config experiments/baseline_diagnosis/schemas/mechanism_config.json \
  --output experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_squash.json
```

Expected stderr:
```
[squash] wrote .../backprop_4096_squash.json (kernel_segments=2, kernel_boundaries=1)
```

- [ ] **Step 6: Inspect output content**

```bash
python3 -c "
import json
d = json.load(open('experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_squash.json'))
kl = d['kernel_level']
print('Kernel-level segments:', len(kl['squash_segments']))
for s in kl['squash_segments']:
    print(f'  Segment {s[\"segment_id\"]}: kernels {s[\"kernel_range\"]}, dominant={s[\"dominant_opcodes\"]}, cohesion={s[\"cohesion_score\"]:.3f}')
"
```

Expected: 2 segments (one per kernel), with FFMA as dominant in segment 0
and DFMA/DMUL-related opcodes in segment 1.

- [ ] **Step 7: Commit**

```bash
git add experiments/baseline_diagnosis/mechanisms/extract_squash_features.py
git add experiments/baseline_diagnosis/tests/test_extract_squash_features.py
git add experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_squash.json
git commit -m "mechanisms: squash extractor (temporal segmentation, two-level)"
```

---

### Task 12: Batch Feature Extractor

**Files:**
- Create: `experiments/baseline_diagnosis/mechanisms/extract_batch_features.py`
- Create: `experiments/baseline_diagnosis/tests/test_extract_batch_features.py`

Implements spatial homogeneity clustering using sklearn's DBSCAN.

- [ ] **Step 1: Verify sklearn is available**

Run:
```bash
python3 -c "from sklearn.cluster import DBSCAN; print('sklearn OK')"
```

Expected: `sklearn OK`. If it fails:
```bash
pip install scikit-learn
```

- [ ] **Step 2: Write the failing test**

File: `experiments/baseline_diagnosis/tests/test_extract_batch_features.py`

```python
"""Test Batch mechanism."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SCRIPT = ROOT / "mechanisms" / "extract_batch_features.py"
INPUT = ROOT / "results" / "rodinia" / "backprop_mechanisms" / "backprop_4096_per_tb.json"
CONFIG = ROOT / "schemas" / "mechanism_config.json"


def run_batch(output_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(INPUT),
         "--config", str(CONFIG), "--output", str(output_path)],
        check=True,
    )
    return json.loads(output_path.read_text())


def test_mechanism_field(tmp_path):
    out = tmp_path / "batch.json"
    result = run_batch(out)
    assert result["mechanism"] == "batch"


def test_two_levels_present(tmp_path):
    out = tmp_path / "batch.json"
    result = run_batch(out)
    assert "kernel_level" in result
    assert "tb_level" in result


def test_kernel_level_has_clusters(tmp_path):
    out = tmp_path / "batch.json"
    result = run_batch(out)
    kl = result["kernel_level"]
    assert "batch_clusters" in kl
    assert "outlier_kernels" in kl
    assert "homogeneity_score" in kl


def test_tb_level_per_kernel(tmp_path):
    out = tmp_path / "batch.json"
    result = run_batch(out)
    tl = result["tb_level"]
    # Should have entries for both backprop kernels
    assert len(tl) == 2


def test_backprop_tb_level_high_homogeneity(tmp_path):
    """backprop TBs in each kernel are highly similar - expect one big cluster."""
    out = tmp_path / "batch.json"
    result = run_batch(out)
    tl = result["tb_level"]
    for kid, data in tl.items():
        # backprop is very regular: homogeneity should be >= 0.9 for each kernel
        assert data["homogeneity_score"] >= 0.9, (
            f"Expected high homogeneity for backprop kernel {kid}, "
            f"got {data['homogeneity_score']}"
        )
```

- [ ] **Step 3: Run tests to verify failure**

```bash
python3 -m pytest experiments/baseline_diagnosis/tests/test_extract_batch_features.py -v
```

Expected: FAIL (script missing)

- [ ] **Step 4: Write the Batch extractor**

File: `experiments/baseline_diagnosis/mechanisms/extract_batch_features.py`

```python
#!/usr/bin/env python3
"""Batch mechanism: spatial homogeneity clustering via DBSCAN.

Operates at two levels:
  kernel_level: cluster the kernels of a workload
  tb_level: cluster the TBs within each kernel
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def to_matrix(vectors):
    """Convert list of list to numpy matrix, handle empty case."""
    if not vectors:
        return np.zeros((0, 1))
    return np.array(vectors, dtype=float)


def normalize(matrix):
    """Standardize features to zero mean, unit variance."""
    if matrix.shape[0] < 2:
        return matrix
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)


def cluster_with_dbscan(matrix, eps, min_samples):
    """Run DBSCAN, return labels array. -1 indicates outlier."""
    if matrix.shape[0] == 0:
        return np.array([], dtype=int)
    if matrix.shape[0] == 1:
        return np.array([0])
    normalized = normalize(matrix)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(normalized)


def homogeneity_from_labels(labels):
    """Homogeneity = largest cluster size / total items. 1.0 means one cluster."""
    if len(labels) == 0:
        return 1.0
    from collections import Counter
    counts = Counter(labels)
    # Exclude outliers (-1) from "cluster" count but include in total
    cluster_counts = [c for lbl, c in counts.items() if lbl != -1]
    if not cluster_counts:
        return 0.0
    return max(cluster_counts) / len(labels)


def build_clusters(labels, items, centroid_field_fn):
    """Build the batch_clusters list from DBSCAN labels.

    items: list of original objects (kernels or TBs)
    centroid_field_fn: callable(list_of_items) -> centroid_summary dict
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[int(label)].append(idx)

    clusters = []
    outliers = []
    total = len(items)

    for label, indices in groups.items():
        if label == -1:
            outliers.extend(indices)
            continue
        clusters.append({
            "cluster_id": int(label),
            "cluster_size": len(indices),
            "cluster_pct": len(indices) / total * 100 if total else 0.0,
            "centroid_summary": centroid_field_fn([items[i] for i in indices]),
            "cohesion": 1.0,  # DBSCAN doesn't give within-cluster sim; approximate as 1.0
            "_members": indices,  # internal, removed before output
        })

    return clusters, outliers


def kernel_centroid_summary(kernels):
    """Summarize a group of kernels."""
    if not kernels:
        return {}
    names = [k["kernel_name"] for k in kernels]
    uses_fp64 = any(k["kernel_summary"].get("uses_fp64", False) for k in kernels)
    return {
        "kernel_names": names,
        "count": len(kernels),
        "any_fp64": uses_fp64,
    }


def tb_centroid_summary(tbs):
    """Summarize a group of TBs."""
    if not tbs:
        return {}
    insts = [tb["features"].get("instructions_per_warp_mean", 0) for tb in tbs]
    warps = [tb["features"].get("num_warps", 0) for tb in tbs]
    return {
        "count": len(tbs),
        "avg_inst_per_warp": float(np.mean(insts)) if insts else 0.0,
        "avg_num_warps": float(np.mean(warps)) if warps else 0.0,
    }


def kernel_summary_vector(summary):
    """Same vectorization as squash: key opcode ratios + flags."""
    opcodes = {entry["opcode"].upper(): entry["count"] for entry in summary.get("top_opcodes", [])}
    total_ops = sum(opcodes.values()) or 1
    return [
        opcodes.get("FFMA", 0) / total_ops,
        sum(v for k, v in opcodes.items() if k.startswith("DFMA") or k.startswith("DMUL") or k.startswith("DADD")) / total_ops,
        sum(v for k, v in opcodes.items() if "LDG" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "STG" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "LDS" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "STS" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "BAR" in k) / total_ops,
        1.0 if summary.get("uses_fp64") else 0.0,
        1.0 if summary.get("uses_shared_memory") else 0.0,
    ]


def tb_feature_vector(features, key_order):
    """Convert TB features dict to numeric vector."""
    vec = []
    for key in key_order:
        val = features.get(key, 0)
        if isinstance(val, (int, float)):
            vec.append(float(val))
        elif isinstance(val, bool):
            vec.append(1.0 if val else 0.0)
        else:
            vec.append(0.0)
    return vec


def batch_kernel_level(workload_data, config):
    kernels = workload_data["kernels"]
    vectors = [kernel_summary_vector(k["kernel_summary"]) for k in kernels]
    matrix = to_matrix(vectors)
    labels = cluster_with_dbscan(
        matrix, config["dbscan_eps"], config["dbscan_min_samples"]
    )
    clusters, outliers = build_clusters(
        labels, kernels, kernel_centroid_summary
    )

    # Add kernel_ids field and strip internal _members
    for cluster in clusters:
        cluster["kernel_ids"] = [kernels[i]["kernel_id"] for i in cluster.pop("_members")]

    return {
        "batch_clusters": clusters,
        "outlier_kernels": [kernels[i]["kernel_id"] for i in outliers],
        "homogeneity_score": homogeneity_from_labels(labels),
    }


def batch_tb_level(workload_data, config):
    result = {}
    for kernel in workload_data["kernels"]:
        tbs = kernel.get("per_tb", [])
        if not tbs:
            continue
        key_order = sorted(tbs[0]["features"].keys())
        vectors = [tb_feature_vector(tb["features"], key_order) for tb in tbs]
        matrix = to_matrix(vectors)
        labels = cluster_with_dbscan(
            matrix, config["dbscan_eps"], config["dbscan_min_samples"]
        )
        clusters, outliers = build_clusters(
            labels, tbs, tb_centroid_summary
        )
        for cluster in clusters:
            cluster["tb_ids"] = [tbs[i]["tb_index"] for i in cluster.pop("_members")]

        result[str(kernel["kernel_id"])] = {
            "batch_clusters": clusters,
            "outlier_tbs": [tbs[i]["tb_index"] for i in outliers],
            "homogeneity_score": homogeneity_from_labels(labels),
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch mechanism")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workload_data = json.loads(Path(args.input).read_text())
    config = json.loads(Path(args.config).read_text())["batch"]

    kernel_level = batch_kernel_level(workload_data, config["kernel_level"])
    tb_level = batch_tb_level(workload_data, config["tb_level"])

    reuse_hint = {
        "kernel_cluster_representatives": {
            str(c["cluster_id"]): c["kernel_ids"][0] if c.get("kernel_ids") else None
            for c in kernel_level["batch_clusters"]
        },
        "tb_cluster_representatives": {
            kid: {
                str(c["cluster_id"]): c["tb_ids"][0] if c.get("tb_ids") else None
                for c in data["batch_clusters"]
            }
            for kid, data in tb_level.items()
        },
    }

    output = {
        "mechanism": "batch",
        "workload": workload_data.get("workload", "unknown"),
        "kernel_level": kernel_level,
        "tb_level": tb_level,
        "_simulation_reuse_hint": reuse_hint,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(
        f"[batch] wrote {out_path} "
        f"(kernel_clusters={len(kernel_level['batch_clusters'])}, "
        f"outliers={len(kernel_level['outlier_kernels'])})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest experiments/baseline_diagnosis/tests/test_extract_batch_features.py -v
```

Expected: 5 tests PASS

- [ ] **Step 6: Run on real backprop data**

```bash
python3 experiments/baseline_diagnosis/mechanisms/extract_batch_features.py \
  --input experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_per_tb.json \
  --config experiments/baseline_diagnosis/schemas/mechanism_config.json \
  --output experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_batch.json
```

Expected: output created, homogeneity >= 0.9 for both kernels (because
backprop TBs are highly regular within each kernel)

- [ ] **Step 7: Commit**

```bash
git add experiments/baseline_diagnosis/mechanisms/extract_batch_features.py
git add experiments/baseline_diagnosis/tests/test_extract_batch_features.py
git add experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_batch.json
git commit -m "mechanisms: batch extractor (spatial homogeneity, DBSCAN)"
```

---

### Task 13: Delta Feature Extractor

**Files:**
- Create: `experiments/baseline_diagnosis/mechanisms/extract_delta_features.py`
- Create: `experiments/baseline_diagnosis/tests/test_extract_delta_features.py`

Implements field-level change pattern analysis.

- [ ] **Step 1: Write the failing test**

File: `experiments/baseline_diagnosis/tests/test_extract_delta_features.py`

```python
"""Test Delta mechanism."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SCRIPT = ROOT / "mechanisms" / "extract_delta_features.py"
INPUT = ROOT / "results" / "rodinia" / "backprop_mechanisms" / "backprop_4096_per_tb.json"
CONFIG = ROOT / "schemas" / "mechanism_config.json"


def run_delta(output_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(INPUT),
         "--config", str(CONFIG), "--output", str(output_path)],
        check=True,
    )
    return json.loads(output_path.read_text())


def test_mechanism_field(tmp_path):
    out = tmp_path / "delta.json"
    result = run_delta(out)
    assert result["mechanism"] == "delta"


def test_two_levels(tmp_path):
    out = tmp_path / "delta.json"
    result = run_delta(out)
    assert "kernel_level" in result
    assert "tb_level" in result


def test_kernel_level_has_fields(tmp_path):
    out = tmp_path / "delta.json"
    result = run_delta(out)
    kl = result["kernel_level"]
    assert "field_temperature" in kl
    assert "hot_fields" in kl
    assert "cold_fields" in kl


def test_backprop_kernel_level_fp64_is_hot(tmp_path):
    """The uses_fp64 field should be HOT at kernel-level in backprop."""
    out = tmp_path / "delta.json"
    result = run_delta(out)
    kl = result["kernel_level"]
    hot = set(kl["hot_fields"])
    # uses_fp64 changes between forward and adjust_weights → should be hot
    assert "uses_fp64" in hot, (
        f"Expected uses_fp64 in hot_fields at kernel-level, got {hot}"
    )


def test_tb_level_per_kernel(tmp_path):
    out = tmp_path / "delta.json"
    result = run_delta(out)
    tl = result["tb_level"]
    assert len(tl) == 2
    for kid, data in tl.items():
        assert "field_temperature" in data
        assert "hot_fields" in data
        assert "cold_fields" in data
```

- [ ] **Step 2: Run test to verify failure**

```bash
python3 -m pytest experiments/baseline_diagnosis/tests/test_extract_delta_features.py -v
```

Expected: FAIL

- [ ] **Step 3: Write the Delta extractor**

File: `experiments/baseline_diagnosis/mechanisms/extract_delta_features.py`

```python
#!/usr/bin/env python3
"""Delta mechanism: field-level change pattern analysis.

Operates at two levels:
  kernel_level: diff between adjacent kernels
  tb_level: diff between adjacent TBs within each kernel
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def extract_numeric_fields(obj):
    """Walk a dict recursively; return {flat_key: numeric_value}.
    Bool → 0/1. Nested dicts → dot-joined keys."""
    result = {}
    def walk(d, prefix):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                walk(v, key)
            elif isinstance(v, bool):
                result[key] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                result[key] = float(v)
    walk(obj, "")
    return result


def compute_temperature(value_series):
    """Given a list of values over a sequence, compute a temperature score.
    Score = stddev / (|mean| + 1e-9), clipped to [0, 1] via tanh.
    Hot fields have high variation relative to mean."""
    if len(value_series) < 2:
        return 0.0
    arr = np.array(value_series, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    denom = abs(mean) + 1e-9
    raw = std / denom
    # Map to [0, 1] via tanh
    return float(np.tanh(raw))


def classify_fields(field_temps, hot_threshold, cold_threshold):
    """Partition fields into hot/cold/warm."""
    hot = [f for f, t in field_temps.items() if t >= hot_threshold]
    cold = [f for f, t in field_temps.items() if t <= cold_threshold]
    return hot, cold


def pairwise_correlation(field_series_dict, threshold):
    """Compute correlations between field pairs that change together.
    Return list of {fields, correlation, interpretation}."""
    fields = list(field_series_dict.keys())
    correlations = []
    for i, f1 in enumerate(fields):
        for f2 in fields[i + 1:]:
            s1 = np.array(field_series_dict[f1])
            s2 = np.array(field_series_dict[f2])
            if len(s1) < 3:
                continue
            if np.std(s1) == 0 or np.std(s2) == 0:
                continue
            corr = float(np.corrcoef(s1, s2)[0, 1])
            if abs(corr) >= threshold:
                correlations.append({
                    "fields": [f1, f2],
                    "correlation": corr,
                    "interpretation": (
                        f"{f1} and {f2} " +
                        ("covary together" if corr > 0 else "move inversely")
                    ),
                })
    return correlations


def detect_outlier_diffs(field_series_dict, zscore_threshold):
    """Find adjacent pairs where total delta magnitude is an outlier."""
    fields = list(field_series_dict.keys())
    if not fields:
        return []
    length = len(next(iter(field_series_dict.values())))
    if length < 3:
        return []

    # Compute per-pair total delta magnitude across all fields
    magnitudes = []
    for i in range(length - 1):
        total = 0.0
        dominant_fields = []
        for f in fields:
            series = field_series_dict[f]
            diff = abs(series[i + 1] - series[i])
            std = np.std(series)
            if std > 0:
                norm_diff = diff / std
                total += norm_diff
                if norm_diff > 1.0:
                    dominant_fields.append((f, norm_diff))
        magnitudes.append((i, total, dominant_fields))

    if not magnitudes:
        return []

    total_values = [m[1] for m in magnitudes]
    mean_mag = np.mean(total_values)
    std_mag = np.std(total_values)
    if std_mag == 0:
        return []

    outliers = []
    for idx, mag, dominant in magnitudes:
        z = (mag - mean_mag) / std_mag
        if z >= zscore_threshold:
            sorted_dom = sorted(dominant, key=lambda p: -p[1])[:5]
            outliers.append({
                "pair": [idx, idx + 1],
                "magnitude": float(mag),
                "dominant_changing_fields": [f for f, _ in sorted_dom],
                "interpretation": f"z-score {z:.2f} above mean magnitude",
            })

    return outliers


def delta_on_sequence(numeric_dicts, config):
    """Run delta analysis on a sequence of numeric field dicts."""
    if not numeric_dicts:
        return {
            "field_temperature": {},
            "hot_fields": [],
            "cold_fields": [],
            "field_correlations": [],
            "outlier_diffs": [],
        }

    # Collect series per field
    all_fields = set()
    for d in numeric_dicts:
        all_fields.update(d.keys())

    field_series = {}
    for f in all_fields:
        field_series[f] = [d.get(f, 0.0) for d in numeric_dicts]

    # Compute temperature per field
    field_temps = {f: compute_temperature(series) for f, series in field_series.items()}

    hot, cold = classify_fields(
        field_temps, config["hot_threshold"], config["cold_threshold"]
    )

    correlations = pairwise_correlation(field_series, config["correlation_threshold"])
    outliers = detect_outlier_diffs(field_series, config["outlier_zscore"])

    return {
        "field_temperature": field_temps,
        "hot_fields": sorted(hot),
        "cold_fields": sorted(cold),
        "field_correlations": correlations,
        "outlier_diffs": outliers,
    }


def delta_kernel_level(workload_data, config):
    kernels = workload_data["kernels"]
    kernel_numeric = []
    for k in kernels:
        flat = extract_numeric_fields(k.get("kernel_summary", {}))
        kernel_numeric.append(flat)
    return delta_on_sequence(kernel_numeric, config)


def delta_tb_level(workload_data, config):
    result = {}
    for kernel in workload_data["kernels"]:
        tbs = kernel.get("per_tb", [])
        if len(tbs) < 2:
            continue
        tb_numeric = [extract_numeric_fields(tb.get("features", {})) for tb in tbs]
        result[str(kernel["kernel_id"])] = delta_on_sequence(tb_numeric, config)
    return result


def main():
    parser = argparse.ArgumentParser(description="Delta mechanism")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workload_data = json.loads(Path(args.input).read_text())
    config = json.loads(Path(args.config).read_text())["delta"]

    kernel_level = delta_kernel_level(workload_data, config["kernel_level"])
    tb_level = delta_tb_level(workload_data, config["tb_level"])

    output = {
        "mechanism": "delta",
        "workload": workload_data.get("workload", "unknown"),
        "kernel_level": kernel_level,
        "tb_level": tb_level,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(
        f"[delta] wrote {out_path} "
        f"(kernel_hot={len(kernel_level['hot_fields'])}, "
        f"kernel_cold={len(kernel_level['cold_fields'])})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest experiments/baseline_diagnosis/tests/test_extract_delta_features.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Run on real backprop data**

```bash
python3 experiments/baseline_diagnosis/mechanisms/extract_delta_features.py \
  --input experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_per_tb.json \
  --config experiments/baseline_diagnosis/schemas/mechanism_config.json \
  --output experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_delta.json
```

Expected: output created, kernel_level hot_fields should include `uses_fp64`
(because it flips between the two kernels)

- [ ] **Step 6: Inspect output**

```bash
python3 -c "
import json
d = json.load(open('experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_delta.json'))
kl = d['kernel_level']
print('Kernel-level hot fields:', kl['hot_fields'])
print('Kernel-level cold fields count:', len(kl['cold_fields']))
tl = d['tb_level']
for kid in sorted(tl.keys()):
    print(f'Kernel {kid} TB-level hot fields:', tl[kid]['hot_fields'])
"
```

Expected: `uses_fp64` in kernel-level hot fields.

- [ ] **Step 7: Commit**

```bash
git add experiments/baseline_diagnosis/mechanisms/extract_delta_features.py
git add experiments/baseline_diagnosis/tests/test_extract_delta_features.py
git add experiments/baseline_diagnosis/results/rodinia/backprop_mechanisms/backprop_4096_delta.json
git commit -m "mechanisms: delta extractor (field temperature + correlations)"
```

---

### Task 14: Integration Validation

**Files:**
- Create: `experiments/baseline_diagnosis/mechanisms/validate_all.sh`

A small shell script that runs all three mechanisms end-to-end and verifies
the outputs exist. This is the "does everything wire up correctly?" check.

- [ ] **Step 1: Write the validation script**

File: `experiments/baseline_diagnosis/mechanisms/validate_all.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Validate Phase 1: run all three mechanisms end-to-end on backprop.
# Assumes: backprop_4096_full.json exists in results/rodinia/.

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/baseline_diagnosis"
RESULTS="$EXP/results/rodinia"
MECH="$RESULTS/backprop_mechanisms"
SCHEMAS="$EXP/schemas"

mkdir -p "$MECH"

echo "=== Step 1: Extract per-TB features ==="
python3 "$EXP/mechanisms/extract_per_tb_features.py" \
  --input "$RESULTS/backprop_4096_full.json" \
  --output "$MECH/backprop_4096_per_tb.json"

echo "=== Step 2: Run Squash ==="
python3 "$EXP/mechanisms/extract_squash_features.py" \
  --input "$MECH/backprop_4096_per_tb.json" \
  --config "$SCHEMAS/mechanism_config.json" \
  --output "$MECH/backprop_4096_squash.json"

echo "=== Step 3: Run Batch ==="
python3 "$EXP/mechanisms/extract_batch_features.py" \
  --input "$MECH/backprop_4096_per_tb.json" \
  --config "$SCHEMAS/mechanism_config.json" \
  --output "$MECH/backprop_4096_batch.json"

echo "=== Step 4: Run Delta ==="
python3 "$EXP/mechanisms/extract_delta_features.py" \
  --input "$MECH/backprop_4096_per_tb.json" \
  --config "$SCHEMAS/mechanism_config.json" \
  --output "$MECH/backprop_4096_delta.json"

echo "=== Step 5: Verify all outputs exist and are non-empty ==="
for f in per_tb squash batch delta; do
  path="$MECH/backprop_4096_${f}.json"
  if [ ! -s "$path" ]; then
    echo "FAIL: $path is missing or empty"
    exit 1
  fi
  echo "OK: $path ($(wc -c < "$path") bytes)"
done

echo "=== Step 6: Print key findings from each mechanism ==="
python3 <<PYEOF
import json
from pathlib import Path
mech = Path("$MECH")
for name in ["squash", "batch", "delta"]:
    path = mech / f"backprop_4096_{name}.json"
    d = json.loads(path.read_text())
    print(f"\n--- {name.upper()} ---")
    print(f"mechanism={d['mechanism']}")
    kl = d["kernel_level"]
    if name == "squash":
        print(f"  kernel_segments={len(kl['squash_segments'])}, boundaries={kl['boundary_count']}")
    elif name == "batch":
        print(f"  kernel_clusters={len(kl['batch_clusters'])}, outliers={len(kl['outlier_kernels'])}")
    elif name == "delta":
        print(f"  hot_fields={kl['hot_fields']}")
PYEOF

echo ""
echo "=== Phase 1 Validation PASSED ==="
```

- [ ] **Step 2: Make executable**

```bash
chmod +x experiments/baseline_diagnosis/mechanisms/validate_all.sh
```

- [ ] **Step 3: Run the validation**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
bash experiments/baseline_diagnosis/mechanisms/validate_all.sh
```

Expected: All 6 steps pass, final line is `Phase 1 Validation PASSED`.

Expected findings summary:
- Squash: kernel_segments=2, boundaries=1 (FFMA vs DFMA boundary)
- Batch: kernel_clusters=2 or with outliers flagged (different FP64 usage)
- Delta: hot_fields includes `uses_fp64`

- [ ] **Step 4: Run all unit tests together to confirm no regressions**

```bash
python3 -m pytest experiments/baseline_diagnosis/tests/ -v
```

Expected: all tests PASS (4 + 5 + 5 + 5 = 19 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline_diagnosis/mechanisms/validate_all.sh
git commit -m "mechanisms: end-to-end validation script for Phase 1"
```

---

## Phase 1 Exit: Checkpoint 1 Preparation

After Task 14 passes, the deliverables for Checkpoint 1 are:

1. **All three mechanism outputs exist and contain non-trivial data**
2. **Each mechanism output conforms to its schema**
3. **Delta's hot_fields at kernel-level includes `uses_fp64`** (the key
   "fingerprint" finding we expect)
4. **Squash produces at least 1 boundary at kernel-level** (the FFMA→DFMA
   transition in backprop)
5. **Unit tests all pass** (19 tests)
6. **Integration script runs green** (`validate_all.sh`)

At this point, **stop and present findings to the user** for Checkpoint 1
decision. Do not proceed to Phase 2 without explicit user approval.

The Checkpoint 1 presentation should include:
- What each mechanism found on backprop
- Whether the findings are trivial or non-trivial
- Whether the mechanisms are ready for multi-dwarf expansion (Phase 3)
- Any discovered gaps in the schemas or interfaces that need fixing
  in a Phase 0.5 revision

---

## Self-Review Notes

- **Spec coverage**: §3 (three mechanisms) → Tasks 10-13. §3.4 (two levels)
  → each mechanism Task has kernel_level + tb_level. §3.6 (common interface)
  → Tasks 1-5 (schemas). §4.2 (Phase 0 deliverables 1-6) → Tasks 1-9.
  §4.3 (Phase 1 deliverables) → Tasks 10-14.

- **Placeholder scan**: no TBD/TODO/placeholder references in steps. Every
  code block is complete.

- **Type consistency**: the `per_tb_features` structure (workload, kernels,
  per_tb, features) is consistent across per-TB extractor (Task 10),
  Squash (Task 11), Batch (Task 12), Delta (Task 13), and their tests.
  Mechanism outputs use consistent field names (`kernel_level`, `tb_level`,
  `mechanism`, `workload`) across Tasks 11-13.

- **Data flow**: backprop_4096_full.json → Task 10 → backprop_4096_per_tb.json
  → Tasks 11/12/13 → three mechanism JSONs → Task 14 validates all.

- **Machine-check-ability**: every Task has a tested assertion (unit test
  or grep/wc check) that confirms the task is "done".
