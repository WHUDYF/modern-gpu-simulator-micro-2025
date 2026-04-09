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
