AGREE:
- Reframing C-1 as a consistency plus sensitivity check is correct. The baseline config already has `gpgpu_shader_registers=65536`, so the validation question is whether the simulator responds correctly around the hardware-faithful point, not whether 65536 itself is a new prescription.
- Promoting `l2_hit_rate_pct + dram_throughput_pct` to the primary C-3 evidence is the right correction. That matches the stated softmax hypothesis much better than `l1_hit_rate_pct`.
- Calling out the `n_mem` to L2 coupling is necessary and materially improves the plan over the earlier version.
- The terminology correction from HBM to GDDR6X/DRAM is right and should stay.
- Adding a data-quality gate before simulator work is good. The repo already shows enough mapping risk that this should be a hard front-door check.
- Using relative control-kernel thresholds instead of a fixed `<2%` cutoff is more defensible.
- The overall lower-bound shape is reasonable: baseline plus one sensitivity run per prescription is a valid minimum closed-loop design if the claims are kept directional.

DISAGREE:
- AC-1 is too strong in the wrong way. In the current raw NCU CSV, `Compute (SM) Throughput` and `Mem Pipes Busy` are already numerically identical launch-by-launch for all 6 kernel families, so “identical for 5/6 => parsing error and halt” would false-stop on existing data. This is not enough to prove parser aliasing. The current parser in [parse_ncu_v2.py](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/parse_ncu_v2.py:1) is reading distinct metric names; the equality is present in the source CSV itself in [mini_transformer_v4_ncu.csv](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_ncu.csv:19).
- “Launch-level vs family-mean matching” cannot remain open. It is already a live correctness problem, not a philosophical choice. The current parser collapses by short kernel name only, while `gemm_tiled` appears with at least two launch shapes in the NCU CSV, and the current per-launch full JSON reuses the same hardware metrics across distinct gemm launches. Any APE computed on top of that is suspect.
- The plan asks for monotonicity in AC-4 and AC-5, but the lower bound only runs a single perturbation. One perturbation can show directionality, not monotonicity.
- AC-5 is too weak where it says “if baseline APE already <10% -> verified without changes.” That verifies current baseline accuracy, not the causal C-3 prescription.

REQUIRED_CHANGES:
- Resolve matching granularity before convergence. The comparison key should be at least `(kernel_short_name, grid_size, block_size)`, and ideally exact launch order if the trace extraction preserves it. Family-mean summaries can still appear in E5, but only as secondary rollups.
- Rewrite AC-1 as a provenance audit, not a numeric-inequality test. Required checks should be:
  1. confirm raw CSV contains distinct rows for each metric name and unit,
  2. confirm parser does not overwrite metrics with same-friendly-name collisions,
  3. confirm launch-shape-specific aggregation is preserved,
  4. compare a few discriminating metrics such as `mem_busy_pct`, `dram_throughput_pct`, `l2_hit_rate_pct`, and `elapsed_cycles`.
- Make the lower-bound path consistent with the acceptance criteria. Either:
  1. add a real 3-point sweep per prescription when claiming monotonicity, or
  2. downgrade the acceptance wording to “directionally consistent sensitivity.”
- Do not allow low baseline APE alone to “validate” C-3. If you skip perturbation, the correct verdict is “baseline already accurate; causal attribution not tested,” not “prescription verified.”
- Rephrase the C-2 verdict semantics. A sweep on `gpgpu_n_mem` validates a coupled DRAM-plus-L2 response, not a pure DRAM-bandwidth model. That is acceptable as a stress test, but not as standalone proof of the bandwidth prescription.
- Explicitly bind the 6 representative traces to the exact NCU comparison targets. Right now the plan says “6 kernels,” but the underlying workload still has multiple launches for at least `gemm_tiled`; that ambiguity must be closed before APE tables are trusted.

OPTIONAL_IMPROVEMENTS:
- Add `elapsed_cycles` as a secondary diagnostic metric. It is useful for spotting bad launch matching and compensating-model-error cases, but I would not make it a primary pass/fail metric until launch matching is fixed.
- Split E5 into two verdict layers: “baseline accuracy” and “prescription sensitivity.” That avoids conflating “the default config is already right” with “the proposed bottleneck explanation was causally validated.”
- For C-1, keep the fallback check on FP32 initiation interval only if occupancy matches hardware but IPC/throughput still miss badly. That makes it a conditional branch instead of scope creep.
- Predefine a failure taxonomy in E5: data issue, launch-matching issue, simulator completeness issue, or prescription not supported. That will make inconclusive results much easier to interpret.

UNRESOLVED:
- Whether the lower bound should include true 3-point sweeps. If the goal is only directional closed-loop evidence, a single perturbation is enough. If the goal is monotonicity claims, it is not.
- Whether `elapsed_cycles` should be primary. I would keep it secondary until launch-level matching is corrected.
- How to handle cache warm/cold semantics. `flush_l1/l2_cache=1` gives a cleaner simulator experiment, but it may bias hit-rate comparisons against hardware steady-state behavior.
- Whether C-2 should be accepted as a coupled validation path now, or deferred until you can isolate channel-count effects from L2-capacity effects. My view is that coupled evidence is acceptable only if the report labels it that way explicitly.
