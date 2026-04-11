AGREE:
- AC-1’s reframing as a provenance audit is the right correction. I directly checked `mini_transformer_v4_ncu.csv`, and the raw CSV does contain distinct rows per metric name; `Compute (SM) Throughput`, `Mem Pipes Busy`, `Achieved Occupancy`, `DRAM Throughput`, `L2 Hit Rate`, and `Elapsed Cycles` each appear 79 times.
- The launch comparison key change to `(kernel_short_name, grid_size, block_size)` is the correct standard. That is necessary because `gemm_tiled` really does appear with multiple launch shapes in the raw CSV.
- Replacing “monotonicity” with “directionally consistent sensitivity” is the right downgrade for a one-step perturbation design.
- C-2 being labeled a coupled DRAM+L2 stress test instead of a pure bandwidth proof is technically correct.
- C-3’s split verdict language is better: “baseline accurate” and “causal attribution tested/not tested” are separate claims and should stay separate.
- Keeping DEC-1 through DEC-4 as explicit user decisions is appropriate; they are genuine study-design choices, not review defects.

DISAGREE:
- The plan is not yet converged because the checked-in parsing path still collapses launches by short name, which violates the new provenance requirement. [parse_ncu_v2.py](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/parse_ncu_v2.py:4) explicitly aggregates “per-kernel type,” and the implementation stores one `grid_size`/`block_size` per short name at [lines 68-108](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/parse_ncu_v2.py:68). The result is already lossy in [mini_transformer_v4_hw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_hw.json:97), where `gemm_tiled` shows `num_launches: 37` but only one grid shape.
- The simulation-side plan is still inconsistent with the new keying rule. [stageC-validation.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/plans/2026-04-12-stageC-validation.md:281) still says “Aggregate by short name (mean across launches),” and `compute_ape.py` joins only on short kernel name at [lines 323-372](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/plans/2026-04-12-stageC-validation.md:323).
- AC-5 says C-3 is about `l2_hit_rate_pct` plus `dram_throughput_pct`, but the current spec/plan still use `l1_hit_rate_pct`, and the proposed simulator parser does not extract any L2 hit metric. See [design spec](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/specs/2026-04-12-stageC-validation-design.md:71) and [plan parser map](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/plans/2026-04-12-stageC-validation.md:238).
- The old acceptance logic is still embedded in the plan/spec. Examples: “baseline APE < 10% => C-1 directly passes” at [design spec lines 97-101](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/specs/2026-04-12-stageC-validation-design.md:97) and again in [Task 5 Step 3](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/plans/2026-04-12-stageC-validation.md:516). That conflicts with the Round 1 correction.
- AC-4’s “control kernels’ response < 50% of residual_add’s response magnitude” is still underdefined. Right now the plan does not specify whether “response magnitude” means raw metric delta, APE delta, normalized delta, or sign-preserving change. Without that, reviewers can reach different verdicts from the same data.
- AC-1(c)/(d) still needs a structural check, not just a value check. In this dataset, `Compute (SM) Throughput` and `Mem Pipes Busy` are distinct raw metrics but numerically identical for the sampled kernels, so a weak spot-check could still miss a parser collision.

REQUIRED_CHANGES:
- Update the actual plan and parser contract so both NCU and simulator data are keyed by `(kernel_short_name, grid_size, block_size)` or by an even more explicit launch identifier. Short-name family means are no longer acceptable anywhere in the validation path.
- Add an explicit Task 1 provenance-audit deliverable to the implementation plan, not just to the acceptance summary. It should produce evidence for: distinct raw rows, distinct parser field mappings, preserved launch-shape aggregation, and 4-metric spot-checks across 3 kernels.
- Align C-3 end-to-end with `l2_hit_rate_pct`. Either prove GPGPU-Sim exposes an L2 hit metric and specify the exact log field to parse, or revise AC-5 before convergence. Leaving the acceptance text on L2 while the implementation still measures L1 is not acceptable.
- Remove stale success language from the plan/spec and replace it with the updated two-layer verdict model: baseline accuracy and perturbation sensitivity. That includes Task 5/6/7 Step 3 and the “APE must decrease / direct pass” wording in the design doc.
- Define `response magnitude` precisely for AC-4, including formula, metric, and sign convention. If the 50% control threshold applies only to `dram_throughput_pct`, state that explicitly.
- Ensure the six validation targets are shown in one explicit table in the plan with kernel short name, full kernel name if needed, grid size, block size, and launch count used for aggregation. Right now that binding is not sufficiently operational in the checked-in docs.

OPTIONAL_IMPROVEMENTS:
- Record the observed total launch count exactly as 79, not “接近 78,” and include the per-shape split for `gemm_tiled`.
- Preserve per-launch `ID` from the NCU CSV in the audit artifact even if the final comparison aggregates by `(short_name, grid, block)`. That makes trace/debugging much easier.
- Add `elapsed_cycles` as a secondary reporting metric in E5 even if DEC-2 leaves it non-primary.
- Give C-3 a control-response threshold analogous to C-2, or explicitly state why you are not requiring one there.

UNRESOLVED:
- DEC-1: single perturbation vs 3-point sweep. Single perturbation is enough only for a directional-sensitivity claim; 3-point is needed for any stronger trend claim.
- DEC-2: `elapsed_cycles` primary vs secondary metric.
- DEC-3: cache warm/cold semantics for `flush_l1/l2_cache=1` vs disable.
- DEC-4: whether the coupled C-2 validation is acceptable for this round or whether an isolated bandwidth test is required before claiming convergence.
