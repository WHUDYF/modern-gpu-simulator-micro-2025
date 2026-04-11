# Ask Codex Input

## Question

This is Round 2 convergence review of the Stage C validation plan for mini-transformer v4 on GPGPU-Sim.

Round 1 REQUIRED_CHANGES addressed:
1. AC-1 rewritten as provenance audit: verify CSV distinct metric rows, parser no name collision, launch-shape aggregation preserved, spot-check 4 discriminating metrics
2. Launch comparison key: (kernel_short_name, grid_size, block_size) - not family mean alone
3. Acceptance language downgraded from "monotonicity" to "directionally consistent sensitivity"
4. C-3 success: removed "baseline APE<10% => verified" shortcut; now says "baseline accurate; causal attribution not tested without perturbation"
5. C-2 verdict explicitly labeled "coupled DRAM+L2 stress test, not pure bandwidth proof"
6. 6 kernels explicitly bound to NCU targets with grid/block dimensions documented

Round 1 UNRESOLVED (pending user decisions - carried forward as DEC items):
- DEC-1: 3-point parameter sweep vs single perturbation (monotonicity vs directionality claim)
- DEC-2: elapsed_cycles as primary vs secondary metric
- DEC-3: Cache warm/cold semantics (keep flush_l1/l2_cache=1 vs disable)
- DEC-4: C-2 coupled validation acceptable now vs deferred isolated test

UPDATED ACCEPTANCE CRITERIA:
- AC-1: NCU data provenance audit - (a) raw CSV has distinct rows per metric, (b) parser has no friendly-name collision for compute_throughput vs mem_pipes_busy, (c) per-(short_name,grid,block) aggregation preserved, (d) 4 discriminating metrics spot-checked across 3 kernels
- AC-2: Baseline simulation completes for all 6 kernels
- AC-3 (C-1): baseline achieved_occupancy_pct APE < 15% for gemm+attention; sensitivity (65536→32768) shows occupancy reduction in expected direction; APE > 30% = fundamental problem flag
- AC-4 (C-2): baseline dram_throughput_pct APE < 20% for residual_add; sensitivity (n_mem 24→12) labeled as coupled DRAM+L2 stress test; direction of change documented; control kernels' response < 50% of residual_add's response magnitude
- AC-5 (C-3): baseline l2_hit_rate_pct + dram_throughput_pct APE characterized; sensitivity (dl2 S:64→S:256) shows l2_hit direction change for softmax; if no perturbation: verdict = "baseline accurate; causal not tested"
- AC-6: E5 report with APE tables, prescription verdicts in two layers (baseline accuracy + prescription sensitivity), failure taxonomy

TASK BREAKDOWN UNCHANGED (9 tasks):
task1 [analyze]: NCU provenance audit
task2 [coding]: Config directories
task3 [coding]: NVBit trace recording
task4 [coding]: parse_sim_output.py + compute_ape.py
task5 [coding]: Baseline simulation
task6 [coding]: C-1 sensitivity
task7 [coding]: C-2 sensitivity (with coupling caveat)
task8 [coding]: C-3 sensitivity
task9 [coding]: E5 report

Please respond in the convergence review format:
AGREE: <points accepted>
DISAGREE: <remaining issues>
REQUIRED_CHANGES: <must-fix before convergence>
OPTIONAL_IMPROVEMENTS: <non-blocking>
UNRESOLVED: <genuine user decisions, not Claude-Codex disagreements>

## Configuration

- Model: gpt-5.4
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-04-12_01-31-09
