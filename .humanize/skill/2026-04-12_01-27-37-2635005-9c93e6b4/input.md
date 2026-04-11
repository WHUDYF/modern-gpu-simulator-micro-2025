# Ask Codex Input

## Question

Please review this candidate plan v1 for the Stage C closed-loop validation of mini-transformer v4 on GPGPU-Sim 4.2.

CANDIDATE PLAN v1:

GOAL: Characterize GPGPU-Sim modeling accuracy against NCU ground truth for mini-transformer v4 on RTX 3080 Ti (GDDR6X, SM_86), validating three prescriptions from Delta mechanism analysis.

KEY CORRECTIONS FROM CODEX ANALYSIS v1:
- C-1: baseline already uses correct HW value (65536); reframed as consistency + sensitivity check
- C-2: n_mem change also affects L2 capacity; coupling with C-3 acknowledged  
- C-3: primary metric = l2_hit_rate_pct + dram_throughput_pct (not l1_hit_rate_pct)
- Terminology: GDDR6X/DRAM (not HBM)
- Added data quality gate: audit NCU metric mapping before proceeding
- Control kernel threshold: relative to target improvement (not fixed <2%)

ACCEPTANCE CRITERIA:
- AC-1: NCU data quality verified (compute_throughput != mem_pipes_busy for compute-bound kernels; if identical for 5/6 kernels, flag parsing error and halt)
- AC-2: Baseline simulation completes for all 6 kernels successfully
- AC-3 (C-1): achieved_occupancy_pct APE < 15% in baseline for gemm+attention; sensitivity test shows expected direction with reduced registers; APE > 30% = fundamental problem
- AC-4 (C-2): dram_throughput_pct APE < 20% for residual_add; sensitivity shows monotonic n_mem→dram relationship; control kernel change < half of residual improvement
- AC-5 (C-3): l2_hit_rate_pct + dram_throughput_pct characterized for softmax; sensitivity with larger dl2 shows l2_hit increase; if baseline APE already < 10% → verified without changes
- AC-6: E5 report with complete APE tables and prescription verdicts

PATH BOUNDARIES:
Upper: 3-point parameter sweep, per-launch trace, repeated NCU measurements
Lower: Single baseline + single perturbation per prescription, family-mean APE comparison

TASK BREAKDOWN (9 tasks):
task1 [analyze]: Audit NCU metric mapping (compute_throughput/mem_pipes_busy aliasing)
task2 [coding]: Create config directories (baseline, rx_C1, rx_C2, rx_C3)
task3 [coding]: Record NVBit trace on RTX 3080 Ti for mini_transformer_v4
task4 [coding]: Write parse_sim_output.py and compute_ape.py tools
task5 [coding]: Run baseline simulation, compute baseline APE
task6 [coding]: C-1 sensitivity (shader_registers 65536→32768)
task7 [coding]: C-2 sensitivity (n_mem 24→12, acknowledge L2 coupling caveat)
task8 [coding]: C-3 sensitivity (dl2 S:64→S:256, primary metric: l2_hit_rate + dram)
task9 [coding]: Generate E5_stageC_validation.md

OPEN ISSUES NOT YET RESOLVED:
- Whether to add elapsed_cycles as primary metric
- Cache warm/cold state semantics (flush_l1/l2_cache=1 = cold start; HW may have warm state)
- Whether 3-point sweep is required at lower bound
- launch-level vs family-mean matching

Please respond in this format:
AGREE: <points accepted as reasonable>
DISAGREE: <points considered unreasonable and why>
REQUIRED_CHANGES: <must-fix items before convergence>
OPTIONAL_IMPROVEMENTS: <non-blocking improvements>
UNRESOLVED: <opposite opinions needing user decisions>

## Configuration

- Model: gpt-5.4
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-04-12_01-27-37
