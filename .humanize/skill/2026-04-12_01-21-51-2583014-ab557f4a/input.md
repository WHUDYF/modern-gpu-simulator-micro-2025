# Ask Codex Input

## Question

Repository: modern-gpu-simulator-micro-2025
Project: GPU performance diagnosis framework using GPGPU-Sim 4.2 trace-driven simulation.

Context:
- Three-stage diagnosis protocol (A: launch config → B: kernel implementation → C: hardware architecture calibration)
- mini-transformer v4 has been cleaned of software defects (Stage A/B complete)
- Stage C generates simulator calibration prescriptions based on Delta mechanism cross-kernel correlation
- Hardware: RTX 3080 Ti (SM_86), NCU measurements available in mini_transformer_v4_ncu.csv
- Simulator: GPGPU-Sim 4.2, config SM86_RTX3080_TI

Draft design doc (Stage C closed-loop validation):

PRESCRIPTIONS TO VALIDATE:
- C-1: gpgpu_shader_registers (target: gemm_tiled, attention_score) - HIGH confidence
- C-2: gpgpu_n_mem + HBM timing (target: residual_add) - HIGH confidence  
- C-3: gpgpu_cache:dl2 (target: softmax) - MEDIUM confidence

CURRENT CONFIG VALUES:
- gpgpu_shader_registers 65536
- gpgpu_n_mem 24
- gpgpu_cache:dl2 S:64:128:16,L:B:m:L:P,A:192:96,32:0,32

PROPOSED VALIDATION FLOW:
1. Record NVBit trace on RTX 3080 Ti for mini_transformer_v4
2. Run baseline simulation with SM86_RTX3080_TI config
3. Compute APE (Absolute Percentage Error) vs NCU measurements
4. For each prescription: modify one parameter, re-run, compare APE delta

6 REPRESENTATIVE KERNELS: gemm_tiled_1, residual_add_9, layernorm_10, attention_score, softmax_kernel, context_mul

5 METRICS:
- achieved_occupancy_pct (C-1)
- compute_throughput_pct (C-1)
- dram_throughput_pct (C-2)
- warp_cycles_per_issued_inst (C-2/C-3)
- l1_hit_rate_pct (C-3)

SUCCESS CRITERIA:
- Target kernel key metric APE decreases
- Control kernel APE change < 2%
- APE delta > 5% (above measurement noise)
APE grades: <10% accurate, 10-30% biased, >30% fundamental problem

Please analyze this design and provide:

CORE_RISKS: highest-risk assumptions and potential failure modes
MISSING_REQUIREMENTS: likely omitted requirements or edge cases
TECHNICAL_GAPS: feasibility or architecture gaps
ALTERNATIVE_DIRECTIONS: viable alternatives with tradeoffs
QUESTIONS_FOR_USER: questions that need explicit human decisions
CANDIDATE_CRITERIA: candidate acceptance criteria suggestions

## Configuration

- Model: gpt-5.4
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-04-12_01-21-51
