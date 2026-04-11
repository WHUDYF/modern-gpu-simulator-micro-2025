# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

在 GPGPU-Sim 4.2（SM86_RTX3080_TI 配置）上，对 mini-transformer v4 的三条 Stage C 架构处方（C-1/C-2/C-3）进行**因果方向性验证**：证明 Delta 机制识别的参数字段（`gpgpu_shader_registers`、`gpgpu_n_mem`、`gpgpu_cache:dl2`）在模拟器中具有与 NCU 实测数据（RTX 3080 Ti，GDDR6X）方向一致的因果响应。判决采用两层结构：Baseline 准确性（APE 水线）+ 处方敏感性（单点 perturbation 方向验证），最终输出 `E5_stageC_validation.md` 报告。

## Acceptance Criteria

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->


每条标准包含正向测试（预期通过）和负向测试（预期拒绝），用于确定性验证。

- AC-1: NCU 数据溯源审计通过
  - Positive Tests (expected to PASS):
    - 原始 CSV 中 `Compute (SM) Throughput`、`Mem Pipes Busy`、`DRAM Throughput`、`L2 Hit Rate`、`Elapsed Cycles`、`Achieved Occupancy` 各作为独立行存在（总行数 = 6 种指标 × N 次 launch，各行 metric name 不同）
    - `parse_ncu_v2.py` 的 METRIC_MAP 中无两个不同 CSV metric name 映射到同一 JSON 字段名
    - 以 `(kernel_short_name, grid_size, block_size)` 为聚合键生成的核心对比表包含 6 个目标 kernel 条目（gemm_tiled、attention_score、residual_add、layernorm、softmax、context_mul），每条包含 grid/block 尺寸和 launch 计数
    - `gemm_tiled` 若存在多种 launch shape，各 shape 作为独立聚合条目存在
  - Negative Tests (expected to FAIL):
    - `compute_throughput_pct` 与 `mem_pipes_busy_pct` 被映射到同一原始 CSV 列名 → 标记"字段别名碰撞"，停止后续 APE 计算
    - 6 个目标 kernel 中任一在 NCU CSV 中找不到对应行 → 标记"launch 覆盖缺失"并停止

- AC-2: 基准模拟成功完成
  - Positive Tests (expected to PASS):
    - 6 个代表 kernel 的模拟日志全部生成（无 crash / timeout）
    - 每个 kernel 日志中可解析：`gpu_ipc`、`gpu_occupancy`、`L1D_total_cache_hit_rate`、`dram_bw_util`、`L2_total_cache_miss_rate`
    - `parse_sim_output.py` 输出的 JSON 包含所有 6 个 kernel 条目，字段完整
  - Negative Tests (expected to FAIL):
    - 模拟日志中有 kernel 缺失或 GPGPU-Sim 输出错误行 → baseline 无效，不得进行处方比较

- AC-3: C-1 寄存器文件模型方向性验证（目标：gemm_tiled、attention_score）
  - Positive Tests (expected to PASS):
    - baseline 中 `achieved_occupancy_pct` APE 已按 < 15% / 15-30% / > 30% 三档分级记录
    - 敏感性测试（`gpgpu_shader_registers` 65536→32768）后，gemm_tiled 和 attention_score 的 `achieved_occupancy_pct` 均出现下降（模拟器正确响应寄存器减少）

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task1: NCU 溯源审计（CSV 独立行、METRIC_MAP 碰撞、(short_name,grid,block) 键表） | AC-1 | pending | analyze | codex | 所有 APE 计算的前置条件 |
| task2: 创建 4 套配置目录（baseline/rx_C1/rx_C2/rx_C3） | AC-2 | pending | coding | claude | 依赖 task1 通过 |
| task3: SSH 录制 NVBit trace（RTX 3080 Ti） | AC-2 | pending | coding | claude | 依赖 task2 |
| task4: 实现 parse_sim_output.py + compute_ape.py | AC-2,3,4,5 | pending | coding | claude | 依赖 task1 比较键格式 |
| task5: 运行 baseline 模拟，计算 baseline APE 表 | AC-2,3,4,5 | pending | coding | claude | 依赖 task3 + task4 |
| task6: C-1 敏感性测试（registers 65536→32768，occupancy 方向验证） | AC-3 | pending | coding | claude | 依赖 task5 |
| task7: C-2 敏感性测试（n_mem 24→12，dram_throughput 方向验证，耦合标注） | AC-4 | pending | coding | claude | 依赖 task5 |
| task8: C-3 敏感性测试（dl2 S:64→S:256，l2_hit_rate 方向验证，冷启动注明） | AC-5 | pending | coding | claude | 依赖 task5 |
| task9: 生成 E5_stageC_validation.md（两层判决 + 失败归因） | AC-6 | pending | coding | claude | 依赖 task6+7+8 |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|

### Open Issues
<!-- Issues discovered during implementation -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
