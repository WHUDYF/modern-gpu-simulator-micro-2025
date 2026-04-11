Read and execute below with ultrathink

## Goal Tracker Setup (REQUIRED FIRST STEP)

Before starting implementation, you MUST initialize the Goal Tracker:

1. Read @/home/dyf/modern-gpu-simulator-micro-2025/.humanize/rlcr/2026-04-12_02-31-41/goal-tracker.md
2. If the "Ultimate Goal" section says "[To be extracted...]", extract a clear goal statement from the plan
3. If the "Acceptance Criteria" section says "[To be defined...]", define 3-7 specific, testable criteria
4. Populate the "Active Tasks" table with tasks from the plan, mapping each to an AC and filling Tag/Owner
5. Write the updated goal-tracker.md

**IMPORTANT**: The IMMUTABLE SECTION can only be modified in Round 0. After this round, it becomes read-only.

---

## Implementation Plan

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList) to track each item in order of importance.
You are strictly prohibited from only addressing the most important issues - you MUST create Tasks for ALL discovered issues and attempt to resolve each one.

## Task Tag Routing (MUST FOLLOW)

Each task must have one routing tag from the plan: `coding` or `analyze`.

- Tag `coding`: Claude executes the task directly.
- Tag `analyze`: Claude must execute via `/humanize:ask-codex`, then integrate Codex output.
- Keep Goal Tracker "Active Tasks" columns **Tag** and **Owner** aligned with execution (`coding -> claude`, `analyze -> codex`).
- If a task has no explicit tag, default to `coding` (Claude executes directly).

# Stage C 闭环验证：mini-transformer v4 模拟器处方验证计划

## Goal Description

在 GPGPU-Sim 4.2（SM86_RTX3080_TI 配置）上，对 mini-transformer v4 的三条 Stage C 架构处方（C-1/C-2/C-3）进行**因果方向性验证**：证明 Delta 机制识别的参数字段（`gpgpu_shader_registers`、`gpgpu_n_mem`、`gpgpu_cache:dl2`）在模拟器中具有与 NCU 实测数据（RTX 3080 Ti，GDDR6X）方向一致的因果响应。判决采用两层结构：Baseline 准确性（APE 水线）+ 处方敏感性（单点 perturbation 方向验证），最终输出 `E5_stageC_validation.md` 报告。

## Acceptance Criteria

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
    - 对照 kernel（residual_add）的 `achieved_occupancy_pct` 变化幅度 < 目标 kernel 变化幅度的 50%
  - Negative Tests (expected to FAIL):
    - 敏感性测试后 occupancy 不降反升 → 标记"模型反转异常"
    - baseline APE > 30% → 标记"根本性建模问题"，需独立调查

- AC-4: C-2 DRAM 带宽模型方向性验证（目标：residual_add）
  - Positive Tests (expected to PASS):
    - baseline 中 residual_add 的 `dram_throughput_pct` APE 已按参考水线（< 20% 为可接受）分级记录
    - 敏感性测试（`gpgpu_n_mem` 24→12）后 residual_add 的 `dram_throughput_pct` 出现下降（响应内存控制器减少）
    - 验证报告中明确标注"此为 DRAM+L2 联合压力测试（n_mem 同时改变 L2 切片数量，非纯带宽单因子）"
    - 对照 kernel（gemm_tiled）`dram_throughput_pct` 绝对变化幅度 < residual_add 变化幅度的 50%
  - Negative Tests (expected to FAIL):
    - 敏感性测试后 residual_add 的 dram_throughput 变化 < 2%（绝对值）→ 标记"DRAM 模型无响应"
    - 所有 kernel 的 dram_throughput 等幅同向变化 → 标记"全局扰动，无法区分处方"

- AC-5: C-3 L2 缓存容量模型方向性验证（目标：softmax）
  - Positive Tests (expected to PASS):
    - baseline 中 softmax 的 `l2_hit_rate_pct`（= (1 − L2_total_cache_miss_rate) × 100）和 `dram_throughput_pct` APE 已记录
    - 敏感性测试（`gpgpu_cache:dl2` S:64:128:16→S:256:128:16）后 softmax 的 l2_hit_rate 出现上升，dram_throughput 出现下降（L2 增大后更多数据命中缓存）
    - 报告中包含声明："冷启动已知限制（flush_l1/l2_cache=1）：模拟器 L2 热态与硬件连续 launch 时的暖态可能不对齐，l2_hit_rate 比较存在系统性偏差"
  - Negative Tests (expected to FAIL):
    - 敏感性测试后 softmax l2_hit_rate 无明显变化（< 2%）→ 标记"L2 模型无响应"
    - dram_throughput 反向变化（L2 增大后 DRAM 利用率上升）→ 标记"模型方向异常"

- AC-6: E5_stageC_validation.md 生成，包含两层判决
  - Positive Tests (expected to PASS):
    - 报告包含：baseline APE 表（6 kernel × 5 指标）、三条处方敏感性 APE 对比表、两层判决（"Baseline 准确性"层 + "处方敏感性"层）
    - 每条处方的判决结论使用以下分类之一：有效（方向一致）/ 无效（无响应或反转）/ 不适用（baseline 已准确，因果性未测）/ 数据质量问题
    - 报告包含失败归因分类：数据问题 / launch 匹配问题 / 模拟器建模限制 / 处方不成立
    - `elapsed_cycles` 作为辅助诊断列出现在 APE 表中
  - Negative Tests (expected to FAIL):
    - 任一处方判决缺失或仅有 APE 数字无文字结论 → 报告不完整

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

实现包含：NVBit 每条 launch 提取的精确 subtrace；以 `(short_name, grid, block, launch_order)` 作为精确比较键；3 点参数 sweep（每条处方测 3 个参数值，声明单调性）；`elapsed_cycles` 作为主要指标；C-2 使用独立 microbench 标定 DRAM 子系统后再做应用验证；GPU 时钟锁定和 3 次重复 NCU 采样。

### Lower Bound (Minimum Acceptable Scope)

实现包含：全量 trace 按 `(short_name, grid, block)` 聚合的 family-mean 比较；baseline 模拟 + 每条处方一个对照点的单点 perturbation；方向性声明（不声明单调性）；`elapsed_cycles` 作为辅助诊断指标；C-2 标注参数耦合但不做独立隔离实验；保持 flush_l1/l2_cache=1 冷启动配置，E5 中注明已知限制。

### Allowed Choices
- Can use: GPGPU-Sim 日志正则解析、NCU CSV 直接解析、Python dict/JSON 存储 APE 表、Markdown 表格输出 E5 报告
- Cannot use: 在 AC-1 溯源审计通过前进行 APE 计算；在 APE < 5% 时声称"精确校准"（仅允许声称"方向一致"）；将 n_mem 扰动的结论描述为"纯 DRAM 带宽证明"

## Feasibility Hints and Suggestions

> **Note**: 本节仅供参考，提供一种可能的实现路径，不作为强制要求。

### Conceptual Approach

```
溯源审计
  读取 mini_transformer_v4_ncu.csv
  按 (短名, grid, block) 分组 → 建立 6 个目标 kernel 对比键表
  检查 METRIC_MAP 无 friendly-name 碰撞

配置准备
  baseline/ : 原始 SM86_RTX3080_TI 配置，不修改任何参数
  rx_C1/    : gpgpu_shader_registers 65536 → 32768
  rx_C2/    : gpgpu_n_mem 24 → 12
  rx_C3/    : gpgpu_cache:dl2 S:64:128:16... → S:256:128:16...

Trace 录制
  SSH to RTX 3080 Ti
  NVBit tracer 完整录制 mini_transformer_v4 运行
  rsync dynamic_trace.pb 到本机 experiments/mini_transformer/traces/

解析工具
  parse_sim_output.py:
    regex 提取 per-kernel 统计块
    关键字段: gpu_ipc, gpu_occupancy,
              L1D_total_cache_hit_rate, dram_bw_util,
              L2_total_cache_miss_rate
    l2_hit_rate = (1 - L2_total_cache_miss_rate) * 100

  compute_ape.py:
    join NCU + sim on (short_name, grid, block)
    APE = abs(sim_val - ncu_val) / ncu_val * 100
    返回 {kernel: {metric: {ncu, sim, ape}}}

两层判决逻辑（E5）
  Layer-1 (Baseline 准确性): APE < 15% → 准确 / 15-30% → 有偏差 / > 30% → 根本问题
  Layer-2 (处方敏感性):
    |target_APE_delta| > 5% AND direction == expected → 有效
    |target_APE_delta| < 2% → 无响应
    direction != expected → 反转异常
    control_delta > target_delta * 0.5 → 特异性不足
```

### Relevant References
- `simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config` — baseline 配置，`gpgpu_shader_registers`（=65536）、`gpgpu_n_mem`（=24）、`gpgpu_cache:dl2` 均在此文件
- `experiments/baseline_diagnosis/parse_ncu_v2.py` — 现有 NCU 解析工具，METRIC_MAP 是 AC-1 溯源审计的对象
- `experiments/mini_transformer/mini_transformer_v4_ncu.csv` — NCU 地面真值数据（原始 CSV，含独立指标行）
- `experiments/mini_transformer/mini_transformer_v4_hw.json` — 已解析的硬件摘要（注意：当前按 family mean 聚合，可能丢失多 launch-shape 信息，需与 AC-1 结论对比）
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc`（L2417-L2423 附近）— GPGPU-Sim 输出 `L2_total_cache_miss_rate` 字段的代码位置

## Dependencies and Sequence

### Milestones

1. **数据质量基础**：task1（溯源审计）完成，建立权威比较键表
   - 产出：`(short_name, grid, block, launch_count)` 对比表，6 个目标 kernel 确认
   - 产出：METRIC_MAP 审计结论（pass / 需修正字段）
   - 后续所有 APE 计算依赖此产出

2. **模拟环境就绪**：task2（配置）+ task3（trace）+ task4（工具）并行进行，均依赖 task1
   - task2 产出：4 套 gpgpusim.config（baseline / rx_C1 / rx_C2 / rx_C3）
   - task3 产出：`experiments/mini_transformer/traces/dynamic_trace.pb`
   - task4 产出：`parse_sim_output.py`、`compute_ape.py`

3. **Baseline APE 建立**：task5 完成（依赖 task3、task4）
   - 产出：baseline APE 表（6 kernel × 5 指标），作为 task6/7/8 的 delta 基准

4. **三条处方验证**：task6、task7、task8 可并行执行（均依赖 task5）
   - task6 产出：C-1 方向性验证结论
   - task7 产出：C-2 验证结论（含耦合标注）
   - task8 产出：C-3 验证结论（含冷启动限制标注）

5. **最终报告**：task9（依赖 task6/7/8 全部完成）
   - 产出：`results/mini_transformer_v4/E5_stageC_validation.md`

## Task Breakdown

每个任务必须包含且仅包含一个路由标签：
- `coding`：由 Claude 实现
- `analyze`：通过 Codex 执行（`/humanize:ask-codex`）

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | 对 `mini_transformer_v4_ncu.csv` 做溯源审计：验证 CSV 独立指标行、`parse_ncu_v2.py` METRIC_MAP 无碰撞、建立 `(short_name, grid, block)` 比较键表、确认 6 个目标 kernel 覆盖 | AC-1 | analyze | — |
| task2 | 在 `results/mini_transformer_v4/configs/` 下创建四套配置目录：`baseline/`（原始配置）、`rx_C1/`（registers 65536→32768）、`rx_C2/`（n_mem 24→12）、`rx_C3/`（dl2 S:64→S:256），各含 gpgpusim.config 和 trace.config | AC-2 | coding | task1 |
| task3 | SSH 至 RTX 3080 Ti，使用 NVBit tracer 录制 mini_transformer_v4 完整运行，rsync `dynamic_trace.pb` 回本机 `experiments/mini_transformer/traces/` | AC-2 | coding | task2 |
| task4 | 实现 `experiments/baseline_diagnosis/parse_sim_output.py`（正则提取 GPGPU-Sim per-kernel 统计，含 `L2_total_cache_miss_rate`）和 `compute_ape.py`（以 (short_name, grid, block) 为键计算 APE，输出 JSON） | AC-2,3,4,5 | coding | task1 |
| task5 | 在 baseline 配置下运行全量 trace 模拟，解析输出，计算并保存 baseline APE 表（6 kernel × 5 指标 + elapsed_cycles 辅助列）至 `results/mini_transformer_v4/baseline_ape.json` | AC-2,3,4,5 | coding | task3, task4 |
| task6 | 在 rx_C1 配置（registers 32768）下重跑模拟，与 baseline 比较 `achieved_occupancy_pct` APE delta；验证 gemm_tiled + attention_score 方向性下降，residual_add 对照 < 50% 变化；记录 C-1 两层判决结论 | AC-3 | coding | task5 |
| task7 | 在 rx_C2 配置（n_mem 12）下重跑模拟，与 baseline 比较 `dram_throughput_pct` APE delta；标注"DRAM+L2 联合压力测试（n_mem 同时改变 L2 切片数量）"；记录 C-2 两层判决结论 | AC-4 | coding | task5 |
| task8 | 在 rx_C3 配置（dl2 S:256）下重跑模拟，与 baseline 比较 `l2_hit_rate_pct`（= 1−L2_total_cache_miss_rate）+ `dram_throughput_pct` APE delta；标注"冷启动已知限制（flush=1）"；记录 C-3 两层判决结论 | AC-5 | coding | task5 |
| task9 | 汇总 task6/7/8 结论，生成 `results/mini_transformer_v4/E5_stageC_validation.md`：baseline APE 表 + 三处方敏感性对比表 + 两层判决 + 失败归因分类 + 已知限制声明 | AC-6 | coding | task6, task7, task8 |

## Claude-Codex Deliberation

### Agreements
- C-1 的正确框架是"一致性检查 + 敏感性验证"，而非"发现偏差再校准"（baseline 已使用真实 HW 值 65536）
- C-3 主指标为 `l2_hit_rate_pct`（由 GPGPU-Sim 输出的 `L2_total_cache_miss_rate` 推导），而非 `l1_hit_rate_pct`
- `n_mem` 变化同时影响 L2 切片数量，C-2 与 C-3 参数层面不独立，报告中必须承认
- 术语修正：RTX 3080 Ti 使用 GDDR6X，文档中统一使用"DRAM 子系统"而非"HBM"
- 数据质量门控（task1）是所有 APE 计算的强前置条件
- control kernel 阈值使用相对量（< 目标 kernel 变化的 50%），不使用固定 < 2%
- 比较键为 `(kernel_short_name, grid_size, block_size)`，不能仅按 family 名称聚合（`gemm_tiled` 存在多种 launch shape）

### Resolved Disagreements
- **AC-1 触发逻辑**（Round 1 解决）：原设计用数值不等式触发 halt，Codex 指出 compute_throughput = mem_pipes_busy 是 CSV 原始数据特性而非解析 bug。修正为溯源审计：独立行验证 + METRIC_MAP 碰撞检查 + 比较键完整性。
- **C-3 指标端到端对齐**（Round 2 解决）：验证 GPGPU-Sim `gpu-sim.cc` 输出 `L2_total_cache_miss_rate`，可推导 l2_hit_rate，与 AC-5 要求一致。
- **下界与 AC 语言一致性**（Round 1 解决）：原计划声明单调性但仅做单点测试，存在矛盾。修正为：单点 perturbation + "方向一致性"声明，与下界设计对齐。
- **旧版直通逻辑**（Round 2 解决）：移除"baseline APE < 10% → C-1 直接通过"的捷径，替换为两层判决（baseline 准确性与处方因果性分开评定）。

### Convergence Status
- Final Status: `partially_converged`（Round 1/2 完成，所有 REQUIRED_CHANGES 已在计划中解决；DEC-1～DEC-4 已由用户明确决策）

## Pending User Decisions

- DEC-1: 参数 sweep 设计（3 点 vs 单点）
  - Claude Position: 单点 perturbation 足以支撑"方向一致性"声明
  - Codex Position: 单点无法声明单调性
  - Tradeoff Summary: 单点实验量最小但结论有限；3 点可声明单调性但实验量约 3×
  - Decision Status: **RESOLVED — 单点 perturbation，声明方向性，不声明单调性**

- DEC-2: `elapsed_cycles` 作为主要/辅助指标
  - Claude Position: 辅助诊断指标，主指标为处方特定字段
  - Codex Position: 辅助即可，需要精确 launch 匹配后再升为主要
  - Tradeoff Summary: 作为主要指标可发现补偿性误差，但依赖精确 launch 匹配
  - Decision Status: **RESOLVED — 辅助诊断指标，在 E5 APE 表中作为附加列**

- DEC-3: 缓存冷/暖态语义（flush=1 vs flush=0）
  - Claude Position: 保持冷启动，E5 中注明已知限制
  - Codex Position: 禁用 flush 可能对齐暖态，但引入其他复杂性
  - Tradeoff Summary: 冷启动实验更干净；禁用 flush 复杂度高且可能引入新偏差
  - Decision Status: **RESOLVED — 保持 flush=1，E5 中注明"冷启动已知限制"**

- DEC-4: C-2 参数耦合处理方式
  - Claude Position: 接受耦合测试，报告中明确标注为"DRAM+L2 联合压力测试"
  - Codex Position: 耦合可接受，前提是报告明确标注
  - Tradeoff Summary: 接受耦合省时但结论粒度受限；隔离测试工作量约 2-3×
  - Decision Status: **RESOLVED — 接受耦合，结论带免责标注**

## Implementation Notes

### Code Style Requirements
- 实现代码和注释中不得包含计划特定术语，如 "AC-"、"Milestone"、"task"、"DEC-" 等工作流标记
- 这些术语仅用于计划文档，不应出现在生成的代码库中
- 代码中使用领域相关命名（如 `validate_ncu_provenance()`、`compute_kernel_ape()`，而非 `task1_ac1_validate()`）

--- Original Design Draft Start ---

# Stage C 闭环验证设计文档

**日期：** 2026-04-12
**目标：** 验证 mini-transformer v4 的三条 Stage C 架构处方（C-1/C-2/C-3）
**硬件：** RTX 3080 Ti (SM_86)
**模拟器：** GPGPU-Sim 4.2，SM86_RTX3080_TI 配置

---

## 背景

mini-transformer v4 完成软件层清洗后，Delta 机制发现
`block_limit_registers` 是中心约束字段，生成了三条 Stage C 处方：

| 处方 | 目标参数 | 目标 Kernel | 置信度 |
|------|---------|------------|:------:|
| C-1 | `gpgpu_shader_registers` | gemm_tiled、attention_score | HIGH |
| C-2 | `gpgpu_n_mem` + HBM 时序 | residual_add | HIGH |
| C-3 | `gpgpu_cache:dl2` | softmax | MEDIUM |

本文档设计如何在模拟器上验证这三条处方。

---

## 整体流程

```
Step 1：录制 trace（RTX 3080 Ti）
  编译 mini_transformer_v4
  NVBit tracer 录制完整运行
  提取 6 个代表 kernel 的 trace 文件

        ↓

Step 2：基准模拟（Baseline）
  SM86_RTX3080_TI 默认配置跑 6 个 kernel
  收集模拟器输出指标
  对比 NCU 实测数据，计算 baseline APE

        ↓

Step 3：处方验证（逐一修改参数）
  C-1：调整 gpgpu_shader_registers
  C-2：调整 gpgpu_n_mem
  C-3：调整 gpgpu_cache:dl2

        ↓

Step 4：汇总报告（E5_stageC_validation.md）
```

---

## 6 个代表 Kernel（来自 Batch 输出）

| 来源 | Kernel | 代表什么 |
|------|--------|---------|
| 聚类 A 代表 | gemm_tiled_1 | 计算密集类（7 次 launch）|
| 聚类 B 代表 | residual_add_9 | HBM 流式类（2 次 launch）|
| 聚类 C 代表 | layernorm_10 | 混合归约类（2 次 launch）|
| Outlier | attention_score | 高 shmem 计算密集 |
| Outlier | softmax_kernel | L2 溢出混合访存 |
| Outlier | context_mul | L1 驻留计算 |

---

## 测量指标

每个 kernel 对比以下 5 个指标：

| 指标 | 模拟器字段 | 对应处方 |
|------|----------|---------|
| `achieved_occupancy_pct` | `gpu_occ` | C-1 |
| `compute_throughput_pct` | `gpu_ipc` 换算 | C-1 |
| `dram_throughput_pct` | `dram_bw_util` | C-2 |
| `warp_cycles_per_issued_inst` | `gpu_ipc` 换算 | C-2/C-3 |
| `l1_hit_rate_pct` | `L1D_total_cache_hit_rate` | C-3 |

APE 计算：
```
APE = |模拟器值 - NCU实测值| / NCU实测值 × 100%
```

---

## 处方验证设计

### C-1：寄存器文件配置

**当前值：** `gpgpu_shader_registers 65536`

**验证逻辑：**
- 目标 kernel：gemm_tiled、attention_score
- 关键指标：occupancy、compute_throughput
- 对照 kernel：residual_add（APE 变化应 < 2%）

**参数调整方向：**
baseline 配置已是 65536（SM_86 真实值），
验证目的是确认模拟器能否正确预测 block_limit_registers=6 带来的 occupancy 限制。
若 baseline APE < 10% 则 C-1 直接通过；
若 APE > 10% 则需要同时检查 `trace_opcode_latency_initiation_sp`。

### C-2：HBM 带宽模型

**当前值：** `gpgpu_n_mem 24`

**验证逻辑：**
- 目标 kernel：residual_add
- 关键指标：dram_throughput、warp_cycles
- 对照 kernel：gemm_tiled（APE 变化应 < 2%）

**参数调整方向：**
若 baseline dram_throughput APE > 10%，
尝试调整 `gpgpu_n_mem`（±4）或 HBM 时序参数（tCL、tRCD）。

### C-3：L2 Cache 容量

**当前值：** `gpgpu_cache:dl2 S:64:128:16,L:B:m:L:P,A:192:96,32:0,32`

**验证逻辑：**
- 目标 kernel：softmax
- 关键指标：dram_throughput、l1_hit_rate
- 对照 kernel：context_mul（APE 变化应 < 2%）

**参数调整方向：**
softmax working set = 12MB > L2 6MB，
若模拟器 L2 配置过大会低估 DRAM 利用率。
尝试调整 L2 大小参数，观察 dram_throughput APE 变化。

---

## 成功标准

**处方有效判定（同时满足）：**
1. 目标 kernel 关键指标 APE 下降
2. 对照 kernel APE 变化 < 2%
3. APE 变化量 > 5%（超过测量噪声）

**APE 分级：**

| APE | 含义 |
|-----|------|
| < 10% | 模拟器建模准确 |
| 10% ~ 30% | 有偏差，处方提供改善方向 |
| > 30% | 建模存在根本性问题 |

**整体判定：**
- C-1/C-2（HIGH）：APE 必须下降，否则重新分析根因
- C-3（MEDIUM）：APE 下降即成功；无变化记录为"模拟器限制"

---

## 输出文件结构

```
results/mini_transformer_v4/
├── E5_stageC_validation.md
└── configs/
    ├── baseline/
    │   └── trace.config + gpgpusim.config
    ├── rx_C1/
    │   └── gpgpusim.config（调整 shader_registers）
    ├── rx_C2/
    │   └── gpgpusim.config（调整 n_mem）
    └── rx_C3/
        └── gpgpusim.config（调整 dl2）
```

---

## 实施前置条件

1. RTX 3080 Ti 可连接（已确认）
2. NVBit tracer 已配置（已确认）
3. mini_transformer_v4 binary 存在（已确认）
4. GPGPU-Sim 编译正常（参考 backprop 闭环验证先例）

--- Original Design Draft End ---

---

## BitLesson Selection (REQUIRED FOR EACH TASK)

Before executing each task or sub-task, you MUST:

1. Read @/home/dyf/modern-gpu-simulator-micro-2025/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task to select relevant lesson IDs
3. Follow the selected lesson IDs (or `NONE`) during implementation

Include a `## BitLesson Delta` section in your summary with:
- Action: none|add|update
- Lesson ID(s): NONE or comma-separated IDs
- Notes: what changed and why (required if action is add or update)

Reference: @/home/dyf/modern-gpu-simulator-micro-2025/.humanize/bitlesson.md

---

## Goal Tracker Rules

Throughout your work, you MUST maintain the Goal Tracker:

1. **Before starting a task**: Mark it as "in_progress" in Active Tasks
   - Confirm Tag/Owner routing is correct before execution
2. **After completing a task**: Move it to "Completed and Verified" with evidence (but mark as "pending verification")
3. **If you discover the plan has errors**:
   - Do NOT silently change direction
   - Add entry to "Plan Evolution Log" with justification
   - Explain how the change still serves the Ultimate Goal
4. **If you need to defer a task**:
   - Move it to "Explicitly Deferred" section
   - Provide strong justification
   - Explain impact on Acceptance Criteria
5. **If you discover new issues**: Add to "Open Issues" table

---

Note: You MUST NOT try to exit `start-rlcr-loop` loop by lying or edit loop state file or try to execute `cancel-rlcr-loop`

After completing the work, please:
0. If you have access to the `code-simplifier` agent, use it to review and optimize the code you just wrote
1. Finalize @/home/dyf/modern-gpu-simulator-micro-2025/.humanize/rlcr/2026-04-12_02-31-41/goal-tracker.md (this is Round 0, so you are initializing it - see "Goal Tracker Setup" above)
2. Commit your changes with a descriptive commit message
3. Write your work summary into @/home/dyf/modern-gpu-simulator-micro-2025/.humanize/rlcr/2026-04-12_02-31-41/round-0-summary.md
