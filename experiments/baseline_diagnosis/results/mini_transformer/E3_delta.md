# 诊断报告：mini-transformer [E3_delta]

**日期：** 2026-04-10
**硬件：** RTX 3080 Ti (SM_86)
**输入特征：** mini_transformer_full.json + mechanisms/delta.json
**启用机制：** 仅 Delta

---

## Delta 输出摘要

### Kernel 级字段温度

**HOT 字段**（跨 kernel 方差高）：

| 字段 | 温度 |
|------|------|
| l1_hit_rate_pct | 0.782 |
| total_dynamic_instructions | 0.826 |
| num_tbs | 0.858 |
| waves_per_sm | 0.887 |
| dram_throughput_pct | 0.665 |
| mem_pipes_busy_pct | 0.501 |
| compute_throughput_pct | 0.501 |
| ipc_active | 0.428 |
| l2_throughput_pct | 0.328 |
| l1_throughput_pct | 0.326 |
| achieved_occupancy_pct | 0.228 |

**COLD 字段**（所有 kernel 近零方差）：

| 字段 | 温度 | 含义 |
|------|------|------|
| uses_fp64 | 0.0 | 无 kernel 使用 FP64 — 正确 |
| num_barriers | 0.0 | Barrier 数量无法区分不同 kernel |
| total_static_instructions | 0.0 | 静态指令数在同类 kernel 中为常量 |
| uses_shared_memory | 0.0 | **特征提取 bug，见发现 3** |

### Kernel 级字段相关性（显著对）

| 字段对 | 相关系数 | 解读 |
|--------|---------|------|
| mem_pipes_busy_pct ↔ compute_throughput_pct | **+1.000** | 完全正相关 |
| l1_hit_rate_pct ↔ compute_throughput_pct | **-0.646** | 反相关 |
| l1_hit_rate_pct ↔ mem_pipes_busy_pct | **-0.646** | 反相关 |
| ipc_active ↔ compute_throughput_pct | **+0.793** | 正相关 |
| ipc_active ↔ mem_pipes_busy_pct | **+0.793** | 正相关 |
| l1_throughput_pct ↔ dram_throughput_pct | **-0.940** | 强反相关 |
| num_tbs ↔ waves_per_sm | **+0.922** | 近恒等 |
| achieved_occupancy_pct ↔ mem_pipes_busy_pct | **+0.634** | 正相关 |

### TB 级：所有 kernel 报告 0 个 HOT 字段

全部 14 个 kernel 的 TB 级字段温度均为 0，所有字段均为 COLD。
与 Squash / Batch 的结论一致：每个 kernel 内部完全均匀，TB 级无方差。

---

## Delta 相对 E0 的增量贡献

### 发现 1：l1_hit_rate ↔ compute_throughput 反相关 — 跨 kernel 规律（发现性）

Delta 报告这两个字段在 14 个 kernel 之间的相关系数为 **-0.646**。

**这是本数据集中最反直觉的发现。** 直觉上：L1 命中率越高（数据取出越快）→
计算吞吐越好。Delta 揭示的是相反规律。

**为什么？** L1 命中率最高的 kernel 是 attention_score（97.2%）和 context_mul（88.5%）。
它们的计算吞吐分别为 22.4% 和 89.4%。而 GEMM 的 L1 命中率仅 2.4%，
计算吞吐却高达 90.3%。

反相关的成因：
- **attention_score** 的 97.2% L1 命中率来自高度重复的访存模式（同一 Q 行
  被多列 K 反复复用），但内层循环串行化（RAW 依赖链）阻止了计算流水线
  消化已就绪的数据。
- **GEMM** 几乎没有 L1 命中（每个 tile 只从全局内存加载一次，之后通过
  shared memory 复用），但 tile 内计算高度并行。

**Delta 将此转化为可操作洞察：** L1 命中率高但计算吞吐低的 kernel 是
**计算串行化**瓶颈的候选者，而非内存瓶颈。E0 可以观察到 attention_score 的
这一现象，但无法将其系统化为跨 kernel 规律。Delta 使这一规律显式可见。

### 发现 2：l1_throughput ↔ dram_throughput 强反相关（-0.940）— 内存系统路由信号（发现性）

L1 吞吐与 DRAM 吞吐之间的强反相关（-0.940）揭示了整个 workload 的内存系统使用分裂：

- **L1 驻留型**（attention_score、context_mul、layernorm）：数据在 L1 附近，DRAM 几乎不涉及。
- **HBM 流式型**（residual_add：DRAM 60.4%，L1 吞吐低）：流式访问模式，无 L1 复用。
- **GEMM**：L1 命中率低（2.4%），DRAM 中等（15.6%）——数据在 L2 中（tile 加载一次后
  全部计算在 shared memory 和寄存器中完成）。

这三种访存机制（L1 驻留 / L2 驻留 / HBM 流式）从任何单一 kernel 的统计数据中
都看不出来，而是从 Delta 的相关分析中涌现。它告诉架构师：mini-transformer 存在
三种结构性不同的内存访问模式，优化策略必须针对每种模式单独设计。

### 发现 3：特征提取 bug — uses_shared_memory 始终为 COLD（系统信号）【已修复】

Delta 报告原始 `uses_shared_memory` 为 COLD 字段（温度 = 0.0，意味着所有 kernel 值相同）。
然而：
- gemm_tiled 每个 block 使用 2048 字节的**静态** shared memory（tile buffer）
- layernorm 使用 dynamic shared memory 存储并行归约的中间结果

**根本原因：** `extract_per_tb_features.py` 仅扫描 trace opcode 中的 LDS/STS 指令，
但 mini_transformer 的 trace 中 `top_opcodes` 为空，导致 `uses_shared_memory` 恒为 False。
同时，`build_kernel_summary` 未读取 `hardware_metrics.static_shmem_per_block` 字段。

**修复（已应用）：** `extract_per_tb_features.py` 中：
1. `uses_shared_memory` 改为同时检查 LDS/STS opcode 和 `hardware_metrics` 中的
   `static_shmem_per_block` / `dynamic_shmem_per_block`
2. `build_kernel_summary` 增加将 `hardware_metrics` 关键字段（14 个）纳入 kernel summary，
   使 Delta 可以在硬件指标维度上计算跨 kernel 方差和相关性

**修复后效果：**
- `uses_shared_memory` 温度从 0.0 → **0.632**（HOT）
- `static_shmem_per_block` 成为新 HOT 字段（温度 0.762）
- 新相关：`static_shmem_per_block ↔ l1_hit_rate = -0.890`（shmem 使用减少 L1 缓存压力，
  与内存三态分解结论一致）

---

## 处方（Delta 信息增强）

| ID | Kernel | Delta 贡献 | 置信度 |
|----|--------|-----------|-------|
| B-1 | attention_score | l1_hit ↔ compute 反相关规律确认：非内存受限，是计算串行化 | HIGH |
| A-1 | softmax | Delta 无信号（softmax 是 L1 命中空间中的单点离群值） | HIGH |
| B-2 | gemm_tiled | 内存路由：GEMM 驻留在 L2（15.6% DRAM），瓶颈在计算流水线而非内存带宽 | MEDIUM |
| 修复 | 特征提取脚本 | uses_shared_memory bug 通过 cold-field 分析发现 | HIGH（正确性修复） |

---

## 机制在本 Workload 上的判定

**Delta 在 mini-transformer 上提供了最强的新颖贡献。**

三项独立发现：
1. `l1_hit ↔ compute` 反相关是 E0 无法系统化呈现的跨 kernel 规律。
   它改变了对 attention_score 的诊断框架：从"这个 kernel 有某种问题"转变为
   "L1 命中高 + 计算吞吐低是由串行化引起的系统性规律，而非内存问题"。
2. `l1_throughput ↔ dram_throughput` 强反相关揭示了三种内存访问机制，
   指导了针对不同 kernel 类别的差异化优化策略。
3. `uses_shared_memory` cold-field 发现捕获了一个特征提取 bug，
   该 bug 会导致未来所有涉及 shared memory kernel 的诊断出现偏差。

在这个 14-kernel workload 上，Delta 产出了 **2 个不可从 E0 推导的跨 kernel 洞察**
和 **1 个系统级正确性问题**。
