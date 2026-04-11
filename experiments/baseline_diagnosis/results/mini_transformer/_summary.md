# Phase 4 消融实验总结：mini-transformer + 跨 Dwarf 对比

**日期：** 2026-04-10
**Workload：** mini-transformer（手写 CUDA，1 层，hidden=768，seq=512，heads=12）
**硬件：** RTX 3080 Ti (SM_86)

---

## mini-transformer 每实验总结

| 实验 | Squash | Batch | Delta | Stage A | Class A 处方 | Class B 处方 | 新发现 vs E0 |
|------|--------|-------|-------|---------|-------------|-------------|-------------|
| E0 baseline | ❌ | ❌ | ❌ | 1 不通过 | 1（softmax） | 2（attention_score, GEMM） | — |
| E1 squash | ✅ | ❌ | ❌ | 1 不通过 | 1 | 2 | 确认性（段 1 凝聚度=0.85，注意力阶段异质） |
| E2 batch | ❌ | ✅ | ❌ | 1 不通过 | 1 | 2 | **发现性：3 个 outlier kernel 机器化标注** |
| E3 delta | ❌ | ❌ | ✅ | 1 不通过 | 1 | 2 | **发现性：l1_hit↔compute 反相关 + 内存三态 + 特征提取 bug** |
| E4 full | ✅ | ✅ | ✅ | 1 不通过 | 1 | 2 | 三机制收敛，最高置信度 |

---

## 每机制贡献分析

### Squash (E1)

**产出：**
- 8 段分解，精确映射 Transformer 层结构（QKV / 注意力分数 / 注意力输出 / 残差 / LN / FFN / 残差 / LN）
- 段 1 凝聚度 = 0.850（最低）：attention_score 和 softmax 行为差异最大
- TB 级：所有 kernel 0 边界，证明所有 kernel 内部完全均匀

**判定：确认性（Confirming）**

Squash 产出的相位分解正确，但对人类读者而言从 kernel 名字即可推断。
段 1 低凝聚度是有意义的信号（提示将 attention_score 和 softmax 当作两个
独立问题处理），但 E0 通过逐行对比也能得出相同结论。

Squash 价值预计在以下场景显著增加：
- Kernel 名称被混淆（编译器内联、匿名 lambda）
- 多层 Transformer 中大量重复 kernel 的阶段需要机器化区分
- 存在 TB 级不均匀的 workload（如稀疏算法的边界 TB）

### Batch (E2)

**产出：**
- 4 个聚类：gemm_tiled ×7（凝聚度 1.0）、residual_add ×2（1.0）、layernorm ×2（1.0）
- 3 个 outlier：attention_score、softmax、context_mul
- TB 级：所有 kernel 单一聚类

**判定：发现性（Discovering）**

Batch 提供了 E0 无法廉价推导的价值：**显式 outlier 标注**。

E0 面对 14 行统计数据，需要人工对比找出异常。Batch 产出：
> "这 3 个 kernel 是结构性异常，其余 11 个 kernel 分属 3 个完全均匀的聚类。"

这是直接可操作的诊断优先级排序。在 6 层 Transformer（~84 个 launch）中，
Batch 将扫描成本从 O(N) 降至 O(outlier 类型数)。

Batch 还识别了 context_mul 是"良性 outlier"——它是 outlier 但性能实际良好
（compute=89.4%）。这一区分（outlier ≠ 瓶颈）避免了误诊。

### Delta (E3)

**产出：**
- HOT：l1_hit_rate（0.782）、waves_per_sm（0.887）、dram_throughput（0.665）等
- COLD：uses_fp64（0.0）、num_barriers（0.0）、**uses_shared_memory（0.0，bug）**
- 关键相关：l1_hit_rate ↔ compute_throughput 反相关（-0.646）
- 强相关：l1_throughput ↔ dram_throughput 反相关（-0.940）
- 一致相关：mem_pipes_busy ↔ compute_throughput 正相关（+1.000）

**判定：发现性（Discovering），贡献最大**

三个独立新发现：

1. **l1_hit ↔ compute 反相关** 是本 workload 中最反直觉的跨 kernel 规律。E0
   可以逐行看到 attention_score（L1 命中 97.2%，compute 22.4%），但不能把它
   系统化为：*在这个 workload 中，L1 命中率高的 kernel 计算吞吐反而更低*。
   这个规律直接指向根因：数据已就绪，但计算流水线无法消化（串行化）。

2. **内存三态分解** 揭示了 mini-transformer 的三种访存机制：L1 驻留（注意力类）、
   L2 驻留（GEMM，通过 shared memory tiling）、HBM 流式（residual_add）。
   这三种机制需要不同的模拟器精度配置，也指向不同的优化策略。

3. **uses_shared_memory = 0（特征提取 bug）** 被 Delta 的 cold-field 分析捕获。
   GEMM 和 layernorm 均使用了 static shared memory，但特征记录为 0。
   此 bug 会导致未来对 GEMM warp stall 的误诊。

---

## 关键观察：mini-transformer vs backprop vs nn

| 属性 | backprop | nn | mini-transformer |
|------|----------|-----|-----------------|
| Kernel 数量 | 2 | 4（1 种重复） | 14（6 种不同） |
| Squash 价值 | 确认 | 确认 | 确认 |
| Batch 价值 | 确认（同 E0） | 发现（Delta 为主） | **发现（3 个 outlier）** |
| Delta 价值 | 确认 | 发现（cold=6 字段） | **发现（反相关 + bug）** |
| 最有价值的机制 | 无 | Delta | **Delta ≥ Batch** |

**跨 Dwarf 规律：**

1. **Squash 在所有测试 workload 上均为确认性。** 它产出正确且非平凡的输出，
   但没有一个案例中 Squash 找到了 E0 完全漏掉的新瓶颈。这符合预期：Squash
   设计用于时序相变检测，在我们测试的三个 workload 中，相变均可从 kernel
   名称推断。TB 级边界持续为 0，说明三个 workload 都没有内部不均匀的 kernel。

2. **Batch 的价值随 kernel 多样性增加而增加。** backprop（2 种 kernel）→ 无新发现；
   nn（1 种 × 4 次）→ 发现"零多样性"；mini-transformer（6 种 × 14 次）→
   发现"3 种 outlier 类型"。趋势符合预期。

3. **Delta 在每个 workload 上均产出了 E0 不可推导的洞察。** backprop（FP64 HOT 字段）、
   nn（zero-diversity cold 字段）、mini-transformer（反相关 + 内存三态 + bug 检测）。
   Delta 是三个机制中最持续有价值的。

---

## 发现的 Bug（已修复）

### uses_shared_memory 特征提取错误【已修复 2026-04-10】

**位置：** `experiments/baseline_diagnosis/mechanisms/extract_per_tb_features.py`

**问题：** `uses_shared_memory` 仅扫描 trace opcode 中的 LDS/STS 指令，
而 mini_transformer 的 `top_opcodes` 为空，同时未读取 `hardware_metrics.static_shmem_per_block`。

**修复内容：**
1. `uses_shared_memory` 检测逻辑增加对 `hardware_metrics.static_shmem_per_block`
   和 `dynamic_shmem_per_block` 的检查（两者取 OR）
2. `build_kernel_summary` 新增将 14 个 `hardware_metrics` 关键字段纳入 kernel summary，
   使 Delta 可以在硬件指标维度上计算跨 kernel 方差和相关性

**修复后结果：**
- gemm_tiled、layernorm 的 `uses_shared_memory` 正确为 True
- Delta 重新生成后 `uses_shared_memory` 温度 0.0 → 0.632（HOT）
- `per_tb.json` 和 `delta.json` 已重新生成

---

## 处方汇总（mini-transformer）

| ID | Kernel | 类别 | 处方内容 | 置信度 | 收敛机制 |
|----|--------|------|---------|--------|---------|
| A-1 | softmax | A | 每行一个 block，grid=6144 | **HIGH** | E0+Batch+Squash |
| B-1 | attention_score | B | 4-wide 累加器展开 | **HIGH** | E0+Squash+Batch+Delta |
| B-2 | gemm_tiled | B | TILE_SIZE 16→32 | MEDIUM | E0+Batch+Delta |
| 修复 | 特征提取脚本 | 系统 | uses_shared_memory = static + dynamic | HIGH | Delta |
| — | residual_add | — | 与 layernorm 融合（可选） | LOW | E0 |

---

## Checkpoint 3 分析

mini-transformer 是目前测试的三个 workload 中结构最复杂的
（6 种不同 kernel，14 次 launch，明确的计算/访存多样性）。

**机制表现：**
- Squash：持续确认，无独立发现。在 Transformer 这类具有显式阶段结构的 workload 上，
  Squash 的 kernel 级价值较小。如需 Squash 提供独立价值，应在 kernel 名称不可读
  或 kernel 内部存在阶段变化的 workload 上测试。
- Batch：首次在多 kernel 场景下产出具有规模优势的发现。随 kernel 数量增加，
  Batch 的 outlier prioritization 价值指数增长。
- Delta：在所有三个 workload 上均提供了 E0 不可推导的跨 kernel 洞察。
  是三个机制中最持续有价值的。

**决定：进行闭环验证（处方 B-1 和 A-1）**

下一步推荐（按优先级）：

1. ~~**修复 uses_shared_memory 特征提取 bug**~~ **已完成（2026-04-10）**
   - `extract_per_tb_features.py` 已修复，`per_tb.json` 和 `delta.json` 已重新生成
2. **在 RTX 3080 Ti 上采集修复后 binary 的 NCU 数据**（attention_score 4-wide 累加器 + softmax 新 launch 配置）
   — 验证处方 B-1 和 A-1 的实际硬件效果
3. **对修复后的 NCU 数据重新运行 E0-E4 消融**，产出干净 workload 的全套报告
4. **生成跨 dwarf 总结报告**（backprop + nn + mini-transformer 三个 workload
   的处方成功率表），作为论文主表的原始数据

---

## 产物清单

- `E0_baseline.md` — 无机制基线诊断
- `E1_squash.md` — Squash 机制诊断
- `E2_batch.md` — Batch 机制诊断
- `E3_delta.md` — Delta 机制诊断
- `E4_full.md` — 三机制完整诊断
- `_summary.md` — 本文件（跨实验分析 + Checkpoint 3）
- `../../mini_transformer/` — 原始数据（.cu, .json, .csv）
