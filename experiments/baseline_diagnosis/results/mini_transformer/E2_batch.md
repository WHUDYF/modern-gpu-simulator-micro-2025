# 诊断报告：mini-transformer [E2_batch]

**日期：** 2026-04-10
**硬件：** RTX 3080 Ti (SM_86)
**输入特征：** mini_transformer_full.json + mechanisms/batch.json
**启用机制：** 仅 Batch

---

## Batch 输出摘要

### Kernel 级聚类

| 聚类 | 大小 | Kernel ID | Kernel 名称 | 凝聚度 |
|------|------|----------|------------|-------|
| 0 | 7（50%） | k1,k2,k3,k4,k8,k11,k12 | gemm_tiled ×7 | 1.000 |
| 1 | 2（14%） | k9, k13 | residual_add ×2 | 1.000 |
| 2 | 2（14%） | k10, k14 | layernorm ×2 | 1.000 |
| outliers | 3（21%） | k5, k6, k7 | attention_score, softmax, context_mul | — |

### TB 级：所有 kernel 内部形成单一聚类（均匀）

所有 kernel 的 TB 均聚类为一组，凝聚度 = 1.0，与 Squash 的 TB 级 0 边界结论一致。

---

## Batch 相对 E0 的增量贡献

### 增量 1：结构性 outlier 的机器化标注（发现性）

3 个 outlier kernel——attention_score（k05）、softmax（k06）、context_mul（k07）——
是唯一无法归入任何聚类的 kernel。7 个 GEMM 实例、2 个 residual_add、2 个 layernorm
全部以凝聚度 = 1.0 完美聚类。

**这是相对 E0 的真实发现。** E0 面对 14 行统计数据，需人工逐行对比才能找出异常。
Batch 产出明确的机器信号：

> "3 个 kernel 是结构性异常。它们无法归入任何聚类。优先将这 3 个 kernel 作为瓶颈候选检查。"

这是可直接操作的优先级排序。分析者不再需要均等审查全部 14 个 kernel，
而是直接聚焦于 3 个特定 kernel。

### 增量 2：GEMM 聚类凝聚度 = 1.0（确认性 + 有用）

7 个 GEMM 实例（4 个 QKV 投影、1 个输出投影、2 个 FFN GEMM）形成凝聚度 = 1.0 的单一聚类。
这机器化地确认：
- 7 个 GEMM 启动的行为特征完全一致
- 处方 B-2（TILE_SIZE 增大）均等适用于全部 7 个实例
- 无任何 GEMM 变体需要特殊处理

在包含更多 GEMM 变体的大型 workload 中（如高矩阵 vs 方形矩阵），
Batch 会区分行为不同的 GEMM 实例，收窄处方范围。

### 增量 3：outlier 分析揭示结构性缺口

3 个 outlier 并非随机——它们都是**非 GEMM 的注意力机制 kernel**：
- k05（attention_score）：每个 head 计算 N×N 分数矩阵，内层循环不规则
- k06（softmax）：逐行归一化，存在逐行串行依赖
- k07（context_mul）：每个 head 对 V 的加权求和，结构类似 attention_score

三者共同特征：都涉及**在线程内对 seq_len 或 head_dim 进行迭代**，
没有 shared memory tiling。GEMM 聚类通过 tile 策略高效处理此类计算，
而这三个 kernel 均未采用。Batch 将这一结构性缺口机器化呈现，
无需依赖 Squash 或 Delta。

---

## 处方（Batch 信息增强）

| ID | Kernel | Batch 贡献 | 置信度 |
|----|--------|-----------|-------|
| A-1 | softmax | Outlier 标注触发优先检查 → 确认 Class A 缺陷 | HIGH |
| B-1 | attention_score | Outlier 标注 → 最高优先级 B 类调查 | HIGH |
| B-3（新） | context_mul | Outlier 标注触发检查：ipc=1.37，compute=89.4% → 实为近最优，无需优化 | LOW（不需要操作）|
| B-2 | gemm_tiled | 均匀聚类 → 一个处方覆盖全部 7 个实例 | MEDIUM |

### Batch 新发现：context_mul 是"良性 outlier"

context_mul 按聚类分配属于 outlier（与 GEMM 行为特征不同），
但实际性能指标良好（compute=89.4%，ipc=1.37）。Batch 的 outlier 标注
触发了对它的检查，检查结论：无需修改。

这是重要规律：**outlier ≠ 瓶颈**，但所有 outlier 均应被检查。
Batch 确保了 context_mul 不被漏查，同时也避免了误诊。

---

## 机制在本 Workload 上的判定

**Batch 在 mini-transformer 上提供了真实的发现性价值。**

对 3 个 outlier kernel（attention_score、softmax、context_mul）的显式标注，
是 E0 无法在不手动逐行对比的情况下产出的机器信号。即使两个 outlier 的
处方与 E0 结论相同，Batch 提供的**优先级排序**和**分组**也降低了
随 kernel 数量增长的诊断成本。

在 6 层 Transformer（~84 个 kernel launch）的情境下，Batch 将检查集合
从 84 个缩减至 ≤ 5 种 outlier 类型，这正是此方法规模化的核心机制。
