# 诊断报告：mini-transformer [E1_squash]

**日期：** 2026-04-10
**硬件：** RTX 3080 Ti (SM_86)
**输入特征：** mini_transformer_full.json + mechanisms/squash.json
**启用机制：** 仅 Squash

---

## Squash 输出摘要

### Kernel 级分段（8 段，7 个边界）

| 段 | Kernel 范围 | Kernel 名称 | 凝聚度 | 阶段解读 |
|----|------------|------------|-------|---------|
| 0 | k1–k4 | gemm_tiled ×4 | 1.000 | QKV 线性投影（Q, K, V + warmup） |
| 1 | k5–k6 | attention_score + softmax | 0.850 | 注意力分数计算 |
| 2 | k7–k8 | context_mul + gemm_tiled | 0.904 | 注意力输出（加权求和 + 输出投影） |
| 3 | k9 | residual_add | 1.000 | 注意力残差连接 |
| 4 | k10 | layernorm | 1.000 | 注意力后层归一化 |
| 5 | k11–k12 | gemm_tiled ×2 | 1.000 | FFN 第一层 + 第二层 |
| 6 | k13 | residual_add | 1.000 | FFN 残差连接 |
| 7 | k14 | layernorm | 1.000 | FFN 后层归一化 |

### TB 级：所有 kernel 边界数均为 0

每个 kernel 的所有 TB 折叠为单一段，凝聚度 ≈ 1.0。
这意味着没有任何 kernel 存在内部子阶段——同一 kernel 内所有 TB 行为完全一致。

---

## Squash 相对 E0 的增量贡献

### 增量 1：Transformer 计算结构得到机器化确认（确认性）

8 段分解精确对应标准 Transformer 层结构：
- **段 0**：QKV 投影
- **段 1**：分数计算（行为最异质的一段）
- **段 2**：注意力输出
- **段 3–4**：注意力后残差 + 归一化
- **段 5**：FFN
- **段 6–7**：FFN 后残差 + 归一化

E0 可从 kernel 名称推断此结构，但 Squash 通过行为相似度而非命名启发式进行机器化确认。
在 kernel 名称被混淆或函数被内联的场景下，Squash 仍能恢复此结构。

### 增量 2：段 1 凝聚度 = 0.850 — attention_score 与 softmax 行为差异显著（确认性 + 有用）

所有其他段的凝聚度 ≥ 0.904，段 1（attention_score + softmax）凝聚度最低（0.850），
尽管两者在执行顺序上相邻。

**解读：** Squash 无法将这两个 kernel 归为同一段，因为它们有不同的行为特征——
attention_score 是计算密集型且 L1 命中率高，softmax 则严重利用不足。
低凝聚度作为机器产出信号表明：**这两个 kernel 应作为独立瓶颈分别诊断，
而不是当作统一的"注意力计算"瓶颈**。

在 E0 中，分析者需手动对比两行数据才能发现差异；Squash 将其转化为机器信号：
凝聚度低于阈值 → 分开处理。

### 增量 3：所有 kernel 的 TB 级均匀性（否定性验证）

所有 kernel 边界数 = 0，确认同一 kernel 内所有 TB 执行完全相同的指令流。这排除了：
- 边界条件 TB 导致的 kernel 内噪声
- 不规则访存模式下不同 TB 需要不同优化策略的情况

当 Squash 返回 0 个 TB 级边界时，分析者可专注于 kernel 级处方，无需调查 kernel 内部异质性。

---

## 处方（Squash 信息增强，与 E0 相同 + 置信度提升）

| ID | Kernel | Squash 贡献 | 置信度 |
|----|--------|------------|-------|
| A-1 | softmax | 段 1 凝聚度=0.85 确认 softmax 与 attention_score 行为差异 → 需要两个独立处方 | HIGH |
| B-1 | attention_score | 同上 | HIGH |
| B-2 | gemm_tiled | 段 0、5 确认所有 GEMM 实例行为完全一致 → 一个处方覆盖全部 7 个实例 | MEDIUM |

---

## 机制在本 Workload 上的判定

**Squash 在 mini-transformer 上是确认性的，而非发现性的。**

Transformer 计算阶段从 kernel 名称和执行顺序即可推断。
段 1 的凝聚度信号有实际价值（标记了注意力阶段内部的异质性），
但得出的处方与 E0 完全相同。

Squash 价值预计在以下场景显著增加：
- Kernel 名称不可读（编译器内联、匿名函数）
- 大量同名 kernel 以非显然顺序交错执行
- 存在 TB 级内部异质性的 workload（如稀疏算法的边界 TB）
