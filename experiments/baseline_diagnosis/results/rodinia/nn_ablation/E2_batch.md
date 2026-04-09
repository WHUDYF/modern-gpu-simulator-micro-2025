# 诊断报告：nn [E2_batch]

**日期：** 2026-04-08
**硬件：** RTX 3080 Ti (SM_86)
**输入特征：** nn_full.json + nn_batch.json
**启用的机制：** batch

---

## Batch 提供的观察

**Kernel-level 聚类：**
- 1 个 cluster，包含全部 4 个 kernel
- 0 个 outlier kernel
- 同质度：**1.0**（完美）
- 质心摘要：所有 4 次 euclid launch 共享一个单一行为

**TB-level 聚类（每个 kernel 内部）：**
- Kernel 1：1 个 cluster，0 outliers，homogeneity=1.000（938 个 TB）
- Kernel 2：1 个 cluster，0 outliers，homogeneity=1.000（938 个 TB）
- Kernel 3：1 个 cluster，0 outliers，homogeneity=1.000（938 个 TB）
- Kernel 4：1 个 cluster，0 outliers，homogeneity=1.000（938 个 TB）

### 和 backprop 的对比

在 backprop 上，Batch 产出 **0 个 cluster 和 2 个 outlier kernel**，
因为 2 个 kernel 之间差异太大，DBSCAN 无法形成 cluster。

在 nn 上，Batch 产出 **1 个包含 4 个 kernel 的 cluster，完美同质度**，
是完全相反的极端。

### Batch 相比 E0 提供了什么

Batch 提供了**机器认证的双向均匀性**：
1. **Kernel 间均匀性**：4 次 launch 都在一个 cluster 里，同质度 1.0
   —— 不同 launch 之间没有任何多样性
2. **Kernel 内部均匀性**：每个 kernel 的 938 个 TB 形成一个完美 cluster
   —— 没有 outlier，没有边界条件 TB

**对处方的影响：**
- E0 可能假设："也许某些特殊的边界 TB 是瓶颈。"
- E2 **机器化地排除了这点**：任何层级都没有 outlier
- 诊断可以聚焦于**整个 workload 的问题**（block_dim），不需要担心特殊
  case 的 TB

---

## Stage A（和 E0 一样）：FAIL

---

## Class A 处方

**处方 A.1：把 block_dim 从 16 改成 64+**（和 E0、E1 一样）

**Batch 强化后的推理：**
- 1-cluster 的结果确认这是一个**整个 workload 统一的问题**，而不是
  局部 TB 出问题
- 修复统一地应用到全部 3752 个 TB（938 × 4 次 launch）
- 不需要做边界情况处理

**置信度：** HIGH（Batch 的均匀性认证支持使用单一全局修复）

---

## 总结

- 总处方数：1（只有 Class A）
- Batch 贡献：排除了 outlier 驱动和 cluster 特定的修复
- 发现的新瓶颈：0
- 处方数相对 E0 的变化：0

### Batch 在 nn vs backprop 上的价值

| 维度 | backprop | nn |
|------|---------|-----|
| Kernel-level 结果 | 0 clusters, 2 outliers | 1 cluster of 4, 0 outliers |
| 解读 | kernel **多样化** | kernel **一致** |
| 可操作信号 | "每个 kernel 需要自己的处方" | "一个处方应用到所有 kernel" |

两种情况下，Batch 的贡献都是**结构性认证**，而不是瓶颈发现。它的价值
在于**排除假设**，而不是产生新假设。

**极端相反的结果**（0 clusters vs 1 完美 cluster）证明 Batch 能正确
区分 workload 结构。
