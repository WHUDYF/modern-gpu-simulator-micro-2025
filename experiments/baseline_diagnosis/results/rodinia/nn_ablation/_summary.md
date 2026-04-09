# Phase 3 消融实验总结：nn + 跨 Dwarf 对比

**日期：** 2026-04-08
**Workload：** Rodinia nn（最近邻搜索，euclid kernel）
**硬件：** RTX 3080 Ti (SM_86)

## nn 每实验总结

| 实验 | Squash | Batch | Delta | Stage A | Class A 处方 | Class B |
|------|--------|-------|-------|---------|-------------|---------|
| E0 baseline | ❌ | ❌ | ❌ | FAIL | 1 (block_dim) | 0 |
| E1 squash | ✅ | ❌ | ❌ | FAIL | 1 | 0 |
| E2 batch | ❌ | ✅ | ❌ | FAIL | 1 | 0 |
| E3 delta | ❌ | ❌ | ✅ | FAIL | 1 | 0 |
| E4 full | ✅ | ✅ | ✅ | FAIL | 1 | 0 |

**所有 5 个实验共享的处方：** 把 `block_dim` 从 16 改成 64+（kernel 源
码修改，不是 simulator config 修改）。

**nn 的 Stage A 根本性地失败**，且**无法通过使用更大的输入数据集修复**
（grid 硬编码为 938，和数据规模无关）。修复必须在 kernel 源码层级进行。

---

## nn 上的机制信号（和 backprop 对比）

### Squash

| | backprop | nn |
|---|----------|-----|
| Kernel-level segments | 2 | **1** |
| Boundary count | 1 | **0** |
| 信号解读 | "workload 有 2 个相位" | "workload 只有 1 个均匀相位" |
| 对诊断的价值 | 确认相位区分 | **排除 launch-pattern 修复** |

### Batch

| | backprop | nn |
|---|----------|-----|
| Kernel-level clusters | 0 | **1** |
| Outlier kernels | 2 | **0** |
| 同质度 | 0.0 | **1.0** |
| 信号解读 | "kernel 差异太大无法聚类" | "所有 kernel 行为完全一致" |
| 对诊断的价值 | 确认 kernel 间多样性 | 确认 kernel 间均匀性 |

### Delta

| | backprop | nn |
|---|----------|-----|
| Kernel-level hot fields | 4（uses_fp64, num_barriers, ...）| **0** |
| Kernel-level cold fields | 1（num_tbs）| **6**（全部）|
| 信号解读 | "FP64 在相位间变化" | **"各次 launch 不携带任何信息多样性"** |
| 对诊断的价值 | 确认 FP64 瓶颈（可从 opcode 推导）| **机器化不可推导的洞察** |

**关键观察**：Delta 在 nn 上提供了**不可推导的信息**。E0 无法廉价地得
出"4 次 launch 完全相同"这个结论 —— 它必须显式对比每个 kernel 的签名。
Delta 把这个结论机器化成 "kernel-level 0 hot fields"，这是**真正的机制
贡献**。

---

## 第一次真正的机制价值

**在 backprop 上，所有 5 个实验都产出了相同的处方** —— 机制只是确认了
E0 已经找到的东西。

**在 nn 上，Delta（E3）提供了一种新形式的价值**：

- E0 的推理："block_dim=16 太小"（静态观察）
- E3 的推理："block_dim=16 太小**且** 4 次 launch 行为相同（所有字段
  都是 cold），所以修复必须在 kernel 内部而不是 per-launch"（机制驱动
  的修复空间收窄）

两条路径都得到同一个处方（block_dim=64），但 E3 的推理更受约束、更
容易解释。在一个含有许多 kernel 且启动结构模糊的 workload（比如一个
大型程序中有 16 个 kernel 变体）上，这个差异会翻译成**正确 vs 错误
的处方**。

---

## 跨 Dwarf 处方对比表

| Dwarf | 主要发现 | Class A | Class B | 机制价值 |
|-------|---------|---------|---------|---------|
| **backprop** | FP64 串行化（adjust_weights）+ L1/shared 带宽（forward） | input size（可通过模拟合并修复）| DP initiation, shmem banks | **CONFIRMING（确认）** —— 所有机制相对 E0 冗余 |
| **nn** | 半 warp block_dim | block_dim 改动（kernel 源码）| （推测：uncoalesced 访存, L2 miss）| **DISCOVERING（发现）** —— Delta 提供不可推导的信号 |

---

## Checkpoint 2 分析

按 spec §4.5：
> 如果机制在 nn 上的效果和 backprop 一致 → 进入 Phase 4
> 如果机制在 nn 上的效果不同 → 停下分析
> 如果机制在 nn 上严重失败 → 回到 Phase 0

### 观察到的结果：**机制在 nn 上的效果和 backprop 不同**

- 在 backprop 上，机制没有增加新洞察（都是冗余的）
- 在 nn 上，Delta 特别地提供了一个 E0 无法廉价推导的新洞察
- Squash 和 Batch 在两个 dwarf 上给出**相反的信号**（backprop：2 段 /
  nn：1 段；backprop：2 outliers / nn：1 cluster）—— 这证明它们能
  **正确区分 workload 结构**

### 解读

**机制是 workload-structure-sensitive 的**：它们在不同 workload 上
产出不同但正确的信号。它们对 AI 诊断的价值取决于**workload 的结构是否
从基础特征就能显而易见**：

- **明显的结构**（backprop：两个清晰不同的 kernel）→ E0 本来就能看到
  → 机制是确认性的，不是发现性的
- **不明显的结构**（nn：4 次看起来相似的 launch，需要跨 launch 对比
  才能确认"零多样性"）→ 机制机器化了这个观察 → 真正的发现价值

**这正是我们希望的行为**：机制应该在 workload 结构越不明显时，边际价
值越大。

### 决定：**继续 Phase 4（lud），并调整期望**

Phase 4 应该瞄准一个结构**比 nn 更不明显**的 workload，测试机制在
复杂度增加时是否继续提供发现价值。

Lud（LU 分解）是一个好的候选，因为：
1. 是一个多相位算法（主元选择、行消元等）
2. 可能有几个结构性非平凡的 kernel
3. 应该同时考验时间维度（Squash）和空间维度（Batch）的机制

---

## Phase 3 发现的已知 bug（已修复）

### Bug 1：Delta 在零方差字段上的虚假相关性

**描述：** 在 nn 的 TB-level 分析上，Delta 报告了 15 个 ±1.0 的字段
相关性，这些字段实际没有真实方差。

**原因：** 相关性计算没有过滤 std 低于某个数值 epsilon 的字段。浮点数
噪声（1e-16 级别）产生了看起来成比例的序列，从而产生虚假相关。

**影响：**
- 曾经影响均匀 workload（nn, backprop）的 TB-level 输出
- **不**影响 kernel-level 输出
- **不**改变主要诊断（仍然产生相同的处方）
- 产生误导性 JSON，如果下游消费者依赖会被坑

**修复：** 已应用（commit `8d55e9b`，2026-04-08）
```python
MIN_STD_FOR_VARIANCE = 1e-10  # 数值阈值
if np.std(s1) < MIN_STD_FOR_VARIANCE or np.std(s2) < MIN_STD_FOR_VARIANCE:
    continue
```

**回归测试：** 新增 `test_no_spurious_correlations_on_constant_fields`
在 100 个完全相同的合成 TB 上测试 Delta，确认未来不会再引入同类 bug。

**修复后的 nn delta 输出**：0 个 hot，6 个 cold，0 个 correlations，
0 个 outliers（和"nn 是完全均匀的"这一事实一致）。

---

## Phase 0-3 发现总结

1. **Phase 0 + Phase 1**：所有三个机制都已实现，19 个单元测试通过
   （修 bug 后 20 个），backprop 上的集成验证通过。

2. **Phase 2（backprop）**：机制都产出了正确的非平凡输出但相对 E0
   没有增加新洞察。backprop 是一个弱的 stress test。

3. **Phase 3（nn）**：
   - 机制产出了**定性不同的信号**（均匀性 vs 多样性），且都正确
   - Delta 提供了第一个**不可推导的洞察**（kernel 间所有字段 cold →
     launches 零信息多样性）
   - Squash 的"1 段"是一个**否定判别信号**，排除 launch-pattern 修复
   - Batch 的"1 完美 cluster"认证了整个 workload 的均匀性
   - **第一个真正证据显示机制对 E0 有额外诊断价值**，但只在特定的
     workload 结构上
   - 发现并修复了 Delta 的浮点噪声 bug

4. **Checkpoint 2 决定**：继续 Phase 4（lud），瞄准更复杂的 workload
   结构。
