# 诊断报告：nn [E1_squash]

**日期：** 2026-04-08
**硬件：** RTX 3080 Ti (SM_86)
**输入特征：** nn_full.json + nn_squash.json
**启用的机制：** squash

---

## Squash 提供的观察

**Kernel-level 分段：**
- **1 个 segment**，涵盖全部 4 次 kernel launch
- 0 个边界
- Dominant opcodes: IMAD.MOV.U32, FFMA, IADD, LDG
- 解读：4 次 launch 是**同一个 kernel 的相同行为**

**TB-level 分段（每个 kernel 内部）：**
- 每个 kernel 1 个 segment，0 个边界
- 每个 kernel 内部的 938 个 TB 均匀一致

### Squash 相比 E0 提供了什么

Squash 提供了**一个新的结构性信息**：4 次 kernel launch 行为上完全相同
（形成单一 segment）。E0 也能从 `stats_csv` 的 4 行数据推出这一点
（同 kernel 名、相似的指令数），但 **Squash 把这个推断形式化了**。

**为什么这对 nn 有意义：**

E0 可能会猜测："也许把 nn 启动更多次可以摊薄启动开销。"
Squash 的 "1 段，0 边界" 结果直接否定了这个猜测：**每次启动都是相同的，
所以启动更多次只是产生更多相同的副本，不会带来新的行为。** 瓶颈不可能是
"kernel 多样性不足"。

这收窄了 Class A 修复空间：Squash 确认我们需要**kernel 内部结构重组**
（改 block_dim，而不是改 launch pattern）。

---

## Stage A（和 E0 一样）

**判定：FAIL**（`waves_per_sm = 0.73`, `theoretical_occupancy = 33.33%`）

---

## Class A 处方

**处方 A.1：把 block_dim 从 16 改成 64+**（和 E0 一样）

**Squash 强化后的推理：**
- E0 的推理："block_dim=16 限制了 warp 利用率"
- E1 额外补充："Squash 确认 4 次 launch 完全一致；改变 launch 数量或
  pattern 无法帮助。修复必须**在 kernel 结构内部**，具体就是 block_dim。"

这比 E0 更精确地收窄了修复空间。

**置信度：HIGH**（Squash 的 "launches 相同" 发现是一个强信号，说明
launch-level 的改动没用）

---

## 总结

- 总处方数：1（Class A）
- Squash 贡献：**收窄了 Class A 修复空间**（排除了 launch-pattern 改动）
- 发现的新瓶颈数：0
- 处方数量相对 E0 的变化：0
- 置信度改进：E0 的 Class A 本来就是 HIGH；Squash 让推理**更明确地说出
  了"launch-pattern 改动为什么无效"**

### Squash 在 nn vs backprop 上的价值

- **在 backprop 上**：Squash 确认了相位区分（2 个相位，1 个边界）。
  信息是冗余的，因为 E0 从 per-kernel opcode 就能看出来。
- **在 nn 上**：Squash 确认了**非区分**（1 段，0 边界）。这是一个
  **不同种类的信号**：它不只是确认 E0 已经看到的东西，**它排除了一整类
  假设**（launch-pattern 改动）。

**结论：** Squash 在 nn 上是一个**有区分力的负向信号**，比它在 backprop
上的确认性正向信号更有价值。
