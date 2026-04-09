# 诊断报告：nn [E4_full]

**日期：** 2026-04-08
**硬件：** RTX 3080 Ti (SM_86)
**输入特征：** nn_full.json + squash + batch + delta
**启用的机制：** 三个全开

---

## 三机制交叉验证

三个机制独立地确认了 nn 的同一个结构性属性：**4 次 kernel launch 行为
上完全一致**。

| 机制 | 信号 | 解读 |
|------|------|------|
| Squash | 1 段，0 边界 | 时间维度均匀 |
| Batch | 1 个 cluster 4 个 kernel，0 outliers，同质度=1.0 | 空间维度均匀 |
| Delta | kernel-level 0 hot fields，6 cold，0 correlations | 字段维度均匀 |

**三重收敛**到 "nn 在各次 launch 上行为均匀" 这个结构性观察，给了这个
观察最高的置信度。

---

## Stage A：FAIL

和 E0-E3 一样：`waves_per_sm = 0.73`，`theoretical_occupancy = 33.33%`，
`achieved_occupancy = 18.1%`，`avg_active_threads_per_warp = 11.62/32`。

---

## Class A 处方

**处方 A.1：把 block_dim 从 16 改成 64（kernel 源码改动）**

**目标：** `nn_cuda.cu` 的 kernel launch 配置

**修改：**
```c
// 修改前:
euclid<<<num_blocks, 16>>>(...);
// 修改后:
euclid<<<num_blocks/4, 64>>>(...);  // 同时调整线程索引
```

（具体修改取决于 kernel 内部逻辑；重点是使用完整的 warp。）

**依据（三重收敛）：**
1. **Squash**：1 段，0 边界 → launch 数量不是优化杠杆
2. **Batch**：1 个均匀 cluster + 0 outliers → 没有特殊 case 的 TB
3. **Delta**：kernel-level 0 hot fields → 4 次 launch 携带零信息多样性；
   "跑更多" 无法帮助
4. **基础指标**：theoretical_occupancy=33.33%（半 warp block size 导致
   的硬帽子），avg_active_threads_per_warp=11.62（每个 warp 总有一半
   是非活跃的）

综合来看：唯一可行的修复杠杆是**每 kernel 的线程结构**，具体就是 block_dim。

**预期效果：**
- avg_active_threads_per_warp: 11.62 → 32（2.75x）
- theoretical_occupancy: 33.33% → 100%
- achieved_occupancy: 18.1% → 60%+（可能仍被寄存器或 shared memory 限制，
  但根本的 block size 瓶颈已经去除）
- 运行时间：**显著改善是可能的**，但精确的加速比取决于内存合并是否
  同时改善

**预期代价：** 需要修改 kernel 源代码。如果原始算法假设 block_dim=16
（例如使用 warp 内 reduction），可能影响正确性。

**验证方法：**
- 修改 `nn_cuda.cu` 的 kernel launch 行和 kernel 主体（索引）
- 重新编译
- 重跑 NCU，对比：theoretical_occupancy, avg_active_threads_per_warp,
  整体运行时间
- **成功判据**：theoretical_occupancy ≥80%，运行时间改善 ≥1.5x

**置信度：** HIGH（三重收敛的机制证据 + 基础指标中清晰的结构信号）

---

## 推测性 Stage B（Class A 应用之后）

一旦 block_dim 增加，我们预期会看到以下额外瓶颈（目前被 occupancy 限制
所掩盖）：

1. **Uncoalesced 全局访存**（NCU 标记 92% uncoalesced）—— 很可能是
   Class A 修复之后的主要问题。这是 kernel 级别的优化（改数据布局或
   访存模式），不是 simulator 配置的改动。

2. **L2 miss rate 84%** —— 更多的 active warp 会增加 L2 压力。一个
   可能的未来 Class B 处方：`-gpgpu_cache:dl2` size 增加。

这些**仅为推测**，直到 Class A 被验证。

---

## 总结

- 总处方数：1（Class A）
- 高置信度：1
- 使用机制特征的处方：1（所有三个机制都支持 Class A 推理）
- 发现的新瓶颈：0（和 E0 相同的 Class A）
- **机制价值**：三重收敛地认证 "launch-pattern 改动无法帮助 —— 必须
  改 kernel 结构"

### 关键对比：E4 在 nn vs E4 在 backprop

| | backprop E4 | nn E4 |
|---|------------|-------|
| Stage A 状态 | PASS（使用 input=65536 合并后）| **FAIL**（无法通过合并修复）|
| 产出的处方 | 2 条（Class B）| 1 条（只有 Class A）|
| 机制是否增加新信息？| 否（所有信号都能从 E0 推导）| **是（Delta 的"全部 cold"是非平凡的）**|
| 机制价值类型 | 确认 / 交叉验证 | **发现 / 机器化**|
| 三重收敛 | 确认 FP64 瓶颈 | 确认结构性均匀 |

**在 nn 上，机制终于显示出了相对 E0 的清晰优势**："全部 cold" 的 Delta
信号和 "1 段" 的 Squash 信号**机器化**了 E0 只能通过显式跨 kernel 比较
才能得到的结论（launches 是一样的）。
