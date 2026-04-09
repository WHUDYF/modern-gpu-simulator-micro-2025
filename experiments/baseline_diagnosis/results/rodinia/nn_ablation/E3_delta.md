# 诊断报告：nn [E3_delta]

**日期：** 2026-04-08
**硬件：** RTX 3080 Ti (SM_86)
**输入特征：** nn_full.json + nn_delta.json
**启用的机制：** delta

---

## Delta 提供的观察

**Kernel-level 字段温度：**

| 字段 | 温度 | 分类 |
|------|------|------|
| num_barriers | 0.0 | COLD |
| num_tbs | 0.0 | COLD |
| total_dynamic_instructions | 0.0 | COLD |
| total_static_instructions | 0.0 | COLD |
| uses_fp64 | 0.0 | COLD |
| uses_shared_memory | 0.0 | COLD |

- **全部 6 个字段都是 COLD**
- **0 个 hot field**
- 0 个字段相关性（4 次 launch 虽然存在，但完全一致 → 没有方差）
- 0 个 outlier diffs

**这是 Delta 对"均匀性"的最强信号**：每一个被测字段在全部 4 次 kernel
launch 之间完全稳定。这 4 次 euclid 调用是行为上的"复印件"。

### Delta 相对 E0 提供的关键洞察

**只用 E0** 时，AI 在 `stats_csv` 和 `enhanced_execution_info` 里看到
4 个 kernel launch。它可以观察到"4 个 kernel，同名"。但它仍然可能假设：
"也许 4 次 launch 做的事情略有不同（不同的数据分区），其中某一次是
瓶颈。"

**E3（Delta）机器化地排除了这个假设**："全部 6 个行为字段都是 COLD
= 4 次 launch 无法区分 = 它们中间没有"坏的那个 kernel"。"

**这是 Phase 2-3 全程中，第一次有机制提供了 E0 无法廉价产生的信号。**
在 backprop 上，Delta 的 hot field（`uses_fp64`）是 E0 也能通过扫描
opcode 得到的信息。在 nn 上，Delta 的"全部 cold"是一个 **全局统计
声明**，E0 需要显式计算才能得到。

### 对 Class A 处方的影响

**E0 的推理：** "block_dim=16 太小"（基于静态信息推断）。

**E3 的推理：** "Class A 修复明确是 **kernel 重构**，因为：
1. **4 次 launch 完全一致**（Delta：kernel 间 0 个 hot field）
2. 所以，改变 **launch 次数** 不会帮助（更多 launch 不会有任何差异）
3. 所以，修复必须改变**每次 launch 内部做什么**
4. 唯一可见的内部结构性问题是 block_dim=16"

E3 的推理**机器化且更紧凑**。

---

## TB-level Delta

**Kernel 1（其他 3 个 kernel 类似）：**
- 0 hot fields
- 9 cold fields
- 15 个"相关性"（可能是虚假的 —— 见下文备注）
- 0 outlier diffs

**关于虚假相关的备注：** Delta 在 nn 的 TB-level 分析上报告了 15 个字段
相关性，但检查底层特征值发现每个 kernel 内所有 TB 特征都是逐 bit 相同的
（零方差，只有 1e-16 级别的浮点数噪声）。±1.0 的相关系数是"几乎零"的
噪声向量相除的假象。**这是 Delta 实现的一个 bug**：它应该先过滤掉 std
低于某个数值 epsilon 的字段再计算相关性。详见报告末尾的"Known Bugs"。

**（注：此 bug 已于 2026-04-08 晚修复，见 commit `8d55e9b`。修复后
TB-level correlations 变成 0，和预期一致。）**

---

## Stage A（和 E0 一样）：FAIL

---

## Class A 处方

**处方 A.1：把 block_dim 从 16 改成 64+**（和 E0/E1/E2 一样）

**Delta 强化后的推理：**
- 所有 kernel-level 字段都是 COLD → 4 次 launch 相同 → "跑更多次" 不是
  有效修复
- 唯一的自由度在 kernel 内部结构
- block_dim=16 是最直接的结构性问题

**置信度：HIGH**（这是三个机制实验中**证据最强**的一个 —— Delta 的
"全部 cold" 信号是推理链的**最直接机器化**）

---

## 总结

- 总处方数：1（只有 Class A）
- 发现的新瓶颈：0
- **机制价值**：Delta 在 nn 上提供了一个**真正新的信号**（"kernel-level
  所有字段都是 cold"），这是 E0 无法廉价得到的
- **这是 ablation 全程第一次有机制提供非冗余信息**

### Delta 在 nn vs backprop 上的价值

| 维度 | backprop | nn |
|---|----------|-----|
| Kernel-level hot fields | 4（uses_fp64, ...） | 0 |
| 信息内容 | 可以从 opcode 扫描推出 | **不能**从简单推理得到，必须计算跨 kernel 差异 |
| 对诊断的价值 | 确认 FP64 瓶颈 | **机器化"launches 完全一致"的发现** |
| 贡献类型 | confirming（确认） | **discovering（发现）** |

在 nn 上，Delta 是**发现型**：它机器化了一个陈述，而 E0 只能通过显式
比较 kernel 签名才能产出这个陈述。在 backprop 上，Delta 是确认型。

---

## Known Bugs（已修复）

**Bug 1：零方差字段上的虚假相关性**
- 描述：Delta TB-level 分析在零真实方差（只有浮点数噪声）的字段对之间
  报告 ±1.0 的相关系数
- 原因：相关性计算没有检查 std 是否在数值上有意义
- 修复：在把字段纳入相关性分析之前添加最小 std 阈值（1e-10）
- 影响：
  - 曾经污染 nn 和 backprop 上 TB-level 的相关性输出
  - 对 kernel-level 输出没有影响
  - 对主要诊断没有影响（诊断基于 hot/cold 分类，不是 correlations）
- 状态：**已修复**（commit `8d55e9b`，2026-04-08）
- 优先级：MEDIUM
