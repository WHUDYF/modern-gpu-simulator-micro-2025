# 诊断报告：nn [E0_baseline]

**日期：** 2026-04-08
**硬件：** RTX 3080 Ti (SM_86)
**输入特征：** nn_full.json
**启用的机制：** 无
**诊断者：** 手动

---

## 第一阶段：软件利用率检查（Stage A）

### 利用率指标

| Kernel | waves_per_sm | achieved_occupancy | theoretical_occupancy | grid_size | block_dim |
|--------|--------------|---------------------|----------------------|-----------|-----------|
| euclid (×4 次启动) | **0.73** | 18.1% | **33.33%** | 938x1x1 | **16x1x1** |

### 第一阶段判定

- [ ] workload 充分利用了硬件
- [x] workload 没有充分利用硬件

**判定：FAIL（失败）**

- `waves_per_sm = 0.73 < 4` → grid 太小，连一波都填不满
- `theoretical_occupancy = 33.33%`（不是通常的 100%）→ **即使理论最大值也只有峰值的 1/3**
- `achieved_occupancy = 18.1%` → 进一步被运行时因素限制
- `avg_active_threads_per_warp = 11.62/32` → warp 利用率仅 36%，每个 warp 有 64% 的线程被浪费

### 根因分析

`theoretical_occupancy = 33.33%` 是决定性的证据。SM_86 每 SM 最多 48 warps，
33.33% 意味着每 SM 最多只能有 16 个 warp 活跃。由于 `block_dim = 16`
（半个 warp），每个 block 仍然占用一个完整的 warp 槽位但只使用其一半线程。
这同时限制了 **每 SM 的 warp 数** 和 **每 warp 的活跃线程数**。

这是一个**kernel 启动配置**的问题，不是数据规模的问题。增加输入数据会添加
更多 block，但每个 block 仍然只有 16 个线程，仍然浪费半个 warp，仍然把
每 SM warp 数卡在 16 个。

### Class A 处方

**处方 A.1：把 block_dim 从 16 改成 64 或更大**

- **修改：** 在 `nn_cuda.cu` 里把 block size 从 16 改成 64（或 128）
- **依据：** block_dim = 16 是半 warp，永久浪费每 warp 50% 的线程槽位
  **且**把每 SM 理论 warp 数卡在 16 个（33%）
- **预期：**
  - avg_active_threads_per_warp: 11.62 → 32（完整的 warp 利用率）
  - theoretical_occupancy: 33% → 100%（不再被 block size 限制）
  - achieved_occupancy: 18% → 60%+
- **验证方法：** 用更大的 block_dim 重新编译 nn，重跑 NCU，对比指标
- **代价：** 需要修改 kernel 源码；如果原算法依赖 block 级别的同步（比如
  block 内部的特定结构），可能会影响正确性

**第二阶段（Stage B）按协议不执行**（因为 Stage A 失败）。

---

## 推测性的 Stage B 备注（用于和开启机制的实验对比）

即使 Stage B 按协议暂停，这里列出原始指标暗示的东西（除非 Class A 修复，
否则应忽略）：

- L1 hit rate 83%（好），L2 hit rate 16%（差，大部分 L1 miss 直接打到 DRAM）
- compute throughput 7%（没有任何东西是计算密集的）
- DRAM 17%，L1/L2 都约 9%（在绝对值上没有任何东西是内存密集的）
- IPC 0.5，warp_cycles_per_issued 16.66（中等 stall）
- NCU 明确标记：**92% 的全局访存是 uncoalesced（非合并）**

如果 Class A 修复被应用，剩下的 Class B 瓶颈很可能是 **uncoalesced 全局
访存模式**（这是 kernel 级别的优化，不是 simulator 配置的改动）。这和
backprop 的 FP64 串行化与 shared memory 带宽问题都不一样。

---

## 总结

- 总处方数：1（只有 Class A —— Stage A 失败）
- Class A 处方：1（改 block_dim）
- Class B 处方：0（Stage B 暂停）
- **关键观察**：nn 的 Class A 失败和 backprop 的 **结构不同**。
  backprop 是"grid 随 input 扩展，只是 input 太小"；
  nn 是"grid 是硬编码的，block_dim 本身就是半个 warp"。
  这是两种**定性不同**的 Class A 失败模式。
