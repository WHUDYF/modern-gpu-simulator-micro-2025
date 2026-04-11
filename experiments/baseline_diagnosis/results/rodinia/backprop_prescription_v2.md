# 处方性诊断报告 v2（两阶段结构）：Rodinia backprop

日期：2026-04-08
平台：RTX 3080 Ti (SM_86 Ampere, 80 SMs)
诊断者：AI Agent (Claude) + 人类方法论校正
特征来源：rodinia2Ampere preset traces + NCU 实测（input=4096 和 65536 两组）

**关键方法论更新：** 本报告遵循 spec 2.5 节的两阶段原则，严格区分：
- **Class A（软件配置处方）**：workload 启动参数层面
- **Class B（硬件微架构处方）**：模拟器 config 层面

前者是后者的**前置条件**。Class B 诊断必须基于已修正 Class A 问题后的数据。

---

## Stage A：软件利用检查（input=4096 的原始数据）

### A.1 利用率快速诊断

**bpnn_layerforward_CUDA @ input=4096：**
- `waves_per_sm = 0.53`
- `achieved_occupancy = 49.09%`
- `achieved_warps_per_sm = 23.56 / 48`
- Grid = 1×256×1 = 256 blocks 分布到 80 SMs ≈ 3.2 blocks/SM

**bpnn_adjust_weights_cuda @ input=4096：**
- `waves_per_sm = 0.53`
- `achieved_occupancy = 44.41%`
- 同样 256 blocks / 80 SMs 分布

### A.2 Stage A 诊断结论

**两个 kernel 都触发了 `waves_per_sm < 4` 的软件利用问题。**

根据方法论原则 2，此时**必须先修复 Class A 问题，不能在这个数据点上开
Class B 处方**。否则所有"distance-to-roof"指标都是被 grid 太小污染的。

### A.3 Class A 处方

**处方 A.1：增加 backprop 输入层大小**

- **修改：** 运行命令从 `./backprop 4096` 改为 `./backprop 65536`
- **依据：** backprop 的 grid 由 input_layer_size 推导（`grid = input/16`），增大
  input 直接线性增加 grid
- **预期：**
  - waves_per_sm: 0.53 → ~8.5（16x）
  - achieved_occupancy: ~45% → ~90%
  - NCU 各项指标的绝对值会显著上升（因为 per-SM 填满了）
- **代价：** 运行时间增加约 4x（GPU 做 16x 的工作，但利用率提升 4x）
- **验证方法：** 重采 NCU，确认 `waves_per_sm ≥ 4` 且 `achieved_occupancy ≥ 80%`
- **置信度：HIGH**（这是机械计算，不是推理）

### A.4 Class A 处方验证结果（已实测）

用 `./backprop_3080 65536` 重跑 NCU，结果：

| 指标 | 4096 (原始) | 65536 (修正后) | 变化 |
|------|------------|----------------|------|
| waves_per_sm (both kernels) | 0.53 | **8.53** | ✅ 16x |
| achieved_occupancy (forward) | 49.09% | **88.92%** | ✅ |
| achieved_occupancy (adjust) | 44.41% | **94.61%** | ✅ |

**Stage A 处方验证成功** ——`waves_per_sm ≥ 4` 达成，可以进入 Stage B。

---

## Stage B：硬件架构瓶颈分析（input=65536 修正后的数据）

### B.1 Kernel 1: bpnn_layerforward_CUDA（前向传播）

#### B.1.1 当前状态（input=65536）

**结构指纹（来自 trace 静态分析，与 input 无关）：**
- Top opcodes: `IADD3(14), NOP(13), LDS(12), IMAD(8), BAR.SYNC(8), STS(7)`
- **Tile-based matrix multiply**：8 BAR.SYNC × 12 LDS + 7 STS
- Static shared memory: 1088 bytes（16×16 float tile + 少量标量）
- block_dim = 16×16 = 256 threads = 8 warps/block

**Distance-to-Roof：**
| 资源 | 4096 | 65536 | 含义 |
|------|------|-------|------|
| **L1/TEX Cache** | 41% | **80.02%** | **🔴 主要瓶颈** |
| **Compute (SM)** | 23% | **72.41%** | **🟡 次要瓶颈** |
| **DRAM** | 8% | 44.19% | 中等 |
| L2 Cache | 8% | 24.63% | 中低 |

**关键约束指标：**
- IPC active: 1.95（forward 没有 FP64 瓶颈）
- L1 hit rate: 57.68%（和 4096 基本一样，没变）
- L2 hit rate: 55.21%
- warp_cycles_per_issued_inst: 21.58（和 4096 基本一样，结构属性）
- avg_active_threads_per_warp: 30.77/32（~96%，轻微 predication）
- registers_per_thread: 20
- achieved_warps_per_sm: 42.68/48（89%）

#### B.1.2 因果分析

**根因（确认）：L1/shared memory 路径的带宽饱和**

- L1/TEX throughput 80%，其他资源远未饱和
- 注意：这里的"L1/TEX"**包括 LDS/STS（shared memory 访问）**，因为它们共享
  同一块物理 SRAM
- 从静态指令看，LDS+STS 占核心循环的大部分 → shared memory bank 带宽是
  真实瓶颈，不是 L1 cache 容量
- 证据：L1 hit rate 只有 57.68%（不是特别高），但 L1 throughput 已经 80%
  → 说明 LDS 贡献了主导流量，不是 LDG hit

**关联因素：** Compute throughput 72% 是被 LDS 推动的 —— 每条 LDS 在读数据后
紧跟着要被 FMA 使用，所以 compute pipe 的 "busy" 实际上和 LDS 吞吐联动。

#### B.1.3 Class B 处方

##### 处方 B.1.1：增加 shared memory bank 数量（HIGH 优先级）

- **修改：** `-gpgpu_shmem_num_banks 32 → 64`
- **依据：**
  - LDS+STS 带宽饱和（80% throughput）
  - 当前 SM_86 默认 32 banks，加倍到 64 可以直接提升 shared memory 带宽
  - 但要注意：Ampere 实际硬件是 32 banks，这个修改是"如果重新设计 SM 会怎样"
- **预期：**
  - L1/TEX throughput 从 80% 降至 50-60%（带宽压力分散了）
  - IPC 从 1.95 提升至 2.5-3.0
  - Duration 降低 20-30%
- **预期代价：** shared memory 面积增加约 30-40%（banks 多意味着每个 bank 更小
  或总容量增大）
- **验证：** 修改 config，对比 `gpu_sim_cycle` 和 shared memory 统计
- **置信度：MEDIUM**（shared memory bank 数和 throughput 的关系需要模拟器
  确认是否敏感）

##### 处方 B.1.2：增大 L1/shared memory 总容量（MEDIUM 优先级）

- **修改：** `-gpgpu_cache:dl1 [size]` 从 128KB 增加到 192KB
- **依据：**
  - L1 hit rate 只有 57.68%，可以提升
  - 增加容量让更多 LDG 命中，减少 L2/DRAM 流量
- **预期：**
  - L1 hit rate 从 57.68% 提升至 70-75%
  - L2 throughput 下降（流量减少）
  - L1/TEX throughput 略微下降
- **预期代价：** L1/shared memory 整体面积增加 50%
- **注意：** 这个处方和 B.1.1 是**正交**的，可以独立测试也可以一起测试
- **置信度：MEDIUM**

##### 处方 B.1.3（对照处方）：增加 warp scheduler 数量

- **修改：** `-gpgpu_num_sched_per_core 4 → 6`
- **依据：** warp_cycles_per_issued = 21.58
- **预期效果较小**，因为：
  - Occupancy 已经 89%，更多 warps 可用
  - 但带宽瓶颈（L1 80%）限制了 scheduler 无论多快都无法推动更多 issue
  - 这个处方在 input=4096 时更有意义（当时 occupancy 只 49%）
- **置信度：LOW**（v1 报告中这是"重点处方"，但 v2 数据显示它效果应该很小）
- **教训：** 修复 Class A 后重新看，原来想开的 Class B 处方可能已经过时

---

### B.2 Kernel 2: bpnn_adjust_weights_cuda（权重更新）

#### B.2.1 当前状态（input=65536）

**结构指纹（与 input 无关）：**
- Top opcodes: `LDG.E(12), F2F.F64.F32(12), NOP(11), IMAD.WIDE(6), DMUL(6),`
  `DFMA(4), F2F.F32.F64(4), STG.E(4)`
- **关键：所有算术是 FP64（DMUL/DFMA）**，数据在 fp32 和 fp64 间反复转换
- Static shared memory: 0 bytes（不使用 shared memory）
- 纯全局访存 + FP64 计算

**Distance-to-Roof：**
| 资源 | 4096 | 65536 | 含义 |
|------|------|-------|------|
| **Compute (SM)** | 51% | **84.98%** | **看起来是主要瓶颈** |
| DRAM | 7% | 21.41% | 中等 |
| L1/TEX | 6% | 10.56% | 很低 |
| L2 | 7% | 10.98% | 很低 |

**关键约束指标 ——  这里是最惊人的部分：**

| 指标 | 4096 | 65536 | 变化 |
|------|------|-------|------|
| achieved_occupancy | 44% | **94.61%** | +2.1x（warps 翻倍）|
| **IPC active** | 0.13 | **0.15** | **+15%（几乎无变化！）**|
| **warp_cycles_per_issued** | 127 | **286.87** | **+2.26x（变得更差）**|
| Compute throughput | 51% | 84.98% | +1.67x |
| DRAM | 7% | 21.41% | +2.9x |
| L1 | 6% | 10.56% | +1.65x |

#### B.2.2 因果分析

**这是一个经典的"增加并行度反而让瓶颈更明显"的案例。**

**证据链：**
1. Warps 数量从 21.32 → 45.41（2.1x）
2. **但 IPC 几乎不变**（0.13 → 0.15）
3. **warp_cycles_per_issued 从 127 → 287**（更差了！）
4. Compute throughput 从 51% → 85%（看似变好）

**唯一自洽的解释：**

**所有 warps 都在等同一个稀缺执行单元（FP64 pipe）。**

想象 6 个 warp 排队等一台机器 vs 45 个 warp 排队等同一台机器：
- 单个 warp 的等待时间（cycles per issue）从 127 变成 287（队列变长）
- 但整体吞吐（IPC）没变 —— 仍然是机器的最大速率
- "Compute throughput" 看起来变高，是因为 SM 的 FP64 pipe 更少空闲时刻（队
  列总是满的）

**这是 Ampere 消费级 GPU 的 FP64 throttling 特征：**
- SM_86 的 FP64 算力是 FP32 的 1/64
- 每个 SM 的 DP 单元数量极少
- 任何使用 FP64 的 kernel 都会被这个配置严重限制

**L1 hit rate 73.32%（不错）、DRAM 21%（远未饱和）** → 确认问题不在 memory
层，纯粹在 FP64 execution pipe。

#### B.2.3 Class B 处方

##### 处方 B.2.1：增加 SM 内 DP 单元数量（HIGH 优先级，决定性处方）

- **修改：** `-gpgpu_num_dp_units 1 → 4`（或更高）
- **依据：**
  - Trace 静态分析：核心循环全是 DMUL/DFMA
  - NCU 矛盾：compute 85% 但 IPC 0.15 + warp_cycles_per_issued 287
  - Class A 已修正（occupancy 94%），排除了软件利用问题
  - 内存系统全部低利用（L1 10%, L2 11%, DRAM 21%）
- **预期：**
  - IPC 从 0.15 → 0.5-0.8（3-5x 提升）
  - warp_cycles_per_issued 从 287 → 50-80
  - Duration 从 90560ns → 20000-30000ns（降低 70%）
  - compute throughput 可能降低（因为现在多个 DP 单元分担负载，每个不再
    100% busy）
  - 内存系统利用率会随之提升（去瓶颈化效应）
- **预期代价：** SM 面积增加 3-5%，功耗相应增加
- **商业意义：** 这正是数据中心 GPU（A100, H100）和消费级 GPU 的主要差别
  之一 —— A100 的 FP64:FP32 = 1:2，消费级 = 1:64
- **验证：**
  - 修改 `gpgpusim.config` 相关参数
  - 重跑模拟器，对比 IPC, duration, compute throughput
  - **成功标准：** IPC 提升 ≥3x，且 DRAM/L1 仍未饱和
- **置信度：HIGH** —— 来自三个独立证据的交叉验证

##### 处方 B.2.2（算法层对照，不作为架构建议）：算法改用 FP32

**说明：** backprop 用 FP64 做梯度累加是为了数值稳定性。但现代 DL 框架大
部分 training 是 FP32 或混合精度。如果改成 FP32，adjust_weights kernel 的
瓶颈立刻消失。

**这个"处方"不是架构处方，是算法处方** —— 列在这里是为了对比：

- 同一瓶颈可以从**两个不同层次**解决：
  - 硬件层：增加 DP 单元（处方 B.2.1）
  - 软件层：改用 FP32（这条）
- **AI 诊断的价值**：能同时识别两层的解法，并说明它们的适用场景
  - 硬件解法适合"workload 必须用 FP64" 的场景（科学计算）
  - 软件解法适合"数值精度可协商"的场景（现代 DL training）

---

## 附录 A：v1 vs v2 对比

| 维度 | v1 报告 | v2 报告 |
|------|---------|---------|
| 数据点 | input=4096 | input=4096 (Stage A) + input=65536 (Stage B) |
| 处方结构 | 混合 | 严格两阶段 |
| forward 主处方 | "增加 scheduler"（置信度 MEDIUM-HIGH） | 降级为 LOW；真正瓶颈是 L1/shared bandwidth |
| adjust_weights 主处方 | "增加 DP 单元"（置信度 HIGH） | 保持，证据更强 |
| 方法论瑕疵 | grid 太小的问题被识别但没前置处理 | 严格分离 Class A 和 Class B |

**v1 的误诊：** 把 forward kernel 的 "warp_cycles_per_issued = 21.57" 解读为
"scheduler 不够"。但 v2 数据（occupancy 89%）证明 scheduler 不是瓶颈 ——
增加 warp 后 IPC 翻倍了（1.0 → 1.95），scheduler 早已够用。**真正的瓶颈是
带宽（L1/shared）**。

**这个案例本身就是方法论升级的动机** —— 如果不做 Class A 修正，v1 的错误
处方会被执行，结果可能微弱提升或无变化，然后得出"AI 处方没用"的错误结论。

---

## 附录 B：v2 的完整处方清单（最终版）

| 编号 | 类别 | 处方 | 置信度 | 是否验证 |
|------|------|------|--------|---------|
| A.1 | Software | input=4096 → 65536 | HIGH | ✅ 已实测验证 |
| B.1.1 | Hardware (forward) | shmem banks 32 → 64 | MEDIUM | ⏳ 待模拟器验证 |
| B.1.2 | Hardware (forward) | L1 size 128KB → 192KB | MEDIUM | ⏳ 待模拟器验证 |
| B.1.3 | Hardware (forward, 对照) | scheduler 4 → 6 | LOW | 不推荐优先验证 |
| B.2.1 | Hardware (adjust_w) | DP units 1 → 4 | HIGH | ⏳ 待模拟器验证（核心处方） |
| B.2.2 | Algorithm (adjust_w, 对照) | FP64 → FP32 | N/A | 非架构处方 |

**下一步：** 在模拟器上验证处方 B.2.1（最高置信度）。需要：
1. 先跑 input=65536 的 backprop trace（当前 trace 只有 input=4096）
2. 或者继续用 input=4096 trace，但接受结果会被小 grid 部分掩盖

## 附录 C：方法论收获

这次实验产出的最大价值**不是处方本身，而是方法论的迭代**：

1. **软件 / 硬件处方分离**是正确设计，之前的 v1 把它们混在一起是错的
2. **Class A 前置检查**必须机械执行，AI 不能跳过
3. **同一数据点不能同时支持 Class A 和 Class B 推理**
4. **小 workload 会掩盖真实架构瓶颈**，也会制造假架构瓶颈（forward 的
   "scheduler 不足"就是假瓶颈）
5. **交叉推理（static trace + hw stats）是诊断非平凡瓶颈的唯一方法**（FP64
   瓶颈）

这些原则应该被写进 spec，后续每个 workload 诊断都按这套流程走。
