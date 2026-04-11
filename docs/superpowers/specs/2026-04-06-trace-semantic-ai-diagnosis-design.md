# 设计规格：GPU Trace 语义提取用于 AI 架构诊断与修改建议

日期：2026-04-06
最后更新：2026-04-07

## 1. 问题陈述

GPU 架构研究依赖 trace-driven 仿真来评估设计决策。仿真产生的 trace 和统计数据
包含丰富的行为信息，但这些信息目前只由人类专家手动检视 stats、波形和日志来消费。

**核心问题：** AI agent 能否从 GPU 模拟器的 trace 数据中**提出可验证的架构修改
建议**？这里"可验证"指的是：修改可以映射到模拟器配置参数，模拟器可以重跑并
定量对比前后差异。

### 描述性 vs 处方性（2026-04-07 关键澄清）

初版设计把目标定为"AI 诊断 trace"，但经过微基准验证发现，仅有描述性诊断
（"这个 workload 在做什么"）对架构研究者没有可执行价值。真正有价值的是
**处方性诊断**（"应该改什么、为什么改、预期效果、如何验证"）。

这一澄清带来两个根本变化：

1. **目标从"诊断"变为"诊断 + 处方 + 闭环验证"**。第三步（闭环评估）从
   可选升级为必需——没有闭环验证，处方性建议就是空话。
2. **特征从"行为描述"变为"距离瓶颈的多维度量"**。要让 AI 能够反推"如果
   改 X，期望效果 Y"，必须输入 workload 离每个潜在瓶颈的距离（多维 Roofline）。

### 为什么微基准不适合

微基准的设计目的是暴露架构极限，而非留出优化空间。MaxFlops 跑出的就是峰值
FLOPS，mem_bw 跑出的就是峰值带宽——它们的"问题"就是它们的"目标"。

微基准仍然有两个用途：
- **验证特征提取流程的正确性**（sanity check）
- **作为"已知上限"参照**，让 AI 判断真实 workload 距离各个 roof 多远

但**处方性诊断的主要目标必须是真实 AI workload**，那里才有"应该更快却跑得
不够快"的空间。

## 2. 目标与非目标

### 目标

1. 验证 AI 能否基于 trace 特征 + 多维 Roofline 指标，对真实 AI workload 产出
   **可验证的架构修改建议**（改什么参数、预期效果量级、验证方法）
2. 建立一套"距离瓶颈的多维度量"作为 AI 输入的标准特征集
3. 搭建闭环验证管线：诊断 → 参数修改 → 重跑模拟器 → 对比 → AI 评估
4. 判断 difftest 风格的压缩机制（Squash、Delta）能否产出对处方性诊断有价值的
   语义特征
5. 基于文献调研和实验，建立"AI 为架构决策需要什么"的参考框架

### 非目标

- 构建生产级压缩管线（这是研究原型）
- 实时或在线分析（从离线开始，后续演进）
- 跨架构泛化证明（先做单架构）
- 取代人类架构师（AI 提供处方建议，人来决策是否采纳）
- 在微基准上追求架构修改建议（微基准没有处方空间）
- 追求"最优"架构（目标是提出有根据的建议，不是找最优解）

## 2.5 方法论原则（2026-04-08 新增）

以下是 baseline diagnosis 第一次实际跑完整流程时（Rodinia backprop on RTX
3080 Ti）暴露出的方法论经验，提升为正式原则。

### 原则 1：软件配置处方 vs 硬件微架构处方必须分离

处方可以分成两类，**来源、验证路径、研究价值完全不同**，必须严格分开：

**Class A（软件配置处方）：**
- 修改对象：workload 的输入/启动参数（grid、block、input size、batch size）
- 不触碰：kernel 代码、硬件设计
- 验证方式：换参数重跑（真实硬件或模拟器都可以）
- 例：backprop input=4096 → 65536，grid 从 256 blocks 扩展到 4096 blocks
- 对架构研究的价值：**排除项** —— 不做架构决策，但必须先做

**Class B（硬件微架构处方）：**
- 修改对象：模拟器 config（代表硬件设计决策）
- 不触碰：workload 代码、启动参数
- 验证方式：同一 trace 在不同 simulator config 上跑
- 例：增加 DP 单元数量、扩大 L1 cache、调整 scheduler 数
- 对架构研究的价值：**核心产出** —— 给架构设计者的实际建议

### 原则 2：Class A 必须先于 Class B

AI 处方诊断必须分两阶段：

**阶段 A（软件利用检查）：**
- 检查 workload 是否已经饱和了硬件
- 检查项：
  - `waves_per_sm ≥ 4`（grid 足够填满 GPU 数波）
  - `achieved_occupancy ≥ 80%`（per-SM 资源被充分使用）
  - `grid_size × block_size ≥ 4 × SM_count × max_warps_per_SM`
  - batch / sequence length 合理
- 如果 A 阶段有问题，输出 Class A 处方，**暂停 B 阶段**
- Class A 处方的目的是"让后续 B 阶段的数据有意义"

**阶段 B（架构瓶颈分析）：**
- **前提**：阶段 A 问题已经修正，workload 正在充分利用硬件
- 只有在这个前提下，NCU 指标才能被当作"硬件真实瓶颈"解读
- 检查项：distance-to-roof、IPC vs throughput 矛盾、stall 原因、
  register pressure、shared memory bank conflict 等
- 输出 Class B 处方（simulator config 修改）

### 原则 3：两类处方必须基于不同的 workload 状态

- Class A 诊断看**原始 workload**
- Class B 诊断看**应用 Class A 处方后的 workload**
- 不能混在同一个数据点上推理，否则**归因混乱**
  - 反例："我改了 DP 单元（B）又改了 input size（A），IPC 提升了 5x" →
    无法判断是 B 有效还是 A 导致了本来就会发生的提升

### 原则 4：AI 要有能力输出"负面处方"

**负面处方（negative prescription）：** "在当前数据点上不要做架构调优，先修复
workload 的软件利用问题"

这种处方表面上"没有产出"，但对架构研究同样有价值 —— 它帮研究者避免在错误
的数据点上浪费时间。一个好的 AI 诊断必须能识别"此时不适合分析架构"的情况，
而不是硬着头皮输出可能误导的建议。

### 原则 5：静态 trace 特征 + 动态硬件指标的交叉推理

单一数据源容易误诊：
- 只看 NCU："compute throughput 51%" → 以为是 compute-bound
- 只看 trace："主要 opcode 是 DMUL/DFMA" → 只知道用了 FP64

交叉推理：
- NCU 的 "compute 51% 但 IPC 0.13" 本身是个矛盾
- trace 的 "用了 FP64" 解释了这个矛盾：FP64 pipe 被打满但 FP64 throughput 本身
  就低，所以"看起来 busy 实际却几乎没工作"

这个例子在实际 backprop 实验中出现过，是 AI 诊断产生非平凡洞察的第一个
真实案例。

---

## 3. 背景：两个源项目

### 3.1 modern-gpu-simulator-micro-2025

基于 Accel-Sim 的增强型 trace-driven GPU 模拟器：
- 三层 trace 压缩：指令级 RLE、warp 级共享 PC 序列、threadblock 级 base+delta
- 从 SASS 指令中解析 control bits
- Protocol Buffers trace 格式
- Per-kernel 模拟器统计（IPC、cache、throughput、occupancy）

关键 trace 文件：
- `dynamic_trace.pb`：kernel 执行顺序、stream 结构
- `threadblocks/`：per-TB 的 warp 指令流、地址、mask
- `enhanced_execution_info.json`：静态指令元数据、control bits
- `compressed_kernel_v8`：最新压缩格式，包含跨 TB delta

### 3.2 difftest

面向 RISC-V 处理器的协同仿真框架，包含三种压缩机制：
- **Squash**：合并连续的、状态相似的时钟周期，仅在行为变化时才输出——产出时序分段
- **Batch**：将多个单元打包成单次传输——减少 IO 开销
- **Delta**：对大型结构（寄存器堆）只传输变化的字段——产出稀疏变化图

## 4. 架构：四步验证（2026-04-07 重构）

设计遵循风险优先原则，但目标从"验证诊断能力"调整为"验证处方能力 +
闭环验证"。

```
第零步：基础设施准备
  输入：当前无法在 5090 上跑 NVBit，NCU 无 sudo
  动作：获取一台支持 NVBit 的 GPU 机器（A100/H100/RTX 4090 等 SM_80~SM_90）
        或解决 NCU 权限，或搭建替代 profiling 管线
  输出：可用的 trace + 硬件 stats 采集环境
  目标：能够在真实 AI workload 上采集完整输入数据

第一步：多维 Roofline 特征提取 + 模拟器校准
  输入：真实 workload 的 trace + 硬件 stats
  动作：
    a. 提取压缩特征 + 计算 distance-to-roof 多维指标
    b. 用同一 trace 跑模拟器（SM86_RTXA6000 config），对比 sim_stats 和 hw_stats
    c. 校准成功后，sim_stats 成为闭环里"修改前后对比"的基准
  输出：每个 kernel 的完整特征包 + 校准后的模拟器基线 stats
  目标：既让 AI 有特征可读，也让模拟器成为可信的"如果改 X 会怎样"的推理工具

第二步：处方性诊断 + 初步闭环
  输入：特征包 + 模拟器可修改参数清单
  动作：AI 输出 "改什么参数、预期效果、验证方法" 的处方报告，
        挑选 1-3 条建议实际修改模拟器配置，重跑对比
  输出：处方报告 + 实测前后对比 + AI 的评估
  目标：验证 AI 的处方建议是否能产出可度量的性能变化

第三步：语义增强（Squash/Delta）
  输入：第二步暴露的特征盲区
  动作：引入 Squash/Delta 产出补充语义特征
  输出：增强特征 + 重新诊断 + 闭环对比
  目标：判断新特征是否让处方建议更准确或发现新瓶颈
```

关键变化：
- **第零步是新增的前置步骤**——现实障碍（5090 硬件限制、NCU 权限）必须先解决
- **第二步合并了原设计的"诊断"和"闭环"**——因为处方性诊断必须通过闭环验证才
  有意义，分开做没有价值
- **第三步（Squash/Delta）只有在第二步证明处方思路可行后才推进**

### 4.0 第零步：基础设施准备（新增）

当前状态（2026-04-07 记录）：
- **NVBit 1.7.6 不支持 Blackwell SM_120**，5090 上 trace 生成失败
  （cuobjdump 报错 `sm_0`）
- **NCU 需要 GPU 性能计数器权限**（`NVreg_RestrictProfilingToAdminUsers=0`），
  需要 sudo，当前账号无权限
- Accel-Sim 社区确认 5090 支持问题仍未解决

基础设施三选一：

**选项 A（推荐）：租用支持 NVBit 的 GPU 服务器**

- 目标架构：RTX 4090（SM_89）、A100（SM_80）、H100（SM_90）——均为 NVBit 1.7.6
  正式支持的架构
- 优势：NVBit trace 和 NCU profiling 都能跑通，数据链路完整
- 代价：云服务商按小时计费
- 一次完整实验周期需要的时间：trace 生成 + NCU 采集 + 模拟器运行 ≈ 几小时

**选项 B：获取 sudo 解决 NCU 权限问题，仅用 NCU 数据**

- 只在 5090 上采集 NCU 硬件 stats，完全跳过 NVBit trace
- 劣势：失去 trace 的压缩特征这一核心输入，压缩即语义提取的思路无法验证
- 不推荐——这等于放弃了我们研究的差异化点

**选项 C：使用已有微基准 trace + 构造的已知 workload**

- 完全不依赖新数据采集，基于现有 19 个微基准 trace
- 劣势：微基准没有处方空间（见 Section 1），无法验证处方性诊断
- 仅作为特征提取流程的 sanity check

**决策：选项 A 是唯一能支撑目标达成的路径。** 微基准数据仅用于验证特征提取
脚本正确性，不作为处方性诊断的主要目标。

### 4.1 第一步：多维 Roofline 特征提取

**前置条件：第零步完成，能够在一个真实 AI workload 上采集 trace + NCU stats。**

#### 4.1.1 工作负载选择

**主要评估对象：GPT-2 decode**

已有实验框架 `experiments/gpt2_decode/`，支持 context length 128/512/1024。
在租用的 SM_89/SM_80/SM_90 机器上运行 `run_trace.sh` 生成 trace，用 NCU
采集硬件 stats。

**辅助参照：微基准 roof 上限**

微基准数据用来建立"各个 roof 的位置"：
- MaxFlops → 峰值 FLOPS roof
- mem_bw → 峰值 HBM 带宽 roof
- l1_bw_32f → 峰值 L1 带宽 roof
- l2_bw → 峰值 L2 带宽 roof
- shared_bw → 峰值 shared memory 带宽 roof
- l1_lat / l2_lat / mem_lat → 各级延迟 roof

这些 roof 上限值将作为"距离瓶颈"计算的分母，让 AI 能定量判断真实 workload
离每个 roof 多远。

#### 4.1.2 特征提取：三类输入

特征包现在有三个独立的来源，分别服务不同的推理目标。

**A. 压缩特征（描述 workload 行为结构）**

从现有 trace protobuf 文件中提取：

```json
{
  "compression_features": {
    "rle_coverage": 0.68,
    "rle_length_distribution": {"mean": 23.5, "p50": 18, "p95": 67},
    "cross_tb_offset_coverage": 0.92,
    "address_override_density": 0.03,
    "full_encoding_fallback_rate": 0.016,
    "warp_diff_distribution": {"mean": 12.3, "p50": 8, "p95": 45},
    "shared_pc_sequence_length": 1847,
    "stall_count_distribution": {"mean": 2.5, "std": 1.8, "p95": 6},
    "yield_rate": 0.23,
    "barrier_wait_rate": 0.11
  }
}
```

这些特征帮助 AI 理解 workload 的**结构性特征**——规则性、控制流、依赖模式。

**B. 硬件性能指标（描述 workload 当前执行质量）**

从 NCU CSV 解析：

```json
{
  "hardware_stats": {
    "ipc": 45.2,
    "l1_miss_rate": 0.32,
    "l2_miss_rate": 0.08,
    "memory_throughput_gbps": 680,
    "compute_utilization_pct": 78,
    "occupancy_pct": 75,
    "shared_mem_bank_conflict_rate": 0.12,
    "warp_divergence_rate": 0.05
  }
}
```

**C. 多维 Distance-to-Roof 指标（描述 workload 距离每个瓶颈多远）**

这是新增的、处方性诊断的核心输入。对每个潜在瓶颈，计算 workload 的实际值
与该 roof 上限值的比值。

```json
{
  "distance_to_roof": {
    "compute": {
      "peak_flops_tflops": 82.6,
      "achieved_flops_tflops": 15.2,
      "utilization": 0.184,
      "roof_source": "MaxFlops microbench"
    },
    "hbm_bandwidth": {
      "peak_gbps": 1008,
      "achieved_gbps": 680,
      "utilization": 0.675,
      "roof_source": "mem_bw microbench"
    },
    "l1_bandwidth": {
      "peak_gbps": 14000,
      "achieved_gbps": 9200,
      "utilization": 0.657,
      "roof_source": "l1_bw_32f microbench"
    },
    "l2_bandwidth": {
      "peak_gbps": 5500,
      "achieved_gbps": 3100,
      "utilization": 0.564,
      "roof_source": "l2_bw microbench"
    },
    "occupancy": {
      "max_active_warps": 64,
      "achieved_active_warps": 48,
      "utilization": 0.75
    },
    "shared_mem_bandwidth": {
      "peak_gbps": 19500,
      "achieved_gbps": 0,
      "utilization": 0.0,
      "roof_source": "shared_bw microbench"
    }
  }
}
```

每一项包括：roof 上限（分母）、当前达到值（分子）、利用率、roof 来源说明。
**最高利用率项就是当前的主要瓶颈**。

#### 4.1.3 Roof 上限的获取

Roof 上限值来源：
- **从微基准直接测量**：MaxFlops、mem_bw、l1_bw 等已有的微基准数据，直接读 NCU
- **从架构规格表**：HBM 带宽、SM 数、peak FLOPS 等厂商标称值
- **从模拟器配置**：如果在模拟器上做研究，从 `gpgpusim.config` 读 roof 参数

微基准实测值优于规格表值——实测反映实际可达上限，规格表反映理论上限。

#### 4.1.4 模拟器可修改参数清单

为了让 AI 的处方建议可以落地为实际模拟器配置修改，需要预先整理出模拟器中
"可以被修改的参数"清单，作为 AI 建议空间的约束。初步清单（来自
`gpgpu-sim/configs/tested-cfgs/SM86_RTX3080/gpgpusim.config`）：

| 参数类别 | 参数名示例 | 含义 |
|---------|----------|------|
| Cache | `-gpgpu_cache:dl1` | L1 数据缓存大小、关联度、行大小 |
| Cache | `-gpgpu_cache:dl2` | L2 缓存配置 |
| Scheduler | `-gpgpu_num_sched_per_core` | 每 SM 的 warp scheduler 数 |
| Scheduler | `-gpgpu_scheduler` | scheduler 策略（lrr / gto / ...）|
| Memory | `-gpgpu_dram_bandwidth` | HBM 带宽 |
| Memory | `-gpgpu_dram_partition` | 内存分区数 |
| Pipeline | `-gpgpu_operand_collector_num_units_sp` | operand collector 单元数 |
| Resource | `-gpgpu_shmem_size` | shared memory 大小 |
| Resource | `-gpgpu_shader_registers` | 寄存器文件大小 |

这个清单会在第零步准备好模拟器环境时细化，定义 AI 可以"提出修改"的边界。

### 4.2 第二步：处方性诊断 + 初步闭环

此步骤将 AI 诊断、处方建议、实际验证三件事合并成一个完整循环。

#### 4.2.1 处方性诊断的输出格式

AI 的输出不再是"描述 workload 做什么"，而是以下格式的处方报告：

```
## Kernel: {name}

### 当前状态
- 分类：{compute/memory/latency-bound}
- 主要瓶颈：{roof name}，利用率 {X%}
- 次要瓶颈：{roof name}，利用率 {Y%}

### 处方 1: {建议标题}
**诊断依据：** {从哪些特征看出来的}
**修改内容：** {具体模拟器参数名 + 新值}
**预期效果：** {哪个指标会变，变多少}
**预期代价：** {硬件面积/功耗/其他维度的代价}
**验证方法：** {要看哪个指标的前后对比，什么阈值算成功}

### 处方 2: ...

### 置信度说明
- 处方 1: HIGH/MEDIUM/LOW，原因：{理由}
- 处方 2: ...
```

每条处方必须有四个要素：**改什么、为什么、预期效果、如何验证**。
缺任何一个都不算处方，只是模糊建议。

#### 4.2.2 闭环验证流程

从 AI 产出的处方列表中挑 1-3 条做实际验证：

1. **筛选标准**：HIGH 置信度优先，修改成本最低的优先
2. **修改模拟器配置**：按处方修改 `gpgpusim.config` 或等价参数
3. **重跑模拟器**：同一 workload，新旧配置各跑一次
4. **对比输出**：收集关心的 metric（IPC、cache miss、throughput 等）
5. **AI 自评估**：把前后对比数据丢回 AI，让 AI 判断处方是否生效，原因是否
   与预期一致

#### 4.2.3 成功判据

处方性诊断能力的成功标志是：

- **至少 1 条处方产出可度量的性能变化**（好或坏都算，只要不是噪声水平）
- **AI 的预期方向与实测方向一致**（预期 IPC 上升，实测 IPC 确实上升）
- **AI 能解释意外结果**（如果实测结果与预期相反，AI 能从数据反推原因）

如果所有处方都在噪声水平内，说明 AI 的建议空间与实际敏感参数不匹配，
需要重新调整参数清单或特征集。

#### 4.2.4 评估表

每条处方都填一张评估表：

| 项 | 内容 |
|---|------|
| 处方内容 | |
| 预期方向 | |
| 实测方向 | |
| 方向是否一致 | |
| 数值量级是否接近预期 | |
| AI 的事后解释质量 | |
| 归类 | 成功 / 方向对但量级错 / 方向错 / 无变化 |

### 4.3 第三步：Squash/Delta 语义增强（条件性）

此步骤以第二步显示"处方性诊断思路可行"为前提。仅在以下情况下推进：
- 第二步产出了至少 1 条方向正确的处方
- 第二步暴露出特征缺失导致 AI 无法识别的瓶颈类型
- 这些盲区可以映射到 Squash/Delta 能提供的语义

如果第二步的处方准确度已经足够，可以跳过第三步直接扩展 workload 广度。

#### 4.3.1 Squash 用于行为分段

**目的：** 让 AI 能识别 workload 的阶段切换，比如 prefill → decode、
attention → FFN、不同 batch 之间的过渡。

**机制：** 对 TB 或 kernel 序列做时序合并，输出分段边界和每段的特征。

**产出的语义特征：**

| 特征 | 描述 | 对处方的价值 |
|------|------|------------|
| `segment_boundaries` | 行为发生显著变化的位置 | AI 可以对不同段分别提处方 |
| `segment_lengths` | 每个稳定阶段持续多长 | 判断哪个阶段是优化重点 |
| `segment_dominant_bottleneck` | 每段的主要瓶颈 | 不同段可能需要不同方向的修改 |

#### 4.3.2 Delta 用于异常检测

**目的：** 识别行为离群的 warp/TB/kernel，让 AI 能提出针对性的处方
（比如"边界 TB 的特殊处理导致 divergence，建议调整 tile 尺寸"）。

**产出的语义特征：**

| 特征 | 描述 | 对处方的价值 |
|------|------|------------|
| `outlier_units` | 偏离群体的单元 | AI 能指出具体问题单元 |
| `hot_fields` | 跨单元频繁变化的字段 | 识别行为差异的维度 |
| `field_correlation` | 一起变化的字段 | 暴露潜在的因果链 |

#### 4.3.3 重新走闭环

用增强特征重新执行第二步的处方 + 闭环验证，对比是否：
- 产出了第二步没发现的新处方
- 原有处方的置信度或预期准确度提升
- 处方覆盖的瓶颈类型更广

## 5. 关键设计决策

### 5.1 仅使用高层语义

AI 诊断特征建立在架构无关的行为维度上：
- 访存模式（coalescing、局部性、规则性）
- 控制流 divergence（warp 级、TB 级）
- 结构相似度（TB 间、kernel 间）
- 资源利用率（计算、内存带宽、occupancy）

指令级和 control bits 特征作为补充，不作为基础。
这确保框架不会被锁定在某个特定的 NVIDIA SM 代际上。

### 5.2 压缩是手段，不是目的

引入 difftest 压缩机制（Squash、Delta）是因为它们能产出有用的语义特征，
而不是为了压缩率。如果某个机制压缩率好但没有产出有用的语义，则不在范围内。
如果某个机制压缩率差但产出有价值的语义，则在范围内。

压缩率是否重要的决策将在原型结果出来后重新审视。

### 5.3 先离线，后在线

第一步和第二步完全是离线的（仿真后分析）。在线分析（仿真过程中）
是未来方向，只有在离线分析证明诊断方法有效后才会推进。

演进路径：离线 → per-kernel 批处理 → per-TB 流式 → 实时

### 5.4 单一工作负载，深入理解

第一次验证使用一个 AI 工作负载，选择标准是深度熟悉。
广度（多工作负载、多架构）在深度证明方法可行之后再扩展。

## 6. 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| **无法获取可用硬件环境**（5090+NVBit 不通，无 sudo）| 已发生 | 阻塞 | 租云服务器（A100/H100/RTX4090）|
| AI 无法基于 distance-to-roof 特征提出可验证的处方 | 中 | 高——动摇核心前提 | 第二步的早期闭环验证能快速暴露这点；一个处方失败就要重新评估特征集 |
| AI 处方"方向对但量级错得离谱" | 中 | 中——说明 AI 对硬件定量关系不理解 | 加强 prompt 中的硬件知识，提供更多参数的"1% 变化对应性能多少%变化"的参照 |
| 模拟器对 AI 修改的参数不敏感 | 中 | 中 | 参数清单要预先用人工实验筛选，保留对结果敏感的参数 |
| Squash/Delta 特征无法改善处方 | 中 | 低——如果描述性特征已足够，跳过第三步就好 | 第二步必须先验证 baseline 处方能力 |
| 架构特定的发现无法泛化 | 低 | 当前低——单架构验证对第一篇论文已足够 | 明确限定 claim 范围；泛化是后续工作 |
| 闭环建议是错的或有害的 | 中 | 低——模拟器可以安全实验 | 始终对比前后；错误建议本身就是数据点，不是失败 |

## 7. 交付物

### 第零步：基础设施
- [x] 压缩特征提取脚本（已完成，在 5090 微基准 trace 上验证）
- [x] NCU CSV 解析脚本（已完成）
- [x] Feature merge 脚本（已完成）
- [x] 诊断 prompt 模板（已完成）
- [x] 微基准 sanity check 报告（已完成，揭示了 stall_count 假设的错误）
- [ ] 可用的 GPU 环境：NVBit 支持 + NCU 权限 + 模拟器能跑
- [ ] 模拟器可修改参数清单（Section 4.1.4）

### 第一步：多维 Roofline 特征
- [ ] 微基准的 NCU 采集，建立 roof 上限参照表
- [ ] distance-to-roof 计算脚本
- [ ] 真实 workload（GPT-2 或替代）的 trace + NCU 采集
- [ ] 完整特征包生成（压缩特征 + 硬件 stats + distance-to-roof）

### 第二步：处方性诊断 + 闭环验证
- [ ] 处方性诊断 prompt 模板（替代原诊断模板）
- [ ] 初版处方报告（至少 3 条处方）
- [ ] 选中处方的模拟器配置修改脚本
- [ ] 前后对比的自动化脚本
- [ ] AI 自评估 prompt 模板
- [ ] 至少 1 条处方的完整闭环验证结果
- [ ] 处方评估表

### 第三步（条件性）：Squash/Delta 语义增强
- [ ] Squash 后处理脚本
- [ ] Delta 后处理脚本
- [ ] 增强特征包
- [ ] 增强后的处方对比
- [ ] 增强后的闭环验证对比

## 8. 与现有文档的关系

- `docs/ai-agent-input-requirements-survey.md`：文献调研，
  指导本规格中的特征选择
- `docs/trace-compression-for-microbench-agent.md`：早期的压缩即特征分析；
  本规格取代其范围
- `docs/ai-workload-driven-workflow.md`：更广泛的研究工作流；
  本规格实现该工作流的"机理层"

## 9. 待决问题

1. ~~第一步使用哪个具体的 AI 工作负载？~~ **已确定：**
   GPT-2 decode（主要评估）+ 已有微基准（sanity check）。

2. Squash 合并的相似度阈值设为多少？这是经验参数——
   从几个值开始尝试，看哪个产出最具可解释性的分段。

3. Delta 应该在 TB 级、warp 级还是两者都操作？
   先从 TB 级开始（更粗、更易验证），如果 TB 级结果有希望再扩展到 warp 级。

4. 如果压缩率和语义质量发生冲突，如何处理？
   推迟决策——先确定它们在实践中是否真的冲突。
