# Trace 压缩结构与 AI AGENT 生产 Microbench 的结合价值

日期：2026-04-05

## 背景

本文档梳理两个项目中已有的压缩手段，并分析这些手段对"AI AGENT 生产 microbench"这一课题的潜在价值。

涉及的两个项目：

- `modern-gpu-simulator-micro-2025`：面向 GPU trace 的多层次压缩体系
- `difftest`：面向 RTL 仿真状态传输的增量压缩体系

---

## 一、两个项目的压缩手段梳理

### 1.1 modern-gpu-simulator-micro-2025

该项目对 GPU SASS trace 实施了三层压缩结构，定义在 `simulator-remodeled/util/traces_enhanced/dynamic_trace/compressed_threadblock.proto`。

#### 层级 1 — 指令级（`compressed_instruction`）

- **PC delta 编码**：PC 存储相对上条指令的偏移，而非绝对地址（`flags` 的 bit 2 控制）
- **字段省略**：`active_mask = 0xFFFFFFFF` 时不存储（bit 0），谓词 mask 等于 active mask 时不存储（bit 1）
- **RLE 游程编码**（`instruction_run`）：对连续的、无内存地址、flags 相同、PC delta 固定的指令序列，只存一条记录 `(pc_start, pc_delta, flags, count)`

#### 层级 2 — Warp 间（`compressed_threadblock_v7`）

- **shared_pc_sequence**：同一 threadblock 内所有 warp 共享一个 PC 执行序列（利用 SIMT 同步特性）
- **warp_diff**：每个 warp 只记录与共享 PC 序列的差异，包括分叉点覆盖（`pc_overrides`）和每条指令各自的地址与 mask

#### 层级 3 — Threadblock 间（`compressed_kernel_v8`）

- **base + delta 结构**：选定一个基准 threadblock，其余 threadblock 只记录与基准的差异
- **global_address_offset**：大量 threadblock 的访存模式完全一致，只有基地址偏移，用一个整数表示
- **address_override**：只存真正不同的少量地址覆盖项
- **is_full_encoding**：标记某个 delta 是否退化为完整编码（无法用 delta 表示时的兜底）

### 1.2 difftest

difftest 的压缩作用在 RTL 仿真的 DUT 状态传输链路上，核心模块为 `Squash`、`Batch`、`Delta`。

| 模块 | 机制 | 本质 |
|------|------|------|
| **Squash** (`Squash.scala`) | 将多个时钟周期内的 DiffTest 事件合并，等到不可压缩时才 tick | 时序批量化，减少 DPI-C 调用次数 |
| **Batch** (`Batch.scala`) | 将 N 个周期的 DUT 状态打包成一次大传输，超出容量时分批发出 | 空间批量化，摊薄传输 overhead |
| **Delta** (`Delta.scala`) | 对寄存器堆等大型结构，只传发生变化的字段；对物理寄存器堆配合 RAT 做 filter，跳过本轮无效更新 | 增量编码 / 稀疏传输 |

---

## 二、这套压缩手段对"AI AGENT 生产 microbench"的价值

### 2.1 压缩结构是 workload 行为特征的隐式编码

这套压缩体系不只是存储优化，它实际上定义了一套**结构化的 workload 行为特征空间**。

| 压缩特征 | 揭示的 workload 行为 |
|----------|---------------------|
| `instruction_run` 的 count 分布 | 计算密集度、循环 pattern 的规律性 |
| `global_address_offset` 的覆盖率 | threadblock 间访存模式的一致性（规则访存 vs 不规则） |
| `address_override` 的密度 | 数据相关的访存分叉程度 |
| `warp_diff` 的非零条目数 | 控制流分叉（warp divergence）的频率 |
| Squash 合并率 | 仿真流中事件的密集程度 |
| Delta 稀疏度 | 寄存器文件的实际更新频率 |

这些特征正是设计 microbench 所需要的核心维度。AI AGENT 如果能解析 trace 的压缩结构（而非原始 trace），就能以极低成本归纳出目标 AI workload 的硬件行为特征。

### 2.2 压缩率可作为 microbench 质量的相似度代理指标

AI AGENT 生产 microbench 面临的核心难题是：**如何判断生成的 microbench 是否真正复现了原 AI workload 的硬件行为？**

完整模拟器的端到端验证成本很高。一个低开销的代理指标是：

> 如果 AI AGENT 生成的 microbench，其 trace 的压缩结构（RLE 分布、delta 密度、cross-TB delta 比例）与目标 AI workload 的 trace 压缩结构相似，则两者的执行行为大概率相似。

这只需要运行 tracer + 分析压缩特征，成本比端到端模拟低 1-2 个数量级。

### 2.3 一个可操作的闭环

```
AI workload trace
      |
      v
分析压缩特征向量
(RLE 长度分布, delta 密度, address override 率, warp divergence 频率, ...)
      |
      v
AI AGENT 以特征向量为输入，生成 microbench 参数或代码
      |
      v
对 microbench 做同样的 trace 压缩分析，提取特征向量
      |
      v
对比两个特征向量的相似度 → 作为 reward signal 反馈给 AGENT
      |
      v
（可选）相似度达标后，再用模拟器做精确验证
```

这套方案的优势：

- 不需要运行完整模拟器就能完成大部分筛选，降低迭代成本
- 特征向量是结构化的，适合作为 LLM AGENT 的 tool call 返回值
- 压缩框架已经存在，不需要从头构建特征提取基础设施

### 2.4 两个课题的数据互补关系

- `modern-gpu-simulator-micro-2025` 提供 trace 压缩框架 + 已有的 microbench trace 集合（`GPU_Microbenchmark`）
- AI AGENT 课题负责生产新的 microbench
- 两者共同的闭环是：**生成 → 压缩特征匹配验证 → 再生成**

`GPU_Microbenchmark` 现有的 trace 可以作为特征向量的参照集（即"已知 microbench 的行为空间"）。AI AGENT 的目标是生成覆盖 AI workload 特征向量的新 microbench，填补这个参照集的空白。

---

## 三、局限与风险

- **压缩格式覆盖范围**：`compressed_kernel_v8` 的 cross-TB delta 目前是最新版本，需要确认模拟器主流程是否已完整消费该格式，否则压缩特征提取链路不完整。
- **特征相似不等于行为完全一致**：active mask 分布、cache miss 率等指标需要配合模拟器统计（APEs 或 per-kernel stats）做二次验证，压缩特征相似只是必要条件，不是充分条件。
- **不规则 workload 的区分度下降**：对稀疏注意力、MoE routing 等访存高度不规则的 AI workload，cross-TB delta 压缩率本来就低，特征向量的信噪比下降，依赖压缩特征做相似度判断的有效性需要单独评估。

---

## 四、结论

这套压缩手段的进一步价值在于：它不只是 trace 存储优化，而是一套**结构化行为抽象框架**。对"AI AGENT 生产 microbench"课题来说，最直接的结合点有三个：

1. 把 trace 压缩结构作为 workload 行为的特征表示，替代昂贵的完整模拟
2. 把"生成 microbench 的 trace 压缩特征与目标 workload 匹配"作为 AGENT 的评估信号
3. 利用现有的 `GPU_Microbenchmark` trace 集合作为参照，让 AI AGENT 朝着"填补特征空间空白"的方向生成新 microbench
