# AI 架构诊断协议 v2：三层过滤框架

**日期：** 2026-04-10
**更新原因：** mini-transformer E0 实验揭示 Stage B 存在软件问题与架构问题混淆，
attention_score 和 softmax 的高置信度发现均为软件实现缺陷，不是架构信号。

---

## 核心原则

**架构诊断的前提是软件层干净。**

软件层存在缺陷时，硬件的表现会被掩盖或失真。AI 在软件问题未清除的情况下
做出的"架构处方"实际上是在优化一个残缺的 workload，结论对架构研究没有意义。

---

## 三层诊断框架

```
输入：trace 特征 + NCU 硬件指标
         ↓
┌─────────────────────────────┐
│  Stage A：启动配置检查      │  ← 检测 grid/block 参数缺陷
│  判据：waves, occupancy     │
│  修复归属：CUDA 启动参数    │
└────────────┬────────────────┘
             │ 通过（或修复后重测）
             ↓
┌─────────────────────────────┐
│  Stage B：Kernel 实现检查   │  ← 检测算法实现层缺陷（新增）
│  判据：warp_cycles 异常     │
│  修复归属：CUDA 源码        │
└────────────┬────────────────┘
             │ 通过（或修复后重测）
             ↓
┌─────────────────────────────┐
│  Stage C：架构瓶颈诊断      │  ← 这才是研究目标
│  判据：pipeline/cache/带宽  │
│  修复归属：模拟器配置/架构  │
└─────────────────────────────┘
```

---

## Stage A：启动配置检查（原有，不变）

### 通过条件

| 指标 | 通过阈值 |
|------|---------|
| waves_per_sm | ≥ 1.0（建议 ≥ 4.0） |
| achieved_occupancy | ≥ 50% |

### 不通过时

- 归类为 **Class A 问题**
- 处方：调整 grid/block 维度，不涉及 kernel 逻辑
- **必须修复后重测，Stage C 的分析对 Class A 未通过的 kernel 无效**

---

## Stage B：Kernel 实现检查（新增）

### 目的

区分两类 warp stall 来源：
- **内存延迟**（合法的架构信号）：数据未就绪，warp 等待内存系统
- **计算串行化**（软件实现缺陷）：数据已就绪，但指令依赖链阻止发射

### 判据一：RAW 依赖链（计算串行化）

**触发条件（同时满足）：**
- `warp_cycles_per_issued_inst` ≥ 50
- `l1_hit_rate_pct` ≥ 70%（数据已就绪，排除内存延迟）
- `compute_throughput_pct` ≤ 50%（计算流水线未饱和）

**解读：** 数据从缓存取出很快，但计算流水线仍然空转。根因是指令间存在长链依赖，
最常见的模式是单累加器内层循环（每次迭代依赖上一次 FMA 结果）。

**重要：** 在应用此处方前，应先用 `cuobjdump -sass` 检查编译器是否已自动展开：
若 FFMA 数量 ≥ head_dim × 2，说明编译器（-O2）已生成多累加器，
手动展开无效——应转为判据四（LDG 软件流水线不足）。

**修复归属：** CUDA 源码——展开循环并引入多个独立累加器。

**验证方法：** 应用处方后，warp_cycles 应下降 ≥ 40%；若下降 < 10%，归入判据四。

---

### 判据四：LDG 软件流水线不足（新增，2026-04-11）

**触发条件（同时满足）：**
- `warp_cycles_per_issued_inst` ≥ 50
- `l1_hit_rate_pct` ≥ 70%（数据在 L1，非 L2/DRAM 延迟）
- `compute_throughput_pct` ≤ 50%（计算流水线未饱和）
- **判据一处方无效**（展开后 warp_cycles 下降 < 10%），
  或 SASS 中 FFMA 数量已 ≥ head_dim × 2（编译器已自动展开）

**解读：** 编译器已消除 RAW 依赖链，但 warp 仍在等待 L1 加载完成（SM_86 L1 hit
latency ≈ 28 cycles）。根因是 LDG 指令与 FFMA 之间缺乏足够深度的软件流水线——
没有足够的独立指令填充 28 cycle 的加载延迟窗口。

与判据一的区别：
- 判据一：编译器未展开，FMA 的目标寄存器直接依赖上一次 FMA 结果
- 判据四：编译器已展开，但 FMA 的源操作数（LDG 结果）还未就绪

**修复归属：** CUDA 源码——
1. 将 inner loop 数据（Q/K 行、K 列等）预加载到 shared memory，再从 shared memory 计算
2. 使用 Ampere `cp.async` 异步预取，在计算第 t 轮时异步加载第 t+1 轮数据
3. 手动循环展开 ≥ 2×（使两轮迭代的加载和计算交错执行）

**例：** attention_score v2（warp_cycles=169.7，l1_hit=97.2%，compute=22.4%，
SASS 中已有 71 FFMA + 98 LDG，4-wide 手动展开无效）

---

### 判据二：共享内存 bank conflict（可能是软件问题）

**触发条件（同时满足）：**
- `warp_cycles_per_issued_inst` ≥ 30（中等偏高）
- `l1_throughput_pct` 高（shared memory 和 L1 共享同一吞吐计量）
- `l1_hit_rate_pct` 低（排除 L1 cache 命中解释）
- `uses_shared_memory = 1`

**解读：** warp 在 shared memory 访问上等待，可能是 bank conflict。
需进一步确认访问 pattern（步长是否为 bank 数的倍数）。

**修复归属：** 视访问 pattern 而定——可能是软件（改变 tile 布局），
也可能是架构（调整 bank 数量），需要区分。

**注意：** `uses_shared_memory` 特征提取 bug 已于 2026-04-10 修复——
现同时检查 `static_shmem_per_block` 和 `dynamic_shmem_per_block`，可靠使用。

---

### 判据三：寄存器溢出到 L2/HBM（软件优化级别问题）

**触发条件（同时满足）：**
- `block_limit_registers` 极小（≤ 2）
- `achieved_occupancy` 远低于理论值
- `dram_throughput_pct` 异常高（超出 workload 的访存需求预期）

**解读：** 编译器寄存器分配过多，导致寄存器溢出（register spilling），
部分数据写入 L2 或 HBM。

**修复归属：** CUDA 源码——使用 `__launch_bounds__` 约束寄存器用量，
或拆分 kernel 降低局部变量数量。

---

### Stage B 通过条件

所有 kernel 满足以下条件：
- 无 RAW 依赖链判据触发
- 无寄存器溢出判据触发
- 共享内存 bank conflict 已确认不存在或已修复

### 不通过时

- 归类为 **Class B 问题**（软件实现缺陷）
- 处方：修改 CUDA 源码
- **必须修复后重新采集 NCU，Stage C 的分析对 Class B 未通过的 kernel 无效**
- 修复后，重新从 Stage A 开始走完整流程

---

## Stage C：架构瓶颈诊断（原 Stage B）

### 前提

Stage A 和 Stage B 均通过，或所有未通过 kernel 已修复并重测。

### 诊断目标

此时 workload 的软件层是干净的，观察到的瓶颈才能归因于硬件架构决策：

| 现象 | 架构归因候选 |
|------|------------|
| DRAM 利用率高但低于理论峰值 | HBM 带宽配置、内存控制器数量 |
| warp_cycles 高 + L2 miss 高 | L2 cache 容量、替换策略 |
| IPC 低 + compute 低 + warp 等待 memory | SM 内存流水线深度、MSHR 数量 |
| occupancy 受限于寄存器 | 寄存器文件大小 |
| DP/SP 吞吐比不匹配 workload 需求 | pipeline 配置（initiation interval） |

### 处方归属

Stage C 处方的修复目标是**模拟器配置参数**，而非源码：
- `trace_opcode_latency_initiation_*`
- `gpgpu_cache:dl1/dl2`
- `gpgpu_shmem_num_banks`
- `gpgpu_n_warp_per_shader`
- 等

---

## 软件问题分类表

| 类别 | 触发来源 | 修复对象 | Stage |
|------|---------|---------|-------|
| Class A-1：启动参数 | waves < 1，occupancy < 50% | grid/block 配置 | A |
| Class B-1：RAW 依赖链 | warp_cycles 高 + L1 命中高 + compute 低 | 累加器展开，循环拆分 | B |
| Class B-2：Bank conflict | warp_cycles 中高 + shared memory 使用 | Tile 布局，padding | B/C 边界 |
| Class B-3：寄存器溢出 | block_limit_registers 极小 + DRAM 异常 | `__launch_bounds__`，kernel 拆分 | B |
| Class C：架构瓶颈 | 软件干净后的残余 stall | 模拟器配置 | C |

---

## 流程应用示例

### mini-transformer（修复前）

| Kernel | Stage A | Stage B | Stage C |
|--------|---------|---------|---------|
| softmax | **FAIL**（Class A-1：waves=0.1） | — | — |
| attention_score | PASS | **FAIL**（Class B-1：warp_cycles=174.7，l1_hit=97.2%） | — |
| gemm_tiled | PASS | PASS（warp_cycles=36.3，在可接受范围） | **进入 Stage C** |
| context_mul | PASS | PASS | **进入 Stage C** |
| residual_add | PASS | PASS | **进入 Stage C** |
| layernorm | PASS | PASS | **进入 Stage C** |

### mini-transformer（修复后，预期）

修复 softmax 和 attention_score 后，重测 NCU：
- softmax：waves 应 ≥ 77，Stage A 通过；warp_cycles 应回落，Stage B 通过
- attention_score：warp_cycles 应从 174.7 降至 ~40，Stage B 通过

此时 6 个 kernel 均进入 Stage C，AI 诊断的结论才对架构研究有意义。

---

## 与方法论文档的关系

本协议更新了：
- `docs/superpowers/specs/2026-04-06-trace-semantic-ai-diagnosis-design.md`
  中的 Stage B 定义（原文档的 Stage B 等同于本文档的 Stage C）
- `experiments/baseline_diagnosis/schemas/diagnosis_template.md`
  中的诊断模板结构

后续所有消融实验报告（E0-E4）应在报告头部注明每个 kernel 通过的 Stage 层级，
未通过 Stage A/B 的 kernel 不应产出 Stage C 处方。
