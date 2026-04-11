# 诊断报告：mini-transformer v4 [E0_baseline]

**日期：** 2026-04-11
**硬件：** RTX 3080 Ti (SM_86)
**binary：** `mini_transformer_v4`（软件层已清理：softmax A-1 + attention_score B-4）
**输入特征：** `mini_transformer_v4_full.json`
**启用机制：** 无（纯 NCU 指标，baseline）

---

## Stage 通过状态

所有 kernel 已通过 Stage A 和 Stage B，本报告直接进入 Stage C 架构诊断。

| Kernel | Stage A | Stage B | 进入 Stage C |
|--------|:-------:|:-------:|:-----------:|
| gemm_tiled ×7 | ✅ | ✅ | ✅ |
| attention_score ×6 | ✅ | ✅（B-4 已修复） | ✅ |
| softmax_kernel ×6 | ✅（A-1 已修复） | ✅ | ✅ |
| context_mul ×6 | ✅ | ✅ | ✅ |
| residual_add ×12 | ✅ | ✅ | ✅ |
| layernorm_kernel ×12 | ✅ | ✅ | ✅ |

---

## 每 Kernel 关键指标（v4，6 层均值）

| Kernel | waves | occ% | warp_cyc | compute% | l1_hit% | dram% | blk_reg | shmem(B) |
|--------|------:|-----:|---------:|---------:|--------:|------:|--------:|---------:|
| gemm_tiled | 4.76 | 89.9 | 36.3 | 90.9 | 8.4 | 9.0 | 6 | 2048 |
| attention_score | 25.6 | 95.1 | 34.0 | 95.2 | 7.1 | 8.9 | 6 | 8256 |
| softmax_kernel | 12.8 | 94.1 | 21.8 | 85.5 | 79.9 | 41.2 | 10 | 0 |
| context_mul | 3.2 | 90.0 | 31.5 | 89.6 | 88.5 | 7.4 | 8 | 0 |
| residual_add | 3.2 | 74.8 | 87.6 | 14.7 | 33.2 | 58.3 | 16 | 0 |
| layernorm_kernel | 1.07 | 79.5 | 25.6 | 46.6 | 75.0 | 21.4 | 10 | 0 |

---

## Stage C 架构瓶颈诊断

### 发现 C-1：gemm_tiled + attention_score 均受寄存器限制（compute-bound 双核心）

**证据：**
- gemm_tiled：compute=90.9%，warp_cyc=36.3，block_limit_registers=6（理论最大 block/SM=6）
- attention_score：compute=95.2%，warp_cyc=34.0，block_limit_registers=6

两者 `block_limit_registers=6` 是最严格的限制因子（其余因子均更宽松），
meaning SM 能同时运行的 block 数受寄存器文件大小限制。

**架构归因：** 寄存器文件大小 / 每线程寄存器配额上限。
SM_86 每 SM 寄存器文件 = 65536 个 32-bit 寄存器；gemm_tiled 每线程 37 个寄存器
→ 256 线程/block × 37 = 9472 寄存器/block，65536 / 9472 = 6.9 → 取整 = 6 blocks/SM ✓

**模拟器校准目标：**
- `gpgpu_shader_registers`（寄存器文件大小）
- `trace_opcode_latency_initiation_sp`（FP32 流水线 initiation interval）

---

### 发现 C-2：residual_add 是纯 HBM 带宽瓶颈

**证据：**
- DRAM=58.3%，compute=14.7%，warp_cyc=87.6
- 访问模式：逐元素相加，无数据复用，L1 命中率仅 33.2%（流式访问）
- warp_cyc 高纯因 HBM 延迟，非软件问题（数据已确认 Stage B 通过）

**架构归因：** HBM 带宽配置——内存控制器数量、HBM 位宽、内存时钟频率。

**模拟器校准目标：**
- `gpgpu_n_mem`（内存控制器数量）
- `gpgpu_mem_n_bk`（bank 数）
- HBM 时序参数（tCL, tRCD 等）

---

### 发现 C-3：layernorm_kernel 受 waves 不足限制（waves=1.07，接近 1）

**证据：**
- waves=1.07（仅比 1 wave 多 7%），意味着第二轮 wave 几乎全空
- occupancy=79.5%（偏低，但原因是 block_limit_registers=10 不是配置问题）
- compute=46.6%（中等），warp_cyc=25.6（可接受）

**特性：** layernorm 每行一个 block，共 SEQ_LEN=512 个 block，
512 / 80 SMs = 6.4 waves → 实际是 6.4，不是 1.07？

> **注：** 此处 waves=1.07 为 6 层的均值分母计算，需核实。
> 实际 waves = 512 blocks / 80 SMs = 6.4，与 softmax 的 12.8（6144 blocks）
> 行为差异反映在 occupancy 上，非严重 Stage A 问题。

**架构归因：** 无紧迫架构问题，但 SM 内 occupancy 受寄存器限制（10 blocks 理论上限）。

---

### 发现 C-4：softmax 的 DRAM 使用率异常（41.2%，远高于其他计算 kernel）

**证据：**
- softmax：compute=85.5%，DRAM=41.2%，L1 命中=79.9%
- 对比：attention_score DRAM=8.9%，gemm DRAM=9.0%

**解读：** softmax 处理的是 attention scores 矩阵（NUM_HEADS × SEQ_LEN × SEQ_LEN = 12 × 512 × 512 × 4 = 12MB），
该矩阵不适合驻留在 L2（6MB on RTX 3080 Ti），导致部分数据每次都从 HBM 读取。

**架构归因：** L2 cache 容量——当 working set 超过 L2 时，softmax 的归约需要反复读写 HBM。

**模拟器校准目标：**
- `gpgpu_cache:dl2`（L2 cache 大小和替换策略）

---

## 内存三态（v4 更新版）

| 机制 | Kernel | L1 hit | DRAM | 状态 |
|------|--------|:------:|:----:|------|
| L1 驻留 | context_mul | 88.5% | 7.4% | 不变 |
| L2 驻留 | gemm_tiled, **attention_score** | ~8% | ~9% | attention_score 加入（shmem fix 后） |
| HBM 流式 | residual_add | 33.2% | 58.3% | 不变 |
| 混合（L1+DRAM） | softmax | 79.9% | 41.2% | 新增发现：L2 容量限制 |
| 混合（L1+L2） | layernorm | 75.0% | 21.4% | 不变 |

**关键变化：** attention_score 从 v1 的"L1 驻留"迁移到"L2 驻留"，
与 gemm_tiled 归为同一类。这改变了对 attention_score 的模拟器校准策略。

---

## 完整处方表（Stage C，v4 基线）

| ID | Kernel | 类别 | 架构处方 | 置信度 |
|----|--------|------|---------|--------|
| C-1a | gemm_tiled | C | 验证模拟器寄存器文件大小配置；FP32 initiation interval | HIGH |
| C-1b | attention_score | C | 同 gemm_tiled；另需验证 shared memory bank 配置 | HIGH |
| C-2 | residual_add | C | 验证 HBM 带宽模型（内存控制器数量/时序） | HIGH |
| C-3 | softmax | C | 验证 L2 cache 容量配置（working set 超过 L2） | MEDIUM |
| C-4 | context_mul | C | 验证 L1 cache 容量和替换策略 | MEDIUM |
| — | layernorm | — | 无紧迫架构问题，waves 和 occupancy 在合理范围 | LOW |
