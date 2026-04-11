# 诊断报告：mini-transformer v4 [E3_delta]

**日期：** 2026-04-11
**硬件：** RTX 3080 Ti (SM_86)
**启用机制：** 仅 Delta

---

## Delta 输出摘要

### Kernel 级字段温度（HOT，温度 > 0.2）

| 字段 | 温度 | 含义 |
|------|:----:|------|
| dynamic_shmem_per_block | 0.957 | softmax（1024B）vs 其余 0 |
| static_shmem_per_block | 0.860 | gemm(2048B), attention(8192B) vs 其余 0 |
| num_tbs | 0.858 | 各 kernel launch 规模差异大 |
| total_dynamic_instructions | 0.826 | GEMM 指令数最多 |
| waves_per_sm | 0.768 | attention(25.6) vs residual(3.2) |
| l1_hit_rate_pct | 0.750 | context_mul(88.5%) vs gemm(8.4%) |
| dram_throughput_pct | 0.714 | residual(58.3%) vs 其余 < 10% |
| warp_cycles_per_issued_inst | 0.452 | residual(87.6) vs softmax(21.8) |

### Kernel 级字段（COLD，温度 ≤ 0.1）

| 字段 | 温度 | 含义 |
|------|:----:|------|
| uses_fp64 | 0.000 | 无 kernel 使用 FP64 |
| total_static_instructions | 0.000 | 静态指令数在同类 kernel 中恒定 |
| num_barriers | 0.000 | barrier 数无法区分 kernel |

### 关键相关性（|r| ≥ 0.80）

| 字段对 | 相关系数 | 解读 |
|--------|:-------:|------|
| mem_pipes_busy ↔ compute_throughput | **+1.000** | 计算与内存流水线完全协同 |
| mem_pipes_busy ↔ l1_throughput | **+0.991** | compute-bound kernel 同时饱和 L1 |
| compute_throughput ↔ l1_throughput | **+0.991** | 同上 |
| l1_throughput ↔ block_limit_registers | **-0.964** | 寄存器限制越严（值小），L1 吞吐越高 |
| compute_throughput ↔ block_limit_registers | **-0.946** | 同上 |
| mem_pipes_busy ↔ block_limit_registers | **-0.946** | 同上 |
| dram_throughput ↔ block_limit_registers | **+0.957** | DRAM 受限 kernel 的寄存器限制反而宽松 |
| dram_throughput ↔ l1_throughput | **-0.893** | 内存三态分解信号 |
| warp_cycles ↔ ipc_active | **-0.925** | warp 等待时间越长，IPC 越低 |
| l2_hit_rate ↔ block_limit_registers | **-0.942** | 寄存器充足的 kernel L2 命中率更高 |

---

## 相对 v1 的关键变化

### 消失的反相关：l1_hit ↔ compute（v1 = -0.646，v4 = ？）

v1 中最反直觉的发现——L1 命中高的 kernel 计算吞吐反而低——在 v4 中消失了：
attention_score 修复后 L1 命中率从 97.2% 降至 7.1%（改用 shmem），
compute 从 22.4% 升至 95.2%，破坏了该反相关的基础数据点。

**这本身是一个重要的元信号：** v1 中的 l1_hit ↔ compute 反相关，
事实上是 attention_score 软件缺陷（LDG 延迟）在 Delta 层面的体现，
而非 workload 的固有架构特性。软件修复后，该反相关消失印证了这一解读。

### 新出现的强相关：block_limit_registers 与多个指标

v4 中 `block_limit_registers` 成为最重要的关联字段：
- 与 l1_throughput：-0.964
- 与 compute_throughput：-0.946
- 与 dram_throughput：+0.957

**解读：** 寄存器限制严格（值小=每个 block 可以占据更多寄存器）的 kernel（gemm=6, attention=6）是
compute/L1 密集型；寄存器限制宽松（值大）的 kernel（residual=16）是 DRAM 密集型。
这个相关性在软件清洗后才变得清晰——v1 中 attention_score 的异常数据点会干扰此规律。

### 内存三态（v4 更新）

`l1_throughput ↔ dram_throughput = -0.893`（仍然强反相关，v1=-0.940）：
三态结构依然成立，但 attention_score 现在归入"L2 驻留"类（与 gemm 相似），
不再是"L1 驻留"类的极端代表。

---

## 处方（Delta 对 Stage C 的信息增强）

| 发现 | Stage C 含义 |
|------|-------------|
| block_limit_registers 是最强关联字段 | 寄存器文件配置是模拟器校准的第一优先级 |
| dram ↔ block_limit 正相关 | DRAM 受限和 compute 受限 kernel 的寄存器配额需分开建模 |
| 三态内存结构仍成立 | L1/L2/HBM 三种带宽模型需独立校准 |
| l1_hit ↔ compute 反相关消失 | 确认该信号是软件伪影，v4 是干净的架构信号 |

**判定：发现性（Discovering）**
Delta 在 v4 上提供了 E0 不可直接推导的架构洞察——特别是
`block_limit_registers` 的中心地位，以及 v1 伪相关的消失这一元信号。
