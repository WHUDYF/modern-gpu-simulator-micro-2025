# 诊断报告：mini-transformer v4 [E2_batch]

**日期：** 2026-04-11
**硬件：** RTX 3080 Ti (SM_86)
**启用机制：** 仅 Batch

---

## Batch 输出：3 聚类 + 3 outlier

### 聚类

| 聚类 | Kernel | n | 凝聚度 |
|------|--------|:-:|:------:|
| A | gemm_tiled | 7 | 1.000 |
| B | residual_add | 2 | 1.000 |
| C | layernorm_kernel | 2 | 1.000 |

### Outlier（不属于任何聚类）

| Kernel | 特征描述 |
|--------|---------|
| attention_score | compute=95.2%，shmem=8256B，waves=25.6 |
| softmax_kernel | DRAM=41.2%，L1_hit=79.9%，dynamic_shmem |
| context_mul | L1_hit=88.5%，compute=89.6%，无 shmem |

---

## 相对 v1 的关键变化

**v1 outlier 原因：**
- attention_score：warp_cycles=174.7（极高），compute=22.4%（极低）→ 因软件问题异质
- softmax：waves=0.1，occupancy=16.6% → 因软件问题异质
- context_mul：良性 outlier（性能良好但特征与 GEMM 不同）

**v4 outlier 原因（三者均为良性）：**
- attention_score：现在 compute=95.2%，性能优秀；但 shmem=8256B 和 waves=25.6
  与 gemm（shmem=2048B，waves=4.76）差异足够大，无法归入同一聚类
- softmax：DRAM=41.2% 是所有 kernel 中最高的，归约模式独特
- context_mul：L1 驻留特征（l1_hit=88.5%）与 GEMM 的 L2 驻留模式不同

**关键结论：** v4 中 3 个 outlier 全部是**良性架构异质**，不再含软件问题信号。
Batch 的功能从"发现软件缺陷"转变为"识别架构特征不同的 kernel 类型"。

---

## 发现与判定

**发现：** outlier 结构与 v1 相同（同样 3 个），但成因完全不同——
v1 的 outlier 由软件缺陷驱动，v4 的 outlier 由真实架构差异驱动。

**对 Stage C 的指导：**
- 3 个聚类（gemm/residual/layernorm）可各用 1 个代表 kernel 做模拟器校准
- 3 个 outlier 需单独校准，各自对应不同的架构参数（寄存器/L2/HBM/L1）

**判定：确认性（Confirming）**，
但对 Stage C 的模拟复用规划提供了直接可操作的优先级排序。
