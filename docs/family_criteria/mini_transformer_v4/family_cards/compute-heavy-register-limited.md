# Family Card: `compute-heavy -> register-limited`

## Core Explanation

这个 family 用来容纳那些整体上明显 `compute-heavy`，且其最强共享架构解释落在寄存器 / occupancy 约束上的 kernel。

当前第一版中，它的主解释不是“它们都是矩阵乘”，而是：

**它们都位于高计算吞吐区，并共享最严格的一档寄存器限制信号。**

## Representative Kernels

- `gemm_tiled`
- `attention_score`（弱共享纳入）

## Boundary Conditions

- 不纳入以 DRAM 带宽为主导的 kernel，例如 `residual_add`
- 不纳入以 cache-capacity / locality 为主导的 memory-side kernel，例如 `softmax_kernel`、`context_mul`
- `attention_score` 只能在保留 shared-memory-coupled 说明的前提下纳入，不能被视为纯 GEMM 副本

## Uncertainty

- `attention_score` 属于弱共享样本，而不是完全同质样本
- 如果后续 shared-memory 特征被证明会改变验证主线，则它可能从该 family 中拆出

## What It Is Not

- 它不是“所有高 compute kernel 的集合”
- 它也不是“所有 attention 相关 kernel 的集合”
- 它是“高 compute 且主解释由 register / occupancy 主导”的 family

## Validation Meaning

- 当前阶段可优先共享寄存器 / occupancy 主线的验证思路
- 但不能因此删除 `attention_score` 的次级 shared-memory 差异记录
