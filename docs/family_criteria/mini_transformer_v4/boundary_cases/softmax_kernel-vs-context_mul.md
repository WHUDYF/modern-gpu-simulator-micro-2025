# Boundary Case: `softmax_kernel` vs `context_mul`

## Case Goal

这份边界 case 文档用于回答：

**`softmax_kernel` 与 `context_mul` 是否应被放进同一个 memory-side family，还是应因为 locality / DRAM 行为差异被拆开。**

本轮分析同样遵循：

- 同时记录共享点与区分点
- 以区分点分析为主
- 结论采用分级判断

## Shared Points

### 1. 二者都不是典型的纯 GEMM 主干

从 [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md) 可见：

- `softmax_kernel`: `compute=85.5%`, `dram=41.2%`, `l1_hit=79.9%`
- `context_mul`: `compute=89.6%`, `dram=7.4%`, `l1_hit=88.5%`

它们都不属于 `residual_add` 那种纯 DRAM 流式带宽瓶颈，也不完全等同于 `gemm_tiled` 的 L2 驻留计算主干。

### 2. 二者都带有明显的 memory-side 行为特征

在同一张表中：

- `softmax_kernel` 的 DRAM 使用率异常高
- `context_mul` 的 L1 命中率异常高

这意味着它们都值得被放到“memory-side boundary”里比较，而不是简单丢回 compute-heavy 主干里。

## Distinguishing Points

### 1. `softmax_kernel` 的主信号是 DRAM / working-set 压力

[E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md) 对 `softmax` 的解读非常明确：

- `dram=41.2%` 明显高于其他计算类 kernel
- `l1_hit=79.9%`
- attention score working set 约 12MB，大于 RTX 3080 Ti 的 6MB L2

[E4_full.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E4_full.md) 进一步把它归结为：

- `softmax` 是 **L2 cache 容量敏感性的标志 kernel**
- 若 simulator 把 L2 配置过大，会低估其 DRAM 利用率

所以 `softmax_kernel` 的主解释更接近：

**memory-side -> cache-capacity / DRAM-pressure**

### 2. `context_mul` 的主信号是 locality / L1 驻留

在 [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md) 的“内存三态”里：

- `context_mul` 被单独归为 **L1 驻留**
- `l1_hit=88.5%`
- `dram=7.4%`

而 [E2_batch.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E2_batch.md) 也明确指出：

- `context_mul` 是 outlier
- 其核心差异在于 **L1 驻留特征**，而非 GEMM 的 L2 驻留模式

所以 `context_mul` 的主解释更接近：

**memory-side -> locality-dominated / L1-resident**

### 3. 二者都“像 memory-side”，但不是同一种 memory-side

这是这一对 case 的关键区分点：

- `softmax_kernel`：主问题是 working set 超过 L2，导致 DRAM 压力升高
- `context_mul`：主问题不是 DRAM 压力，而是明显的 locality / L1 驻留特征

因此，它们不能因为都“看起来有 memory 行为”就被简单并入同一个 family。

## Graded Conclusion

**结论等级：边界未定（倾向拆分）**

理由：

- 共享点存在，但主要是“都不属于纯计算主干”
- 区分点更强，而且它们分别指向不同的 memory-side 主机制

因此，当前更稳妥的结论不是“弱共享”，而是：

**边界未定，但倾向拆分为两个不同的 memory-side 子族。**

## Current Execution Advice

### Family 划分建议

- 当前阶段不建议把 `softmax_kernel` 与 `context_mul` 并入同一个稳定 family
- 可以先把二者都挂在 `memory-heavy / mixed` 的外层讨论之下
- 但子类解释应分开保留：
  - `softmax_kernel`：偏 `cache-capacity / DRAM-pressure`
  - `context_mul`：偏 `locality-dominated / L1-resident`

### 验证组织建议

- 当前阶段不建议直接共享后续验证主线
- `softmax_kernel` 更适合作为 cache-capacity / DRAM-side 验证入口
- `context_mul` 更适合作为 locality / L1-side 验证入口

## Evidence References

- [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “每 Kernel 关键指标（v4，6 层均值）”与“内存三态（v4 更新版）”
- [E2_batch.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E2_batch.md): `softmax_kernel` 与 `context_mul` 的 outlier 描述
- [E4_full.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E4_full.md): “发现 C-3：softmax 揭示 L2 cache 容量限制”
