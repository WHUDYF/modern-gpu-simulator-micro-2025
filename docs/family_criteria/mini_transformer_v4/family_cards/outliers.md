# Outlier Card

## Current Outliers

- `layernorm_kernel`

## Why It Is Not Absorbed Yet

`layernorm_kernel` 当前既不稳定落入 `memory-heavy -> dram-dominated`，也不稳定落入两个 mixed 子类中的任一个。

它表现出：

- 中等计算比例
- 中等 DRAM 使用率
- 较高 L1 hit
- 极低 `waves`

这些信号说明它是一个典型的第二轮 mixed/outlier 检验样本，而不是当前第一轮就应被强行并类的对象。

## Boundary Reasoning

- 它不像 `residual_add` 那样是纯 DRAM 流式样本
- 它也不像 `softmax_kernel` 那样主导问题落在 cache-capacity / DRAM-pressure
- 它更不像 `context_mul` 那样体现强 locality 主解释

因此第一版更稳妥的处理方式是：

**先保留为 outlier，再在后续轮次中判断它究竟应形成新的 mixed 子类，还是能并入已有子类。**

## Validation Meaning

- 当前阶段不将其纳入任何稳定共享验证主线
- 后续若 family 判据成熟，可优先把它作为第二轮 mixed 边界检验对象
