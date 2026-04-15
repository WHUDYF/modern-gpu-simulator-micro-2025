# Family Card: `mixed -> cache-capacity-sensitive`

## Core Explanation

这个 family 用于表示那些既具有显著计算成分、又呈现强 memory-side 压力，但其主导解释最终落在 cache-capacity / DRAM-pressure 上的 kernel。

## Representative Kernels

- `softmax_kernel`

## Boundary Conditions

- 不纳入 `context_mul`，因为后者虽然也处在 memory-side 边界上，但更接近 locality / L1-resident 解释
- 不纳入 `residual_add`，因为后者是更纯粹的 DRAM-bandwidth family

## Uncertainty

- 当前该 family 仍是单成员 family
- 第一版只把它当作 memory-side 细分的必要子类，不声称该子类规则已经稳定可迁移

## What It Is Not

- 它不是“所有 mixed kernel”的总类
- 它也不是“所有 softmax-like kernel”的语义族
- 它特指“主导问题来自 cache 容量不足导致 DRAM 压力抬升”的 mixed 子族

## Validation Meaning

- 当前阶段不建议和 `context_mul` 共享验证主线
- 它更适合作为 cache-capacity / DRAM-pressure 方向的独立验证入口
