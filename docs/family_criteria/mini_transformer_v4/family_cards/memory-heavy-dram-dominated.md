# Family Card: `memory-heavy -> dram-dominated`

## Core Explanation

这个 family 用来容纳那些主行为明显 `memory-heavy`，且其最强解释直接落在 DRAM 带宽 / 流式访存压力上的 kernel。

## Representative Kernels

- `residual_add`

## Boundary Conditions

- 不纳入虽然带有 memory-side 特征、但主问题是 cache-capacity 的 kernel，例如 `softmax_kernel`
- 不纳入以 locality / L1-resident 为主的 kernel，例如 `context_mul`

## Uncertainty

- 第一版里它是单成员 family，这并不构成方法问题
- 后续若出现其它纯流式带宽样本，再决定是否扩展该 family

## What It Is Not

- 它不是所有 memory-heavy kernel 的总类
- 它只表示“主导解释直接是 DRAM bandwidth”的子族

## Validation Meaning

- 当前阶段可把它视为 memory-system 验证中的稳定中心样本
- 它适合承担后续 DRAM-side 验证主线的锚点角色
