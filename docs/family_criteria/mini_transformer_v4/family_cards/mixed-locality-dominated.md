# Family Card: `mixed -> locality-dominated`

## Core Explanation

这个 family 表示那些既有较高计算成分、又带明显 memory-side 特征，但其主解释更接近 locality / L1-resident 行为的 kernel。

## Representative Kernels

- `context_mul`

## Boundary Conditions

- 不纳入 `softmax_kernel`，因为其主问题不在 locality，而在 cache-capacity / DRAM-pressure
- 不纳入 `gemm_tiled` / `attention_score`，因为它们的主解释仍在 compute-heavy / register-limited

## Uncertainty

- 第一版中它仍是单成员 family
- 后续若更多 workload 中出现类似 L1-resident mixed 样本，再判断该子类是否稳定

## What It Is Not

- 它不是泛化意义上的“访存命中率高”集合
- 它是“mixed 外层下，由 locality 主导解释的子类”

## Validation Meaning

- 当前阶段不建议与 `softmax_kernel` 共享后续验证主线
- 若后续需要压缩验证复杂度，可以先在 locality / L1-side 方向单独展开
