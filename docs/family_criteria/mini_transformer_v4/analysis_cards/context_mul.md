# Kernel Analysis Card: `context_mul`

## Basic Info

- kernel name: `context_mul`
- operator semantics: context value accumulation
- workload role: attention 路径中的 context update
- representative note: memory-side 边界 case 的另一侧锚点，用于代表 locality / L1-resident 解释

## Execution Mode

- tentative mode: `mixed`

## Key Observed Metrics

- `compute=89.6%`
- `dram=7.4%`
- `l1_hit=88.5%`
- `occ=90.0%`
- `warp_cyc=31.5`
- `block_limit_registers=8`
- `shmem=0`

## Dominant Resource Candidates

- primary: `cache / locality`
- secondary: `L1-resident access behavior`

## Family Decision

- tentative family: `mixed -> locality-dominated`
- boundary note: 不宜和 `softmax_kernel` 直接并类；二者共享的是 memory-side 外层语义，而不是相同的主导机制
- ambiguity / outlier note: 当前保持为 memory-side 边界样本，而不是稳定吸收入 `softmax` 所在子类

## Evidence References

- [E0_baseline.md](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “每 Kernel 关键指标（v4，6 层均值）”
- [E0_baseline.md](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “内存三态（v4 更新版）”
- [E2_batch.md](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/E2_batch.md): context_mul outlier 描述
- [softmax_kernel-vs-context_mul.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/boundary_cases/softmax_kernel-vs-context_mul.md): “Distinguishing Points / Current Execution Advice”
