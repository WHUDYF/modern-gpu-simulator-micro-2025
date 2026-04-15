# Kernel Analysis Card: `layernorm_kernel`

## Basic Info

- kernel name: `layernorm_kernel`
- operator semantics: normalization / reduction
- workload role: attention 后与 FFN 后的规范化路径
- representative note: 当前 family 体系中的第二轮检验样本，用于测试 mixed / outlier 边界是否稳定

## Execution Mode

- tentative mode: `mixed`

## Key Observed Metrics

- `compute=46.6%`
- `dram=21.4%`
- `l1_hit=75.0%`
- `occ=79.5%`
- `warp_cyc=25.6`
- `waves=1.07`
- `block_limit_registers=10`
- `shmem=0`

## Dominant Resource Candidates

- primary: `mixed reduction / normalization behavior`
- secondary: `cache / locality`

## Family Decision

- tentative family: `mixed -> reduction-coupled`
- boundary note: 目前不强行并入 `softmax` 或 `context_mul` 的 memory-side 子类；它更适合作为第二轮 mixed/outlier 判据检验对象
- ambiguity / outlier note: 第一版保留明显不确定性；必要时可升级为 outlier 候选

## Evidence References

- [E0_baseline.md](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “每 Kernel 关键指标（v4，6 层均值）”
- [E0_baseline.md](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “发现 C-3：layernorm_kernel 受 waves 不足限制”
- [E2_batch.md](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/E2_batch.md): `layernorm_kernel` 独立聚类说明
