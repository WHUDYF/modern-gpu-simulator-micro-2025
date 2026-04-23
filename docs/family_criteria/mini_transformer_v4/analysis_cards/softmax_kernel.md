# Kernel Analysis Card: `softmax_kernel`

## Basic Info

- kernel name: `softmax_kernel`
- operator semantics: attention score normalization / reduction
- workload role: attention 子路径中的归约与归一化核心
- representative note: memory-side 边界 case 的关键样本，主要用于检验 cache-capacity / DRAM-pressure 解释

## Execution Mode

- tentative mode: `mixed`

## Key Observed Metrics

- `compute=85.5%`
- `dram=41.2%`
- `l1_hit=79.9%`
- `occ=94.1%`
- `warp_cyc=21.8`
- `block_limit_registers=10`
- `shmem=1024B`

## Dominant Resource Candidates

- primary: `cache / locality`
- secondary: `DRAM pressure`

## Family Decision

- tentative family: `mixed -> cache-capacity-sensitive`
- boundary note: 与 `context_mul` 虽然都表现出 memory-side 特征，但 `softmax` 的主问题是 working set 超过 L2 带来的 DRAM 压力，而不是 L1 驻留
- ambiguity / outlier note: 当前更适合保留在 memory-side 边界讨论中，不宜直接并入 `context_mul`

## Evidence References

- [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “每 Kernel 关键指标（v4，6 层均值）”
- [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “发现 C-4：softmax 的 DRAM 使用率异常”
- [E4_full.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E4_full.md): “发现 C-3：softmax 揭示 L2 cache 容量限制”
- [baseline_ape.json](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/baseline_ape.json): `softmax_kernel` 的 baseline APE 条目
- [softmax_kernel-vs-context_mul.md](../boundary_cases/softmax_kernel-vs-context_mul.md): “Distinguishing Points / Graded Conclusion”
