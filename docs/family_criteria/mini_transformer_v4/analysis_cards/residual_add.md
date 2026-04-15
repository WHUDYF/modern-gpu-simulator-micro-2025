# Kernel Analysis Card: `residual_add`

## Basic Info

- kernel name: `residual_add`
- operator semantics: elementwise residual accumulation
- workload role: 注意力后与 FFN 后的逐元素加法路径
- representative note: mini-transformer v4 中最稳定、最极端的 DRAM-side 样本

## Execution Mode

- tentative mode: `memory-heavy`

## Key Observed Metrics

- `dram=58.3%`
- `compute=14.7%`
- `warp_cyc=87.6`
- `occ=74.8%`
- `l1_hit=33.2%`
- `block_limit_registers=16`
- `shmem=0`

## Dominant Resource Candidates

- primary: `DRAM bandwidth`
- secondary: `memory latency / streaming access`

## Family Decision

- tentative family: `memory-heavy -> dram-dominated`
- boundary note: 不能和 `softmax_kernel` 混成同一类 memory-side family，因为 `softmax` 的主解释更偏 cache-capacity / DRAM-pressure，而 `residual_add` 是纯带宽流式样本
- ambiguity / outlier note: 当前不需要保留为 outlier；它更像稳定 family 的中心样本

## Evidence References

- [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “每 Kernel 关键指标（v4，6 层均值）”
- [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “发现 C-2：residual_add 是纯 HBM 带宽瓶颈”
- [E2_batch.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E2_batch.md): residual_add 独立聚类说明
- [E4_full.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E4_full.md): “发现 C-2：residual_add 是孤立的 HBM 带宽瓶颈”
- [baseline_ape.json](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/baseline_ape.json): `residual_add` 的 baseline APE 条目
