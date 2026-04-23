# Kernel Analysis Card: `attention_score`

## Basic Info

- kernel name: `attention_score`
- operator semantics: attention score compute
- workload role: attention 子路径中的核心计算 kernel
- representative note: 与 `gemm_tiled` 在主解释上高度接近，但在实现特征上仍保留明显异质性

## Execution Mode

- tentative mode: `compute-heavy`

## Key Observed Metrics

- `compute=95.2%`
- `warp_cyc=34.0`
- `occ=95.1%`
- `dram=8.9%`
- `l1_hit=7.1%`
- `block_limit_registers=6`
- `shmem=8256B`
- `waves=25.6`

## Dominant Resource Candidates

- primary: `register / occupancy`
- secondary: `shared-memory-coupled execution`

## Family Decision

- tentative family: `compute-heavy -> register-limited`
- boundary note: 与 `gemm_tiled` 弱共享；shared memory 和 waves 特征足以阻止在第一版中把它当作纯 GEMM 副本
- ambiguity / outlier note: 保留“良性计算异质”说明，不作为完全稳定并类样本

## Evidence References

- [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “每 Kernel 关键指标（v4，6 层均值）”
- [E2_batch.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E2_batch.md): “Batch 输出：3 聚类 + 3 outlier”
- [E4_full.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E4_full.md): “发现 C-1：gemm_tiled + attention_score 共享计算瓶颈”
- [baseline_ape.json](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/baseline_ape.json): `attention_score` 的 baseline APE 条目
- [gemm_tiled-vs-attention_score.md](../boundary_cases/gemm_tiled-vs-attention_score.md): “Distinguishing Points / Graded Conclusion”
