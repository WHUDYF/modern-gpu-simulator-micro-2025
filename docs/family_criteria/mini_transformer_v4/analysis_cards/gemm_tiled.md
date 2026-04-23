# Kernel Analysis Card: `gemm_tiled`

## Basic Info

- kernel name: `gemm_tiled`
- operator semantics: GEMM / matrix multiply
- workload role: mini-transformer v4 中的主计算骨架
- representative note: 在六层结构中重复出现，是 compute 主干的代表样本

## Execution Mode

- tentative mode: `compute-heavy`

## Key Observed Metrics

- `compute=90.9%`
- `warp_cyc=36.3`
- `occ=89.9%`
- `dram=9.0%`
- `l1_hit=8.4%`
- `block_limit_registers=6`
- `shmem=2048B`

## Dominant Resource Candidates

- primary: `register / occupancy`
- secondary: `compute pipeline saturation`

## Family Decision

- tentative family: `compute-heavy -> register-limited`
- boundary note: 与 `attention_score` 共享主解释，但不能忽略后者更强的 shared-memory 特征
- ambiguity / outlier note: 当前不是 outlier 候选，但它是与 `attention_score` 形成边界 case 的一侧锚点

## Evidence References

- [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “每 Kernel 关键指标（v4，6 层均值）”
- [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “发现 C-1：gemm_tiled + attention_score 均受寄存器限制”
- [E4_full.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E4_full.md): “发现 C-1：gemm_tiled + attention_score 共享计算瓶颈”
- [baseline_ape.json](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/baseline_ape.json): `gemm_tiled` 的 baseline APE 条目
- [gemm_tiled-vs-attention_score.md](../boundary_cases/gemm_tiled-vs-attention_score.md): “Graded Conclusion / Current Execution Advice”
