# GNN Autotuning and Knob-Prior Papers

Downloaded on 2026-06-04 for the GCL / GNNExplainer / simulator-knob-prior discussion.

This folder collects papers that are close to the idea:

```text
program / kernel graph representation
  -> GNN or graph-aware cost model
  -> performance / QoR / configuration-effect prediction
  -> search, ranking, or validation-priority reduction
```

The papers do not directly solve GPU simulator knob attribution. Their value is that they show credible variants of a cheaper model-guided tuning loop: predict or rank optimization candidates before running expensive validation.

| File | Paper | Source |
| --- | --- | --- |
| `gtuner-dac2022.pdf` | GTuner: Tuning DNN Computations on GPU via Graph Attention Network | https://www.cse.cuhk.edu.hk/~byu/papers/C143-DAC2022-GTuner.pdf |
| `calo-gnn-tvm-meta-schedule-2025.pdf` | CALO-GNN: Calibrated-Uncertainty Graph Cost Models for TVM Meta-Schedule | https://openreview.net/forum?id=wSTa0FRjVB |
| `ironman-gnn-hls-dse-2102.08138.pdf` | IronMan: GNN-assisted Design Space Exploration in High-Level Synthesis via Reinforcement Learning | https://arxiv.org/abs/2102.08138 |
| `comparexplore-hls-2409.13138.pdf` | compareXplore: Learning to Compare Hardware Designs for High-Level Synthesis | https://arxiv.org/abs/2409.13138 |
| `diffhls-2604.09240.pdf` | DiffHLS: Differential Learning for HLS QoR Prediction with GNNs and LLM Code Embeddings | https://arxiv.org/abs/2604.09240 |

## Related Paper Not Downloaded

| Paper | Reason |
| --- | --- |
| Fast selection of compiler optimizations using performance prediction with graph neural networks, DOI `10.1002/cpe.6869` | Wiley article. Search results and institutional records indicate restricted / closed full-text access, so no PDF is committed here. DOI page: https://doi.org/10.1002/cpe.6869 |

## Relevance to Our Design

- `GTuner` is the closest GPU autotuning reference: graph attention is used as a performance estimator to reduce tuning cost.
- `CALO-GNN` is useful for uncertainty-aware graph cost modeling, which fits our concern that full closed-loop simulator validation is expensive.
- `IronMan` shows the pattern `program graph -> GNN predictor -> search / RL engine` for parameterized hardware design exploration.
- `compareXplore` supports pairwise ranking of candidate designs, which is closer to "which knob should we validate first" than direct absolute importance regression.
- `DiffHLS` is relevant to knob-conditioned effect prediction: compare a baseline program and a parameterized variant, then predict the induced QoR delta.

For our first implementation, these papers support a conservative claim:

```text
GNN / graph models can produce a knob-ranking or validation-priority prior.
They should not be presented as causal knob importance until promoted by validation.
```
