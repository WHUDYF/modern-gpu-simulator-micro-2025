# Unsupervised GNN Soft Prototype Assignment Papers

Downloaded on 2026-06-05 for the question:

```text
Can unsupervised GNNs produce class / family proportions instead of only hard clusters?
```

The papers here support the design idea:

```text
graph embedding
  -> prototype / cluster assignment distribution
  -> family-distribution prior
```

The assignments are not hardware mechanism truth. In our method, hardware semantics must still come from GNNExplainer motifs, LLM / taxonomy interpretation, microbench anchors, counters, and validation.

| File | Paper | Source |
| --- | --- | --- |
| `dmon-graph-clustering-with-gnns-jmlr2023.pdf` | Graph Clustering with Graph Neural Networks / DMoN | https://www.jmlr.org/papers/volume24/20-998/20-998.pdf |
| `x-goal-multiplex-heterogeneous-gpcl-2109.03560.pdf` | X-GOAL: Multiplex Heterogeneous Graph Prototypical Contrastive Learning | https://arxiv.org/abs/2109.03560 |

## Existing Related PDFs In This Repository

| Existing file | Paper | Source |
| --- | --- | --- |
| `papers/mechanism-prototypical-gcl/pgcl-prototypical-graph-contrastive-learning-2106.09645.pdf` | Prototypical Graph Contrastive Learning | https://arxiv.org/abs/2106.09645 |
| `papers/mechanism-prototypical-gcl/pcl-prototypical-contrastive-learning-2005.04966.pdf` | Prototypical Contrastive Learning | https://arxiv.org/abs/2005.04966 |
| `papers/mechanism-prototypical-gcl/swav-contrasting-cluster-assignments-2006.09882.pdf` | SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments | https://arxiv.org/abs/2006.09882 |

## Related Papers Not Downloaded

| Paper | Reason |
| --- | --- |
| Graph Prototypical Contrastive Learning, Information Sciences 2022 | ScienceDirect / Elsevier page found, but no stable open PDF was archived here. Link: https://www.sciencedirect.com/science/article/pii/S002002552201057X |
| scGPCL: graph prototypical contrastive learning for single-cell transcriptomics | PMC page found, direct PDF fetch returned HTML during this archive step. Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10246584/ |

## Relevance To Our Design

- `DMoN` is useful for understanding differentiable soft assignment matrices. It is node-clustering oriented, but the idea of a learned assignment matrix motivates a distribution over latent groups.
- `PGCL` and `PCL` motivate prototype-aware contrastive learning, where each sample can be related to learned prototypes rather than only hard nearest-neighbor clusters.
- `SwAV` is not a GNN paper, but it is useful for understanding swapped cluster assignments and online prototype assignment.
- `X-GOAL` shows a prototypical contrastive learning formulation on heterogeneous graphs.

Our safe claim:

```text
The prototype assignment distribution is a mechanism-family prior.
It is not a calibrated hardware bottleneck distribution.
```
