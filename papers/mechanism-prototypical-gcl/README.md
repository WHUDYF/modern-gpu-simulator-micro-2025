# Mechanism-Prototypical GCL Reading Set

This folder stores papers for the trace-compression-industrial branch discussion on extending GCL-style kernel structure learning with mechanism microbench anchors and learned mechanism candidate scoring.

## Downloaded PDFs

| File | Paper | Source |
| --- | --- | --- |
| `gcl-sampler-2603.00551.pdf` | GCL-Sampler: Discovering Kernel Similarity for Sampled GPU Simulation via Graph Contrastive Learning | https://arxiv.org/abs/2603.00551 |
| `pgcl-prototypical-graph-contrastive-learning-2106.09645.pdf` | Prototypical Graph Contrastive Learning | https://arxiv.org/abs/2106.09645 |
| `pcl-prototypical-contrastive-learning-2005.04966.pdf` | Prototypical Contrastive Learning of Unsupervised Representations | https://arxiv.org/abs/2005.04966 |
| `swav-contrasting-cluster-assignments-2006.09882.pdf` | Unsupervised Learning of Visual Features by Contrasting Cluster Assignments | https://arxiv.org/abs/2006.09882 |

## Tracked but PDF Not Downloaded

| Paper | Source | Note |
| --- | --- | --- |
| Graph Prototypical Contrastive Learning | https://doi.org/10.1016/j.ins.2022.09.013 | DOI / publisher page found; no stable open PDF was downloaded. |

## Why These Papers

The intended method direction is:

```text
GCL-Sampler reproduction
  -> mechanism microbench anchors
  -> prototype-aware graph contrastive learning
  -> learned mechanism candidate scoring
  -> registry-constrained knob candidates
  -> validation planning
```

The papers map to that direction as follows:

- `GCL-Sampler`: trace graph construction, R-GCN encoder, graph contrastive learning, kernel embeddings, clustering, representative selection.
- `PGCL`: graph-specific prototype contrastive learning and prototype-aware graph representation.
- `PCL`: general prototypical contrastive learning with cluster/prototype assignments.
- `SwAV`: clustering/assignment consistency without fixed pairwise labels.
- `GPCL`: related graph prototype contrastive learning reference; keep citation even though no PDF is stored here.

## File Checksums

```text
a244f021b19fc4307f6638d814bc5b874838895ba09851652d6f651820347e4d  gcl-sampler-2603.00551.pdf
a89d8d2e79f281f7c646e890dc7a7ebf7b50e2dc6c1d9ae56a792a1f5ab2f309  pgcl-prototypical-graph-contrastive-learning-2106.09645.pdf
47783f785c00130659a29d79f28f112a6e8914f4c84490d19669b61db8a612d2  pcl-prototypical-contrastive-learning-2005.04966.pdf
4249b19a6c31946e27b4244d050b69d7283b87dde1d134c051cd596a8bd4e5dc  swav-contrasting-cluster-assignments-2006.09882.pdf
```
