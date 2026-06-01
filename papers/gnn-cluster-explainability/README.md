# GNN and Clustering Explainability Reading Set

This folder stores papers for explaining GNN embeddings, graph clusters, and unsupervised clustering assignments in the trace-compression-industrial branch.

## Downloaded PDFs

| File | Paper | Source |
| --- | --- | --- |
| `gnnexplainer-1903.03894.pdf` | GNNExplainer: Generating Explanations for Graph Neural Networks | https://arxiv.org/abs/1903.03894 and https://pmc.ncbi.nlm.nih.gov/articles/PMC7138248/ |
| `page-prototype-based-explanations-for-gnns-aaai2022.pdf` | Prototype-Based Explanations for Graph Neural Networks | https://ojs.aaai.org/index.php/AAAI/article/view/21660 |
| `algorithm-agnostic-explainability-unsupervised-clustering-2105.08053.pdf` | Algorithm-Agnostic Explainability for Unsupervised Clustering | https://arxiv.org/abs/2105.08053 |

## Why These Papers

The intended use is to explain GCL-derived representative clusters:

```text
GCL kernel embeddings
  -> cluster / representative kernel
  -> cluster-level feature explanation
  -> instance-level assignment explanation
  -> graph motif / prototype explanation
  -> mechanism candidate audit
```

Mapping:

- `GNNExplainer`: explains which subgraph and node features matter for a GNN prediction; useful for explaining why a kernel graph is assigned to a mechanism candidate or cluster.
- `PAGE`: clusters graph embeddings and extracts a human-interpretable prototype via maximum common subgraph; useful for summarizing what a representative cluster structurally shares.
- `Algorithm-Agnostic Clustering Explainability`: explains any clustering method with global permutation percent change and local perturbation percent change; useful for explaining which feature groups drive GCL / KMeans cluster assignments.

## File Checksums

```text
515567da60b85b4f3112b1e6ef3cb692b0a4e39fa4d59fbe341a7c6ed3688e5b  gnnexplainer-1903.03894.pdf
897e50477418eafadbbd4eaa2e4c60363719a51d75a2e99d867a283ff9a2101a  page-prototype-based-explanations-for-gnns-aaai2022.pdf
e4236a53b80e1579d0f35145f041120fdfcfb2fe126166fdd82afe44a9abbbc3  algorithm-agnostic-explainability-unsupervised-clustering-2105.08053.pdf
```
