# GNNExplainer Application Motif Papers

Downloaded on 2026-06-04 for the explanation-aware knob-prior paper design.

This folder collects papers that use GNN explainability, including GNNExplainer-style masks, as a bridge from a graph model prediction to human-readable or domain-specific structure.

The direct target is not GPU simulator tuning. The relevance is the common pattern:

```text
graph model prediction
  -> compact explanation subgraph / feature mask
  -> domain motif, rule, or analyst-facing evidence
```

This supports our design:

```text
GCL cluster assignment
  -> GNNExplainer compact trace motif
  -> LLM / registry mechanism interpretation
  -> knob-prior ranking
```

| File | Paper | Source |
| --- | --- | --- |
| `fuzzygnn-enhanced-fuzzy-modeling-2026.pdf` | Enhanced fuzzy modeling with graph neural network-based explainability | https://link.springer.com/article/10.1007/s00521-026-11967-7 |
| `cfgexplainer-dsn2022.pdf` | CFGExplainer: Explaining Graph Neural Network-Based Malware Classification from Control Flow Graphs | https://www.dinalherath.com/papers/2022dsn.pdf |
| `explainable-malware-graph-reduction-gnnexplainer-2412.03634.pdf` | Explainable Malware Detection Using Graph Reduction and GNNExplainer | https://arxiv.org/abs/2412.03634 |
| `geco-community-gnn-explainer-2026.pdf` | GECo: Unveiling communities in graphs with neural networks | https://link.springer.com/article/10.1007/s00607-026-01642-z |

## Related Papers Not Downloaded

| Paper | Reason |
| --- | --- |
| Explaining decisions of graph convolutional neural networks: patient-specific molecular subnetworks responsible for metastasis prediction in breast cancer | Open page found, but direct PDF download from the institutional repository was slow/unreliable during this archive step. Link: https://doi.org/10.1186/s12859-021-04447-7 |
| Application of Graph Attention Network for Breast Cancer Data and Explanation of Prediction Basis | IEEE article; no stable open PDF committed. DOI: https://doi.org/10.1109/ACCESS.2025.3575946 |

## Relevance To Our Design

- `FuzzyGNN` uses GNN explanation masks to convert graph model behavior into interpretable fuzzy rules. This supports our idea that compact graph explanations can become structured semantic evidence.
- `CFGExplainer` and explainable malware detection papers are close to our program-graph setting: large control-flow / function-call graphs are classified by a GNN, then explanation subgraphs are extracted for analyst-facing evidence.
- `GECo` is useful as a cluster/community explanation reference: explanation masks can be used to expose what structure drives graph grouping.
- Breast-cancer GNN explanation papers are useful analogies for `motif -> domain mechanism`: explanation subgraphs are interpreted as pathway or molecular mechanism evidence. Our analog is `trace motif -> hardware mechanism candidate`.

The main claim boundary remains:

```text
GNNExplainer subgraphs are model-relative explanations.
They are not hardware-causal proof until promoted by counter, microbench, or simulator validation.
```
