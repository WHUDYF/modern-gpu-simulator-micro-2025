# Unsupervised GNN Soft Family Distribution Design Spec

日期：2026-06-05

## 1. 问题

我们希望从无监督 GNN / GCL 中得到的不只是 hard cluster：

```text
kernel_i -> cluster_03
```

而是类似：

```json
{
  "kernel_id": "kernel_i",
  "family_distribution_prior": {
    "memory_latency_like": 0.62,
    "fp64_pipeline_like": 0.18,
    "register_pressure_like": 0.12,
    "sync_like": 0.08
  }
}
```

但必须注意：

```text
无监督 cluster / prototype 本身没有硬件语义。
```

因此 family distribution 不能直接来自：

```text
prototype_id -> hardware family
```

而应来自：

```text
soft prototype assignment
  + prototype explanation motif
  + LLM / taxonomy interpretation
  + anchor / counter / validation evidence
```

## 2. 推荐链路

第一版推荐链路：

```text
trace graph
  -> GCL / graph encoder
  -> graph embedding
  -> prototype / cluster assignment distribution
  -> GNNExplainer prototype motif
  -> LLM / taxonomy maps motif to mechanism family candidate
  -> family_distribution_prior
  -> registry-constrained knob candidates
  -> validation priority
```

其中：

```text
prototype assignment distribution
```

回答：

```text
这个 kernel 在 embedding space 中像哪些 prototype？
```

而：

```text
family_distribution_prior
```

回答：

```text
这些 prototype 经过 motif interpretation 后，像哪些硬件 family？
```

二者不能混为一谈。

## 3. 为什么看 Prototype / Soft Assignment

GCL-Sampler 默认路径更接近：

```text
embedding -> K-Means -> hard cluster -> representative kernel
```

这适合 sampled simulation representative selection，但不适合表达混合机制。

真实 kernel 可能同时具有：

```text
global memory latency motif
fp32 / fp64 compute chain
register pressure
sync behavior
```

如果只输出 hard cluster，会丢掉这种混合性。

prototype / soft assignment 可以提供：

```text
kernel -> prototype probability / assignment weights
```

再通过 prototype motif interpretation 转为：

```text
kernel -> weak family distribution prior
```

## 4. 关键边界

不能声称：

```text
soft assignment probability = calibrated hardware mechanism probability
```

可以声称：

```text
soft assignment provides an unsupervised structural prior.
Hardware family meaning is assigned only after motif explanation and taxonomy mapping.
```

也就是说：

```text
0.62 memory_latency_like
```

不是说：

```text
这个 kernel 62% 时间受 memory latency 支配。
```

而是说：

```text
这个 kernel 的 graph embedding / motif evidence 更接近被解释为 memory-latency-like 的 prototype。
```

## 5. 与 GNNExplainer 的关系

prototype assignment 给出：

```text
kernel_i is close to prototype_p
```

GNNExplainer 给出：

```text
prototype_p / cluster_p 的 assignment 依赖哪些 compact subgraph 和 feature mask？
```

然后我们才能解释：

```text
prototype_p looks like global memory latency
prototype_q looks like fp64 dependency chain
prototype_r looks like sync fan-in
```

因此 soft family distribution 应由两步生成：

```text
Step 1:
kernel -> prototype assignment weights

Step 2:
prototype -> interpreted family candidates
```

组合：

```text
family_score(kernel, family)
  = sum_over_prototypes(
      assignment_weight(kernel, prototype)
      * interpreted_family_support(prototype, family)
    )
```

这个公式只产生 prior，不产生 validated causal importance。

## 6. 输出 Schema

建议输出：

```json
{
  "record_id": "kernel_i",
  "prototype_assignment": [
    {
      "prototype_id": "proto_03",
      "assignment_weight": 0.62,
      "assignment_source": "soft_prototype_head"
    }
  ],
  "prototype_explanations": [
    {
      "prototype_id": "proto_03",
      "motif_id": "motif_global_load_to_use",
      "motif_stability": 0.74,
      "claim_status": "model_explanation_not_hardware_truth"
    }
  ],
  "family_distribution_prior": [
    {
      "family_id": "global_memory_latency_like",
      "support": 0.58,
      "evidence_sources": [
        "prototype_assignment",
        "gnnexplainer_motif",
        "taxonomy_interpretation"
      ],
      "claim_status": "weak_family_prior_not_calibrated_probability"
    }
  ]
}
```

## 7. Related Work

PDFs are archived in:

```text
papers/unsupervised-gnn-soft-prototype-assignment/
```

Existing prototype-contrastive PDFs are also in:

```text
papers/mechanism-prototypical-gcl/
```

| Paper | Link | Use in our paper |
| --- | --- | --- |
| Graph Clustering with Graph Neural Networks / DMoN | https://www.jmlr.org/papers/volume24/20-998/20-998.pdf | Shows differentiable soft assignment matrices for unsupervised graph clustering. Useful for thinking about assignment distributions rather than hard cluster IDs. |
| Prototypical Graph Contrastive Learning | https://arxiv.org/abs/2106.09645 | Supports graph-level prototype-aware contrastive learning and prototype consistency. |
| Prototypical Contrastive Learning | https://arxiv.org/abs/2005.04966 | General prototype contrastive learning reference; useful for prototype assignment in representation learning. |
| SwAV | https://arxiv.org/abs/2006.09882 | Useful for online swapped cluster assignments and prototype-style unsupervised learning. |
| X-GOAL: Multiplex Heterogeneous Graph Prototypical Contrastive Learning | https://arxiv.org/abs/2109.03560 | Shows prototypical contrastive learning on heterogeneous graphs. |
| Graph Prototypical Contrastive Learning, Information Sciences 2022 | https://www.sciencedirect.com/science/article/pii/S002002552201057X | Most directly aligned with graph prototype assignment probability, but no stable open PDF is archived here. |

## 8. Minimal First Version

第一版不需要完整训练复杂 prototype GNN。

可执行版本：

1. 使用 GCL embedding；
2. 用 K-Means / prototype centroids 得到 distance-based soft assignment；
3. 对每个 prototype / cluster representative 跑 GNNExplainer；
4. 聚合 motif，得到 prototype explanation；
5. LLM / taxonomy 把 prototype motif 映射到 family candidates；
6. 将 assignment weights 和 family support 合成为 family_distribution_prior；
7. 只把它作为 knob validation priority 的输入，不声称 causal importance。

第一版 soft assignment 可用：

```text
assignment_weight_i =
  softmax(-distance(kernel_embedding, prototype_embedding) / temperature)
```

其中 temperature 必须记录在 manifest 中，并作为 calibration-sensitive hyperparameter，而不是自然概率。

