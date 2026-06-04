# Explanation-Aware Knob Prior Paper Design Spec

日期：2026-06-04

## 1. 核心问题

我们不应让 GNN 直接输出 simulator knob 的 causal importance ratio。

原因：

```text
kernel graph -> GNN embedding -> FC head -> knob importance
```

这条路径看起来简单，但有两个风险：

1. 缺少足够 closed-loop validation ground truth；
2. FC head 可能只是在黑盒 embedding 上学习弱标签或人工规则的影子。

如果论文声称：

```text
GNN predicts knob importance.
```

审稿人会追问：

```text
importance label 从哪里来？
为什么这个比例代表真实 causal tuning contribution？
```

因此第一篇工作应把目标改成：

```text
GNN predicts an explanation-aware, registry-constrained knob prior.
```

这个 prior 不是最终调参结论，而是有限验证预算下的候选排序。

## 2. 推荐方法链路

完整链路：

```text
GCL cluster
  -> GNNExplainer compact explanation subgraph
  -> cluster-level shared motif
  -> LLM / rule interpreter over fixed taxonomy
  -> mechanism candidate
  -> subtype / knob registry
  -> knob-prior ranking head
  -> budget-aware validation planner
  -> closed-loop validation promotes or rejects candidates
```

关键设计选择：

```text
GNNExplainer is a structural bottleneck before knob ranking.
```

也就是说，knob head 不应只吃：

```text
whole_cluster_embedding
```

而应吃：

```text
whole_cluster_embedding
motif_embedding
feature_mask_summary
edge_mask_summary
mechanism_candidate
knob_id
optional_knob_delta
```

这样可以把方法从黑盒预测变成：

```text
解释子图证据 -> 机制候选 -> 合法 knob 候选 -> 验证优先级
```

## 3. 为什么必须先用 GNNExplainer

GNNExplainer 的作用不是替代 GCL，也不是直接输出 knob。

它回答：

```text
GCL / GNN 为什么把这个 kernel 放进当前 cluster？
这个判断依赖哪些 node、edge、feature？
```

输出：

```text
explanation_subgraph
edge_mask
feature_mask
important_node_types
important_edge_types
important_feature_groups
```

如果对 cluster 内多个 kernel 做解释并聚合，可得到：

```text
cluster_shared_motif
motif_stability
motif_coverage
```

这些才是后续 mechanism interpretation 和 knob ranking 的结构证据。

没有 GNNExplainer，FC head 容易退化为：

```text
kernel embedding -> knob score
```

这种说法很难解释模型为什么认为某个 knob 重要。

有了 GNNExplainer，论文可以说：

```text
The knob prior is conditioned on compact structural evidence extracted from the cluster assignment.
```

## 4. Knob Prior Head

第一版不预测绝对 importance ratio，而预测 knob ranking prior。

输入：

```text
cluster_embedding
motif_embedding
motif_feature_summary
mechanism_candidate_embedding
knob_embedding
optional_knob_delta
cluster_scale_features
```

输出：

```json
{
  "cluster_id": "cluster_03",
  "ranked_knob_candidates": [
    {
      "knob": "dram_latency",
      "score": 0.71,
      "uncertainty": 0.24,
      "claim_status": "knob_prior_not_validated"
    }
  ]
}
```

允许的训练目标：

1. pairwise ranking：

```text
knob A should be validated before knob B
```

2. binary validation-readiness：

```text
this knob is worth a validation run
```

3. weak target from registry and anchors：

```text
mechanism candidate maps to legal knob candidates
```

4. validated feedback when available：

```text
knob A reduced simulator error more than knob B
```

不允许第一版声称：

```text
score is calibrated causal importance.
```

## 5. Budget-Aware Validation

闭环实验成本高，因此 validation 不能全量跑。

推荐三层验证：

```text
cheap structural prior
  -> cheap proxy validation
  -> expensive simulator validation for top candidates only
```

cheap structural prior：

```text
GCL cluster scale
GNNExplainer motif stability
LLM taxonomy interpretation
microbench anchor similarity
registry knob readiness
counter support if available
```

cheap proxy validation：

```text
mechanism microbench knob sensitivity
short replay / reduced input
counter consistency
representative-SM scoped evidence
```

expensive validation：

```text
top clusters only
top knobs only
few knob deltas only
```

论文主张应是：

```text
We reduce validation search cost by ranking knob candidates using explanation-aware graph evidence.
```

不是：

```text
We infer exact knob importance without validation.
```

## 6. Paper Claim Boundaries

可以声称：

```text
GCL compresses kernels into structural clusters.
GNNExplainer extracts compact motifs explaining cluster assignment.
LLM / taxonomy maps motifs to weak mechanism candidates.
Registry constrains legal knob candidates.
The model ranks knob candidates for limited validation.
Closed-loop validation promotes or rejects top-ranked candidates.
```

不能声称：

```text
LLM labels are ground truth.
GNNExplainer subgraph is hardware-causal proof.
Knob prior score is calibrated causal importance.
All knob candidates are exhaustively validated.
Compression ratio equals simulator speedup.
```

## 7. Related Work To Cite

这些论文分成两类：

1. 图模型作为 tuning / search / validation-priority prior；
2. GNNExplainer-style explanation 把黑盒图预测转换成领域可读 motif / rule / evidence。

第一类支持后续 knob-prior head，第二类支持我们为什么要先提取 compact explanation subgraph。

### 7.1 Graph Models for Tuning / Search Priors

| Paper | Link | Use in our paper |
| --- | --- | --- |
| GTuner: Tuning DNN Computations on GPU via Graph Attention Network | https://www.cse.cuhk.edu.hk/~byu/papers/C143-DAC2022-GTuner.pdf | GPU autotuning 中使用 GAT performance estimator，支持 graph model 先缩小 tuning search。 |
| CALO-GNN: Calibrated-Uncertainty Graph Cost Models for TVM Meta-Schedule | https://openreview.net/forum?id=wSTa0FRjVB | 支持 uncertainty-aware cost model，适合我们输出 validation priority 和 uncertainty。 |
| IronMan: GNN-assisted Design Space Exploration in High-Level Synthesis via Reinforcement Learning | https://arxiv.org/abs/2102.08138 | 支持 `program graph -> GNN predictor -> search/RL` 的参数探索模式。 |
| compareXplore: Learning to Compare Hardware Designs for High-Level Synthesis | https://arxiv.org/abs/2409.13138 | 支持 pairwise ranking，比直接回归 absolute importance 更适合低数据场景。 |
| DiffHLS: Differential Learning for HLS QoR Prediction with GNNs and LLM Code Embeddings | https://arxiv.org/abs/2604.09240 | 支持 baseline-vs-variant delta prediction，可迁移为 knob-conditioned effect prediction。 |
| Fast selection of compiler optimizations using performance prediction with graph neural networks | https://doi.org/10.1002/cpe.6869 | 支持用 GNN 预测快速筛选 compiler optimization；Wiley 受限访问，暂不保存 PDF。 |

### 7.2 GNNExplainer as a Motif / Rule / Evidence Bridge

PDF 已归档在：

```text
papers/gnnexplainer-application-motifs/
```

| Paper | Link | Use in our paper |
| --- | --- | --- |
| GNNExplainer: Generating Explanations for Graph Neural Networks | https://arxiv.org/abs/1903.03894 | 原始方法：通过 edge mask 和 feature mask 提取 compact subgraph，解释 GNN prediction；支持 single-instance 和 multi-instance / prototype explanation。 |
| Enhanced fuzzy modeling with graph neural network-based explainability | https://link.springer.com/article/10.1007/s00521-026-11967-7 | 使用 GNNExplainer-style masks 把图模型预测转成 fuzzy IF-THEN rules。对应我们的 `trace motif -> mechanism evidence` 桥接思路。 |
| CFGExplainer: Explaining Graph Neural Network-Based Malware Classification from Control Flow Graphs | https://www.dinalherath.com/papers/2022dsn.pdf | 程序 CFG 场景中解释 GNN 分类，说明在程序图上抽取重要子图可服务人工分析。对应我们的 `trace graph cluster -> explanation subgraph`。 |
| Explainable Malware Detection Using Graph Reduction and GNNExplainer | https://arxiv.org/abs/2412.03634 | 在大规模 malware graph 上结合 graph reduction 和 GNNExplainer，说明 explanation subgraph 可降低图规模并提供 analyst-facing evidence。 |
| GECo: Unveiling communities in graphs with neural networks | https://link.springer.com/article/10.1007/s00607-026-01642-z | 说明 explanation masks 可以用于解释 graph grouping / community structure。对应我们的 cluster-level motif explanation。 |
| Explaining decisions of graph convolutional neural networks: patient-specific molecular subnetworks responsible for metastasis prediction in breast cancer | https://doi.org/10.1186/s12859-021-04447-7 | 生物医学中把 explanation subgraph 解释为 domain mechanism / pathway evidence。对应我们的 `trace motif -> hardware mechanism candidate`，但我们必须再经 validation。 |
| Application of Graph Attention Network for Breast Cancer Data and Explanation of Prediction Basis | https://doi.org/10.1109/ACCESS.2025.3575946 | 使用 GAT 预测并解释 prediction basis，支持“图预测后需要解释其依据”的论文动机。IEEE PDF 暂未归档。 |

这类工作共同支持一个方法边界：

```text
GNNExplainer is used to expose model-relative structural evidence.
Domain semantics are assigned by an interpreter or expert layer.
Final causal claims require external validation.
```

这正是我们的论文叙事：

```text
GCL/GNN prediction
  -> GNNExplainer compact trace motif
  -> LLM / taxonomy mechanism interpretation
  -> registry-constrained knob prior
  -> limited validation
```

## 8. Minimal First Paper Version

第一篇最小可行版本：

1. GCL 复现 / 简化复现，得到 cluster 和 representative kernel；
2. 对 representative kernel 或 cluster members 跑 GNNExplainer；
3. 聚合 explanation subgraphs 得到 cluster motif；
4. LLM 在固定 taxonomy 内解释 motif，输出 weak mechanism candidate；
5. registry 映射 legal knobs；
6. knob-prior head 或规则 baseline 排序 top candidates；
7. 只跑少量 top candidate validation；
8. 报告 validation search reduction 和 top-ranked candidate success rate。

第一版可以先用规则 baseline：

```text
motif + registry + scale -> knob-prior ranking
```

再加入学习 head：

```text
motif-aware representation -> FC ranking head
```

这样即使训练数据少，也能形成可发表的渐进路线。
