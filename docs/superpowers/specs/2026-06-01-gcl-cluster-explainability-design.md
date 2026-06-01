# GCL Cluster Explainability Design

日期：2026-06-01

## 1. 目标

本文档定义如何解释 GCL / graph embedding 聚类后得到的 representative clusters。目标不是让无监督图网络自动命名硬件机制，而是让系统输出可审计的证据：

```text
这个 cluster 为什么形成？
哪些 feature groups / graph motifs 支撑这个 cluster？
某个 kernel 为什么被分到这个 cluster？
这个 cluster 和哪些 mechanism microbench anchors 相似？
这些解释是否足以进入 subtype / knob candidate ranking？
```

核心链路：

```text
GCL kernel embeddings
  -> cluster / representative selection
  -> cluster explainability module
  -> feature-level explanation
  -> instance-level assignment explanation
  -> graph motif / prototype explanation
  -> mechanism candidate audit
```

第一版解释模块只输出 `candidate explanation`，不输出 validated mechanism conclusion。

---

## 2. 相关论文与可借鉴点

### 2.1 GNNExplainer

论文：

```text
GNNExplainer: Generating Explanations for Graph Neural Networks
https://arxiv.org/abs/1903.03894
https://pmc.ncbi.nlm.nih.gov/articles/PMC7138248/
```

本地 PDF：

```text
papers/gnn-cluster-explainability/gnnexplainer-1903.03894.pdf
```

核心思路：

```text
给定一个已训练 GNN 和一个 prediction，
学习一个 edge / subgraph mask 和 feature mask，
找出最能保持该 prediction 的小子图和特征子集。
```

它的优化目标可以理解成：

```text
最大化 GNN prediction 与解释子图之间的 mutual information。
```

对我们的启发：

```text
对于某个 kernel 被分到某个 cluster，
或某个 kernel 获得某个 mechanism candidate score，
可以学习一个解释 mask：
  哪些 instruction nodes 重要？
  哪些 dependency edges 重要？
  哪些 node/edge features 重要？
```

可用于：

```text
instance-level explanation
single-kernel assignment audit
mechanism candidate score audit
```

限制：

```text
GNNExplainer 原本解释的是已有 prediction；
如果 GCL 是纯无监督 embedding，需要先定义要解释的 target：
  cluster assignment
  nearest-anchor assignment
  mechanism candidate score
```

### 2.2 PAGE

论文：

```text
Prototype-Based Explanations for Graph Neural Networks
https://ojs.aaai.org/index.php/AAAI/article/view/21660
```

本地 PDF：

```text
papers/gnn-cluster-explainability/page-prototype-based-explanations-for-gnns-aaai2022.pdf
```

核心思路：

```text
先从 GNN 的 embedding space 做 clustering；
再从每个 cluster 里抽取靠近 cluster center 的 graphs；
最后估计这些 graphs 的 maximum common subgraph，作为 human-interpretable prototype。
```

PAGE 的重要点是：

```text
它解释的是模型层面的共同模式，
不是只解释单个样本。
```

对我们的启发：

```text
对于 GCL cluster，
可以从 cluster members 中抽取最靠近 prototype embedding 的 kernel graphs，
再找共同 graph motif：
  shared opcode motif
  repeated dependency chain
  memory access motif
  control-flow motif
  synchronization motif
```

可用于：

```text
cluster-level prototype explanation
representative cluster summary
mechanism-anchor comparison report
```

限制：

```text
maximum common subgraph 计算可能较贵；
GPU trace graph 很大，第一版需要先压缩到 typed motif graph 或 summary graph。
```

### 2.3 Algorithm-Agnostic Explainability for Unsupervised Clustering

论文：

```text
Algorithm-Agnostic Explainability for Unsupervised Clustering
https://arxiv.org/abs/2105.08053
```

本地 PDF：

```text
papers/gnn-cluster-explainability/algorithm-agnostic-explainability-unsupervised-clustering-2105.08053.pdf
```

核心思路：

```text
不用关心聚类算法内部是什么；
通过扰动输入 feature，观察 cluster assignment 变化；
变化越大，说明该 feature 对聚类越重要。
```

它提出两个指标：

```text
G2PC: Global Permutation Percent Change
  打乱某一组 feature 后，统计全体样本有多少比例换了 cluster。

L2PC: Local Perturbation Percent Change
  扰动单个样本的某一组 feature，统计这个样本有多少比例换了 cluster。
```

对我们的启发：

```text
G2PC 可解释整个 GCL clustering 主要依赖哪些 feature groups；
L2PC 可解释某个 kernel 为什么属于当前 cluster。
```

可用于：

```text
feature-level cluster explanation
global cluster audit
per-kernel assignment audit
```

限制：

```text
如果 GCL embedding 是由 trace graph encoder 生成，
不能直接扰动原始 embedding 后声称解释硬件机制；
需要把 feature groups 定义成可解释来源：
  opcode group
  memory edge group
  dependency depth group
  control-flow group
  counter group
```

---

## 3. 我们的解释目标分层

第一版解释模块分三层。

### 3.1 Feature-Level Explanation

回答：

```text
这个 cluster 主要由哪些 feature groups 区分？
```

方法：

```text
G2PC for global cluster importance
L2PC for local kernel assignment importance
```

输入 feature groups：

```text
opcode_mix_group
memory_access_group
dependency_graph_group
control_flow_group
sync_barrier_group
counter_group
microbench_anchor_similarity_group
```

输出示例：

```json
{
  "cluster_id": "gcl_cluster_03",
  "global_feature_importance": [
    {
      "feature_group": "dependency_graph_group",
      "g2pc": 0.42
    },
    {
      "feature_group": "memory_access_group",
      "g2pc": 0.31
    }
  ]
}
```

### 3.2 Instance-Level Explanation

回答：

```text
为什么 real_kernel_A 被分到 cluster_03？
```

方法：

```text
L2PC over feature groups
or GNNExplainer-style mask for cluster / nearest-anchor score
```

输出示例：

```json
{
  "record_id": "real_kernel_A",
  "cluster_id": "gcl_cluster_03",
  "local_assignment_explanation": [
    {
      "feature_group": "memory_access_group",
      "l2pc": 0.58
    },
    {
      "feature_group": "control_flow_group",
      "l2pc": 0.09
    }
  ],
  "claim_status": "cluster_assignment_explanation_not_mechanism_proof"
}
```

### 3.3 Graph Motif / Prototype Explanation

回答：

```text
这个 cluster 共享什么图结构模式？
```

方法：

```text
PAGE-style embedding cluster prototype
  -> select nearest graphs to cluster center
  -> estimate common typed motif / approximate common subgraph
```

第一版不直接在完整 SASS trace graph 上做 MCS。先构造 motif summary graph：

```text
opcode family nodes
memory-space nodes
dependency-depth buckets
control-flow motif nodes
sync motif nodes
typed edges with counts / normalized weights
```

输出示例：

```json
{
  "cluster_id": "gcl_cluster_03",
  "prototype_motifs": [
    {
      "motif_id": "mem_dep_chain_ldg_imad_use",
      "support_ratio": 0.76,
      "description": "LDG-like memory load followed by address arithmetic and dependent consumer chain"
    }
  ],
  "prototype_member_ids": [
    "kernel_A",
    "kernel_B",
    "kernel_C"
  ]
}
```

---

## 4. Adaptation to GCL Mechanism Attribution

### 4.1 输入

解释模块消费：

```text
gcl_kernel_embeddings
gcl_kernel_clusters
gcl_representative_selection
kernel trace graphs or motif summary graphs
raw evidence / counter feature groups
nearest mechanism anchor output
registry projection output
```

### 4.2 输出

推荐输出：

```text
gcl_cluster_feature_explanations.json
gcl_kernel_assignment_explanations.json
gcl_cluster_prototype_motifs.json
gcl_cluster_explainability_report.md
```

### 4.3 Claim Boundary

解释模块可以说：

```text
cluster_03 is characterized by memory-access and dependency-chain feature groups.
cluster_03 has a prototype motif involving LDG-like load and dependent consumer chains.
cluster_03 is nearest to a counter-confirmed memory-latency microbench anchor.
```

解释模块不能说：

```text
cluster_03 is proven memory-bound.
cluster_03's bottleneck is definitely L2 cache.
changing a memory knob will improve performance.
```

机制语义仍然需要：

```text
microbench anchor
counter evidence
registry mapping
simulator validation
```

---

## 5. 推荐第一版算法

第一版采用轻量组合：

```text
1. GCL embedding clustering
2. G2PC on interpretable feature groups
3. L2PC for representative kernels and boundary samples
4. PAGE-style prototype motif extraction on compressed motif graphs
5. optional GNNExplainer mask for mechanism scoring head after a scorer exists
```

第一版不建议直接在完整 trace graph 上跑 expensive common subgraph search。原因：

```text
trace graph 太大；
MCS 计算代价高；
完整 graph motif 难以人工解释；
compressed motif graph 更适合论文展示和审计。
```

---

## 6. 与 Learned Mechanism Candidate Score 的关系

如果后续实现 joint model：

```text
shared GCL encoder
  -> representative selection head
  -> mechanism scoring head
  -> uncertainty head
```

解释模块对应：

```text
mechanism scoring head:
  用 GNNExplainer-style mask 解释 subtype score。

representative selection head:
  用 G2PC / L2PC 解释 cluster assignment 和 representative selection。

cluster-level semantics:
  用 PAGE-style prototype motif 解释 cluster 共同结构。
```

这使论文主张更稳：

```text
模型不只是输出 candidate score；
模型还能给出该 score 的 graph / feature / cluster-level evidence。
```

---

## 7. 验收标准

第一版完成时应满足：

1. 每个 GCL cluster 有 global feature importance；
2. 每个 representative kernel 有 local assignment explanation；
3. 每个 cluster 至少有一个 prototype motif 或明确说明 motif extraction failed；
4. explainability report 区分 feature explanation、assignment explanation 和 mechanism interpretation；
5. 输出不把 cluster explanation 写成 validated mechanism；
6. low-separation 或 mixed cluster 必须标记 boundary；
7. 所有 explanation artifacts 记录 source graph、feature group definition 和 claim_status。

---

## 8. 推荐落地顺序

```text
1. 定义 interpretable feature groups。
2. 在 GCL clusters 上实现 G2PC。
3. 对 representative / boundary kernels 实现 L2PC。
4. 从 trace graph 导出 compressed motif graph。
5. 实现 PAGE-style prototype motif extraction。
6. 生成 cluster explainability report。
7. 有 learned mechanism scorer 后，再加入 GNNExplainer-style mask。
```

一句话总结：

```text
无监督 GCL 可以发现相似结构；
G2PC / L2PC 解释聚类依赖哪些特征；
PAGE 解释 cluster 共享哪些图原型；
GNNExplainer 解释具体 GNN score 依赖哪些子图和特征。
这些解释共同帮助我们把 representative clusters 从“黑盒相似”推进到“可审计机制候选”。
```
