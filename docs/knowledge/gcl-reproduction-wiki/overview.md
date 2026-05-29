# GCL Reproduction Overview

GCL-Sampler 的核心思想是把 PKA baseline 中的手工行为特征替换为从 trace graph 学到的 kernel embedding。

完整路径可以概括为：

```text
NVBit SASS trace
  -> heterogeneous relational trace graph
  -> RGCN contrastive learning
  -> kernel embedding
  -> K-Means clustering
  -> representative simulation points
```

本 worktree 的目标不是一次性复现完整 pipeline，而是分阶段建立可验证的 contract。

阶段顺序是：

```text
[[gcl-m0-offline-embedding-selector]]
  -> [[gcl-m1-trace-graph-construction]]
  -> [[gcl-m2-rgcn-embedding-and-selector]]
  -> [[gcl-m3-simulator-evaluation]]
```

每个阶段都必须产出可审计、可 replay 的 artifacts。跨阶段 artifact 关系集中在 [[artifact-contracts]]，claim 边界集中在 [[stage-boundaries]]。

## 与 PKA 的关系

PKA 和 GCL 共享外层 representative compression 语义：

```text
selector input representation
  -> clustering
  -> representative anchors
  -> structural compression evaluation
```

差异在于 representation 来源。

PKA 使用 measured 12D behavior features。GCL 最终应使用 RGCN-derived kernel embeddings。

[[gcl-m0-offline-embedding-selector]] 先验证 embedding table 能否替换 PKA 12D feature table。[[gcl-m1-trace-graph-construction]] 和 [[gcl-m2-rgcn-embedding-and-selector]] 再逐步把 embedding 的来源替换为真实 trace graph learning。

