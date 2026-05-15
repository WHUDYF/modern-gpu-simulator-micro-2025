# A 线 GCL-M0 专门说明

日期：2026-05-15

## 1. 文档目的

这份文档专门说明 GCL-M0 要做什么、为什么要做、做完以后会得到什么。

它不是 implementation plan，也不是完整的 GCL 总体 spec。它只解释 GCL-M0 这一阶段的定位：

**先用 fixture/offline embedding 验证 GCL 的 selector 接口，再决定后续是否把 trace graph 和 RGCN training 接进来。**

---

## 2. GCL-M0 的一句话定义

GCL-M0 是一个最小闭环：

```text
offline embedding table
  -> validation
  -> normalization
  -> clustering
  -> representative anchors
  -> structural compression evaluation
```

它的任务不是证明 GCL 已经比 PKA 更强，而是证明：

1. GCL 可以替换 PKA 的 representation layer；
2. 输出仍然能沿用 anchor / membership / coverage / weight 这套语义；
3. 这一层的实现可以独立 replay、独立测试、独立比较。

---

## 3. 为什么先做 M0

GCL-Sampler 真正完整的路径是：

```text
SASS trace
  -> trace graph
  -> RGCN contrastive learning
  -> kernel embedding
  -> K-Means
```

但这条路一开始就全做，风险太高，问题也难定位。M0 的作用就是先把最关键的接口拆出来：

- 输入可以不是真实 graph embedding，而是 fixture/offline embedding；
- 聚类策略先明确成可控、可测、可比较的形态；
- 先把结果 artifact 化，验证输出语义是否和 PKA 对齐。

换句话说，M0 解决的是“接口和语义是否成立”，不是“模型是否最终最优”。

---

## 4. M0 和 PKA 的关系

PKA 的 selector 输入是 12D measured feature table，大致流程是：

```text
12D feature
  -> preprocessing
  -> PCA
  -> K-Means
  -> anchors
```

GCL-M0 的 selector 输入改成：

```text
embedding table
  -> validation
  -> z-score normalization
  -> K-Means
  -> anchors
```

两者保留相同的外层语义：

- cluster
- representative anchor
- coverage
- weight
- deterministic replay hash
- structural compression evaluation

因此，GCL-M0 的意义不是新造一套输出，而是把 PKA 的“行为特征表示”替换成“学习到的 embedding 表示”。

---

## 5. GCL-M0 默认做什么

GCL-M0 同时实现两种 K selection mode：

### 5.1 `silhouette_k`

这是默认模式。

它会在候选 K 上计算 silhouette coefficient，选择得分最高的 K。这样做的原因是：

- 更贴近 GCL-Sampler 论文；
- 让 embedding space 自己决定更合适的 cluster count；
- 避免一开始只用固定 K 绑死结果。

### 5.2 `deterministic_fixed_k`

这是显式 ablation / baseline mode。

它仍然保留：

```text
k = ceil(sqrt(n_records))
```

并 clamp 到 `[2, n_records]`。

这个模式的作用是和 PKA 做直接对照，避免后面解释不清：

- 变化来自 embedding；
- 还是变化来自 K 选择策略。

---

## 6. M0 的输入是什么

M0 输入不是 trace，也不是 graph，而是 embedding table。每条记录至少包含：

- `record_id`
- `kernel_invocation_id`
- `representation_mode`
- `embedding_dim`
- `embedding`
- `source_graph_hash`
- `encoder_manifest_hash`
- `embedding_hash`
- `weight_input`

这里的 `embedding` 可以来自：

- fixture；
- 离线导出的 embedding；
- 后续的模型输出快照。

M0 只关心这组 row 是否可用于 selector，不关心它们是怎么来的。

---

## 7. M0 不做什么

M0 不做：

- trace acquisition
- graph construction
- RGCN training
- graph augmentation
- simulator execution
- cross-architecture evaluation

这些都在后续 M1/M2/M3 里处理。

M0 也不应该偷偷使用这些信息来做 clustering：

- `kernel_name`
- `shape_hint`
- `trace_order`
- `family`
- `regime`
- `grid_dim`
- `block_dim`

这些字段最多只能出现在 audit 里，不能进入 selector 主逻辑。

---

## 8. M0 的处理流程

M0 的核心处理顺序是：

```text
1. validate embedding rows
2. sort by stable record_id
3. normalize embeddings
4. choose K
5. run K-Means
6. select representatives
7. compute coverage / compression metrics
8. write artifacts
```

### 8.1 Validation

验证内容包括：

- 记录数至少为 2；
- `record_id` / `kernel_invocation_id` 可识别；
- `representation_mode` 一致；
- embedding 数值有限；
- `embedding_dim` 与实际维度一致；
- `source_graph_hash` / `encoder_manifest_hash` / `embedding_hash` 都存在；
- forbidden fields 未进入 selector 输入。

### 8.2 Normalization

M0 对 embeddings 做 z-score normalization，然后再 clustering。这样可以避免不同维度量纲差异影响距离计算。

### 8.3 Clustering

默认使用 silhouette-K。

如果显式指定 `deterministic_fixed_k`，则使用固定 K 对照模式。

### 8.4 Representative selection

每个 cluster 选距离 centroid 最近的真实 record 作为 representative。

---

## 9. M0 的输出是什么

M0 会写出四类 formal artifacts：

```text
gcl_embedding_table_l1.json
gcl_kmeans_clusters_l1.json
gcl_representative_anchor_table_l1.json
gcl_compression_evaluation_l1.json
```

它们分别表示：

- `gcl_embedding_table_l1.json`：标准化前后的 embedding 输入视图；
- `gcl_kmeans_clusters_l1.json`：聚类分配和 centroid 信息；
- `gcl_representative_anchor_table_l1.json`：代表性 anchor 与 coverage 信息；
- `gcl_compression_evaluation_l1.json`：结构压缩评价。

这些 artifact 的目的不是“看起来像输出”，而是能让后续直接比较 GCL 与 PKA。

---

## 10. 评价重点

M0 先看结构性指标，不看 simulator accuracy。

主要看：

- `anchor_count`
- `compression_ratio`
- `coverage_count`
- `weighted_coverage`
- `top_k_coverage`
- `anchor_balance`
- `cluster_size_distribution`

M0 的重点是：

**embedding-based selector 能否稳定地产出合理的 representative compression 结果。**

---

## 11. 为什么默认 silhouette

这是 GCL-M0 最关键的策略选择。

默认 silhouette 的原因不是“固定 K 不好”，而是：

1. GCL 的初衷就是让 learned embedding 自己表达 kernel similarity；
2. silhouette 可以让数据自己决定 K；
3. 这更接近论文里的 GCL-Sampler 思路；
4. fixed-K 仍然保留，所以不会失去 PKA 对照能力。

因此，M0 的默认策略可以理解为：

```text
主结果看 silhouette
对照结果看 fixed-K
```

---

## 12. M0 成功的标准

当以下条件满足时，GCL-M0 就算完成：

1. 可以读取 embedding table；
2. 可以稳定验证和标准化；
3. 默认 silhouette-K 可以跑通；
4. deterministic fixed-K 也可以作为显式模式运行；
5. 可以输出可比较的 anchor / coverage / evaluation artifacts；
6. PKA 回归测试不受影响。

这意味着 M0 已经具备了一个可用的 selector 层，但还没有进入真实 trace graph 和模型训练阶段。

---

## 13. M0 之后会接什么

M0 之后的自然顺序是：

### M1

把 trace acquisition 和 graph construction 接进来，开始生成真实 graph artifact。

### M2

加入 RGCN contrastive learning，从 graph 生成 embedding。

### M3

做 full-vs-sampled 的 simulator / metric evaluation。

所以 M0 不是最终系统，而是一个可测的起点。

---

## 14. 最终结论

GCL-M0 的核心不是“做一个新的聚类器”，而是：

**证明 GCL 的 embedding-based representation 可以替换 PKA 的 feature-based representation，并且仍然保持相同的 representative compression 语义。**

如果这一步成立，后续 M1/M2 才值得继续往真实 trace graph 和 RGCN 方向推进。
