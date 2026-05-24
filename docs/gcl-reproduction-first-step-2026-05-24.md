# 复现 GCL 的第一步应该做什么

日期：2026-05-24

## 1. 文档目的

这份文档回答一个具体问题：

**我们要复现 GCL-Sampler 时，第一步应该先做什么？**

结论是：

**第一步不应该直接做 NVBit trace、trace graph construction 或 RGCN contrastive training，而应该先做 GCL-M0：Offline Embedding Selector。**

也就是说，先用 fixture / offline embedding 建立一个最小 selector 闭环，确认 GCL 的 representation 可以替换 PKA 的 12D feature representation，并且输出仍然保持 representative compression 的语义。

---

## 2. 为什么第一步不是直接复现完整 GCL-Sampler

GCL-Sampler 论文里的完整方法链是：

```text
NVBit SASS trace
  -> heterogeneous relational trace graph
  -> RGCN contrastive learning
  -> kernel embedding
  -> K-Means clustering
  -> representative simulation points
```

这条链里至少有四个高风险部分：

1. trace acquisition 依赖 NVBit 和运行环境；
2. trace graph construction 需要定义节点、边、变量版本、warp 合并等细节；
3. RGCN contrastive training 需要训练框架、数据增强、模型配置和 GPU 资源；
4. embedding clustering 还要稳定输出代表点、coverage 和 evaluation artifacts。

如果第一步就把这些全部做在一起，任何结果异常都很难定位：

- 是 trace 采集不完整？
- 是 graph schema 不对？
- 是 augmentation 破坏了语义？
- 是 RGCN 没训练好？
- 是 clustering / K selection 策略不稳？
- 还是 artifact contract 本身没有和 PKA 对齐？

因此，复现 GCL 的第一步应该先切出最小可验证接口：

```text
offline embedding -> selector -> anchor/evaluation artifacts
```

这就是 GCL-M0。

---

## 3. GCL-M0 的定位

GCL-M0 是 GCL 复现路线的第一阶段。

它的目标不是证明 learned embedding 已经有效，也不是证明 GCL 比 PKA 精度更高。

它只验证三件事：

1. **接口是否成立**

   Selector 能否从 PKA 的 12D feature input 切换到 embedding input。

2. **输出语义是否成立**

   GCL selector 能否继续输出 cluster、representative anchor、membership、coverage、weight 和 structural evaluation。

3. **比较边界是否成立**

   GCL-M0 的 artifacts 是否能和 PKA-M0 / PKA-M1 在同一层比较。

一句话说：

```text
GCL-M0 复现的是 GCL 的 selector interface，不复现完整 trace-to-model pipeline。
```

---

## 4. GCL-M0 的最小闭环

GCL-M0 的最小闭环是：

```text
fixture/offline embedding table
  -> embedding validation
  -> z-score normalization
  -> K selection
  -> deterministic K-Means
  -> nearest-centroid representative selection
  -> structural compression evaluation
  -> formal artifacts
```

其中输入 embedding 可以是人工 fixture，也可以是离线导出的 embedding。第一版不要求这些 embedding 来自真实 RGCN。

这不是偷懒，而是为了先验证：

- selector 是否能消费 embedding；
- K selection 是否能稳定运行；
- anchor artifact 是否和 PKA 对齐；
- 后续机制网络或 evaluator 是否能消费这些结果。

---

## 5. 输入契约

GCL-M0 的输入是一张 embedding table。

每条 row 至少包含：

```text
record_id
kernel_invocation_id
representation_mode
embedding_dim
embedding
source_graph_hash
encoder_manifest_hash
embedding_hash
weight_input
```

第一版 `representation_mode` 固定为：

```text
gcl_m0_embedding_fixture
```

这里有三个重要约束：

1. `embedding_dim` 必须和 `embedding` 实际长度一致；
2. embedding 中所有值必须是 finite numeric values；
3. `source_graph_hash`、`encoder_manifest_hash` 和 `embedding_hash` 不能省略。

即使 M0 的 embedding 是 fixture，也必须保留这些 hash 字段。这样后续从 fixture 切换到真实 graph / model output 时，artifact contract 不需要重写。

---

## 6. 禁止进入 selector 的字段

GCL-M0 必须避免用非 representation 信息偷做 clustering。

Selector input 不得包含或使用：

```text
kernel_name
source_path
shape_hint
trace_order
family
regime
grid_dim
block_dim
simulator outcome fields
full-workload cycle totals
```

这些字段可以出现在单独 audit 或 explanation artifact 中，但不能进入 selector 主逻辑。

原因很直接：

如果 selector 使用了这些字段，那么 GCL-M0 就不再是“embedding representation 替换 PKA feature representation”，而会变成“embedding + 手工语义标签混合 grouping”。

这会破坏后续和 PKA 的公平比较。

---

## 7. K selection 第一版怎么做

GCL-M0 第一版同时实现两种 K selection mode：

```text
silhouette_k
deterministic_fixed_k
```

### 7.1 默认模式：`silhouette_k`

默认使用 `silhouette_k`。

它的作用是让 embedding space 自己决定合适的 cluster count：

```text
for k in candidate_k:
    run K-Means
    compute silhouette_score
choose k with highest score
if tie:
    choose smaller k
```

这个选择更接近 GCL-Sampler 论文的思路，因为 GCL 的目标就是学习一个能表达 kernel similarity 的 embedding space。

### 7.2 对照模式：`deterministic_fixed_k`

同时保留 `deterministic_fixed_k`：

```text
k = ceil(sqrt(n_records)), clamped to [2, n_records]
```

这个模式用于和 PKA 直接做 ablation 对照。

有了这两个模式，我们后续可以拆开解释：

```text
PKA + fixed-K
GCL embedding + fixed-K
GCL embedding + silhouette-K
```

这样可以判断结果变化到底来自 embedding，还是来自 K selection policy。

---

## 8. Clustering 和 representative selection

GCL-M0 的 clustering 应保持 deterministic。

推荐第一版使用：

```text
deterministic farthest-first K-Means
```

要求：

- 输入 records 按 `record_id` 稳定排序；
- centroid 初始化有稳定 tie-breaker；
- 相同输入得到相同 cluster assignment；
- 输出 deterministic replay hash。

每个 cluster 的 representative selection 规则为：

```text
选择距离 centroid 最近的真实 record
```

不要生成 synthetic representative。第一阶段只选择已有 kernel invocation 作为 representative。

---

## 9. 输出 artifacts

GCL-M0 应输出四类 artifacts：

```text
gcl_embedding_table_l1.json
gcl_kmeans_clusters_l1.json
gcl_representative_anchor_table_l1.json
gcl_compression_evaluation_l1.json
```

### 9.1 `gcl_embedding_table_l1.json`

记录：

- raw embedding；
- normalized embedding；
- representation mode；
- source hash；
- encoder manifest hash；
- embedding hash；
- normalization config。

### 9.2 `gcl_kmeans_clusters_l1.json`

记录：

- K selection mode；
- selected K；
- silhouette scores；
- cluster assignments；
- centroids；
- distance to centroid；
- inertia；
- replay hash。

### 9.3 `gcl_representative_anchor_table_l1.json`

记录：

- representative record；
- cluster id；
- members；
- coverage count；
- coverage weight；
- representative distance to centroid；
- forbidden-field audit。

### 9.4 `gcl_compression_evaluation_l1.json`

记录：

- input record count；
- anchor count；
- compression ratio；
- weighted coverage；
- top-k coverage；
- anchor balance；
- cluster size distribution。

---

## 10. 第一阶段不应该声称什么

GCL-M0 不应该声称：

- GCL 已完整复现；
- embedding 已经来自真实 trace graph；
- sampled simulation accuracy 已验证；
- GCL 比 PKA 更准；
- GCL 导致 causal speedup；
- 某个 knob 有因果贡献。

M0 的结论只能是：

**embedding-based selector interface 已经跑通，并且输出 artifacts 可以和 PKA baseline 对齐比较。**

---

## 11. 第一阶段的验收标准

GCL-M0 第一阶段完成的标准是：

1. 可以读取 fixture/offline embedding table；
2. 可以校验 embedding row schema；
3. 可以拒绝 forbidden fields；
4. 可以做 z-score normalization；
5. 默认 `silhouette_k` 可以运行；
6. 显式 `deterministic_fixed_k` 可以运行；
7. 可以输出四类 formal artifacts；
8. 可以计算 structural compression metrics；
9. 不影响现有 PKA baseline tests。

---

## 12. 做完第一步以后再做什么

GCL-M0 完成以后，下一步才进入真正的 GCL 复现主体：

### 第二步：Trace Graph Construction

构建：

```text
SASS trace -> heterogeneous relational graph
```

解决：

- instruction node；
- variable node；
- pseudo node；
- control-flow edge；
- data-flow edge；
- warp-level graph；
- kernel-level graph union。

### 第三步：RGCN Contrastive Learning

构建：

```text
trace graph -> RGCN encoder -> kernel embedding
```

解决：

- node feature initialization；
- graph augmentation；
- InfoNCE loss；
- graph-level readout；
- model checkpoint；
- embedding export。

### 第四步：End-to-End Evaluation

验证：

```text
selected representatives -> sampled reconstruction -> full workload comparison
```

解决：

- sampled simulation error；
- cycle / IPC / cache / occupancy metrics；
- speedup；
- cross-architecture transfer。

---

## 13. 最终结论

复现 GCL 的第一步应该是 GCL-M0。

它不是完整 GCL-Sampler，也不是最终性能结果。

它是一个必要的 interface checkpoint：

```text
Can embeddings replace PKA features as selector input
while preserving representative compression artifacts?
```

只有这个问题先回答清楚，后续 trace graph、RGCN training 和 simulator evaluation 才有稳定的接口基础。
