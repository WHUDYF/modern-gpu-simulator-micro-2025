# A 线 GCL-M0 含义说明 Spec

日期：2026-05-28

## 1. 背景

A 线已经把 PKA baseline 作为正式前端锚点来推进。PKA 路径的核心是：

```text
measured 12D behavior features
  -> preprocessing / PCA
  -> clustering
  -> representative anchors
  -> structural compression evaluation
```

GCL-Sampler 的核心变化不在于“后端多一个输出”，而在于把 kernel 的表示方式从手工行为特征换成 learned embedding：

```text
SASS trace
  -> trace graph
  -> RGCN contrastive learning
  -> kernel embedding
  -> clustering
  -> representative anchors
```

完整 GCL-Sampler 一次性复现风险很高，因为它同时涉及 trace acquisition、trace graph construction、RGCN training、embedding export、K selection 和 simulator-side evaluation。GCL-M0 的意义，就是先把其中最小、最关键、最容易验证的一段切出来。

---

## 2. GCL-M0 的定义

GCL-M0 是 A 线 GCL 复现路线的第一阶段。

它的定义是：

**使用 fixture/offline embedding table 作为 selector input，验证 embedding-based representative compression 能否产出与 PKA baseline 可比较的 cluster、anchor、coverage 和 evaluation artifacts。**

换句话说，GCL-M0 不是完整 GCL-Sampler，也不是最终效果证明。它是一个 selector-interface validation stage。

GCL-M0 回答的问题是：

```text
如果我们已经有 kernel embedding，
它能不能替代 PKA 的 12D feature，
并驱动同一套 representative compression 输出语义？
```

---

## 3. GCL-M0 的最小闭环

GCL-M0 的最小闭环必须覆盖从 embedding row 到 formal artifacts 的完整路径：

```text
fixture/offline embedding table
  -> embedding row validation
  -> stable record ordering
  -> z-score embedding normalization
  -> K selection
  -> deterministic K-Means
  -> nearest-centroid representative selection
  -> structural compression evaluation
  -> replayable GCL artifacts
```

这个闭环不能只做到 cluster assignment。只有当它能写出 anchor table 和 evaluation artifact 时，才算真正形成闭环。

### 3.1 为什么叫“最小”

“最小”指它暂时不依赖：

- NVBit trace；
- heterogeneous trace graph；
- RGCN encoder；
- graph augmentation；
- model checkpoint；
- simulator replay；
- closed-loop validation。

这些部分都属于后续 M1/M2/M3。

### 3.2 为什么叫“闭环”

“闭环”指它必须从输入到输出形成一个完整、可验证、可 replay 的 selector path：

```text
input row
  -> selector decision
  -> representative object
  -> coverage / weight
  -> evaluation summary
```

如果只完成 embedding validation，没有 clustering，不是闭环。

如果只完成 clustering，没有 representative anchor，不是闭环。

如果只输出 anchor，没有 coverage / evaluation，也不是闭环。

---

## 4. M0 与 PKA 的关系

GCL-M0 应被理解为 PKA selector 输入层的替换实验。

PKA 使用：

```text
12D measured features
```

GCL-M0 使用：

```text
embedding vectors
```

但二者在外层输出上应尽量保持一致：

- cluster membership；
- representative anchor；
- coverage count；
- coverage weight；
- weight；
- representative distance；
- structural compression evaluation。

这样做的原因是：只有输出语义一致，GCL-M0 才能和 PKA-M0 / PKA-M1 做同层比较。

---

## 5. 输入契约

GCL-M0 的 canonical input 是 embedding table。

每条 row 必须至少包含：

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

### 5.1 字段语义

`record_id`：

- selector 内部稳定排序、dedup 和 replay 的主 id。

`kernel_invocation_id`：

- 连接回 workload invocation 的外部 id。

`representation_mode`：

- 标记该 embedding 来源和阶段。M0 中必须是 `gcl_m0_embedding_fixture`。

`embedding_dim`：

- embedding 维度声明，必须和 `embedding` 实际长度一致。

`embedding`：

- selector 使用的 numeric vector。

`source_graph_hash`：

- M0 中可以指向 fixture source hash；M1/M2 后应指向真实 graph artifact hash。

`encoder_manifest_hash`：

- M0 中可以指向 fixture embedding manifest hash；M2 后应指向 encoder/model manifest hash。

`embedding_hash`：

- 当前 row 的 embedding hash，用于 replay 和审计。

`weight_input`：

- member-count fallback 或 timing weight 输入。

### 5.2 输入验证要求

M0 必须拒绝以下输入：

- 少于 2 条 records；
- 缺少 `record_id` / `kernel_invocation_id`；
- 重复 `record_id`；
- `representation_mode` 不一致；
- `embedding_dim` 与实际长度不一致；
- embedding 为空；
- embedding 包含 non-finite value；
- 缺少 hash fields；
- selector row 中出现 forbidden fields。

---

## 6. Forbidden Fields

GCL-M0 的 selector input 不得包含或使用：

```text
kernel_name
source_path
expected_behavior_axis
family
regime
shape_hint
trace_order
grid_dim
block_dim
simulator outcome fields
full-workload cycle totals
B-line semantic metadata
```

这些字段可以进入单独 audit / explanation artifact，但不能进入 selector input table。

原因是：M0 要验证 embedding representation 本身能否驱动 representative compression。如果 selector 偷用了 kernel name、shape、family 或 simulator outcome，结果就不再能说明 embedding 替换是否成立。

---

## 7. Normalization 语义

GCL-M0 必须在 clustering 前对 embedding 做 z-score normalization。

要求记录：

- normalization mode；
- embedding dimension；
- per-dimension mean；
- per-dimension standard deviation；
- zero-std dimensions；
- normalized embedding。

如果某一维 std 为 0，则使用 safe std，并在 `zero_std_dimensions` 中记录。

Normalization 的目的不是改变 embedding 语义，而是避免某些维度因数值尺度过大主导 Euclidean distance。

---

## 8. K Selection 语义

GCL-M0 必须支持两种 K selection mode：

```text
silhouette_k
deterministic_fixed_k
```

### 8.1 默认模式：`silhouette_k`

M0 默认使用 `silhouette_k`。

行为：

```text
for k in candidate_k:
    run deterministic K-Means
    compute silhouette_score
choose k with highest silhouette_score
if tie:
    choose smaller k
```

Artifact 必须记录：

- `k_selection.mode = "silhouette_k"`；
- candidate K list；
- silhouette score per K；
- selected K；
- tie-breaker rule。

### 8.2 对照模式：`deterministic_fixed_k`

M0 同时保留 `deterministic_fixed_k`：

```text
k = ceil(sqrt(n_records)), clamped to [2, n_records]
```

该模式不是默认模式，而是用于 PKA 对照和 ablation：

```text
PKA + fixed-K
GCL embedding + fixed-K
GCL embedding + silhouette-K
```

这样可以拆清楚结果变化来自 embedding 还是来自 K policy。

---

## 9. Clustering 语义

GCL-M0 使用 deterministic K-Means。

第一版推荐沿用 deterministic farthest-first initialization：

- 输入按 `record_id` 稳定排序；
- 第一个 centroid 使用稳定规则选出；
- 后续 centroid 使用 farthest-first；
- tie-breaker 使用 `record_id`；
- 相同输入应得到相同 assignments。

Distance metric 第一版使用：

```text
squared Euclidean distance in normalized embedding space
```

K-Means metadata 必须写入 artifact，至少包括：

- method；
- selected K；
- initial center record ids；
- centroids；
- iterations；
- distance metadata。

---

## 10. Representative Selection 语义

每个 cluster 必须选择一个真实 record 作为 representative。

第一版规则：

```text
nearest real record to centroid
```

禁止第一版生成 synthetic representative。

Anchor row 至少包含：

```text
anchor_id
cluster_id
representative_record_id
members
coverage_count
coverage_weight
weight
representative_distance_to_centroid
cluster_label
```

---

## 11. 输出 Artifacts

GCL-M0 必须写出四个 artifacts。

### 11.1 `gcl_embedding_table_l1.json`

用途：

- 固化 selector 实际消费的 embedding input；
- 记录 raw / normalized embedding；
- 保存 normalization config 和 replay hash。

### 11.2 `gcl_kmeans_clusters_l1.json`

用途：

- 固化 K selection 结果；
- 保存 cluster assignment；
- 保存 centroids、distance 和 inertia；
- 记录 deterministic replay hash。

### 11.3 `gcl_representative_anchor_table_l1.json`

用途：

- 输出 representative anchors；
- 输出 members；
- 输出 coverage / weight；
- 输出 forbidden-field audit；
- 对齐 PKA anchor artifact 的核心语义。

### 11.4 `gcl_compression_evaluation_l1.json`

用途：

- 汇总 structural compression metrics；
- 不报告 simulator accuracy；
- 不报告 causal speedup。

---

## 12. Evaluation 语义

GCL-M0 只做 structural evaluation。

必须计算：

- input record count；
- anchor count；
- compression ratio；
- coverage count；
- weighted coverage；
- top-k coverage；
- anchor balance；
- cluster size distribution。

禁止在 M0 中报告：

- sampled simulation APE；
- cycle accuracy；
- IPC / cache / occupancy accuracy；
- simulator speedup；
- causal performance contribution。

这些属于后续 M3。

---

## 13. Replay 与 Audit 要求

GCL-M0 必须保证相同输入可以 replay。

至少记录：

- input embedding fixture hash；
- normalization config；
- K selection metadata；
- K-Means metadata；
- forbidden-field audit；
- deterministic replay hash。

Audit 的重点是回答：

```text
这个 representative 为什么被选中？
它代表哪些 members？
这个 cluster 的 coverage 是多少？
K 是如何选出来的？
selector 是否偷用了 forbidden fields？
```

---

## 14. 非目标

GCL-M0 不做：

- NVBit trace collection；
- SASS trace parsing；
- heterogeneous trace graph construction；
- graph tensorization；
- graph augmentation；
- RGCN training；
- learned embedding quality evaluation；
- simulator execution；
- cross-architecture transfer；
- mechanism attribution；
- knob matching；
- validation priority。

M0 也不声称：

- 已完整复现 GCL-Sampler；
- GCL 比 PKA 更准；
- sampled simulation error 已降低；
- 任何 causal speedup。

---

## 15. 成功标准

GCL-M0 完成必须满足：

1. 可以读取 fixture/offline embedding table；
2. 可以严格验证 input schema；
3. 可以拒绝 forbidden fields；
4. 可以做 deterministic ordering；
5. 可以做 z-score normalization；
6. 默认 `silhouette_k` 可以运行；
7. 显式 `deterministic_fixed_k` 可以运行；
8. 可以运行 deterministic K-Means；
9. 可以选择 nearest-centroid real representative；
10. 可以写出四个 formal artifacts；
11. 可以计算 structural compression evaluation；
12. 现有 PKA baseline tests 不受影响。

---

## 16. 与后续阶段的关系

GCL-M0 完成后，后续阶段按以下顺序推进。

### GCL-M1

把输入从 fixture/offline embedding 前移到 trace graph artifact：

```text
trace-like input
  -> deterministic trace graph
  -> graph artifact
```

### GCL-M2

把 embedding 来源从 fixture/offline rows 替换为 RGCN encoder 输出：

```text
graph artifact
  -> RGCN contrastive encoder
  -> kernel embedding
  -> GCL-M0 selector contract
```

### GCL-M3

在 simulator / metric 层验证 selected representatives 是否真的保留 full workload 行为：

```text
representatives
  -> sampled reconstruction
  -> full workload comparison
```

因此，M0 是后续 M1/M2/M3 的 selector contract base。

---

## 17. 最终结论

GCL-M0 的具体含义是：

**在没有真实 trace graph 和 RGCN 训练的情况下，先验证 embedding-based selector 是否能替代 PKA 12D feature selector，并产出可比较、可 replay、可审计的 representative compression artifacts。**

它是 GCL 复现的第一层接口保证，不是完整方法复现，也不是最终性能结论。
