# A 线 GCL-M0 Selector Interface Design Spec

日期：2026-05-29

## 1. 定位

GCL-M0 是复现 GCL-Sampler 的第一阶段，但它不复现完整论文 pipeline。

它只做一件事：

```text
用 offline / fixture embedding 替代 PKA 12D feature，
验证 embedding-based selector 能否产出可比较的 representative compression artifacts。
```

因此，GCL-M0 是：

```text
selector interface validation stage
```

不是：

```text
trace graph stage
RGCN training stage
simulator accuracy stage
```

M0 的关键价值是先把 selector 接口、artifact 契约和 comparison boundary 固定下来。只有这一层成立，后续 M1/M2 才有稳定目标。

---

## 2. 目标

GCL-M0 需要回答四个问题：

1. 如果已经有 kernel embedding，能不能替代 PKA 的 12D feature？
2. embedding 能不能驱动 clustering / representative selection？
3. 输出能不能和 PKA-M0 / PKA-M1 在同一层比较？
4. artifact 是否可 replay、可审计、可作为后续 M1/M2 的接口基础？

一句话：

```text
先证明 GCL 的 representation interface 成立。
```

M0 不以最终精度为目标，也不声称 learned embedding 已经有效。它只确认 selector path 能否从 feature-based input 切换到 embedding-based input。

---

## 3. 最小闭环

GCL-M0 的最小闭环是：

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

这里“闭环”的意思是：

- 不能只做到 embedding validation；
- 不能只做到 cluster assignment；
- 不能只输出 representative id；
- 必须最终写出 anchor artifact 和 evaluation artifact。

一个 M0 run 必须能回答：

```text
哪些 records 被分到同一个 cluster？
哪个真实 record 是 representative？
这个 representative 覆盖哪些 members？
coverage / weight 是多少？
K 是怎么选出来的？
selector 是否使用了 forbidden fields？
```

只有这些问题都能从 artifacts 中回答，M0 才形成最小闭环。

---

## 4. GCL 压缩方法在 M0 中的含义

GCL-Sampler 的最终目标不是压缩 trace 文件本身，而是在 sampled simulation 场景下用少量 representative kernel invocations 代表完整 workload。

在 M0 中，压缩方法被限定为 selector-side representative compression：

```text
N 个 kernel invocation records
  -> embedding space 中的 K 个 clusters
  -> K 个 representative anchors
```

因此，M0 的压缩对象是：

```text
kernel invocation set
```

不是：

```text
SASS trace bytes
trace graph nodes / edges
RGCN training samples
simulator execution cycles
```

### 4.1 压缩流程

M0 的 representative compression 由四步组成。

第一，使用 embedding 表示每个 kernel invocation：

```text
kernel invocation -> embedding vector
```

第二，在 normalized embedding space 中执行 K selection 和 deterministic K-Means：

```text
embedding vectors -> clusters
```

第三，在每个 cluster 中选择距离 centroid 最近的真实 record：

```text
cluster -> nearest-centroid real representative
```

第四，用 representative anchors 覆盖原始 members，并计算 structural compression metrics：

```text
representative anchors -> coverage / compression evaluation
```

### 4.2 压缩比含义

M0 中的 compression ratio 是 structural compression ratio：

```text
compression_ratio = input_record_count / anchor_count
```

例如：

```text
input_record_count = 100
anchor_count = 8
compression_ratio = 12.5
```

这只表示 100 个 kernel invocation records 被 8 个 representative anchors 覆盖。

它不等价于：

```text
trace file size reduction
measured simulator speedup
sampled reconstruction accuracy
```

这些 claim 必须等到后续 M3 存在 full-vs-sampled simulator evaluation 后才能提出。

### 4.3 与完整 GCL-Sampler 的关系

完整 GCL-Sampler 的 compression path 是：

```text
trace graph / RGCN embedding
  -> clustering
  -> representative simulation points
  -> sampled simulation
```

M0 只验证后半段：

```text
embedding
  -> clustering
  -> representative anchors
  -> structural compression evaluation
```

M0 不验证前半段：

```text
trace acquisition
trace graph construction
graph size control
RGCN contrastive learning
kernel embedding quality
```

所以 M0 可以证明 GCL 的 selector-side compression contract 已经闭合，但不能证明 GCL 的 trace-to-embedding pipeline 已经可用。

### 4.4 与 PKA 的比较边界

M0 应和 PKA 在同一层比较：

```text
PKA feature space -> clustering -> representative anchors
GCL embedding space -> clustering -> representative anchors
```

比较对象是：

```text
anchor_count
compression_ratio
coverage_count
weighted_coverage
cluster_size_distribution
anchor_balance
```

不是：

```text
cycle error
IPC error
cache metric error
measured speedup
```

后者属于 simulator-side evaluation，不属于 M0。

---

## 5. 输入契约

输入是 embedding table。每条 row 至少包含：

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

M0 第一版固定：

```text
representation_mode = gcl_m0_embedding_fixture
```

M0 不关心 embedding 是怎么来的，可以是 fixture，也可以是离线导出结果。它只关心该 embedding table 是否满足 selector contract。

### 5.1 字段语义

`record_id`：

- selector 内部稳定排序、dedup 和 replay 的主 id。

`kernel_invocation_id`：

- 对应 workload 中的 kernel invocation。

`representation_mode`：

- 标记当前 representation 来源和阶段。M0 中必须为 `gcl_m0_embedding_fixture`。

`embedding_dim`：

- embedding 的声明维度，必须等于 `embedding` 的实际长度。

`embedding`：

- selector 用于 clustering 的 numeric vector。

`source_graph_hash`：

- M0 中可以指向 fixture source hash；后续 M1/M2 中应指向真实 graph artifact hash。

`encoder_manifest_hash`：

- M0 中可以指向 fixture embedding manifest hash；后续 M2 中应指向 encoder / model manifest hash。

`embedding_hash`：

- 当前 embedding row 的 replay / audit hash。

`weight_input`：

- member-count fallback 或 timing weight 输入。

---

## 6. 输入验证

GCL-M0 必须严格校验 input rows。

必须拒绝：

- 少于 2 条 records；
- 缺少 `record_id` / `kernel_invocation_id`；
- 重复 `record_id`；
- `representation_mode` 不一致；
- `embedding_dim` 与实际 embedding 长度不一致；
- embedding 为空；
- embedding 包含 non-finite numeric value；
- 缺少 `source_graph_hash`；
- 缺少 `encoder_manifest_hash`；
- 缺少 `embedding_hash`；
- selector row 中出现 forbidden fields。

输入验证失败时，M0 不应继续运行 clustering。

---

## 7. 禁止字段

selector input 不允许包含或使用：

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
B-line semantic metadata
```

这些字段可以出现在单独 audit / explanation artifact 中，但不能进入 selector input。

原因：

```text
M0 要验证 embedding 本身能否做 representative compression，
不能偷偷用名字、形状、family 或 simulator 结果辅助 grouping。
```

如果 selector 使用这些字段，GCL-M0 就不再是 embedding representation replacement，而是混合语义标签的 grouping。

---

## 8. Normalization

M0 在 clustering 前必须对 embedding 做 z-score normalization。

必须记录：

```text
normalization mode
embedding dimension
per-dimension mean
per-dimension std
zero-std dimensions
normalized embedding
```

如果某一维 std 为 0，则使用 safe std，并把该维度记录到 `zero_std_dimensions`。

Normalization 的目的不是改变 embedding 语义，而是避免某些维度因数值尺度过大主导距离计算。

---

## 9. K Selection

GCL-M0 支持两种 mode：

```text
silhouette_k
deterministic_fixed_k
```

默认：

```text
silhouette_k
```

### 9.1 `silhouette_k`

`silhouette_k` 用来贴近 GCL-Sampler 论文，让 embedding space 自己决定 cluster count。

行为：

```text
for k in candidate_k:
    run deterministic K-Means
    compute silhouette_score
choose k with highest score
if tie:
    choose smaller k
```

artifact 必须记录：

- candidate K list；
- silhouette score per K；
- selected K；
- tie-breaker rule；
- final K-Means metadata。

### 9.2 `deterministic_fixed_k`

`deterministic_fixed_k` 用作 PKA 对照：

```text
k = ceil(sqrt(n_records)), clamped to [2, n_records]
```

后续可以做清晰 ablation：

```text
PKA + fixed-K
GCL embedding + fixed-K
GCL embedding + silhouette-K
```

这样可以判断结果变化到底来自 embedding，还是来自 K selection policy。

---

## 10. Clustering

M0 使用 deterministic K-Means。

要求：

- records 按 `record_id` 稳定排序；
- centroid 初始化 deterministic；
- tie-breaker 使用 `record_id`；
- 相同输入得到相同输出；
- artifact 记录 replay hash。

距离空间：

```text
normalized embedding space
```

第一版距离度量：

```text
squared Euclidean distance
```

K-Means metadata 至少记录：

- method；
- selected K；
- initial center record ids；
- centroids；
- distance metadata；
- iterations；
- deterministic replay hash。

---

## 11. Representative Selection

每个 cluster 选一个真实 record：

```text
nearest real record to centroid
```

第一版不生成 synthetic representative。

原因是后续 simulator / evaluator 需要能回到真实 kernel invocation。Synthetic representative 会增加解释和验证成本，不适合 M0。

---

## 12. 输出 Artifacts

M0 输出四类 artifacts：

```text
gcl_embedding_table_l1.json
gcl_kmeans_clusters_l1.json
gcl_representative_anchor_table_l1.json
gcl_compression_evaluation_l1.json
```

### 12.1 `gcl_embedding_table_l1.json`

记录：

- raw embedding；
- normalized embedding；
- representation mode；
- source hash；
- encoder manifest hash；
- embedding hash；
- normalization config。

### 12.2 `gcl_kmeans_clusters_l1.json`

记录：

- K selection mode；
- selected K；
- silhouette scores；
- cluster assignments；
- centroids；
- distance to centroid；
- inertia；
- replay hash。

### 12.3 `gcl_representative_anchor_table_l1.json`

记录：

- representative record；
- cluster id；
- members；
- coverage count；
- coverage weight；
- representative distance to centroid；
- forbidden-field audit。

### 12.4 `gcl_compression_evaluation_l1.json`

记录：

- input record count；
- anchor count；
- compression ratio；
- weighted coverage；
- top-k coverage；
- anchor balance；
- cluster size distribution。

---

## 13. Evaluation Scope

M0 只做 structural evaluation。

必须计算：

```text
anchor_count
compression_ratio
coverage_count
weighted_coverage
top_k_coverage
anchor_balance
cluster_size_distribution
```

M0 不报告：

```text
sampled simulation accuracy
cycle error
IPC / cache / occupancy accuracy
simulator speedup
causal contribution
```

这些属于后续 M3。

---

## 14. Replay 与 Audit

M0 的 artifacts 必须足以 replay 和审计。

至少记录：

- input embedding table hash；
- normalization config；
- K selection metadata；
- K-Means metadata；
- forbidden-field audit；
- deterministic replay hash。

Audit 应能回答：

```text
为什么选这个 representative？
它代表哪些 members？
coverage / weight 是多少？
K 是如何选择的？
selector 是否使用 forbidden fields？
```

---

## 15. 与 Joint Mechanism Network 的关系

GCL-M0 本身不做 mechanism attribution、knob matching 或 validation priority。

但它的 artifacts 应为后续 joint mechanism network 预留可消费接口。

M0 可以在不改变 selector 主逻辑的前提下，保留以下 audit / metadata 字段：

```text
cluster_id
representative_record_id
members
coverage_weight
selection_confidence
evidence_status
claim_status
embedding_ref
```

其中第一版必须稳定的是：

- `cluster_id`
- `representative_record_id`
- `members`
- `coverage_weight`

`selection_confidence`、`evidence_status`、`claim_status` 和 `embedding_ref` 可以作为后续扩展字段，但不能影响 M0 clustering。

---

## 16. 非目标

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

## 17. 成功标准

GCL-M0 完成标准：

1. 能读取 fixture/offline embedding table；
2. 能严格验证 schema；
3. 能拒绝 forbidden fields；
4. 能做 z-score normalization；
5. 默认 `silhouette_k` 可运行；
6. 显式 `deterministic_fixed_k` 可运行；
7. 能运行 deterministic K-Means；
8. 能选择 nearest-centroid real representative；
9. 能输出四类 formal artifacts；
10. 能计算 structural compression metrics；
11. 不影响现有 PKA baseline tests。

---

## 18. 和后续阶段的关系

M0 完成后：

```text
M1: trace -> graph artifact
M2: graph artifact -> RGCN embedding
M3: representative -> sampled/full metric evaluation
```

所以 M0 是后续阶段的 selector contract base。

M1/M2 的目标不是重写 M0 selector，而是逐步替换 M0 的 embedding 来源：

```text
fixture/offline embedding
  -> graph-derived embedding
  -> RGCN-derived embedding
```

只要最终 embedding table 满足 M0 输入契约，后续就可以复用同一套 selector / anchor / evaluation 语义。
