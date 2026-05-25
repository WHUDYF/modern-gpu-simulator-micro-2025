# A 线 GCL-Sampler 总体架构设计

日期：2026-05-13

## 1. 背景

A 线当前已经有一条 PKA-compatible 的 measured baseline 路径。该路径把 invocation-level records 转成固定的 12 维行为特征空间，经过 preprocessing 和 PCA 后，对投影点进行聚类，并输出带 membership 与 coverage metadata 的 representative anchors。

GCL-Sampler 解决的是同一个 sampled GPU simulation 问题，但它替换的是 representation layer。它不再依赖 PKA 的手工行为特征，而是从 trace graph 中学习 kernel similarity。论文中的主流程是：

```text
NVBit SASS trace
  -> heterogeneous relational trace graph
  -> RGCN contrastive learning
  -> kernel embedding
  -> K-Means clustering
  -> representative simulation points
```

这份设计定义 A 线应该如何吸收 GCL-Sampler：既不破坏已经稳定的 PKA baseline，也不把 learned representation 工作混入下游 family 或 simulator 语义。

## 2. 目标

GCL 架构必须满足：

- 保留当前 representative compression contract：anchors、memberships、weights 和 evaluation metadata。
- 在 GCL mode 下，只替换 PKA 的 representation layer。
- 定义 trace、graph、embedding、selector 和 evaluation artifacts，并保证输入可 replay。
- 支持从 offline embeddings 到真实 trace graph learning 的分阶段实现路径。
- 保持 PKA 与 GCL 在 selector 和 anchor artifact 层面的可比较性。
- 每个阶段都要足够可审计，能够解释两个 kernel invocations 为什么被分到一起。

## 3. 非目标

这份设计不做：

- 声称完整复现 GCL-Sampler。
- 要求立即集成 NVBit。
- 要求立即搭建 RGCN training infrastructure。
- 替换 PKA-M1 baseline。
- 把 kernel name 作为主 grouping key。
- 把 B 线 family、regime、route primitive 或 simulator semantic metadata 引入 GCL selector。
- 在 simulator evaluation 阶段存在之前，声称 simulator accuracy 或 measured speedup。
- 定义生产环境 GPU tracing 权限、集群调度或长时间训练 orchestration。

## 4. 与 PKA Baseline 的关系

PKA 和 GCL 应共享同一个外层 selector 角色：

```text
selector input representation
  -> clustering
  -> representative anchors
  -> structural compression evaluation
```

二者的差异在于 selector input representation 如何产生。

PKA：

```text
measured 12D feature record
  -> log/clip/z-score preprocessing
  -> PCA projection
  -> K-Means
```

GCL：

```text
SASS trace
  -> heterogeneous relational graph
  -> RGCN encoder
  -> kernel embedding
  -> K-Means
```

Anchor table、cluster membership table、coverage weight、deterministic replay hash 和 structural compression summary 应保持结构可比较。这样 PKA 继续作为正式 baseline，GCL 则作为 representation replacement experiment。

## 5. 端到端 Pipeline

完整 GCL pipeline 分为六层。

### 5.1 Trace Acquisition

输入：

- workload invocation manifest
- selected kernel invocation identifiers
- tracing configuration

输出：

- 每个 kernel invocation 一个 trace bundle
- trace acquisition manifest
- acquisition gap report

Trace layer 只负责收集动态 SASS-level execution evidence。它不得决定 cluster membership。

### 5.2 Trace Graph Construction

输入：

- 按 kernel invocation 和 warp 分组后的 normalized trace entries

输出：

- 每个 kernel invocation 一个 graph bundle
- graph construction audit
- node and edge schema summary

Graph layer 把时间顺序 trace records 转换为带 typed nodes 和 typed directed edges 的 heterogeneous relational graphs。

### 5.3 Graph Preprocessing and Augmentation

输入：

- canonical graph bundles

输出：

- node feature tensors
- relation-indexed edge tensors
- augmentation manifest

这一层负责初始化 node vectors，并定义 contrastive views。它必须把 canonical graph bundle 与 augmented training views 分开保存。

### 5.4 RGCN Contrastive Learning

输入：

- training graph tensors
- augmentation configuration
- training configuration

输出：

- trained encoder checkpoint
- training metrics
- validation metrics
- model provenance manifest

这一层在无标签条件下学习 graph encoder，并产出可复用的 representation model。

### 5.5 Embedding Generation

输入：

- canonical graph tensors
- trained encoder checkpoint

输出：

- 每个 kernel invocation 一行 kernel embedding
- embedding metadata
- embedding replay hash

Embedding layer 是 GCL 中对应 PKA PCA projection artifact 的部分。

### 5.6 Clustering and Anchor Export

输入：

- GCL embedding table
- selector eligibility/audit metadata
- weight input

输出：

- GCL cluster assignment artifact
- GCL representative anchor table
- GCL structural compression evaluation

这一层应尽量复用现有 anchor 和 evaluation semantics。

## 6. 阶段计划

### 6.1 GCL-M0：Offline Embedding Selector

目的：

- 在不要求 trace graph generation 或 model training 的前提下，先验证 GCL 作为 selector representation 的可行性。

最小闭环定义：

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

这里的“最小闭环”有四层含义。

第一，M0 的输入已经是 embedding table，而不是 raw trace、trace graph 或 RGCN checkpoint。M0 不追问 embedding 如何产生，只要求 embedding row 满足 selector input contract。这样可以先验证 GCL representation 能否接入现有 representative compression 语义。

第二，M0 必须覆盖从输入校验到 artifact 写出的完整 selector 路径。它不能只停在“读入 embedding”或“跑出 cluster assignment”。一个 M0 run 必须能产出 anchor、membership、coverage、weight 和 structural evaluation，否则还没有形成可与 PKA 对齐的闭环。

第三，M0 的输出必须可 replay、可比较、可审计。相同 embedding rows、相同 `k_selection_mode` 和相同排序规则应得到稳定的 cluster / anchor / evaluation artifacts，并记录 replay hash、normalization config、K selection metadata 和 forbidden-field audit。

第四，M0 的闭环只验证 selector interface 和 artifact semantics，不验证 learned representation quality。它不能声称 simulator accuracy、cycle error、cross-architecture robustness 或 causal speedup。

输入：

- fixture embedding table，每个 kernel invocation 一行
- 可选 fixture weight input

输出：

- `gcl_embedding_table_l1.json`
- `gcl_kmeans_clusters_l1.json`
- `gcl_representative_anchor_table_l1.json`
- `gcl_compression_evaluation_l1.json`

必要行为：

- 校验 `record_id` / `kernel_invocation_id`、`representation_mode`、embedding dimensionality、finite numeric values、hash fields 和 forbidden fields。
- 按 stable `record_id` 排序，保证相同输入的 replay 顺序稳定。
- 对 embeddings 做 z-score normalization，并在 artifact 中记录 mean、std、zero-std dimensions 和 normalization mode。
- 默认使用 `silhouette_k` 做 K selection，同时支持 `deterministic_fixed_k` 作为显式 ablation mode。
- 在 normalized embeddings 上运行 deterministic K-Means，而不是使用 PKA 12D features。
- 每个 cluster 选择距离 centroid 最近的真实 record 作为 representative，不生成 synthetic representative。
- 计算 structural compression metrics，包括 anchor count、compression ratio、coverage count、weighted coverage、top-k coverage、anchor balance 和 cluster size distribution。
- 在字段重合处，输出与 PKA 结构一致的 anchors，并保留 GCL-specific `representation_mode` / embedding hash / K selection metadata。
- 保留 `representation_mode = "gcl_m0_embedding_fixture"`。

闭环边界：

- M0 不采集 NVBit trace。
- M0 不构建 trace graph。
- M0 不训练 RGCN。
- M0 不做 graph augmentation。
- M0 不运行 simulator。
- M0 不报告 sampled simulation accuracy。
- M0 不报告 causal performance contribution。

当 PKA-style selector 能由 embedding rows 驱动，并稳定产出可比较、可 replay、可审计的 GCL anchor / cluster / evaluation artifacts 时，M0 完成。

### 6.2 GCL-M1：Trace Graph Construction

目的：

- 从 trace-like inputs 构建 deterministic graph artifacts。

输入：

- trace fixture 或真实 trace subset
- trace schema manifest

输出：

- `gcl_trace_manifest_l1.json`
- `gcl_trace_graphs_l1.jsonl` 或 sharded graph bundle
- `gcl_graph_construction_audit_l1.json`

必要行为：

- 将 trace entries 解析为 per-kernel、per-warp sequences。
- 创建 instruction、pseudo 和 variable nodes。
- 创建 control-flow 和 data-flow edges。
- 记录 graph size、node type counts、edge type counts，以及 dropped/invalid trace entries。
- 对相同输入 trace 保持 graph construction deterministic。

当 graph artifacts 可以在不训练模型的情况下 replay 和 validate 时，M1 完成。

### 6.3 GCL-M2：RGCN Embedding and Selector

目的：

- 训练或加载 RGCN contrastive encoder，并从 canonical graph artifacts 生成 GCL embeddings。

输入：

- M1 产出的 graph bundles
- training configuration
- augmentation configuration

输出：

- `gcl_rgcn_training_report_l1.json`
- `gcl_rgcn_model_manifest_l1.json`
- `gcl_embedding_table_l1.json`
- `gcl_kmeans_clusters_l1.json`
- `gcl_representative_anchor_table_l1.json`
- `gcl_compression_evaluation_l1.json`

必要行为：

- 为每个 training graph 生成两个 augmented graph views。
- 使用 symmetric InfoNCE 训练。
- 为 selector 导出 canonical、non-augmented graph embeddings。
- 在 embeddings 上运行 clustering 和 anchor selection。
- 保留 training seed、model config、data split、checkpoint hash 和 embedding hash。

当 learned embeddings 能驱动与 M0 相同结构的 compression outputs 时，M2 完成。

### 6.4 GCL-M3：Simulator and Cross-Architecture Evaluation

目的：

- 评估 GCL-selected representatives 是否保持 simulator-relevant metrics。

输入：

- representative anchor table
- full workload metric table
- sampled representative metric table
- 可选 cross-GPU metric tables

输出：

- `gcl_simulator_accuracy_l1.json`
- `gcl_microarchitectural_metric_error_l1.json`
- `gcl_cross_architecture_transfer_l1.json`

必要行为：

- 比较 sampled reconstruction 与 full workload metrics。
- 报告 cycles 和 selected microarchitectural metrics 的 error。
- 将 structural compression speedup 与 measured simulator speedup 分开。
- 在 PKA 和 GCL 都可用时，对同一 workload set 进行比较。

只有存在 measured full-vs-sampled evaluation 时，M3 才算完成。

## 7. Artifact Contracts

### 7.1 Trace Manifest

Canonical path：

```text
artifacts/a_line/l1/gcl/gcl_trace_manifest_l1.json
```

每条 trace row 必须包含：

- `record_id`
- `kernel_invocation_id`
- `workload_id`
- `trace_path`
- `trace_format_version`
- `collection_scope`
- `selected_sm`
- `warp_count`
- `instruction_count`
- `status`
- `gap_reason`
- `trace_hash`

允许的 `status`：

- `collected`
- `missing`
- `invalid`
- `unsupported`

只有 `collected` rows 可以进入 graph construction。

### 7.2 Graph Bundle

Canonical path：

```text
artifacts/a_line/l1/gcl/gcl_trace_graphs_l1.jsonl
```

每条 graph record 必须包含：

- `record_id`
- `kernel_invocation_id`
- `graph_id`
- `graph_format_version`
- `node_table`
- `edge_table`
- `warp_partitions`
- `graph_statistics`
- `source_trace_hash`
- `graph_hash`

Graph bundle 是 canonical data。Training augmentations 不得覆盖它。

### 7.3 Embedding Table

Canonical path：

```text
artifacts/a_line/l1/gcl/gcl_embedding_table_l1.json
```

每条 embedding row 必须包含：

- `record_id`
- `kernel_invocation_id`
- `representation_mode`
- `embedding_dim`
- `embedding`
- `source_graph_hash`
- `encoder_manifest_hash`
- `embedding_hash`
- `weight_input`

Selector 可以读取 `record_id`、`kernel_invocation_id`、`embedding`、`representation_mode` 和 `weight_input`。它不得为了 clustering 读取 kernel name 或 downstream semantic metadata。

在 GCL-M0 中，embeddings 来自 fixture 或 offline rows，而不是真实 model outputs。此时 `source_graph_hash` 必须指向 fixture source hash，`encoder_manifest_hash` 必须指向 fixture embedding manifest hash。这些字段不得省略，也不得设置为无法解释的 null value。

### 7.4 Cluster Artifact

Canonical path：

```text
artifacts/a_line/l1/gcl/gcl_kmeans_clusters_l1.json
```

Cluster artifact 必须包含：

- `artifact_name`
- `mode`
- `representation_mode`
- `input_embedding_table_path`
- `input_embedding_table_hash`
- `k_selection`
- `k`
- `cluster_assignments`
- `members_by_cluster`
- `distance_to_centroid`
- `centroids`
- `inertia`
- `deterministic_replay_hash`

### 7.5 Anchor Artifact

Canonical path：

```text
artifacts/a_line/l1/gcl/gcl_representative_anchor_table_l1.json
```

Anchor artifact 必须包含：

- `artifact_name`
- `mode`
- `representation_mode`
- `selector_name`
- `embedding_dim`
- `clustering_config`
- `selection_rule`
- `forbidden_field_audit`
- `anchors`
- `input_embedding_table_hash`
- `deterministic_replay_hash`

每条 anchor row 必须包含：

- `anchor_id`
- `cluster_id`
- `representative_record_id`
- `members`
- `coverage_count`
- `coverage_weight`
- `weight`
- `representative_distance_to_centroid`
- `cluster_label`

## 8. Trace Acquisition Contract

Trace acquisition layer 最终应使用 NVBit-style SASS tracing，但架构允许在真实 NVBit 集成前先使用 fixture trace records。

每条 trace entry 应在可用时归一化这些字段：

- CTA coordinates：`tbx`、`tby`、`tbz`
- `warp_id`
- program counter：`pc`
- active lane mask：`mask`
- destination registers
- opcode
- source registers
- memory width
- dynamic operand values
- memory addresses when available

第一版实现可以只支持严格子集，但缺失字段必须显式进入 acquisition gap report。缺失 dynamic values 可能削弱 variable node initialization，但不得静默替换为伪造值。

Collection scope 必须被记录。长期推荐 scope 与论文一致：每个 kernel invocation 选择一个代表性 SM，并 trace 该 SM 上执行的所有 CTAs。如果使用不同 scope，必须在 manifest 和 evaluation report 中可见。

## 9. Trace Graph Construction

Graph construction 以 kernel invocation 为单位。

每个 invocation 内部：

1. 按 warp 对 trace entries 分组。
2. 保留每个 warp 内的 temporal order。
3. 为每个 warp 构建一个 directed graph。
4. 将 warp graphs union 成 kernel graph。
5. 记录 warp partitions，使 readout 可以先聚合到 warp，再聚合到 kernel level。

Graph construction 必须 deterministic：

- node ids 从 stable tuple keys 分配。
- edge ids 在 construction 后按稳定顺序输出。
- duplicate edge handling 必须写入 graph audit。
- invalid entries 必须记录 counts 和 reasons。

Graph builder 在创建 graph topology 时不得使用 runtime cycle counts、full-workload metric labels 或 simulator outcomes。

## 10. Node Schema

GCL 使用三类 nodes。

### 10.1 Instruction Nodes

Instruction nodes 表示已执行的 SASS instructions。

必要字段：

- `node_id`
- `node_type = "instruction"`
- `warp_id`
- `sequence_index`
- `pc`
- `opcode`
- `active_mask`

Initial feature inputs：

- opcode token id
- normalized PC
- optional active mask statistics

### 10.2 Pseudo Nodes

Pseudo nodes 表示应被 graph learning 看见、但并不作为单独 SASS instruction 存在的内部操作概念。

初始 pseudo node classes：

- `mem_ref`
- `address_calc`
- `predicate`

第一版实现可以只支持 `mem_ref`，但 schema 必须保留 typed pseudo nodes，使 graph 后续扩展时不需要改 artifact format。

### 10.3 Variable Nodes

Variable nodes 表示 dynamic values。

初始 variable node classes：

- register version node
- memory value/address node
- predicate value node

Variable nodes 按写入进行 versioning。每次 write 创建一个新的 variable node。之后的 reads 连接到 warp trace 内最近可见的版本。如果没有可见 producer，该 node 标记为 input variable。

Initial feature inputs：

- variable token id
- dynamic value summary when available
- 对 derived values 使用 zero vector，让其值通过 graph propagation 计算得到

## 11. Edge Schema

GCL 使用 typed directed edges。

必要 edge categories：

- `control_flow`：连接同一 warp trace 中连续的 instruction nodes。
- `data_left_source`：从 source variable 或 pseudo node 指向 operation/instruction node。
- `data_right_source`：从 source variable 或 pseudo node 指向 operation/instruction node。
- `data_destination`：从 operation/instruction node 指向 destination variable node。

如果 operand ordering 不可用，M0/M1 中可以把 source-side edge labels 泛化为 `data_source`，但 graph format 必须记录 operand position 是否已知。

每条 edge row 必须包含：

- `edge_id`
- `src_node_id`
- `dst_node_id`
- `edge_type`
- `warp_id`
- `source_trace_index`

## 12. Node Feature Initialization

Canonical graph artifact 存 semantic fields，而不是最终 tensors。Tensorization 会把 graph records 转换为 model inputs。

初始 tensorization target：

- uniform node feature width：64
- instruction token embedding 加 PC positional encoding
- variable token embedding 加 dynamic value summary
- pseudo token embedding
- 必要时 zero padding 到 64 维

如果 dynamic values 可用，dynamic value summary 应包括：

- mean
- standard deviation
- median
- minimum
- maximum
- 25th percentile
- 75th percentile
- skewness

如果 dynamic values 缺失，tensorizer 必须标记 summary missing，并使用 deterministic zero summary，同时在 tensorization metadata 中记录 missingness。

## 13. Graph Augmentation

Contrastive training 使用从 canonical graph tensors 派生出的 augmented views。

允许的 augmentations：

- node dropping
- edge dropping
- feature noise injection

默认配置：

- node dropping rate：`0.15`
- edge dropping rate：`0.15`
- feature noise standard deviation：`0.01`
- 每个 view 使用一种或两种 augmentations
- 每个 graph 生成两个 views

Augmentation 只用于 training。Selector embeddings 必须从 canonical、non-augmented graphs 生成。

Augmentation manifest 必须记录：

- random seed
- augmentation pool
- rates
- view generation policy
- 用于 training 和 validation 的 graph ids

## 14. RGCN Encoder and Contrastive Training

默认 encoder 与论文保持一致，除非 M0/M1 scope 需要更轻量的 fixture path。

默认 architecture：

- 3 个 RGCN layers
- input dimension：64
- hidden dimension：128
- graph embedding dimension：256
- 启用 basis decomposition
- convolution 后使用 layer normalization
- ReLU activation
- final RGCN layer 之外使用 dropout
- 从 node embeddings mean pooling 到 warp embeddings
- 从 warp embeddings average pooling 到 kernel embedding

Training projection head：

- MLP hidden dimension：128
- output dimension：64
- projection layers 之间使用 ReLU 和 dropout

Loss：

- symmetric InfoNCE
- 在 L2-normalized projection outputs 上计算 cosine similarity
- default temperature：`0.05`

Training metadata 必须包含：

- optimizer
- learning rate
- scheduler
- batch size
- epoch count
- random seeds
- train/validation split
- graph bundle hash
- model checkpoint hash

Selector 消费 projection head 之前的 256 维 kernel embedding，而不是 contrastive loss 使用的 64 维 projection output。

## 15. Embedding Generation

Embedding generation 是一个独立、可 replay 的步骤。

对每个 canonical graph：

1. 加载 graph tensor。
2. 在无 augmentation 条件下运行 trained encoder。
3. 读取 256 维 kernel embedding。
4. 每个 `record_id` 输出一条 embedding row。
5. 对 numeric normalization 后的 embedding row 计算 hash。

用于 clustering 的 embedding normalization 必须显式记录。默认做法是在 K-Means 前对 embedding dimensions 做 z-score normalization。Raw embedding 和 normalized embedding metadata 都必须能从 artifacts 中恢复。

## 16. Clustering and Representative Selection

GCL selector 应支持两种 K selection modes。

### 16.1 Deterministic Fixed-K Mode

目的：

- 作为 GCL-M0 的 baseline/ablation mode，让结果能与当前 deterministic PKA selector 直接比较。

默认：

```text
k = ceil(sqrt(n_records)), clamped to [2, n_records]
```

Initialization：

- deterministic farthest-first
- 第一个 center 是 lexicographically smallest `record_id`
- tie-breakers 使用 `record_id`

Representative：

- 选择距离 centroid 最近的真实 record

### 16.2 Silhouette-K Mode

目的：

- 对齐 GCL-Sampler 论文中的 clustering intent。
- 作为 GCL-M0 默认 mode，让 embedding space 自己决定更合适的 cluster count。

行为：

- 在有界 candidate K values 中评估。
- 选择 silhouette coefficient 最大的 K。
- 当多个 K 近似等价时，选择更小的 K。

GCL-M0 第一版应同时实现 deterministic fixed-K 和 silhouette-K。默认使用 silhouette-K，以便 M0 更贴近 GCL-Sampler 论文；deterministic fixed-K 必须作为显式 mode 保留，用于和 PKA selector 做 ablation 对照。

## 17. Evaluation Semantics

GCL 有三层 evaluation。

### 17.1 Structural Compression Evaluation

这一层只需要 selector artifacts。

Metrics：

- input record count
- anchor count
- compression ratio
- coverage count
- coverage weight
- top-k coverage
- anchor balance
- cluster size distribution

这一层可以在 M0 实现。

### 17.2 Representation Comparison

这一层在同一 record set 上比较 PKA 和 GCL。

Metrics：

- PKA anchor count vs GCL anchor count
- PKA top-k coverage vs GCL top-k coverage
- cluster agreement
- representative overlap
- embedding nearest-neighbor examples

这一层不得声称 simulator accuracy。

### 17.3 Full Metric Reconstruction Evaluation

这一层需要 full 和 sampled metric tables。

Metrics：

- cycle error
- IPC error
- L1 hit-rate error
- L2 hit-rate error
- achieved occupancy error
- measured or estimated speedup

这一层属于 M3。

## 18. Forbidden Fields and Audit

GCL selector 不得使用以下字段进行 clustering：

- `kernel_name`
- `source_path`
- `expected_behavior_axis`
- `family`
- `regime`
- `shape_hint`
- `trace_order`
- `grid_dim` string
- `block_dim` string
- simulator outcome fields
- full-workload cycle totals
- B-line semantic metadata

这些字段可以出现在单独的 audit 或 explanation artifacts 中，但 selector input table 不得包含它们，除非 selector 明确忽略并报告 forbidden-field violations。

## 19. Determinism and Replay

每个阶段都必须输出足够 metadata，用于 replay 或解释 outputs。

必要 hashes：

- trace file hash
- graph hash
- tensorization config hash
- augmentation config hash
- training graph bundle hash
- model checkpoint hash
- embedding table hash
- cluster assignment hash
- anchor table hash

随机阶段必须记录 seeds。Deterministic 阶段必须按 `record_id` 排序输入，除非 temporal order 是该阶段显式语义的一部分。

Floating point artifacts 应使用 stable JSON formatting 和 recorded precision。后续实现可以使用 sidecar binary files 存 arrays，但第一版 architecture-level artifacts 应保持 JSON 或 JSONL，便于 audit。

## 20. Acceptance Criteria

当 repo 中的 specs 和 implementation plans 能支持以下序列时，该总体架构满足要求：

1. 不改变 PKA-M1，继续作为 formal baseline 运行。
2. 在 fixture embeddings 上运行 GCL-M0，并产出 GCL anchors。
3. 从 trace-like inputs 构建 GCL-M1 trace graph artifacts。
4. 从 graph artifacts 运行 GCL-M2 embedding generation 和 clustering。
5. 在同一 record set 上比较 PKA 和 GCL structural compression。
6. 在声称 accuracy 或 speedup 之前，运行 GCL-M3 full-vs-sampled metric reconstruction。

## 21. Open Risks

主要风险：

- NVBit trace collection 在当前环境中可能成本高，且难以稳定复现。
- Trace graph size 可能很大，在 model training 之前就需要 sharding。
- Dynamic operand values 在早期 traces 中可能不可用，从而削弱 variable node features 的价值。
- RGCN training 引入了 PKA 中不存在的 dependency 和 GPU-resource requirements。
- Silhouette-K 更贴近论文，但如果不单独评估，可能降低与 deterministic PKA 的可比较性。
- Learned embedding 难以解释，因此必须保留 nearest-neighbor、cluster membership 和 graph statistics。

这些风险不阻塞 M0。它们应在 M1 和 M2 specs 中逐步处理。
