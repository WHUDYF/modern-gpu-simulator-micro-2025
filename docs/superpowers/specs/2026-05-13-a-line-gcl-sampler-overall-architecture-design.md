# A-line GCL-Sampler Overall Architecture Design

Date: 2026-05-13

## 1. Background

A-line currently has a PKA-compatible measured baseline path. That path turns invocation-level records into a fixed 12-dimensional behavior feature space, applies preprocessing and PCA, clusters the projected points, and emits representative anchors with membership and coverage metadata.

GCL-Sampler targets the same sampled GPU simulation problem, but changes the representation layer. Instead of relying on hand-crafted PKA features, it learns kernel similarity from trace graphs. The paper's pipeline is:

```text
NVBit SASS trace
  -> heterogeneous relational trace graph
  -> RGCN contrastive learning
  -> kernel embedding
  -> K-Means clustering
  -> representative simulation points
```

This design defines how A-line should absorb GCL-Sampler without losing the stable PKA baseline and without mixing learned representation work into downstream family or simulator concerns.

## 2. Goals

The GCL architecture must:

- Preserve the current representative compression contract: anchors, memberships, weights, and evaluation metadata.
- Replace only the PKA representation layer when running in GCL mode.
- Define trace, graph, embedding, selector, and evaluation artifacts with replayable inputs.
- Support a staged implementation path from offline embeddings to real trace graph learning.
- Keep PKA and GCL results comparable at the selector and anchor artifact level.
- Make every stage auditable enough to explain why two kernel invocations were grouped together.

## 3. Non-goals

This design does not:

- Claim a full GCL-Sampler reproduction.
- Require immediate NVBit integration.
- Require immediate RGCN training infrastructure.
- Replace the PKA-M1 baseline.
- Use kernel name as a primary grouping key.
- Introduce B-line family, regime, route primitive, or simulator semantic metadata into the GCL selector.
- Claim simulator accuracy or measured speedup before a simulator evaluation stage exists.
- Define production GPU tracing permissions, cluster scheduling, or long-running training orchestration.

## 4. Relationship to PKA Baseline

PKA and GCL should share the same outer selector role:

```text
selector input representation
  -> clustering
  -> representative anchors
  -> structural compression evaluation
```

They differ in how the selector input representation is produced.

PKA:

```text
measured 12D feature record
  -> log/clip/z-score preprocessing
  -> PCA projection
  -> K-Means
```

GCL:

```text
SASS trace
  -> heterogeneous relational graph
  -> RGCN encoder
  -> kernel embedding
  -> K-Means
```

The anchor table, cluster membership table, coverage weight, deterministic replay hash, and structural compression summary should remain structurally comparable. This keeps PKA as the formal baseline and GCL as a representation replacement experiment.

## 5. End-to-End Pipeline

The full GCL pipeline has six layers.

### 5.1 Trace Acquisition

Input:

- workload invocation manifest
- selected kernel invocation identifiers
- tracing configuration

Output:

- one trace bundle per kernel invocation
- trace acquisition manifest
- acquisition gap report

The trace layer is responsible only for collecting dynamic SASS-level execution evidence. It must not decide cluster membership.

### 5.2 Trace Graph Construction

Input:

- normalized trace entries grouped by kernel invocation and warp

Output:

- one graph bundle per kernel invocation
- graph construction audit
- node and edge schema summary

The graph layer turns temporal trace records into heterogeneous relational graphs with typed nodes and typed directed edges.

### 5.3 Graph Preprocessing and Augmentation

Input:

- canonical graph bundles

Output:

- node feature tensors
- relation-indexed edge tensors
- augmentation manifest

This layer initializes node vectors and defines contrastive views. It must preserve the canonical graph bundle separately from augmented training views.

### 5.4 RGCN Contrastive Learning

Input:

- training graph tensors
- augmentation configuration
- training configuration

Output:

- trained encoder checkpoint
- training metrics
- validation metrics
- model provenance manifest

This layer learns a graph encoder without labels. It produces a reusable representation model.

### 5.5 Embedding Generation

Input:

- canonical graph tensors
- trained encoder checkpoint

Output:

- one kernel embedding row per kernel invocation
- embedding metadata
- embedding replay hash

The embedding layer is the GCL equivalent of PKA's PCA projection artifact.

### 5.6 Clustering and Anchor Export

Input:

- GCL embedding table
- selector eligibility/audit metadata
- weight input

Output:

- GCL cluster assignment artifact
- GCL representative anchor table
- GCL structural compression evaluation

This layer should reuse the existing anchor and evaluation semantics wherever possible.

## 6. Stage Plan

### 6.1 GCL-M0: Offline Embedding Selector

Purpose:

- Validate GCL as a selector representation without requiring trace graph generation or model training.

Inputs:

- fixture embedding table, one row per kernel invocation
- optional fixture weight input

Outputs:

- `gcl_embedding_table_l1.json`
- `gcl_kmeans_clusters_l1.json`
- `gcl_representative_anchor_table_l1.json`
- `gcl_compression_evaluation_l1.json`

Required behavior:

- Validate embedding dimensionality and numeric values.
- Run clustering on embeddings, not PKA 12D features.
- Emit anchors in the same structural shape as PKA where fields overlap.
- Preserve `representation_mode` as `gcl_m0_embedding_fixture`.

M0 is complete when a PKA-style selector can be driven by embedding rows and produce comparable anchor artifacts.

### 6.2 GCL-M1: Trace Graph Construction

Purpose:

- Build deterministic graph artifacts from trace-like inputs.

Inputs:

- trace fixture or real trace subset
- trace schema manifest

Outputs:

- `gcl_trace_manifest_l1.json`
- `gcl_trace_graphs_l1.jsonl` or sharded graph bundle
- `gcl_graph_construction_audit_l1.json`

Required behavior:

- Parse trace entries into per-kernel and per-warp sequences.
- Create instruction, pseudo, and variable nodes.
- Create control-flow and data-flow edges.
- Record graph size, node type counts, edge type counts, and dropped/invalid trace entries.
- Keep graph construction deterministic for the same input trace.

M1 is complete when graph artifacts can be replayed and validated without training a model.

### 6.3 GCL-M2: RGCN Embedding and Selector

Purpose:

- Train or load an RGCN contrastive encoder and generate GCL embeddings from canonical graph artifacts.

Inputs:

- graph bundles from M1
- training configuration
- augmentation configuration

Outputs:

- `gcl_rgcn_training_report_l1.json`
- `gcl_rgcn_model_manifest_l1.json`
- `gcl_embedding_table_l1.json`
- `gcl_kmeans_clusters_l1.json`
- `gcl_representative_anchor_table_l1.json`
- `gcl_compression_evaluation_l1.json`

Required behavior:

- Generate two augmented graph views per training graph.
- Train with symmetric InfoNCE.
- Export canonical, non-augmented graph embeddings for selector use.
- Run clustering and anchor selection on embeddings.
- Preserve training seed, model config, data split, checkpoint hash, and embedding hash.

M2 is complete when the learned embeddings can drive the same structural compression outputs as M0.

### 6.4 GCL-M3: Simulator and Cross-Architecture Evaluation

Purpose:

- Evaluate whether GCL-selected representatives preserve simulator-relevant metrics.

Inputs:

- representative anchor table
- full workload metric table
- sampled representative metric table
- optional cross-GPU metric tables

Outputs:

- `gcl_simulator_accuracy_l1.json`
- `gcl_microarchitectural_metric_error_l1.json`
- `gcl_cross_architecture_transfer_l1.json`

Required behavior:

- Compare sampled reconstruction against full workload metrics.
- Report error for cycles and selected microarchitectural metrics.
- Keep structural compression speedup separate from measured simulator speedup.
- Compare PKA and GCL on the same workload set when both inputs are available.

M3 is complete only when measured full-vs-sampled evaluation exists.

## 7. Artifact Contracts

### 7.1 Trace Manifest

Canonical path:

```text
artifacts/a_line/l1/gcl/gcl_trace_manifest_l1.json
```

Each trace row must include:

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

Allowed `status` values:

- `collected`
- `missing`
- `invalid`
- `unsupported`

Only `collected` rows can enter graph construction.

### 7.2 Graph Bundle

Canonical path:

```text
artifacts/a_line/l1/gcl/gcl_trace_graphs_l1.jsonl
```

Each graph record must include:

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

The graph bundle is canonical data. Training augmentations must not overwrite it.

### 7.3 Embedding Table

Canonical path:

```text
artifacts/a_line/l1/gcl/gcl_embedding_table_l1.json
```

Each embedding row must include:

- `record_id`
- `kernel_invocation_id`
- `representation_mode`
- `embedding_dim`
- `embedding`
- `source_graph_hash`
- `encoder_manifest_hash`
- `embedding_hash`
- `weight_input`

The selector may read `record_id`, `kernel_invocation_id`, `embedding`, `representation_mode`, and `weight_input`. It must not read kernel name or downstream semantic metadata for clustering.

In GCL-M0, where embeddings are fixture or offline rows rather than model outputs, `source_graph_hash` must point to the fixture source hash and `encoder_manifest_hash` must point to the fixture embedding manifest hash. These fields must not be omitted or set to an unexplained null value.

### 7.4 Cluster Artifact

Canonical path:

```text
artifacts/a_line/l1/gcl/gcl_kmeans_clusters_l1.json
```

The cluster artifact must include:

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

Canonical path:

```text
artifacts/a_line/l1/gcl/gcl_representative_anchor_table_l1.json
```

The anchor artifact must include:

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

Each anchor row must include:

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

The trace acquisition layer should eventually use NVBit-style SASS tracing, but the architecture allows fixture trace records before real NVBit integration.

Each trace entry should normalize these fields when available:

- CTA coordinates: `tbx`, `tby`, `tbz`
- `warp_id`
- program counter: `pc`
- active lane mask: `mask`
- destination registers
- opcode
- source registers
- memory width
- dynamic operand values
- memory addresses when available

The first implementation may support a strict subset, but missing fields must be explicit in the acquisition gap report. Missing dynamic values may make variable node initialization weaker, but they must not be silently replaced with fabricated values.

Collection scope must be recorded. The preferred long-term scope follows the paper's single-representative-SM strategy: trace all CTAs executed on one selected SM for each kernel invocation. If a different scope is used, the scope must be visible in the manifest and evaluation report.

## 9. Trace Graph Construction

Graph construction is per kernel invocation.

Within each invocation:

1. Group trace entries by warp.
2. Preserve temporal order inside each warp.
3. Build one directed graph per warp.
4. Union warp graphs into the kernel graph.
5. Record warp partitions so readout can aggregate by warp before kernel-level pooling.

Graph construction must be deterministic:

- node ids are assigned from stable tuple keys
- edge ids are emitted in stable sorted order after construction
- duplicate edge handling is specified in the graph audit
- invalid entries are reported with counts and reasons

The graph builder must not use runtime cycle counts, full-workload metric labels, or simulator outcomes when creating graph topology.

## 10. Node Schema

GCL uses three node categories.

### 10.1 Instruction Nodes

Instruction nodes represent executed SASS instructions.

Required fields:

- `node_id`
- `node_type = "instruction"`
- `warp_id`
- `sequence_index`
- `pc`
- `opcode`
- `active_mask`

Initial feature inputs:

- opcode token id
- normalized PC
- optional active mask statistics

### 10.2 Pseudo Nodes

Pseudo nodes represent internal operation concepts that should be visible to graph learning but do not exist as separate SASS instructions.

Initial pseudo node classes:

- `mem_ref`
- `address_calc`
- `predicate`

The first implementation may support only `mem_ref`, but the schema must reserve typed pseudo nodes so the graph can grow without changing the artifact format.

### 10.3 Variable Nodes

Variable nodes represent dynamic values.

Initial variable node classes:

- register version node
- memory value/address node
- predicate value node

Variable nodes are versioned by writes. A new variable node is created for each write. Later reads connect to the most recent visible version within the warp trace. If no visible producer exists, the node is marked as an input variable.

Initial feature inputs:

- variable token id
- dynamic value summary when available
- zero vector for derived values whose value should be computed through graph propagation

## 11. Edge Schema

GCL uses typed directed edges.

Required edge categories:

- `control_flow`: connects consecutive instruction nodes in a warp trace.
- `data_left_source`: source variable or pseudo node to operation/instruction node.
- `data_right_source`: source variable or pseudo node to operation/instruction node.
- `data_destination`: operation/instruction node to destination variable node.

The exact source-side edge labels may be generalized to `data_source` in M0/M1 if operand ordering is unavailable, but the graph format must record whether operand position is known.

Each edge row must include:

- `edge_id`
- `src_node_id`
- `dst_node_id`
- `edge_type`
- `warp_id`
- `source_trace_index`

## 12. Node Feature Initialization

The canonical graph artifact stores semantic fields, not final tensors. Tensorization converts graph records into model inputs.

Initial tensorization target:

- uniform node feature width: 64
- instruction token embedding plus PC positional encoding
- variable token embedding plus dynamic value summary
- pseudo token embedding
- zero padding to 64 dimensions where needed

Dynamic value summary should include, when values are available:

- mean
- standard deviation
- median
- minimum
- maximum
- 25th percentile
- 75th percentile
- skewness

If dynamic values are missing, the tensorizer must mark the summary as missing and use a deterministic zero summary, with the missingness recorded in tensorization metadata.

## 13. Graph Augmentation

Contrastive training uses augmented views derived from canonical graph tensors.

Allowed augmentations:

- node dropping
- edge dropping
- feature noise injection

Default configuration:

- node dropping rate: `0.15`
- edge dropping rate: `0.15`
- feature noise standard deviation: `0.01`
- one or two augmentations per view
- two views per graph

Augmentation must be training-only. Selector embeddings must be generated from canonical, non-augmented graphs.

The augmentation manifest must record:

- random seed
- augmentation pool
- rates
- view generation policy
- graph ids used for training and validation

## 14. RGCN Encoder and Contrastive Training

The default encoder mirrors the paper unless M0/M1 scope requires a lighter fixture path.

Default architecture:

- 3 RGCN layers
- input dimension: 64
- hidden dimension: 128
- graph embedding dimension: 256
- basis decomposition enabled
- layer normalization after convolution
- ReLU activation
- dropout except on the final RGCN layer
- mean pooling from node embeddings to warp embeddings
- average pooling from warp embeddings to kernel embedding

Projection head for training:

- MLP hidden dimension: 128
- output dimension: 64
- ReLU and dropout between projection layers

Loss:

- symmetric InfoNCE
- cosine similarity on L2-normalized projection outputs
- default temperature: `0.05`

Training metadata must include:

- optimizer
- learning rate
- scheduler
- batch size
- epoch count
- random seeds
- train/validation split
- graph bundle hash
- model checkpoint hash

The selector consumes the 256-dimensional kernel embedding before the projection head, not the 64-dimensional projection output used for contrastive loss.

## 15. Embedding Generation

Embedding generation is a separate replayable step.

For each canonical graph:

1. Load graph tensor.
2. Run the trained encoder without augmentation.
3. Read out the 256-dimensional kernel embedding.
4. Emit one embedding row per `record_id`.
5. Hash the embedding row after numeric normalization.

Embedding normalization for clustering should be explicit. The default is z-score normalization across embedding dimensions before K-Means. The raw embedding and normalized embedding metadata must both be recoverable from artifacts.

## 16. Clustering and Representative Selection

The GCL selector should support two K selection modes.

### 16.1 Deterministic Fixed-K Mode

Purpose:

- Make GCL-M0 comparable to the current deterministic PKA selector.

Default:

```text
k = ceil(sqrt(n_records)), clamped to [2, n_records]
```

Initialization:

- deterministic farthest-first
- first center is lexicographically smallest `record_id`
- tie-breakers use `record_id`

Representative:

- nearest real record to centroid

### 16.2 Silhouette-K Mode

Purpose:

- Match GCL-Sampler's paper-level clustering intent.

Behavior:

- evaluate candidate K values in a bounded range
- choose K with maximum silhouette coefficient
- if multiple K values are comparable, choose the smaller K

The first implementation should keep deterministic fixed-K as the default and treat silhouette-K as an explicit mode. This prevents early results from mixing representation quality with changing K policy.

## 17. Evaluation Semantics

GCL has three evaluation layers.

### 17.1 Structural Compression Evaluation

This layer requires only selector artifacts.

Metrics:

- input record count
- anchor count
- compression ratio
- coverage count
- coverage weight
- top-k coverage
- anchor balance
- cluster size distribution

This layer can be implemented in M0.

### 17.2 Representation Comparison

This layer compares PKA and GCL on the same record set.

Metrics:

- PKA anchor count vs GCL anchor count
- PKA top-k coverage vs GCL top-k coverage
- cluster agreement
- representative overlap
- embedding nearest-neighbor examples

This layer should not claim simulator accuracy.

### 17.3 Full Metric Reconstruction Evaluation

This layer requires full and sampled metric tables.

Metrics:

- cycle error
- IPC error
- L1 hit-rate error
- L2 hit-rate error
- achieved occupancy error
- measured or estimated speedup

This layer belongs to M3.

## 18. Forbidden Fields and Audit

The GCL selector must not use these fields for clustering:

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

These fields may appear in separate audit or explanation artifacts, but the selector input table must not contain them unless the selector explicitly ignores and reports them as forbidden-field violations.

## 19. Determinism and Replay

Every stage must emit enough metadata to replay or explain outputs.

Required hashes:

- trace file hash
- graph hash
- tensorization config hash
- augmentation config hash
- training graph bundle hash
- model checkpoint hash
- embedding table hash
- cluster assignment hash
- anchor table hash

Randomized stages must record seeds. Deterministic stages must sort inputs by `record_id` unless temporal order is part of the stage's explicit semantics.

Floating point artifacts should use stable JSON formatting and recorded precision. The implementation may use arrays in sidecar binary files later, but the first architecture-level artifacts should remain JSON or JSONL for auditability.

## 20. Acceptance Criteria

The overall architecture is satisfied when the repo has specs and implementation plans that make the following sequence possible:

1. Run PKA-M1 unchanged as the formal baseline.
2. Run GCL-M0 on fixture embeddings and produce GCL anchors.
3. Build GCL-M1 trace graph artifacts from trace-like inputs.
4. Run GCL-M2 embedding generation and clustering from graph artifacts.
5. Compare PKA and GCL structural compression on the same record set.
6. Run GCL-M3 full-vs-sampled metric reconstruction before making accuracy or speedup claims.

## 21. Open Risks

The main risks are:

- NVBit trace collection may be expensive or hard to reproduce in the current environment.
- Trace graph size may require sharding before model training is practical.
- Dynamic operand values may be unavailable in early traces, reducing the value of variable node features.
- RGCN training introduces dependency and GPU-resource requirements not present in PKA.
- Silhouette-K may improve paper alignment but reduce comparability with deterministic PKA unless evaluated separately.
- A learned embedding can be hard to explain unless nearest-neighbor, cluster membership, and graph statistics are preserved.

These risks do not block M0. They should be addressed incrementally in M1 and M2 specs.
