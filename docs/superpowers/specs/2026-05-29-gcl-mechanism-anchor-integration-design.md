# GCL Mechanism Anchor Integration Design

日期：2026-05-29

## 1. 目标

本文档定义如何在复现 GCL-Sampler 的基础上，把它扩展为服务 mechanism attribution 和 simulator tuning validation 的结构相似性模块。

核心判断：

```text
GCL 不直接输出调参 knob；
GCL 输出 kernel / anchor 的结构 embedding、相似性和 cluster；
mechanism microbench anchors 提供机制原型；
registry 提供 subtype-to-knob 约束；
validation planner 决定哪些候选 knob 值得跑 closed-loop validation。
```

改造后的链路：

```text
real kernel trace graph
  + mechanism microbench anchor graph
  -> shared GCL encoder
  -> kernel / anchor embeddings
  -> nearest mechanism-anchor matching
  -> family / subtype candidates
  -> registry-constrained knob candidates
  -> validation priority
  -> simulator before/after validation
```

第一版目标是跑通：

```text
real kernel -> top-k mechanism microbench anchors -> top-k subtype/knob candidates
```

而不是立即训练一个端到端调参模型。

---

## 2. 非目标

第一版不做：

1. 不让 GCL embedding 单独决定 final family / subtype；
2. 不让 GCL 直接修改 simulator config；
3. 不把 nearest anchor similarity 当作 calibrated probability；
4. 不把 design-intent-only microbench 当作 validated label；
5. 不直接预测 speedup 或 validated contribution；
6. 不要求第一版 fine-tune GCL encoder；
7. 不要求完整实现 multi-task GNN head。

GCL 的职责是结构相似性和 representative selection。最终 claim-bearing 结论必须来自 C-Line simulator validation。

---

## 3. 保留的 GCL 原始能力

第一阶段尽量复现 GCL-Sampler 的原始能力，不急于修改训练目标。

保留链路：

```text
SASS / trace
  -> heterogeneous relational graph
  -> R-GCN / graph contrastive learning
  -> kernel embedding
  -> KMeans / clustering
  -> representative selection
```

第一阶段必须导出：

```text
kernel_embeddings
kernel_clusters
representative_kernels
cluster_members
cluster_time_weight
cluster_purity_or_stability
```

这些 artifact 回答：

```text
哪些 kernel 在结构上相似？
哪些 kernel 可以由同一个 representative 覆盖？
cluster 内部是否稳定？
```

---

## 4. 新增机制锚点

### 4.1 Microbench Anchor

新增 `mechanism_microbench_anchor` 作为 GCL embedding 空间中的机制原型。

每个 anchor 至少包含：

```json
{
  "anchor_id": "mb_fp64_pipeline_v1",
  "anchor_type": "mechanism_microbench",
  "target_family": "dense_compute",
  "target_subtype": "fp64_dp_pipeline_compute",
  "target_knobs": [
    "trace_opcode_latency_initiation_dp"
  ],
  "label_quality": "knob_validated",
  "claim_status": "simulator_internal_closed_loop"
}
```

### 4.2 Anchor Label Quality

microbench anchor 标签质量分层：

| Quality | Meaning | Weight |
| --- | --- | ---: |
| `design_intent_only` | LLM / human 设计目标，尚未被 profiler 确认 | 0.30 |
| `counter_confirmed` | profiler / counter 显示目标机制被激活 | 0.70 |
| `knob_validated` | simulator before/after 验证通过 | 1.00 |
| `rejected_or_invalid` | 编译、运行、counter 或 validation 失败 | 0.00 |

`label_quality` 用于调节 anchor 对 real kernel candidate score 的影响。它不是概率。

### 4.3 Real Kernel 与 Anchor 同空间编码

real workload kernel 和 microbench anchor 都构造成 GCL graph，并使用同一个 encoder：

```text
embedding(real_kernel_i) = GCL_encoder(graph(real_kernel_i))
embedding(microbench_anchor_j) = GCL_encoder(graph(microbench_anchor_j))
```

这样可以计算：

```text
similarity(real_kernel_i, microbench_anchor_j)
```

---

## 5. Nearest-Anchor Mechanism Matching

### 5.1 基本匹配

对每个 real kernel / representative anchor，找 top-k mechanism microbench anchors：

```text
top_k_anchors(real_kernel) =
  nearest microbench anchors in embedding space
```

推荐第一版使用：

```text
cosine similarity
top_k = 5
```

输出示例：

```json
{
  "record_id": "real_kernel_A",
  "nearest_microbench_anchors": [
    {
      "anchor_id": "mb_fp64_pipeline_v1",
      "similarity": 0.82,
      "target_subtype": "fp64_dp_pipeline_compute",
      "label_quality": "knob_validated",
      "label_quality_weight": 1.0
    }
  ]
}
```

### 5.2 Mechanism Candidate Score

第一版 score 使用确定性公式：

```text
anchor_candidate_score =
  similarity
  * label_quality_weight
  * evidence_quality_weight
```

如果多个 anchors 指向同一个 subtype：

```text
subtype_score =
  weighted_top_mean(anchor_candidate_scores for subtype)
```

推荐第一版使用 top-2 weighted mean，避免单个异常 anchor 主导。

### 5.3 Mixed Mechanism

真实 kernel 可能同时靠近多个 anchors。第一版不强行单标签，输出 distribution / ranking：

```json
{
  "subtype_candidates": [
    {
      "subtype_id": "tensor_core_pipeline_compute",
      "score": 0.45
    },
    {
      "subtype_id": "shared_memory_tiled_compute",
      "score": 0.32
    },
    {
      "subtype_id": "occupancy_limited_compute",
      "score": 0.18
    }
  ],
  "claim_status": "candidate_not_validated"
}
```

如果 top candidates 分数接近且来自不同 family，应标记：

```text
boundary_or_mixed_candidate
```

---

## 6. Registry 与 Knob 映射

GCL embedding 不知道 simulator knob 合法性，因此必须接 registry。

输入：

```text
docs/family_criteria/canonical_family_enum_v1_2026-05-14.json
docs/family_criteria/canonical_family_subtype_registry_v1_2026-05-14.json
docs/family_criteria/subtype_to_simulator_knob_map_v1_2026-05-16.json
```

规则：

1. subtype 必须属于合法 family；
2. knob 必须来自 subtype-to-knob map；
3. 没有 knob map 的 subtype 可以进入 mechanism candidates，但不能进入 executable validation plan；
4. registry-invalid candidate 必须从输出中删除，并记录 audit；
5. knob candidate 的 claim status 只能是 `ready_for_validation` 或更低，不能是 validated。

输出示例：

```json
{
  "record_id": "real_kernel_A",
  "knob_candidates": [
    {
      "knob": "trace_opcode_latency_initiation_dp",
      "source_subtype": "fp64_dp_pipeline_compute",
      "score": 0.78,
      "registry_status": "valid",
      "claim_status": "ready_for_validation"
    }
  ],
  "registry_projection": {
    "invalid_candidates_removed": 1
  }
}
```

---

## 7. Validation Priority

GCL 与调参真正连接在 validation priority，而不是直接 prescription。

第一版 priority：

```text
validation_priority =
  time_weight
  * anchor_similarity_score
  * anchor_label_quality
  * group_purity
  * registry_knob_readiness
  * evidence_quality
```

字段含义：

| Field | Meaning |
| --- | --- |
| `time_weight` | kernel / cluster 在 workload 中的时间占比 |
| `anchor_similarity_score` | real kernel 与 mechanism anchor 的结构相似度 |
| `anchor_label_quality` | anchor 是 design-only、counter-confirmed 还是 knob-validated |
| `group_purity` | GCL cluster 内部一致性 |
| `registry_knob_readiness` | subtype 是否有明确 knob map |
| `evidence_quality` | measured / proxy / missing-feature 状态 |

输出必须声明：

```text
validation_priority 不是 importance contribution；
validation_priority 不是 predicted speedup；
validation_priority 只是有限验证预算下的排序分。
```

---

## 8. 分阶段改造计划

### Stage 0 - GCL Reproduction

目标：

```text
复现原 GCL-Sampler 的 graph construction、embedding、clustering、representative selection。
```

输出：

```text
gcl_kernel_embeddings.json / .npy
gcl_kernel_clusters.json
gcl_representative_selection.json
gcl_reproduction_report.md
```

验收：

```text
能对一组 real kernels 生成 stable embeddings 和 clusters。
```

### Stage 1 - Microbench Anchor Encoding

目标：

```text
把 mechanism microbench anchors 构造成与 real kernels 同 schema 的 graph，并送入同一个 GCL encoder。
```

输出：

```text
microbench_anchor_graphs/
microbench_anchor_embeddings.json
microbench_anchor_dataset_card.md
```

验收：

```text
每个 anchor 有 label_quality、provenance 和 graph embedding。
```

### Stage 2 - Nearest-Anchor Matching

目标：

```text
对每个 real kernel / representative 找 top-k mechanism anchors。
```

输出：

```text
gcl_nearest_mechanism_anchors.json
gcl_anchor_matching_report.md
```

验收：

```text
每个 real kernel 有 top-k anchors、similarity、anchor label quality 和 audit。
```

### Stage 3 - Registry-Constrained Candidate Export

目标：

```text
把 nearest anchors 转换成 family / subtype / knob candidates。
```

输出：

```text
gcl_mechanism_candidates.json
gcl_knob_candidates.json
gcl_registry_projection_audit.json
```

验收：

```text
所有 subtype / knob candidates 都通过 registry 校验。
```

### Stage 4 - Validation Priority Export

目标：

```text
生成 C-Line validation planner 可消费的 priority artifact。
```

输出：

```text
gcl_validation_priority.json
gcl_to_validation_planner_report.md
```

验收：

```text
priority 包含 time_weight、similarity、label_quality、group_purity、knob_readiness 和 evidence_quality。
```

### Stage 5 - Feedback Loop

目标：

```text
把 validation result 回写为新的 high-quality anchor / feedback edge。
```

输出：

```text
gcl_validation_feedback_edges.json
updated_microbench_anchor_quality.json
```

验收：

```text
successful validation 可提升相关 anchor / subtype 的 label_quality；
failed validation 可降低或 block 对应 candidate。
```

### Stage 6 - Optional Fine-Tuning

目标：

```text
在有足够 counter-confirmed / knob-validated 数据后，fine-tune GCL encoder 或增加 ranking heads。
```

可选训练目标：

```text
prototype contrastive loss
family ranking loss
subtype ranking loss
knob ranking loss
abstain / boundary loss
```

第一版不依赖该阶段。

---

## 9. Artifact Schema

### 9.1 `gcl_nearest_mechanism_anchors.json`

```json
{
  "artifact_name": "gcl_nearest_mechanism_anchors",
  "schema_version": "v0.1",
  "rows": [
    {
      "record_id": "real_kernel_A",
      "cluster_id": "gcl_cluster_03",
      "nearest_anchors": [
        {
          "anchor_id": "mb_fp64_pipeline_v1",
          "similarity": 0.82,
          "target_family": "dense_compute",
          "target_subtype": "fp64_dp_pipeline_compute",
          "label_quality": "knob_validated",
          "claim_status": "simulator_internal_closed_loop"
        }
      ]
    }
  ]
}
```

### 9.2 `gcl_mechanism_candidates.json`

```json
{
  "artifact_name": "gcl_mechanism_candidates",
  "schema_version": "v0.1",
  "rows": [
    {
      "record_id": "real_kernel_A",
      "family_candidates": [
        {
          "family_id": "dense_compute",
          "score": 0.78,
          "source": "nearest_microbench_anchor",
          "claim_status": "candidate_not_validated"
        }
      ],
      "subtype_candidates": [
        {
          "subtype_id": "fp64_dp_pipeline_compute",
          "family_id": "dense_compute",
          "score": 0.78,
          "source_anchor_ids": [
            "mb_fp64_pipeline_v1"
          ],
          "claim_status": "candidate_not_validated"
        }
      ],
      "boundary_flags": []
    }
  ]
}
```

### 9.3 `gcl_validation_priority.json`

```json
{
  "artifact_name": "gcl_validation_priority",
  "schema_version": "v0.1",
  "rows": [
    {
      "record_id": "real_kernel_A",
      "cluster_id": "gcl_cluster_03",
      "candidate_knob": "trace_opcode_latency_initiation_dp",
      "source_subtype": "fp64_dp_pipeline_compute",
      "priority_score": 0.71,
      "priority_terms": {
        "time_weight": 0.76,
        "anchor_similarity_score": 0.82,
        "anchor_label_quality": 1.0,
        "group_purity": 0.86,
        "registry_knob_readiness": 1.0,
        "evidence_quality": 0.95
      },
      "claim_status": "validation_priority_not_contribution"
    }
  ]
}
```

---

## 10. 与现有 B-Line / C-Line 的接口

### 10.1 输入给 B-Line Graph Reasoner

B-Line 可以把 GCL 输出作为 graph edges：

```text
kernel_to_microbench_anchor_similarity
microbench_anchor_to_subtype
subtype_to_knob
cluster_member_similarity
validation_supports_anchor
validation_rejects_anchor
```

这些 edges 进入 NetworkX reasoner 或后续 learned GNN。

### 10.2 输入给 C-Line Validation Planner

C-Line 消费：

```text
gcl_knob_candidates.json
gcl_validation_priority.json
gcl_registry_projection_audit.json
```

C-Line 仍负责：

```text
选择 top-k validation items
加入 control kernels
生成 baseline / modified run plan
导入 validation results
回写 feedback
```

---

## 11. 风险与防线

### 11.1 Microbench 与 real kernel 分布差

防线：

```text
只把 microbench 当 prototype，不当 real workload ground truth。
输出 top-k candidates 和 boundary flags。
用 validation feedback 校正。
```

### 11.2 Similarity 被误读成概率

防线：

```text
字段命名使用 similarity / score，不使用 probability。
只有校准后才允许输出 calibrated_probability。
```

### 11.3 GCL cluster 内部混合机制

防线：

```text
输出 group_purity。
低 purity cluster 只能进入 boundary / review，不进入 full validation。
```

### 11.4 Registry 缺 knob

防线：

```text
没有 knob map 的 subtype 只保留 mechanism candidate，不生成 executable validation item。
```

### 11.5 过早 fine-tune

防线：

```text
第一版只做 frozen encoder + nearest-anchor matching。
等 counter-confirmed / knob-validated 数据足够后再训练 ranking heads。
```

---

## 12. 第一版完成标准

第一版完成时应满足：

1. 已复现或读取 GCL kernel embeddings；
2. microbench anchors 能进入同一个 embedding 空间；
3. real kernel 能输出 top-k nearest mechanism anchors；
4. anchor candidates 能映射到合法 family / subtype；
5. subtype candidates 能通过 registry 映射到 knob candidates；
6. validation priority 明确包含 time、similarity、quality、purity、readiness；
7. 输出中没有 final attribution、speedup prediction 或 validated contribution；
8. C-Line 可以基于输出生成 simulator validation plan；
9. validation feedback 能回写为 anchor quality 或 candidate support 的更新；
10. 所有 artifact 都有 provenance 和 claim_status。

---

## 13. 推荐落地顺序

推荐顺序：

```text
1. Stage 0: 复现 GCL embeddings / clusters
2. Stage 1: 准备 microbench anchor graphs
3. Stage 2: nearest-anchor matching
4. Stage 3: registry-constrained subtype / knob export
5. Stage 4: validation priority export
6. Stage 5: 少量 simulator closed-loop validation
7. Stage 6: 有足够反馈后再 fine-tune
```

一句话总结：

```text
我们不是把 GCL 改成调参模型；
我们把 GCL 改成 real kernel 与 mechanism microbench anchor 的结构匹配器，
再通过 registry 和 validation planner 把结构相似性转成可验证的 knob candidates。
```
