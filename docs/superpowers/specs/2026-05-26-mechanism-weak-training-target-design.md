# Mechanism Weak Training Target Design

日期：2026-05-26

## 1. 目标

本文档定义第一版 mechanism attribution 训练目标：给定一个 kernel / invocation / regime 的 evidence pack，训练模型输出 `family`、`family_subtype` 和 `simulator_knob` 的候选排序，并显式支持 abstain / boundary 判断。

核心链路：

```text
Measured / proxy evidence records
  + registry constraints
  + LLM weak judgments
  + rule-based weak votes
  + sparse validation feedback
  -> weak training targets
  -> candidate ranking model
  -> registry-constrained ranked candidates
  -> graph reasoner / validation planner
```

第一版训练目标不是预测唯一真值，而是学习一个可审计的候选排序器。它服务于后续 B-Line graph reasoner 和 C-Line validation planner，让系统优先验证最可能有效的 family / subtype / knob 路径。

---

## 2. 非目标

第一版不做：

1. 不直接预测 speedup；
2. 不直接预测 validated contribution ratio；
3. 不把 instruction ratio、opcode ratio 或 memory ratio 当作重要性归因；
4. 不把 LLM 输出当作 ground truth；
5. 不训练 end-to-end 黑盒调参模型；
6. 不绕过 family / subtype / knob registry；
7. 不强迫 boundary / mixed / insufficient-evidence 样本给出单一标签。

LLM 在本设计中只提供 weak candidate judgments。真正 claim-bearing 的标签只能来自 simulator closed-loop validation 或明确的人类审计。

---

## 3. 第一版训练任务

### 3.1 任务定义

输入：

```text
kernel evidence pack
```

输出：

```text
ranked family candidates
ranked subtype candidates
ranked knob candidates
abstain / boundary decision
```

候选必须满足：

```text
subtype belongs to selected family
knob maps to selected subtype
invalid registry combinations are removed after scoring
```

### 3.2 推荐训练目标

第一版采用 multi-task candidate ranking：

```text
Task A: family ranking
Task B: subtype ranking conditioned on family
Task C: knob ranking conditioned on subtype
Task D: abstain / boundary detection
```

不建议第一版做 flat classification。原因是 family、subtype、knob 存在层级依赖，且第一版标签质量不均匀。ranking 比单标签分类更适合 weak supervision 场景，也更容易服务 validation planner。

---

## 4. 数据集分层

### 4.1 Raw Evidence Dataset

文件建议：

```text
artifacts/mechanism_training/<run_name>/raw_evidence_records.jsonl
```

每行一个 evidence record：

```json
{
  "record_id": "rodinia_backprop.bpnn_adjust_weights_cuda",
  "workload_id": "rodinia_backprop",
  "kernel_name": "bpnn_adjust_weights_cuda",
  "source_anchor_id": null,
  "phase_id": "backprop",
  "time_weight": 0.63,
  "evidence_status": "measured",
  "features": {
    "compute_to_memory_ratio": 8.3,
    "memory_ratio": 0.118,
    "branch_ratio": 0.021,
    "sync_ratio": 0.0,
    "dp_opcode_ratio": 0.64,
    "fp32_opcode_ratio": 0.126,
    "tensor_op_ratio": 0.0,
    "shared_memory_ratio": 0.0,
    "global_memory_ratio": 0.111,
    "instruction_density": 0.82
  },
  "provenance": {
    "source_artifact": "artifacts/rodinia_backprop_dp_prescription_rerun/comparison_summary.json",
    "claim_status": "measured"
  }
}
```

要求：

1. `record_id` 必须唯一；
2. `features` 缺失不能默认补 0；
3. `kernel_name` 可以提供给 LLM 和 audit，但不能单独作为 validated label 来源；
4. `evidence_status` 必须区分 `measured`、`proxy`、`synthetic_fixture`、`manual_seed`；
5. ratio 字段只作为 descriptive feature，不作为 attribution weight。

### 4.2 LLM Weak Judgment Dataset

文件建议：

```text
artifacts/mechanism_training/<run_name>/llm_weak_judgments.jsonl
```

每行记录 Claude Code 或其他 LLM 对一个 evidence record 的候选判断：

```json
{
  "record_id": "rodinia_backprop.bpnn_adjust_weights_cuda",
  "annotator": {
    "type": "llm",
    "model": "claude-code",
    "prompt_version": "mechanism_weak_judgment_v0.1",
    "temperature": 0.0
  },
  "family_candidates": [
    {
      "family_id": "dense_compute",
      "support": 0.78,
      "evidence_sources": ["kernel_name", "dp_opcode_ratio", "compute_to_memory_ratio"],
      "claim_status": "weak_candidate_not_calibrated_probability"
    }
  ],
  "subtype_candidates": [
    {
      "family_subtype_id": "fp64_dp_pipeline_compute",
      "support": 0.82,
      "evidence_sources": ["dp_opcode_ratio", "validated_seed_similarity"],
      "claim_status": "weak_candidate_not_calibrated_probability"
    }
  ],
  "knob_candidates": [
    {
      "knob": "trace_opcode_latency_initiation_dp",
      "support": 0.86,
      "evidence_sources": ["subtype_to_knob_registry", "validated_seed_similarity"],
      "claim_status": "weak_candidate_not_calibrated_probability"
    }
  ],
  "abstain": false,
  "boundary_flags": [],
  "rationale": "Audit-only natural language explanation."
}
```

LLM judgment 规则：

1. `support` 是 evidence support，不是概率；
2. LLM 必须允许 abstain；
3. LLM 必须输出 evidence sources；
4. rationale 只用于 audit，默认不进入训练特征；
5. LLM 不能输出 `true_family`、`true_subtype`、`validated_contribution`；
6. 所有 LLM 候选必须经过 registry 校验。

### 4.3 Rule / Registry Weak Vote Dataset

文件建议：

```text
artifacts/mechanism_training/<run_name>/programmatic_votes.jsonl
```

来源包括：

```text
labeling functions
registry matches
validated feedback LFs
graph consistency checks
```

这些 votes 与 LLM judgments 并列进入 weak target builder。registry constraint 不应被当作普通 vote；它是合法性过滤器。

### 4.4 Validation Feedback Dataset

文件建议：

```text
artifacts/mechanism_training/<run_name>/validation_feedback_cases.jsonl
```

每个 case 描述一次可审计的 before / after 验证：

```json
{
  "case_id": "rodinia_backprop_dp_init_4",
  "record_ids": ["rodinia_backprop.bpnn_adjust_weights_cuda"],
  "control_record_ids": ["rodinia_backprop.bpnn_layerforward_CUDA"],
  "family_id": "dense_compute",
  "family_subtype_id": "fp64_dp_pipeline_compute",
  "knob": "trace_opcode_latency_initiation_dp",
  "baseline_value": "24,16",
  "modified_value": "24,4",
  "target_cycle_delta_ratio": -0.43572,
  "target_ipc_delta_ratio": 0.77216,
  "control_stable": true,
  "claim_status": "simulator_internal_closed_loop"
}
```

Validation feedback 是最高质量训练信号，但 evidence boundary 必须保留。例如 simulator-internal closed-loop 不能写成 real hardware speedup。

---

## 5. Weak Training Target Builder

### 5.1 输出文件

```text
artifacts/mechanism_training/<run_name>/weak_training_targets.jsonl
artifacts/mechanism_training/<run_name>/weak_training_dataset_card.md
```

每行一个训练样本：

```json
{
  "record_id": "rodinia_backprop.bpnn_adjust_weights_cuda",
  "sample_quality": "validated_seed",
  "sample_weight": 1.0,
  "family_target": {
    "target_type": "hard",
    "labels": {
      "dense_compute": 1.0
    }
  },
  "subtype_target": {
    "target_type": "hard",
    "labels": {
      "fp64_dp_pipeline_compute": 1.0
    }
  },
  "knob_target": {
    "target_type": "hard",
    "labels": {
      "trace_opcode_latency_initiation_dp": 1.0
    }
  },
  "abstain_target": {
    "should_abstain": false,
    "boundary_type": null
  },
  "target_provenance": {
    "sources": ["validation_feedback", "registry"],
    "claim_status": "simulator_internal_closed_loop"
  }
}
```

### 5.2 样本质量等级

训练样本按质量分层：

| Quality | Meaning | Default Weight |
| --- | --- | ---: |
| `validated_seed` | simulator closed-loop with target/control comparison | 1.00 |
| `human_reviewed_seed` | 人工审计确认的机制候选 | 0.85 |
| `graph_constrained_weak` | LLM / LF / registry / graph 一致支持 | 0.60 |
| `proxy_weak` | proxy evidence 支持但 non-claim-bearing | 0.35 |
| `llm_only_weak` | 只有 LLM 候选，无其他支持 | 0.20 |
| `boundary_abstain` | mixed / insufficient / conflict 样本 | 0.50 for abstain head only |

这些权重是第一版训练配置，不是研究结论。dataset card 必须记录每类样本数量和权重。

### 5.3 Target Fusion Policy

第一版采用 deterministic weighted fusion：

```text
validated feedback > human reviewed seed > graph constrained agreement > LF / registry-supported weak vote > LLM-only weak vote
```

规则：

1. 如果存在 validated feedback，则对应 family / subtype / knob 生成 hard target；
2. 如果只有 weak votes，则生成 soft target；
3. 如果候选冲突且无高质量来源压制冲突，则生成 boundary / abstain target；
4. 如果 subtype 与 family registry 不一致，丢弃该 subtype vote；
5. 如果 knob 无法从 subtype map 反查，丢弃该 knob vote；
6. 如果样本为 proxy-only，不能生成 validated target。

Soft target 可以用归一化支持分生成：

```text
soft_label(label_i) =
  weighted_support(label_i) / sum(weighted_support(all valid labels))
```

其中 `weighted_support` 来自 source weight 和 candidate support。输出字段必须命名为 `weak_training_target` 或 `soft_target`，不能命名为 `ground_truth`。

---

## 6. 模型结构

### 6.1 第一版模型

第一版推荐轻量结构化模型：

```text
Structured feature encoder
  -> shared hidden representation
  -> family ranking head
  -> subtype ranking head
  -> knob ranking head
  -> abstain / boundary head
  -> registry projection
```

输入特征：

```text
numeric mechanism features
missing-feature mask
evidence_status embedding
workload type embedding
cluster / anchor context features
LLM candidate support features
LF vote score features
graph consistency features
```

默认不把 LLM rationale 文本直接输入模型。原因是第一版数据少，文本解释容易让模型学习 annotator style，而不是 mechanism evidence。

### 6.2 输出 schema

模型推理输出：

```json
{
  "record_id": "rodinia_backprop.bpnn_adjust_weights_cuda",
  "ranked_family_candidates": [
    {
      "family_id": "dense_compute",
      "score": 2.13,
      "rank": 1,
      "claim_status": "model_candidate_not_validated"
    }
  ],
  "ranked_subtype_candidates": [
    {
      "family_subtype_id": "fp64_dp_pipeline_compute",
      "family_id": "dense_compute",
      "score": 2.47,
      "rank": 1,
      "claim_status": "model_candidate_not_validated"
    }
  ],
  "ranked_knob_candidates": [
    {
      "knob": "trace_opcode_latency_initiation_dp",
      "family_subtype_id": "fp64_dp_pipeline_compute",
      "score": 2.92,
      "rank": 1,
      "claim_status": "model_candidate_not_validated"
    }
  ],
  "abstain_score": 0.04,
  "boundary_flags": [],
  "registry_projection": {
    "invalid_candidates_removed": 0
  }
}
```

`score` 是模型排序分，不是 calibrated probability。只有经过单独校准后，才允许输出 calibrated probability 字段。

---

## 7. Loss 设计

第一版使用组合 loss：

```text
L_total =
  w_family * L_family
  + w_subtype * L_subtype
  + w_knob * L_knob
  + w_rank * L_pairwise_rank
  + w_abstain * L_abstain
```

建议初始权重：

```text
w_family = 1.0
w_subtype = 1.0
w_knob = 1.2
w_rank = 0.5
w_abstain = 0.8
```

目标类型对应 loss：

| Target Type | Loss |
| --- | --- |
| hard validated target | cross entropy |
| soft weak target | KL divergence / soft cross entropy |
| candidate preference pair | pairwise margin ranking |
| abstain / boundary | binary cross entropy |

每个样本 loss 还要乘以 `sample_weight`。`llm_only_weak` 样本不得主导训练；训练报告必须输出不同质量层的 loss contribution。

---

## 8. 训练和切分策略

### 8.1 数据切分

必须支持 split-by-workload：

```text
train workloads
validation workloads
test workloads
```

禁止只按 record 随机切分。原因是同一 workload 中 kernel name、trace context、cluster 信息高度相关，随机切分会导致泄漏。

### 8.2 训练顺序

推荐 curriculum：

```text
Stage 1: validated_seed + human_reviewed_seed
Stage 2: add graph_constrained_weak
Stage 3: add proxy_weak with lower weight
Stage 4: add llm_only_weak only for regularization and abstain contrast
```

### 8.3 第一版 fixture

第一版训练 dataset 至少包含：

```text
Rodinia DP validated positive:
  dense_compute / fp64_dp_pipeline_compute / trace_opcode_latency_initiation_dp

Rodinia DP control:
  control kernel stable under DP knob change

ResNet ref_layer3 proxy samples:
  conv_compute, layout_transform, batchnorm boundary, activation/residual mixed split

Boundary / abstain samples:
  helper_index_precompute, mixed cluster, insufficient PKA measured features
```

---

## 9. 评估指标

第一版评估不只看 accuracy。

必须输出：

```text
family top-1 accuracy
family top-3 recall
subtype top-k recall
knob top-k recall
mean reciprocal rank
registry legality rate
abstain precision / recall
boundary false-positive rate
calibration error if calibrated probability is emitted
```

下游指标：

```text
validation planner top-k hit rate
validated seed recovery rate
negative/control stability recognition
```

如果有足够 closed-loop cases，再报告：

```text
validation lift over time-only baseline
validation lift over rule-only baseline
```

第一版不要求报告 speedup prediction error，因为模型不预测 speedup。

---

## 10. LLM 生成训练数据协议

### 10.1 Prompt 输入

每个 LLM annotation prompt 应包含：

```text
kernel identity and workload context
structured features
missing feature list
registry label space
subtype-to-knob legal map
known validation feedback if available
required output JSON schema
explicit instruction that support is not probability
explicit permission to abstain
```

### 10.2 Prompt 输出约束

LLM 必须输出机器可解析 JSON。解析失败的样本不进入训练集，只进入 audit report。

必须包含：

```text
family_candidates
subtype_candidates
knob_candidates
abstain
boundary_flags
evidence_sources
claim_status
```

禁止包含：

```text
ground_truth
true_label
validated_contribution
real_hardware_speedup
```

### 10.3 Multi-pass Consistency

如果成本允许，同一个 record 可以运行多次 annotation：

```text
same prompt, deterministic model setting
or prompt variants with identical output schema
```

一致候选可以提升到 `graph_constrained_weak` 的候选来源之一。不一致候选应增加 conflict / boundary 信号，而不是强行平均成高置信标签。

---

## 11. Reviewer 风险边界

论文或报告中不能声称：

```text
LLM produced ground-truth mechanism labels
weak support scores are calibrated probabilities
instruction ratios measure attribution contribution
proxy cases validate simulator knobs
model output is a prescription without validation
```

可以声称：

```text
LLM is used as one weak candidate generator.
Registry constraints remove illegal mechanism / knob combinations.
Sparse closed-loop validation provides high-quality seed labels.
The trained model ranks mechanism candidates for validation planning.
Final claim-bearing conclusions require simulator before/after validation.
```

---

## 12. 与 B-Line / C-Line 的接口

### 12.1 输出给 B-Line Graph Reasoner

```text
model_ranked_candidates.json
model_training_audit.md
training_dataset_card.md
```

B-Line 使用模型输出作为 graph evidence edge：

```text
kernel_to_family_model_score
kernel_to_subtype_model_score
subtype_to_knob_model_score
kernel_to_abstain_model_score
```

这些 edge 的 `claim_status` 必须是：

```text
model_candidate_not_validated
```

### 12.2 输出给 C-Line Validation Planner

C-Line 不直接信任模型 top-1。它接收经过 graph reasoner 约束后的：

```text
accepted candidate groups
knob candidates
validation priority scores
blocked / boundary report
```

模型分数只影响 validation ordering，不直接构成 validated contribution。

---

## 13. 验收标准

第一版完成时应满足：

1. 生成 `raw_evidence_records.jsonl`、`llm_weak_judgments.jsonl`、`weak_training_targets.jsonl` 和 `weak_training_dataset_card.md`；
2. 每个 target 都有 provenance；
3. 每个 LLM 输出都标明 `prompt_version` 和 `claim_status`；
4. dataset card 统计每个 sample quality 层级的数量；
5. registry-invalid family / subtype / knob 组合不会进入训练 target；
6. proxy 样本不会生成 validated target；
7. boundary 样本可以训练 abstain head；
8. 训练 split 使用 split-by-workload；
9. 评估报告包含 top-k、MRR、registry legality、abstain 指标；
10. 模型输出不包含 calibrated probability，除非有单独校准步骤。

---

## 14. 下一步

如果该设计被确认，下一步应写 implementation plan，优先实现：

```text
1. dataset schema and fixtures
2. Claude Code / LLM annotation runner contract
3. weak target builder
4. baseline ranking model
5. evaluation report
6. export adapter to B-Line graph reasoner
```

第一版可以先不接入完整 GNN。训练目标稳定后，再把模型输出作为 B-Line graph 的一类 evidence edge，并逐步扩展到 learned graph network。
