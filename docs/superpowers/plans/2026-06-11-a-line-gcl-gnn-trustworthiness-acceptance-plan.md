# A 线 GCL GNN Trustworthiness Acceptance Implementation Plan

## Goal Description

实现一个可重复、可审计的 GCL GNN 可信性验收层，用于消费现有 Gate5、Gate6、Gate7 和 ResNet-50 full-trace artifacts，并输出 GNN acceptance manifest、summary 和 markdown report。

该计划不重新设计 RGCN，不重新采集 trace，不重新构造 graph，也不把当前结果升级为正确性通过。目标是把 `docs/superpowers/specs/2026-06-11-a-line-gcl-gnn-trustworthiness-acceptance-design.md` 中定义的验收标准落地为代码和测试，使系统能够明确区分：

```text
结构真实、链路可运行
embedding 有弱信号
训练不足
baseline / stability / semantic / downstream 证据缺失
```

当前 ResNet-50 full trace 的预期输出仍应保持：

```text
gnn_acceptance_status =
  weak_acceptance_structure_valid_but_correctness_unproven

claim_status =
  quantified_no_correctness_claim
```

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: GNN acceptance evaluator validates required artifact provenance
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_acceptance_evaluator_accepts_real_resnet50_full_trace_artifacts`
    - 输入包含 `resnet50_full_trace_reproduction_manifest.json`，且字段满足 `run_scope = real_resnet50_full_trace`、`formal_full_trace_run = true`、`input_kernel_invocation_count > 0`、`input_cta_record_count > 0` 时，evaluator 必须生成 `input_provenance = PASS`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_acceptance_evaluator_rejects_synthetic_or_debug_input`
    - 当 `formal_full_trace_run = false`、`run_scope` 不是 `real_resnet50_full_trace`、或缺少 kernel / CTA count 时，formal acceptance 必须拒绝，不能输出 `PASS`。

- AC-2: RGCN structure acceptance verifies architecture and relation-aware contract
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_rgcn_structure_acceptance_passes_for_gate5_contract`
    - `rgcn_training_run_manifest.json` 记录 `layers = 3`、`input_dim = 64`、`hidden_dim = 128`、`kernel_embedding_dim = 256`、`relation_count = 3`、`readout_hierarchy = node_to_warp_to_cta_to_selected_sm_to_kernel` 时，必须输出 `rgcn_structure = PASS`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_rgcn_structure_acceptance_rejects_pooling_only_or_projection_output_selector`
    - 如果缺少 relation schema、`relation_count < 3`、readout hierarchy 错误、或 manifest 标记 selector 使用 projection head output，必须拒绝结构通过。

- AC-3: Training adequacy is conservative and fails smoke-level training
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_training_adequacy_fails_for_single_step_smoke_training`
    - 当前 artifact 中 `train_graph_count = 4` 且 `optimizer_step_count = 1` 时，必须输出 `training_adequacy = FAIL`，并把训练不足加入 `blocking_gaps`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_training_adequacy_cannot_pass_without_loss_curve_and_multi_step_training`
    - 当缺少 `loss_curve`、`epoch_count`、`positive_pair_count` 或 `negative_pair_count` 时，即使 `training_status = formal_gate5_complete`，也不能输出 `training_adequacy = PASS`。

- AC-4: Embedding geometry is scored as weak evidence unless baseline evidence exists
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_embedding_geometry_weak_pass_with_current_gate7_metrics`
    - 当前 Gate7 指标 `silhouette = 0.48186617`、`inter_intra_ratio = 2.01633922`、`davies_bouldin = 0.78974237` 可计算时，必须输出 `embedding_geometry_signal = WEAK_PASS`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_embedding_geometry_cannot_be_final_pass_without_baseline_ablation`
    - 当没有 baseline ablation report 时，embedding geometry 不得输出最终 `PASS`，也不得升级 `claim_status`。

- AC-5: Baseline ablation report supports no-claim when missing and graph-signal rejection when weak
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_missing_baseline_ablation_is_not_available`
    - 缺少 baseline report 时，必须输出 `baseline_ablation = NOT_AVAILABLE`，并保留 `claim_status = quantified_no_correctness_claim`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_full_rgcn_must_not_pass_when_no_edge_baseline_matches`
    - 如果 baseline report 显示 full RGCN 与 no-edge baseline 接近，或者不优于 random / opcode histogram baseline，必须输出 `rejected_no_graph_signal` 或保持不通过，不能输出 `accepted`。

- AC-6: Multi-seed stability requires hard seed-count floors
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_single_run_stability_is_not_available`
    - 当前 `stability_status = single_run_not_evaluated` 时，必须输出 `multi_seed_stability = NOT_AVAILABLE`。
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_multi_seed_stability_accepts_minimum_seed_counts`
    - 当 stability report 记录 `training_seed_count >= 3`、`kmeans_seed_count >= 5`，且包含 `k_stability`、`assignment_stability_ari`、`assignment_stability_nmi`、`centroid_drift`、`representative_stability_rate` 时，才允许进入 stability 支持状态。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_multi_seed_stability_rejects_under_seeded_reports`
    - `training_seed_count < 3` 或 `kmeans_seed_count < 5` 时，不能输出 `cluster_stability_supported`。

- AC-7: Semantic cluster correctness handles coarse labels conservatively
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_coarse_resnet50_label_keeps_semantic_correctness_unproven`
    - 当 family evidence 只有粗粒度 `resnet50_real_trace` 覆盖，且 `ari = 0.0`、`nmi = 0.0`、`completeness = 0.0` 时，必须输出 `semantic_cluster_correctness = UNPROVEN`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_purity_alone_cannot_upgrade_semantic_claim`
    - 即使 `cluster_purity = 1.0` 或 `weighted_purity = 1.0`，如果 label 粒度不足，不能输出 `semantic_cluster_supported`。

- AC-8: Representative downstream usefulness requires measured or simulator metrics
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_missing_metric_rows_keep_downstream_usefulness_not_available`
    - 当 Gate7 metric report 记录 `metric_claim_status = unavailable` 或 `status = not_provided` 时，必须输出 `downstream_representative_usefulness = NOT_AVAILABLE`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_downstream_claim_requires_error_metrics`
    - 缺少 `cluster_weighted_mape`、`global_weighted_mape`、`global_p95_relative_error` 或 metric source provenance 时，不能输出 `representative_downstream_supported`。

- AC-9: Claim-status upgrade policy is monotonic and conservative
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_current_resnet50_run_keeps_quantified_no_correctness_claim`
    - 当前 artifact 组合必须输出 `gnn_acceptance_status = weak_acceptance_structure_valid_but_correctness_unproven` 和 `claim_status = quantified_no_correctness_claim`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_claim_status_cannot_upgrade_with_any_blocking_gap`
    - 只要 training adequacy、baseline ablation、multi-seed stability、semantic alignment 或 downstream usefulness 任一项缺失或失败，就不能输出 `gnn_trustworthiness_accepted`。

- AC-10: Acceptance artifacts are written with stable hashes and replayable references
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_acceptance_artifacts_written_with_hashes`
    - runner 必须写出 `gnn_acceptance_manifest.json`、`gnn_acceptance_summary.json`、`gnn_acceptance_report.md`，并记录 `input_artifact_hashes`、`acceptance_items`、`blocking_gaps`、`recommended_next_gates`、`report_hash`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_phase_b/test_gnn_acceptance.py::test_acceptance_manifest_rejects_missing_hash_or_report_mismatch`
    - manifest 缺少 source artifact hash、report hash 与 markdown 内容不匹配、或 summary 与 manifest 状态不一致时，validator 必须拒绝。

- AC-11: Full trace runner can append Gate10 report without mutating Gate1-Gate9 artifacts
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_full_trace_reproduction_runner.py::test_full_trace_runner_can_append_gnn_acceptance_report`
    - 在 existing artifact root 上运行 acceptance stage 时，只能新增或更新 Gate10 acceptance artifacts，不能重写 `kernel_embedding_table.json`、`selector_artifacts.json`、`gate7_cluster_correctness_manifest.json`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_full_trace_reproduction_runner.py::test_acceptance_stage_rejects_missing_gate5_gate6_gate7_inputs`
    - 缺少 Gate5 training manifest、Gate6 selector artifacts 或 Gate7 correctness manifest 时，runner 必须输出 blocker，不得生成 formal acceptance。

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

完整实现 Gate10 风格的 GNN trustworthiness acceptance stage：

- 新增独立 evaluator 模块，消费 Gate5/Gate6/Gate7/ResNet-50 manifest；
- 生成 acceptance manifest、summary、markdown report；
- 实现 baseline ablation report 的 schema 和 conservative validator；
- 实现 multi-seed stability report 的 schema 和 hard seed-count validation；
- 支持 full trace runner 在 Gate9 后追加 Gate10 report；
- 测试覆盖正向、负向、hash replay、claim-status downgrade / upgrade policy。

### Lower Bound (Minimum Acceptable Scope)

最低可接受实现必须包括：

- 独立 acceptance evaluator；
- 当前 ResNet-50 artifact 的确定性验收输出；
- `PASS / WEAK_PASS / FAIL / NOT_AVAILABLE / UNPROVEN` item 状态；
- `gnn_acceptance_status` 与 `claim_status` 保守决策；
- `gnn_acceptance_manifest.json`、`gnn_acceptance_summary.json`、`gnn_acceptance_report.md` 三个输出；
- 覆盖 AC-1 到 AC-11 的 pytest。

如果没有 baseline 或 multi-seed 真实实验，第一版可以只实现 schema、validator 和缺失证据下的保守判断；不能伪造实验结果。

### Allowed Choices

- Can use:
  - `experiments/gcl_phase_b` 下新增 `trustworthiness.py` 或同等独立模块；
  - 复用 `experiments/gcl_phase_b/utils.py::stable_hash`；
  - 复用现有 Gate7 correctness artifact 和 selector artifact；
  - 使用 JSON manifest 加 Markdown report 的输出组合；
  - 在 `scripts/run_resnet50_full_trace_gcl.py` 中追加可选 acceptance stage。
- Cannot use:
  - 不能重新采集 trace；
  - 不能重新构造 canonical graph 或 tensor schema；
  - 不能把 projection head output 用作 selector embedding；
  - 不能用 synthetic fixture 冒充 real ResNet-50 full trace formal result；
  - 不能在缺少 baseline / stability / downstream metric 时升级 claim；
  - 不能修改 Gate5、Gate6、Gate7 已有 artifact 的语义来适配 Gate10。

## Feasibility Hints and Suggestions

### Conceptual Approach

一个可行实现路径：

```text
load artifact root
  -> validate required files exist
  -> validate provenance and hashes
  -> evaluate input / graph / RGCN / training / embedding / selector / stability / semantic / downstream items
  -> combine item statuses into gnn_acceptance_status
  -> derive conservative claim_status
  -> write manifest + summary + markdown report
```

状态合并可以采用保守优先级：

```text
missing required artifact
  -> not_evaluable_missing_artifacts

training_adequacy = FAIL
  + structure/input/export pass
  -> weak_acceptance_structure_valid_but_correctness_unproven

baseline shows no graph signal
  -> rejected_no_graph_signal

multi-seed unstable
  -> rejected_unstable_embedding

downstream metric provided but representative error too high
  -> rejected_downstream_unproven

all required evidence passes
  -> accepted
```

### Relevant References

- `docs/superpowers/specs/2026-06-11-a-line-gcl-gnn-trustworthiness-acceptance-design.md` - 设计来源。
- `docs/reports/gcl-resnet50-gnn-acceptance-report-2026-06-11.md` - 当前 artifact 的人工验收结论。
- `experiments/gcl_phase_b/correctness.py` - Gate7 correctness report 和 metric 计算逻辑。
- `experiments/gcl_phase_b/selector.py` - Gate6 selector artifacts 和 silhouette-K 输出结构。
- `experiments/gcl_phase_b/embedding_export.py` - Gate5 embedding table validator。
- `scripts/run_resnet50_full_trace_gcl.py` - ResNet-50 full trace runner 和 artifact root 管理。
- `tests/gcl_phase_b/` - Phase B gate 单元测试风格。
- `tests/gcl_resnet50/test_full_trace_reproduction_runner.py` - full trace runner 测试风格。

## Dependencies and Sequence

### Milestones

1. Acceptance data model and validators
   - 定义 item status、overall status、claim status、blocking gaps。
   - 校验 required artifacts 和 provenance。
   - 校验 RGCN architecture、readout hierarchy、training manifest。

2. Evidence evaluators
   - 实现 input provenance evaluator。
   - 实现 RGCN structure evaluator。
   - 实现 training adequacy evaluator。
   - 实现 embedding geometry evaluator。
   - 实现 baseline ablation schema / missing-evidence handling。
   - 实现 multi-seed stability schema / seed-count validation。
   - 实现 semantic cluster alignment evaluator。
   - 实现 downstream representative usefulness evaluator。

3. Status combiner and claim policy
   - 将 item statuses 合并为 `gnn_acceptance_status`。
   - 实现保守 claim upgrade / downgrade。
   - 确保任何 blocking gap 都阻止 `gnn_trustworthiness_accepted`。

4. Artifact writer
   - 写出 `gnn_acceptance_manifest.json`。
   - 写出 `gnn_acceptance_summary.json`。
   - 写出 `gnn_acceptance_report.md`。
   - 计算并校验 stable hash。

5. Runner integration
   - 在 ResNet-50 full trace runner 增加 acceptance stage。
   - 支持对 existing artifact root 追加 Gate10 report。
   - 缺少输入时输出 blocker，不重写 Gate1-Gate9 artifacts。

6. Regression tests and replay verification
   - 添加 `tests/gcl_phase_b/test_gnn_acceptance.py`。
   - 扩展 `tests/gcl_resnet50/test_full_trace_reproduction_runner.py`。
   - 运行针对性 pytest。

## Task Breakdown

Each task must include exactly one routing tag:

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 新增 GNN acceptance 状态常量、artifact schema 和 required-file validator | AC-1, AC-10 | coding | - |
| task2 | 实现 provenance、RGCN structure、training adequacy 三个 evaluator | AC-1, AC-2, AC-3 | coding | task1 |
| task3 | 实现 embedding geometry、baseline ablation、multi-seed stability evaluator | AC-4, AC-5, AC-6 | coding | task1 |
| task4 | 实现 semantic cluster correctness 和 downstream representative usefulness evaluator | AC-7, AC-8 | coding | task1 |
| task5 | 实现 overall status combiner 和 conservative claim-status policy | AC-9 | coding | task2, task3, task4 |
| task6 | 实现 manifest / summary / markdown report writer 和 validator | AC-10 | coding | task5 |
| task7 | 将 acceptance stage 接入 ResNet-50 full trace runner，保证只追加 Gate10 artifacts | AC-11 | coding | task6 |
| task8 | 编写 gcl_phase_b 单元测试覆盖 AC-1 到 AC-10 | AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8, AC-9, AC-10 | coding | task6 |
| task9 | 编写 full trace runner 集成测试覆盖 AC-11 | AC-11 | coding | task7 |
| task10 | 审核生成 artifacts 与人工报告结论是否一致 | AC-9, AC-10, AC-11 | analyze | task8, task9 |

## Claude-Codex Deliberation

### Agreements

- GNN acceptance 必须独立于 Gate5/Gate6/Gate7，不能通过修改旧 gate 语义来制造可信性。
- 当前 ResNet-50 run 只能得到弱验收，不能升级正确性声明。
- 缺少 baseline、multi-seed、semantic 和 downstream metric 时，claim status 必须保持保守。
- 第一版可以实现 baseline / stability schema 和 validator，而不伪造真实多次实验。

### Resolved Disagreements

- Topic: 是否把当前 embedding geometry 的正向指标视为通过。
  - Resolution: 只允许 `WEAK_PASS`，因为没有 baseline ablation 和 multi-seed stability。
- Topic: 是否在 Gate7 内扩展可信性判断。
  - Resolution: 新增独立 GNN acceptance stage，避免 Gate7 correctness 文件继续膨胀。

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- DEC-1: 是否在第一版就运行真实 multi-seed RGCN training。
  - Claude Position: 第一版先实现 schema、validator 和缺失证据保守判断，真实多 seed 训练作为后续实验。
  - Codex Position: 同意先做 evaluator 和 artifact 合约，否则很难判断后续实验是否合格。
  - Tradeoff Summary: 先实现 evaluator 能快速建立验收边界；直接跑 multi-seed 成本更高，且当前训练配置还需要调整。
  - Decision Status: `PENDING`

- DEC-2: 是否把 Gate10 artifact 纳入默认 full trace runner。
  - Claude Position: 可以默认追加 report-only Gate10，不改变 Gate1-Gate9 artifact。
  - Codex Position: 同意，但必须支持缺失输入时输出 blocker。
  - Tradeoff Summary: 默认追加便于复现报告统一；可选开关更保守但容易漏跑。
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.
- Artifact status strings must match spec exactly，避免出现同义但不同拼写的状态。
- Markdown report 可以使用中文，但 JSON manifest 字段名和值应保持英文稳定标识。
