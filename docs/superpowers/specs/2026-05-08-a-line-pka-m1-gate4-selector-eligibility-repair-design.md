# A 线 PKA-M1 Gate 4 Selector Eligibility + Backward Repair Design Spec

日期：2026-05-08

## 1. 背景

Gate 1 负责确认 L1 P0 workload 能否解析为真实可运行命令。

Gate 2 负责对 Gate 1 resolved commands 执行 exact-metric NCU capture，并输出 capture artifacts。

Gate 3 负责从 Gate 2 eligible artifacts 中抽取 12 维 measured PKA feature，生成 feature table 或 acquisition gap。

Gate 4 是进入 selector 前的总检查站。它不跑 PCA，不跑 k-means，不生成 anchor。它只决定：

```text
当前 measured feature table 是否可以进入 Gate 5 selector？
如果不能，应该回到哪个 gate 优先修？
如果能，剩余 gap 应该如何继续修？
```

## 2. 目标

Gate 4 的目标是：

- 统计 Gate 3 输出的 measured rows。
- 检查 selector 输入是否满足 formal M1 要求。
- 决定是否允许进入 Gate 5。
- 对所有未 measured P0 entries 输出 backward repair report。
- 在 `measured_rows < 3` 时强制阻止 Gate 5，并给出逐 entry 修复路径。

输入：

- `artifacts/a_line/l1/m1_workload_resolution_l1.json`
- `artifacts/a_line/l1/m1_workload_resolution_gap_l1.json`
- `artifacts/a_line/l1/m1_ncu_capture_attempts_l1.json`
- `artifacts/a_line/l1/m1_ncu_capture_gap_l1.json`
- `artifacts/a_line/l1/pka_feature_table_l1.json`
- `artifacts/a_line/l1/pka_acquisition_gap_l1.json`
- `artifacts/a_line/l1/pka_feature_audit_l1.json`
- `artifacts/a_line/l1/pka_join_audit_l1.json`

输出：

- `artifacts/a_line/l1/m1_selector_eligibility_l1.json`
- `artifacts/a_line/l1/m1_backward_repair_report_l1.json`
- `artifacts/a_line/l1/m1_backward_repair_report_l1.md`

## 3. 非目标

Gate 4 不做：

- workload build。
- smoke run。
- NCU capture。
- NCU CSV parsing。
- 12D feature extraction。
- PCA。
- k-means。
- anchor selection。
- compression evaluation。
- feature 补齐。

Gate 4 不能通过修改 feature table 来让 selector 通过。

## 4. 核心定位

Gate 4 是：

```text
selector 前总检查站 + backward repair 仲裁层
```

它有两个职责：

1. 判断 Gate 5 是否可以运行。
2. 如果仍有 gap，给出每个 P0 entry 的 earliest failed gate 和 repair action。

## 5. Selector Eligibility States

Gate 4 必须输出一个 selector eligibility state。

允许状态：

- `selector_ready`
- `selector_ready_with_remaining_gaps`
- `selector_blocked_insufficient_measured_records`
- `selector_blocked_invalid_feature_table`
- `selector_blocked_mixed_timing_unit`

### 5.1 `selector_ready`

条件：

- `measured_rows >= 3`
- 所有 L1 P0 entries 都 measured。
- feature table preflight 通过。
- timing unit 检查通过。

处理：

- 允许进入 Gate 5。
- backward repair report 可以为空，但仍必须输出 summary。

### 5.2 `selector_ready_with_remaining_gaps`

条件：

- `measured_rows >= 3`
- feature table preflight 通过。
- timing unit 检查通过。
- 仍有 L1 P0 entries 处于 gap。

处理：

- 允许进入 Gate 5。
- 必须输出 remaining gap repair report。
- remaining gaps 不参与 Gate 5 selector。

### 5.3 `selector_blocked_insufficient_measured_records`

条件：

```text
measured_rows < 3
```

处理：

- 禁止进入 Gate 5。
- 必须输出 backward repair report。
- 不得宣布 M1 selector 完成。

### 5.4 `selector_blocked_invalid_feature_table`

条件：

- feature table 缺失。
- feature table schema 不合法。
- measured row 不完整。
- feature status 不是 `measured`。
- feature mode 不是 `pka_m1_measured`。
- selector input projection 包含 forbidden metadata。

处理：

- 禁止进入 Gate 5。
- 必须输出 blocking reason 和 repair action。

### 5.5 `selector_blocked_mixed_timing_unit`

条件：

- measured rows 中存在混合 timing unit。

例如同时存在：

- `duration_ns`
- `elapsed_cycles`

处理：

- 禁止进入 Gate 5。
- 必须输出 timing conflict report。

## 6. Feature Table Preflight

Gate 4 必须对 `pka_feature_table_l1.json` 做 selector input preflight。

### 6.1 行数检查

统计：

- `measured_rows`
- `gap_rows`
- `total_p0_entries`

`measured_rows >= 3` 是进入 Gate 5 的必要条件，但不是充分条件。

### 6.2 12D 完整性检查

每条 measured row 必须包含完整 12D feature：

- `coalesced_global_loads`
- `coalesced_global_stores`
- `coalesced_local_loads`
- `thread_global_loads`
- `thread_global_stores`
- `thread_local_loads`
- `thread_shared_loads`
- `thread_shared_stores`
- `thread_global_atomics`
- `num_instructions`
- `divergence_efficiency`
- `num_thread_blocks`

每个 feature 必须：

- `status == measured`
- 有 numeric value。
- 有 `canonical_metric`。
- 有 `actual_source_metric`。
- 有 provenance。

### 6.3 feature_mode 检查

每条 measured row 必须满足：

```text
feature_mode == pka_m1_measured
```

禁止 M0 fixture row 进入 M1 selector。

### 6.4 Selector input projection

Gate 4 应生成或验证 selector input projection。

Selector 只允许读取：

- `record_id`
- `kernel_invocation_id`
- `features`
- `feature_mode`

Selector input projection 不得包含：

- `kernel_name`
- `source_path`
- `expected_behavior_axis`
- `family`
- `regime`
- `shape_hint`
- `trace_order`
- B-line semantic metadata

如果 projection 中出现 forbidden field，Gate 4 必须阻止 Gate 5。

## 7. Timing Unit Check

Gate 4 必须检查 measured rows 的 timing unit。

允许情况：

### 7.1 全部无 timing

如果所有 measured rows 都没有 timing：

- 允许进入 Gate 5。
- Gate 5 evaluation 必须使用 `member_count_fallback`。
- Gate 4 输出：

```text
weight_mode = member_count_fallback
```

### 7.2 全部同一种 timing unit

如果所有 measured rows 都使用同一种 timing unit：

- `duration_ns`
- 或 `elapsed_cycles`

允许进入 Gate 5。

Gate 4 输出：

```text
weight_mode = timing_weight
timing_unit = duration_ns | elapsed_cycles
```

### 7.3 混合 timing unit

如果 measured rows 混用 timing unit：

- 禁止进入 Gate 5。
- 输出 `selector_blocked_mixed_timing_unit`。
- 输出 timing conflict report。

原因：weighted coverage 不能混合不可比较单位。

## 8. Backward Repair Report

Gate 4 必须输出 backward repair report。

即使 selector 可以运行，只要存在 remaining gaps，也必须报告后续修复方向。

输出：

- `artifacts/a_line/l1/m1_backward_repair_report_l1.json`
- `artifacts/a_line/l1/m1_backward_repair_report_l1.md`

### 8.1 每个 P0 entry 必须有状态

每个 L1 P0 manifest entry 必须在 repair report 中出现。

每个 entry 状态为：

- `measured`
- `blocked`
- `not_attempted`

### 8.2 earliest_failed_gate

每个未 measured entry 必须有一个 `earliest_failed_gate`。

优先级固定：

```text
Gate1 > Gate2 > Gate3 > Gate4
```

规则：

- 如果 Gate 1 unresolved，`earliest_failed_gate = Gate1`。
- 如果 Gate 1 resolved 但 Gate 2 没有 usable capture，`earliest_failed_gate = Gate2`。
- 如果 Gate 2 gate3_eligible 但 Gate 3 没有 measured 12D，`earliest_failed_gate = Gate3`。
- 如果 Gate 1-3 都完成但全局 selector 条件失败，`earliest_failed_gate = Gate4`。

禁止跳过更早 gate 去修后面的 gate。

### 8.3 Repair action type

每条 blocked entry 必须有：

- `repair_action_type`
- `suggested_repair_action`
- `allowed_to_auto_run`

允许的 `repair_action_type`：

- `executable_command`
- `manual_action`
- `code_fix_required`
- `environment_action`

### 8.4 executable command 约束

如果 `repair_action_type == executable_command`，必须包含：

- `command`
- `working_directory`
- `source_of_command`
- `allowed_to_auto_run`

`source_of_command` 只能来自：

- Gate 1 workload registry。
- Gate 2 capture command artifact。
- 已存在 repo script。
- metric resolution artifact。

禁止 RLCR / agent 临场编造 repair command。

### 8.5 environment action

如果是环境问题，例如：

- `permission_blocked`
- `environment_blocked`
- GPU 不可见
- NCU 不可用

则必须输出：

- `repair_action_type: environment_action`
- `allowed_to_auto_run: false`
- 清晰的人工操作说明。

## 9. M1SelectorEligibilityReport

`m1_selector_eligibility_l1.json` 至少包含：

- `artifact_name`
- `generated_at`
- `selector_eligibility_state`
- `gate5_allowed`
- `measured_rows`
- `gap_rows`
- `total_p0_entries`
- `feature_table_path`
- `acquisition_gap_path`
- `feature_table_preflight`
- `timing_check`
- `selector_input_projection_path`
- `backward_repair_report_path`
- `blocking_reasons`

`feature_table_preflight` 至少包含：

- `status`
- `checked_rows`
- `complete_12d_rows`
- `invalid_rows`
- `forbidden_field_violations`
- `feature_mode_violations`

`timing_check` 至少包含：

- `status`
- `weight_mode`
- `timing_unit`
- `conflicting_units`
- `conflict_records`

## 10. M1BackwardRepairReport

JSON report 至少包含：

- `artifact_name`
- `generated_at`
- `selector_eligibility_state`
- `gate5_allowed`
- `entries`
- `summary`

每条 entry 至少包含：

- `manifest_entry_id`
- `kernel_or_case`
- `entry_status`
- `gate1_status`
- `gate2_status`
- `gate3_status`
- `earliest_failed_gate`
- `blocking_reason`
- `repair_action_type`
- `suggested_repair_action`
- `executable_command`
- `allowed_to_auto_run`

summary 至少包含：

- `total_p0_entries`
- `measured_entries`
- `blocked_entries`
- `not_attempted_entries`
- `gate1_blocked_count`
- `gate2_blocked_count`
- `gate3_blocked_count`
- `gate4_blocked_count`
- `auto_runnable_repairs`
- `manual_repairs`
- `environment_actions`
- `code_fixes_required`

Markdown report 必须面向人工 review，按 gate 分组：

- Gate 1 blockers
- Gate 2 blockers
- Gate 3 blockers
- Gate 4 blockers
- Selector readiness summary

## 11. Stop-hook / Completion Rules

Gate 4 是强 stop-hook gate。

### 11.1 measured_rows >= 3

如果：

```text
measured_rows >= 3
```

且 preflight / timing checks 通过：

- 可以进入 Gate 5。
- 但必须已经输出 eligibility report。
- 如果存在 remaining gaps，必须已经输出 backward repair report。

### 11.2 measured_rows < 3

如果：

```text
measured_rows < 3
```

则：

- 不得进入 Gate 5。
- 不得宣布 M1 selector 完成。
- 必须输出 backward repair report。
- backward repair report 必须包含 per-entry earliest failed gate。

如果 report 缺失，RLCR 必须 STOP/FAIL。

如果 report 只有空泛建议，没有 per-entry earliest gate，RLCR 必须 STOP/FAIL。

### 11.3 invalid feature table

如果 feature table preflight 失败：

- 不得进入 Gate 5。
- 必须输出 invalid rows 和修复动作。

### 11.4 mixed timing unit

如果 timing unit 混用：

- 不得进入 Gate 5。
- 必须输出 conflict records。

## 12. Determinism

Gate 4 output 必须 deterministic：

- P0 entries 按 manifest order 输出。
- blocking reasons 使用稳定排序。
- JSON 使用 stable key order。
- selector input projection 使用 stable row order。

## 13. 测试要求

Gate 4 测试至少覆盖：

- measured rows < 3 -> `selector_blocked_insufficient_measured_records`。
- measured rows >= 3 且无 gap -> `selector_ready`。
- measured rows >= 3 且有 gap -> `selector_ready_with_remaining_gaps`。
- incomplete 12D row -> `selector_blocked_invalid_feature_table`。
- feature status 非 measured -> block。
- feature mode 是 `pka_m0_algorithmic_fixture` -> block。
- selector input projection 含 forbidden metadata -> block。
- 全无 timing -> allowed with `member_count_fallback`。
- 单一 timing unit -> allowed with `timing_weight`。
- mixed timing unit -> `selector_blocked_mixed_timing_unit`。
- Gate 1 blocker 的 earliest failed gate 是 Gate1。
- Gate 2 blocker 的 earliest failed gate 是 Gate2。
- Gate 3 blocker 的 earliest failed gate 是 Gate3。
- repair executable command 只能来自允许来源。
- measured rows < 3 且缺 backward repair report -> stop/fail。

## 14. 与 Gate 5 的接口

Gate 5 只能在以下条件满足时运行：

```text
gate5_allowed == true
selector_eligibility_state in [
  "selector_ready",
  "selector_ready_with_remaining_gaps"
]
```

Gate 5 输入必须来自 Gate 4 selector input projection，而不是直接读取完整 feature table metadata。

## 15. 简短结论

Gate 4 的核心是：

```text
feature table + gap artifacts
  -> selector eligibility decision
  -> selector input preflight
  -> timing unit check
  -> backward repair report
```

它保证 M1 不会在 measured rows 不足、feature table 不干净、timing 不可解释时强行进入 PCA / k-means。
