# A 线 PKA-M1 Gate 3 Measured Feature Extractor Design Spec

日期：2026-05-08

## 1. 背景

Gate 1 将 L1 P0 manifest entries 解析为真实可运行命令。

Gate 2 将 Gate 1 resolved commands 转成 exact-metric NCU capture jobs，并输出 capture CSV、selected metrics、environment manifest、stderr、exit code 等 provenance。

Gate 3 的职责是消费 Gate 2 中 `gate3_eligible == true` 的 capture artifacts，将它们解析成正式 12 维 PKA measured feature records，或者将失败对象明确写入 acquisition gap。

Gate 3 不负责 workload resolution，不负责 NCU capture，也不负责判断 measured rows 是否足够运行 PCA / k-means。`measured_rows >= 3` 是 Gate 4 的职责。

## 2. 目标

Gate 3 的目标是：

```text
Gate 2 eligible capture jobs
  -> parse NCU CSV
  -> join CSV invocations to consuming manifest entries
  -> extract 12D measured PKA features
  -> emit pka_feature_table_l1.json or pka_acquisition_gap_l1.json
  -> emit audit / join audit artifacts
```

输入：

- `artifacts/a_line/l1/m1_ncu_capture_attempts_l1.json`
- `experiments/baseline_diagnosis/results/m1_ncu/<capture_job_id>/capture.csv`
- `experiments/baseline_diagnosis/results/m1_ncu/<capture_job_id>/selected_metrics.json`
- `experiments/baseline_diagnosis/results/m1_ncu/<capture_job_id>/capture_env_manifest.json`
- `artifacts/a_line/l1/ncu_metric_resolution_table_l1.json`

输出：

- `artifacts/a_line/l1/pka_feature_table_l1.json`
- `artifacts/a_line/l1/pka_acquisition_gap_l1.json`
- `artifacts/a_line/l1/pka_feature_audit_l1.json`
- `artifacts/a_line/l1/pka_join_audit_l1.json`

## 3. 非目标

Gate 3 不做：

- workload build。
- smoke run。
- NCU capture。
- metric query。
- metric selection。
- PCA。
- k-means。
- anchor selection。
- compression evaluation。
- B-line consumption。

Gate 3 不得为了让 selector 运行而补齐 feature。

## 4. 核心原则

### 4.1 Measured-only

Gate 3 formal path 只接受真实 measured feature。

禁止：

- `proxy`
- `derived`
- section label fallback
- semantic substitution
- default-zero fill
- 使用 M0 fixture 填补缺口

如果任一 feature 无法从允许的 NCU actual metric / launch metadata 中获得 measured value，该 invocation 必须进入 acquisition gap。

### 4.2 Exactly-one-outcome

每个 Gate 2 eligible capture job 的每个 consuming manifest entry 必须产生 exactly one outcome：

- 一条 measured feature record。
- 或一条 acquisition gap row。

不能既进入 feature table 又进入 gap。

不能静默丢弃 consuming manifest entry。

### 4.3 Gate 3 不判断 selector eligibility

Gate 3 允许输出 0 条、1 条、2 条或更多 measured records。

Gate 3 不因为 measured rows < 3 而失败。它只负责正确分类 measured / gap。

Gate 4 负责检查：

```text
measured_rows >= 3
```

并决定是否运行 selector 或触发 backward repair。

## 5. 12D Feature Contract

Gate 3 固定输出以下 12 维 PKA feature：

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

每个 feature 必须包含：

- `value`
- `status`
- `canonical_metric`
- `actual_source_metric`
- `source_artifact_path`
- `provenance`

进入 formal feature table 的 feature 必须满足：

```text
status == measured
```

## 6. Metric Source Rules

### 6.1 PM counter features

除 `num_thread_blocks` 之外，其他 11 个 feature 必须来自 Gate 2 `selected_metrics.json` 中允许的 actual source metric。

允许的 resolution status：

- `available`
- `rollup_resolved`

如果 NCU CSV 中出现未在 `selected_metrics.json` 中声明的 metric，不能作为 measured source。

### 6.2 launch_grid_size

`num_thread_blocks` 对应 canonical metric：

```text
launch_grid_size
```

它不是 PM counter，不来自 `--metrics`。

它只能来自：

- NCU CSV `Grid Size` column。
- 或等价 profiler launch metadata。

Gate 3 必须同时记录：

- raw Grid Size text。
- normalized scalar value。
- normalization rule。

例如：

```json
{
  "raw_grid_size": "(48, 32, 1)",
  "normalized_value": 1536,
  "normalization_rule": "product_of_grid_dimensions"
}
```

### 6.3 禁止 section label fallback

以下 label 不得作为 formal measured source：

- `Executed Instructions`
- `Grid Size` 作为 PM counter substitute。
- `num_blocks`
- `grid_dim`
- `total_dynamic_insts`

`Grid Size` 只能用于 `num_thread_blocks` 的 launch metadata provenance，不能替代任何 PM counter feature。

## 7. NCU CSV Parsing

Gate 3 parser 必须支持 Nsight Compute CSV preamble。

Parser 应通过 header row 定位表格主体。常见 header 第一列为：

```text
ID
```

每行通常表示一个 metric observation。

Gate 3 必须按 invocation 聚合 rows。建议聚合 key：

- NCU `ID`
- `Kernel Name`
- occurrence order

每个 invocation metric map 至少记录：

- `csv_invocation_id`
- `kernel_name`
- `metric_name -> metric_value`
- `grid_size_raw`
- `grid_size_normalized`
- `duration`
- `elapsed_cycles`

## 8. Consuming Manifest Entry Join

Gate 2 一个 capture job 可能服务多个 manifest entries。

例如：

```text
capture.csv
  -> L1_AI_01 gemm_tiled
  -> L1_AI_02 attention_score
  -> L1_AI_03 softmax_kernel
```

Gate 3 必须对每个 consuming manifest entry 独立 join。

Join key 使用：

- primary: `kernel_or_case` 与 NCU `Kernel Name` matching。
- secondary: occurrence order。

Gate 3 必须输出 join audit：

```text
artifacts/a_line/l1/pka_join_audit_l1.json
```

每条 join audit row 至少包含：

- `capture_job_id`
- `manifest_entry_id`
- `kernel_or_case`
- `csv_invocation_id`
- `kernel_name`
- `occurrence_index`
- `join_status`
- `auxiliary_grid_size_evidence`
- `reason`

允许的 join status：

- `matched`
- `missing_kernel`
- `ambiguous_kernel`
- `occurrence_mismatch`
- `empty_kernel_name`

## 9. Partial CSV / Non-zero Exit Provenance

如果 Gate 2 capture status 是：

```text
capture_non_zero_exit_with_partial_csv
```

Gate 3 允许解析该 CSV。

如果某个 invocation 仍然满足 12D 全 measured，可以进入 `pka_feature_table_l1.json`。

但 record 必须带 provenance：

- `capture_status: capture_non_zero_exit_with_partial_csv`
- `capture_warning: non_zero_exit`
- `capture_exit_code`
- `capture_stderr_path`

Gate 3 不能隐藏 non-zero-exit 风险。

如果 Gate 2 capture status 是 `permission_blocked`、`environment_blocked` 或 `malformed_ncu_csv`，该 job 不应被 Gate 3 消费，因为它应有：

```text
gate3_eligible == false
```

## 10. PkaFeatureRecord

进入 `pka_feature_table_l1.json` 的每条 record 至少包含：

- `record_id`
- `dataset_level`
- `source_type`
- `benchmark_name`
- `kernel_or_case`
- `kernel_invocation_id`
- `feature_mode`
- `features`
- `feature_status`
- `source_path`
- `capture_job_id`
- `manifest_entry_id`
- `capture_status`
- `feature_provenance`

要求：

- `dataset_level == L1`
- `feature_mode == pka_m1_measured`
- `feature_status == complete_measured`
- `features` 包含完整 12D。
- 每个 feature `status == measured`。
- 每个 feature provenance 可追溯到 NCU CSV / selected metric / Grid Size。

## 11. Acquisition Gap Row

任一 consuming manifest entry 或 invocation 不能生成完整 12D measured feature 时，必须进入：

```text
artifacts/a_line/l1/pka_acquisition_gap_l1.json
```

每条 gap row 至少包含：

- `record_id`
- `manifest_entry_id`
- `capture_job_id`
- `dataset_level`
- `source_type`
- `benchmark_name`
- `kernel_or_case`
- `kernel_invocation_id`
- `failed_gate: Gate3`
- `gap_reason`
- `missing_features`
- `join_status`
- `capture_status`
- `source_artifact_path`
- `selected_metrics_path`
- `environment_manifest_path`
- `suggested_repair_action`

## 12. Gate 3 Gap Reasons

允许的 Gate 3 gap reason：

- `missing_kernel_in_csv`
- `ambiguous_kernel_match`
- `occurrence_mismatch`
- `empty_kernel_name`
- `missing_canonical_metric`
- `metric_not_in_selected_allowlist`
- `section_label_rejected`
- `grid_size_missing`
- `grid_size_parse_failed`
- `invalid_metric_value`
- `incomplete_12d_feature_vector`
- `capture_status_not_gate3_eligible`
- `env_manifest_missing`
- `env_manifest_invalid`
- `selected_metrics_missing`
- `selected_metrics_invalid`

## 13. Feature Audit

Gate 3 必须输出：

```text
artifacts/a_line/l1/pka_feature_audit_l1.json
```

Audit summary 至少包含：

- `total_consuming_manifest_entries`
- `gate3_eligible_capture_jobs`
- `parsed_capture_jobs`
- `measured_record_count`
- `gap_record_count`
- `complete_12d_count`
- `incomplete_12d_count`
- `feature_missing_counts`
- `gap_reason_counts`

每条 audit entry 至少包含：

- `manifest_entry_id`
- `kernel_or_case`
- `capture_job_id`
- `kernel_invocation_id`
- `feature_status`
- `measured_features`
- `missing_features`
- `gap_reason`

## 14. Output Determinism

Gate 3 输出必须 deterministic：

- 按 Gate 2 capture job order 处理。
- 同一 capture job 内按 CSV first-seen invocation order 处理。
- consuming manifest entries 按 Gate 2 记录顺序处理。
- JSON 输出使用 stable key order。
- record_id 生成规则 deterministic。

推荐 record_id：

```text
m1rec_<manifest_entry_id>_<occurrence_index>
```

## 15. Gate 3 通过条件

Gate 3 通过，当且仅当：

- 所有 `gate3_eligible == true` 的 capture jobs 都被 attempted parse。
- 每个 consuming manifest entry 都产生 exactly one outcome。
- `pka_feature_table_l1.json` 中没有 incomplete 12D record。
- `pka_acquisition_gap_l1.json` 中每个 gap 都有明确 gap reason。
- `pka_feature_audit_l1.json` 存在。
- `pka_join_audit_l1.json` 存在。

Gate 3 通过不要求 measured rows >= 3。

## 16. 测试要求

Gate 3 测试至少覆盖：

- `gate3_eligible == false` 的 capture job 不被解析。
- valid CSV + complete 12D metrics -> measured feature record。
- missing one canonical metric -> acquisition gap。
- section label fallback 被拒绝。
- metric 不在 selected allowlist 中被拒绝。
- Grid Size missing -> gap `grid_size_missing`。
- Grid Size parse failed -> gap `grid_size_parse_failed`。
- shared capture CSV 为多个 manifest entries 产生独立 outcomes。
- ambiguous kernel match -> gap `ambiguous_kernel_match`。
- occurrence mismatch -> gap `occurrence_mismatch`。
- non-zero exit partial CSV 可解析，但 measured record 带 capture warning。
- env manifest missing -> gap `env_manifest_missing`。
- selected metrics missing -> gap `selected_metrics_missing`。
- feature table 不包含 incomplete 12D rows。
- each consuming manifest entry exactly one outcome。

## 17. 与其他 Gates 的接口

Gate 3 输入来自 Gate 2：

```text
m1_ncu_capture_attempts_l1.json
```

Gate 3 输出给 Gate 4：

```text
pka_feature_table_l1.json
pka_acquisition_gap_l1.json
pka_feature_audit_l1.json
pka_join_audit_l1.json
```

Gate 4 根据 feature table 中 measured rows 数量决定：

- `measured_rows >= 3` -> 运行 selector。
- `measured_rows < 3` -> 触发 backward repair。

## 18. 简短结论

Gate 3 的核心是：

```text
eligible NCU artifacts
  -> parse invocations
  -> join manifest entries
  -> extract strict measured 12D features
  -> feature table or acquisition gap
```

它必须严格 measured-only。任何缺失都进入 gap，不允许 proxy、derived、label fallback 或默认补 0。
