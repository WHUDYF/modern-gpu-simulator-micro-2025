# A 线 PKA-M1 Gate 2 NCU Capture Dispatcher Design Spec

日期：2026-05-07

## 1. 背景

Gate 1 Workload Resolver 已经把每个 L1 P0 manifest entry 解析成真实可运行命令，并通过非 NCU smoke run 验证 workload 可启动。

Gate 2 的职责是把 Gate 1 通过的 resolved commands 转成正式 NCU exact-metric capture jobs，并输出完整 capture provenance。

Gate 2 不负责解析 PKA 12D feature，也不决定 invocation 是否进入 measured feature table。Gate 2 只判断 capture job 是否产生了可供 Gate 3 解析的 NCU artifact。

## 2. 目标

Gate 2 的目标是：

```text
Gate 1 resolved workload records
  -> command-level deduplication
  -> exact-metric NCU capture jobs
  -> capture attempt table
  -> capture gap table
  -> per-job NCU artifact directory
```

输入：

- `artifacts/a_line/l1/m1_workload_resolution_l1.json`
- `artifacts/a_line/l1/ncu_metric_query_l1.json`
- `artifacts/a_line/l1/ncu_metric_resolution_table_l1.json`
- `experiments/baseline_diagnosis/shared_acquisition.py`

输出：

- `artifacts/a_line/l1/m1_ncu_capture_attempts_l1.json`
- `artifacts/a_line/l1/m1_ncu_capture_gap_l1.json`
- `experiments/baseline_diagnosis/results/m1_ncu/<capture_job_id>/`

## 3. 非目标

Gate 2 不做：

- workload build。
- workload smoke run。
- NCU CSV 到 12D PKA feature 的解析。
- measured feature table 生成。
- PCA / k-means / anchor。
- proxy / derived feature 补齐。

Gate 2 可以标记 capture artifact 是否可交给 Gate 3，但不负责判断 12D feature 是否完整。

## 4. 核心原则

### 4.1 只消费 Gate 1 resolved records

Gate 2 只允许处理：

```text
resolution_status == resolved
```

的 Gate 1 records。

Gate 1 gap records 不进入 Gate 2。

### 4.2 Command-level capture，entry-level attribution

Gate 2 必须按 `resolved_run_command` 去重生成 capture jobs。

同一个 `resolved_run_command` 只运行一次 NCU capture。

但 capture job 必须记录它服务的所有 manifest entries：

```json
"consuming_manifest_entry_ids": [
  "L1_AI_01",
  "L1_AI_02",
  "L1_AI_03"
]
```

这用于 mini-transformer 这类 shared binary / multi-kernel workload：Gate 2 只 capture 一次，Gate 3 再从同一个 CSV 中按 kernel name / occurrence order 拆分 invocation。

### 4.3 Exact metrics only

正式 M1 capture 必须使用 explicit `--metrics`。

禁止 formal path 使用：

```text
--set full
```

`launch_grid_size` 不是 PM counter，不得出现在 `--metrics` 参数中。它必须由 NCU CSV 的 `Grid Size` column 或等价 launch metadata 提供。

### 4.4 Query provenance 必须可追溯

每次 M1 run 至少执行一次：

```bash
ncu --query-metrics
```

生成：

- `artifacts/a_line/l1/ncu_metric_query_l1.json`
- `artifacts/a_line/l1/ncu_metric_resolution_table_l1.json`

每个 capture job 必须记录：

- `query_artifact_path`
- `query_artifact_hash`
- `resolution_table_path`
- `resolution_table_hash`
- `environment_signature`

如果 environment signature 改变，必须重新 query，不能复用旧 query artifact。

### 4.5 Capture timeout 独立于 smoke timeout

Gate 1 的 `smoke_timeout_seconds` 只用于普通 workload run。

Gate 2 使用独立的 `capture_timeout_seconds`，因为 NCU profiling 会显著放慢 workload。

`capture_timeout_seconds` 可以来自：

- `workload_registry_l1.json`
- Gate 2 默认配置

## 5. Capture Job

Gate 2 将 Gate 1 resolved records 按 `resolved_run_command` 分组。

每个 group 生成一条 `M1NcuCaptureAttempt`。

每条 attempt 至少包含：

- `capture_job_id`
- `capture_job_index`
- `resolved_run_command_hash`
- `target_run_command`
- `ncu_capture_command`
- `working_directory`
- `consuming_manifest_entry_ids`
- `consuming_workload_ids`
- `consuming_kernel_or_cases`
- `selected_metrics`
- `query_artifact_path`
- `query_artifact_hash`
- `resolution_table_path`
- `resolution_table_hash`
- `environment_manifest_path`
- `capture_stdout_path`
- `capture_stderr_path`
- `capture_csv_path`
- `capture_exit_code_path`
- `capture_timeout_seconds`
- `capture_exit_code`
- `capture_status`
- `gate3_eligible`
- `gap_reason`

## 6. capture_job_id

`capture_job_id` 必须 deterministic。

禁止使用随机 UUID。

推荐格式：

```text
m1cap_<stable_index>_<readable_workload_token>_<short_hash>
```

示例：

```text
m1cap_000_l1_bw_32f_ab12cd34
m1cap_001_mini_transformer_v4_9f81e2aa
```

`short_hash` 来自 canonical `resolved_run_command`。

## 7. Per-job Output Directory

每个 capture job 输出到：

```text
experiments/baseline_diagnosis/results/m1_ncu/<capture_job_id>/
```

目录中至少包含：

- `capture.csv`
- `capture_stdout.log`
- `capture_stderr.log`
- `capture_exit_code.txt`
- `capture_env_manifest.json`
- `query_artifact_ref.json`
- `selected_metrics.json`
- `capture_command.json`

可选包含：

- `query_metrics_raw.txt`

如果 job 复用全局 query artifact，可以不复制 raw query，只写 `query_artifact_ref.json`。

## 8. Selected Metrics

Gate 2 的 selected metrics 来自 `ncu_metric_resolution_table_l1.json`。

只允许选择状态为以下之一的 PM counter：

- `available`
- `rollup_resolved`

禁止选择：

- `launch_metadata`
- `permission_blocked`
- `environment_blocked`
- `unsupported_metric`
- `not_found`

`num_thread_blocks` 对应的 `launch_grid_size` 是 launch metadata，不进入 selected metrics。

`selected_metrics.json` 至少记录：

- canonical metric
- actual source metric
- pka feature name
- resolution status
- rollup operation
- query evidence

## 9. NCU Command

Gate 2 生成的 command 必须满足：

```text
ncu --csv --target-processes all --metrics <selected_metrics> -- <target_run_command>
```

或者等价形式。

要求：

- 必须包含 `--metrics`。
- 不得包含 `--set full`。
- `selected_metrics` 必须非空。
- `target_run_command` 必须来自 Gate 1 resolved record。
- command 必须写入 `capture_command.json`。

`capture_command.json` 至少包含：

- `target_run_command`
- `ncu_capture_command`
- `selected_metrics`
- `working_directory`
- `capture_timeout_seconds`

## 10. Environment Manifest

每个 capture job 必须生成：

```text
capture_env_manifest.json
```

至少包含：

- GPU name
- compute capability
- driver version
- CUDA version
- Nsight Compute version
- environment signature
- capture timestamp
- target run command
- NCU capture command
- selected metrics
- output CSV path

Gate 3 只能消费带有 valid environment manifest 的 CSV。

## 11. Capture Status

Gate 2 capture status 使用以下枚举。

### 11.1 `captured`

条件：

- NCU exit code 为 0。
- `capture.csv` 存在。
- CSV header 可被识别。
- 未出现 `ERR_NVGPUCTRPERM`。

处理：

- `gate3_eligible: true`

### 11.2 `capture_non_zero_exit_with_partial_csv`

条件：

- NCU exit code 非 0。
- `capture.csv` 存在。
- CSV header 可被识别。
- 未出现 `ERR_NVGPUCTRPERM`。

处理：

- `gate3_eligible: true`
- 所有 Gate 3 解析出的 invocation 必须带 non-zero-exit provenance。

### 11.3 `permission_blocked`

条件：

- stdout / stderr / CSV 中出现 `ERR_NVGPUCTRPERM`。

处理：

- `gate3_eligible: false`
- 进入 capture gap。

### 11.4 `environment_blocked`

条件：

- `ncu` 不存在。
- driver / CUDA / GPU environment 无法初始化。
- `ncu --query-metrics` 无法在当前环境正常运行，且不是权限错误。

处理：

- `gate3_eligible: false`
- 进入 capture gap。

### 11.5 `ncu_capture_timeout`

条件：

- capture 超过 `capture_timeout_seconds`。

处理：

- 如果 timeout 后存在可解析 CSV 且不是权限错误，可以设置 `gate3_eligible: true`，但必须保留 timeout provenance。
- 如果没有可解析 CSV，则 `gate3_eligible: false`。

### 11.6 `malformed_ncu_csv`

条件：

- CSV 不存在。
- CSV header 不可识别。
- CSV 为空。

处理：

- `gate3_eligible: false`
- 进入 capture gap。

### 11.7 `metric_selection_failed`

条件：

- selected metrics 为空。
- 所有 canonical PM counters 都无法 resolution 到 actual metric。

处理：

- 不运行 capture。
- `gate3_eligible: false`
- 进入 capture gap。

## 12. Capture Gap

Gate 2 gap 输出到：

```text
artifacts/a_line/l1/m1_ncu_capture_gap_l1.json
```

每条 gap 至少包含：

- `capture_job_id`
- `consuming_manifest_entry_ids`
- `failed_gate: Gate2`
- `capture_status`
- `gap_reason`
- `target_run_command`
- `ncu_capture_command`
- `query_artifact_path`
- `query_artifact_hash`
- `environment_manifest_path`
- `stderr_path`
- `exit_code_path`
- `suggested_repair_action`
- `gate3_eligible`

## 13. Attempt Table

`m1_ncu_capture_attempts_l1.json` 包含：

- `artifact_name`
- `generated_at`
- `gate1_resolution_path`
- `query_artifact_path`
- `resolution_table_path`
- `attempts`
- `summary`

`summary` 至少包含：

- `resolved_gate1_records`
- `capture_job_count`
- `deduplicated_record_count`
- `captured_count`
- `partial_csv_count`
- `permission_blocked_count`
- `environment_blocked_count`
- `timeout_count`
- `malformed_csv_count`
- `gate3_eligible_count`

## 14. Determinism

Gate 2 output 必须 deterministic：

- capture jobs 按首次出现的 Gate 1 record order 排序。
- 相同 `resolved_run_command` 归并到同一个 job。
- `capture_job_id` 使用 stable index + command hash。
- JSON 输出 stable key order。

## 15. 测试要求

Gate 2 测试至少覆盖：

- Gate 1 unresolved records 不进入 capture。
- 相同 `resolved_run_command` 去重为一个 capture job。
- shared mini-transformer command 记录多个 `consuming_manifest_entry_ids`。
- NCU command 包含 `--metrics`。
- NCU command 不包含 `--set full`。
- `launch_grid_size` 不进入 selected metrics。
- selected metrics 只来自 `available` / `rollup_resolved`。
- selected metrics 为空时进入 `metric_selection_failed`。
- query artifact hash 被每个 capture job 记录。
- environment signature 变化时拒绝复用旧 query。
- exit 0 + valid CSV -> `captured` 且 `gate3_eligible: true`。
- exit nonzero + valid CSV -> `capture_non_zero_exit_with_partial_csv` 且 `gate3_eligible: true`。
- `ERR_NVGPUCTRPERM` -> `permission_blocked` 且 `gate3_eligible: false`。
- timeout with no valid CSV -> `ncu_capture_timeout` 且 `gate3_eligible: false`。
- malformed CSV -> `malformed_ncu_csv`。
- output directory 按 `capture_job_id` 组织。

## 16. 与其他 Gates 的接口

Gate 2 输入来自 Gate 1：

```text
m1_workload_resolution_l1.json
```

Gate 2 输出给 Gate 3：

```text
m1_ncu_capture_attempts_l1.json
```

Gate 3 只允许消费：

```text
gate3_eligible == true
```

的 capture jobs。

Gate 4 backward repair 必须根据 Gate 2 gap reason 判断是否修：

- NCU permission / environment
- metric resolution
- capture timeout
- capture command
- malformed CSV

## 17. 简短结论

Gate 2 的核心是：

```text
resolved commands
  -> deduplicated capture jobs
  -> exact --metrics NCU capture
  -> complete provenance
  -> gate3_eligible or Gate2 gap
```

它不负责让 feature measured，但必须生成足够严谨的 capture evidence，让 Gate 3 能判断是否能抽取 12D measured features。
