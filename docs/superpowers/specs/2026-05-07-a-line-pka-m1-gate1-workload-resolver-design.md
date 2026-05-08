# A 线 PKA-M1 Gate 1 Workload Resolver Design Spec

日期：2026-05-07

## 1. 背景

`PKA-M1` 的目标是把真实 L1 P0 workload 接入 NCU measured acquisition，并在采齐足够 measured rows 后复用 M0 的 PCA / k-means / anchor 逻辑。

Gate 1 是 M1 的第一个 stage。它不负责 profiling，也不负责抽取 PKA feature。它只回答一个问题：

```text
每个 L1 P0 manifest entry 是否能解析成一个真实、可执行、可 smoke run、可交给 NCU capture 的 workload command？
```

如果 Gate 1 没有把 workload id 解析成具体 executable 和 run command，后续 Gate 2 NCU capture 与 Gate 3 feature extraction 都没有可靠输入。

## 2. 目标

Gate 1 的目标是生成机器可读 workload resolution table。

输入：

- `artifacts/a_line/l1/kernel_validation_manifest_l1.json`
- `experiments/baseline_diagnosis/workload_registry_l1.json`

输出：

- `artifacts/a_line/l1/m1_workload_resolution_l1.json`
- `artifacts/a_line/l1/m1_workload_resolution_gap_l1.json`
- smoke run stdout / stderr sidecar files

Gate 1 必须覆盖所有 L1 P0 manifest entries。

## 3. 非目标

Gate 1 不做：

- NCU capture。
- NCU metric resolution。
- PKA 12D feature extraction。
- PCA / k-means / anchor selection。
- measured feature table 生成。
- proxy / derived feature 补齐。

Gate 1 的 smoke run 只证明 workload 可以普通运行，不证明 profiling 成功。

## 4. 核心原则

### 4.1 每个 P0 entry 一条 resolution record

每个 L1 P0 manifest entry 必须生成一条 `M1WorkloadResolutionRecord`。

即使多个 manifest entries 共享同一个 binary，也不能在 Gate 1 合并。

例如 mini-transformer：

- `gemm_tiled`
- `attention_score`
- `softmax_kernel`

它们可以共享同一个 `resolved_binary_path` 和 `resolved_run_command`，但必须输出三条 resolution records。这样后续 gap report 可以精确定位到每个 P0 kernel_or_case。

### 4.2 Dispatcher 不能作为最终 provenance

现有 manifest 中很多 P0 entry 使用：

```text
binary_path = experiments/baseline_diagnosis/dispatch_ncu_capture.sh
run_args = workload_id
```

Gate 1 允许通过 dispatcher / registry / override 解析 workload。

但通过 Gate 1 的 record 必须包含真实 `resolved_binary_path`。不能只把 dispatcher path 当成最终 binary provenance。

### 4.3 允许受控 build

如果 binary 不存在，Gate 1 resolver 可以执行 build command。

但 build command 必须来自 `workload_registry_l1.json` 的 allowlisted 字段。

禁止 RLCR / agent 临场猜测 build command。

### 4.4 必须 smoke run

Gate 1 通过前必须执行一次非 NCU smoke run。

Smoke run 的目的：

- 检查 binary 是否能启动。
- 检查 run args / working directory 是否正确。
- 检查输入文件、动态库、运行环境是否满足。
- 在进入 NCU 前暴露 workload 自身运行问题。

Smoke run 不得使用 NCU，不得产生 measured feature，不得替代 Gate 2。

## 5. Workload Registry

新增 registry：

```text
experiments/baseline_diagnosis/workload_registry_l1.json
```

每个 registry entry 至少包含：

- `workload_id`
- `source_type`
- `binary_path`
- `build_command`
- `run_args`
- `run_command_template`
- `working_directory`
- `expected_kernel_or_case`
- `capture_target_type`
- `smoke_args`
- `smoke_timeout_seconds`
- `expected_output_regex`

字段含义：

- `workload_id`: 与 manifest 中 `run_args` 或 workload key 对齐的稳定 id。
- `source_type`: `local_microbench`、`local_benchmark_result`、`local_ai_workload` 等。
- `binary_path`: 真实 executable 的期望路径，不是 dispatcher。
- `build_command`: allowlisted build 命令；可以为空。
- `run_args`: formal capture 使用的默认参数。
- `run_command_template`: 如何从 binary 和 args 生成命令。
- `working_directory`: 命令执行目录。
- `expected_kernel_or_case`: 预期出现的 kernel / case 名。
- `capture_target_type`: `single_kernel_binary`、`multi_kernel_binary`、`dispatcher_resolved_binary` 等。
- `smoke_args`: smoke run 使用的小输入参数；可以不同于 formal capture args。
- `smoke_timeout_seconds`: smoke run timeout。
- `expected_output_regex`: 可选输出匹配规则。

## 6. M1WorkloadResolutionRecord

每条 resolution record 至少包含：

- `manifest_entry_id`
- `workload_id`
- `benchmark_name`
- `kernel_or_case`
- `source_type`
- `registry_entry_id`
- `dispatcher_path`
- `dispatcher_arg`
- `resolved_binary_path`
- `resolved_run_args`
- `resolved_run_command`
- `working_directory`
- `build_command`
- `build_attempted`
- `build_status`
- `binary_exists`
- `binary_executable`
- `smoke_run`
- `resolution_status`
- `gap_reason`

`smoke_run` 至少包含：

- `enabled`
- `command`
- `timeout_seconds`
- `exit_code`
- `elapsed_ms`
- `stdout_tail_path`
- `stderr_tail_path`
- `status`
- `failure_reason`

## 7. Resolution Flow

Gate 1 resolver 按以下顺序运行：

```text
load manifest
  -> filter L1 P0 entries
  -> load workload_registry_l1.json
  -> join manifest entry with registry entry
  -> resolve dispatcher to real binary when needed
  -> check binary exists / executable
  -> if missing and build_command exists, run build
  -> recheck binary exists / executable
  -> construct resolved run command
  -> run smoke command with timeout
  -> emit resolution record or resolution gap
```

## 8. Gate 1 通过条件

一个 manifest entry 通过 Gate 1，当且仅当：

- registry 中存在对应 workload。
- resolver 能得到真实 `resolved_binary_path`。
- `resolved_binary_path` 不是 dispatcher。
- binary 存在。
- binary 可执行。
- `resolved_run_command` 明确。
- `working_directory` 明确且存在。
- smoke run 在 timeout 内完成。
- smoke run exit code 为 0。
- 如果配置了 `expected_output_regex`，smoke output 必须匹配。
- `resolution_status == resolved`。

## 9. Gap Reasons

Gate 1 失败时进入：

```text
artifacts/a_line/l1/m1_workload_resolution_gap_l1.json
```

允许的 gap reason：

- `registry_missing`
- `binary_unresolved`
- `binary_missing`
- `binary_not_executable`
- `build_command_missing`
- `build_failed`
- `run_command_missing`
- `working_directory_missing`
- `smoke_timeout`
- `smoke_non_zero_exit`
- `smoke_missing_input`
- `smoke_missing_library`
- `smoke_runtime_error`
- `smoke_output_mismatch`

每条 gap row 必须包含：

- `manifest_entry_id`
- `workload_id`
- `kernel_or_case`
- `failed_gate: Gate1`
- `gap_reason`
- `attempted_resolution_steps`
- `suggested_repair_action`
- `can_enter_ncu_capture: false`

## 10. Smoke Run

### 10.1 Smoke run 的定义

Smoke run 是一次普通 workload run：

```text
timeout <smoke_timeout_seconds> <resolved_binary_path> <smoke_args>
```

或者使用 `run_command_template` 生成等价命令。

### 10.2 Smoke run 记录

每次 smoke run 必须记录：

- command
- working directory
- timeout seconds
- exit code
- elapsed ms
- stdout tail path
- stderr tail path
- status

stdout / stderr 不要求完整写入主 JSON。主 JSON 记录 tail sidecar 路径即可。

### 10.3 Smoke run 失败分类

Resolver 应根据 stderr / exit code / timeout 做基础分类：

- timeout -> `smoke_timeout`
- exit code 非 0 -> `smoke_non_zero_exit`
- stderr 包含 missing input 相关信号 -> `smoke_missing_input`
- stderr 包含 shared library 相关信号 -> `smoke_missing_library`
- stderr 包含 CUDA runtime / illegal instruction / segmentation fault -> `smoke_runtime_error`
- expected output 不匹配 -> `smoke_output_mismatch`

### 10.4 Smoke run 边界

Smoke run：

- 不使用 NCU。
- 不生成 measured feature。
- 不写 `pka_feature_table_l1.json`。
- 不写 `representative_anchor_table_l1.json`。
- 不能替代 Gate 2 capture。

## 11. Shared Binary Handling

对于 shared binary workload，例如 mini-transformer：

```text
mini_transformer_v4 binary
  -> gemm_tiled
  -> attention_score
  -> softmax_kernel
```

Gate 1 输出三条 records。

三条 records 可以共享：

- `resolved_binary_path`
- `resolved_run_command`
- `working_directory`
- `smoke_run` evidence

但三条 records 必须分别保留：

- `manifest_entry_id`
- `benchmark_name`
- `kernel_or_case`
- `expected_kernel_or_case`
- `resolution_status`
- `gap_reason`

Gate 2 可以基于相同 `resolved_run_command` 去重 capture，但 Gate 1 不做 entry 合并。

## 12. Artifact Requirements

### 12.1 Resolution Table

`m1_workload_resolution_l1.json` 包含：

- `artifact_name`
- `generated_at`
- `manifest_path`
- `registry_path`
- `records`
- `summary`

`summary` 至少包含：

- `total_p0_entries`
- `resolved_count`
- `gap_count`
- `smoke_passed_count`
- `smoke_failed_count`

### 12.2 Gap Table

`m1_workload_resolution_gap_l1.json` 包含：

- `artifact_name`
- `generated_at`
- `failed_gate: Gate1`
- `gaps`
- `summary`

### 12.3 Determinism

Resolution output 必须 deterministic：

- records 按 manifest entry order 输出。
- JSON 使用 stable key order。
- 相同输入和相同文件系统状态下，输出 records 顺序稳定。

## 13. 测试要求

Gate 1 测试至少覆盖：

- registry missing -> gap `registry_missing`。
- dispatcher path 不能作为最终 binary provenance。
- binary missing + build command missing -> gap `build_command_missing` 或 `binary_missing`。
- binary missing + build success -> resolved。
- binary not executable -> gap `binary_not_executable`。
- working directory missing -> gap `working_directory_missing`。
- smoke timeout -> gap `smoke_timeout`。
- smoke non-zero exit -> gap `smoke_non_zero_exit`。
- smoke output mismatch -> gap `smoke_output_mismatch`。
- shared binary three manifest entries -> three resolution records。
- resolution JSON deterministic order。

## 14. 与后续 Gates 的接口

Gate 1 成功 records 是 Gate 2 的输入。

Gate 2 只允许对 `resolution_status == resolved` 的 records 执行 NCU capture。

Gate 1 gap records 不进入 Gate 2。

Gate 4 backward repair 必须优先修复 Gate 1 gap，因为没有 resolved executable 时，NCU capture 与 feature extraction 都不可诊断。

## 15. 简短结论

Gate 1 的核心是把“抽象 workload”变成“真实可运行命令”。

它必须做到：

```text
manifest entry
  -> registry lookup
  -> resolved binary
  -> optional controlled build
  -> smoke run
  -> resolution record or Gate1 gap
```

只有通过 Gate 1 的 workload 才能进入正式 NCU acquisition。
