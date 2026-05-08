# A 线 PKA-M1 Measured Loop Design Spec

日期：2026-05-07

## 1. 背景

A 线已经完成 `PKA-M0` 最小算法闭环。M0 使用人工构造的 12 维 PKA-like fixture，验证了以下链路：

```text
12D fixture feature table
  -> preprocessing
  -> numpy SVD PCA
  -> deterministic farthest-first k-means
  -> nearest-centroid anchor selection
  -> compression evaluation
```

M0 的价值是证明 selector / PCA / k-means / anchor / evaluation 这条算法链路可以稳定运行。但 M0 不是 PKA measured reproduction，不能作为正式 baseline。

`PKA-M1` 的目标是把 M0 中已经稳定的算法链路接到真实 NCU measured 12D feature table 上，从而形成第一版正式 measured PKA baseline loop。

## 2. M1 目标

M1 第一版采用策略：

```text
全 L1 P0 workload acquisition attempt
  + measured-only 12D feature extraction
  + measured_rows >= 3 时运行 PCA/k-means selector
  + measured_rows < 3 时强制 backward repair
```

M1 不是只修一个 workload，也不是要求所有 workload 第一次全部采齐。

M1 的目标是：

- 对所有 L1 P0 workload 执行 acquisition attempt。
- 对能采齐 12 维 measured feature 的 invocation，写入正式 `pka_feature_table_l1.json`。
- 对采不齐的 invocation，写入 `pka_acquisition_gap_l1.json`，并记录明确 gap reason。
- 当 measured rows >= 3 时，复用 M0 的 PCA/k-means/anchor/evaluation 逻辑，跑通正式 M1 selector loop。
- 当 measured rows < 3 时，不允许伪完成，必须回看 Gate 1 到 Gate 3 并启动 backward repair。

## 3. 非目标

M1 第一版不追求：

- 一次性让所有 L1 P0 workload 全部 measured。
- 直接扩展到 L2 workload。
- 做 simulator accuracy / speedup 结论。
- 引入新的 PCA 或 k-means 算法。
- 用 PKA-M0 fixture 混入 formal measured pipeline。
- 用 proxy / derived / semantic fallback 伪造 measured feature。

## 4. 核心原则

### 4.1 Measured-only

M1 formal path 只接受真实 measured feature。

禁止：

- `proxy`
- `derived`
- section label fallback
- semantic substitution
- default-zero fill
- 使用 M0 fixture 填补真实 feature 缺口

如果任一 feature 无法从允许的 NCU metric / launch metadata 中获得 measured value，该 invocation 必须进入 acquisition gap。

### 4.2 M0 / M1 artifact 隔离

M0 artifact 必须继续使用 `pka_m0_*` 路径。

M1 formal artifact 使用正式路径：

- `artifacts/a_line/l1/pka_feature_table_l1.json`
- `artifacts/a_line/l1/representative_anchor_table_l1.json`
- `artifacts/a_line/l1/pka_compression_evaluation_l1.json`
- `artifacts/a_line/l1/pka_acquisition_gap_l1.json`

M1 不得读取 `pka_m0_feature_table_l1.json` 作为 formal measured input。

### 4.3 算法复用

M1 不重新发明 selector。

M1 selector 应复用 M0 已验证的算法逻辑：

- fixed 12D feature order
- count-like feature `log1p`
- `divergence_efficiency` clip to `[0, 1]`
- z-score normalization
- `numpy SVD` PCA
- deterministic farthest-first k-means
- nearest-centroid representative selection
- deterministic replay hash

允许为了复用而把 M0 pipeline 中的纯算法函数抽取为公共模块，但不得改变 M0 已验证语义。

## 5. M1 总流程

```text
L1 P0 manifest
  -> Gate 1: Workload Resolution
  -> Gate 2: NCU Exact Capture
  -> Gate 3: 12D Measured Feature Extraction
  -> Gate 4: Selector Eligibility + Backward Repair Trigger
  -> Gate 5: Formal M1 Selector + Evaluation
```

## 6. Gate 1: Workload Resolution

### 6.1 目的

确认每个 L1 P0 workload 都能解析为可运行对象。

### 6.2 每个 workload 必须记录

- `workload_id`
- `benchmark_name`
- `kernel_or_case`
- `source_origin`
- `build_command`
- `run_command`
- `binary_path`
- `input_args`
- `output_path`
- `resolution_status`

### 6.3 通过条件

一个 workload 通过 Gate 1，当且仅当：

- binary path 存在且可执行，或 build command 能生成该 binary。
- run command 明确。
- input args 明确。
- output path 明确。
- invocation 可以被后续 NCU capture 定位。

### 6.4 未通过处理

未通过 Gate 1 的 workload 不进入 NCU capture。

它必须进入 acquisition gap，gap reason 使用：

- `launcher_unresolved`
- `binary_missing`
- `build_failed`
- `run_command_missing`
- `input_args_missing`

## 7. Gate 2: NCU Exact Capture

### 7.1 目的

对 Gate 1 通过的 workload 执行正式 NCU exact-metric capture。

### 7.2 强制要求

- 必须使用 explicit `--metrics`。
- 禁止 formal path 使用 `--set full`。
- capture 前必须运行 `ncu --query-metrics`。
- 必须记录 actual metrics list。
- 必须记录 environment manifest。
- 必须保留 stderr、exit code、query output。
- 必须保留 capture command line。

### 7.3 Metric resolution

M1 允许 canonical metric 到 actual source metric 的精确 resolution。

例如，若当前 NCU query 输出只暴露 base metric：

```text
smsp__inst_executed
```

而 canonical metric 是：

```text
smsp__inst_executed.sum
```

则允许记录为：

```text
canonical_metric: smsp__inst_executed.sum
actual_source_metric: smsp__inst_executed
rollup_operation: sum
status: measured
```

但这必须来自 `ncu --query-metrics` 的真实 metric availability，不允许 section label 替代。

### 7.4 未通过处理

Capture 失败不直接终止整个 M1 pipeline，而是进入 gap。

允许的 gap reason 包括：

- `permission_blocked`
- `environment_blocked`
- `unsupported_metric`
- `metric_not_found`
- `launcher_non_zero_exit`
- `malformed_ncu_csv`
- `ncu_capture_failed`

## 8. Gate 3: 12D Measured Feature Extraction

### 8.1 目的

从 NCU CSV / launch metadata 中抽取正式 12 维 PKA feature。

### 8.2 12 维 feature

M1 feature order 固定为：

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

### 8.3 进入 feature table 的条件

一个 invocation 进入 `pka_feature_table_l1.json`，必须满足：

- 12 个 feature 全部存在。
- 每个 feature 的 `status == measured`。
- 每个 feature 有 `canonical_metric`。
- 每个 feature 有 `actual_source_metric`。
- 每个 feature 有 `source_artifact_path`。
- 每个 feature 能追溯到 NCU CSV、profiler field 或 launch metadata。
- `num_thread_blocks` 来自 NCU `Grid Size` 或等价 launch metadata。
- `num_thread_blocks` 同时记录 raw Grid Size 和 normalized scalar。

### 8.4 未通过处理

如果任一 feature 缺失，该 invocation 不进入 feature table。

它进入 `pka_acquisition_gap_l1.json`，并记录：

- `record_id`
- `kernel_invocation_id`
- `workload_id`
- `failed_gate`
- `missing_features`
- `gap_reason`
- `source_artifact_path`
- `suggested_repair_action`

## 9. Gate 4: Selector Eligibility + Backward Repair Trigger

### 9.1 目的

Gate 4 判断 formal selector 是否有足够 measured rows 可以运行。

PCA + k-means 至少需要 3 条 measured records。少于 3 条时，selector 不应运行。

### 9.2 measured_rows >= 3

当：

```text
measured_rows >= 3
```

则允许进入 Gate 5。

要求：

- selector 只读取 measured 12D features。
- gap rows 保留在 acquisition report 中。
- gap rows 不参与 PCA、k-means、anchor 或 evaluation。

### 9.3 measured_rows < 3

当：

```text
measured_rows < 3
```

则 selector 必须停止，并输出：

```text
insufficient_measured_records_for_pca
```

但这不能被视为 M1 完成。

此时必须启动 backward repair：

```text
measured_rows < 3
  -> classify blockers by failed gate
  -> repair earliest failing gate first
  -> rerun acquisition
  -> recheck measured_rows
```

### 9.4 Backward repair 优先级

修复顺序必须从最早失败 gate 开始：

1. Gate 1: build / binary / run command / workload id resolution。
2. Gate 2: NCU exact metrics / permission / environment / exit code / CSV output。
3. Gate 3: metric resolution / CSV parser / Grid Size provenance / kernel invocation matching。
4. Gate 4: 重新检查 measured rows 是否达到 3。

禁止跳过 Gate 1 的 launcher 问题去调 Gate 3 parser。

禁止在 Gate 2 capture 未成功时伪造 Gate 3 feature。

### 9.5 Backward repair report

如果 RLCR 在 `measured_rows < 3` 时无法继续修复，必须输出 backward repair report：

- `artifacts/a_line/l1/m1_backward_repair_report_l1.json`
- `artifacts/a_line/l1/m1_backward_repair_report_l1.md`

report 必须包含：

- 每个 P0 workload 的当前 gate status。
- 每个失败 workload 的 earliest failed gate。
- 已尝试的 repair action。
- repair 后 measured row count 是否增加。
- 下一步需要人工操作还是代码修复。
- 为什么当前不能进入 selector。

## 10. Gate 5: Formal M1 Selector + Evaluation

### 10.1 前置条件

Gate 5 只能在以下条件满足时运行：

- `pka_feature_table_l1.json` 存在。
- feature table 中 measured rows >= 3。
- 每条 row 都有完整 12D measured features。
- 每条 row 的 feature provenance 完整。
- forbidden-field guard 通过。

### 10.2 Selector 读取字段

Selector 只能读取：

- `record_id`
- `kernel_invocation_id`
- `features`
- `feature_mode`

Selector 不得读取：

- `kernel_name`
- `source_path`
- `expected_behavior_axis`
- `family`
- `regime`
- `shape_hint`
- `trace_order`
- B-line semantic metadata

### 10.3 输出

Gate 5 输出：

- `artifacts/a_line/l1/representative_anchor_table_l1.json`
- `artifacts/a_line/l1/pka_compression_evaluation_l1.json`

允许额外输出调试 artifact，例如：

- `artifacts/a_line/l1/pka_pca_projection_l1.json`
- `artifacts/a_line/l1/pka_kmeans_clusters_l1.json`

这些调试 artifact 不得使用 `pka_m0_*` 前缀。

### 10.4 Evaluation 范围

M1 evaluation 只计算结构性压缩指标：

- compression ratio
- coverage count
- weighted coverage
- cluster feature variance
- top-k coverage
- deterministic replay hash

M1 第一版不得声称 simulator accuracy 或 measured speedup。

## 11. Completion Rules

### 11.1 成功完成

M1 第一版可以被判定为成功完成，当且仅当：

- 所有 L1 P0 workload 都完成 acquisition attempt。
- 至少 3 条 invocation 进入 formal measured feature table。
- 每条 measured row 都有完整 12D measured features。
- gap rows 都有明确 failed gate 和 gap reason。
- Gate 5 selector 成功运行。
- `representative_anchor_table_l1.json` 输出成功。
- `pka_compression_evaluation_l1.json` 输出成功。
- M0 artifact 未被覆盖。
- regression tests 通过。

### 11.2 允许阻塞但不得伪完成

如果 measured rows < 3，M1 不得宣布 selector 完成。

此时 RLCR 只能以 blocked-on-acquisition 状态结束，并且必须满足：

- 已输出 `m1_backward_repair_report_l1.json`。
- 已输出 `m1_backward_repair_report_l1.md`。
- report 清楚说明每个 P0 workload 卡在哪个 gate。
- report 清楚说明下一步 repair action。

### 11.3 禁止完成

出现以下任一情况，不能判定 M1 完成：

- 只采集了一个 workload，没有尝试所有 L1 P0 workload。
- measured rows < 3 且没有 backward repair report。
- selector 使用了 proxy / derived / label fallback feature。
- selector 使用 forbidden metadata。
- gap rows 被补 0 后进入 feature table。
- M0 fixture 混入 M1 feature table。
- M1 输出覆盖 `pka_m0_*` artifact。

## 12. 推荐实现切分

### 12.1 Workload launcher resolver

负责将 manifest 中的 workload id 解析为：

- binary path
- build command
- run command
- args
- output path

### 12.2 NCU capture dispatcher

负责：

- 运行 `ncu --query-metrics`。
- 生成 metric resolution table。
- 运行 explicit-metrics capture。
- 生成 environment manifest。
- 保存 stderr / exit code / raw query output。

### 12.3 Feature extractor hardening

负责：

- 从 NCU CSV 抽取 12D measured feature。
- 保证 allowlist validation。
- 保证 Grid Size raw + normalized provenance。
- 把缺失 feature route 到 gap。

### 12.4 M1 selector wrapper

负责：

- 检查 measured rows >= 3。
- 复用 M0 preprocessing / PCA / k-means / anchor 逻辑。
- 输出 formal M1 artifacts。
- 保证 forbidden-field guard。

### 12.5 Backward repair reporter

负责：

- 聚合 Gate 1 到 Gate 4 的失败状态。
- 找出 earliest failed gate。
- 输出 JSON / Markdown repair report。

## 13. 推荐测试

M1 测试至少覆盖：

- 所有 P0 workload 都被 attempted。
- binary missing 进入 Gate 1 gap。
- `--set full` 在 formal path 被拒绝。
- explicit `--metrics` capture command 被接受。
- NCU query failure 进入 `environment_blocked` 或 `permission_blocked`。
- missing canonical metric 进入 gap，而不是 measured。
- section label fallback 不能成为 formal measured。
- 12D 全 measured row 能进入 feature table。
- measured rows < 3 时 selector 停止并生成 backward repair report。
- measured rows >= 3 时 selector 跑通。
- selector 不读取 forbidden metadata。
- M0 artifact 不被覆盖。

## 14. 与后续工作的关系

M1 第一版完成后，A 线将拥有：

- M0: fixture-based algorithm loop。
- M1: real measured L1 loop。

后续可以继续推进：

- 扩大 measured workload 覆盖率。
- 将同一 acquisition contract 推广到 L2。
- 与 Photon / PKA / 其他 baseline 做消融对比。
- 把 M1 anchors 接入 B 线消费流程。

## 15. 简短结论

M1 的核心不是“采不到就停”，而是：

```text
全 workload 尝试采集
  -> 采齐的进入 measured selector
  -> 采不齐的进入 gap
  -> measured rows 不足 3 时强制回溯修复最早失败 gate
```

这样可以同时满足两个目标：

- 暴露并修复真实 NCU acquisition 问题。
- 尽快打通正式 measured PKA selector 闭环。
