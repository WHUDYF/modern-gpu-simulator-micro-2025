# A 线 PKA-M1 Measured Loop 中文实施 Plan

日期：2026-05-08

## 1. Goal Description

本 plan 的目标是实现 A 线第一版 `PKA-M1` measured loop：对所有 L1 P0 workload 执行真实 acquisition attempt，通过 NCU exact metrics 获取严格 measured 的 12 维 PKA feature，在满足 selector 条件时复用 M0 已验证的 PCA / k-means / anchor 算法，输出正式 M1 representative anchors 和 structural compression evaluation。

整体链路如下：

```text
L1 P0 manifest
  -> Gate 1: workload resolver + smoke run
  -> Gate 2: exact-metric NCU capture dispatcher
  -> Gate 3: strict measured 12D feature extractor
  -> Gate 4: selector eligibility + backward repair
  -> Gate 5: formal M1 selector + structural evaluation
```

本 plan 允许两种正确结局：

- `measured_rows >= 3`：Gate 5 运行成功，输出正式 M1 PCA / k-means / anchor / evaluation artifacts。
- `measured_rows < 3`：Gate 5 不运行，但 Gate 1 到 Gate 4 完整执行，并输出逐 P0 entry 的 backward repair report。

不允许的结局是伪完成：例如 measured rows 不足却声称 selector 完成，或用 proxy / derived / fallback 补出 feature table。

## 2. Normative References

实现、review 和 stop-hook 判断必须遵循以下文档。若本 plan 与 spec 冲突，以 spec 为准。

- `docs/superpowers/specs/2026-05-07-a-line-pka-m1-measured-loop-design.md`
- `docs/superpowers/specs/2026-05-07-a-line-pka-m1-gate1-workload-resolver-design.md`
- `docs/superpowers/specs/2026-05-07-a-line-pka-m1-gate2-ncu-capture-dispatcher-design.md`
- `docs/superpowers/specs/2026-05-08-a-line-pka-m1-gate3-measured-feature-extractor-design.md`
- `docs/superpowers/specs/2026-05-08-a-line-pka-m1-gate4-selector-eligibility-repair-design.md`
- `docs/superpowers/specs/2026-05-08-a-line-pka-m1-gate5-formal-selector-evaluation-design.md`
- `docs/superpowers/plans/2026-05-08-a-line-pka-m1-measured-loop.plan.md`
- `docs/superpowers/plans/2026-05-06-a-line-pka-m0-minimal-loop-ablation.plan.md`

## 3. Non-Negotiable Constraints

- 必须覆盖所有 L1 P0 workload，不能只挑选容易的 3 个 workload。
- Gate 1 必须给每个 L1 P0 manifest entry 一个 outcome。
- Gate 2 必须覆盖所有 Gate 1 resolved records。
- Gate 3 必须 attempted parse 所有 `gate3_eligible == true` 的 capture jobs。
- Formal measured path 只接受 `status == measured` 的真实 NCU / profiler / launch metadata 来源。
- 禁止 `proxy`、`derived`、section label fallback、semantic substitution、default-zero fill。
- `--set full` 禁止用于 formal M1 capture；必须使用 explicit `--metrics`。
- `launch_grid_size` 不得进入 `--metrics`，只能来自 `Grid Size` 或 launch metadata。
- `measured_rows >= 3` 是运行 Gate 5 的硬门槛。
- `measured_rows < 3` 时 Gate 5 不得运行，必须输出完整 backward repair report。
- M0 / M1 必须共用 `pka_selector_core.py`，不能复制出一份漂移的 M1 PCA/k-means 实现。
- Gate 5 只能读取 Gate 4 生成的 selector input projection，不能直接读取完整 feature table metadata 决定 clustering。
- M1 不包含 B-line consumption；B-line 后续单独做 integration spec。
- M1 evaluation 只允许 structural metrics，不能声称 simulator accuracy 或 measured speedup。
- M1 不得写入或覆盖 `pka_m0_*` artifacts。

## 4. Acceptance Criteria

### AC-M1-1：L1 P0 workload registry 完整

Positive Tests:

- `experiments/baseline_diagnosis/workload_registry_l1.json` 存在。
- registry 至少覆盖 `l1_bw_32f`、`l2_bw_32f`、`mem_bw`、`mem_lat`、`shared_bw`、`MaxFlops`、`rodinia_nn`、`mini_transformer_v4`。
- 每个 registry entry 包含 `workload_id`、`binary_path`、`build_command`、`run_args`、`run_command_template`、`working_directory`、`smoke_timeout_seconds`、`capture_timeout_seconds`。
- `build_command` 要么是 `null`，要么是显式 list command。

Negative Tests:

- registry 缺少任一 P0 workload 时 Gate 1 测试失败。
- `build_command` 是字符串或缺失时测试失败。
- registry 只指向 dispatcher 且无真实 binary path 时测试失败。

### AC-M1-2：Gate 1 对所有 P0 entries 产生 resolution 或 gap

Positive Tests:

- `m1_workload_resolver.py` 读取 `kernel_validation_manifest_l1.json` 和 `workload_registry_l1.json`。
- 每个 L1 P0 manifest entry 产生一条 `M1WorkloadResolutionRecord` 或一条 Gate 1 gap。
- 通过 Gate 1 的 record 包含真实 `resolved_binary_path`，且不是 `dispatch_ncu_capture.sh`。
- binary 缺失时，只允许执行 registry 中 allowlisted `build_command`。
- smoke run 成功后才能 `resolution_status == resolved`。
- mini-transformer 三个 P0 entries 即使共享 binary，也输出三条 resolution records。

Negative Tests:

- registry missing -> `registry_missing` gap。
- dispatcher 被当作最终 binary -> `binary_unresolved` gap。
- binary missing 且无 build command -> Gate 1 gap。
- smoke timeout 或非 0 exit -> Gate 1 gap。
- 多个 P0 entry 被错误合并成一条 resolution record -> 测试失败。

### AC-M1-3：Gate 2 只对 Gate 1 resolved records 执行 exact NCU capture

Positive Tests:

- `m1_ncu_capture_dispatcher.py` 只消费 `resolution_status == resolved` 的 records。
- 相同 `resolved_run_command` 被去重为一个 capture job。
- capture job 保留所有 `consuming_manifest_entry_ids`。
- 每个 M1 run 至少有一个 `ncu --query-metrics` artifact。
- 每个 capture job 记录 query artifact path / hash 和 resolution table path / hash。
- NCU command 包含 `--metrics`，不包含 `--set full`。
- selected metrics 只来自 `available` 或 `rollup_resolved`。
- output directory 使用 deterministic `capture_job_id`。

Negative Tests:

- Gate 1 gap record 进入 Gate 2 -> 测试失败。
- `launch_grid_size` 被加入 selected metrics -> 测试失败。
- selected metrics 为空仍执行 capture -> 测试失败。
- command 使用 `--set full` -> 测试失败。
- 随机 UUID 作为 capture job id -> 测试失败。

### AC-M1-4：Gate 2 capture status 分类正确

Positive Tests:

- exit 0 + valid CSV -> `captured` 且 `gate3_eligible == true`。
- exit 非 0 + valid partial CSV -> `capture_non_zero_exit_with_partial_csv` 且 `gate3_eligible == true`。
- `ERR_NVGPUCTRPERM` -> `permission_blocked` 且 `gate3_eligible == false`。
- NCU / driver / GPU 环境不可用 -> `environment_blocked`。
- timeout 无 valid CSV -> `ncu_capture_timeout` 且 `gate3_eligible == false`。
- malformed CSV -> `malformed_ncu_csv`。

Negative Tests:

- permission error 被传给 Gate 3 -> 测试失败。
- malformed CSV 被标记为 eligible -> 测试失败。
- non-zero partial CSV 没有 provenance -> 测试失败。

### AC-M1-5：Gate 3 严格抽取 12D measured feature

Positive Tests:

- `m1_measured_feature_extractor.py` 只消费 `gate3_eligible == true` 的 capture jobs。
- valid CSV + complete 12D metrics 生成 `feature_mode == pka_m1_measured` 的 measured record。
- 每个 feature 包含 `value`、`status`、`canonical_metric`、`actual_source_metric`、`source_artifact_path`、`provenance`。
- `num_thread_blocks` 来自 `Grid Size` 或 launch metadata，并记录 raw + normalized provenance。
- non-zero partial CSV 可解析时，measured record 必须带 `capture_warning: non_zero_exit`。
- `pka_feature_table_l1.json` 中不存在 incomplete 12D row。

Negative Tests:

- 缺少任一 canonical metric -> acquisition gap。
- section label fallback 被当作 measured -> 测试失败。
- metric 不在 selected allowlist 中却被使用 -> 测试失败。
- Grid Size 缺失或解析失败 -> acquisition gap。
- default-zero fill -> 测试失败。

### AC-M1-6：Gate 3 对每个 consuming manifest entry exactly one outcome

Positive Tests:

- 每个 Gate 2 eligible capture job 的每个 consuming manifest entry 产生一个 measured record 或一个 acquisition gap。
- shared capture CSV 能为多个 manifest entries 独立 join。
- `pka_join_audit_l1.json` 记录 `matched`、`missing_kernel`、`ambiguous_kernel`、`occurrence_mismatch`、`empty_kernel_name`。
- `pka_feature_audit_l1.json` 统计 measured / gap / missing feature counts。

Negative Tests:

- consuming manifest entry 被静默丢弃 -> 测试失败。
- 同一个 entry 同时进入 feature table 和 gap -> 测试失败。
- ambiguous kernel match 被强行 measured -> 测试失败。

### AC-M1-7：Gate 4 正确判断 selector eligibility

Positive Tests:

- `measured_rows >= 3` 且无 gap -> `selector_ready`。
- `measured_rows >= 3` 且仍有 gap -> `selector_ready_with_remaining_gaps`。
- `measured_rows < 3` -> `selector_blocked_insufficient_measured_records`。
- invalid feature table -> `selector_blocked_invalid_feature_table`。
- mixed timing unit -> `selector_blocked_mixed_timing_unit`。
- Gate 4 输出 `m1_selector_eligibility_l1.json`。
- Gate 4 输出 `m1_selector_input_l1.json`，且只包含 selector allowed fields。

Negative Tests:

- measured rows < 3 仍允许 Gate 5 -> 测试失败。
- M0 fixture row 进入 selector input -> 测试失败。
- selector projection 包含 `kernel_name` / `source_path` / `expected_behavior_axis` -> 测试失败。
- mixed timing unit 被允许进入 Gate 5 -> 测试失败。

### AC-M1-8：Gate 4 backward repair report 完整

Positive Tests:

- 无论 Gate 5 是否可运行，只要存在 remaining gaps，Gate 4 都输出 `m1_backward_repair_report_l1.json` 和 `.md`。
- 每个 L1 P0 entry 在 report 中出现。
- 每个未 measured entry 有且仅有一个 `earliest_failed_gate`。
- earliest failed gate 优先级固定为 `Gate1 > Gate2 > Gate3 > Gate4`。
- repair action type 是 `executable_command`、`manual_action`、`code_fix_required` 或 `environment_action`。
- executable command 的来源只能是 registry、repo script、capture artifact 或 metric artifact。

Negative Tests:

- measured rows < 3 但缺 repair report -> RLCR stop/fail。
- report 只有空泛建议、没有 per-entry earliest gate -> RLCR stop/fail。
- repair command 由模型临场编造 -> 测试失败。

### AC-M1-9：M0 / M1 共用 selector core

Positive Tests:

- 新增 `experiments/baseline_diagnosis/pka_selector_core.py`。
- M0 的 preprocessing / PCA / k-means / anchor / evaluation 逻辑迁移到 shared selector core。
- `pka_m0_pipeline.py` 调用 shared selector core。
- M0 tests 继续通过。
- M0 artifact schema 和 `pka_m0_*` 输出路径保持不变。

Negative Tests:

- M1 复制一份独立 PCA / k-means 实现 -> review 不通过。
- M0 artifact mode / name 改变 -> 测试失败。
- M0 tests 回归 -> RLCR stop/fail。

### AC-M1-10：Gate 5 只消费 Gate 4 selector input projection

Positive Tests:

- `pka_m1_selector.py` 只读取 `m1_selector_eligibility_l1.json` 和 `m1_selector_input_l1.json`。
- Gate 5 在 `gate5_allowed == false` 时 abort。
- Gate 5 检查 selector input 中无 forbidden metadata。
- Gate 5 检查至少 3 条 `pka_m1_measured` records。

Negative Tests:

- Gate 5 直接读取完整 `pka_feature_table_l1.json` metadata 决定 clustering -> 测试失败。
- Gate 4 blocked 时 Gate 5 仍输出 formal artifacts -> 测试失败。
- selector input 包含 forbidden fields -> Gate 5 abort。

### AC-M1-11：Gate 5 输出 formal M1 artifacts

Positive Tests:

- Gate 5 成功时输出：
  - `artifacts/a_line/l1/pka_pca_projection_l1.json`
  - `artifacts/a_line/l1/pka_kmeans_clusters_l1.json`
  - `artifacts/a_line/l1/representative_anchor_table_l1.json`
  - `artifacts/a_line/l1/pka_compression_evaluation_l1.json`
- PCA method 是 `numpy_svd`。
- k-means method 是 deterministic farthest-first。
- representative 是 nearest-centroid real record。
- anchor membership 覆盖所有 measured selector input records，且每条 record 只属于一个 anchor。
- artifacts 包含 Gate 4 hash、selector input hash、deterministic replay hash。

Negative Tests:

- 使用 sklearn PCA 或 sklearn KMeans -> 测试失败。
- random initialization -> 测试失败。
- centroid 平均点被当作 representative -> 测试失败。
- M1 写入 `pka_m0_*` -> RLCR stop/fail。

### AC-M1-12：M1 evaluation 只做 structural metrics

Positive Tests:

- `pka_compression_evaluation_l1.json` 包含 compression ratio、coverage count、weighted coverage、cluster feature variance、top-k coverage、PCA diagnostics、k-means diagnostics、replay hash。
- `weight_mode` 来自 Gate 4。
- 全无 timing 使用 `member_count_fallback`。
- 单一 timing unit 使用 `timing_weight`。

Negative Tests:

- evaluation 声称 simulator accuracy -> 测试失败。
- evaluation 声称 measured speedup -> 测试失败。
- Gate 4 weight contract 缺失时仍计算 weighted coverage -> 测试失败。

### AC-M1-13：Orchestrator 串起 Gate1-Gate5

Positive Tests:

- 新增 `run_m1_measured_loop.py`。
- 执行顺序固定为 Gate1、Gate2、Gate3、Gate4、条件式 Gate5。
- `measured_rows >= 3` 且 Gate 4 允许时运行 Gate 5。
- `measured_rows < 3` 时不运行 Gate 5，但输出 blocked-on-acquisition 状态。
- dry-run capture 模式可用于无 NCU 环境下验证 artifact flow。

Negative Tests:

- Gate 4 blocked 仍运行 Gate 5 -> 测试失败。
- Gate 1/2/3 任一阶段静默跳过 P0 entries -> 测试失败。
- `<3` 且缺 repair report 被判定成功 -> 测试失败。

### AC-M1-14：最终验证命令通过

Positive Tests:

- M1 unit tests 全部通过。
- M0 tests 继续通过。
- `test_l1_regression.py` 通过。
- M1 loop 产生以下两种状态之一：
  - `completed_gate5_formal_selector`
  - `blocked_on_acquisition_with_repair_report`

Negative Tests:

- tests 失败仍宣布完成 -> RLCR stop/fail。
- 没有运行 M0 regression -> RLCR stop/fail。
- 工作区存在未解释的冲突变更 -> RLCR stop/fail。

## 5. Path Boundaries

### Upper Bound

本轮 RLCR 的最大可接受范围：

- Gate1-Gate5 全部实现。
- 所有 L1 P0 workload 都 attempted。
- 至少 3 条 measured rows 成功进入 Gate 5。
- 输出 formal M1 PCA / k-means / anchor / evaluation artifacts。
- remaining gaps 输出 backward repair report。
- M0/M1 共用 selector core。
- M0 regression 通过。
- 所有新增 M1 tests 通过。

### Lower Bound

本轮 RLCR 的最低可接受范围：

- Gate1-Gate4 全部实现并覆盖所有 L1 P0 entries。
- Gate2 可以因真实环境返回 permission / environment / metric / launcher blocker，但必须生成结构化 evidence。
- Gate3 对 eligible captures attempted parse。
- `measured_rows < 3` 时不运行 Gate 5。
- 输出完整 `m1_backward_repair_report_l1.json` 和 `.md`。
- 每个 P0 entry 有 measured 状态或 earliest failed gate。
- M0 tests 继续通过。

最低可接受范围不是“只实现 Gate1”或“只跑一个 workload”。

### Allowed Choices

Can use:

- Python standard library。
- `numpy`。
- `pytest`。
- JSON artifacts。
- Existing `shared_acquisition.py` metric allowlist。
- Existing NCU query / resolution logic。
- Dry-run NCU mode for deterministic tests。
- Controlled build commands from `workload_registry_l1.json`。

Cannot use:

- `--set full` formal capture。
- sklearn PCA。
- sklearn KMeans。
- random k-means initialization。
- proxy / derived / fallback feature。
- default-zero fill。
- section label fallback as measured source。
- M0 fixture as M1 input。
- Gate5 direct metadata reads from full feature table。
- B-line consumption in this plan。

## 6. Dependencies and Sequence

### Milestone 1：Gate1 Workload Resolver

目标：

- 建立 `workload_registry_l1.json`。
- 实现 `m1_workload_resolver.py`。
- 对所有 L1 P0 entries 输出 resolution 或 Gate1 gap。

验收：

- AC-M1-1。
- AC-M1-2。

### Milestone 2：Gate2 NCU Capture Dispatcher

目标：

- 实现 command-level dedup capture jobs。
- 实现 selected metrics resolution。
- 支持 dry-run 和真实 NCU capture。
- 输出 Gate2 attempts / gap artifacts。

验收：

- AC-M1-3。
- AC-M1-4。

### Milestone 3：Gate3 Measured Feature Extractor

目标：

- 解析 Gate2 eligible CSV。
- 按 consuming manifest entries join。
- 严格生成 12D measured feature table 或 acquisition gap。
- 输出 feature audit 和 join audit。

验收：

- AC-M1-5。
- AC-M1-6。

### Milestone 4：Gate4 Eligibility + Backward Repair

目标：

- 生成 selector input projection。
- 判断 Gate5 是否允许运行。
- 实现 timing unit policy。
- 输出完整 backward repair report。

验收：

- AC-M1-7。
- AC-M1-8。

### Milestone 5：Shared Selector Core

目标：

- 从 M0 抽取 `pka_selector_core.py`。
- M0 pipeline 改为调用 shared core。
- 保持 M0 artifact 和测试稳定。

验收：

- AC-M1-9。

### Milestone 6：Gate5 Formal M1 Selector

目标：

- 实现 `pka_m1_selector.py`。
- 只消费 Gate4 selector input projection。
- 输出 formal M1 artifacts。
- evaluation 只做 structural metrics。

验收：

- AC-M1-10。
- AC-M1-11。
- AC-M1-12。

### Milestone 7：End-to-End Orchestrator

目标：

- 实现 `run_m1_measured_loop.py`。
- 串起 Gate1-Gate5。
- 支持 full success 和 blocked-on-acquisition 两种结局。

验收：

- AC-M1-13。
- AC-M1-14。

## 7. Implementation Notes

- 代码中不要用 `AC-M1-*` 或 `Milestone` 命名业务函数。
- 每个 gate 的主脚本都应能独立运行。
- 每个 gate 的 artifact 都应 deterministic JSON 输出。
- Gate2 的真实 NCU capture 可能受环境影响；tests 应优先使用 dry-run 和 fixture CSV。
- Gate5 不能为了方便直接复用完整 feature table；必须通过 Gate4 projection。
- 若真实 NCU 环境导致 measured rows 不足，正确行为是 blocked-on-acquisition，不是失败，也不是成功伪装。

## 8. Required Verification

最小验证命令：

```bash
python experiments/baseline_diagnosis/m1_workload_resolver.py
python experiments/baseline_diagnosis/m1_ncu_capture_dispatcher.py
python experiments/baseline_diagnosis/m1_measured_feature_extractor.py
python experiments/baseline_diagnosis/m1_selector_eligibility.py
python experiments/baseline_diagnosis/pka_m1_selector.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_ncu_capture_dispatcher.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_measured_feature_extractor.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_selector_eligibility.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_selector.py
pytest -q experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py
pytest -q experiments/baseline_diagnosis/test_l1_regression.py
```

End-to-end 验证：

```bash
python experiments/baseline_diagnosis/run_m1_measured_loop.py
```

允许输出：

- `completed_gate5_formal_selector`
- `blocked_on_acquisition_with_repair_report`

不允许输出：

- `completed` 但缺 Gate5 artifacts。
- `completed` 但 measured rows < 3。
- `blocked` 但缺 backward repair report。

## 9. Stop-Hook Completion Rule

RLCR 必须在以下情况 STOP/FAIL：

- 不是所有 L1 P0 entries 都有 Gate1 outcome。
- Gate2 忽略 Gate1 resolved record。
- Gate3 忽略 Gate2 eligible capture job。
- `pka_feature_table_l1.json` 含 incomplete 12D row。
- `measured_rows < 3` 但缺完整 backward repair report。
- `measured_rows < 3` 但 Gate5 artifacts 被写出为成功。
- Gate5 直接读取完整 feature table metadata。
- Gate5 写入或覆盖 `pka_m0_*`。
- M0 tests 失败。
- evaluation 声称 simulator accuracy 或 measured speedup。
- B-line consumption 被混入本 plan。

## 10. 简短结论

本 plan 的核心是把 M1 做成一个可审计的 measured pipeline，而不是一个“能跑就行”的 selector demo。

正确实现后，A 线会得到两种有价值结果之一：

- 如果真实 NCU acquisition 足够成功，则获得正式 measured PKA selector anchors。
- 如果真实 acquisition 仍不足，则获得逐 workload、逐 gate 的 backward repair report，明确下一步应该修 build、capture、metric resolution、CSV parser 还是 join。
