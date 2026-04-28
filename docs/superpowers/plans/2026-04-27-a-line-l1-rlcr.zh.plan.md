# A 线 L1 RLCR 中文实施计划

日期：2026-04-27

## 1. 目标说明

本计划用于实现 A 线 L1 RLCR 的第一轮闭环。

L1 的目标不是证明 A 线压缩效果最好，
而是先在一组小而可解释的 kernel 上证明：

- PKA baseline 输入契约成立；
- PKA 12 维 measured feature 能被稳定抽取；
- selector 不依赖禁止字段；
- representative anchor table 能被输出；
- B 线可以读取并校验 A 线 anchor 输出。

L1 有两种合法完成状态。

### 1.1 完整闭环成功

所有 10 个 P0 对象都能生成 12 维 measured PKA feature。

此时 pipeline 可以继续：

```text
manifest
  -> PKA feature table
  -> pka_baseline selector
  -> representative anchor table
  -> B-line consumption report
```

这证明 L1 输入、selector、anchor 输出和 B 线接口可以完整闭环。

### 1.2 Acquisition-gate 成功

一个或多个 P0 对象无法生成完整 12 维 measured feature。

此时 pipeline 必须停在 Stage 2，
只输出：

- `kernel_validation_manifest_l1.json`
- `pka_feature_audit_l1.md`
- `pka_feature_audit_l1.json`
- `pka_acquisition_gap_l1.json`

这不是失败。
这是 measured-only stage-gate 在现有数据不完整时的正确行为。

核心不变量是：

**只要存在未解决的 P0 acquisition gap，pipeline 就不能继续 selector、anchor table 或 B-line consumption。**

---

## 2. 验收标准

每个验收标准都应有正向测试和反向测试，
保证行为可重复、可验证。

### AC-1：L1 manifest 可机器读取并通过 schema 校验

正向要求：

- 所有 10 个 P0 manifest entries 出现在 JSON 输出中；
- 每个 entry 有稳定 `id`、`source_type`、`benchmark_name`、`kernel_or_case`、`priority`、`local_input_path`；
- manifest 能通过 `kernel_validation_manifest_schema.json`；
- P1 entries 在开启配置时也能输出；
- 每个 entry 有 `expected_behavior_axis`，但该字段只用于人工 sanity check，不得进入 grouping。

反向要求：

- 缺少 `id` 的 manifest 必须被拒绝；
- 重复 `id` 必须被拒绝；
- `local_input_path` 不存在时必须报错；
- 非法 `source_type` 必须被 schema 拒绝；
- `local_microbench` 指向非 JSON 文件时必须被拒绝。

补充要求：

- manifest builder 使用现有 schema 字段名；
- draft 中的 `validation_id` 对应现有 schema 的 `id`；
- draft 中的 `source_path` 对应现有 schema 的 `local_input_path`。

### AC-2：PKA feature table 只包含 12 维 measured features

正向要求：

- 每个进入 feature table 的 P0 invocation 都有且只有 12 个 PKA feature；
- 每个 feature 都有非空数值、`status: "measured"` 和可追溯 `source`；
- `source` 必须是 Nsight metric name 或 profiler / launch metadata field；
- `num_thread_blocks` 必须来自 `launch_grid_size` 或等价 profiler / launch record；
- 当 12 维全齐时，`feature_mode` 为 `pka_l1_measured_only`；
- 相同输入重复运行，输出必须完全一致。

反向要求：

- 缺少任意 PKA feature 的 invocation 不得进入 feature table；
- 该 invocation 只能进入 `PkaAcquisitionGap`；
- 使用填充值、默认 0、语义替代值的 invocation 必须被拒绝；
- `num_thread_blocks` 如果不是来自 profiler / launch metadata，必须被拒绝；
- 任一 P0 acquisition gap 未解决时，AC-3 / AC-4 不可达。

### AC-3：PKA baseline selector 只能使用 12 维 feature space

正向要求：

- selector 输出实际使用字段列表；
- 输出每个使用字段的 status；
- 输出 `feature_mode`；
- 相同输入下 cluster assignment 和 anchor choice 必须确定；
- 每个 cluster 有 `cluster_id`、member list、representative 和 membership count；
- anchor table schema 可机器校验。

反向要求：

- grouping key 中出现 `kernel_name` 必须报 forbidden-field error；
- grouping key 中出现 `grid_dim` 或 `block_dim` string 必须报错；
- grouping key 中出现 `cross_tb_offset_coverage`、`squash_boundary_crossing_flag` 或 compression-side 字段必须报错；
- grouping key 中出现 `family_id`、`regime_id`、`route_primitive`、`execution_template`、`simulator_lane_id` 必须报错；
- anchor table 输出中包含任何禁止字段时，validator 必须拒绝。

前置 gate：

- 只有当 `PkaAcquisitionGap` 中没有阻塞 P0 对象时，selector 才能运行；
- 任一 P0 gap 未解决时，selector 必须拒绝运行。

### AC-4：Representative anchor table 可被 B 线解析和校验

正向要求：

- B-line consumer 能读取 anchor table JSON；
- 每个 anchor row 都有必需字段：
  `rep_kernel_id`、`kernel_name`、`cluster_id`、`member_invocations`、`coverage_count`、`coverage_weight`、`time_weight`；
- 每个 anchor row 都不得包含：
  `family_id`、`regime_id`、`route_primitive`、`execution_template`、`simulator_lane_id`；
- consumption report 记录：
  anchor count、每行 schema check 结果、缺失字段、泄露字段、总体 pass/fail。

反向要求：

- 缺少 `rep_kernel_id` 时必须拒绝；
- 缺少 `member_invocations` 时必须拒绝；
- anchor row 中包含 downstream forbidden key 时必须拒绝。

前置 gate：

- 只有 `RepresentativeAnchorTable` 存在且 schema 通过时，B-line consumption 才能运行；
- 空 anchor table 或 schema-invalid anchor table 必须被拒绝。

### AC-5：回归测试自动化且可重复运行

必须覆盖：

- manifest schema validation；
- feature table completeness；
- acquisition gap routing；
- stage-gate blocking；
- selector forbidden-field rejection；
- anchor output schema；
- B-line parse smoke test。

反向测试必须能清楚报告具体 violation，
例如缺少哪个字段、哪个 invocation 有 gap、哪个 forbidden field 泄露。

### AC-6：每个 P0 invocation 只有一个确定结果

每个 P0 invocation 必须产生且只产生以下两种结果之一：

- 一个有效 `PkaFeatureRecord`，包含 12 维 measured features；
- 一个 acquisition-gap row，列出缺失 metric、invocation identity 和 source path。

如果同一个 invocation 同时产生 feature record 和 gap row，
必须拒绝为 ambiguous。

如果同一个 invocation 两者都没有产生，
pipeline 必须报错并指出 invocation 与 source file。

### AC-7：Acquisition gap 阻塞下游产物

如果任何 P0 invocation 出现在 acquisition gap report 中：

- 不得输出 `RepresentativeAnchorTable`；
- 不得输出 `BLineConsumptionReport`；
- stage-gate validator 必须拒绝继续。

### AC-8：Selector 运行时输出实际 12 维 allowlist

selector 必须输出实际用于 grouping 的 allowlist。

要求：

- allowlist 必须恰好包含 12 个 PKA 字段；
- 11 个或 13 个字段都必须拒绝；
- 包含任何非 PKA 字段都必须拒绝。

### AC-9：`kernel_invocation_id` 唯一且稳定

规则：

```text
{kernel_or_case}#{occurrence_index}
```

其中 `occurrence_index` 从 1 开始，
按 `trace_order` 排序；
如果没有 `trace_order`，按文件顺序。

要求：

- 所有 P0 invocation 的 `kernel_invocation_id` 必须唯一；
- 相同输入重复运行必须生成相同 ID；
- 重复 ID 必须 fail fast。

### AC-10：默认拒绝混合 timing unit

如果所有 invocation 都使用同一种 timing unit，
例如全部是 `duration_ns` 或全部是 `elapsed_cycles`，
weight computation 可以继续。

如果不同 invocation 混用 `duration_ns` 和 `elapsed_cycles`，
weight computation 必须 abort，
并报告冲突来源。

selector 不消费 timing，
但 coverage weight 计算必须执行该检查。

### AC-11：Source adapters 独立测试覆盖

必须分别测试：

- microbench JSON adapter；
- Rodinia artifact adapter；
- mini-transformer JSON adapter。

每个 adapter 至少验证：

- 能正确抽取 12 个 PKA 字段；
- malformed input 缺少必需 metric 时，能报告字段名和 source file。

### AC-12：Audit 输出记录 per-feature provenance

`PkaFeatureAudit` 对每个 P0 invocation 的每个 feature 都必须记录：

- metric name；
- source artifact path；
- measured status；
- 缺失原因，如果适用。

如果 audit 中某条记录标记为 `measured`，
但 `source` 为空，
validator 必须拒绝。

---

## 3. 路径边界

### 3.1 上界：最大可接受范围

上界实现包括：

- manifest builder 能读取 L1 manifest 文档，输出包含 P0 和 P1 entries 的 schema-valid JSON；
- 对所有 P0 source path 执行 path-existence pre-check；
- PKA feature extractor 支持 microbench JSON、Rodinia NCU / trace artifact、mini-transformer full / dual-source JSON；
- 每个 adapter 都输出 12 维 measured feature 和 per-feature provenance；
- PKA baseline selector 只基于 12 维 feature space grouping；
- selector 具备 forbidden-field guard；
- grouping algorithm 可以是 exact-vector、bucketed、distance-threshold，或更接近 PKA 的 PCA + k-means；
- representative selection 使用 `first_chronological`；
- 输出完整 audit artifacts；
- B-line stub consumer 能解析 anchor table 并输出 consumption report；
- 回归测试覆盖 AC-1 到 AC-12；
- 旧 selector modes 保留，但新 PKA selector 必须在独立模块中实现。

### 3.2 下界：最低可接受范围

下界实现必须至少包括：

- manifest builder 输出包含全部 10 个 P0 objects 的 `kernel_validation_manifest_l1.json`；
- manifest 能通过现有 schema 校验；
- PKA feature extractor 支持所有 P0-bearing source types；
- 如果任一 P0 invocation 无法生成 12 维 measured features，则进入 acquisition gap report，pipeline 停在 Stage 2；
- audit / gap report 清楚列出每个 P0 invocation 缺少哪些 metric；
- B-line consumer 只做 parse + required / forbidden field validation；
- AC-1 到 AC-5 的测试存在；
- Stage 2 输出的测试独立于 Stage 3 / Stage 4，可以在 pipeline 被 acquisition gap 阻塞时仍然通过。

下界不是“只实现 microbench adapter 并跑完整闭环”。

下界是：

**所有 adapter 都存在；运行时是否能进入 selector 由 measured-only stage-gate 决定。**

### 3.3 允许选择

允许：

- grouping 只使用 12 个 PKA feature；
- representative selection 使用 `first_chronological`；
- grouping algorithm 可选 exact-vector、bucketed、distance-threshold，或 PCA + k-means；
- 使用 Python standard library、现有 JSON 工具和 `pytest`；
- 以现有 `kernel_validation_manifest_schema.json` 为基础扩展。

禁止：

- 在 PKA baseline grouping key 中使用 `kernel_name`；
- 使用 `grid_dim` string、`block_dim` string、`shape_hint`、`trace_order`；
- 使用 `cross_tb_offset_coverage`、`squash_boundary_crossing_flag` 或 compression-side features；
- 使用 `family_id`、`regime_id`、`route_primitive`、`execution_template`、`simulator_lane_id`；
- 使用填充值、默认 0 或语义替代值伪装成 `measured`；
- 让 L1 B-line consumer 依赖现有 curated middle-layer bundle。

---

## 4. 五阶段实现方案

### Stage 1：Manifest Builder

输入：

- `docs/a-line-l1-validation-manifest-2026-04-26.md`
- 本地 source paths
- `kernel_validation_manifest_schema.json`

输出：

- `artifacts/a_line/l1/kernel_validation_manifest_l1.json`

职责：

- 解析 L1 manifest 文档；
- 生成机器可读 manifest；
- 校验 schema；
- 检查 P0 `local_input_path` 是否存在；
- 对多 kernel source file 只检查文件存在，具体 kernel/case 在 feature extraction 阶段校验。

### Stage 2：Feature Extractor + Audit

输入：

- manifest JSON；
- microbench JSON；
- Rodinia artifacts；
- mini-transformer JSON；
- 后续 NCU CSV / profile reports。

输出：

- `pka_feature_table_l1.json`
- `pka_feature_audit_l1.json`
- `pka_feature_audit_l1.md`
- `pka_acquisition_gap_l1.json`

职责：

- 按 source type 分发给 adapter；
- 按 invocation 展开；
- 为每个 invocation 尝试抽取 12 个 PKA feature；
- 只接受 exact Nsight metric / profiler / launch record；
- 不做 name-mapping、semantic substitution、approximate fallback；
- 12 维全 measured 时生成 `PkaFeatureRecord`；
- 任一字段缺失时生成 acquisition gap row；
- 如果任何 P0 invocation 有 gap，停止在 Stage 2。

### Stage 3：PKA Selector

前置条件：

- Stage 2 通过；
- P0 acquisition gap 清零；
- feature table 仅包含 measured PKA records。

职责：

- 只基于 12 维 feature space grouping；
- 不读取 metadata；
- 实现 forbidden-field guard；
- 输出实际 feature allowlist；
- 使用 `first_chronological` 选择 representative；
- 输出 `representative_anchor_table_l1.json`。

推荐 grouping：

- 更符合 PKA baseline 的实现是 PCA-like dimensionality reduction 后接 k-means；
- 若第一版为了工程简化使用 bucketed / threshold grouping，必须明确记录选择，并保证只使用 12 维 feature。

### Stage 4：B-line Consumption

前置条件：

- Stage 3 通过；
- anchor table 存在且 schema-valid。

职责：

- 读取 `representative_anchor_table_l1.json`；
- 校验 required fields；
- 校验 forbidden downstream fields 不存在；
- 输出 `b_line_consumption_report_l1.md`；
- 不生成 family / regime / writeback lineage；
- 不依赖现有 `artifacts/middle_layer/mini_transformer_v4/bundle.json`。

### Stage 5：Regression Tests

测试必须伴随每个 stage 实现。

覆盖：

- valid / invalid manifest；
- feature table completeness；
- acquisition gap routing；
- selector forbidden-field rejection；
- anchor table schema；
- B-line parse smoke；
- mixed timing unit rejection；
- source adapter malformed input。

---

## 5. 里程碑与依赖

### Milestone 1：Manifest Builder

目标：

- 生成机器可读 L1 输入。

阶段：

- Phase A：解析 L1 manifest 文档；
- Phase B：输出 `kernel_validation_manifest_l1.json` 并校验 schema；
- Phase C：检查 P0 source path 是否存在。

### Milestone 2：Feature Extractor and Audit

目标：

- 生成 12 维 PKA feature table 或 acquisition gap report。

阶段：

- Phase A：实现 PKA 12 维抽取逻辑；
- Phase B：实现 microbench / Rodinia / mini-transformer adapters；
- Phase C：生成 measured invocations 和 incomplete invocations；
- Phase D：生成 per-feature audit；
- Gate：如果 P0 未全部 measured，则输出 gap 并停止。

### Milestone 3：PKA Baseline Selector

目标：

- 实现不依赖禁止字段的 PKA selector。

阶段：

- Phase A：实现 12 维 feature-space grouping；
- Phase B：实现 forbidden-field guard；
- Phase C：实现 `first_chronological` representative selection；
- Phase D：输出并校验 anchor table。

### Milestone 4：B-line Consumption Check

目标：

- 证明 anchor table schema 能被 B 线 parse-and-validate。

阶段：

- Phase A：实现 B-line consumer；
- Phase B：校验 required / forbidden fields；
- Phase C：输出每行 pass/fail report。

### Milestone 5：Regression Tests

目标：

- 自动化验证所有关键约束。

阶段：

- Phase A：manifest schema tests；
- Phase B：feature completeness / gap routing tests；
- Phase C：selector forbidden-field tests；
- Phase D：anchor schema tests；
- Phase E：B-line smoke tests。

依赖关系：

```text
Milestone 1 -> Milestone 2 -> Milestone 3 -> Milestone 4
      \             \             \             /
       \------------- Milestone 5 -------------/
```

---

## 6. 任务拆分

| Task ID | 任务 | 目标 AC | 执行标签 | 依赖 |
|---|---|---|---|---|
| T1 | Manifest builder：解析 L1 manifest，生成 JSON，schema 校验，path pre-check | AC-1, AC-1.1 | coding | - |
| T2 | PKA feature extractor：实现 12 维抽取、source adapters、invocation expansion、`kernel_invocation_id`、gap routing | AC-2, AC-6, AC-9, AC-10, AC-12 | coding | T1 |
| T3 | PKA feature audit generator：输出 audit md/json 和 acquisition gap json | AC-6, AC-7, AC-12 | coding | T2 |
| T4 | Stage-gate validator：阻止 unresolved P0 gap 进入 selector / B-line；B-line 前校验 anchor table | AC-3.1, AC-4.1, AC-7 | coding | T2, T3 |
| T5 | PKA baseline selector：12 维 grouping、forbidden-field guard、representative selection、anchor export | AC-3, AC-3.1, AC-8 | coding | T2, T4 |
| T6 | B-line consumer：解析 anchor table，校验 required / forbidden fields，输出 report | AC-4, AC-4.1 | coding | T4, T5 |
| T7 | Regression tests：覆盖 manifest、feature、stage-gate、selector、anchor、B-line | AC-1 到 AC-12 | coding | T1, T2, T4, T5, T6 |
| T8 | Codex review：检查 PKA feature extraction completeness 和 adapter 边界 | AC-2, AC-6, AC-9 | analyze | T2 |
| T9 | Codex review：检查 selector forbidden-field isolation 和 12 维 algorithm coherence | AC-3, AC-8 | analyze | T5 |
| T10 | Codex review：检查 B-line interface contract 和 parse-and-validate consumer | AC-4 | analyze | T6 |

---

## 7. 已解决设计决策

- DEC-1：实现新的 `pka_baseline` selector，放在独立模块；旧 selector modes 保持不动。
- DEC-2：B-line success 定义为 parse-only；只校验 required / forbidden fields，不生成 family / regime / writeback。
- DEC-3：P0 acquisition gap 行为是输出 audit / gap 并停止；这不是失败。
- DEC-4：manifest schema 使用现有 `kernel_validation_manifest_schema.json` 字段名。
- DEC-5：grouping algorithm 采用接近 PKA 的 dimensionality reduction + k-means；若第一版简化，必须记录并保持 12 维输入纯净。
- DEC-6：representative selection 使用 `first_chronological`。
- DEC-7：`num_thread_blocks` 必须来自 profiler / launch metadata，不放宽。
- DEC-8：P1 entries 可进入 manifest，但不阻塞 stage-gate。
- DEC-9：默认拒绝 mixed timing units；weight computation 遇到混用单位必须 abort。
- DEC-10：旧 selector modes 保留，不重构、不废弃。

当前无待用户决策项。

---

## 8. 实现注意事项

代码和注释中不要出现 plan 内部术语，例如：

- `AC-`
- `Milestone`
- `Phase`
- `Step`

这些术语只属于计划文档。
代码中应使用领域命名，例如：

- `manifest_builder`
- `feature_extractor`
- `acquisition_gap`
- `stage_gate`
- `pka_selector`
- `anchor_table_validator`

L1 约束：

- selector 前必须 12 维全 measured；
- 不允许 imputation；
- 不允许 default value；
- 不允许 semantic substitution；
- `kernel_name` 只能在 metadata / audit 中出现；
- `expected_behavior_axis` 只能用于人工 sanity check；
- 任一 P0 gap 未解决时 Stage 3 / Stage 4 不得执行；
- 输出统一放在 `artifacts/a_line/l1/`；
- anchor table 不得包含 downstream forbidden fields；
- mixed timing units 默认导致 weight computation abort。

---

## 9. 风险说明

### 9.1 Acquisition Risk：CRITICAL

现有仓库 artifacts 很可能完全不包含 PKA 12 个 canonical Nsight metric names。

当前代码使用的是另一套 13 维左右的 feature vector，
例如：

- `compute_throughput_pct`
- `dram_throughput_pct`
- `ipc_active`

这些不是 PKA 12 维 feature 的子集。

因此第一轮 L1 很可能停在 Stage 2，
主要产物是：

- 10 x 12 metric availability matrix；
- `pka_feature_audit_l1.*`；
- `pka_acquisition_gap_l1.json`。

这不是计划缺陷，
而是 correctness gate 面对旧采集数据时的正确输出。

### 9.2 NCU Data Availability

必须采集的 12 个 PKA feature 包括：

- 3 个 `l1tex__t_sectors_pipe_lsu_mem_*`；
- 6 个 `smsp__inst_executed_op_*`；
- 1 个 `smsp__sass_inst_executed_op_global_atom`；
- 1 个 `smsp__inst_executed`；
- 1 个 `smsp__thread_inst_executed_per_inst_executed`；
- `num_thread_blocks` 来自 profiler / launch metadata。

每个字段必须以 exact Nsight metric name 或 profiler / launch metadata 形式出现，
才能标记为 `measured`。

### 9.3 Backend Coupling Risk：HIGH

当前 B 线 `backend_builder.py` 依赖 curated middle-layer bundle。

L1 B-line consumer 不应依赖这个 bundle。

L1 consumer 是新的 parse-and-validate 路径，
只负责证明 anchor table schema 与 B 线消费预期兼容。

---

## 10. 输出文件约定

建议输出：

- `artifacts/a_line/l1/kernel_validation_manifest_l1.json`
- `artifacts/a_line/l1/pka_feature_table_l1.json`
- `artifacts/a_line/l1/pka_feature_audit_l1.md`
- `artifacts/a_line/l1/pka_feature_audit_l1.json`
- `artifacts/a_line/l1/pka_acquisition_gap_l1.json`
- `artifacts/a_line/l1/representative_anchor_table_l1.json`
- `artifacts/a_line/l1/b_line_consumption_report_l1.md`

注意：

- 如果 Stage 2 被 acquisition gap 阻塞，则不输出 anchor table 和 B-line report；
- `pka_feature_table_l1.json` 只包含完整 measured records；
- gap records 不得混入 feature table。

---

## 11. 执行顺序

推荐执行：

1. 实现 manifest builder；
2. 实现 source adapters 和 invocation expansion；
3. 实现 12 维 measured feature extractor；
4. 实现 audit / gap 输出；
5. 实现 stage-gate validator；
6. 如果 P0 gap 清零，实现 PKA selector；
7. 实现 anchor table validator / exporter；
8. 如果 anchor table 有效，实现 B-line parse consumer；
9. 补齐回归测试；
10. 请求 Codex review：feature extraction、selector isolation、B-line contract。

如果第 4 / 5 步发现 P0 acquisition gap，
必须停止后续 selector / B-line，
并把本轮结论标记为 acquisition-gate success。

---

## 12. 简短结论

L1 RLCR 的第一目标不是“跑出 anchor”，
而是证明：

**A 线不会在输入不完整、字段不 measured、selector 偷看 metadata 的情况下继续向下游推进。**

如果现有数据采不齐 PKA 12 维，
本轮最有价值的结果就是精确的 gap report。

只有当 12 维 measured feature 全部通过后，
才允许进入 PKA selector、RepresentativeAnchorTable 和 B-line consumption。
