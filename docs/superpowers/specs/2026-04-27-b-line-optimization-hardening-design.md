# B 线优化加固设计

日期：2026-04-27

## 1. 文档目的

这份设计文档用于定义当前阶段 B 线的下一轮优化方案。

当前项目已经形成第一版可运行方法链：

`frontend anchor -> middle structure -> backend planning -> execution bridge -> result summary -> writeback interface`

其中 B 线位于 `frontend anchor` 和 `backend planning` 之间，负责把 A 线输出的 representative anchors 提升成 backend 能消费的结构化决策对象。

本轮优化的核心判断是：

**在 A 线正式数据集和 L1/L2 验证集尚未完全稳定之前，B 线不做最终分组优化，而做 semantic / interface hardening。**

也就是说，本轮目标不是证明当前 family / regime / lane 已经是最终结构，而是让 B 线对象系统具备稳定接收 A 线 L1 输出、稳定输出 C 线可消费对象、稳定承接 execution evidence 和 writeback 的能力。

这里的 A 线 L1 输出不是任意原始 profiling 结果，而是经过下面链路形成的对象：

`raw local result -> KernelValidationRecord -> PkaFeatureRecord -> pka_baseline selector -> RepresentativeAnchorTable`

B 线只消费 `RepresentativeAnchorTable` 及其必要 audit metadata，不直接消费混合来源原始结果，也不重新实现 PKA selector。

---

## 2. 背景与当前状态

当前 B 线已经具备第一版主干：

- `experiments/baseline_diagnosis/build_middle_layer.py`
- `docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml`
- `artifacts/middle_layer/mini_transformer_v4/`

当前 `mini_transformer_v4` 上已有：

- 9 个 anchors
- 4 个 families
- 9 个 regimes
- 9 条 lanes
- importance scoring sheet
- writeback lane-to-regime mapping
- middle-layer bundle

下游 C 线也已经具备：

- backend priority lane table
- validation worksheet
- scenario matrix
- run manifest
- baseline plan
- result summary
- writeback map

execution bridge 已经能完成 smoke execution，并将部分运行结果解析进 `backend_result_summary_v1.json`。

因此，当前 B 线已经不是文档整理层，而是一个真实的对象系统。

当前主要问题也随之变化：下一步最重要的不是“再多列几个对象”，而是保证这些对象的语义边界、字段契约、证据状态和下游回写路径足够稳定。

同时，A 线 L1 spec 已经把 L1 定位为：

- correctness gate
- feature sanity gate
- downstream interface gate

因此 B 线在本轮必须把 L1 作为 interface gate 来消费，而不是把 L1 当作 final grouping gate。

---

## 3. 目标与非目标

### 3.1 本轮目标

本轮 B 线优化要完成四件事：

1. 明确 A 线传给 B 线的 anchor 是带权重的 representative object，而不是最终语义对象。
2. 明确 family 主要承担硬件分组职责。
3. 明确 regime 是硬件 family 和算法功能角色汇合的位置。
4. 明确 lane 只负责 backend 参数方向和验证入口，不重新承担语义分类。

成功后的 B 线应能回答：

- 这个 anchor 有多重要？
- 它属于哪类硬件执行模板？
- 在这个 hardware family 内，它承担什么算法功能角色？
- 它是否需要作为单独 regime 进入 backend？
- 它应该映射到哪条 backend validation lane？
- 当前结论是 measured、derived、manual、provisional 还是 missing？
- 当前 anchor 来自 `pka_l1_compatible` 还是 `pka_complete` feature mode？
- 当前 B 线分组是否误用了 A 线 selector 禁止字段？

### 3.2 非目标

本轮不做以下事情：

- 不大幅调整 family 数量。
- 不大幅调整 regime 数量。
- 不重设全部 lane。
- 不基于当前不完整 A 线输入重新计算最终 importance 排序。
- 不把 smoke execution 结果解释成 formal validation 成功。
- 不证明 importance-guided priority 已经统计显著优于所有 baseline。
- 不把 `mini_transformer_v4` 小样本结构包装成最终普适结论。
- 不让 `kernel_name`、`grid_dim`、`block_dim`、`trace_order`、`shape_hint`、squash / batch / delta 机制字段进入 A 线 PKA baseline 主 grouping。
- 不让 B 线反向污染 A 线 selector，例如把 family / regime / lane 字段写回 selector 输入。

---

## 4. 设计路线选择

### 4.1 路线 A：立即重做 family / regime 分组

优点：

- 可以快速产出看起来更干净的新表。
- 容易把当前方法叙事重新包装成更整齐的结构。

缺点：

- A 线输入还未完全变硬，过早重分组会制造新的不稳定源。
- 当前 execution evidence 主要是 smoke，不足以支撑结构重排。
- 容易把 B 线变成主观重命名层。

### 4.2 路线 B：先做语义和接口加固，保持对象数量基本稳定

优点：

- 保留当前可运行主干。
- 降低对下游 C 线和 execution bridge 的破坏。
- 为 A 线 L1/L2 新输入预留稳定消费接口。
- 能显式区分 provisional evidence 和 formal validation evidence。

缺点：

- 短期内 family / regime 数量不会显得“焕然一新”。
- 需要补充 schema、provenance 和一致性测试。

### 4.3 路线 C：等待 A 线完整数据集后再修改 B 线

优点：

- 避免对不稳定输入做过多假设。

缺点：

- 当前 B 线和 C 线之间的 stale artifact、ID 对齐、字段语义问题会继续累积。
- A 线 L1/L2 回来后，B 线可能没有准备好消费新输入。
- execution evidence 和 writeback 仍可能落在语义不干净的对象上。

### 4.4 最终选择

本设计采用路线 B：

**先做 B 线 semantic / interface hardening，保持当前 family / regime / lane 数量基本稳定，等 A 线 L1/L2 与 formal execution evidence 回来后再做结构重排。**

---

## 5. A 线 L1 到 B 线的输入契约

### 5.1 L1 数据流边界

B 线必须按照 A 线 L1 spec 的数据流消费输入：

`raw local result -> KernelValidationRecord -> PkaFeatureRecord -> pka_baseline selector -> RepresentativeAnchorTable -> B line consumption`

其中：

- `KernelValidationRecord` 是验证集对象，用于溯源、审计和回归。
- `PkaFeatureRecord` 是 PKA baseline selector 的真正输入。
- `RepresentativeAnchorTable` 是 B 线的直接输入。

B 线不得绕过 `RepresentativeAnchorTable` 直接读取 microbench JSON、Rodinia 本地结果、mini-transformer full JSON 或 NCU CSV 来做主对象分组。

### 5.2 KernelValidationRecord 的 B 线使用方式

`KernelValidationRecord` 中的字段只允许作为 audit / provenance 信息进入 B 线。

这些字段包括：

- `validation_id`
- `dataset_level`
- `source_type`
- `benchmark_name`
- `kernel_or_case`
- `kernel_invocation_id`
- `kernel_name`
- `exec_time_or_cycle_proxy`
- `expected_behavior_axis`
- `pka_feature_vector`
- `feature_status`
- `source_path`

B 线可以使用这些字段做：

- provenance 展示
- missing / proxy 解释
- writeback 链路定位
- BLineConsumptionReport 中的错误定位

B 线不得使用这些字段替代 PKA behavior features 来做 family / regime 主分组。

特别是：

- `kernel_name` 只能用于溯源和报告。
- `expected_behavior_axis` 只能用于 sanity check。
- `source_type`、`benchmark_name`、`dataset_level` 不能作为分组依据。
- `exec_time_or_cycle_proxy` 可以用于 weight / audit，但不能作为 PKA 主特征。

### 5.3 PkaFeatureRecord 的 B 线使用方式

`PkaFeatureRecord` 是 selector 的输入，不是 B 线重新分类的原始材料。

B 线可以读取其 audit metadata，用于判断 anchor 证据强度：

- `feature_mode`
- 实际使用的 PKA 12 维字段列表
- 每个字段的 `measured / derived / proxy / missing` 状态
- selector 是否使用了 proxy 字段

如果 `feature_mode = pka_l1_compatible`，B 线基于该 anchor 形成的 family / regime / lane 结论必须默认标记为 provisional 或 pending。

只有当 `feature_mode = pka_complete` 且后续 formal validation 支撑时，相关结论才可以向 validated 状态推进。

### 5.4 RepresentativeAnchorTable 契约

B 线直接消费的 `RepresentativeAnchorTable` 至少应包含：

- `anchor_id`
- `representative_record_id`
- `member_invocations`
- `member_validation_ids`
- `member_count`
- `coverage_count`
- `coverage_weight`
- `time_weight` 或 `exec_time_or_cycle_proxy_weight`
- `feature_mode`
- `selector_used_features`
- `feature_status_summary`
- `anchor_provenance`
- `audit_metadata`

`RepresentativeAnchorTable` 不应包含：

- `family_id`
- `regime_id`
- `lane_id`
- backend priority 字段
- writeback status 字段

如果这些 downstream 字段出现在 A 线 anchor 输出中，B 线消费检查应将其标记为 schema contamination。

### 5.5 Selector 禁止字段继承

B 线 spec 继承 A 线 L1 selector 约束。

以下字段不得进入 `pka_baseline` 主 grouping：

- `kernel_name`
- `grid_dim`
- `block_dim`
- `shape_hint`
- `trace_order`
- `cross_tb_offset_coverage`
- squash boundary fields
- batch / delta 机制字段
- family / regime / lane 字段

B 线可以在消费报告中检查这些字段是否被误用于 selector，或是否污染了 anchor 输出。

---

## 6. 对象边界设计

### 6.1 Anchor

Anchor 是 A 线 representative compression 的输出对象，也是 B 线的输入对象。

Anchor 回答的问题是：

**哪些 kernel invocation 可以由同一个代表对象近似？**

Anchor 不应被理解为完整算法语义对象，也不应被理解为后端最终调参对象。

每个 anchor 进入 B 线时至少应携带：

- `anchor_id`
- `representative_record_id`
- `member_invocations`
- `member_validation_ids`
- `member_count`
- `coverage_count`
- `coverage_weight`
- `time_weight` 或 `exec_time_or_cycle_proxy_weight`
- `workload_scale`
- `feature_mode`
- `selector_used_features`
- `feature_status_summary`
- `provenance`

如果 A 线暂时不能直接提供这些字段，B 线可以过渡性重算，但必须在输出中标注来源。

`kernel_name` 可以保留在 anchor audit metadata 中，但不能作为 B 线 family 主分组的直接依据。

### 6.2 Family

Family 是 hardware execution-template grouping。

Family 回答的问题是：

**哪些 anchors 在 GPU 上共享主要执行模板、资源行为和后端调参方向？**

Family 不应主要按模型模块名、kernel 名字或具体算子名建立。

每个 family 至少应补齐：

- `family_id`
- `input_anchor_ids`
- `execution_template`
- `route_primitive`
- `resource_sensitivity`
- `expected_parameter_direction`
- `coverage_weight`
- `time_weight`
- `importance_score`
- `input_feature_mode_summary`
- `evidence_status`
- `provenance`

当前可保留的 family 主轴包括：

- `dense_tiled_compute`
- `reduction_normalize`
- `streaming_aggregation`
- `elementwise_residual`
- 预留 `layout_or_data_movement`

如果 family 的全部或部分 anchors 来自 `pka_l1_compatible`，family 的 `evidence_status` 不得高于 provisional，除非后续 formal validation 已经补足证据。

### 6.3 Regime

Regime 是 hardware family 和 algorithm function group 的交汇对象。

Regime 回答的问题是：

**在同一个 hardware family 内，这个对象是否因为算法功能、shape/context 或 resource signature 不同，而值得单独进入 backend？**

每个 regime 至少应补齐：

- `regime_id`
- `family_id`
- `source_anchor_ids`
- `algorithm_function_group`
- `shape_context`
- `resource_signature`
- `separation_reason`
- `merge_risk_if_absorbed`
- `validation_role`
- `regime_priority_score`
- `simulator_lane_id`
- `input_feature_mode_summary`
- `evidence_status`
- `provenance`

推荐的算法功能标签为：

- `primary_compute`
- `score_or_transform`
- `reduction_normalization`
- `aggregation_or_fusion`
- `elementwise_postprocess`
- `layout_or_data_movement`
- `constraint_or_bookkeeping`

不应继续把算法功能标签写成 `qkv_projection`、`ffn_expand`、`softmax` 这类强绑定具体模型模块的名称。

如果 regime 的拆分理由依赖 PKA proxy 字段或 L1 expected behavior sanity check，必须在 `separation_reason` 中明确标记，不得包装成 measured hardware evidence。

### 6.4 Lane

Lane 是 backend validation entry。

Lane 回答的问题是：

**这个 regime 应该从哪个 simulator 参数方向进入验证？**

Lane 不应承担 family 或 algorithm group 的语义分类职责。

每条 lane 至少应补齐：

- `lane_id`
- `target_regime_id`
- `target_family_id`
- `parameter_direction`
- `scenario_ids`
- `expected_signal`
- `validation_metric`
- `baseline_type`
- `writeback_target`
- `evidence_status`
- `provenance`

当前 9 条 lane 可以先保留，但需要检查每条 lane 是否真正落到可验证的硬件方向，而不是只描述模型模块。

---

## 7. Schema 与 Provenance 设计

### 7.1 字段来源

B 线输出中的关键字段必须标记来源。

推荐使用以下来源标签：

- `a_line`
- `kernel_validation_record`
- `pka_feature_record`
- `representative_anchor_table`
- `b_line_derived`
- `rule_config`
- `manual_seed`
- `execution_observed`
- `provisional`
- `unavailable`

### 7.2 证据状态

关键对象和字段必须区分证据强度。

推荐使用以下状态：

- `measured`
- `derived`
- `proxy`
- `manual`
- `provisional`
- `missing`
- `inconclusive`
- `validated`

其中 `validated` 只能来自 formal validation，不能来自 smoke execution。

### 7.3 Feature mode

B 线必须透传或汇总 A 线 L1 的 `feature_mode`。

允许值至少包括：

- `pka_l1_compatible`
- `pka_complete`

语义如下：

- `pka_l1_compatible`：selector 使用了 proxy 或部分 missing 字段，当前 anchor 可用于接口闭环和 sanity check，但不能支持最终分组结论。
- `pka_complete`：PKA 12 维字段完整且来源满足正式特征要求，但仍需要 B/C 线 formal validation 才能升级 backend 结论。

### 7.4 不允许的默认行为

B 线不应：

- 静默把缺失权重当作 0。
- 只用 anchor 名字或 kernel 名字推断 importance。
- 只用 `expected_behavior_axis` 证明 family / regime 正确。
- 只用 `member_count` 替代 `time_weight`。
- 只用平均时间替代总时间贡献。
- 把 rule config 中的人工先验包装成 measured evidence。
- 把 `pka_l1_compatible` 输出包装成 final grouping evidence。

---

## 8. 执行语义设计

当前 execution bridge 已经能生成 smoke result summary，但 B 线和 writeback 必须严格区分：

- execution success
- parse success
- smoke run success
- formal validation success

### 8.1 Smoke execution

Smoke execution 只能证明：

- command plan 可生成。
- simulator 可启动。
- stdout / stderr / metadata 可落盘。
- parser 能提取最小指标。
- result summary schema 可用。

Smoke execution 不证明：

- regime 已经 validated。
- lane 有真实 tuning gain。
- importance-guided priority 优于 baseline。
- family / regime 分组已经正确。

### 8.2 Formal validation

Formal validation 至少需要：

- 使用 formal trace / formal profile。
- 没有 smoke-only cycle cap 或 trimmed trace 语义污染。
- 具备 baseline 对照。
- 能计算 `baseline_delta` 或等价比较指标。
- result summary 明确标记 `execution_mode = validation`。
- writeback 只基于 formal result promotion 状态。

### 8.3 Writeback 约束

Writeback 层必须遵守：

- `execution_mode = smoke` 的结果只能记录，不得提升为 `validated`。
- `result_status = inconclusive` 不得提升对象状态。
- `parse_status = parsed-smoke` 不得作为 formal validation evidence。
- failed run 可以保留可见状态，但不能覆盖 unrelated successful formal evidence。

---

## 9. L1 集成设计

在完整数据集准备好之前，B 线应优先接入 A 线 L1 小验证集。

L1 的作用不是证明 compression quality 最优，而是作为：

- correctness gate
- feature sanity gate
- downstream interface gate

B 线在 L1 上需要验证：

1. `RepresentativeAnchorTable` 能被 B 线读取。
2. `RepresentativeAnchorTable` 不包含 family / regime / lane 等 downstream contamination 字段。
3. anchor 权重字段能被透传或显式标记缺失。
4. `feature_mode` 和 feature status summary 能被透传。
5. anchors 能映射到 family。
6. family 能拆出 regime。
7. regime 能映射到 lane。
8. C 线能基于这些对象生成 run manifest。
9. writeback map 能从 lane / regime 回到 family / anchor。

### 9.1 L1 P0 输入范围

B 线消费检查的第一批对象应与 A 线 L1 P0 保持一致：

- `l1_bw_32f`
- `l2_bw_32f`
- `mem_bw`
- `mem_lat`
- `shared_bw`
- `MaxFlops`
- Rodinia `nn`
- mini-transformer `gemm_tiled`
- mini-transformer `attention_score`
- mini-transformer `softmax_kernel`

第二批可选对象包括：

- `shared_lat`
- `atomic_add_bw`
- `atomic_add_lat`
- Rodinia `backprop`
- mini-transformer `context_mul`
- mini-transformer `layernorm_kernel`
- mini-transformer `residual_add`

### 9.2 BLineConsumptionReport

`artifacts/a_line/l1/b_line_consumption_report_l1.md` 至少应报告：

- 输入 anchor 数量。
- 输入 anchor 的 `feature_mode` 分布。
- 输入 anchor 的 feature status summary。
- 成功映射 family 的 anchors 数量。
- 无法映射 family 的 anchors 及原因。
- 生成或更新的 regime 数量。
- 生成或更新的 lane 数量。
- 是否发现 selector 禁止字段污染。
- 是否发现 downstream 字段污染 anchor 输出。
- writeback 链路是否能从 lane / regime 回到 family / anchor / invocation。
- 当前结论是 interface-pass、interface-fail 还是 inconclusive。

L1 通过后，再进入 L2 数据规模和稳定性验证。

---

## 10. 优先级与实施顺序

### P0：L1 contract alignment pass

先把 B 线 spec 和 builder 输入契约对齐到 A 线 L1：

- 明确 B 线直接消费 `RepresentativeAnchorTable`。
- 明确 `KernelValidationRecord` 只作为 audit / provenance。
- 明确 `PkaFeatureRecord` 只作为 selector 输入和 feature audit 来源。
- 明确 `feature_mode` 透传规则。
- 明确 selector 禁止字段和 downstream contamination 检查。

### P1：Artifact consistency pass

检查并修复 B 线和 C 线之间的 ID / schema 不一致。

重点检查：

- middle-layer artifact 中的 regime / family / lane ID
- backend priority lane table
- validation worksheet
- run manifest
- result summary
- writeback map
- validation status

当前应特别避免旧 ID 继续残留，例如旧式 `R1_projection_dense` 与当前 `R1_qkv_projection_dense` 混用。

### P2：Schema hardening pass

给 anchor / family / regime / lane 补充：

- provenance
- feature_mode summary
- evidence_status
- validation_role
- source字段
- missing / provisional 显式标记

### P3：Regime semantics pass

给每个 regime 补齐：

- `algorithm_function_group`
- `shape_context`
- `resource_signature`
- `separation_reason`
- `merge_risk_if_absorbed`

目标是让 regime 的存在理由可检查，而不是只靠名称成立。

### P4：Lane validation pass

检查每条 lane 是否具备：

- 明确 parameter direction
- 明确 expected signal
- 明确 validation metric
- 明确 writeback target

弱 lane 应先标记为 `needs-review`，不要直接删除。

### P5：A-L1 integration pass

接入 A 线 L1 representative anchors，生成 B 线消费报告。

该报告至少说明：

- 输入 anchors 数量
- feature_mode 分布
- 成功映射 family 的 anchors 数量
- 无法映射或证据不足的 anchors
- selector 禁止字段污染检查结果
- regime / lane 生成情况
- 下游 C 线最小闭环状态

### P6：Post-evidence revision

等 L1/L2 和 formal execution evidence 回来后，再决定：

- 哪些 family 应合并或拆分。
- 哪些 regime 拆得过细。
- 哪些 lane 没有信息增益。
- importance formula 是否需要调整。

---

## 11. 产物要求

本轮优化完成后，应至少更新或生成以下产物：

1. `artifacts/middle_layer/mini_transformer_v4/bundle.json`
2. `artifacts/middle_layer/mini_transformer_v4/anchors.json`
3. `artifacts/middle_layer/mini_transformer_v4/families.json`
4. `artifacts/middle_layer/mini_transformer_v4/regimes.json`
5. `artifacts/middle_layer/mini_transformer_v4/lanes.json`
6. `artifacts/middle_layer/mini_transformer_v4/importance_scoring_sheet.json`
7. `artifacts/middle_layer/mini_transformer_v4/writeback_lane_to_regime.json`
8. `experiments/backend_pipeline/results/mini_transformer_v4/backend_priority_lane_table_v1.json`
9. `experiments/backend_pipeline/results/mini_transformer_v4/backend_validation_status_v1.json`
10. `artifacts/a_line/l1/b_line_consumption_report_l1.md`

如果某些产物在本轮只完成 schema 加固而未接入 L1，应在 metadata 中明确标注 `pending_l1_input`。

B 线本轮可以读取但不负责生成以下 A 线 L1 产物：

- `artifacts/a_line/l1/kernel_validation_manifest_l1.json`
- `artifacts/a_line/l1/pka_feature_table_l1.json`
- `artifacts/a_line/l1/pka_feature_audit_l1.json`
- `artifacts/a_line/l1/representative_anchor_table_l1.json`

如果这些上游文件不存在，B 线应输出 `pending_l1_input`，而不是退回读取原始 profiling 文件。

---

## 12. 验收标准

本轮 B 线优化完成后，必须满足：

1. `anchor -> family -> regime -> lane` 每层对象职责清楚。
2. 所有关键权重字段都有 provenance。
3. 缺失字段显式标记为 `missing` 或 `provisional`。
4. 当前 middle-layer 和 backend artifacts 不再存在 regime / family / lane ID 冲突。
5. `execution_mode = smoke` 的结果不会提升为 `validated`。
6. B 线可以消费 A 线 L1 representative anchor table。
7. C 线仍能生成 run manifest、scenario matrix、result summary 和 writeback map。
8. 每个 regime 都有算法功能角色、硬件 family、拆分理由和 merge risk。
9. 每条 lane 都有参数方向、预期信号、验证指标和回写目标。
10. 所有未被 formal validation 支撑的结论都标记为 `provisional`、`pending` 或 `inconclusive`。
11. B 线消费检查能报告 `feature_mode` 分布和 feature status summary。
12. B 线消费检查能发现 `kernel_name`、`grid_dim`、`block_dim`、family / regime / lane 等禁止字段污染。
13. `RepresentativeAnchorTable` 中不得出现 downstream 字段；若出现，B 线消费报告必须标记 schema contamination。
14. `pka_l1_compatible` 输入只能形成 interface-pass 或 provisional 结论，不能形成 final grouping 结论。

---

## 13. 风险与缓解

### 13.1 风险：过早固化当前 B 线结构

缓解：

- 保持当前对象数量基本稳定，但不把它们标记为 final。
- 对所有未验证字段添加 provenance 和 evidence status。

### 13.2 风险：schema 加固影响 C 线消费

缓解：

- 新字段应优先向后兼容。
- C 线读取逻辑应允许未知字段存在。
- 核心 ID 字段保持稳定。

### 13.3 风险：smoke result 被误用为 validation evidence

缓解：

- 在 result summary 和 writeback 中强制区分 `execution_mode`。
- 测试覆盖 smoke 不得 promotion。

### 13.4 风险：A 线 L1 输入变化导致 B 线重工

缓解：

- B 线消费 L1 时先做 `RepresentativeAnchorTable` schema adapter。
- 对缺失字段显式标记，不静默推断。
- 将 L1 作为 interface gate，而不是 final grouping gate。

### 13.5 风险：A 线 selector 被 downstream 字段污染

缓解：

- B 线消费报告检查 anchor 输出中是否含有 family / regime / lane 字段。
- L1 regression tests 覆盖 selector forbidden fields。
- B 线只消费 selector 输出，不向 selector 输入写回 downstream metadata。

---

## 14. 后续衔接

这份 spec 通过后，下一步应进入 implementation plan。

实施计划应拆成以下工作包：

1. L1 input contract adapter。
2. artifact consistency 检查与 ID 对齐。
3. middle-layer schema 扩展。
4. builder provenance / evidence status 输出。
5. feature_mode 和 feature status summary 透传。
6. regime semantic fields 补齐。
7. lane validation fields 补齐。
8. backend artifact 兼容性检查。
9. smoke / validation promotion 测试。
10. A-L1 B 线消费报告。

在 implementation plan 之前，不应直接修改 B 线规则数量或重排 family / regime 结构。
