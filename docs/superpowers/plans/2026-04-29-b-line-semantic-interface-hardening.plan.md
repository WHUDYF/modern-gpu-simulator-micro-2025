# B 线语义与接口加固实施计划

## Goal Description

基于当前 B 线已经形成的第一版对象系统，完成一轮 `semantic / interface hardening`。

本计划的 canonical spec 是：

- `docs/superpowers/specs/2026-04-27-b-line-optimization-hardening-design.md`

方法论边界摘要是：

- `docs/superpowers/specs/2026-04-28-b-line-semantic-interface-hardening-design.md`

本轮目标不是重做 family / regime / lane，也不是基于当前不完整 A 线输入重排最终 importance，而是让 B 线稳定承接 A 线 L1/L2 输出，并继续向 C 线输出可消费的结构化对象。

本轮应保持当前 `mini_transformer_v4` 的对象数量基本稳定：

- 9 个 anchors
- 4 个 families
- 9 个 regimes
- 9 条 lanes

本轮实施后的目标链路是：

`RepresentativeAnchorTable -> B-line consumer -> anchors/families/regimes/lanes/bundle -> backend planning`

其中 `RepresentativeAnchorTable` 是 B 线直接消费的 A 线输出。B 线不得绕过它直接读取原始 profiling 文件来做主对象分组。

---

## Acceptance Criteria

### AC-1: B 线必须以 `RepresentativeAnchorTable` 作为直接输入契约

Positive Tests:

- B 线 consumer 能读取 A 线 L1 `RepresentativeAnchorTable` 或等价 fixture。
- consumer 能验证每行至少包含 anchor ID、representative ID、member list、coverage、time weight、feature mode、feature status summary 和 provenance。
- 当 A 线 L1 输入缺失时，B 线输出 `pending_l1_input` 或等价状态，而不是退回读取原始 profiling 文件。

Negative Tests:

- 如果 consumer 直接读取 microbench JSON、Rodinia 本地结果、mini-transformer full JSON 或 NCU CSV 作为主分组输入，应判定越界。
- 如果 anchor 输出包含 `family_id`、`regime_id`、`lane_id`、backend priority 或 writeback status，应标记 schema contamination。
- 如果缺少权重字段但被静默当作 0，应判定失败。

### AC-2: Anchor 权重与 provenance 必须显式进入 B 线 artifacts

Positive Tests:

- `anchors.json` 或 `bundle.json` 中每个 anchor 至少包含 `member_count`、`member_invocations`、`time_weight` 或等价字段。
- 如果可以可靠获得工作量尺度，应包含 `total_dynamic_insts`、`avg_dynamic_insts` 或等价字段。
- 每个关键字段都有来源标记，例如 `a_line`、`representative_anchor_table`、`b_line_derived`、`rule_config`、`provisional`、`missing`。
- `time_weight` 被记录为第一版 importance 的主信号。

Negative Tests:

- 只用 anchor 名字或 kernel 名字推断 importance，应判定失败。
- 只用 `member_count` 替代 `time_weight`，应判定失败。
- 只用平均时间替代总时间贡献，应判定失败。

### AC-3: Family 必须明确表达硬件执行模板分组

Positive Tests:

- `families.json` 中每个 family 包含 `hardware_group_basis` 或等价字段。
- family 的主依据落在 `execution_template`、`route_primitive` 或共享执行模式上。
- `resource_sensitivity` 和 `expected_parameter_direction` 作为解释字段或 lane 依据保留。
- 当前 family 数量基本不变，除非发现明确 schema bug。

Negative Tests:

- family 主要按模型模块名、kernel 名字或具体算子名建立，应判定失败。
- 在没有 formal execution evidence 的情况下按资源瓶颈大幅重分 family，应判定越界。
- 为了让 family 看起来更复杂而强行合并或拆分，应判定越界。

### AC-4: Regime 必须承接算法功能角色与拆分理由

Positive Tests:

- `regimes.json` 中每个 regime 包含 `algorithm_function_group`。
- 每个 regime 包含 `separation_reason` 或等价拆分理由。
- 每个 regime 包含 `validation_role`、`shape_context`、`resource_signature` 或等价字段。
- 当前 9-regime 结构基本保持稳定。

Negative Tests:

- `algorithm_function_group` 写成 `qkv_projection`、`ffn_expand`、`softmax` 这类强绑定具体模型模块的标签，应判定不合格。
- regime 只复述 family 名称，没有拆分理由，应判定失败。
- 为了追求更细粒度继续扩 regime，应判定越界，除非 spec 明确更新。

### AC-5: Lane 必须保持 backend 参数方向入口

Positive Tests:

- `lanes.json` 中每条 lane 包含 `parameter_direction`。
- 每条 lane 能反查到唯一 regime 和 family。
- 每条 lane 包含 `expected_signal`、`validation_metric`、`writeback_target` 或等价字段。
- 弱证据 lane 可标记为 `needs-review`、`provisional` 或 `pending`，但不应静默删除。

Negative Tests:

- lane 变成算法分组标签，应判定失败。
- lane 只是 family 的别名，应判定失败。
- lane 只描述模型模块，不落到 backend 参数方向，应判定失败。

### AC-6: B 线必须继承 A 线 selector 禁止字段约束

Positive Tests:

- consumer 或测试能检查 `kernel_name`、`grid_dim`、`block_dim`、`shape_hint`、`trace_order`、`cross_tb_offset_coverage`、squash/batch/delta 机制字段是否污染主 grouping。
- consumer 或测试能检查 family / regime / lane 字段是否污染 A 线 anchor 输出。
- `feature_mode` 和 feature status summary 能被透传或汇总。

Negative Tests:

- B 线把 selector 禁止字段当成主分组证据，应判定失败。
- B 线把 `pka_l1_compatible` 输出包装成 final grouping evidence，应判定失败。

### AC-7: C 线消费兼容性必须保持

Positive Tests:

- middle-layer `bundle.json` 仍能被 backend builder 或 backend planning 消费。
- backend priority lane table、validation worksheet、run manifest 和 writeback map 仍能生成。
- 新增字段不破坏旧读取路径。

Negative Tests:

- schema 加固导致 C 线无法读取核心 ID 字段，应判定失败。
- 旧式 regime / family / lane ID 和当前 ID 混用，应判定失败。

### AC-8: Tests 必须覆盖本轮语义边界

Positive Tests:

- 增加或更新 B 线 consumer 测试。
- 增加或更新 middle-layer builder 测试。
- 增加或更新 C 线兼容性测试。
- 相关测试通过。

Negative Tests:

- 只更新 artifacts，不更新测试，应判定不完整。
- 只更新文档，不更新 builder / artifacts / tests，应判定不完整。

---

## Path Boundaries

### Upper Bound

本轮最多完成：

- L1 `RepresentativeAnchorTable` consumer / adapter
- B 线 artifacts schema 加固
- YAML rule config 字段扩展
- builder 字段透传或过渡计算
- `feature_mode` / provenance / evidence status 透传
- regime / lane 语义字段补齐
- BLineConsumptionReport
- B/C 线兼容性测试

### Lower Bound

本轮至少完成：

- 明确 B 线只消费 `RepresentativeAnchorTable`
- anchor 权重字段进入 artifacts 或显式标记 `missing / provisional`
- family 有硬件分组依据
- regime 有算法功能角色
- lane 有参数方向
- 不改变当前对象数量
- 相关测试通过

### Cannot Do

本轮不得做：

- 重写 A 线 selector
- 复刻完整 PKA / STEM / ROOT
- 大改 family / regime / lane 数量
- 基于当前输入重排最终 importance
- 把 smoke execution 当成 formal validation
- 把 downstream 字段写回 A 线 selector 输入
- 绕过 `RepresentativeAnchorTable` 读取原始 profiling 文件做主分组

---

## Dependencies and Sequence

### Milestone 1: Spec and Artifact Baseline Check

检查当前 spec、checklist、YAML、builder、artifacts 和 tests 的状态。

Target files:

- `docs/superpowers/specs/2026-04-27-b-line-optimization-hardening-design.md`
- `docs/superpowers/specs/2026-04-28-b-line-semantic-interface-hardening-design.md`
- `docs/b-line-revision-checklist-2026-04-25.md`
- `docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml`
- `experiments/baseline_diagnosis/build_middle_layer.py`
- `artifacts/middle_layer/mini_transformer_v4/`

Expected output:

- 确认当前对象数量和 ID。
- 确认当前 artifacts 中哪些字段已存在，哪些缺失。
- 不做结构重排。

### Milestone 2: L1 Input Contract Adapter

实现或补齐 B 线对 A 线 L1 `RepresentativeAnchorTable` 的消费检查。

Likely target files:

- `experiments/baseline_diagnosis/b_line_consumer_l1.py`
- `experiments/baseline_diagnosis/test_l1_regression.py`

Required behavior:

- 校验 required fields。
- 校验 field types。
- 检查 downstream contamination。
- 检查 selector forbidden fields。
- 输出 `pending_l1_input`、`interface-pass`、`interface-fail` 或 `inconclusive`。

### Milestone 3: Middle-Layer Schema Hardening

扩展 YAML 和 builder，使 artifacts 能表达本轮新增语义字段。

Likely target files:

- `docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml`
- `experiments/baseline_diagnosis/build_middle_layer.py`

Required additions:

- `hardware_group_basis`
- `algorithm_function_group`
- `anchor_weight_source`
- `regime_split_rationale`
- `lane_parameter_direction`
- `feature_mode_summary`
- `evidence_status`
- `provenance`

Implementation rule:

- 缺失字段必须显式标记，不得静默默认。
- 新字段应向后兼容。
- 不改变对象数量。

### Milestone 4: Artifact Regeneration and Consistency Check

重新生成 B 线 artifacts，并检查 C 线消费兼容性。

Target artifacts:

- `artifacts/middle_layer/mini_transformer_v4/anchors.json`
- `artifacts/middle_layer/mini_transformer_v4/families.json`
- `artifacts/middle_layer/mini_transformer_v4/regimes.json`
- `artifacts/middle_layer/mini_transformer_v4/lanes.json`
- `artifacts/middle_layer/mini_transformer_v4/importance_scoring_sheet.json`
- `artifacts/middle_layer/mini_transformer_v4/writeback_lane_to_regime.json`
- `artifacts/middle_layer/mini_transformer_v4/bundle.json`

Required checks:

- anchors 有权重和 provenance。
- families 有 hardware grouping basis。
- regimes 有 algorithm function group 和 split rationale。
- lanes 有 parameter direction。
- 所有 ID 引用一致。

### Milestone 5: Backend Compatibility Pass

确保 C 线仍能消费 B 线输出。

Likely target files:

- `experiments/backend_pipeline/build_backend_outputs.py`
- `experiments/backend_pipeline/backend_builder.py`
- `experiments/backend_pipeline/plan_backend_validation.py`
- `experiments/backend_pipeline/apply_backend_writeback.py`

Expected behavior:

- 新增字段不会破坏 backend builder。
- backend priority lane table 仍能生成。
- validation worksheet、scenario matrix、run manifest、writeback map 仍能生成。
- schema 变化不应引入 stale ID。

### Milestone 6: Tests

更新或新增测试覆盖本轮语义边界。

Likely target files:

- `tests/test_build_middle_layer.py`
- `experiments/baseline_diagnosis/test_l1_regression.py`
- `experiments/baseline_diagnosis/tests/`
- `experiments/backend_pipeline/tests/`

Required checks:

- L1 consumer required fields。
- downstream contamination detection。
- selector forbidden fields detection。
- middle-layer artifacts 新字段存在。
- B/C 线核心 pipeline 可运行。

---

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---|---|---|---|---|
| task1 | 审查当前 artifacts、YAML 和 builder 字段缺口 | AC-1..AC-8 | analyze | - |
| task2 | 实现或补齐 L1 `RepresentativeAnchorTable` consumer 校验 | AC-1, AC-6 | coding | task1 |
| task3 | 扩展 YAML schema，加入 hardware / algorithm / lane / provenance 字段 | AC-2, AC-3, AC-4, AC-5 | coding | task1 |
| task4 | 修改 middle-layer builder，透传或生成新字段 | AC-2, AC-3, AC-4, AC-5 | coding | task3 |
| task5 | 重新生成 middle-layer artifacts 并检查 ID 一致性 | AC-2..AC-7 | coding | task4 |
| task6 | 补 B 线消费报告或等价 audit 输出 | AC-1, AC-6 | coding | task2, task5 |
| task7 | 检查并修复 C 线消费兼容性 | AC-7 | coding | task5 |
| task8 | 补充测试并运行相关回归 | AC-8 | coding | task2, task4, task7 |

---

## Feasibility Hints

- 优先保证字段来源和证据状态清楚，再考虑是否优化字段命名。
- 如果 A 线 L1 产物缺失，优先输出 `pending_l1_input`，不要退回读取原始文件。
- 如果某个字段当前只能由 rule config 或人工规则给出，标记为 `rule_config` 或 `manual`，不要写成 `measured`。
- 如果某条 lane 的参数方向不够硬，先标记为 `needs-review`，不要删除。
- 如果 C 线不认识新增字段，应让读取逻辑忽略未知字段，而不是删除新增字段。

---

## Implementation Notes

- 代码中不要写 `AC-1`、`Milestone`、`task1` 等计划术语。
- JSON 字段名使用英文。
- 文档、报告、case note 使用中文。
- 新字段尽量向后兼容，避免破坏已有 artifacts 消费方。
- `validated` 只能来自 formal validation，不能来自 smoke execution。
- `pka_l1_compatible` 只能支撑 interface-pass 或 provisional 结论。

---

## First-Version Success Definition

本轮成功不是：

- 得到新的 family 数量
- 得到新的 regime 排序
- 宣称当前结构已被最终验证

本轮成功是：

- B 线明确只消费 A 线 representative anchor 输出
- 权重、provenance、feature status 和 evidence status 能进入对象系统
- family / regime / lane 的职责边界在 artifacts 中可检查
- C 线能继续消费 B 线输出
- superpower 后续可以在明确边界内继续执行，而不会把本轮任务误解成最终分组优化
