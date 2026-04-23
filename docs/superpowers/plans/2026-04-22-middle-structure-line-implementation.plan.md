# 中端结构线实现计划

## Goal Description

基于 [2026-04-22-middle-structure-line-implementation-design.md](/home/dyf/modern-gpu-simulator-micro-2025/.worktrees/middle-structure-layer/docs/superpowers/specs/2026-04-22-middle-structure-line-implementation-design.md)，在 `mini_transformer_v4` 上实现一条最小可运行的中端结构线，把 frontend 输出继续提升为：

`Representative Anchors -> Family -> Representative Regime -> Importance -> Lane Mapping`

该计划的目标不是完成完整后端验证，而是先把中端结构线做成：

- 有稳定对象 ID
- 有机器可读 artifacts
- 有最小 builder
- 有最小 validator / tests
- 能直接向 backend validation 提供 regime / lane 输入

并且严格遵守当前已经拍板的实现边界：

- Anchor 采用 `kernel + phase + semantic route + shape/context`
- Family 主判据是共享硬件执行模板
- Regime 偏保守、多拆一点
- `decision_weight` 允许人工判断，但必须附来源说明
- `importance_score` 采用 observed/provisional 双轨制
- writeback 设计上保留完整链，第一版实现先做 `lane -> regime`
- 下一步优先级 1 是把 `family/regime` 规则抽成 config
- rule config 第一版采用偏声明式、family-centered、single-file YAML
- config 直接带上 `lane mapping`
- `decision_weight` 在 config 中采用半结构化来源说明
- config 路径放在 `docs/family_criteria/mini_transformer_v4/`

## Acceptance Criteria

- AC-1: `mini_transformer_v4` 的中端结构线必须生成稳定的 Anchor artifacts，并采用 `kernel + phase + semantic route + shape/context` 粒度。
  - Positive Tests (expected to PASS):
    - builder 生成的 `anchors.json` 至少包含 `anchor_id / kernel_name / kernel_name_raw / phase_id / context_scope / member_invocations / route_hint / template_hint`
    - `gemm_tiled` 被拆成多个 context-aware anchors，而不是退化成单一 name-aware anchor
    - `attention_score`、`softmax_kernel`、`context_mul`、`layernorm_kernel`、`residual_add` 均有稳定 anchor 记录
  - Negative Tests (expected to FAIL):
    - `anchors.json` 中缺少 `anchor_id` 或 `member_invocations`
    - 所有 `gemm_tiled` 只输出成一个总 anchor
    - anchor 主键退化成只看 `kernel_name`

- AC-2: Family builder 必须以共享硬件执行模板为主判据生成 Family Table，并显式保留 boundary 状态。
  - Positive Tests (expected to PASS):
    - `families.json` 至少包含 `family_id / input_anchor_ids / route_primitive / hardware_template / boundary_status / importance_score`
    - `attention_score` 所在 dense object 被保留在 dense backbone family 内，但 family 备注中显式标注 boundary / weak-share 性质
    - `softmax` 和 `layernorm` 同属 `Reduction / Normalize` family
  - Negative Tests (expected to FAIL):
    - family 只按 kernel name 聚类，不输出 `hardware_template`
    - `softmax` 和 `layernorm` 被无理由拆成两个 family
    - family 结果不包含 `boundary_status`

- AC-3: Regime builder 必须把同一 Family 内会影响 backend 参数映射的对象继续拆开。
  - Positive Tests (expected to PASS):
    - `regimes.json` 至少包含 `regime_id / family_id / shape_regime / context_scope / resource_signature / regime_priority_score`
    - `attention_score` 必须成为独立 regime，而不是被并回 generic dense projection
    - `softmax` 和 `layernorm` 必须是不同 regime
    - dense backbone 内至少区分前段 projection、attention score、FFN heavy regime
  - Negative Tests (expected to FAIL):
    - Family 直接充当 final simulator object，没有 regime 层
    - `attention_score` 和普通 projection 在 regime 层被揉平
    - regime 不包含 `context_scope` 或 `resource_signature`

- AC-4: Importance 表达必须采用双轨制，显式区分 observed/measured 字段和 provisional/label-based 字段。
  - Positive Tests (expected to PASS):
    - artifacts 中同时存在 `observed_coverage_ratio / observed_time_ratio`
    - artifacts 中同时存在 `coverage_label / time_label / decision_label`
    - `importance_score` 与 `regime_priority_score` 可被计算并输出
    - 每类权重至少能追溯到 measured / derived / provisional / placeholder 之一
  - Negative Tests (expected to FAIL):
    - 只有标签，没有 observed ratio
    - 只有 observed ratio，没有 provisional labels
    - 把 provisional 分数字段当成没有来源说明的最终值

- AC-5: 中端结构线必须产出可直接被 backend 消费的 Lane Mapping，而不是只停留在 family/regime 表。
  - Positive Tests (expected to PASS):
    - `lanes.json` 至少包含 `lane_id / target_regime_id / target_family_id / parameter_direction / baseline_type / validation_metric / writeback_target`
    - 每个高优先级 regime 都能映射到明确 lane
    - lane mapping 能表达 backend 需要看的参数方向和比较基线
  - Negative Tests (expected to FAIL):
    - lane 只保留一个名称，没有 parameter direction 或 metric
    - lane 无法反查到具体 regime
    - lane mapping 与 family/regime ID 不一致

- AC-6: 中端结构线必须以 builder + artifacts + tests 的形式实现，而不是只靠 prose 文档。
  - Positive Tests (expected to PASS):
    - 存在可执行 builder，例如 `experiments/baseline_diagnosis/build_middle_layer.py`
    - builder 运行后能在 `artifacts/middle_layer/mini_transformer_v4/` 写出 `anchors / families / regimes / lanes / bundle`
    - 存在针对 builder 的测试，并能通过
  - Negative Tests (expected to FAIL):
    - 只有文档，没有生成脚本
    - 只有脚本，没有 artifacts 输出
    - builder 运行后不生成稳定 JSON 产物

- AC-7: 中端结构线必须把 `family/regime/lane` 规则收敛成一份可维护的 single-file YAML config。
  - Positive Tests (expected to PASS):
    - 存在一个位于 `docs/family_criteria/mini_transformer_v4/` 下的 YAML config 文件
    - 该 config 以 family 为顶层组织单位，而不是以零散对象平铺
    - config 能同时表达 `anchor -> family -> regime -> lane`
    - config 中显式包含 `boundary_status / boundary_notes / lane mapping / decision_weight source note`
  - Negative Tests (expected to FAIL):
    - 规则仍只分散写在 builder 常量里
    - family、regime、lane 各自维护不同 source-of-truth 文件而没有主配置入口
    - config 不包含 lane mapping 或 decision-weight 来源说明

- AC-8: 中端结构线必须提供最小 validator，保证 ID、归属关系和覆盖关系一致。
  - Positive Tests (expected to PASS):
    - 所有 `family.input_anchor_ids` 都能在 `anchors` 中找到
    - 所有 `regime.source_anchor_ids` 都能在 `anchors` 中找到
    - 所有 `lane.target_regime_id` 都能在 `regimes` 中找到
    - observed coverage/time 的汇总逻辑可被测试验证
  - Negative Tests (expected to FAIL):
    - family 引用不存在的 anchor
    - regime 引用不存在的 family
    - lane 引用不存在的 regime

- AC-9: 第一版必须保留 writeback contract，并至少实现 `lane -> regime` 的最小闭环。
  - Positive Tests (expected to PASS):
    - spec 或 artifact 中存在 `writeback_target`
    - lane 结果能稳定回指到唯一 regime
    - spec 中保留完整写回链 `lane -> regime -> family -> workload explanation`
  - Negative Tests (expected to FAIL):
    - 完全没有 writeback 字段
    - 一个 lane 对多个 regime 没有显式规则
    - spec 不说明完整写回链，只留下局部结论

## Path Boundaries

### Upper Bound (Maximum Scope)

第一版可接受的最大范围包括：

- 支持 `mini_transformer_v4` 的完整 middle-layer builder
- 支持一份 `family-centered` 的 single-file YAML rule config
- 输出 `anchors / families / regimes / lanes / bundle / markdown snapshots`
- 提供显式 `Importance Scoring Sheet`
- 提供最小 validator 和 `lane -> regime` writeback record
- 将当前 builder 中的 rule 常量进一步抽成单独 rule config

### Lower Bound (Minimum Scope)

第一版最低必须达到：

- 稳定生成 `anchors.json / families.json / regimes.json / lanes.json / bundle.json`
- 生成结果与 spec 中已拍板的 9 条原则一致
- 存在一份能表达 `family / regime / lane / decision_weight note` 的单文件 YAML config
- 有最小测试验证对象计数、ID 映射和 observed ratio 汇总
- backend 至少能直接读取 `regime_id / family_id / parameter_direction / baseline_type / validation_metric`

### Allowed Choices

- Can use:
  - Python builder
  - Single-file YAML config under `docs/family_criteria/mini_transformer_v4/`
  - JSON artifacts
  - Markdown snapshots for human-readable review
  - 显式 rule config 或先嵌入 builder 的过渡性规则
  - 半定量 label 与 observed ratio 并存
- Cannot use:
  - 把 representative compression 直接等同于 family
  - 把 family 直接等同于 final simulator object
  - 把 provisional 分数写成 final fact
  - 只新增 prose 文档而不实现 builder/artifacts/tests
  - 在代码或 artifacts 中引入“plan/AC”术语作为业务字段

## Dependencies and Sequence

### Milestones

1. Milestone 1: 固定对象边界与 artifact 结构
   - Phase A: 固定 `Anchor / Family / Regime / Lane` 四类对象字段
   - Phase B: 固定 `artifacts/middle_layer/mini_transformer_v4/` 输出结构

2. Milestone 2: 设计并落地 single-file YAML rule config
   - Phase A: 设计 `family-centered` 顶层结构
   - Phase B: 把 `anchor -> family -> regime -> lane` 显式写入 config
   - Phase C: 为 `decision_weight` 增加半结构化来源说明字段

3. Milestone 3: 实现最小 builder
   - Phase A: 实现 `anchor builder`
   - Phase B: 实现 `family builder`
   - Phase C: 实现 `regime builder`
   - Phase D: 让 builder 读取 YAML config 并实现 `lane mapper`

4. Milestone 4: 接 observed/provisional 双轨字段
   - Phase A: 从 `mini_transformer_v4_full.json` 计算 observed coverage/time
   - Phase B: 保留 `coverage_label / time_label / decision_label`
   - Phase C: 计算 `importance_score / regime_priority_score`

5. Milestone 5: 增加验证与 writeback contract
   - Phase A: 增加 ID/映射/coverage 测试
   - Phase B: 增加 config 完整性测试
   - Phase C: 增加 `writeback_target`
   - Phase D: 保证 `lane -> regime` 最小闭环成立

6. Milestone 6: 收紧 rule config
   - Phase A: 减少 builder 内部硬编码规则
   - Phase B: 为 `decision_weight` 增加更明确的 rule note / source status

## Feasibility Hints

- 先优先保证对象 ID 和引用关系稳定，再去优化权重公式。
- 先把已拍板的规则搬进 YAML config，再继续优化 builder 内部逻辑。
- `decision_weight` 第一版不要强行完全自动化，先保证“有判断且可追溯”。
- Family 可以偏聚合，但 Regime 应偏保守拆分，因为它是 backend 的直接入口对象。
- `attention_score` 和 `softmax/layernorm` 这些边界对象，是验证中端结构线是否成立的关键，不要为了压缩对象数而过早合并。
- `importance_score` 一定要保留 observed/provisional 双轨字段，否则后续很容易写过满。

## Implementation Notes

- 代码中不要把 `AC-1`、`Milestone`、`plan` 等计划术语写成业务对象字段。
- YAML config 应该是当前阶段 middle-layer 规则的 source of truth，JSON artifacts 是 builder 生成的机器输出，Markdown 主要用于审阅。
- `kernel_name` 与 `kernel_name_raw` 必须并存，避免后续 canonical 名与原始 trace 名混淆。
- writeback 设计上必须保留完整链，但第一版实现可先只做 `lane -> regime`。
