# 后端执行线 V1 结构化实施计划

## Goal Description

在当前 `backend_builder` 已经能够输出后端结构对象的基础上，继续实现后端执行线第一版，使项目具备：

- 从 `priority / lane / validation worksheet` 生成可执行 `run manifest`
- 以统一格式接收执行结果形成 `result summary`
- 将执行结果回写到 `regime / family / anchor` 的 `writeback` 链路

本计划只覆盖：

- `run manifest -> result summary -> writeback`

不覆盖：

- 真实 frontend compression 重建
- family / regime 判据重写
- 完整 simulator 自动执行系统

本计划中的量化项，如：

- `Top-3 families`
- `Top-4 regimes`
- 每对象 `1-2` 个 scenarios

均作为第一版建议目标与趋势，不作为必须严格命中的硬性验收数值。

---

## Acceptance Criteria

- AC-1：系统能够从已有后端结构输出生成一份完整的执行规划产物集合。
  - Positive Tests (expected to PASS):
    - 给定 `backend_priority_lane_table_v1.json` 和 `backend_validation_worksheet_v1.json`，能够生成 `backend_run_manifest_v1.json`
    - 输出中同时包含 `importance-guided`、`time-only`、`name-based`、`no-priority` 四种策略
    - 每条 manifest 行都包含稳定字段：`run_id`、`object_level`、`object_id`、`priority_source`、`priority_rank`、`simulator_lane_id`、`parameter_scenario_id`
  - Negative Tests (expected to FAIL):
    - 缺少 `priority_source` 或 `parameter_scenario_id` 的条目不能通过生成校验
    - 仅输出一种 baseline 策略时，应判定为不满足该标准

- AC-2：执行规划必须遵守当前已经确认的第一轮策略边界。
  - Positive Tests (expected to PASS):
    - 规划先进行 `family-level` 预筛选，再展开 `regime-level`
    - `layernorm_kernel` 对应对象被标记为 `review-object`
    - `residual_add` 对应对象被标记为 `constraint-object`
    - `no-priority` 使用固定原始顺序，而不是随机顺序
  - Negative Tests (expected to FAIL):
    - 直接跳过 family 预筛选，全量展开所有 regimes，应判定越界
    - `layernorm_kernel` 被当作与主对象完全等价的默认主对象，应判定不符合当前执行策略
    - `no-priority` 引入随机打乱而未固定顺序，应判定不符合当前 baseline 定义

- AC-3：系统能够定义并落地第一版 `result summary` 接口。
  - Positive Tests (expected to PASS):
    - 存在结构化 `backend_result_summary_v1.json`，并包含 `run_id`、`baseline_delta`、`sensitivity_score`、`result_status`
    - `result_status` 至少支持 `success`、`weak`、`inconclusive`、`failed`
    - 第一轮允许该文件先为空模板或样例数据，但字段必须稳定
  - Negative Tests (expected to FAIL):
    - 结果文件只是一段 prose 说明，而非结构化对象，应判定失败
    - 缺少 `run_id` 导致结果无法与 manifest 对齐，应判定失败

- AC-4：系统能够将执行结果回写到结构层，而不是停留在局部结果层。
  - Positive Tests (expected to PASS):
    - 给定 `run manifest`、`result summary` 和已有 `writeback map`，能够生成 `backend_writeback_updates_v1.json`
    - 回写结果至少包含：`regime_id`、`family_id`、`rep_kernel_ids`、`decision_update`、`validation_status_update`
    - `review-needed` 对象默认可以保留 `keep-review`，而不是被强行结论化
  - Negative Tests (expected to FAIL):
    - 回写结果无法追溯到 `run_id` 时，应判定失败
    - 执行结果只更新单个 regime，而不能映射到 family / anchor，应判定失败

- AC-5：执行线实现必须保持与当前方法链兼容，而不重新定义前端或中间结构层。
  - Positive Tests (expected to PASS):
    - 执行线只消费 `backend_builder` 产物或其下游文件
    - 代码中不重新实现 frontend compression 或 family 判据逻辑
    - 结构层与执行层职责保持分离：`builder` 负责对象，`planner/writeback` 负责执行
  - Negative Tests (expected to FAIL):
    - 在 `plan_backend_validation.py` 中重新生成 family assignment，应判定越界
    - 在执行线代码中直接把 representative anchors 当成 final simulator object，应判定越界

---

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

在不引入过度工程化的前提下，完成：

- `plan_backend_validation.py`
- `apply_backend_writeback.py`
- 对应测试
- 一套可运行的 `backend_run_manifest_v1.json`
- 一套稳定的 `backend_result_summary_v1.json` 模板/样例
- 一套稳定的 `backend_writeback_updates_v1.json`

并确保：

- baseline 对比可复用
- review / constraint / main-object 三类角色显式保留
- 结果可回写到 family / regime / anchor

### Lower Bound (Minimum Acceptable Scope)

至少完成：

- `plan_backend_validation.py`
- `backend_run_manifest_v1.json`
- `backend_result_summary_v1.json` schema/template
- `apply_backend_writeback.py` 的最小回写逻辑

并保证：

- manifest 可生成
- result summary 接口固定
- writeback 能从 `run_id` 回到 `regime_id / family_id`

### Allowed Choices

- Can use:
  - Python CLI
  - JSON 文件作为主输出
  - 现有 `experiments/backend_pipeline/` 目录
  - 样例数据或空模板作为第一轮 `result summary`
- Cannot use:
  - 重写 frontend compression
  - 重写 family / regime canon
  - 在本轮把真实 simulator 自动执行绑定为硬前置条件
  - 将 provisional 权重包装成 final fact

---

## Feasibility Hints and Suggestions

### Conceptual Approach

推荐的实现方式是三段式：

1. `backend_builder.py`
   - 已有
   - 负责生成结构对象

2. `plan_backend_validation.py`
   - 新增
   - 读取 `priority_lane_table + validation_worksheet`
   - 输出 `run_manifest + scenario_matrix + baseline_plan`

3. `apply_backend_writeback.py`
   - 新增
   - 读取 `run_manifest + result_summary + writeback_map`
   - 输出 `writeback_updates + validation_status`

### Relevant References

- `/home/dyf/modern-gpu-simulator-micro-2025/experiments/backend_pipeline/backend_builder.py`
- `/home/dyf/modern-gpu-simulator-micro-2025/experiments/backend_pipeline/build_backend_outputs.py`
- `/home/dyf/modern-gpu-simulator-micro-2025/experiments/backend_pipeline/tests/test_backend_builder.py`
- `/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/specs/2026-04-22-backend-execution-line-design.md`
- `/home/dyf/modern-gpu-simulator-micro-2025/docs/parallel-session-briefing-instructions-2026-04-22.md`

---

## Dependencies and Sequence

### Milestones

1. 执行规划层落地
   - Phase A: 实现 `plan_backend_validation.py`
   - Phase B: 生成 `backend_run_manifest_v1.json`、`backend_scenario_matrix_v1.json`、`backend_baseline_plan_v1.json`

2. 结果接口固定
   - Phase A: 定义 `backend_result_summary_v1.json` 的字段
   - Phase B: 生成空模板或样例结果文件

3. 回写层落地
   - Phase A: 实现 `apply_backend_writeback.py`
   - Phase B: 生成 `backend_writeback_updates_v1.json` 与 `backend_validation_status_v1.json`

4. 测试与样例闭环
   - Phase A: 新增 planner / writeback 测试
   - Phase B: 跑通第一版样例 outputs

---

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 设计并实现 `plan_backend_validation.py` 的输入输出和选择规则 | AC-1, AC-2, AC-5 | coding | - |
| task2 | 新增 `test_plan_backend_validation.py`，覆盖 baseline、角色与顺序策略 | AC-1, AC-2 | coding | task1 |
| task3 | 固定 `backend_result_summary_v1.json` schema/template | AC-3 | coding | task1 |
| task4 | 设计并实现 `apply_backend_writeback.py` | AC-4, AC-5 | coding | task3 |
| task5 | 新增 `test_apply_backend_writeback.py`，覆盖 review / constraint / writeback 行为 | AC-4 | coding | task4 |
| task6 | 生成第一版样例执行产物并人工检查结构一致性 | AC-1, AC-3, AC-4 | coding | task2, task5 |

---

## Claude-Codex Deliberation

### Agreements

- 第一轮应优先做执行协议，而不是直接绑定完整 simulator 自动执行。
- 第一轮 baseline 必须可重复，因此 `no-priority` 不应使用随机顺序。
- `family -> regime` 的两级执行策略比纯 family-level 或纯 regime-level 更适合当前阶段。

### Resolved Disagreements

- 量化预算项：
  - 早期候选：是否把 `Top-3 families / Top-4 regimes / 1-2 scenarios` 写成硬性验收值
  - 最终选择：作为第一版建议目标与趋势，而非硬性 AC
  - 原因：当前执行线仍处于原型期，固定结构比固定数值更重要

### Convergence Status

- Final Status: `converged`

---

## Pending User Decisions

- DEC-1: 量化预算项的约束方式
  - Claude Position: 作为建议目标/趋势
  - Codex Position: 作为建议目标/趋势
  - Tradeoff Summary: 有利于保留执行线迭代空间，避免把原型期预算写成硬性契约
  - Decision Status: `建议目标 / 趋势`

- DEC-2: 计划语言
  - Claude Position: 中文
  - Codex Position: 中文
  - Tradeoff Summary: 与当前项目文档主语言一致，集成成本最低
  - Decision Status: `中文`

---

## Implementation Notes

### Code Style Requirements

- 实现代码与注释中不要出现 `AC-`、`Milestone`、`Phase` 等计划术语
- 这些术语只属于计划文件，不属于最终代码
- 代码中应使用领域语义命名，例如：
  - `validation_role`
  - `priority_source`
  - `review_status`
  - `writeback_update`

### Current Repository Constraint

- 当前若找不到原始草稿，可将本计划作为主工作区可见版本继续使用
- 后续如草稿回流，可再做差异核对，但不影响本计划作为当前实现基准使用
