# mini_transformer_v4 后端输出层 Schema（第一版）

日期：2026-04-22

## 1. 文档目的

这份文档用于把当前方法线中 `Representative Anchor Table` 之后的后端输出层固定下来。

这里的“后端输出层”不是指：

- Web backend
- 服务接口层
- 数据库系统

而是指：

**compression 之后，面向 simulator decision layer 的结构化输出层。**

它的目标是把：

`anchors -> family -> regime -> priority -> validation -> writeback`

这条链路正式定义成一组稳定对象。

---

## 2. 为什么现在必须定义后端输出层

当前我们已经有：

- `Representative Anchor Table`
- `Family / Regime Table`
- `importance ratio` 的第一版定义
- `importance ratio` 的验证计划

但这些内容目前仍然分散在多份文档里。

如果没有统一的后端输出层 schema，就会出现：

- family / regime 表能写，但不能稳定进入 validation
- importance score 能定义，但不能稳定进入 simulator lane
- simulator 结果能观察，但不能稳定回写到 family / anchor / workload

因此，当前最需要的是：

**把后端真正要输出哪些表、这些表之间如何引用、哪些字段必须可回写，一次性固定下来。**

---

## 3. 后端输出层在总方法线中的位置

当前推荐的对象流如下：

`Representative Anchor Table -> Family Table -> Regime Table -> Priority & Lane Table -> Validation Worksheet -> Writeback Map`

其中：

- `Anchor Table` 是输入接口
- `Family / Regime` 是结构对象层
- `Priority & Lane` 是决策映射层
- `Validation Worksheet` 是验证协议层
- `Writeback Map` 是结果回写层

---

## 4. 设计原则

### 原则 1：后端输出层只吃 anchor 接口，不吃前端内部实现

后端层的输入对象必须固定为：

**phase-annotated representative anchors**

而不是：

- PKA 内部 cluster 过程
- 前端特征工程细节

这样后续当前占位版 anchors 替换成真实 PKA 输出时，后端层无需重写。

### 原则 2：后端输出层必须区分“结构对象”和“验证对象”

- `Family / Regime` 解决对象是否成立
- `Priority / Validation` 解决对象是否有用

不能把这两层混成一张表。

### 原则 3：后端输出层必须显式支持回写

每个 regime 的 simulator 结果最终必须能回写到：

- `family_id`
- `rep_kernel_id`
- `member_invocations`

否则 importance 和 tuning 结果无法回到 workload 解释。

### 原则 4：第一版优先输出结构化 JSON，再决定是否同步生成 Markdown

因为当前最重要的是：

- 可引用
- 可排序
- 可比较
- 可继续计算

所以第一版后端主输出应优先是结构化对象，而不是只写 prose。

---

## 5. 前置条件

后端输出层开始稳定实现前，必须满足两个前提。

### 5.1 有一份唯一的 Anchor Table

至少字段齐全：

- `rep_kernel_id`
- `kernel_name`
- `cluster_id`
- `member_invocations`
- `coverage_count`
- `coverage_weight`
- `time_weight`
- `phase_id`
- `trace_order_summary`
- `shape_hint_summary`
- `route_hint`
- `template_hint`

### 5.2 有一份唯一的 family canon

也就是必须先定清楚：

- 哪些 anchors 进入哪个 family
- 哪些对象仍然是 boundary / unresolved
- 哪些对象已经能稳定进入 regime

如果 family canon 仍然在冲突状态，后端输出层只能生成临时表，不能生成稳定决策表。

---

## 6. 后端输出层的 5 张核心表

## 6.1 Family Table

### 角色

把 anchors 提升成共享机制层的组织对象。

### 最小字段

| 字段 | 含义 |
|---|---|
| `family_id` | family 标识 |
| `phase_scope` | 覆盖 phase |
| `route_primitive` | 路径角色 |
| `hardware_template` | 执行模板 |
| `member_rep_kernels` | 包含的 anchor 列表 |
| `member_count` | anchor 数量 |
| `canonical_status` | stable / weak-share / absorbed-with-note |
| `boundary_status` | strong share / weak share / unresolved |
| `boundary_notes` | 边界说明 |
| `shape_regime_summary` | family 内 shape 摘要 |
| `resource_signature_summary` | 主导资源行为摘要 |
| `coverage_weight` | family 覆盖权重 |
| `time_weight` | family 时间权重 |
| `decision_weight` | family 决策权重 |
| `importance_score` | family 综合重要性 |
| `priority_class` | High / Medium / Low |
| `recommended_tuning_target` | 推荐调参方向 |
| `notes` | 备注 |

### 输出作用

- 作为 importance ratio 的第一层载体
- 作为 family-level tuning target 的直接输入

---

## 6.2 Regime Table

### 角色

把 family 进一步拆成真正可进入 simulator lane 的执行对象。

### 最小字段

| 字段 | 含义 |
|---|---|
| `regime_id` | regime 标识 |
| `family_id` | 所属 family |
| `phase_id` | 所属 phase |
| `route_primitive` | 路径角色 |
| `hardware_template` | 执行模板 |
| `source_rep_kernels` | 来源 anchor |
| `canonical_status` | stable / singleton / review-needed |
| `shape_regime` | shape 区间 |
| `context_scope` | 工作负载上下文 |
| `resource_signature` | 主导资源机制 |
| `coverage_weight` | regime 覆盖权重 |
| `time_weight` | regime 时间权重 |
| `family_importance_score` | 上层 family importance |
| `local_decision_weight` | regime 局部调参权重 |
| `regime_priority_score` | regime 综合优先级 |
| `simulator_lane_id` | 进入哪个 lane |
| `validation_status` | pending / selected / validated |
| `notes` | 备注 |

### 输出作用

- 作为 simulator perturbation 的直接入口对象
- 作为 regime-level priority 的排序表

---

## 6.3 Priority & Lane Table

### 角色

把 importance 和 regime priority 映射成真正的 simulator-side action。

### 为什么需要单独成表

因为：

- `Family Table` 主要是结构层
- `Regime Table` 主要是对象层

而后续 simulator 执行需要的是：

- 优先级顺序
- lane 类型
- 参数方向
- 预算入口

这些内容不应混进 family / regime 基础表。

### 最小字段

| 字段 | 含义 |
|---|---|
| `priority_item_id` | 该条优先级记录标识 |
| `object_level` | family / regime |
| `object_id` | 对应 family_id 或 regime_id |
| `canonical_status` | 从上游继承的 canonical 状态 |
| `priority_rank` | 当前排序位置 |
| `priority_source` | importance-guided / time-only / manual |
| `simulator_lane_id` | lane 标识 |
| `lane_type` | compute / cache / locality / reduction / constraint |
| `recommended_tuning_target` | 调参方向 |
| `parameter_scenario_ids` | 关联参数场景 |
| `expected_signal` | 希望看到的响应 |
| `selection_reason` | 进入该 lane 的理由 |
| `status` | pending / running / completed |

### 输出作用

- 作为 simulator 任务编排入口
- 作为 baseline 与 ours 的排序对比入口

---

## 6.4 Validation Worksheet

### 角色

把验证计划从说明文档变成可执行验证协议。

### 最小字段

| 字段 | 含义 |
|---|---|
| `validation_round_id` | 当前验证轮次 |
| `target_scope` | 当前验证范围 |
| `baseline_defs` | baseline 定义集合 |
| `importance_strategy` | ours 的排序策略 |
| `parameter_scenarios` | 本轮参数场景 |
| `budget_definition` | runs / families / regimes 预算 |
| `metrics` | Object Reduction / Early Gain / Sensitivity / Tuning Efficiency |
| `success_criteria` | 本轮成功判据 |
| `selected_families` | 入选 family |
| `selected_regimes` | 入选 regime |
| `notes` | 备注 |

### 输出作用

- 作为 validation 执行协议
- 作为不同 baseline 之间的公平对照约束

---

## 6.5 Writeback Map

### 角色

把 simulator 结果回写到结构对象与原始 workload。

### 为什么它是必需层

当前最容易被忽略的一点是：

**validation 不是终点，回写才是方法闭环成立的关键。**

如果没有这张表，结果只会停在：

- 某个 regime 对某参数敏感

而不会稳定变成：

- 某个 family 的 decision weight 被修正
- 某些 anchors 的 importance 被上调或下调
- workload 主路径解释被更新

### 最小字段

| 字段 | 含义 |
|---|---|
| `writeback_id` | 回写记录标识 |
| `simulator_lane_id` | 来源 lane |
| `regime_id` | 来源 regime |
| `family_id` | 对应 family |
| `rep_kernel_ids` | 关联 anchors |
| `member_invocations` | 关联原始 invocations |
| `parameter_scenario_id` | 触发该结果的场景 |
| `observed_response` | 观测到的响应摘要 |
| `sensitivity_score` | 响应强度 |
| `decision_update` | 对 decision weight 的更新建议 |
| `importance_update` | 对 importance 的更新建议 |
| `validation_status_update` | pending -> selected -> validated 等状态更新 |
| `review_status_update` | no-review / keep-review / resolved-review |
| `workload_explanation_note` | 回写到 workload 解释的摘要 |

### 输出作用

- 把 simulator 结果回写到 family / regime / anchor / workload
- 形成端到端方法闭环

---

## 7. 五张表之间的引用关系

推荐的引用链如下：

1. `Anchor Table` 提供 `rep_kernel_id`
2. `Family Table` 引用 `member_rep_kernels`
3. `Regime Table` 引用 `source_rep_kernels`
4. `Priority & Lane Table` 引用 `family_id / regime_id`
5. `Validation Worksheet` 引用 `selected_families / selected_regimes / parameter_scenarios`
6. `Writeback Map` 反向引用 `regime_id / family_id / rep_kernel_ids / member_invocations`

这样形成一个可追踪闭环：

`anchor -> family -> regime -> lane -> validation -> writeback -> anchor/workload`

---

## 8. 第一版推荐的文件级输出

当前建议先固定下面 5 个输出文件：

1. `backend_family_table_v1.json`
2. `backend_regime_table_v1.json`
3. `backend_priority_lane_table_v1.json`
4. `backend_validation_worksheet_v1.json`
5. `backend_writeback_map_v1.json`

如需方便人工讨论，可同步生成对应 Markdown 摘要，但 Markdown 不应替代 JSON 主输出。

---

## 8.1 当前 canon 对 schema 的补充约束

根据当前 `mini_transformer_v4` family canon，后端 schema 第一版应额外满足：

- `Family Table` 必须能表达 `weak-share`
- `Regime Table` 必须能表达 `review-needed`
- `Priority & Lane Table` 必须保留上游 canonical 状态
- `Writeback Map` 必须能表达 review 是否被保留、升级或解除

也就是说，当前后端层不能假设所有进入 family / regime 的对象都已经“完全稳定”。

---

## 9. 第一版实现顺序

当前建议按以下顺序落地：

### Step 1

先固定：

- Anchor Table
- Family canon

### Step 2

生成：

- `Family Table`
- `Regime Table`

### Step 3

补上：

- `Priority & Lane Table`

让后端对象真正进入 simulator-side decision layer。

### Step 4

补上：

- `Validation Worksheet`

让 baseline、budget、metrics 与 success criteria 固定。

### Step 5

最后补：

- `Writeback Map`

让 simulator 结果能回写到结构层与 workload 解释。

---

## 10. 当前阶段最重要的判断

当前最重要的不是马上把所有表都填满，而是：

**先把后端输出层的对象边界固定下来。**

也就是说，当前阶段真正要保的是：

- 输出对象不能混层
- 每层引用关系清楚
- 回写链路存在

这样后续即使：

- anchor 还是占位版
- family 权重还是半定量
- validation 只做小规模原型

整个后端层也已经是一个可运行、可替换、可扩展的稳定框架。

---

## 11. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. 后端输出层应由 `Family Table`、`Regime Table`、`Priority & Lane Table`、`Validation Worksheet`、`Writeback Map` 五层组成。
2. `Family / Regime` 负责构造对象，`Priority / Validation` 负责证明对象有用，`Writeback` 负责闭环回写。
3. 后端输出层必须只依赖 Anchor 接口，而不依赖前端 compression 的内部实现。
4. 第一版最关键的不是填满所有数值，而是先把对象、引用关系和回写逻辑固定下来。
