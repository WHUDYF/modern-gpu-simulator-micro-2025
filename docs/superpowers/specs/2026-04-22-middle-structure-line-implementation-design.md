# 中端结构线实现设计

日期：2026-04-22

## 目标

本设计文档用于把当前项目方法主线中的中端结构线正式收敛为一份可执行 spec。

当前中端结构线负责的方法位置是：

`frontend compression -> representative anchors -> family -> representative execution regime -> importance ratio -> tuning priority -> simulator validation`

其中中端结构线只负责下面这一段：

`representative anchors -> family -> representative execution regime -> importance ratio`

本 spec 的目标不是重新定义整条方法线，而是明确：

- 中端结构线的对象边界
- 中端结构线应交付的结构化产物
- 中端结构线的实现架构
- 中端结构线与前后两段的对接字段
- 当前阶段的验收标准与风险边界

---

## 边界

本设计只做下面这些事：

- 固定 `Anchor / Family / Regime / Importance` 的对象边界
- 规定中端结构线必须交付的表、schema、artifact 和 builder
- 规定中端结构线的最小 validator 与 writeback contract
- 规定 `mini_transformer_v4` 上的第一版落地方式

本设计明确不做下面这些事：

- 不重新定义 frontend compression
- 不把 representative compression 直接等同于 family
- 不直接输出完整 simulator 参数处方
- 不直接承担后端 validation 的实验结论
- 不把 provisional importance 分数写成 final fact

---

## 第一版完成标准

第一版完成后，至少应满足：

1. `mini_transformer_v4` 能稳定生成一组 `phase-aware / context-aware / shape-aware` anchors
2. 这些 anchors 能稳定映射到 `Family Table`
3. Family 能继续稳定拆成 `Representative Regime Table`
4. `importance_score` 和 `regime_priority_score` 能作为结构字段存在
5. 输出产物能被 backend validation 直接消费，而不是只停留在 prose

---

## 一、问题定义

当前中端结构线要解决的问题不是：

- 再提出一个新的 representative compression 前端
- 再提出一个新的 kernel clustering 技巧
- 单独给某个 kernel 做更细 profiling

它要解决的是一个更具体的问题：

**在已有 representative compression 之后，如何把压缩后的 workload 对象继续组织成 simulator 可用、可比较、可排序、可回写的结构化决策对象。**

换句话说：

- frontend 主要回答“压谁”
- 中端结构线要继续回答：
  - “怎么组织”
  - “哪些对象共享机制”
  - “哪些对象必须继续拆”
  - “优先级挂在哪里”

因此，中端结构线的真正价值不在于“更好的 cluster”，而在于：

**补上 compression 之后的 simulator-side organization layer。**

---

## 二、对象层级定义

### 2.1 Representative Anchors

代表前端 compression 之后的输入锚点。

Anchor 的职责是：

- 承接 representative compression 的输出
- 保留 membership / coverage / time 接口
- 在进入 Family 之前补回 phase / shape / route / template 提示

Anchor 不是：

- 最终机制解释对象
- 最终 simulator object

### 2.2 Family

代表共享机制层的组织对象。

Family 的职责是：

- 识别共享 data path / execution template 的对象集合
- 作为 importance ratio 的第一层承载表
- 为 regime 拆分提供上层约束

Family 不是：

- representative cluster 的别名
- 最终 backend 入口对象

### 2.3 Representative Execution Regime

代表最终进入 simulator lane 的执行区间对象。

Regime 的职责是：

- 在 Family 内继续按 `shape / context / resource signature` 拆分
- 提供 backend validation / tuning 的直接输入对象
- 承接最终 `regime_priority_score`

### 2.4 Importance

代表中端结构线中的优先级字段层。

它至少包括：

- `coverage_weight`
- `time_weight`
- `decision_weight`
- `importance_score`
- `regime_priority_score`

当前阶段这些字段允许：

- measured
- derived
- provisional
- placeholder

但每个字段必须显式标注来源性质。

---

## 三、中端结构线必须坚持的术语边界

### 3.1 representative anchor 不等于 family

compression 结果只给出输入锚点。

family 需要额外引入：

- phase
- route primitive
- hardware template
- boundary-first 判据

### 3.2 family 不等于 final simulator object

同一 family 内可能仍然包含多个不同的：

- shape 区间
- context scope
- resource signature

因此 family 之后必须继续进入 regime。

### 3.3 importance 不直接挂在裸 cluster 上

importance 的主要承载层应是：

- family-level importance
- regime-level priority

而不是直接停在 frontend cluster。

---

## 四、必须交付的结构化产物

根据并行 session briefing，中端结构线必须至少交付下面三类主产物：

1. `Family Table`
2. `Regime Table`
3. `Importance Scoring Sheet`

为了让这三类产物真正可实现，第一版实现中额外要求交付：

4. `Representative Anchor Table`
5. `Middle Layer Builder`
6. `Middle Layer Artifacts`
7. `Middle Layer Validator / Tests`

---

## 五、最小字段要求

### 5.1 Representative Anchor Table

最小字段必须包括：

- `anchor_id`
- `kernel_name`
- `kernel_name_raw`
- `phase_id`
- `context_scope`
- `cluster_id`
- `member_invocations`
- `coverage_count`
- `coverage_weight`
- `time_weight`
- `trace_order_summary`
- `grid_dim_summary`
- `block_dim_summary`
- `shape_hint_summary`
- `route_hint`
- `template_hint`

当前设计要求：

- 主键不能再只用 `kernel_name`
- 必须至少提升到 `kernel + phase/context + shape`

### 5.2 Family Table

最小字段必须包括：

- `family_id`
- `input_anchor_ids`
- `phase_scope`
- `route_primitive`
- `hardware_template`
- `boundary_status`
- `boundary_notes`
- `shape_regime_summary`
- `resource_signature_summary`
- `coverage_weight`
- `time_weight`
- `decision_weight`
- `importance_score`
- `priority_class`
- `recommended_tuning_target`

### 5.3 Regime Table

最小字段必须包括：

- `regime_id`
- `family_id`
- `source_anchor_ids`
- `phase_id`
- `route_primitive`
- `hardware_template`
- `shape_regime`
- `context_scope`
- `resource_signature`
- `coverage_weight`
- `time_weight`
- `family_importance_score`
- `local_decision_weight`
- `regime_priority_score`
- `simulator_lane_id` 或可对接 lane 字段

### 5.4 Importance Scoring Sheet

最小字段必须包括：

- `coverage_weight`
- `time_weight`
- `decision_weight`
- `importance_score`
- `priority_class`
- `family_importance_score`
- `local_decision_weight`
- `regime_priority_score`
- `weight_source`

其中 `weight_source` 至少要标识：

- `measured`
- `derived`
- `provisional`
- `placeholder`

---

## 六、实现架构

第一版中端结构线应采用：

**文档 + builder + artifacts + tests**

而不是只靠 prose 讨论推进。

### 6.1 文档层

负责定义对象、边界、解释和当前 provisional judgment。

当前 `mini_transformer_v4` 已存在的文档层包括：

- middle layer blueprint
- anchor table note
- family table note
- regime table note
- lane mapping note

### 6.2 Builder 层

负责把证据自动提升为中端对象。

第一版 builder 应至少实现：

1. `anchor builder`
2. `family builder`
3. `regime builder`
4. `lane mapper`

当前 builder 的目标不是全自动学习，而是：

**把稳定规则从 prose 提升成可重复运行的构建过程。**

### 6.3 Artifacts 层

负责输出机器可读对象。

第一版要求同时输出：

- `json`
- `markdown snapshot`

推荐输出路径：

- `artifacts/middle_layer/<workload>/anchors.json`
- `artifacts/middle_layer/<workload>/families.json`
- `artifacts/middle_layer/<workload>/regimes.json`
- `artifacts/middle_layer/<workload>/lanes.json`
- `artifacts/middle_layer/<workload>/bundle.json`

### 6.4 Validator / Tests 层

负责保证：

- schema 不漂移
- 对象归属一致
- ID 引用稳定
- coverage/time 汇总逻辑正确

---

## 七、Builder 的四个核心模块

### 7.1 Anchor Builder

输入：

- `mini_transformer_v4_full.json`
- `squash.json`
- `batch.json`
- `baseline_ape.json`

职责：

- 生成 `phase-aware / context-aware / shape-aware` anchors
- 补 canonical kernel name
- 保留 raw kernel name 和 invocation membership
- 生成 observed coverage / time ratio

### 7.2 Family Builder

职责：

- 从 anchors 生成共享机制层对象
- 固定 `phase_scope / route_primitive / hardware_template`
- 记录 boundary status 和 tuning target
- 挂 family-level importance fields

### 7.3 Regime Builder

职责：

- 从 family 内部继续拆出 backend 入口对象
- 按 `shape / context / resource signature` 划分 regime
- 生成 `regime_priority_score`

### 7.4 Lane Mapper

职责：

- 为每个 regime 生成后续可消费的 lane mapping
- 明确 `parameter_direction / baseline_type / validation_metric / writeback_target`

注意：

Lane 是 backend 对接接口，不属于中端结构线的最终实验结论。

---

## 八、当前第一版实现策略

### 8.1 先用 placeholder-friendly pipeline

当前第一版允许：

- membership 部分真实、部分 placeholder
- decision weight 是 provisional
- 部分 boundary notes 仍靠规则配置

但不允许：

- 没有结构化对象
- 只有 prose 没有 artifacts

### 8.2 先固定 IDs，再逐步替换权重来源

实现优先级应是：

1. 先固定 `anchor_id / family_id / regime_id`
2. 再固定对象引用关系
3. 最后再逐步替换权重来源

### 8.3 先支持 `mini_transformer_v4`

第一版实现只要求在 `mini_transformer_v4` 上闭环。

当前不要求：

- 多 workload 泛化
- 通用自动 rule inference

---

## 九、必须回答的三个核心问题

根据并行 session B 线要求，中端结构线的实现必须显式回答：

### 9.1 family 判据最关键的边界是什么

当前最关键的边界包括：

- `gemm_tiled` vs `attention_score`
- `softmax_kernel` vs `context_mul`
- `softmax` vs `layernorm`

也就是说，Family 的核心难点不在“列出 family”，而在：

**哪些对象不能因为名字相似或同属一条大路径就被错误合并。**

### 9.2 为什么 Family 之后还必须继续拆 Regime

因为同一 Family 内仍然可能存在：

- 不同 shape
- 不同 context
- 不同 resource signature

如果不继续拆，Family 就会被错误当成 final simulator object。

### 9.3 importance 哪些部分还是 provisional

当前仍然 provisional 的主要包括：

- `decision_weight`
- `importance_score` 的部分来源
- `regime_priority_score` 的一部分排序逻辑

因此第一版 spec 必须要求：

所有权重字段显式标注来源状态。

---

## 十、对接字段

中端结构线必须向其它工作线稳定输出下面这些字段：

### 对 A 线的输入依赖

- `anchor_id`
- `cluster_id`
- `member_invocations`
- `coverage_count`
- `coverage_weight`
- `time_weight`
- `phase_id`

### 向 C 线输出的核心字段

- `family_id`
- `regime_id`
- `route_primitive`
- `hardware_template`
- `boundary_status`
- `importance_score`
- `regime_priority_score`
- `recommended_tuning_target`
- `simulator_lane_id` 或等价 lane 对接字段

---

## 十一、验收标准

中端结构线的第一版实现完成后，至少应满足下面这些验收条件：

1. 能稳定运行 builder 并生成 `anchors / families / regimes / lanes` artifacts
2. 能通过最小测试，保证 ID 映射、覆盖关系、基础权重汇总正确
3. artifacts 中区分 `kernel_name` 与 `kernel_name_raw`
4. artifacts 中显式保留 provisional labels 和 observed ratios
5. backend 能直接读取 `regime` 和 `lane` 相关字段

---

## 十二、当前最主要的风险

### 风险 1：Anchor 粒度仍然过粗

如果 anchor 继续退化成 kernel-name-only 对象，后续 Family 和 Regime 都会失真。

### 风险 2：Family / Regime 规则只存在于 prose

如果规则不进入 builder，中端结构线无法稳定复现。

### 风险 3：Importance 被误当成 final fact

当前 provisional 分数如果不显式标注来源，很容易在后续主线整合时被误写成最终实验结论。

---

## 十三、下一步实现建议

第一版 spec 落地后，下一步建议按下面顺序继续推进：

1. 把 builder 中目前 still-manual 的 family / regime 配置进一步抽成 rule config
2. 给 `decision_weight` 增加显式 `source_status`
3. 增加 `writeback contract`，把 lane 结果回写到 regime / family
4. 补一个 `Importance Scoring Sheet` artifact，而不是只在 bundle 中隐含
5. 再决定是否扩到第二个 workload

---

## 十四、当前阶段的简短结论

如果压成最短形式，可以写成：

1. 中端结构线的真正任务不是“再写一张 family 文档”，而是把 anchors 提升成 `family / regime / importance` 三层正式对象。
2. 第一版实现必须采用 `文档 + builder + artifacts + tests` 的形式，而不是只停留在 prose。
3. 中端结构线既不能把 representative compression 直接等同于 family，也不能把 family 直接等同于 final simulator object。
4. 当前最合理的落地方向，是先在 `mini_transformer_v4` 上做出最小闭环，再逐步替换 provisional weights 与 rule config。

---

## 十五、当前已拍板的实现决策

下面这些点已经在当前 session 中明确拍板，后续实现应默认遵守，除非主线再明确推翻。

### 15.1 Anchor 粒度

第一版 Anchor 默认采用：

`kernel + phase + semantic route + shape/context`

这意味着：

- 同名 kernel 不自动等于同一个 anchor
- 只要 semantic route 不同，就允许拆成不同 anchor
- `Q/K/V projection`、`output projection`、`FFN expansion`、`FFN contraction` 这类角色应进入正式对象定义

### 15.2 `attention_score` 的 Family 位置

当前拍板为：

- `attention_score` 保留在 dense backbone family 中
- 但在 regime 层必须强制单独拆出

也就是说：

- 在共享机制层，它仍视为 `Dense Tiled Compute` 主干的一部分
- 在 backend 入口层，它不能被当作普通 dense projection regime

### 15.3 `softmax` 与 `layernorm`

当前拍板为：

- 两者同属 `Reduction / Normalize family`
- 但必须在 regime 层拆开

原因是：

- 它们共享 reduction / normalize 主模板
- 但其 semantic route、context 和参数敏感方向不同

### 15.4 `decision_weight`

当前拍板为：

- 第一版以人工方法判断为主
- 但每个判断都必须附来源说明或简短 rule note

这意味着第一版不强求完全自动 rule-based scoring，但不允许无解释的 `High / Medium / Low`。

### 15.5 Regime 数量策略

当前拍板为：

- 第一版 regime 采用“受控地多拆一点”的策略

判断标准是：

**只要差异已经足以改变 backend 的 `parameter_direction / lane_goal / expected signal`，就应优先拆成不同 regime。**

### 15.6 Lane Mapping 的责任范围

当前拍板为：

- B 线不只做到接口层
- B 线要做到半完整 backend 对接层

因此第一版 lane mapping 至少应明确：

- `simulator_lane_id`
- `parameter_direction`
- `baseline_type`
- `validation_metric`
- `writeback_target`

### 15.7 Family 的主判据

当前拍板为：

- `共享硬件执行模板` 是 Family 的主判据
- `semantic route / operator role` 是强辅助边界
- `parameter sensitivity` 不作为 Family 主判据，而主要进入 importance 与 lane 层

### 15.8 Importance 的表示方式

当前拍板为：

- 第一版 importance 采用双轨制

也就是同时保留：

- `observed / measured fields`
- `provisional / label-based fields`

例如：

- `observed_coverage_ratio`
- `observed_time_ratio`
- `coverage_label`
- `time_label`
- `decision_label`
- `importance_score`

### 15.9 Writeback 策略

当前拍板为：

- 设计上定义完整 writeback 链：
  `lane -> regime -> family -> workload explanation`
- 实现上第一版先优先做：
  `lane -> regime`

这意味着：

- spec 里必须保留完整 writeback contract
- builder / artifact 第一版至少要支持最小回写闭环
