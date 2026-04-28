# B 线语义与接口加固设计

日期：2026-04-28

## 目标

本设计文档用于定义当前阶段 B 线的下一轮实现方向。

当前项目已经形成第一版可运行方法链：

`frontend anchor -> middle structure -> backend planning -> execution bridge -> result summary -> writeback interface`

其中 B 线位于 `frontend anchor` 和 `backend planning` 之间，负责把 A 线输出的 representative anchors 提升为 backend 可消费的结构化决策对象。

本轮设计的核心判断是：

**在 A 线尚未完全稳定之前，B 线不做最终分组优化，而做 semantic / interface hardening。**

也就是说，本轮目标不是证明当前 family / regime / lane 已经是最终结构，而是让 B 线对象系统具备：

- 稳定接收 A 线输出的能力
- 稳定表达对象权重的能力
- 稳定承接执行证据与回写的能力
- 稳定向 C 线输出可消费结构的能力

---

## 边界

本设计只做下面这些事：

- 固定 `Anchor / Family / Regime / Lane` 的职责边界
- 规定 B 线必须接收的权重字段和 provenance 字段
- 规定 family / regime / lane 的语义边界
- 规定 B 线必须交付的 schema、artifact 和 builder contract
- 规定 `mini_transformer_v4` 上的第一版硬化方式

本设计明确不做下面这些事：

- 不重新定义 A 线 compression
- 不在当前阶段大幅重分 family / regime / lane
- 不把 family 直接等同于 anchor
- 不把 regime 直接等同于算子语义标签
- 不直接输出完整 simulator 参数处方
- 不把 smoke execution 结果写成 final validation fact

---

## 第一版完成标准

第一版完成后，至少应满足：

1. B 线能够稳定消费带权重的 representative anchors。
2. Anchor 能携带或补齐 membership / time_weight / workload_scale 相关信息。
3. Family 能稳定表达硬件执行模板分组。
4. Regime 能在同一 hardware family 内继续结合算法功能角色、shape/context 和 resource signature 做拆分。
5. Lane 能稳定表达 backend 参数方向和验证入口。
6. 输出产物能被 backend planning 和 execution bridge 直接消费。

---

## 一、问题定义

当前 B 线要解决的问题不是：

- 再提出一个新的 representative compression 前端
- 再做一个新的 kernel clustering 技巧
- 单独给某个 kernel 做更细 profiling

它要解决的是：

**在已有 representative compression 之后，如何把压缩后的 workload 对象继续组织成 simulator 可用、可比较、可排序、可回写的结构化决策对象。**

换句话说：

- A 线主要回答“压谁”
- B 线要继续回答：
  - “怎么组织”
  - “哪些对象共享机制”
  - “哪些对象必须继续拆”
  - “优先级挂在哪里”

因此，B 线的真正价值不在于“更好的 cluster”，而在于：

**补上 compression 之后的 simulator-side organization layer。**

---

## 二、当前阶段边界

本轮 B 线迭代的性质是：

**semantic / interface hardening**

而不是：

**final grouping optimization**

原因是当前 A 线输入还没有完全变硬：

- A 线主分组逻辑仍可能继续向 PKA-core behavior feature space 对齐。
- A 线还需要更稳定地输出 `membership / time_weight / workload_scale`。
- 当前 anchor 权重和 behavior evidence 仍可能在下一轮 A 线迭代后变化。

因此，本轮 B 线可以做的是：

- 补 schema
- 补字段
- 补语义定义
- 补 provenance / provisional 标记
- 补 builder 对新字段的透传或过渡计算
- 补测试，确保 C 线仍能消费
- 保持现有 family / regime / lane 数量基本不动

本轮 B 线不应做的是：

- 大幅调整 family 数量
- 大幅调整 regime 数量
- 大幅重设 lane
- 基于当前不完整 A 线输入重新计算最终 importance 排序
- 宣称当前 family / regime / lane 已经被正式 validation 证明

本轮修改的目标是：

**让 B 线对象系统准备好接收更好的 A 线输入，而不是用当前不完整输入做最终结构决策。**

---

## 三、A 线到 B 线的输入契约

虽然这份设计主要服务 B 线，但下一轮 B 线不能再默认 A 线只输出“代表对象名字”。

### 3.1 必须确认的 anchor 权重字段

每个进入 B 线的 anchor，应尽量携带以下信息：

- `member_count`：该 anchor 覆盖多少 invocation。
- `member_invocations`：该 anchor 覆盖哪些原始 invocation。
- `total_time` 或 `time_weight`：该 anchor 覆盖的总运行时间或时间占比。
- `avg_time`：该 anchor 成员的平均运行时间。
- `total_dynamic_insts` 或 `avg_dynamic_insts`：该 anchor 的工作量尺度。
- `source_status`：该字段来自原始输出、重算结果还是过渡填充。

第一版 importance 应以 `time_weight` 作为主信号，因为时间贡献与 squash 的动机一致，也最接近后续 simulator tuning 的优化关注点。

`member_count` 和 `dynamic_inst_count` 应作为辅助信号，用于解释代表范围和工作量尺度，不应直接压过时间贡献。

### 3.2 本轮允许的处理方式

- 如果 A 线当前已经输出这些字段，B 线应直接消费。
- 如果 A 线暂未输出，但 B 线可从现有输入可靠重算，可以先在 B 线中生成兼容字段，并记录为过渡方案。
- 如果某些字段暂时无法获得，应在 artifacts 中显式标记为 `missing` 或 `provisional`，不要静默当成 0 或默认值。

### 3.3 不应做的事

- 不要只用 anchor 名字或 kernel 名字推断重要性。
- 不要只用成员数量替代运行时间权重。
- 不要只用平均时间替代总贡献，因为小组高均值和大组中等均值的意义不同。

---

## 四、Family 设计

family 在下一版中应更明确地承担硬件分组职责。

### 4.1 Family 的主判据

family 的第一分组轴应优先是执行模板，例如：

- `dense_tiled_compute`
- `reduction_template`
- `streaming_or_locality_aggregation`
- `elementwise_template`
- 必要时预留 `layout_or_data_movement`

检查 YAML 和 builder 中 family 的判据是否主要落在：

- `execution_template`
- `route_primitive`
- 共享硬件瓶颈或共享执行模式

### 4.2 Family 的解释字段

`resource_sensitivity` 和 `expected_parameter_direction` 本轮更适合作为 family 的解释字段和 lane 的依据，而不应作为 family 的第一分组轴。

这些字段可以帮助说明：

- 这个 family 为什么属于某个硬件模板
- 这个 family 后续更可能受哪些资源约束
- 这个 family 进入 backend 时优先看什么方向

### 4.3 Family 不应主要依赖的判据

需要检查并弱化这些写法：

- 按模型模块名直接建 family
- 按 kernel 名字直接建 family
- 按 attention / FFN / residual / layernorm 这类模型语义直接建 family
- 为了让 family 看起来更复杂而强行合并或拆分
- 在 execution evidence 不足时，过早按资源瓶颈重分 family

### 4.4 当前 family 可接受的状态

第一版中允许出现：

- 单 anchor family
- family 与 anchor 边界高度相似
- 小 workload 下没有大规模跨 anchor 合并

但必须在文档或 artifact metadata 中说明：

- 这是小 workload 和细粒度 anchor 条件下的自然结果
- family 在定义上仍然是硬件等价类，不是 anchor 的重命名

---

## 五、算法功能分组设计

算法分组不应再绑定具体算子或模型模块，而应改成更普适的计算功能角色。

### 5.1 推荐的算法功能标签

下一轮 regime 或 anchor metadata 中可优先使用以下标签：

- `primary_compute`
- `score_or_transform`
- `reduction_normalization`
- `aggregation_or_fusion`
- `elementwise_postprocess`
- `layout_or_data_movement`
- `constraint_or_bookkeeping`

### 5.2 标签定义检查

每个标签都应回答：

- 这个 kernel 在整体计算流程里承担什么功能？
- 它是否是主要算量来源？
- 它是否只是后处理、约束或回归检查对象？
- 它是否会影响同一硬件 family 内的 regime 拆分？

### 5.3 不应做的事

- 不要把 `algorithm_group` 写成 `qkv_projection`、`ffn_expand`、`softmax` 这类具体算子名。
- 不要让算法功能标签替代硬件 family。
- 不要让算法功能标签直接决定 lane。

---

## 六、Regime 设计

regime 应成为硬件信息和算法功能信息汇合的位置。

### 6.1 Regime 的拆分依据

每个 regime 都应至少说明：

- 它属于哪个 hardware family
- 它对应哪个 algorithm function group
- 它的 shape / context 是否与同 family 其他对象不同
- 它的 resource signature 是否与同 family 其他对象不同
- 它是否会改变 backend parameter direction
- 它是否值得作为单独 backend validation object

### 6.2 Regime 的必要性说明

每个 regime 最好能回答：

- 如果把它并回同 family 其他 regime，会丢失什么信息？
- 如果把它单独保留，后端验证能获得什么额外信息？
- 它是主调参对象、辅助对象，还是 constraint / regression object？

### 6.3 当前结构的处理原则

- 暂时不为了形式美强行减少 regime 数量
- 暂时不为了“更细”继续扩 regime
- 优先补齐每个 regime 的硬件判据、算法功能判据和重要性权重
- 等 execution evidence 回来后，再决定是否合并或拆分

---

## 七、Lane 设计

lane 应保持为 backend 参数方向和验证入口。

### 7.1 每条 lane 应说明

- 对应的 regime
- 主要硬件敏感方向
- 预期观察的资源或结构
- 关联的 backend validation scenario
- result summary 回来后应写回哪个 regime / family

### 7.2 lane 不应承担的职责

- 不要把 lane 当成算法分组标签
- 不要把 lane 当成 family 的别名
- 不要让 lane 只描述“这是哪个模型模块”

### 7.3 lane 与调参的关系

每条 lane 最好能落到至少一种可验证方向：

- register / occupancy
- shared memory
- cache / locality
- DRAM / memory pressure
- reduction path
- elementwise regression / constraint

---

## 八、实现与产物约束

重点文件包括：

- `docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml`
- `experiments/baseline_diagnosis/build_middle_layer.py`
- `artifacts/middle_layer/mini_transformer_v4/`
- `experiments/backend_pipeline/results/mini_transformer_v4/`

本轮实现应保证：

- YAML 能表达 hardware family basis、algorithm function group、lane direction 和 provenance 语义
- builder 能透传或补齐 anchor 权重字段
- artifacts 能显式表示 measured / derived / provisional / missing
- C 线能够继续消费 middle-layer bundle
- 不因 schema hardening 破坏现有回归

---

## 九、验收标准

本轮完成后，应满足：

- 每个 anchor 有权重字段或明确 provisional 标记
- 每个 family 有 hardware grouping basis
- 每个 regime 有 algorithm function group 和 split rationale
- 每条 lane 有 parameter direction
- 当前对象数量不因本轮加固被大幅改变
- B 线 artifacts 能继续被 C 线消费
- 相关测试通过
- 文档明确说明当前是接口加固，不是最终分组优化

---

## 十、风险边界

当前最主要的风险不是对象太少，而是语义过早固化。

因此本轮需要避免：

- 用不完整 A 线输入重新定义最终 importance
- 用烟雾式 execution evidence 直接证明最终分组成立
- 把 family 写成算法语义标签
- 把 regime 写成纯硬件模板标签
- 把 lane 写成分析文本而不是 backend 接口

---

## 十一、结论

本轮 B 线的核心不是增加对象数量，而是让现有对象更有方法含义。

最重要的改动方向是：

**让 family 更硬件化，让 regime 更明确地承接算法功能信息，让 A 线权重信息进入 B 线重要性判断。**

只有这样，B 线才不会退化成对 A 线结果的重命名，也不会退化成按算子名贴标签的解释层。
