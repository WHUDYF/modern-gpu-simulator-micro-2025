# B 线下一轮修改执行清单

日期：2026-04-25

## 1. 文档目的

这份文档把当前 B 线方法论修正转成可执行 checklist，供下一次 RLCR 或手工修改时使用。

它承接：

- `docs/b-line-method-update-note-2026-04-24.md`
- `docs/a-line-pka-alignment-note-2026-04-24.md`

本轮修改的目标不是重新设计整条方法链，而是把 B 线从当前可运行 v1 进一步收敛到更清楚的方法边界：

`representative anchor -> hardware family -> algorithm-aware regime -> backend lane`

---

## 2. 本轮修改总目标

本轮 B 线修改应完成四件事：

1. 明确 A 线传给 B 线的 anchor 不是最终语义对象，而是带权重的 representative object。
2. 明确 family 主要承担硬件分组职责。
3. 明确 regime 是硬件分组和算法功能分组汇合的位置。
4. 明确 lane 只负责 backend 参数方向和验证入口，不重新承担语义分类。

成功后的 B 线应能清楚回答：

- 这个 anchor 有多重要？
- 它属于哪类硬件执行模板？
- 在这个硬件 family 内，它承担什么算法功能角色？
- 它是否需要作为单独 regime 进入 backend？
- 它应该映射到哪条 backend validation lane？

---

## 3. 当前阶段边界

本轮 B 线迭代的性质是：

**semantic / interface hardening**

而不是：

**final grouping optimization**

原因是当前 A 线输入还没有完全变硬：

- A 线主分组逻辑还需要进一步向 PKA-core behavior feature space 对齐。
- A 线还需要更稳定地输出 `membership / time_weight / workload_scale`。
- 当前 anchor 权重和 behavior evidence 仍可能在下一轮 A 线 RLCR 后变化。

因此，本轮 B 线可以做的是：

- [ ] 补 schema。
- [ ] 补字段。
- [ ] 补语义定义。
- [ ] 补 provenance / provisional 标记。
- [ ] 补 builder 对新字段的透传或过渡计算。
- [ ] 补测试，确保 C 线仍能消费。
- [ ] 保持现有 family / regime / lane 数量基本不动。

本轮 B 线不应做的是：

- [ ] 大幅调整 family 数量。
- [ ] 大幅调整 regime 数量。
- [ ] 大幅重设 lane。
- [ ] 基于当前不完整 A 线输入重新计算最终 importance 排序。
- [ ] 宣称当前 family / regime / lane 已经被正式 validation 证明。

本轮修改的目标是：

**让 B 线对象系统准备好接收更好的 A 线输入，而不是用当前不完整输入做最终结构决策。**

---

## 4. 修改前检查

开始修改前先确认当前基线稳定。

- [ ] 确认当前分支是合流后的工作分支。
- [ ] 确认 `docs/b-line-method-update-note-2026-04-24.md` 存在。
- [ ] 确认 `docs/a-line-pka-alignment-note-2026-04-24.md` 存在。
- [ ] 运行或至少检查当前 B 线 builder 入口：
  `experiments/baseline_diagnosis/build_middle_layer.py`
- [ ] 检查当前 B 线规则真源：
  `docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml`
- [ ] 检查当前 B 线 artifacts：
  `artifacts/middle_layer/mini_transformer_v4/`

---

## 5. A 线到 B 线的输入契约

虽然这份清单主要服务 B 线，但下一轮 B 线不能再默认 A 线只输出“代表对象名字”。

### 5.1 必须确认的 anchor 权重字段

每个进入 B 线的 anchor，应尽量携带以下信息：

- [ ] `member_count` 或等价字段：该 anchor 覆盖多少 invocation。
- [ ] `member_invocations` 或等价字段：该 anchor 覆盖哪些原始 invocation。
- [ ] `total_time` 或 `time_weight`：该 anchor 覆盖的总运行时间或时间占比。
- [ ] `avg_time`：该 anchor 成员的平均运行时间。
- [ ] `total_dynamic_insts` 或 `avg_dynamic_insts`：该 anchor 的工作量尺度。
- [ ] 明确字段来源：来自 A 线原始输出、B 线重算，还是暂时由 full-feature source 补齐。

第一版 importance 应以 `time_weight` 作为主信号，因为时间贡献与 squash 的动机一致，也最接近后续 simulator tuning 的优化关注点。

`member_count` 和 `dynamic_inst_count` 应作为辅助信号，用于解释代表范围和工作量尺度，不应直接压过时间贡献。

### 5.2 本轮允许的处理方式

- [ ] 如果 A 线当前已经输出这些字段，B 线应直接消费。
- [ ] 如果 A 线暂未输出，但 B 线可从现有输入可靠重算，可以先在 B 线中生成兼容字段，并记录为过渡方案。
- [ ] 如果某些字段暂时无法获得，应在 artifacts 中显式标记为 `missing` 或 `provisional`，不要静默当成 0 或默认值。

### 5.3 不应做的事

- [ ] 不要只用 anchor 名字或 kernel 名字推断重要性。
- [ ] 不要只用成员数量替代运行时间权重。
- [ ] 不要只用平均时间替代总贡献，因为小组高均值和大组中等均值的意义不同。

---

## 6. family 修改清单

family 在下一版中应更明确地承担硬件分组职责。

### 6.1 family 的主判据

检查 YAML 和 builder 中 family 的判据是否主要落在：

- [ ] `execution_template`
- [ ] `route_primitive`
- [ ] 共享硬件瓶颈或共享执行模式

本轮 family 的第一分组轴应优先是执行模板，例如：

- [ ] `dense_tiled_compute`
- [ ] `reduction_template`
- [ ] `streaming_or_locality_aggregation`
- [ ] `elementwise_template`
- [ ] 必要时预留 `layout_or_data_movement`

`resource_sensitivity` 和 `expected_parameter_direction` 本轮更适合作为 family 的解释字段和 lane 的依据，不应作为 family 的第一分组轴。

### 6.2 family 不应主要依赖的判据

需要检查并弱化这些写法：

- [ ] 按模型模块名直接建 family。
- [ ] 按 kernel 名字直接建 family。
- [ ] 按 attention / FFN / residual / layernorm 这类模型语义直接建 family。
- [ ] 为了让 family 看起来更复杂而强行合并或拆分。
- [ ] 在 execution evidence 不足时，过早按资源瓶颈重分 family。

### 6.3 当前 family 可接受的状态

第一版中允许出现：

- [ ] 单 anchor family。
- [ ] family 与 anchor 边界高度相似。
- [ ] 小 workload 下没有大规模跨 anchor 合并。

但必须在文档或 artifact metadata 中说明：

- [ ] 这是小 workload 和细粒度 anchor 条件下的自然结果。
- [ ] family 在定义上仍然是硬件等价类，不是 anchor 的重命名。

---

## 7. 算法功能分组修改清单

算法分组不应再绑定具体算子或模型模块，而应改成更普适的计算功能角色。

### 7.1 推荐的算法功能标签

下一轮 regime 或 anchor metadata 中可优先使用以下标签：

- [ ] `primary_compute`
- [ ] `score_or_transform`
- [ ] `reduction_normalization`
- [ ] `aggregation_or_fusion`
- [ ] `elementwise_postprocess`
- [ ] `layout_or_data_movement`
- [ ] `constraint_or_bookkeeping`

### 7.2 标签定义检查

每个标签都应回答：

- [ ] 这个 kernel 在整体计算流程里承担什么功能？
- [ ] 它是否是主要算量来源？
- [ ] 它是否只是后处理、约束或回归检查对象？
- [ ] 它是否会影响同一硬件 family 内的 regime 拆分？

### 7.3 不应做的事

- [ ] 不要把 `algorithm_group` 写成 `qkv_projection`、`ffn_expand`、`softmax` 这类具体算子名。
- [ ] 不要让算法功能标签替代硬件 family。
- [ ] 不要让算法功能标签直接决定 lane。

---

## 8. regime 修改清单

regime 应成为硬件信息和算法功能信息汇合的位置。

### 8.1 regime 的拆分依据

每个 regime 都应至少说明：

- [ ] 它属于哪个 hardware family。
- [ ] 它对应哪个 algorithm function group。
- [ ] 它的 shape / context 是否与同 family 其他对象不同。
- [ ] 它的 resource signature 是否与同 family 其他对象不同。
- [ ] 它是否会改变 backend parameter direction。
- [ ] 它是否值得作为单独 backend validation object。

### 8.2 regime 的必要性说明

每个 regime 最好能回答：

- [ ] 如果把它并回同 family 其他 regime，会丢失什么信息？
- [ ] 如果把它单独保留，后端验证能获得什么额外信息？
- [ ] 它是主调参对象、辅助对象，还是 constraint / regression object？

### 8.3 当前 9-regime 结构的处理原则

- [ ] 暂时不为了形式美强行减少 regime 数量。
- [ ] 暂时不为了“更细”继续扩 regime。
- [ ] 优先补齐每个 regime 的硬件判据、算法功能判据和重要性权重。
- [ ] 等 execution evidence 回来后，再决定是否合并或拆分。

---

## 9. lane 修改清单

lane 应保持为 backend 参数方向和验证入口。

### 9.1 每条 lane 应说明

- [ ] 对应的 regime。
- [ ] 主要硬件敏感方向。
- [ ] 预期观察的资源或结构。
- [ ] 关联的 backend validation scenario。
- [ ] result summary 回来后应写回哪个 regime / family。

### 9.2 lane 不应承担的职责

- [ ] 不要把 lane 当成算法分组标签。
- [ ] 不要把 lane 当成 family 的别名。
- [ ] 不要让 lane 只描述“这是哪个模型模块”。

### 9.3 lane 与调参的关系

每条 lane 最好能落到至少一种可验证方向：

- [ ] register / occupancy
- [ ] shared memory
- [ ] cache / locality
- [ ] DRAM / memory pressure
- [ ] reduction path
- [ ] elementwise regression / constraint

---

## 10. YAML 修改清单

重点文件：

`docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml`

需要检查或补充：

- [ ] family 条目是否显式包含硬件判据。
- [ ] regime 条目是否显式包含算法功能角色。
- [ ] regime 条目是否包含必要性说明。
- [ ] lane 条目是否只描述 backend 验证入口和参数方向。
- [ ] 是否存在把模型模块名直接当 family 判据的写法。
- [ ] 是否存在把 anchor 名字直接复制成 family / regime 解释的写法。
- [ ] 是否有字段可以记录 `member_count / time_weight / workload_scale` 的来源或引用。

如果需要新增字段，建议优先考虑：

- [ ] `hardware_group_basis`
- [ ] `algorithm_function_group`
- [ ] `anchor_weight_source`
- [ ] `regime_split_rationale`
- [ ] `lane_parameter_direction`

---

## 11. builder 修改清单

重点文件：

`experiments/baseline_diagnosis/build_middle_layer.py`

需要检查或补充：

- [ ] builder 是否能读取或计算 anchor 权重信息。
- [ ] builder 是否能把权重信息写入 `anchors.json` 或 `bundle.json`。
- [ ] builder 是否能把 `algorithm_function_group` 写入 regime artifacts。
- [ ] builder 是否能把 hardware family 判据写入 family artifacts。
- [ ] builder 是否能把 lane 的 parameter direction 写入 lane artifacts。
- [ ] builder 是否对缺失权重字段有显式处理，不静默吞掉。
- [ ] builder 输出是否保持 deterministic。

---

## 12. artifact 修改清单

重点目录：

`artifacts/middle_layer/mini_transformer_v4/`

需要检查以下文件是否反映本轮语义：

- [ ] `anchors.json`
- [ ] `families.json`
- [ ] `regimes.json`
- [ ] `lanes.json`
- [ ] `importance_scoring_sheet.json`
- [ ] `writeback_lane_to_regime.json`
- [ ] `bundle.json`

每类 artifact 的最低要求：

- [ ] anchors 应包含代表关系和权重信息。
- [ ] families 应包含硬件分组依据。
- [ ] regimes 应包含算法功能角色和拆分理由。
- [ ] lanes 应包含 backend 参数方向。
- [ ] importance scoring 不应只依赖名称或人工排序。

---

## 13. 测试修改清单

重点测试文件可能包括：

- `tests/test_build_middle_layer.py`
- `experiments/baseline_diagnosis/tests/`
- 与 backend output 对接的相关测试

需要补充或确认：

- [ ] 缺失 anchor 权重字段时，builder 行为可预期。
- [ ] 每个 regime 都有 `algorithm_function_group` 或等价字段。
- [ ] 每个 family 都有硬件分组依据。
- [ ] 每条 lane 都有 backend parameter direction。
- [ ] 输出对象数在未主动调整规则时保持稳定。
- [ ] `bundle.json` 能继续被 C 线消费。
- [ ] backend pipeline 相关测试不因 schema 调整破坏。

---

## 14. 验收标准

本轮修改完成后，应满足：

- [ ] B 线文档中不再把算法分组写成算子名分组。
- [ ] family 的定义明确是硬件等价类。
- [ ] regime 明确承接算法功能角色和硬件 family 的交汇。
- [ ] lane 明确是 backend 参数方向入口。
- [ ] anchor 权重信息进入 B 线 artifacts 或被明确标记为暂缺。
- [ ] 当前 `mini_transformer_v4` 的对象链能从 anchor 解释到 lane。
- [ ] 现有测试通过。
- [ ] 若 schema 有变化，C 线消费方同步更新或保持兼容。

---

## 15. 本轮不做的事

为避免 scope 膨胀，本轮先不做：

- [ ] 不重新实现完整 A 线 PKA-core selector。
- [ ] 不扩大 workload。
- [ ] 不把 current 9 regimes 强行改成新的数量。
- [ ] 不把 execution bridge 的 smoke / validation 语义问题混进 B 线修改。
- [ ] 不宣称当前 family 已经被正式 validation 证明。
- [ ] 不把 STEM+ROOT 完整复刻进当前 pipeline。

---

## 16. 建议执行顺序

建议下一轮 RLCR 按这个顺序执行：

1. 更新 B 线文档口径。
2. 检查并修改 YAML rule config。
3. 修改 middle-layer builder 的字段透传和生成逻辑。
4. 重新生成 middle-layer artifacts。
5. 更新或补充 tests。
6. 运行 B 线和 C 线相关回归。
7. 检查 artifacts 是否能支持下一轮 backend planning。

---

## 17. 最终判断

下一轮 B 线修改的核心不是增加对象数量，而是让现有对象更有方法含义。

最重要的改动方向是：

**让 family 更硬件化，让 regime 更明确地承接算法功能信息，让 A 线权重信息进入 B 线重要性判断。**

只有这样，B 线才不会退化成对 A 线结果的重命名，也不会退化成按算子名贴标签的解释层。
