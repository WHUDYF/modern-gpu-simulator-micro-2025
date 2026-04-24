# 第一版工作分层地图

日期：2026-04-24

## 1. 文档目的

这份文档用于把当前第一版工作的组成部分系统拆开，作为后续优化、改进、排优先级时的统一参考。

当前仓库的第一版工作已经不再只是：

- 若干分析文档
- 若干中间 JSON
- 若干独立脚本

而是已经形成了一条可运行的方法链：

`frontend anchor -> middle structure -> backend planning -> execution bridge -> result summary -> writeback interface`

因此，后续优化不应该再笼统地说“改 B 线”或“补后端”，而应该明确：

1. 当前每一层的目标是什么
2. 当前每一层已经做到了什么
3. 每一层的核心文件在哪里
4. 每一层目前还差什么
5. 下一步优化应该优先打在哪一层

---

## 2. 当前第一版总状态

当前第一版已经具备：

- A 线：frontend anchor 输入层
- B 线：middle structure 对象层
- C 线：backend planning 层
- backend execution bridge 第一版
- result summary 与 writeback interface
- 对应测试与阶段性真实 smoke execution

当前最准确的状态不是“全部完成”，而是：

**第一版主干已经完成，当前正在做最后一层执行语义收尾。**

这里所谓“执行语义收尾”，主要是指：

- 区分 `smoke execution`
- 区分 `formal validation`
- 避免把 smoke 成功误写成正式 validation 成功

---

## 3. 第一版工作分层表

| 层级 / 模块 | 目标 | 核心文件 | 当前状态 | 下一步优化点 |
|---|---|---|---|---|
| A 线：frontend anchor 输入层 | 把原始 invocation 压成后段可消费的 representative anchors | `experiments/baseline_diagnosis/build_frontend_anchor_outputs.py` `experiments/baseline_diagnosis/build_invocation_table.py` `experiments/baseline_diagnosis/frontend_anchor/` `experiments/mini_transformer/frontend_anchor_sources/` | 已完成第一版。当前已有 `14` 条 invocation records、`8` 条 representative anchors、`3` 条 comparison methods。已能生成 `kernel_invocation_table / representative_anchor_table / comparison_table / case_note / frontend note`。 | 提高 anchor 的输入稳定性与解释性；继续检查 `member_invocations / coverage_weight / time_weight` 是否足够稳；后续如需提升，可补 stronger shape/context 信号，但不建议先重写此层。 |
| B 线：middle structure 对象层 | 把 frontend anchor 提升成 simulator-side decision objects：`anchor / family / regime / lane` | `experiments/baseline_diagnosis/build_middle_layer.py` `docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml` `artifacts/middle_layer/mini_transformer_v4/` | 已完成第一版主干。当前已有 `9` anchors、`4` families、`9` regimes、`9` lanes，并产出 `importance_scoring_sheet / writeback_lane_to_regime / bundle`。这一层已经不是文档整理，而是 YAML + builder + artifacts 的对象系统。 | 后续最主要优化是 rule config 的机制边界：哪些 family 该并、哪些 regime 该拆、哪些 lane 的 parameter direction 太弱。当前不建议大动对象数量，建议先观察 execution evidence 再调整。 |
| C 线：backend planning 层 | 把 middle structure 对象转成 validation planning、scenario matrix、baseline plan、writeback map | `experiments/backend_pipeline/backend_builder.py` `experiments/backend_pipeline/build_backend_outputs.py` `experiments/backend_pipeline/plan_backend_validation.py` `experiments/backend_pipeline/apply_backend_writeback.py` `experiments/backend_pipeline/results/mini_transformer_v4/` | 已完成第一版 planning 层。当前已有 `36` 条 run manifest、priority lane table、scenario matrix、validation worksheet、writeback map。已形成规划闭环和接口闭环。 | 后续优化点是 planning 与真实 execution 的一致性：manifest 是否过密、priority source 是否合理、scenario 分配是否真有信息增益。 |
| backend execution bridge v1 | 把 `backend_run_manifest` 真正接到 simulator 执行层，形成 `command plan -> execution -> parser -> result summary` | `experiments/backend_pipeline/execution_bridge.py` `experiments/backend_pipeline/workload_profiles.py` `experiments/backend_pipeline/run_backend_execution.py` `experiments/backend_pipeline/tests/test_execution_bridge.py` | 第一版主体已完成。当前已经有真实 command plan、run dir、stdout/stderr、parser report、execution records、result summary。还做了多轮 review 修复。 | 当前最优先优化点不是“更多功能”，而是“执行语义校正”：默认 `validation` 路径和显式 `smoke` 路径必须彻底分开。 |
| result summary 层 | 把运行结果变成统一结构化证据，供后续 writeback 使用 | `experiments/backend_pipeline/results/mini_transformer_v4/backend_result_summary_v1.json` `experiments/backend_pipeline/execution_bridge.py` | 已能产出第一版 summary。当前已有 `4` 条 result summary rows，并且 `sim_cycles` 已可从真实 run 的 `stdout.log` 中解析。 | 最关键的优化是明确 `execution success` 与 `validation success` 的区别，避免 smoke runs 被误包装成 formal validation evidence。 |
| writeback interface 层 | 把结果摘要回写到 regime / family / anchor 的状态层 | `experiments/backend_pipeline/apply_backend_writeback.py` `experiments/backend_pipeline/results/mini_transformer_v4/backend_writeback_map_v1.json` `experiments/backend_pipeline/results/mini_transformer_v4/backend_validation_status_v1.json` | 已有 writeback 协议与状态更新逻辑，也已有对应测试。当前已修复“parse-failed 不应提升状态”“成功 run 不应被失败 comparison 覆盖”等问题。 | 后续优化点是：让 smoke-mode 结果只做记录，不直接 promotion 到 `validated`；同时保持 writeback 对 formal validation 的升级路径清晰。 |
| 测试与回归层 | 确保 A/B/C 线与 execution bridge 在工程上可复现、可回归 | `experiments/baseline_diagnosis/tests/` `experiments/backend_pipeline/tests/` `tests/test_build_middle_layer.py` | 第一版回归基础已经建立。当前 backend 相关测试集合可通过，execution bridge 也已有专门测试。 | 后续优化重点是补“语义正确性”测试，不只是“命令能跑通”测试。尤其要补 smoke / validation 模式分离后的 regression tests。 |
| 文档 / 基线层 | 固定当前第一版的解释口径、设计边界和优化顺序 | `docs/experiment-round2-merged-summary-2026-04-23.md` `docs/superpowers/specs/2026-04-24-backend-execution-bridge-design.md` `docs/superpowers/plans/2026-04-24-backend-execution-bridge-implementation.plan.md` | 当前已经有 merged summary、execution bridge design、implementation plan 等基线文档。 | 后续不建议先扩更多新文档，而应以代码和 artifacts 为主，文档只做收敛性更新。 |

---

## 4. 当前第一版的核心成果

如果把当前第一版压成最关键的几条成果，可以概括为：

1. 已经有稳定的 frontend anchor 输入层，而不是停留在原始 invocation 数据。
2. 已经有 middle-layer object system，而不是停留在“压缩结果 + prose 解释”。
3. 已经有 backend planning artifacts，而不是只说“后面可以验证”。
4. 已经把 `backend_run_manifest` 接到了真实 simulator 执行侧，哪怕目前还是 smoke-first 的执行桥第一版。
5. 已经有 result summary 和 writeback interface，不再只是“跑完看日志”。

因此，第一版最大的意义不是“做完了所有工作”，而是：

**已经把方法链从概念、结构、规划推进到了初步执行。**

---

## 5. 当前第一版的主要风险

当前第一版最主要的风险，不再是“对象层没建起来”，而是下面三类问题：

### 5.1 smoke 与 formal validation 语义混淆

这是当前最核心的风险。

如果：

- `smoke trace`
- `gpgpu_max_cycle` 限制
- parser 解析出的最小 cycle

被直接包装成正式 validation 成功，
那么 execution bridge 就会在方法语义上产生误导。

### 5.2 B 线对象边界过早固化

当前 family / regime / lane 已经稳定到可用，
但它们还不应被过早视为“最终真值”。

更合理的做法是：

- 先通过 execution evidence 看哪些对象真的值得保持拆分
- 再决定是否合并或继续细分

### 5.3 execution bridge scope 膨胀

当前 execution bridge 最合理的边界是：

`manifest -> command plan -> smoke/full execution -> minimal result summary`

如果过早把：

- 大规模批跑系统
- 多 workload 同时支持
- 完整自动 writeback orchestration
- 丰富统计分析

全部塞进第一版，会让这层失控。

---

## 6. 后续优化优先级建议

当前最合理的优化顺序如下：

### Priority 1：收尾 execution 语义分离

先明确：

- 默认 profile 走 `full trace`
- 只有显式 `smoke mode` 才走 trimmed trace
- smoke 成功不能自动升格为 formal validation success

这一步完成之后，第一版才真正从“能跑”变成“证据语义干净”。

### Priority 2：固定 execution bridge v1 的 schema

包括固定：

- `execution_mode`
- `result_status`
- `parse_status`
- `parser_report`
- `backend_result_summary_v1.json`

让后续 writeback 和分析层不再被字段漂移影响。

### Priority 3：用 execution evidence 回看 B 线规则

等 execution bridge 语义干净之后，再回头看：

- 哪些 family 合理
- 哪些 regime 拆得过细
- 哪些 lane 没信息增益

这样调整才有证据基础。

### Priority 4：扩大验证范围

最后才考虑：

- 更多 scenario
- 更多 priority source 对照
- 更多 workload
- 更完整 writeback 回流

---

## 7. 建议的后续使用方式

后续做优化时，建议始终按下面的方式定位问题：

1. 先判断问题属于哪一层：
   - 输入层
   - 结构层
   - 规划层
   - 执行层
   - 证据语义层

2. 再判断是：
   - 工程 bug
   - schema 不稳
   - 对象边界问题
   - 方法语义问题

3. 最后再决定：
   - 是修 execution bridge
   - 是修 YAML rule config
   - 还是回到 frontend anchor

这样可以避免后续优化时把所有问题都混成“B 线还需要继续改”。

---

## 8. 结论

当前第一版已经不是早期探索态，而是：

**可运行、可测试、可解释、可继续优化的 v1 方法基线。**

它最重要的价值不在于“所有部分都最终定型了”，而在于：

- 方法链各层已经成型
- 层与层之间已经有接口
- 已经有初步执行证据
- 后续优化已经可以按模块、有依据地推进

因此，接下来最好的推进方式不是再整体泛化讨论，
而是按这份分层地图逐层优化。
