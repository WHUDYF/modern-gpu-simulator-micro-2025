# 第二次实验汇总文档（A/B/C 合流版）

日期：2026-04-23

## 1. 文档目的

这份文档用于记录当前 `mini_transformer_v4` 方法链在 A/B/C 三条线合流后的统一状态，
作为第二次实验的直接工作入口。

当前分支已经不再只是前端锚点线，而是一个可执行的合流分支：

- A 线：前端锚点线
- B 线：中端结构线
- C 线：后端验证线

因此，这份文档的目标不是重复设计过程，而是明确：

1. 当前分支已经合入了什么
2. 三条线各自的入口、产物和边界是什么
3. 第二次实验应该按什么顺序运行和汇总

---

## 2. 当前分支状态

当前工作分支：

- `dyf/docs/frontend-anchor-model`

本次合流已经包含两个 merge commit：

- `4caad3c`
  - `Merge branch 'round1/backend-output-layer' into dyf/docs/frontend-anchor-model`
- `f611cd1`
  - `Merge branch 'round1/middle-structure-layer' into dyf/docs/frontend-anchor-model`

这意味着当前分支已经同时包含：

- A 线前端锚点实现
- B 线 family / regime / lane / scoring sheet / YAML rule config
- C 线 backend validation planning / scenario matrix / writeback mapping

当前统一方法链可以写成：

`frontend compression -> representative anchors -> family -> representative regime -> importance / priority -> backend validation`

---

## 3. 三条线当前入口

### 3.1 A 线：前端锚点线

主入口：

- [build_frontend_anchor_outputs.py](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/build_frontend_anchor_outputs.py:1)
- [build_invocation_table.py](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/build_invocation_table.py:1)

主要输入：

- [mini_transformer_v4_identity.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/frontend_anchor_sources/mini_transformer_v4_identity.json:1)
- [mini_transformer_v4_features.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/frontend_anchor_sources/mini_transformer_v4_features.json:1)
- [squash.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mechanisms/squash.json:1)

当前产物目录：

- [frontend_anchor_v1](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/mini_transformer_v4/frontend_anchor_v1)

当前对象规模：

- `KernelInvocationRecord`：14 条
- `Representative Anchor Table v1`：8 条
- `Comparison Table`：3 条方法对照

当前定位：

- A 线已经能提供稳定的 `representative anchors`
- 仍然属于 `frontend anchor v1`
- `member_invocations / coverage_weight / time_weight` 已可下游消费
- `shape_hint_summary` 仍是 placeholder

### 3.2 B 线：中端结构线

主入口：

- [build_middle_layer.py](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/build_middle_layer.py:1)

主配置：

- [mini_transformer_middle_layer_rules_v1_2026-04-22.yaml](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml:1)

当前产物目录：

- [artifacts/middle_layer/mini_transformer_v4](/home/dyf/modern-gpu-simulator-micro-2025/artifacts/middle_layer/mini_transformer_v4)

当前对象规模：

- `anchors.json`：9 条
- `families.json`：4 条
- `regimes.json`：9 条
- `lanes.json`：9 条

当前 B 线还同时产出：

- `importance_scoring_sheet.json`
- `writeback_lane_to_regime.json`
- `bundle.json`
- 对应 Markdown snapshots

当前定位：

- B 线已经不只是 prose 文档，而是 builder + YAML config + artifacts
- 当前 middle-layer rule 的 source of truth 是单文件 YAML
- 当前重点对象已经从 “压缩后的 kernel” 过渡为：
  - anchor
  - family
  - regime
  - lane

### 3.3 C 线：后端验证线

主入口：

- [build_backend_outputs.py](/home/dyf/modern-gpu-simulator-micro-2025/experiments/backend_pipeline/build_backend_outputs.py:1)
- [backend_builder.py](/home/dyf/modern-gpu-simulator-micro-2025/experiments/backend_pipeline/backend_builder.py:1)

当前产物目录：

- [experiments/backend_pipeline/results/mini_transformer_v4](/home/dyf/modern-gpu-simulator-micro-2025/experiments/backend_pipeline/results/mini_transformer_v4)

当前对象规模：

- `backend_anchor_table_v1.json`：6 条
- `backend_family_table_v1.json`：4 条
- `backend_regime_table_v1.json`：6 条
- `backend_priority_lane_table_v1.json`：40 条
- `backend_scenario_matrix_v1.json`：24 条
- `backend_run_manifest_v1.json`：36 条
- `backend_writeback_map_v1.json`：9 条

当前定位：

- C 线已经把 `importance-guided validation` 的后端对象和 baseline 对照框架写成了结构化 JSON
- 但它仍然主要是规划/映射层，不是完整 simulator run 结果层
- `backend_result_summary_v1.json` 与 `backend_writeback_updates_v1.json` 目前仍为空，说明这条线已到“验证编排可跑”，但还没有进入完整实验回写阶段

---

## 4. 当前统一口径

第二次实验应统一采用下面这套口径：

### 4.1 A 线负责输入可信性

A 线负责回答：

- representative anchors 从哪里来
- 为什么 `hybrid` 比简单 baseline 更有价值
- `member_invocations / coverage / time` 是否可解释

A 线当前不负责输出：

- family
- regime
- simulator lane

### 4.2 B 线负责结构对象化

B 线负责回答：

- anchor 如何进入 family
- family 为什么还要继续拆成 regime
- importance / priority 如何挂在这些结构对象上

B 线当前是第二次实验最核心的方法对象层。

### 4.3 C 线负责验证路径

C 线负责回答：

- 哪些 regime 值得先验证
- 每个 regime 进入哪个 lane
- baseline 如何定义
- parameter scenarios 如何组织

因此，C 线当前最重要的角色是：

**把 importance 从“排序概念”变成“可执行验证计划”。**

---

## 5. 第二次实验的建议运行顺序

建议按 A -> B -> C 顺序重新生成，并保持每一步结果单独落盘。

### Step 1：重跑 A 线

```bash
python3 experiments/baseline_diagnosis/build_frontend_anchor_outputs.py \
  --identity-json experiments/mini_transformer/frontend_anchor_sources/mini_transformer_v4_identity.json \
  --features-json experiments/mini_transformer/frontend_anchor_sources/mini_transformer_v4_features.json \
  --squash-json experiments/mini_transformer/mechanisms/squash.json \
  --output-dir experiments/baseline_diagnosis/results/mini_transformer_v4/frontend_anchor_v1
```

重点检查：

- invocation 数是否仍为 14
- hybrid anchor 数是否仍稳定
- `case_note_v1.md` 是否仍能给出 split case

### Step 2：重跑 B 线

```bash
python3 experiments/baseline_diagnosis/build_middle_layer.py
```

如果需要指定 rule config：

```bash
python3 experiments/baseline_diagnosis/build_middle_layer.py \
  --rule-config docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml \
  --output-dir artifacts/middle_layer/mini_transformer_v4
```

重点检查：

- anchors / families / regimes / lanes 数量是否稳定
- `bundle.json` 的 metadata 是否正确记录当前 rule config
- `importance_scoring_sheet.json` 和 `writeback_lane_to_regime.json` 是否同步更新

### Step 3：重跑 C 线

```bash
python3 experiments/backend_pipeline/build_backend_outputs.py \
  --input experiments/mini_transformer/mini_transformer_v4_full.json \
  --output-dir experiments/backend_pipeline/results/mini_transformer_v4
```

重点检查：

- `backend_priority_lane_table_v1.json`
- `backend_scenario_matrix_v1.json`
- `backend_run_manifest_v1.json`
- `backend_validation_worksheet_v1.json`

---

## 6. 第二次实验建议重点记录的对比项

第二次实验不建议只记录“文件生成成功”，而应至少汇总下面几类差异：

### 6.1 A 线差异

- `hybrid` 与 `name-only / pka-like-coarse` 的 anchor 数差异
- split case 是否变化
- `coverage_weight / time_weight` 是否稳定

### 6.2 B 线差异

- anchor -> family -> regime 的对象数是否变化
- 哪些 family / regime 的 boundary 状态变化
- `importance_score / regime_priority_score` 排序是否变化
- YAML rule config 是否有新增/删减规则

### 6.3 C 线差异

- 高优先级 lane 是否变化
- scenario 分配是否变化
- baseline plan 与 run manifest 是否变化
- 是否开始出现非空的 result summary / writeback updates

---

## 7. 当前稳定部分与仍然 provisional 的部分

### 7.1 相对稳定

- A 线 dual-source 输入链
- A 线 `Representative Anchor Table v1`
- B 线单文件 YAML rule config
- B 线 anchors / families / regimes / lanes artifacts
- C 线 baseline / scenario / lane / run-manifest 结构

### 7.2 仍然 provisional

- A 线部分字段仍是 placeholder 或 derived approximation
- B 线 `decision_weight` 仍带人工判断色彩
- B 线 family / regime 划分仍依赖当前 rule config，不应过度表述为“最终真值”
- C 线目前更接近 validation orchestration，而不是完整 simulator evidence writeback

因此，第二次实验的合理目标不是“把所有层都做成最终版”，而是：

**把 A/B/C 三层之间的接口、对象和验证顺序固化下来。**

---

## 8. 当前建议的第二次实验最小闭环

如果这次只保一个最小闭环，建议保下面 6 项：

1. A 线重新生成 frontend anchor artifacts
2. B 线重新生成 middle-layer bundle
3. C 线重新生成 backend planning artifacts
4. 记录 A/B/C 三层对象数量与排序变化
5. 记录 rule config 与 lane mapping 的变化
6. 输出一份 round-2 差异总结

只要这 6 项完成，就能说明：

- 合流分支可跑
- 三条线接口已打通
- 第二次实验已经从“单线试做”进入“方法链联合演化”

---

## 9. 现阶段推荐的统一检查命令

```bash
python3 -m pytest experiments/baseline_diagnosis/tests experiments/backend_pipeline/tests tests -q
```

当前该测试集合在合流分支上已经通过，可作为第二次实验前后的统一 sanity check。

---

## 10. 结论

当前分支已经不是单纯的 frontend 支线，而是：

**A 线输入、B 线结构、C 线验证编排同时存在的实验主工作分支。**

因此，第二次实验最应该做的不是再单独补某一条线，而是：

- 固定 A/B/C 接口
- 重新生成三层 artifacts
- 记录对象、分数、lane 和 baseline 计划的变化
- 为后续真正的 simulator-side round 提供统一入口
