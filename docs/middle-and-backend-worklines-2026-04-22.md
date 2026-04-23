# 中间结构线与后端验证线当前任务说明

日期：2026-04-22

## 1. 文档目的

这份文档用于把当前方法线中最关键的两条工作线拆清楚：

1. `中间结构线`
2. `后端验证线`

当前我们已经确定：

- 前端以 `PKA` 为锚点
- `STEM+ROOT` 作为异质性 refinement 的重要启发
- 我们的主贡献位于 compression 之后

因此，当前最需要明确的不是“前端还差多少”，而是：

**在 compression 之后，我们自己的两条核心工作线到底分别需要做什么。**

---

## 2. 当前总方法线位置

当前最合理的方法链可以写成：

`frontend compression -> representative anchors -> family -> representative execution regime -> importance ratio -> tuning priority -> simulator validation`

在这条链路里：

- 前端已经有比较清楚的方向
- 但真正属于我们的方法部分，主要集中在：
  - `family / regime / importance`
  - `tuning priority / simulator validation`

因此当前需要重点推进的是：

1. `中间结构线`
2. `后端验证线`

---

## 3. 中间结构线：它的目标是什么

中间结构线的核心目标是：

**把 representative kernels 继续组织成 simulator 可理解、可比较、可排序的结构化对象。**

更具体地说，它解决的是：

- representative anchors 如何变成 family
- family 如何继续拆成 representative execution regimes
- importance ratio 如何真正挂在这些对象上

如果没有这条线，前端压缩得到的对象仍然只是：

- 少量 representative kernels

而不会自然变成：

- 可进入 simulator side reasoning 的对象

---

## 4. 中间结构线当前需要做什么

### 4.1 建立 Representative Anchor Table

这是中间结构线的输入层。

当前至少需要把下面这些字段固定下来：

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

这一步的意义在于：

**让后续 family 层不再面对抽象的“压缩结果”，而是面对一张明确的输入表。**

### 4.2 完成 Anchor -> Family 的映射

这一步要解决的问题是：

- 哪些 anchors 可以进入同一 family
- 哪些 anchors 必须分开
- 哪些对象仍处于 boundary / unresolved 状态

这里必须继续使用当前已形成的判据：

- `phase`
- `route primitive`
- `hardware execution template`
- `boundary-first`

### 4.3 完成 Family Table 的真实字段填充

当前 Family Table 已经有第一版对象，但很多字段仍然是半定量。

接下来要做的是补齐：

- `coverage_weight` 的来源说明
- `time_weight` 的来源说明
- `decision_weight` 的 provisional 赋值依据
- `importance_score` 的实际计算过程

### 4.4 完成 Family -> Regime 的拆分

这一步要真正回答：

- 为什么同一个 family 还不够
- 哪些对象在 family 内部还要继续拆成多个 regime

当前 regime 的拆分建议至少依据：

- `shape_regime`
- `context_scope`
- `resource_signature`

### 4.5 把 importance ratio 挂到 Family / Regime 上

中间结构线最终必须至少得到：

- family-level importance
- regime-level priority

否则 family / regime 仍然只是分类对象，而不是：

- 决策对象

### 4.6 继续补 boundary cases

边界判据仍然是中间结构线最重要的“硬度来源”。

当前最需要做的不是再多列对象，而是继续逼问：

- 哪些看起来相似的对象为什么不能并
- 哪些不同 route 的对象为什么仍然 weak-share
- 哪些 family 内部必须继续拆 regime

---

## 5. 中间结构线的当前阶段产物

当前这条线的阶段产物应至少包括：

1. `Representative Anchor Table`
2. `Family Table`
3. `Regime Table`
4. `Importance Scoring Sheet`
5. `Boundary Case Notes`

如果这五样东西都存在，那么 compression 之后的结构层就基本成立了。

---

## 6. 后端验证线：它的目标是什么

后端验证线的核心目标是：

**证明 importance ratio 与 tuning priority 不是概念，而是真的能指导 simulator-side decision making。**

也就是说，后端线不再主要回答：

- family 合不合理

而是回答：

- high importance family 是否真的更值得先调
- regime priority 是否真的能改善验证流程
- importance-guided tuning 是否比 baseline 更有效

---

## 7. 后端验证线当前需要做什么

### 7.1 定义 Simulator Lane Mapping

首先要回答：

- 每个高优先级 regime 后面准备接哪类 simulator lane
- 哪些 regime 对应哪类参数扰动
- 哪些 regime 更适合作为主验证对象

如果没有 lane mapping，regime 仍然只是静态对象。

### 7.2 定义 Family-Level Tuning Targets

对于每个 family，至少要给出当前最合理的 tuning direction：

- `Dense Tiled Compute`
  - register / occupancy / tiled compute path
- `Reduction / Normalize`
  - cache-capacity / reduction / normalization path
- `Streaming Aggregation`
  - locality / L1 / aggregation path
- `Elementwise Fusion`
  - constraint / regression checking

这一步的作用是：

**把 family importance 映射成真正的调参方向。**

### 7.3 定义 Baseline 对照组

importance ratio 要证明自己有价值，必须有 baseline。

当前至少要对照：

- `No Priority`
- `Time-Only Priority`
- `Name-Based / Manual Priority`
- `Importance-Guided Priority`

### 7.4 定义第一组 Parameter Scenarios

不要一开始追求全设计空间。

第一版只需要选少量有代表性的参数方向，例如：

- register-sensitive
- cache-sensitive
- locality-sensitive
- reduction-sensitive

### 7.5 定义验证指标

后端线至少应明确下面几个验证指标：

- Top-K family 覆盖多少 time weight
- high importance family 是否更可能对关键参数敏感
- importance-guided 顺序是否更早发现有效 tuning object
- 在同等预算下，importance-guided 是否优于 baseline

### 7.6 想清楚结果如何回写

后端验证线还必须考虑：

- simulator 结果如何回写到 family
- family 结果如何回写到 regime
- regime 结果如何回写到原始 workload 解释

如果没有回写逻辑，importance ratio 最后只会变成：

- 独立局部结论

而不是：

- 端到端方法的一部分

---

## 8. 后端验证线的当前阶段产物

当前这条线的阶段产物应至少包括：

1. `Simulator Lane Mapping Table`
2. `Family-Level Tuning Target Table`
3. `Baseline Comparison Note`
4. `Parameter Scenario List`
5. `Importance Validation Worksheet`

如果这些东西存在，那么 importance ratio 就开始从“定义”变成“可验证对象”。

---

## 9. 两条线之间的关系

### 9.1 中间结构线解决“对象是否成立”

它主要回答：

- family 合不合理
- regime 能不能稳定定义
- priority 字段有没有合理来源

### 9.2 后端验证线解决“对象是否有用”

它主要回答：

- family importance 能不能指导 tuning
- regime priority 能不能改善 validation workflow

所以：

- 中间结构线是对象构造层
- 后端验证线是对象证明层

两者不能互相替代。

---

## 10. 当前最小闭环如何拆到两条线上

当前最小闭环可以拆成：

### 中间结构线必须完成

1. `Representative Anchor Table`
2. `Family Table`
3. `Regime Table`
4. `Importance Template`

### 后端验证线必须完成

1. `Lane Mapping`
2. `Baseline Definition`
3. `Parameter Scenario`
4. `Validation Metrics`

只要这两部分都成立，你们的方法就不再只是“整理结构”，而是真正开始进入：

**compression 之后的 simulator decision layer**

---

## 11. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. 中间结构线的任务，是把 representative kernels 组织成 family / regime / importance 这些结构对象。
2. 后端验证线的任务，是证明这些结构对象真的能指导 simulator tuning priority。
3. 前者解决“对象是否成立”，后者解决“对象是否有用”。
4. 当前最重要的，不是等待前端完全建立，而是让这两条线并行推进，先形成最小闭环。
