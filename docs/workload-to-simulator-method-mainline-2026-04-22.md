# 从 Workload 到 Simulator 的方法主线总文档

日期：2026-04-22

## 1. 文档目的

这份文档用于把当前已经形成的整条方法线收束成一个统一入口文档。

当前我们已经陆续生成了：

- 前端锚点相关文档
- family / regime 相关文档
- importance ratio 相关文档
- 后端验证计划
- related work 总结

但如果没有一份总文档，后续无论是：

- 自己继续推进
- 给导师汇报
- 在其他 terminal 中交给 RLCR / agent 继续工作

都会反复丢上下文。

因此，这份文档的目标是：

1. 固定当前最稳的方法主线
2. 说明每一层到底解决什么问题
3. 给出当前已形成的核心文档索引
4. 说明接下来最优先的推进方向

---

## 2. 当前最稳的主问题

当前我们最稳的主问题定义是：

**在已有 representative compression 工作之后，如何把压缩后的 workload 对象继续组织成 simulator 可用、可验证、可调参的结构化决策对象。**

换句话说：

- 现有工作已经能帮助我们回答“压谁”
- 我们要继续回答：
  - “怎么组织”
  - “先调谁”
  - “怎么验证”

因此，这项工作的主价值不在于再次提出新的 sampling trick，而在于：

**补上 compression 之后的 simulator-side decision layer。**

---

## 3. 当前最稳的方法链

当前最推荐的完整方法链写法是：

`frontend compression -> representative anchors -> squash / family / regime -> importance ratio -> tuning priority -> simulator validation`

更具体地说，可以拆成下面几层：

### 3.1 前端锚点层

这一层当前以：

- `PKA` 作为前端锚点
- `STEM+ROOT` 作为异质性 refinement 启发

作用是：

- 从完整 workload 中得到可信的 representative anchors

### 3.2 时间结构层

由：

- `squash`

负责。

作用是：

- 从长 trace 中提取稳定 phase
- 给后续 family 层补回 phase context

### 3.3 共享机制层

由：

- `batch`
- `family`

负责。

作用是：

- 把 representative anchors 继续组织成共享机制 family
- 明确哪些对象共享执行模板
- 明确哪些对象必须作为边界或例外保留

### 3.4 执行区间层

由：

- `representative execution regime`

负责。

作用是：

- 在 family 内继续按 shape / context / resource 拆出最终 simulator 对象

### 3.5 决策层

由：

- `importance ratio`

负责。

作用是：

- 给 family / regime 排优先级
- 生成 tuning priority

### 3.6 验证层

由：

- `simulator lane`
- `baseline comparison`
- `priority-guided validation`

负责。

作用是：

- 验证 importance-guided 策略是否真的改善后续 workflow

---

## 4. 当前与相关工作的关系

### 4.1 PKA

PKA 解决的是：

- representative kernel compression

它的价值是：

- 让大 workload 变得可模拟

它在我们方法中的角色是：

- **frontend anchor**

### 4.2 Sieve

Sieve 解决的是：

- work-size stratification
- strata 内 execution time variance 控制

它的重要启发是：

- grouping 不能只看行为相似
- 还必须显式控制工作量尺度

### 4.3 Photon

Photon 解决的是：

- 在线执行路径结构驱动的 adaptive sampling

它的重要启发是：

- 在线执行结构本身可以成为有效特征

### 4.4 STEM+ROOT

STEM+ROOT 解决的是：

- runtime heterogeneity
- refined cluster construction
- sample budget optimization

它的重要启发是：

- 前端对象不能忽略 invocation 级 runtime distribution 差异

### 4.5 GCL-Sampler

GCL-Sampler 解决的是：

- learned kernel similarity

它提醒我们：

- 不要把自己的贡献点重新压回“更好的 kernel clustering”

### 4.6 GainSight

GainSight 解决的是：

- workload-derived structure -> hardware decision

它给我们的支撑是：

- 证明“从 workload 提结构化信号，再进入硬件决策”这条方法论是成立的

---

## 5. 当前已经形成的核心判断

当前这条方法线已经形成了几条相对稳定的认识：

### 5.1 前端必须有锚点

如果没有前端锚点，reviewer 很容易认为：

- 输入对象不稳
- 后段方法结论依赖前端偏差

所以当前最稳的选择是：

- `PKA` 作为主锚点
- `STEM+ROOT` 作为 refinement 启发

### 5.2 family 不等于 representative cluster

当前必须坚持：

- PKA cluster 是 representative compression 的结果
- family 是共享机制组织层

两者相关，但不等同。

### 5.3 family 不等于最终 simulator 对象

family 之后还必须有：

- `representative execution regime`

否则：

- shape
- context
- resource signature

这些差异会被错误揉平。

### 5.4 importance ratio 是真正的后端决策层

当前方法的真正独特价值不在于：

- family 分类本身

而在于：

- 从 family / regime 中提取 importance ratio
- 把 compression output 推进成 tuning priority

---

## 6. 当前已有核心文档索引

下面这些文档构成了当前方法线的主骨架。

### 6.1 方向与方法定位

- [current-goal-and-method-clarification-2026-04-19.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/current-goal-and-method-clarification-2026-04-19.md)
- [iteration-analysis-report-2026-04-19.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/iteration-analysis-report-2026-04-19.md)
- [family-method-positioning-and-abstracts.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family-method-positioning-and-abstracts.md)

### 6.2 family / regime 核心协议

- [family_selection_boundary_protocol.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/family_selection_boundary_protocol.md)
- [representative_execution_regime_protocol.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/representative_execution_regime_protocol.md)
- [mini_transformer_v4_route_primitive_template_table.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/mini_transformer_v4_route_primitive_template_table.md)

### 6.3 前端锚点与接口

- [pka-interface-notes-2026-04-20.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/pka-interface-notes-2026-04-20.md)
- [pka-to-family-interface-design-2026-04-20.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/pka-to-family-interface-design-2026-04-20.md)
- [why-we-need-a-frontend-anchor-2026-04-22.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/why-we-need-a-frontend-anchor-2026-04-22.md)

### 6.4 importance 与验证

- [family-importance-weight-definition-2026-04-22.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family-importance-weight-definition-2026-04-22.md)
- [family-table-schema-with-importance-2026-04-22.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family-table-schema-with-importance-2026-04-22.md)
- [importance-ratio-validation-plan-2026-04-22.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/importance-ratio-validation-plan-2026-04-22.md)
- [middle-and-backend-worklines-2026-04-22.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/middle-and-backend-worklines-2026-04-22.md)

### 6.5 对象表

- [mini_transformer_representative_anchor_table_v1_2026-04-22.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/mini_transformer_representative_anchor_table_v1_2026-04-22.md)
- [mini_transformer_family_regime_table_v1_2026-04-22.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/mini_transformer_family_regime_table_v1_2026-04-22.md)

### 6.6 相关工作总结

- [related-work-summary-2026-04-22.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/related-work-summary-2026-04-22.md)
- [compression-dimensions-summary-2026-04-21.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/compression-dimensions-summary-2026-04-21.md)
- [research-overlap-assessment-2026-04-20.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/research-overlap-assessment-2026-04-20.md)

---

## 7. 当前已经形成的最小闭环

当前已经形成的最小闭环可以写成：

`representative anchors`
-> `family`
-> `regime`
-> `importance ratio`
-> `tuning priority`
-> `validation plan`

这里仍然有不少字段是：

- half-quantitative
- provisional

但最重要的是：

**方法对象已经齐了。**

这意味着当前工作已经从“纯概念”进入：

- 接口可定义
- 对象可填表
- 优先级可排序
- 验证可规划

---

## 8. 当前最需要继续推进的方向

如果只选最值得优先做的事情，当前建议顺序如下：

### 8.1 用真实字段替换 anchor / family / regime 表中的半定量字段

优先替换：

- `coverage_weight`
- `time_weight`
- `decision_weight`

### 8.2 明确 anchor -> family -> regime 的字段映射

把三张表真正接死，避免后续实现时再反复摇摆。

### 8.3 设计第一组 simulator parameter scenarios

让：

- dense family
- reduction family
- aggregation family
- elementwise family

分别对应到不同的 tuning target。

### 8.4 跑第一版 importance-guided validation

哪怕是很小的规模，也要尽快开始：

- `No Priority`
- `Time-Only`
- `Importance-Guided`

三者对比。

---

## 9. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. 当前这项工作的最稳定位，是在已有 representative compression 之后补上 simulator-side organization and decision layer。
2. 前端以 PKA 为锚点，STEM+ROOT 提供异质性 refinement 启发。
3. 中间层通过 squash、family 与 regime，把 compressed objects 组织成结构化分析对象。
4. 后端通过 importance ratio 和 tuning priority，把这些对象进一步推进成 simulator validation workflow。
5. 当前最关键的不是再扩概念，而是逐步用真实数据替换 anchor / family / regime 表中的半定量字段，并开始 importance-guided validation。
