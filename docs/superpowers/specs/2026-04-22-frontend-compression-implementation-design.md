# Frontend Compression Implementation Design

日期：2026-04-22

## 1. 目标

本设计文档定义前端压缩线的第一版实现方案。

当前要实现的不是完整的 sampled simulation 前端论文，也不是 family / regime / tuning 层，
而是：

**一个可运行、可解释、可替换的 frontend anchor v1。**

它必须满足两类目标：

1. 交付 A 线当前必须给出的结构化产物
2. 为后续 family 线和 importance / tuning 线提供稳定输入

因此，v1 的实现目标固定为：

- 生成 `Representative Anchor Table`
- 生成 `Frontend Compression Note`
- 生成最小比较证据，说明为什么 `hybrid` 比 `name-only` 和 `PKA-like coarse` 更有价值

---

## 2. 方法定位

本实现严格遵循当前已经确认的方法定位：

- 前端是 `Constrained PKA Extension`
- 前端只允许最小必要扩展
- 前端的角色是 reviewer-safe 的输入锚点
- 前端不提前输出 family / regime 结论

当前前端压缩线服务的主方法链为：

`frontend compression -> representative anchors -> family -> representative execution regime -> importance ratio -> tuning priority -> simulator validation`

因此，前端实现必须同时满足：

- 对前：保持 `PKA-style representative compression` 的角色清晰
- 对后：输出可被 B 线直接消费的 anchor 对象

---

## 3. v1 工作边界

### 3.1 v1 要做什么

- 建立标准输入表 `KernelInvocationRecord`
- 基于该输入表生成三套前端方案：
  - `name-only baseline`
  - `PKA-like coarse baseline`
  - `hybrid`
- 统一导出 `Representative Anchor Table`
- 生成比较表和 case note
- 明确标注所有 provisional / placeholder 字段

### 3.2 v1 不做什么

- 不完整复现 PKA
- 不完整复现 STEM+ROOT
- 不输出 family membership
- 不输出 regime label
- 不输出 route primitive 或 execution template
- 不直接进入 simulator lane 映射

### 3.3 v1 完成标志

v1 完成时，至少应有：

1. 一份稳定 schema 的 `Representative Anchor Table`
2. 一个最小可跑入口，能从 dual-source 生成该表
3. 一份 `Comparison Table`
4. 一份短的 `Case Note`
5. 一份 `Frontend Compression Note`

---

## 4. 输入来源与对象定义

### 4.1 双数据源输入

v1 采用 `dual-source` 输入：

- Source A：`enhanced_execution_info.json`
  - 提供 identity / context / static instruction metadata
- Source B：merged feature JSON
  - 提供 trace / hardware / compression feature summary

这种双源设计的角色分工是：

- Source A 负责 identity 和最小上下文
- Source B 负责 feature 和 weight 相关信息

### 4.2 标准输入对象

双源输入在前端第一步必须统一为：

**`KernelInvocationRecord`**

推荐最小字段集：

- `kernel_invocation_id`
- `kernel_name`
- `trace_order`
- `grid_dim`
- `block_dim`
- `shape_hint`
- `exec_time`
- `dynamic_inst_count`
- `feature_vector`
- `feature_source_note`

这里的 `feature_vector` 不等同于 family 判据，
它只用于 frontend compression。

### 4.3 统一对象的理由

v1 先做输入表，而不是先做 selector，有三个原因：

1. 保证三套前端方案共用同一输入对象
2. 减少临时字段污染主线 schema
3. 让后续 B 线接入时只关心表结构，而不关心比较逻辑

---

## 5. 总体实现链路

v1 的最小链路固定为：

`dual-source raw files`
-> `KernelInvocationRecord table`
-> `3-way anchor selection`
-> `Representative Anchor Table`
-> `Comparison Table + Case Note`

其中必须明确区分两类输出：

### 5.1 主线输出

主线输出只有：

- `Representative Anchor Table`
- `Frontend Compression Note`

这两者是后续 B 线真正应消费的对象。

### 5.2 证据输出

证据输出包括：

- `Comparison Table`
- `Case Note`

这两者只用于证明前端设计合理，
不应被下游直接当作主线输入对象。

---

## 6. 模块划分

v1 推荐按 4 个最小模块实现。

### 6.1 `invocation_table_builder`

职责：

- 读取 dual-source 输入
- 统一成 `KernelInvocationRecord`
- 对字段做最小标准化
- 输出标准输入表

它不负责：

- 聚类
- 代表选择
- family 解释

### 6.2 `anchor_selector`

职责：

- 在统一输入表上运行三套前端方案
- 输出中间 anchor grouping 结果

它必须支持：

- `name-only baseline`
- `PKA-like coarse baseline`
- `hybrid`

### 6.3 `anchor_exporter`

职责：

- 将 selector 中间结果统一导出为 `Representative Anchor Table`
- 补齐 A 线要求的关键字段
- 标注权重字段与 membership 字段的状态

### 6.4 `comparison_report_builder`

职责：

- 生成 `Comparison Table`
- 生成 `Case Note`

它不应反向影响主线对象定义。

### 6.5 文件形态建议

v1 推荐采用：

- 一个主入口脚本
- 若干可复用 Python 模块

即：

**hybrid file structure**

这样既能快速跑通，也不会把逻辑全塞进一个脚本里。

---

## 7. 三套前端方案的最小定义

### 7.1 `name-only baseline`

最小定义：

- 按 `kernel_name` 分组
- 每组生成一个 anchor

它的作用是：

**证明只按名字分组过于粗糙。**

### 7.2 `PKA-like coarse baseline`

最小定义：

- 不做 clustering
- 只使用显式 metadata / coarse execution signature 分桶
- 每桶生成一个 anchor

推荐的 coarse bucket 形式：

- `kernel_name + shape/grid/block`
或
- 等价的 coarse signature

它的作用是：

**提供一个不依赖 clustering 的合理前端基线。**

### 7.3 `hybrid`

最小定义：

- 先做和 coarse baseline 相同的显式分桶
- 再在每个桶内做 lightweight clustering
- 每个 cluster 生成一个 anchor

它的作用是：

**证明在保持 frontend anchor 边界的前提下，引入轻量 clustering 具有额外价值。**

### 7.4 三者的关系

三者在论证链中的角色分别是：

- `name-only`
  - 证明不能只按名字分
- `PKA-like coarse`
  - 证明只做显式粗分桶仍然不够
- `hybrid`
  - 证明 clustering 是一个有价值的 constrained frontend optimization

---

## 8. 主线输出对象

v1 的主线输出对象定义为：

**`Representative Anchor Table`**

这是 A 线必须交付的核心产物。

### 8.1 最小必备字段

这些字段必须出现：

- `rep_kernel_id`
- `kernel_name`
- `cluster_id`
- `member_invocations`
- `coverage_count`
- `coverage_weight`
- `time_weight`

### 8.2 强烈建议字段

这些字段建议在 v1 就保留：

- `trace_order_summary`
- `shape_hint_summary`
- `grid_dim`
- `block_dim`

### 8.3 状态说明字段

由于 v1 允许 provisional 结果，
必须增加状态说明字段，例如：

- `coverage_weight_source`
- `time_weight_source`
- `member_invocations_status`
- `heterogeneity_flag`
- `notes`

### 8.4 主线对象语义

`Representative Anchor Table` 表示：

- 前端压缩后的代表锚点集合
- 可被后续 B 线继续消费

它不表示：

- final family
- final regime
- final simulator lane

---

## 9. provisional 策略

v1 明确采用：

**允许 provisional，但必须透明。**

这意味着：

- 权重字段可以先是 provisional
- membership 可以先是摘要形式
- 但必须在输出 note 中明确写出：
  - 哪些字段是 measured
  - 哪些字段是 derived
  - 哪些字段是 provisional
  - 哪些字段仍是 placeholder

当前推荐的策略是：

- 先拿到初版可运行结果
- 再逐步把关键字段替换成真实测量值

---

## 10. comparison 证据设计

v1 不只交主线 anchor 表，
还要交最小比较框架。

### 10.1 比较表的角色

`Comparison Table` 的作用是：

- 证明三种前端方案的差异存在
- 为后续 paper 论证保留证据

### 10.2 case note 的角色

`Case Note` 的作用是：

- 解释为什么 `hybrid` 拆分更合理
- 服务于 interpretability，而不是主线接口

### 10.3 最小比较指标

第一版 `Comparison Table` 推荐至少包含：

- `method`
- `num_anchors`
- `time_weight_covered`
- `avg_cluster_size`
- `intra_cluster_exec_time_var`
- `intra_cluster_inst_var`
- `split_cases_count`
- `notes`

### 10.4 指标解释

- `num_anchors`
  - 衡量压缩复杂度
- `time_weight_covered`
  - 衡量对 workload 重要部分的覆盖
- `avg_cluster_size`
  - 衡量分组是否过粗或过碎
- `intra_cluster_exec_time_var`
  - 衡量 cluster 内稳定性
- `intra_cluster_inst_var`
  - 衡量 cluster 内工作量稳定性
- `split_cases_count`
  - 衡量 `hybrid` 相比 baseline 的额外辨别能力

### 10.5 比较目标的主次顺序

当前比较目标优先级为：

1. `interpretability`
2. `stability`
3. `coverage`

因此：

- 比较表先证明“差异存在”
- case note 再解释“为什么这种差异有意义”

---

## 11. squash 相关特征在前端中的角色

### 11.1 为什么前端需要关心 squash

当前仓库中的 `squash` 机制负责：

- 对 workload 的 kernel 序列做 temporal segmentation
- 对 kernel 内部 TB 序列做 segmentation
- 输出 `squash_segments`、`boundary_count`、`cohesion_score`、`representative_kernel / representative_tb`

这些信息本质上是在描述：

- 时间结构
- 段内稳定性
- 显著 boundary

因此，`squash` 对前端 anchor 线的重要性在于：

- 它可以帮助识别某个 coarse bucket 或 cluster 是否内部不稳定
- 它可以提供时间结构相关的辅助上下文

### 11.2 为什么 squash 不能变成前端主轴

尽管 `squash` 很重要，v1 仍然不应把它直接提升为前端压缩的主特征轴。

原因是：

- 当前主线已把 `squash + batch` 放在中间结构层
- 前端仍然需要保持 `Constrained PKA Extension` 的角色
- 如果把 `squash` 主导前端分组，前端很容易越界到 phase / regime 层

因此，v1 中 `squash` 的合理定位是：

- `context-for-family`
或
- `heterogeneity-guardrail`

而不是：

- frontend compression 的唯一主判据

### 11.3 v1 中 squash 的使用原则

v1 对 squash 的使用采用下面三个原则：

1. `squash` 可以提供前端辅助信息，但不能直接替代 frontend anchor 主特征
2. `squash` 可以参与 cluster 稳定性判断，但不直接输出 family 结论
3. `squash` 产出的信息更适合作为 summary / guardrail，而不是完整机制标签

---

## 12. v1 squash 输入特征设计

### 12.1 总体原则

v1 的 squash 特征采用：

**behavior skeleton + resource behavior + light scale/context**

也就是：

- 行为骨架特征
- 资源行为特征
- 轻量规模 / 上下文特征

这三类特征共同作用于：

- temporal segmentation
- cluster stability observation
- heterogeneity guarding

### 12.2 v1 推荐的 squash 最小特征集合

v1 不追求一次加入大量 squash 特征，
而采用一组约 8 个左右的最小集合。

当前推荐采用：

- `ffma_ratio`
- `fp64_op_ratio`
- `global_load_ratio`
- `shared_mem_op_ratio`
- `compute_utilization`
- `dram_throughput_pct`
- `occupancy_pct`
- `warp_divergence_rate`

这一组属于：

**resource-balanced squash feature set**

它的目标是同时保留：

- 行为骨架识别能力
- compute vs memory 区分能力
- 执行稳定性区分能力

### 12.3 两层优先级

为了保证 v1 的可落地性，这 8 个特征进一步分为两层。

#### 第一层：必须尽量真实的 5 个

- `ffma_ratio`
- `global_load_ratio`
- `compute_utilization`
- `dram_throughput_pct`
- `occupancy_pct`

这组特征构成 v1 squash 的硬骨架。

如果它们失真，segmentation 本身就容易失真。

#### 第二层：允许先近似或缺省的 3 个

- `fp64_op_ratio`
- `shared_mem_op_ratio`
- `warp_divergence_rate`

这组特征更像：

- boundary difference amplifier
- phase distinction enhancer

它们很重要，但不应阻塞 v1 的整体推进。

### 12.4 为什么不用更重的 squash 特征

v1 暂不引入：

- route primitive
- hardware template
- family_id
- regime label
- importance-related score

这些信息都属于 compression 之后的方法层，
进入 squash 特征会导致前端越界。

### 12.5 对当前实现的直接要求

由于当前 `extract_squash_features.py` 已经在使用 opcode ratio 与部分 NCU 指标，
v1 最直接的实现要求不是重写 squash，而是：

1. 检查并对齐 squash 期望字段与 NCU / merged features 的命名
2. 确认 resource-balanced 特征是否真正进入 squash 向量
3. 把 squash 输出转成前端可用的 summary / guardrail 字段

---

## 13. squash 输出在前端中的接入方式

### 13.1 推荐接入位置

v1 推荐把 squash 相关结果接入到两个位置：

#### A. `KernelInvocationRecord` 的辅助上下文字段

例如：

- `kernel_squash_segment_id`
- `kernel_squash_boundary_count`
- `kernel_squash_cohesion`

#### B. `Representative Anchor Table` 的 guardrail 字段

例如：

- `heterogeneity_flag`
- `squash_boundary_crossing_flag`
- `squash_context_note`

### 13.2 不推荐的接入方式

v1 不推荐：

- 直接把 squash 输出作为 anchor_selector 的唯一主输入
- 直接按 squash segment 替代 frontend clustering
- 直接把 squash segment 当作 final family / regime

### 13.3 最稳的前端定位

因此，当前最稳的说法是：

**squash 在 v1 中主要用于提供时间结构上下文和 cluster stability guardrail，而不是直接决定前端 anchor 的最终分组。**

---

## 14. v1 的完成标准

v1 不追求完整，
只追求最小闭环。

当前推荐的最小完成标准是：

1. 能从 dual-source 构建 `KernelInvocationRecord`
2. 能运行三种前端方案
3. 能导出 `Representative Anchor Table`
4. 能导出 `Comparison Table`
5. 能写出一份短的 `Frontend Compression Note`
6. 能说明哪些字段是 provisional

---

## 15. 风险与限制

### 12.1 输入字段仍可能不完整

`enhanced_execution_info.json` 与 merged features 之间的键对齐关系，
可能限制某些字段在 v1 的真实性。

### 12.2 PKA-like coarse baseline 不是完整 PKA

因此它只能作为：

- 非聚类合理基线

不能被表述成：

- 完整复现 PKA

### 12.3 hybrid 的 clustering 优势在 v1 只能做初步证明

第一版更适合回答：

- 差异是否存在
- 某些 split case 是否更合理

而不是：

- clustering 已被完全证明最优

### 12.4 主线输出与证据输出必须严格分离

如果把比较逻辑混进主线输出，
后续 B 线会难以稳定接入。

---

## 16. 最终结论

前端压缩线 v1 的最稳实现方案是：

- 采用 `dual-source`
- 先做 `KernelInvocationRecord` 标准输入表
- 采用 `Table-First`
- 同时运行 `name-only`、`PKA-like coarse`、`hybrid`
- 只把 `Representative Anchor Table` 作为主线输出
- 把 `Comparison Table + Case Note` 作为证据输出
- 明确允许 provisional，但要求状态透明

一句话总结是：

**v1 应该交付一个可运行的 frontend anchor 初版结果，同时附带最小比较证据，用来证明 lightweight clustering 作为 constrained frontend optimization 的价值，而不越界到后续 family / regime 层。**
