# Trace Compression Behavior Signature for Microbench Generation Design

> **学术线定位：** 本 spec 只定义 trace compression 的研究方法线，不负责在线压缩系统实现，也不修改 L1 的 PKA selector 契约。

**Goal:** 将 trace compression 产生的结构与失败模式抽象成可解释的 behavior signature，并用它来驱动 AI agent 生成与目标 workload 行为相近的 microbench。

**Architecture:** 从 L1 选出的 representative target 出发，读取其 trace 并经过现有 compression substrate 生成分层压缩表示，再从压缩表示中抽取结构签名、退化签名和相似度分数。AI agent 以该签名为输入生成 microbench，microbench 再走同样的压缩与签名流程，最后用 signature distance 作为生成反馈。

**Tech Stack:** 现有 `simulator-remodeled/util/traces_enhanced/dynamic_trace` 压缩格式、Python 分析脚本、L1 RLCR 输出、Nsight trace 产物、AI agent 生成循环。

---

## 1. 这条线在回答什么问题

这条线要回答的不是“trace 能不能压得更小”，而是：

> trace 的哪些压缩结构，能稳定暴露 kernel 的执行规律、warp / TB 级差异和数据相关不规则性？

如果这个问题成立，trace compression 就不只是存储优化，而是一种行为抽象层。它能把原始 workload 的复杂执行过程压成一个足够小、足够稳定、足够可解释的 signature，供 agent 生成 microbench 使用。

这条线和 L1 的关系是：

- L1 负责给出 representative target
- 本 spec 负责把 target 的 trace 结构变成可比较的 behavior signature
- agent 负责利用 signature 生成 microbench

这条线和工程线的边界是：

- 本 spec 关心 signature、相似性和生成价值
- 工程线关心在线压缩、读写吞吐、传输和 replay 成本

---

## 2. 非目标

这份 spec 不做下面几件事：

- 不实现新的在线压缩系统
- 不把 trace compression 直接改造成 backend planning 输入
- 不替代 L1 的 PKA measured-only baseline
- 不要求 microbench 在语义上等价于目标 kernel
- 不把压缩比本身当作唯一研究结论

microbench 的目标是行为接近，不是语义等价。压缩结构的目标是暴露行为规律，不是单纯减文件大小。

---

## 3. 核心定义

### 3.1 Target

`target` 是由 L1 选出的 representative kernel / invocation。

它至少携带这些信息：

- `kernel_invocation_id`
- `source_trace_path`
- `launch_shape`
- `membership`
- `time_weight`
- `workload_scale`

这些字段来自 L1 的代表对象输出，不在本 spec 中重新定义。

### 3.2 Microbench

`microbench` 是人为构造的小程序或小 CUDA kernel。它不需要复现真实模型语义，只需要尽量复现 target 的硬件行为结构。

### 3.3 Behavior signature

`behavior signature` 是从压缩 trace 中抽取的一组结构化特征，描述 kernel 在以下方面的规律性：

- 指令序列是否可被长 run 压缩
- warp 之间是否共享相近 PC 序列
- threadblock 之间是否主要只差 base address
- 地址覆盖是否高度规则
- 压缩器是否频繁退化为 full encoding

### 3.4 Failure profile

`failure profile` 是压缩器解释不了的部分，或者压缩不得不退化的部分。它不是噪声，而是信号的一部分，因为它直接反映执行路径的不规则性。

---

## 4. 压缩签名的组成

本 spec 以当前 `compressed_kernel_v8` 作为主 substrate，必要时允许回退到 `v7` / `v6` 做对照。

### 4.1 结构性特征

这些特征描述压缩结构本身：

- `instruction_run_length_histogram`
- `instruction_run_coverage`
- `shared_pc_sequence_coverage`
- `warp_diff_density`
- `cross_tb_global_address_offset_coverage`
- `address_override_density`
- `full_encoding_fallback_rate`

### 4.2 诊断性特征

这些特征描述压缩器在哪些位置开始解释失败：

- `pc_override_count`
- `per_warp_diff_count`
- `per_tb_delta_override_count`
- `full_encoding_tb_ratio`
- `address_override_position_entropy`

### 4.3 规模控制特征

这些特征不是主签名，但用于防止把 shape 差异误判成行为差异：

- `launch_grid_size`
- `launch_block_size`
- `dynamic_instruction_count`
- `kernel_runtime`

规模特征不应单独决定相似性分数，只能作为归一化或分层比较的条件。

---

## 5. 相似性定义

这个方向不把“相似”定义成单一数值，而是定义成一组可解释的距离项。

### 5.1 距离结构

对 target 和 microbench 的 signature，计算以下距离：

- histogram 类特征使用 Jensen-Shannon divergence
- scalar rate 类特征使用归一化绝对差
- count 类特征使用对数化后的归一化绝对差

最终总分采用加权和：

```text
signature_distance =
  w1 * dist(run_length_histogram)
  + w2 * |instruction_run_coverage_t - instruction_run_coverage_m|
  + w3 * |shared_pc_sequence_coverage_t - shared_pc_sequence_coverage_m|
  + w4 * |warp_diff_density_t - warp_diff_density_m|
  + w5 * |cross_tb_global_address_offset_coverage_t - cross_tb_global_address_offset_coverage_m|
  + w6 * |address_override_density_t - address_override_density_m|
  + w7 * |full_encoding_fallback_rate_t - full_encoding_fallback_rate_m|
```

首轮权重采用归一化后的 equal-weight 配置，所有 component 都必须单独报告。后续只有在 ablation 证明某些 component 对区分力没有贡献时，才调整权重。初始化原则是：

- 先保证解释性
- 再强调跨 TB 规则性和退化模式
- 最后再调 runtime 和 launch shape 的校正项

### 5.2 评分用途

`signature_distance` 用于三件事：

1. 作为 agent 的 reward signal
2. 作为 microbench 候选排序分数
3. 作为对比不同生成策略的统一评价指标

它不是最终的模拟正确性证明，只是更便宜的行为对齐信号。

---

## 6. 工作流

```text
L1 representative target
  -> target trace
  -> compression substrate
  -> behavior signature
  -> similarity score
  -> agent generates microbench
  -> microbench trace
  -> same compression substrate
  -> microbench signature
  -> score and iteration
```

### 6.1 目标输入

L1 输出的 representative target 进入这条线时，不需要带入 A 线的全部细节，只要保留：

- 代表对象 id
- raw trace path
- membership
- time weight
- launch shape
- source provenance

### 6.2 特征抽取

特征抽取阶段应同时读取：

- 压缩后的结构信息
- 压缩过程中的退化信息
- 必要的规模校正信息

这一步的产物不是一串原始 protobuf，而是一个面向分析和生成的 signature record。

### 6.3 生成与回传

agent 读取 signature record 后生成 microbench。microbench 运行后产生新的 trace，再走同样的压缩与抽取流程。这样可以得到稳定的闭环。

---

## 7. 与 L1 的接口

这条线和 L1 的接口是单向清晰的。

### 7.1 L1 -> 学术线

L1 提供：

- representative target
- membership
- weight
- launch shape
- source trace path

### 7.2 学术线 -> L1

学术线回传：

- target 的 behavior signature
- microbench 候选的 signature
- target / microbench 的 distance report
- 哪些压缩结构最能区分该 target 的解释报告

### 7.3 不进入 L1 的内容

以下内容不应倒灌回 L1 selector：

- compression-side signature 作为主分组轴
- microbench 生成结果作为 selector 证据
- online compression 的工程指标

L1 继续只负责 representative compression，不负责 microbench 生成。

---

## 8. 评估方法

这条线的评估分三层。

### 8.1 可分性

检查不同 kernel family 是否能被 signature 清楚区分。

这里的 family label 只作为评估标签，用来判断 signature 是否有行为区分力；它不进入 L1 selector，也不替代 B 线的 family / regime 对象。

对照基线至少包括：

- kernel name
- launch shape
- runtime only
- PKA measured feature summary

报告形式至少包括：

- silhouette score
- nearest-neighbor retrieval table
- behavior family confusion table

如果 compression signature 不能比这些基线更好地区分行为类型，这条线就没有方法价值。

### 8.2 稳定性

同一个 target 在多个运行、多个采样窗口下，其 signature 应保持稳定。

如果 signature 只是在某次运行里偶然成立，那它不能作为 agent 的可靠输入。

### 8.3 生成价值

比较以下两类 microbench：

- 只按 runtime / shape / name 生成的 baseline microbench
- 按 compression signature 生成的 microbench

如果后者在 signature_distance 上持续更低，并且更接近 target 的结构行为，那么这条线成立。

---

## 9. 风险与控制

### 9.1 过度依赖 launch shape

launch shape 很容易把“形状相近”误判成“行为相近”。因此 shape 只能用于分层比较和归一化，不能直接决定结论。

### 9.2 不规则 kernel 的信号稀释

对高度不规则的 sparse 或 control-heavy kernel，压缩率和 signature 的区分度可能下降。此时应把 failure profile 当作主信号，而不是只看压缩率。

### 9.3 压缩版本漂移

如果不同 trace 版本的压缩 substrate 不一致，signature 也会漂移。评估时必须固定版本，或明确标注版本转换。

### 9.4 误把工程指标当研究结论

在线读取速度、传输速度和文件大小下降是工程收益，不是本 spec 的主结论。它们可以作为附加观察，但不能替代 behavior signature 的方法论结果。

---

## 10. 成功标准

这条线成立的最低标准是：

1. 能用压缩结构和失败模式，稳定区分 `GEMM-like`、`reduction`、`elementwise`、`irregular-memory`、`control-heavy` 中至少三类典型 kernel 行为
2. 能让 agent 生成的 microbench 在 signature 层面比 baseline 更接近 target
3. 能解释“为什么这个 target 难压缩 / 为什么这个 microbench 更像目标”
4. 能和 L1 接上，但不侵入 L1 的 selector 契约

如果只得到“压缩后更小”或“看起来更快”，但没有行为解释力，这条线不算成立。

---

## 11. 现有文档关系

这份 spec 继承并收敛了以下材料：

- [trace-compression-for-microbench-agent.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/trace-compression-for-microbench-agent.md)
- [2026-04-03-trace-compression-design.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/specs/2026-04-03-trace-compression-design.md)
- [2026-04-27-a-line-l1-rlcr.spec](/home/dyf/modern-gpu-simulator-micro-2025/docs/a-line-l1-rlcr-spec-2026-04-27.md)

它们的关系是：

- 早期 trace compression design 提供压缩 substrate
- 旧 microbench note 提供研究直觉
- L1 spec 提供代表对象入口
- 本 spec 把这些东西收束成学术线方法
