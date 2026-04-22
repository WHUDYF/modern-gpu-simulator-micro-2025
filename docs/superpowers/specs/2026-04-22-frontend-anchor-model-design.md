# Frontend Anchor Model Design

日期：2026-04-22

## 1. 目标

本设计文档定义一个面向完整 workload 的前端压缩层，用于把大量 kernel invocations
压缩成少量、可追踪、可加权、可进入后续结构化分析的 representative anchors。

当前要解决的问题不是：

- family 如何定义
- regime 如何划分
- simulator lane 如何组织
- tuning priority 如何最终决定

当前只解决：

**如何从完整 workload 中稳定地产生可信的 frontend anchors。**

因此，本文档的目标是把方法线固定为：

`full workload -> invocation-level anchor compression -> representative anchors + membership + weights + context metadata`

---

## 2. 方法定位

当前前端压缩层的定位是：

- 以前端 `PKA-style representative compression` 作为主锚点
- 吸收 `STEM+ROOT` 对 runtime heterogeneity 的提醒
- 但不在第一版完整复现 heterogeneity refinement

这意味着：

- 前端 anchor 层的主职责是 **compression**
- heterogeneity 在第一版中只作为 **guardrail / refinement interface**
- family / regime / importance weighting 仍然属于 compression 之后的方法层

一句话概括：

**前端负责选可信代表对象，后端负责组织解释结构。**

---

## 3. 设计边界

### 3.1 本设计做什么

- 定义前端压缩层的输入对象
- 定义 coarse anchor compression 的输出对象
- 定义 `hybrid-lite` 特征结构
- 定义 membership / weight / context metadata 的保留方式
- 定义 heterogeneity refinement 的预留接口
- 定义前端层应如何与后续 family 层解耦

### 3.2 本设计明确不做什么

- 不直接定义 family merge / split 规则
- 不直接定义 representative execution regime
- 不直接定义 simulator tuning parameter mapping
- 不完整复现 PKA / STEM+ROOT 的全部论文细节
- 不用前端输出直接证明某个机制解释为真

### 3.3 第一版完成标准

第一版完成后，至少应满足：

- 能从 workload invocation 表中生成 representative anchor 表
- 每个 anchor 都保留 membership 和 weight 信息
- 输出对象可被后续 phase / family 层继续消费
- 设计上已为 cluster 内异质性 refinement 留接口

---

## 4. 核心原则

### 原则 1：压缩对象采用 invocation-level，而不是 kernel-name-level

当前压缩基本对象定义为：

**kernel invocation**

原因如下：

- `kernel name` 级别过粗，容易把 runtime heterogeneous invocations 提前混合
- invocation-level 压缩更接近真实 workload 行为分布
- 后续如需 family / regime / phase 组织，仍可在 invocation-level anchor 之上继续收缩

### 原则 2：主特征采用 `hybrid-lite`

第一版前端特征不采用纯 profile，也不采用重度 trace 结构特征，而采用：

- 以 profile / workload summary 为主
- 以少量 context / trace 字段为辅

这样做有三个目的：

- 保持 `PKA-style` 前端锚点的清晰性
- 避免前端模型过早长成 family 判据
- 为后续 family / regime 提供最小必要上下文

### 原则 3：主权重采用时间权重，辅助保留 count 和 inst

前端层的 anchor 重要性排序优先依据：

- `time_weight`

同时保留：

- `count_weight`
- `inst_weight`

原因是：

- `time_weight` 最接近 workload 整体代价与模拟优先级
- `count_weight` 反映重复性
- `inst_weight` 反映工作量规模

前端层不对这三者做 family-level 解释，但必须完整保留。

### 原则 4：前端输出的是 anchor，不是最终解释对象

前端压缩后的 representative object 应被定义为：

**Representative Kernel Anchor**

它是：

- 压缩后的代表锚点
- 进入后续 phase / family / regime 分析的输入对象

它不是：

- 最终 family
- 最终 regime
- 最终 simulator lane

### 原则 5：异质性意识进入架构，不进入第一版主体复杂度

第一版不做重型 heterogeneity refinement，但必须承认：

- 某些 coarse cluster 可能内部 runtime distribution 不稳定
- 某些同名 invocations 可能并不应共享同一 anchor

因此，第一版采用：

**two-stage architecture, stage-1 first**

即：

- 架构上保留 Stage 2 refinement
- 实现上先完成 Stage 1 coarse compression

---

## 5. 总体方法线

当前推荐的前端方法线如下：

`full workload -> raw invocation table -> hybrid-lite feature construction -> coarse anchor compression -> representative anchor table`

其后保留一个可选接口：

`heterogeneous cluster -> optional refinement -> refined anchor subset`

因此，完整但克制的前端流程是：

1. 构建原始 invocation 表
2. 提取 `hybrid-lite` 特征
3. 进行 coarse clustering / representative selection
4. 输出 anchor、membership、weights、metadata
5. 对高风险异质 cluster 仅做标记或保留 refinement 接口

---

## 6. 输入对象定义

### 6.1 原始输入对象

前端压缩层的基本输入对象定义为：

**KernelInvocationRecord**

推荐字段如下：

| 字段 | 含义 |
|---|---|
| `kernel_invocation_id` | invocation 唯一标识 |
| `kernel_name` | kernel 名称 |
| `trace_order` | 在 workload trace 中的顺序 |
| `grid_dim` | grid 配置 |
| `block_dim` | block 配置 |
| `exec_time` / `cycle_proxy` | 执行时间或时间代理 |
| `dynamic_inst_count` | 动态指令数 |
| `memory_stats` | memory 行为统计 |
| `shape_hint` | M/N/K、sequence length、batch size 等可选形状提示 |
| `phase_hint_optional` | 可选 phase hint，不要求第一版完整提供 |

### 6.2 输入对象的语义约束

- 每条记录对应一个真实 invocation，而不是一个 kernel name 聚合体
- `exec_time` 与 `dynamic_inst_count` 必须可比较
- `trace_order` 必须保留，以支持后续 phase context 补回
- `shape_hint` 若存在，应原样透传，不在前端层解释其机制含义

---

## 7. 特征设计

### 7.1 设计选择

当前选择：

**hybrid-lite feature vector**

### 7.2 主特征

主特征用于 coarse compression，本质上是 `PKA-style behavior feature space` 的轻量扩展。

建议主特征包括：

- `exec_time` / `cycle_proxy`
- `dynamic_inst_count`
- DRAM / global memory 行为统计
- local / shared memory 行为统计
- `grid_dim`
- `block_dim`

### 7.3 辅助上下文字段

这些字段不作为第一版强判据，但必须保留：

- `trace_order`
- `shape_hint`
- `phase_hint_optional`

### 7.4 为什么不用更重的 trace-structure 特征

第一版不以：

- BBV
- warp path structure
- phase boundary sequence
- route primitive

作为前端主特征，原因是：

- 会让前端与后续 family 结构层边界混乱
- 会显著增加特征工程复杂度
- 会使论文主贡献看起来前移到 compression 本身

因此，第一版前端特征只承担：

**稳定压缩 representative objects**

而不承担：

**解释 shared mechanism**

---

## 8. 权重设计

### 8.1 主权重

前端层的主权重定义为：

**`time_weight = normalized_time_share`**

它表示某个 anchor 或 cluster 对整体 workload 时间占比的贡献。

### 8.2 辅助权重

同时保留：

- `count_weight = normalized_invocation_share`
- `inst_weight = normalized_dynamic_instruction_share`

### 8.3 设计理由

- `time_weight` 最适合作为 compression 后的重要性排序信号
- `count_weight` 为后续识别重复模式提供依据
- `inst_weight` 为后续识别工作量规模差异提供依据

前端层只负责存储和透传这些权重，不在此层决定 importance ratio 的最终组合方式。

---

## 9. Stage 1：Coarse Anchor Compression

### 9.1 目标

Stage 1 的目标是：

**在 invocation-level feature space 中生成一组 coarse representative anchors。**

### 9.2 流程

推荐流程如下：

1. 构建原始 invocation 表
2. 对主特征做标准化
3. 在 feature space 中执行 coarse clustering
4. 为每个 cluster 选 representative invocation
5. 计算 membership 与权重
6. 输出 representative anchor 表

### 9.3 representative invocation 选择规则

每个 cluster 的 anchor 优先选择：

- 距离 cluster 中心较近
- `time_weight` 较高
- metadata 完整

这样可以避免：

- 选到数学上居中但 workload 上不重要的样本
- 选到缺少上下文信息、后续难以接 phase / family 的样本

### 9.4 输出约束

Stage 1 输出必须满足：

- 每个 invocation 只能归属到一个 coarse cluster
- 每个 cluster 必须有一个 anchor
- 每个 anchor 必须显式记录 membership
- 每个 anchor 必须显式记录 `time_weight / count_weight / inst_weight`

---

## 10. 输出对象定义

### 10.1 输出对象

Stage 1 的标准输出对象定义为：

**RepresentativeKernelAnchor**

### 10.2 推荐字段

| 字段 | 含义 |
|---|---|
| `rep_kernel_id` | anchor 标识 |
| `anchor_invocation_id` | 作为 anchor 的 invocation |
| `kernel_name` | anchor 对应 kernel 名称 |
| `cluster_id` | coarse cluster 标识 |
| `covered_invocations` | 被该 anchor 覆盖的 invocation 集 |
| `coverage_count` | 覆盖 invocation 数 |
| `time_weight` | 时间权重 |
| `count_weight` | 次数权重 |
| `inst_weight` | 指令权重 |
| `trace_order_center` | cluster 在 trace 上的大致中心位置 |
| `grid_dim` | anchor 对应 grid 配置 |
| `block_dim` | anchor 对应 block 配置 |
| `shape_hint` | 透传 shape metadata |
| `heterogeneity_flag` | 是否存在异质性风险 |

### 10.3 语义说明

`RepresentativeKernelAnchor` 表示：

- 这是一个压缩后的代表对象
- 它代表一组 workload invocations
- 它携带后续结构化分析所需的最小 metadata

它不表示：

- 这些 invocations 已经共享同一 mechanism family
- 这些 invocations 已经属于同一 execution regime

---

## 11. Stage 2：Heterogeneity Refinement Interface

### 11.1 定位

Stage 2 在第一版中不是主体实现，而是：

**optional heterogeneity refinement interface**

### 11.2 触发条件

下列情况可触发 refinement：

- cluster 内 `exec_time` 方差异常大
- runtime distribution 呈现明显多峰
- `inst/time` 比例不稳定
- 同 cluster 内 `shape_hint` 差异过大
- 同名 invocations 呈现明显异质行为

### 11.3 第一版处理方式

第一版推荐处理方式是：

- 不自动细化全部 cluster
- 仅对满足触发条件的 cluster 设置 `heterogeneity_flag`
- 将 refinement 保留给后续扩展阶段

### 11.4 为什么保留而不完整实现

原因不是 heterogeneity 不重要，而是：

- 第一版需要先建立稳定前端锚点
- 过早引入重型 refinement 会抬高前端复杂度
- 会削弱后续 family / regime 层的主创新空间

因此，第一版对 heterogeneity 的立场是：

**承认、标记、留接口，但不全面展开。**

---

## 12. 与后续 family 层的接口

### 12.1 family 层不直接吃裸 anchor

后续 family 层不应直接消费 `RepresentativeKernelAnchor`，而应消费：

**Phase-Annotated Representative Anchor**

即在 anchor 之上补回：

- `phase_id`
- `phase_local_order`
- `optional route hint`
- `optional shape context`

### 12.2 这样分层的原因

这样做可以保证：

- 前端 compression 不提前长成 family 判据
- phase / family / regime 仍然保持为 compression 之后的方法层
- 后续结构组织有稳定输入对象，但不受前端假设绑死

---

## 13. 验证要求

前端层第一版只验证三类性质。

### 13.1 压缩率

验证：

- 从全部 invocations 压缩到多少 representative anchors

### 13.2 覆盖性

验证：

- 少量 anchors 是否覆盖了 workload 大部分 `time_weight`

### 13.3 稳定性

验证：

- 在小范围参数扰动或随机初始化变化下，anchor 结果是否基本稳定

### 13.4 明确不在前端层验证的内容

前端层不验证：

- family 是否正确
- regime 是否正确
- simulator tuning 是否正确

因为这些问题属于 compression 之后的方法层。

---

## 14. 风险与限制

### 14.1 coarse cluster 可能掩盖 invocation-level heterogeneity

这是第一版最主要的风险，因此必须保留 `heterogeneity_flag` 与 refinement 接口。

### 14.2 仅用 `hybrid-lite` 特征可能不足以分辨细粒度执行模板差异

这是有意接受的边界，因为执行模板差异应主要由后续 family 层处理，而不是由前端层完成。

### 14.3 shape / phase metadata 可能不完整

第一版允许这类 metadata 仅作为 optional 字段透传，但不能在前端层强行根据不完整 metadata 做机制归类。

---

## 15. 最终结论

当前最推荐的前端设计是：

- 对象粒度采用 `invocation-level`
- 特征采用 `hybrid-lite`
- 权重采用 `time` 为主、`count` 与 `inst` 为辅
- 架构采用 `two-stage`
- 实现先完成 `Stage 1 coarse anchor compression`
- `Stage 2 heterogeneity refinement` 只保留接口

因此，第一版 frontend anchor model 的最稳表述是：

**我们以前端 `PKA-style` representative invocation compression 作为输入锚点，在 invocation-level feature space 中生成带 membership、weight 与 context metadata 的 representative anchors，并为 runtime heterogeneity refinement 预留接口，而不在前端层直接进入 family / regime 解释。**
