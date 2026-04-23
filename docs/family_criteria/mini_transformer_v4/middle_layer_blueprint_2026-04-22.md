# mini_transformer_v4 中段结构层 Blueprint

日期：2026-04-22

## 1. 文档目的

这份文档用于把 `mini_transformer_v4` 上的中段结构层收敛成一套可执行 blueprint。

当前最关键的判断是：

**Anchor Table 只是输入层，不是 middle layer 本身。**

真正的中段结构层应该由下面四层组成：

`frontend anchors -> family objects -> representative regimes -> simulator lanes`

这四层分别回答：

1. 前端到底给了我们什么输入对象
2. 哪些输入对象共享同一工作模式
3. 哪些对象是 backend 真正需要单独处理的执行区间
4. 这些执行区间进入哪条 simulator validation / tuning 路线

---

## 2. 当前最小闭环

对 `mini_transformer_v4`，当前最小闭环不应再写成：

`representative kernels -> family`

而应写成：

`phase-aware anchors -> family -> representative regimes -> simulator lanes`

这意味着 middle layer 至少要落成四张正式表：

1. `Representative Anchor Table`
2. `Family Table`
3. `Representative Regime Table`
4. `Simulator Lane Mapping Table`

---

## 3. 当前中段结构层的核心原则

### 3.1 Anchor 不等于 Family

Anchor 只是前端压缩后的输入锚点。

它应该保留：

- membership 接口
- coverage / time 接口
- phase / shape / route / template 提示

但它本身不是最终解释对象。

### 3.2 Family 不按 kernel 名字组织

Family 的组织依据应该是：

- `phase-aware execution context`
- `route primitive`
- `hardware execution template`
- `boundary-first` 判据

也就是说，Family 的本质是：

**共享 data path / execution template 的结构对象。**

### 3.3 Regime 才是 backend 入口对象

即使进入同一 Family，后续仍然可能因为：

- shape 区间不同
- trace 上下文不同
- 资源签名不同

而继续拆成多个 regime。

因此，真正进入 simulator lane 的单位不应是：

- 单个 kernel 名
- 单个 family 名

而应是：

**representative execution regime**

### 3.4 Lane 不是备注栏

如果 regime 后面没有 lane mapping，那么 middle layer 仍然只是分类系统，而不是 decision layer。

每个高优先级 regime 都应明确：

- 去哪条 lane
- 主要验证什么
- 主要扰动什么参数方向
- 结果如何回写

---

## 4. 当前推荐对象层级

### 4.1 输入层：Phase-Aware Anchors

当前第一版 anchor 不应继续停留在“6 个 kernel 名”的层级。

更合理的对象层级应至少提升到：

`kernel + phase/context + shape`

在 `mini_transformer_v4` 上，推荐的第一版 operational anchors 为：

1. `A1_qkv_projection_dense_48x32`
2. `A2_attention_score_dense_32x32x12`
3. `A3_softmax_reduce_24x1`
4. `A4_context_stream_4x32x12`
5. `A5_output_projection_dense_48x32`
6. `A6_residual_elementwise_1536`
7. `A7_layernorm_reduce_512`
8. `A8_ffn_expand_dense_192x32`
9. `A9_ffn_contract_dense_48x32`

这个集合仍然是 placeholder version，但比“kernel-name-only anchors”更适合作为 middle layer 输入。

### 4.2 组织层：Family Objects

当前推荐的第一版 Family 为：

1. `F1_dense_tiled_backbone`
2. `F2_reduction_normalize`
3. `F3_streaming_aggregation`
4. `F4_elementwise_residual`

其中：

- `F1` 不是单一 regime，而是 dense backbone 的共享机制层
- `F2` 不是单一上下文，而是 reduction/normalize 的共享机制层

### 4.3 决策层：Representative Regimes

当前推荐把 backend 入口拆成 9 个 regime：

1. `R1_qkv_projection_dense`
2. `R2_attention_score_dense`
3. `R3_output_projection_dense`
4. `R4_ffn_expand_dense`
5. `R5_ffn_contract_dense`
6. `R6_softmax_reduction`
7. `R7_layernorm_reduction`
8. `R8_context_streaming`
9. `R9_residual_elementwise`

### 4.4 执行层：Simulator Lanes

每个 regime 后面应对应至少一条 lane。

Lane 的目标不是一开始覆盖完整 DSE，而是先把：

- parameter direction
- baseline type
- validation metric
- writeback target

这四个接口固定下来。

---

## 5. 当前推荐的实现顺序

### Step 1：先把 Anchor Table 升级成 phase-aware / context-aware / shape-aware

这是 middle layer 的输入修正。

### Step 2：再做 Anchor -> Family

这一步固定共享机制层。

### Step 3：再做 Family -> Regime

这一步生成真正的 backend 入口对象。

### Step 4：最后做 Regime -> Lane

这一步把 middle layer 接到 backend validation。

---

## 6. 当前最重要的工作重点

接下来最应该优先做的，不是继续扩展 analysis cards，而是把以下四张表做硬：

1. `Representative Anchor Table`
2. `Family Table`
3. `Representative Regime Table`
4. `Simulator Lane Mapping Table`

只要这四张表成立，`mini_transformer_v4` 的中段结构层就不再只是方法说明，而开始变成：

**compression 之后的 simulator decision layer**

---

## 7. 当前阶段的简短结论

如果压成最短形式，可以写成：

1. Anchor Table 负责提供 middle layer 的输入对象。
2. Family Table 负责组织共享 data path / execution template。
3. Regime Table 负责给 backend 提供真正的 simulator 入口对象。
4. Lane Mapping Table 负责把 regime 接入验证与调参路径。
5. 因此 middle layer 的真正任务不是“补一个 anchor 表”，而是把 anchors 提升成 `family / regime / lane` 三层正式对象。
