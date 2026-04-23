# mini_transformer_v4 Family Canon（第一版）

日期：2026-04-22

## 1. 文档目的

这份文档用于解决当前 `mini_transformer_v4` family 体系中的一个关键问题：

**不同文档对同一 kernel 的 family 归属存在角色差异与阶段差异，导致后端输出层缺少唯一可引用的 canon。**

这里的目标不是重新讨论所有边界 case，而是明确：

1. 对后端输出层来说，当前哪一版 family / regime 映射算唯一 canon
2. 旧的 family cards / outlier cards / synthesis 文档在当前应如何解释
3. 哪些对象已经可以稳定进入后端输出层
4. 哪些对象仍然保留 uncertainty，但不再阻塞后端表输出

---

## 2. 为什么现在必须定义 canon

当前已经存在三类不同角色的文档：

### 2.1 方法生长文档

例如：

- boundary cases
- analysis cards
- family cards
- outlier cards
- synthesis

这些文档的职责主要是：

- 暴露边界
- 记录不确定性
- 说明 family 判据是如何长出来的

### 2.2 结构落表文档

例如：

- `Representative Anchor Table`
- `Family / Regime Table`

这些文档的职责主要是：

- 把当前对象真正落成可引用表
- 让后续 importance / lane / validation 能继续展开

### 2.3 后端输出层文档

例如：

- `backend_output_schema_v1`

这些文档的职责主要是：

- 定义后端究竟输出什么对象
- 定义对象之间如何引用
- 定义结果如何回写

问题就在于：

**方法生长文档允许保留 ambiguity，但后端输出层不能在 family 归属上继续悬空。**

因此必须专门定义：

**for backend v1, what is canonical.**

---

## 3. 当前冲突的本质

当前最明显的冲突主要集中在两个对象上。

### 3.1 `layernorm_kernel`

一组文档把它当成：

- 第二轮 mixed/outlier 检验样本
- 当前应保留明显 uncertainty

另一组文档则已经把它吸收到：

- `F2_reduction_normalize`
- `R4_layernorm_reduction`

### 3.2 `residual_add`

一组文档把它写成：

- `memory-heavy -> dram-dominated`

另一组文档则已经把它写成：

- `F4_elementwise_fusion`
- `R6_residual_elementwise`

因此当前冲突不是“有没有证据”，而是：

**family 判据文档与后端结构落表文档使用的归类视角不同。**

---

## 4. 当前第一版 canon 的原则

### 原则 1：后端 canon 以“当前结构落表版本”为准

对于后端输出层而言，当前 family canon 的最高优先级来源是：

1. `mini_transformer_representative_anchor_table_v1_2026-04-22.md`
2. `mini_transformer_family_regime_table_v1_2026-04-22.md`

因为这两份文档已经承担了：

- 输入接口固定
- 对象实体化
- importance / lane / validation 的承接角色

### 原则 2：方法生长文档作为证据层，不直接覆盖 canon

下列文档当前仍然非常重要，但它们的角色是：

- 提供边界证据
- 记录不确定性
- 解释为什么当前表这样落

而不是：

- 直接改写 backend v1 的 family 归属

这些文档包括：

- analysis cards
- family cards
- outlier cards
- synthesis
- boundary cases

### 原则 3：family assignment 与 bottleneck explanation 必须分开

当前 backend v1 中：

- family assignment 优先由 `phase + route primitive + hardware template + current backend usefulness` 决定
- bottleneck explanation 优先作为 `resource_signature / tuning target / validation note` 保留

也就是说：

**一个 kernel 可以在 family 层被放入某条结构路径，同时继续保留与该 family 不完全同质的瓶颈解释。**

### 原则 4：backend v1 允许“带保留条款的吸收”

当前第一版不是要求所有 family 都完全纯净，而是允许：

- weak-share family
- context-split regime
- canonically absorbed but still review-needed objects

只要这些保留条款被显式记录，就足以支撑后端输出层。

---

## 5. 当前 backend v1 的唯一 canonical mapping

## 5.1 Anchor -> Family Canon

| rep_kernel_id | kernel_name | canonical_family_id | canonical_status | rationale |
|---|---|---|---|---|
| `A1` | `gemm_tiled` | `F1_dense_tiled` | `stable` | 是 dense compute 主锚点，family 归属稳定 |
| `A2` | `attention_score` | `F1_dense_tiled` | `weak-share` | 与 `gemm_tiled` 共享 dense tiled compute 主线，但保留 shmem 差异 |
| `A3` | `softmax_kernel` | `F2_reduction_normalize` | `stable-with-context-split` | 当前后端 v1 中按 reduction / normalize 主线吸收，并在 regime 层保留 cache / DRAM 特征 |
| `A4` | `context_mul` | `F3_streaming_aggregation` | `stable-singleton` | 当前作为独立 aggregation family 进入后端层 |
| `A5` | `layernorm_kernel` | `F2_reduction_normalize` | `canonically-absorbed-review-needed` | backend v1 先吸收到 reduction family，但保留第二轮 review 标记 |
| `A6` | `residual_add` | `F4_elementwise_fusion` | `canonically-absorbed-with-bottleneck-note` | backend v1 优先按 route/template 进入 elementwise family，同时保留 DRAM-dominated note |

## 5.2 Family -> Regime Canon

| family_id | canonical_regimes | notes |
|---|---|---|
| `F1_dense_tiled` | `R1_projection_dense`, `R2_attention_score_dense` | 共享 compute 主线，但必须拆成两个 regime |
| `F2_reduction_normalize` | `R3_softmax_reduction`, `R4_layernorm_reduction` | family 层共享 reduction template，regime 层按上下文拆分 |
| `F3_streaming_aggregation` | `R5_context_streaming` | 当前为 singleton regime |
| `F4_elementwise_fusion` | `R6_residual_elementwise` | 当前为 singleton regime，但需保留 DRAM note |

---

## 6. 如何解释当前最关键的两个冲突对象

### 6.1 `layernorm_kernel`

当前 canon 的处理方式是：

- **family 判据生长阶段**：它曾被当作 mixed/outlier 检验样本，这个判断仍然保留为证据
- **backend v1 输出阶段**：它先被 canonically absorbed 到 `F2_reduction_normalize`

原因不是说 earlier outlier 判断“错了”，而是：

1. 它在 `route primitive / template` 上已经稳定属于 `Reduction / Normalize`
2. 后端输出层需要一个可进入 lane 的对象，而不能长期悬空
3. 它的 uncertainty 可以通过：
   - `canonical_status = canonically-absorbed-review-needed`
   - regime-level note
   - writeback review
   来保留

因此当前最准确的表述不是：

- `layernorm_kernel` 已被彻底证明属于稳定 reduction family

而是：

- `layernorm_kernel` 在 backend v1 中先被吸收到 reduction family，以支持后端闭环，但仍保留第二轮 review 权利

### 6.2 `residual_add`

当前 canon 的处理方式是：

- **瓶颈解释层**：它仍然是最稳定的 DRAM-dominated 样本
- **backend v1 family assignment 层**：它归入 `F4_elementwise_fusion`

这个决定的核心原因是：

1. 当前 backend family 层优先组织的是 `route primitive + hardware template`
2. `residual_add` 在 workload 路径角色上稳定属于 `Elementwise Fusion`
3. 它的 `DRAM bandwidth` 特征应保留在：
   - `resource_signature`
   - `recommended_tuning_target`
   - `validation lane`
   - `writeback`

因此它不是被“改写”为非 DRAM 样本，而是：

**在结构归属层归入 elementwise family，在解释与验证层继续保留 DRAM 主导性。**

---

## 7. 当前各类文档在 backend v1 中的角色定位

| 文档类型 | 当前角色 | 是否直接改写 canon |
|---|---|---|
| `Representative Anchor Table` | 输入接口 | `Yes` |
| `Family / Regime Table` | 结构落表 | `Yes` |
| route/template 对照表 | 辅助解释与对齐 | `Yes, as support` |
| analysis cards | 证据层 | `No` |
| family cards | 方法生长记录 | `No` |
| outlier cards | 边界保留记录 | `No` |
| synthesis | 方法说明层 | `No` |
| validation documents | 后端证明层 | `No` |

因此当前 backend v1 的文档优先级可以简写为：

`Anchor / Family-Regime Tables > route-template table > cards / synthesis / outlier notes`

---

## 8. 对后端输出层的直接影响

当前 canon 一旦固定，后端输出层就可以稳定继续生成：

1. `Family Table`
2. `Regime Table`
3. `Priority & Lane Table`
4. `Validation Worksheet`
5. `Writeback Map`

其中需要特别注意：

### 8.1 `canonical_status` 应进入后端表

建议在后续结构化 JSON 中补入：

- `canonical_status`

用于区分：

- `stable`
- `weak-share`
- `stable-singleton`
- `canonically-absorbed-review-needed`
- `canonically-absorbed-with-bottleneck-note`

### 8.2 `residual_add` 的 DRAM 性不能丢

虽然其 canonical family 是 `F4_elementwise_fusion`，但后端表中必须继续保留：

- `resource_signature = dram-dominated elementwise path`
- `lane_type = constraint / memory-side check`

### 8.3 `layernorm_kernel` 的 review 权不能丢

虽然其 canonical family 是 `F2_reduction_normalize`，但后端表中必须继续保留：

- `review_flag = second_round_review`
- `validation_status = pending-review`

---

## 9. 当前第一版最推荐的实现姿态

当前不建议再把所有历史文档硬改成完全一致。

更稳的做法是：

1. 新增这份 `Family Canon` 文档
2. 让后端输出层只引用这份 canon
3. 保留旧文档作为方法演化与边界证据
4. 在第二轮 family review 时，再决定是否反向清理旧表述

这样可以避免：

- 为了表面一致性重写证据文档
- 把方法演化过程抹掉
- 提前消灭本来有价值的 uncertainty

---

## 10. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. 当前 backend v1 的 family canon 以 `Representative Anchor Table` 和 `Family / Regime Table` 为准。
2. `layernorm_kernel` 在 backend v1 中先 canonically 吸收到 `F2_reduction_normalize`，但保留第二轮 review。
3. `residual_add` 在 backend v1 中归入 `F4_elementwise_fusion`，但继续保留 DRAM-dominated bottleneck note。
4. analysis cards / family cards / outlier cards 当前是证据层与方法生长层，不直接改写 backend v1 canon。
5. 有了这份 canon，后端输出层就可以继续稳定生成 priority、lane、validation 与 writeback 对象。
