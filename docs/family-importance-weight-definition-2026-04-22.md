# Family Importance Ratio 第一版定义

日期：2026-04-22

## 1. 文档目的

这份文档用于给出当前方法线中最关键但仍未固定的一部分：

**family-level importance ratio 的第一版定义。**

当前我们已经明确：

- `PKA` 负责前端 representative compression
- `STEM+ROOT` 提醒我们必须考虑 runtime heterogeneity
- `squash + batch` 负责把 compressed objects 继续组织成：
  - phase
  - family
  - representative execution regime

但如果我们最终希望把这些结构真正用于：

- simulator validation
- tuning priority
- design-space exploration

那么就必须回答：

**不同 family 之间，谁更重要，应该先调谁。**

这份文档的目标不是给出最终完美公式，而是给出：

**可实现、可解释、可逐步演化的第一版 importance ratio 定义。**

---

## 2. 为什么需要 importance ratio

当前我们已经知道，family 不能只是分类结果。

如果只有 family，而没有 importance ratio，那么后续仍然会出现：

- 所有 family 被同等对待
- 后续 simulator 调参优先级仍然依赖人工经验
- representative kernels 只是换了个更整齐的组织方式，但没有真正减少后端决策复杂度

因此，importance ratio 的作用是：

1. 把 family 从“结构对象”变成“决策对象”
2. 把 representative compression 延伸到 simulator tuning priority
3. 为后续 family-aware validation 和 tuning 提供排序依据

---

## 3. 当前最重要的原则

### 原则 1：importance ratio 不是单一来源

当前不能把 importance ratio 简化为：

- 时间占比
或
- invocation 数量
或
- 某个 hotspot 指标

因为这些都只能反映 family 的某一个面。

更合理的理解是：

**importance ratio 是多种权重共同作用后的结果。**

### 原则 2：工作模式本身不能直接推出 importance

family 告诉我们：

- 对象共享什么机制

但不能直接告诉我们：

- 它占 workload 的多少
- 它是不是值得优先调参

所以：

**family 是 importance 的结构骨架，不是 importance 本身。**

### 原则 3：工作量也不能直接推出 importance

如果一个 family：

- 很大
- 很重
- 占时间很多

它当然重要，但不一定最值得优先调。

因为还有一种可能：

- 它很重
- 但对 simulator 参数不敏感
- 你调它几乎没收益

所以真正的 importance 必须同时考虑：

- workload relevance
- decision leverage

### 原则 4：第一版先做半定量，而不是追求闭式真理

当前阶段最稳的策略不是一开始就追求：

- 严格最优
- 全数学化闭式公式

而是先做：

**结构清楚、字段明确、可实现的第一版。**

---

## 4. 第一版 importance ratio 的三层结构

当前最推荐的第一版定义，把 importance ratio 拆成三层：

1. `Coverage Weight`
2. `Time Weight`
3. `Decision Weight`

这三层的关系可以理解为：

- `Coverage Weight` 回答：**它覆盖多广**
- `Time Weight` 回答：**它花了多久**
- `Decision Weight` 回答：**它值不值得先调**

最终 importance ratio 则由这三层共同决定。

---

## 5. Coverage Weight

### 5.1 定义

`Coverage Weight` 表示：

**某个 family 在整个 compressed workload 中覆盖了多大范围。**

它反映的是：

- 这个 family 代表了多少 representative kernels
- 这些 representative kernels 覆盖了多少原始 kernel / invocation

### 5.2 直观意义

如果一个 family 覆盖了大量代表对象，那么它在 workload 中更具有：

- 结构广度
- 覆盖范围
- 代表性

所以 Coverage Weight 更像：

**family 的范围权重**

### 5.3 第一版实现建议

当前第一版可以定义为：

`CoverageWeight(f) = family f 覆盖的原始 invocation 数 / 全 workload invocation 总数`

如果当前没有完整 invocation 级 membership，也可以退化为：

`CoverageWeight(f) = family f 覆盖的 representative anchors 数 / 全部 representative anchors 数`

### 5.4 注意事项

Coverage Weight 不反映：

- 单次执行有多重
- 对性能有多敏感

它只反映：

**覆盖范围**

---

## 6. Time Weight

### 6.1 定义

`Time Weight` 表示：

**某个 family 在整个 workload 执行时间中占多大比例。**

### 6.2 直观意义

如果一个 family 贡献了大量运行时间，那么它很可能是：

- 主要 bottleneck candidate
- 主要优化对象

所以 Time Weight 更像：

**family 的运行时重量**

### 6.3 第一版实现建议

当前第一版可以定义为：

`TimeWeight(f) = family f 的加权执行时间 / 全 workload 的总加权执行时间`

其中：

- 单个 representative kernel 的执行时间，可来自：
  - profiling
  - silicon measurement
  - sampled simulation result
- 加权方式可用：
  - representative anchor 的 time weight
  - 或 membership count

### 6.4 注意事项

Time Weight 比 Coverage Weight 更接近：

- hotspot
- performance impact

但它仍然不能单独决定 tuning priority。

---

## 7. Decision Weight

### 7.1 定义

`Decision Weight` 表示：

**某个 family 对后续 simulator 参数决策的影响程度。**

这是三层里最重要、也最难定义的一层。

### 7.2 为什么需要这一层

因为不是所有时间占比高的 family 都值得优先调参。

一个 family 可能：

- 时间占比很大
- 但对关键参数几乎不敏感

另一个 family 可能：

- 时间占比中等
- 但一旦调参数，其影响会明显改变最终结论

因此，Decision Weight 的作用是：

**把“运行时重要”转换成“调参时重要”。**

### 7.3 第一版的定义方式

当前第一版不建议直接定义成复杂公式，而建议用：

**半定量评分**

例如可以从以下三个因素构成：

#### 1. Structural Priority

表示该 family 是否位于：

- 主计算路径
- 核心 phase
- 关键 algorithm route

#### 2. Mechanism Sensitivity

表示该 family 是否对当前目标参数组敏感，例如：

- 对 cache capacity 敏感
- 对 register pressure 敏感
- 对 shared memory / occupancy 敏感

#### 3. Reuse Value

表示该 family 的 simulator reasoning 是否可复用：

- 是否能代表一整组 regime
- 是否能带动多个对象的验证

### 7.4 第一版实现建议

当前可以先给每个 family 一个离散评分：

- `High`
- `Medium`
- `Low`

或数值评分：

- `3 / 2 / 1`

再根据：

- 主路径位置
- sensitivity class
- 可复用范围

人工或半自动赋值。

### 7.5 注意事项

Decision Weight 当前阶段不需要追求绝对自动化。  
第一版只要：

- 规则清楚
- 赋值依据明确
- 能和后续 tuning priority 对接

就已经足够。

---

## 8. 第一版 importance ratio 的组合方式

当前推荐的第一版组合方式不是一开始就上复杂模型，而是：

### 方案 A：乘法型

`Importance(f) = CoverageWeight(f) × TimeWeight(f) × DecisionWeight(f)`

优点：

- 直观
- 容易实现
- 易于解释“其中一层很弱会拉低整体重要性”

缺点：

- 对 Decision Weight 的定义敏感

### 方案 B：加权和型

`Importance(f) = α·CoverageWeight(f) + β·TimeWeight(f) + γ·DecisionWeight(f)`

优点：

- 更平滑
- 不容易被单项极小值压死

缺点：

- 需要先定 α / β / γ

### 当前建议

第一版更建议先用：

**加权和型**

因为当前 Decision Weight 还比较粗，乘法型容易让结果不稳定。

一个最简单的第一版可以是：

`Importance(f) = 0.3·CoverageWeight(f) + 0.4·TimeWeight(f) + 0.3·DecisionWeightNorm(f)`

其中：

- `DecisionWeightNorm(f)` 把 `High / Medium / Low` 映射到数值区间，例如：
  - `High = 1.0`
  - `Medium = 0.6`
  - `Low = 0.3`

这只是第一版默认配置，后续可以再调。

---

## 9. Importance Ratio 如何从 family 推到 tuning priority

importance ratio 的最终用途不是停留在表格里，而是生成：

**family-level tuning priority**

当前最稳的映射方式是：

### Priority 1：High-Importance Families

这类 family：

- 优先进入 simulator perturbation
- 优先分配 tuning budget
- 优先形成 representative regimes

### Priority 2：Medium-Importance Families

这类 family：

- 用于补充验证
- 作为约束对象
- 避免主参数优化伤害关键次级 family

### Priority 3：Low-Importance Families

这类 family：

- 可只做轻量验证
- 暂不作为主调参对象

所以 importance ratio 的真正作用是：

**把 family 从“被分析对象”变成“被排序对象”。**

---

## 10. 当前最小可实现字段

为了让第一版 importance ratio 可落地，当前建议至少补齐以下字段。

### 在 Family Table 中增加

| 字段 | 含义 |
|---|---|
| `coverage_weight` | family 覆盖权重 |
| `time_weight` | family 时间权重 |
| `decision_weight` | family 决策权重 |
| `importance_score` | 综合 importance 分数 |
| `priority_class` | High / Medium / Low |

### 在 Regime Table 中增加

| 字段 | 含义 |
|---|---|
| `parent_family_id` | 所属 family |
| `family_importance_score` | 上层 family 重要性 |
| `regime_local_weight` | family 内局部权重 |
| `regime_priority` | 进入 simulator lane 的优先级 |

这样后续就能自然走到：

`family importance -> regime selection -> simulator tuning order`

---

## 11. 当前阶段不该做的事情

### 11.1 不要把 importance 简化成单一 hotspot ratio

这样会退回传统 profiling 逻辑。

### 11.2 不要一开始追求完全自动化

第一版更重要的是：

- 结构清楚
- 字段明确
- 能服务后续实验

### 11.3 不要把 family importance 和 regime importance 混在一起

family 是上层组织对象，regime 是最终 simulator lane 对象。  
两层权重应区分。

---

## 12. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. family importance ratio 的作用，是把 family 从结构对象提升为调参决策对象。
2. 第一版 importance ratio 应至少由 `Coverage Weight`、`Time Weight` 和 `Decision Weight` 三层构成。
3. `Coverage Weight` 反映 family 覆盖多广，`Time Weight` 反映 family 花了多久，`Decision Weight` 反映 family 是否值得优先调参。
4. 当前阶段最稳的第一版实现，是采用半定量的 Decision Weight，并用加权和形成综合 importance score。
5. importance ratio 的最终目的，是生成 family-level tuning priority，并进一步指导 representative regimes 的 simulator 进入顺序。
