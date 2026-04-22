# Importance Ratio 验证计划

日期：2026-04-22

## 1. 文档目的

这份文档用于回答：

**我们如何证明 family-level importance ratio 不是一个好听但不可验证的概念，而是真能指导 simulator tuning priority。**

当前我们已经有：

- importance ratio 的第一版定义
- family / regime 的 schema

但如果没有验证计划，那么：

- importance score 只是表格里的一个数字
- tuning priority 仍然可能停留在人工直觉上

因此，这份文档的目标是：

1. 定义 importance ratio 应该被验证的命题
2. 给出第一版实验验证路径
3. 明确什么结果可以支撑方法有效

---

## 2. 要验证的核心命题

importance ratio 的验证，不应只看“分数是否好看”，而要看它是否真的改善后续 workflow。

当前最关键的验证命题有三条。

### 命题 1：importance ratio 能压缩后端调参对象

也就是说，基于 family / regime importance 排序后，我们不需要：

- 平均地看所有对象
- 手工同强度分析所有 family

而可以更集中地看少量高重要性对象。

### 命题 2：importance ratio 能更早定位真正值得调参的 family

也就是说，高 importance family 应该更可能：

- 占主要时间
- 对关键参数敏感
- 对后续结果有更大影响

### 命题 3：importance ratio 能改善 tuning workflow

也就是说，用 importance-guided 顺序做 tuning / validation 时，应比：

- 随机顺序
- 仅按 kernel 名字
- 仅按时间热点

更快达到有意义的结论。

---

## 3. 当前建议的验证对象

第一版验证不应铺太大，建议集中在：

### 3.1 mini-transformer 原型

因为当前已有：

- squash
- family boundary
- representative execution regime

这是最容易形成闭环的对象。

### 3.2 代表性 kernel / family 子集

第一版可以优先围绕：

- Dense Tiled Compute Family
- Reduction / Normalize Family
- Streaming Aggregation Family

不一定需要所有 family 一次做满。

---

## 4. 建议的对照基线

importance ratio 要证明自己有价值，必须有 baseline。

当前建议至少设置三种 baseline。

### Baseline A：No Priority

所有 family 一视同仁：

- 不排序
- 平均分配验证预算

### Baseline B：Time-Only Priority

只按 `time_weight` 排序：

- 谁耗时高，先看谁

### Baseline C：Name-Based / Manual Priority

按：

- kernel 名字
或
- 人工经验判断

给出优先级

### Ours：Importance-Guided Priority

按：

- `coverage_weight`
- `time_weight`
- `decision_weight`

综合形成的 `importance_score`

---

## 5. 第一版验证指标

当前建议至少看四类指标。

### 5.1 Object Reduction

验证问题：

**importance-guided 策略能否减少需要优先处理的对象数量。**

建议指标：

- Top-K families 覆盖的总 time weight
- Top-K regimes 覆盖的总 time weight
- 达到某覆盖率所需 family 数量

### 5.2 Early Gain

验证问题：

**importance-guided 顺序能否更早捕获主要调参对象。**

建议指标：

- 前 1 个 family 覆盖的 tuning-relevant time
- 前 2 / 3 个 families 覆盖的综合 importance
- 在固定预算下，能解释多少关键 runtime 行为

### 5.3 Sensitivity Concentration

验证问题：

**高 importance family 是否真的更可能对关键参数敏感。**

建议指标：

- 对选定参数扰动后，不同 family 的性能变化幅度
- importance 排序与 sensitivity 排序的相关性
- 高 importance family 的 sensitivity 是否显著高于低 importance family

### 5.4 Tuning Efficiency

验证问题：

**importance-guided tuning 能否更快得到有意义的 tuning 结果。**

建议指标：

- 达到某性能改善需要尝试多少 family / regime
- 达到某解释覆盖率需要多少 simulator runs
- importance-guided 与 baseline 在相同预算下的收益比较

---

## 6. 第一版实验路径

当前建议按三步走。

### Step 1：静态排序验证

先不跑复杂 simulator perturbation，只验证：

- importance score 排序是否稳定
- 与 time-only 排序有何不同
- 高 importance families 是否覆盖主路径与主 runtime

这一步主要验证：

**importance ratio 是否形成合理排序。**

### Step 2：局部参数扰动验证

选少量典型参数，例如：

- register-sensitive
- cache-sensitive
- reduction-related

观察：

- 不同 family 的响应
- importance 高的 family 是否更常成为主要响应对象

这一步主要验证：

**importance ratio 是否和 simulator sensitivity 对齐。**

### Step 3：priority-guided tuning workflow 对比

在有限预算下比较：

- 随机顺序
- time-only 顺序
- importance-guided 顺序

观察：

- 谁更快定位有效调参对象
- 谁更快覆盖主要 family

这一步主要验证：

**importance ratio 是否真正改善 workflow。**

---

## 7. 当前最推荐的成功判据

第一版实验不需要一开始就证明绝对最优，但至少应达到下面三条中的两条。

### 成功判据 A

Top 少量高 importance families 覆盖了大部分关键 runtime/time weight。

### 成功判据 B

importance-guided 顺序比 time-only / random 更早定位高 sensitivity family。

### 成功判据 C

importance-guided workflow 在相同 simulator budget 下，达到更高的解释覆盖率或调参收益。

如果满足这三条中的两条，就足以说明：

**importance ratio 不是附属标签，而是真正有 workflow 价值的排序层。**

---

## 8. 当前阶段不该做的事情

### 8.1 不要一开始追求全参数空间验证

第一版应选择少量代表性参数。

### 8.2 不要把“时间热点命中”误当成全部成功

如果 importance ratio 只是和 time-only 一样，那它的增量价值有限。

### 8.3 不要忽略低 importance family 的约束角色

低 importance family 不一定完全没价值，它们可能是：

- tuning constraint
- regression detector

因此验证时也要保留“约束对象”的视角。

---

## 9. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. importance ratio 的价值不在于数字本身，而在于它是否能改善后续 tuning workflow。
2. 第一版应至少验证三类问题：对象压缩、敏感性集中、workflow 效率。
3. baseline 至少应包括：不排序、仅按时间排序、人工/名字排序。
4. 第一版最稳的验证路径是：静态排序 -> 局部参数扰动 -> priority-guided workflow 对比。
5. 如果 importance-guided 策略能更早定位高价值 family，并在相同预算下获得更高收益，就足以支撑该层方法的有效性。
