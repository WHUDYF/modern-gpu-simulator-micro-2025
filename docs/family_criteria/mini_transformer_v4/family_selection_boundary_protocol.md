# Family Selection / Boundary Protocol

日期：2026-04-19

## 1. 文档目的

这份文档用于把当前 `mini_transformer_v4` 方法原型里的 family 选择逻辑，整理成一套可执行的边界协议。

当前阶段我们已经有：

- `squash` 给出的 phase-level 时间结构
- `Route Primitive`
- `Hardware Execution Template`
- 两组关键 boundary case
- 六个 kernel 的 analysis cards

但在没有协议的情况下，我们仍然容易在下面几个问题上反复摇摆：

- 什么时候两个 kernel 可以归到同一 family
- 什么时候必须拆开
- 什么时候暂时不下最终结论，保留为 outlier

因此，这份文档的目标不是定义最终全领域 family taxonomy，而是先回答：

**在当前原型里，family selection 到底按什么顺序、依据什么证据、在什么条件下做出“并 / 拆 / 暂缓”决策。**

---

## 2. family 在当前阶段的定义

当前阶段，我们不把 family 定义成：

- 算子名分组
- 上层模块名分组
- 单一瓶颈项分组

当前最稳的定义是：

**family 是一组在相同 phase 上下文中，能够共享同一条 simulator reasoning lane 的 kernel / primitive 对象集合。**

这个定义有三个关键词：

1. **phase 上下文**
   - family 不是脱离时间结构定义的
2. **共享工作模式**
   - family 不等于语义相似
3. **共享 simulator reasoning lane**
   - family 的价值在于后续验证和调参可复用

所以，当前阶段的 family 是一个：

**面向 simulator 组织的结构对象**

而不是单纯的标签系统。

---

## 3. 当前协议的核心原则

### 原则 1：先看工作模式，再看调参方向

family 的第一层任务是回答：

**这些对象是否共享同一种执行模式。**

它不应该先由：

- 当前谁最慢
- 当前哪个瓶颈最大
- 当前哪个指标最突出

来决定。

瓶颈项应留给 family 内调参与权重组织，而不是直接拿来定义 family。

---

### 原则 2：先看边界 case，再做并类

当前阶段我们明确采用：

**boundary-first**

也就是：

- 先用最容易混淆的样本逼出判据
- 再决定哪些对象值得共享解释

而不是：

- 先铺满全量卡片
- 再事后强行找规则

---

### 原则 3：并类要保守，拆分类要有理由

当前阶段的默认策略应是：

- **并类保守**
- **拆分类有证据**
- **证据不够时保留 outlier / unresolved**

因为当前方法还在生长过程中，过早强行并类比保留不确定性更危险。

---

### 原则 4：family 决策必须同时看 Route Primitive 和 Hardware Template

当前阶段任何 family 决策，如果只看：

- 上层语义
或
- 硬件模板

都不够稳。

更合理的做法是同时检查：

1. `Route Primitive`
2. `Hardware Execution Template`
3. `phase` 上下文
4. 关键 shape / size regime

---

## 4. family 决策输入项

当前协议中，每个候选对象至少应有以下输入信息：

### 4.1 Phase Context

- 该对象主要出现在哪个 phase
- 是否属于稳定主 phase
- 是否只是短暂过渡段

### 4.2 Route Primitive

- 它在 workload 主计算路径中的角色是什么

### 4.3 Hardware Execution Template

- 它在 GPU 上主要通过什么执行模板实现

### 4.4 Evidence Metrics

至少包括：

- compute / dram / l1_hit / occupancy / warp_cyc
- register / shmem / waves 等边界性指标

### 4.5 Boundary Notes

- 当前与哪个已有对象最容易混淆
- 当前共享点是什么
- 当前区分点是什么

---

## 5. 协议流程

当前阶段最稳的选择流程如下。

### Step 1：先确定是否在同一 phase 上下文中讨论

如果两个对象：

- 不在同一稳定 phase 中出现
- 或者一个属于主 phase，一个只是边界过渡段

那么默认：

**先不进入同一 family 候选讨论。**

理由很简单：

family 的共享前提之一是能够共享后续 simulator reasoning lane；如果 phase 上下文不同，这个前提通常不成立。

---

### Step 2：检查 Route Primitive 是否相同

如果两个对象的 `Route Primitive` 不同，则默认：

**不直接并为同一个强 family。**

但这不意味着它们一定要完全断开，因为仍然要继续看下一步：

- 它们是否共享同一种 hardware template
- 它们是否属于同一条更高层 route

当前阶段推荐的处理方式是：

- `Route Primitive` 不同
- 但 `Hardware Template` 相同

则进入：

**弱共享 / 边界候选**

而不是直接强并。

---

### Step 3：检查 Hardware Execution Template 是否相同

如果 `Hardware Execution Template` 也不同，则默认：

**应拆开。**

因为这说明：

- workload 角色不同
- GPU 执行方式也不同

这种情况下继续并类，往往只会得到语义层的假相似。

---

### Step 4：如果 Route 不同但 Template 相同，进入“弱共享”检查

这一步是当前协议里最关键的一层。

如果两个对象：

- `Route Primitive` 不同
- 但 `Hardware Execution Template` 相同

那么当前更稳的做法不是立刻并类，而是进入：

**弱共享检查**

当前可以把它理解成：

**共享底层执行骨架，但在 workload 路线中扮演不同角色。**

如果还存在显著边界指标差异，例如：

- `shmem`
- `waves`
- locality 特征

则当前更适合：

**保留在同一候选 family 边界，而不是完全合并。**

---

### Step 5：如果 Route 相同但 Template 不同，优先拆分

这一步是为了防止上层语义误导。

如果两个对象：

- 属于同一条上层 route
- 但 `Route Primitive` 或 `Hardware Template` 不同

当前应默认：

**拆分为不同 primitive / family 子类。**

理由是：

同一条 workload 路线内部，本来就可能包含多个执行骨架。

如果只因为“都属于 attention”就并类，会直接削弱方法的硬度。

---

### Step 6：证据不足时，保留 unresolved / outlier

如果当前证据出现：

- 共享点很强
- 区分点也很强
- 但还不足以稳定决定“并”还是“拆”

那么当前阶段更推荐：

**保留 unresolved boundary 或 outlier。**

不要强行做绝对裁决。

这也是当前阶段 protocol 的一条底线：

**不为了结构完整而牺牲边界真实性。**

---

## 6. 当前协议下的三种输出

当前阶段，family 选择的结果不应只有“并”或“拆”两种。

更稳的输出应有三类：

### 6.1 Strong Share

条件：

- phase 上下文一致
- Route Primitive 一致
- Hardware Template 一致
- 没有强边界性异质指标

含义：

**可以进入同一强 family，并优先共享 simulator lane。**

---

### 6.2 Weak Share / Boundary Candidate

条件：

- 共享底层执行模板
或
- 共享同一主架构解释

但仍存在：

- Route 角色差异
- 次级实现特征差异
- shape / locality / shmem 等边界差异

含义：

**暂时可放入同一候选 family 边界讨论，但不能直接当成完全同质对象。**

---

### 6.3 Split / Outlier / Unresolved

条件：

- Route 和 Template 都不共享
或
- 共享点过弱
或
- 区分点强到足以改变后续 simulator lane
或
- 当前证据不足以下最终结论

含义：

**应拆开，或先作为 outlier / unresolved 保留。**

---

## 7. 当前协议下的两个关键案例

### 7.1 `gemm_tiled` vs `attention_score`

当前已知：

- `Route Primitive`
  - `gemm_tiled`: `Dense Projection/Transform`
  - `attention_score`: `Pairwise Score`
- `Hardware Template`
  - 二者都接近 `Dense Tiled Compute`
- 关键边界差异：
  - `attention_score` 的 shared memory / waves 特征更强

按当前协议判断：

- phase 上下文：可比
- Route Primitive：不同
- Hardware Template：相同
- 边界性指标：明显存在

因此当前应输出为：

**Weak Share / Boundary Candidate**

而不是：

- 完全并类
或
- 完全无关

---

### 7.2 `softmax_kernel` vs `context_mul`

当前已知：

- 上层 route：
  - 同属 `attention readout route`
- `Route Primitive`
  - `softmax_kernel`: `Reduction / Normalize`
  - `context_mul`: `Weighted Aggregation`
- `Hardware Template`
  - `softmax_kernel`: `Reduction Template`
  - `context_mul`: `Streaming Aggregation Template`

按当前协议判断：

- phase 上下文：可比
- Route Primitive：不同
- Hardware Template：不同

因此当前应输出为：

**Split / Unresolved Boundary（倾向拆分）**

这组 case 的意义在于：

**同属一条上层 route，不等于属于同一个 family。**

---

## 8. 当前协议对调参层的影响

当前协议最重要的意义，不只是帮助分组，而是帮助后续调参对象收缩。

如果一个对象已经进入：

- `Strong Share`

那么后续调参与 simulator lane 可以优先复用。

如果一个对象仍处于：

- `Weak Share`
或
- `Unresolved`

那么后续调参时应先保留独立观察，不应过早共享结论。

因此，当前协议不是静态分类规则，而是：

**面向 simulator reasoning lane 和 tuning reuse 的前置过滤器。**

---

## 9. 当前协议的边界

这份协议仍然有明确边界：

### 9.1 当前不处理最终全领域 taxonomy

它当前只服务于：

- `mini_transformer_v4`
- 当前已有的 6 个核心 kernel
- 当前已有的两组 boundary case

### 9.2 当前不把 shape regime 完整纳入强规则

shape / size regime 当前已明确重要，但还没有形式化到协议主规则中。  
后续在 `representative execution regime` 文档中，应把这部分补上。

### 9.3 当前不直接输出 simulator 参数处方

协议只回答：

- 哪些对象值得共享 lane
- 哪些对象不能直接共享 lane

它不直接给出：

- cache / shmem / register 具体怎么调

那属于 family 内调参与 decision weight 层的后续工作。

---

## 10. 当前阶段的最简结论

当前阶段，我们已经可以比较稳地把 family selection / boundary protocol 总结为：

1. family 不应由算子名、模块名或单一瓶颈项直接定义。
2. family 决策必须同时看：
   - phase 上下文
   - Route Primitive
   - Hardware Execution Template
   - 边界性指标
3. `Route Primitive` 不同但 `Template` 相同，应优先视为弱共享边界，而不是直接强并。
4. `Route Primitive` 和 `Template` 都不同时，应优先拆分。
5. 证据不足时，应保留 unresolved / outlier，而不是为了结构完整强行合并。

因此，当前协议最重要的作用是：

**把 family 讨论从“经验分组”推进成“有顺序、有条件、有保留项的结构化边界决策过程”。**
