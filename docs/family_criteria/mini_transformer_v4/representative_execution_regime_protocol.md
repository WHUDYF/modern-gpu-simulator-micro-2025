# Representative Execution Regime Protocol

日期：2026-04-19

## 1. 文档目的

这份文档用于回答当前方法线里的下一个关键问题：

**当我们已经有了 phase、Route Primitive、Hardware Execution Template 和 family boundary protocol 之后，究竟应该把什么对象送进 simulator lane？**

当前最重要的结论是：

**后续进入 simulator lane 的单位，不应只是“一个代表 kernel”，而应是“一个代表执行区间（representative execution regime）”。**

因此，这份文档的目标不是继续解释 family，而是把下面几件事固定下来：

1. 什么叫 representative execution regime
2. 为什么它比 representative kernel 更稳
3. 在当前原型中，应按什么规则提取 regime
4. 它如何服务后续 simulator lane / tuning lane

---

## 2. 为什么不能只提取 representative kernel

当前阶段已经可以明确，直接提取“代表 kernel”有三个明显风险。

### 2.1 同名 kernel 不一定属于同一调参对象

例如在 Transformer 中，多个 GEMM 虽然都叫 GEMM，但可能分别来自：

- Q/K/V projection
- output projection
- FFN up projection
- FFN down projection

它们在 `Hardware Execution Template` 上可能都接近 `Dense Tiled Compute`，  
但它们的：

- shape
- phase 上下文
- 在 workload 路线中的角色
- 权重

不一定相同。

因此，单纯说“挑一个代表 GEMM”过于粗糙。

### 2.2 同一 family 内部仍然可能包含多个稳定工作区间

即便两个对象已经被放到同一个 family，它们也可能在：

- sequence length
- head dimension
- batch size
- M/N/K 形状
- memory locality

上落入不同 regime。

这意味着：

**family 仍然只是结构压缩层，不等于最终 simulator lane 单位。**

### 2.3 调参真正复用的不是名字，而是“工作区间”

后续 simulator 调参与验证真正需要复用的是：

- 相同 phase 上下文
- 相同 route / template
- 相近 shape / size
- 相近资源敏感性

所以最终应复用的对象不是“一个算子名字”，而是：

**一段稳定的执行工作区间。**

---

## 3. 当前定义：什么是 representative execution regime

当前阶段，我们将 representative execution regime 定义为：

**在同一稳定 phase 中，具有相同或相近 Route Primitive、Hardware Execution Template、shape / size 区间以及主导资源行为的一组执行实例所对应的代表对象。**

这个定义里有五个关键词：

1. **同一稳定 phase**
2. **相同或相近 Route Primitive**
3. **相同或相近 Hardware Execution Template**
4. **相近 shape / size 区间**
5. **相近主导资源行为**

也就是说，一个 regime 不是单点样本，而是：

**一类在当前方法视角下值得被统一解释和统一调参的执行区间。**

---

## 4. regime 的最小描述字段

当前阶段，每个 representative execution regime 至少应包含以下字段。

### 4.1 Phase ID

表示：

- 它属于哪个稳定 phase
- 是否属于主 phase 或边界 phase

### 4.2 Route Primitive

表示：

- 它在 workload 路线中的功能角色

### 4.3 Hardware Execution Template

表示：

- 它在 GPU 上的主执行骨架

### 4.4 Shape / Size Regime

表示：

- 它的形状区间，而不是单个孤立 shape

对于当前 Transformer 原型，至少应关注：

- M / N / K
- sequence length
- head dimension
- batch size

### 4.5 Resource Signature

表示：

- 该 regime 的主导资源压力或敏感性

当前阶段至少可记录：

- register / occupancy
- shared memory
- DRAM bandwidth
- cache-capacity / DRAM-pressure
- locality / L1-resident
- reduction / synchronization

### 4.6 Weights

表示：

- coverage weight
- time weight
- decision weight（当前阶段可先定性）

---

## 5. regime 提取协议

当前阶段最稳的 regime 提取流程如下。

### Step 1：先按 phase 切开

每个 representative execution regime 必须先从某个稳定 phase 内部提取。

默认规则：

- 不跨主 phase 直接合并 regime
- 过渡 phase 中的对象优先单独观察

原因：

如果 phase 上下文不同，即使 kernel 名字相同，也不一定应该共用同一个 simulator lane。

---

### Step 2：在 phase 内按 Route Primitive 切第一刀

在同一个 phase 内，先按 `Route Primitive` 分开。

原因：

这一步先保住 workload 主计算路径的结构。

例如在 attention 路线中：

- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`

不应被直接揉成一个大 attention cluster。

---

### Step 3：在同一 primitive 内按 Hardware Template 切第二刀

如果同一个 `Route Primitive` 内存在多个不同硬件模板，则应继续拆开。

默认规则：

- template 不同，不直接共享同一个 regime

原因：

代表对象最终要服务 simulator lane 和调参，因此必须保留执行骨架差异。

---

### Step 4：在同一 primitive + template 内按 shape / size 形成 regime

这是当前 protocol 里最关键的新环节。

只有当对象已经满足：

- 同 phase
- 同 Route Primitive
- 同 Hardware Template

时，才进入 shape / size regime 合并。

当前阶段的规则应是：

- 不用单个具体 shape 作为 regime
- 用一组相近 shape 区间作为 regime

更直觉地说：

**regime 是“这一段 shape 空间里的典型工作方式”，而不是“某个孤立矩阵尺寸”。**

例如：

- 小序列长度 attention score
- 中序列长度 attention score
- 大序列长度 attention score

或：

- 小 M / 中 K projection
- 大 M / 中 K projection

当前阶段不强求写出严格数值边界，但至少要明确：

- regime 是区间，不是单点
- regime 划分必须服务后续 simulator 复用

---

### Step 5：用资源签名检查是否值得继续拆分

即使对象已经：

- 同 phase
- 同 primitive
- 同 template
- 同 shape regime

如果其资源签名仍然明显不同，例如：

- 一个是 register-limited
- 一个是 shmem-coupled
或
- 一个是 locality-dominated
- 一个是 DRAM-pressure-dominated

那么当前更稳的做法是：

**继续拆成两个 regime，或先保留 unresolved。**

也就是说，regime 合并的最后一关不是 shape，而是：

**它们是否真的值得共用同一条 simulator reasoning lane。**

---

### Step 6：在每个 regime 内选代表对象

当一个 regime 已经稳定后，再从其中挑选代表对象。

这里的“代表对象”可以是：

- 一个代表 kernel instance
- 一个代表 layer instance
- 一个代表 shape 点

但需要强调：

这些对象是 regime 的代表，不是直接从原始 workload 中盲选出来的“代表 kernel”。

---

## 6. 当前阶段 regime 选择的输出格式

当前阶段，每个 representative execution regime 至少应写成下面这种格式：

### 6.1 标准字段

- `phase`
- `route primitive`
- `hardware template`
- `shape regime`
- `resource signature`
- `coverage weight`
- `time weight`
- `decision weight`
- `representative object`
- `lane advice`

### 6.2 最小示意

例如：

`phase=P1`

`route primitive=Dense Projection/Transform`

`hardware template=Dense Tiled Compute`

`shape regime=mid-M / large-K projection regime`

`resource signature=register-limited`

`coverage weight=high`

`time weight=high`

`decision weight=high`

`representative object=layer-3 qkv gemm`

`lane advice=primary tuning lane`

这里最重要的是：

regime 的主体不是最后那个 `representative object`，  
而是前面这整个结构描述。

---

## 7. 当前原型里的三个典型 regime 方向

基于当前 `mini_transformer_v4` 的理解，至少已经可以预想出几类典型 regime。

### 7.1 Dense Projection / Transform Regime

典型特征：

- 主 phase 中反复出现
- `Route Primitive = Dense Projection/Transform`
- `Template = Dense Tiled Compute`
- shape 主要由不同投影路径决定
- 资源签名常常偏 `register-limited`

当前意义：

这是最有可能形成主调参对象的一类 regime。

---

### 7.2 Attention Score Regime

典型特征：

- `Route Primitive = Pairwise Score`
- `Template = Dense Tiled Compute`
- 与 projection 路线共享底层模板
- 但在 route 角色上独立
- shape 受 sequence / head 维度影响更强

当前意义：

这类 regime 不能因为 template 相近就被 projection 全吸收。

---

### 7.3 Attention Readout Sub-Regimes

这里至少应拆成两类：

#### A. `softmax` side

- `Route Primitive = Reduction / Normalize`
- `Template = Reduction Template`
- 资源签名偏 `cache-capacity / DRAM-pressure`

#### B. `context_mul` side

- `Route Primitive = Weighted Aggregation`
- `Template = Streaming Aggregation Template`
- 资源签名偏 `locality / L1-resident`

当前意义：

它们同属 attention readout route，  
但不应合成同一个 regime。

---

## 8. regime 与 family 的关系

当前阶段最稳的关系应写成：

- `family`
  - 是结构归属层
  - 决定哪些对象值得共享解释

- `representative execution regime`
  - 是落地执行层
  - 决定哪些对象真正进入 simulator lane

因此：

**family 先于 regime，regime 细于 family。**

可以把它理解成：

- family 先回答：谁属于同一种工作模式
- regime 再回答：在这种工作模式里，哪些具体区间值得保留为代表对象

---

## 9. regime 与权重的关系

当前阶段，regime 不应脱离权重讨论。

因为我们后续不是要平均保留所有 regime，而是要根据它们的权重决定：

- 哪些 regime 是主优化对象
- 哪些 regime 是约束对象
- 哪些 regime 可以暂时降优先级

因此，每个 regime 至少需要绑定：

### 9.1 Coverage Weight

它在主路径里覆盖了多少位置。

### 9.2 Time Weight

它实际花了多少时间。

### 9.3 Decision Weight

它对参数方向到底有多重要。

当前阶段不要求马上精确数值化，但要求：

**regime 的保留优先级必须能被这些权重解释。**

---

## 10. 当前协议的边界

### 10.1 当前还没有给出严格的 shape 数学划分

这一步当前仍以“工作区间”概念为主，后续如果需要更精细，可以再形式化。

### 10.2 当前还没有把 regime 直接绑定到 simulator 参数处方

这份文档只回答：

- regime 怎么来
- regime 为什么是比 representative kernel 更稳的对象

它不直接回答：

- 这个 regime 应该怎样改 cache / shmem / register 参数

### 10.3 当前只服务于 `mini_transformer_v4` 原型

后续扩展到更广 workload 时，应补：

- sparse gather/scatter 型 regime
- spatial local stencil 型 regime

---

## 11. 当前阶段的最简结论

到目前为止，我们可以比较稳地说：

1. 后续进入 simulator lane 的单位，不应只是“代表 kernel”，而应是“representative execution regime”。
2. 一个 regime 至少应由：
   - phase
   - route primitive
   - hardware template
   - shape regime
   - resource signature
   共同决定。
3. regime 提取顺序应为：
   - 先按 phase
   - 再按 route primitive
   - 再按 hardware template
   - 再按 shape regime
   - 最后用资源签名做一致性检查。
4. family 决定结构归属；regime 决定真正进入 simulator lane 的代表对象。
5. 这一步的意义，是把方法从“能分组”推进到“能真正组织 simulator 侧的代表对象与调参复用”。 
