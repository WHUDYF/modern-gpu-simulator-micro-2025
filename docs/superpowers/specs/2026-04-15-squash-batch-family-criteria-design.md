# Squash + Batch Family 判据框架设计

日期：2026-04-15

## 目标

本设计文档定义一套面向 `squash + batch` 的 family 判据框架，用于把复杂 workload 的 kernel
行为组织成可解释、可比较、可服务后续 simulator 验证的结构化对象；并以
`mini-transformer` 作为第一应用场景，验证这套框架是否能稳定产出中等粒度的 family 与 outlier。

这里的核心目标不是直接给出最终规则阈值，也不是立即实现一个全自动 pipeline，而是先建立：

- family 判据框架的输入边界
- family 判据的层级结构
- kernel 分析卡与 family 解释卡片
- mixed / ambiguous / outlier 的处理规则
- 在 `mini-transformer` 上的首轮试运行方式

## 边界

第一版设计只做下面这些事：

- 定义 family 判据框架的目标、输入、结构和输出
- 定义分析卡和 family 解释卡片的结构
- 定义不确定性、mixed kernel 和 outlier 的处理方式
- 规定如何在 `mini-transformer` 上应用这套框架

## 非目标

第一版明确不做下面这些事：

- 不展开 `delta`
- 不直接定义 simulator 具体参数处方
- 不追求全自动 family 划分
- 不立即做跨 workload 泛化证明
- 不把经验性 simulator 知识直接写入判据

## 第一版完成标准

第一版设计完成后，至少要能支持：

- 用统一框架分析 `mini-transformer` 的代表 kernel
- 产出中等粒度的 family
- 显式保留不确定性
- 给出为什么这些 family 对后续 simulator 验证组织有意义的定性说明

---

## 一、问题定义

这套框架要解决的问题不是：

- 单独对某一个 kernel 做更细 profiling
- 单独提出一个 simulator 参数调节技巧
- 单独展示某个 cluster 结果好不好看

它要解决的是一个更上游的问题：

**如何把复杂 workload 的执行行为，逐层收缩成后续可进入 simulator 验证的结构化对象。**

从这个角度看，`squash` 与 `batch` 都不是附属分析工具，而是中间结构层：

- `squash` 负责组织执行流的时间结构
- `batch` 负责组织 kernel 之间的共享机制结构

因此，两者的学术价值不主要体现在“单独发现了多少新现象”，而体现在：

**它们作为必要结构层，使后续架构解释和 simulator 验证能够成立。**

---

## 二、Squash 与 Batch 的分工

### Squash 的职责

`squash` 的作用不是简单地“把 kernel 分成几段”，而是：

- 将长执行流压缩为若干行为稳定的 phase
- 提取 workload 在时间维度上的结构
- 为后续代表 trace 选择和阶段级分析提供基础

如果没有 `squash`，复杂 workload 的执行流通常会出现：

- 时间序列过长，难以直接理解
- 不同阶段混在一起，无法稳定映射到后续分析对象
- 无法判断哪些 trace 可以代表某一整段行为

因此，`squash` 的价值在于把“原始执行流”转成“阶段化执行结构”，为端到端流程提供
phase-level 中间表示。

### Batch 的职责

`batch` 的核心定义是：

**一个解释层模块，用于识别共享同一架构机制的 kernel family。**

这意味着 `batch` 不应该只是：

- 按算子名分组
- 对已有标签做结果整理
- 为了展示方便做语义聚合

因为这些弱定义默认了：

- 同算子名 = 同架构行为
- 不同算子名 = 不同架构行为

而这在复杂 workload 中并不成立。

`batch` 真正要回答的是：

- 哪些 kernel 虽然名字不同，但受同一类架构机制主导
- 哪些 kernel 虽然名字相同，但行为异质，不能共用解释
- 当前 workload 的共享机制边界在哪里

### 当前对 Batch 的落地约束

在定义上，`batch` 停留在解释层；
在实现上，`batch` 识别出的 family 会进一步指导：

- 哪些 kernel 可以优先共用一类 simulator 验证主线
- 哪些 case 可以合并验证
- 哪些 kernel 必须作为 outlier 单独进入后续分析

因此，本设计采用一个明确原则：

**定义上选解释层，落地上实现验证分流。**

---

## 三、Family 判据框架的总体结构

family 判据框架采用三层结构。

### 第 1 层：执行模式粗分层

这一层只回答：

**这个 kernel 整体更像 compute-heavy、memory-heavy，还是 mixed。**

允许的状态包括：

- `compute-heavy`
- `memory-heavy`
- `mixed`
- `uncertain`

这里的作用不是得出最终 family，而是建立第一层组织结构，避免所有 kernel 在同一层直接平铺细分。

### 第 2 层：资源主导判据层

这一层是 family 边界的核心。

它回答：

**在当前执行模式下，哪个资源主导特征最能解释这个 kernel 的行为。**

第一版暂定允许的资源主导特征包括：

- register / occupancy
- DRAM bandwidth
- cache / locality
- shared memory

这里先不规定最终全集，也不写死阈值；第一版只要求：

- 每个 kernel 至少给出 1 个主导资源候选
- 必要时给出 1 个次级候选
- 允许写“边界不稳”

### 第 3 层：family / outlier 决策层

这一层把前两层的信息收束成结构化输出：

- 若多个 kernel 共享相近的执行模式和资源主导特征，则形成同一 family
- 若某个 kernel 无法稳定并入已有 family，则先宽松地标记为 outlier
- 若某个 kernel 同时呈现两个强信号，则允许保留为 `mixed / ambiguous`

### 该结构的核心原则

本设计采用如下暂定原则：

**执行模式用于粗分，资源主导特征用于决定 family 边界。**

---

## 四、输入与证据边界

### 主输入

第一版 family 判据框架的主输入只包括两类：

#### 1. Profiling / diagnosis 指标

例如：

- achieved occupancy
- compute throughput
- dram throughput
- l1/l2 hit rate
- warp cycles
- shmem usage
- waves / launch shape / block limit 一类结构性指标

#### 2. Kernel 语义信息

例如：

- kernel 属于哪类算子
- 它在 workload 中承担什么角色
- 它与上下游 kernel 的关系

这里的原则是：

**指标提供主证据，语义提供约束与解释。**

也就是说，语义不能单独决定 family，但可以帮助避免纯指标相似带来的误判。

### 辅助输入

第一版允许保留但不直接进入 family 判据的输入，包括：

- 更细粒度的实现细节
- 经验性 simulator 知识
- 已有处方经验
- 后验验证结果

这些信息可以写在分析说明里，但暂时不作为判据本身。

### 明确排除项

第一版明确不把下面这些东西直接纳入判据：

- “过去这个 kernel 常常是寄存器瓶颈”
- “在 simulator 里之前调这个参数有用”
- “某一轮 delta 指向了某个字段”

原因在于：

**在还没有通过闭环验证确认之前，这些知识属于后验经验，而不是前验中立的 family 判据。**

### 输入策略总原则

本设计采用如下输入策略：

**指标为主、语义约束、经验隔离。**

---

## 五、输出对象：分析卡与 Family 解释卡片

本设计明确拆成两个输出对象：

- kernel 分析卡
- family 解释卡片

两者不能混在一起。

### 5.1 Kernel 分析卡

其作用是：

**用统一模板记录单个 kernel 的可判据信息。**

第一版建议包含五个区块。

#### 区块 1：基本信息

- kernel 名称
- 算子语义
- 在 workload 中的位置或作用
- 代表性说明

#### 区块 2：执行模式粗分

- `compute-heavy / memory-heavy / mixed / uncertain`
- 允许“暂定”
- 这里不是最终结论，只是第一层组织

#### 区块 3：关键观测指标

- 只保留和 family 判断相关的指标
- 指标先记录事实，不急着解释

#### 区块 4：主导资源候选

- 1 个主候选
- 必要时 1 个次候选
- 允许写“边界不稳”

#### 区块 5：归属判断

- 暂定属于哪个 family
- 为什么能归进去
- 与相邻 family 的边界疑点
- 若不能归入已有 family，先标成 outlier / ambiguous

### 5.2 Family 解释卡片

其作用是：

**把多个 kernel 的共同解释压缩成一个可用于论文和后续验证的对象。**

第一版建议每张卡片包含以下内容。

#### 区块 1：family 标识

- family 名称
- 粗类名称
- 子类名称

第一版 family 命名采用分层命名：

- 粗类使用现象型命名
- 子类使用机制型命名

例如：

- `memory-heavy -> dram-dominated`
- `compute-heavy -> register-limited`

#### 区块 2：核心解释

- 这个 family 为什么成立
- 共享的架构主导特征是什么
- 哪些现象支持这一解释

#### 区块 3：代表 kernel

- 该 family 的代表 kernel
- 为什么这些 kernel 可以代表这个 family

#### 区块 4：边界条件

- 哪些 kernel 不应被纳入
- 为什么不能纳入
- 哪些情况会让该 family 定义失效

#### 区块 5：不确定性

- 当前还不稳的部分
- 哪些边界依然需要后续验证
- 是否存在可能拆成更细子类的风险

#### 区块 6：后续意义

- 该 family 是否适合共享后续 simulator 验证主线
- 此处只做定性表述，不展开具体处方

### 5.3 两者关系

本设计明确区分：

- 分析卡是单 kernel 证据载体
- family 卡片是多 kernel 解释载体

分析卡帮助我们从案例中长出 family；
family 卡片帮助我们把这些结果稳定地用于论文表达和后续验证组织。

---

## 六、不确定性、Mixed Kernel 与 Outlier 规则

### 不确定性的总原则

第一版采用如下原则：

**不确定性显式保留，不强行消解。**

因此：

- 暂定判断可以写成“暂定”
- family 边界可以写“边界不稳”
- mixed 状态可以合法存在
- outlier 可以先宽松定义，再后续收紧

### Mixed / ambiguous kernel

当某个 kernel 同时表现出两个强信号时，第一版不强行归入已有 family。

处理规则如下：

1. 先标为 `mixed / ambiguous`
2. 记录两个强信号分别是什么
3. 说明它为什么无法稳定并入已有 family
4. 后续再判断它是：
   - 某个现有 family 的边界样本
   - 一个潜在新 family
   - 或一个需要单独看待的 outlier

### 第一版 outlier 定义

第一版先宽松地把：

**不能稳定并入已有 family 的 kernel**

视为 outlier。

这里的关键点是：

- 第一版 outlier 不是最终定案
- 它只是表明：当前 family 体系还不能稳定解释它
- 后续可以再判断它是否真会影响验证主线，从而决定是否保留为核心 outlier

### 后续收紧方向

后续可以把 outlier 收紧为：

**既不能稳定并入已有 family，又会显著影响后续验证组织的 kernel。**

但这个收紧动作不在第一版强制完成。

### 这一部分的核心原则

**第一版优先保留结构性不确定性，而不是过早追求形式上的完整分组。**

---

## 七、在 `mini-transformer` 上的第一应用场景

`mini-transformer` 在本设计中的角色是：

**family 判据框架的首个试运行场景，而不是其普适性的唯一证明。**

### 7.1 应用目标

在 `mini-transformer` 上使用这套框架，目标不是立刻证明规则已经完美，而是验证：

1. 这套框架能否稳定组织代表 kernel
2. 它能否产出中等粒度 family
3. 它能否显式保留 mixed / outlier
4. 它能否为后续 simulator 验证组织提供定性帮助

因此，`mini-transformer` 的角色是：

**第一应用样例 + 第一轮 family 判据提炼来源。**

### 7.2 第一版应用对象

第一版不需要覆盖所有 launch，而是优先从已反复讨论过、且在 E0-E5 中有明确观察基础的代表
kernel 入手：

- `gemm_tiled`
- `attention_score`
- `residual_add`
- `softmax_kernel`
- `context_mul`
- `layernorm_kernel`

### 7.3 第一版应用流程

第一版在 `mini-transformer` 上的流程如下：

1. 为每个代表 kernel 填写分析卡
2. 先按执行模式做粗分
3. 再根据资源主导特征判断 family 边界
4. 对不能稳定归类的 kernel 保留 mixed / outlier 标记
5. 汇总生成 family 解释卡片
6. 最后检查这些 family 是否已经能自然组织出少数几条后续验证主线

最后一步只做定性检查，不展开具体 simulator 参数处方。

### 7.4 第一版成功标准

在 `mini-transformer` 场景下，第一版成功不要求“完全自动”或“完全正确”，而要求：

- 能形成一组中等粒度 family
- family 边界有可解释性
- mixed / outlier 被显式保留
- family 卡片能自然支撑后续验证组织的叙事

也就是说，第一版成功标准是：

**框架能工作，并且能生成有研究价值的结构。**

### 7.5 当前阶段不做的事

在 `mini-transformer` 应用场景中，第一版暂时不要求：

- 跨 workload 立即复现
- 定量证明 family 能减少多少验证成本
- 自动生成最终阈值
- 直接推出 simulator 参数处方

这些属于下一阶段。

---

## 八、当前设计的小结

本设计的核心可以压缩为一句话：

**先定义一套以执行模式粗分和资源主导边界为核心的 family 判据框架，再以
`mini-transformer` 作为第一应用场景，验证它能否生成中等粒度、可解释、可服务后续验证组织的
family 结构。**

它同时落实了当前已确认的设计前提：

1. `mini-transformer` 单点原型优先
2. 先把 idea 讲扎实，再搭系统
3. `batch` 定义上是解释层，落地上指导验证分流
4. 执行模式用于粗分，资源主导特征用于决定 family 边界
5. 输入采用“指标为主、语义约束、经验隔离”
6. 框架内部用层级结构，输出用 family 解释卡片
7. 第一版显式保留不确定性
