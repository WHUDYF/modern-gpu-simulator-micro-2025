# 导师汇报 PPT 大纲：从 Workload 到 Simulator 的结构化 Family 接口

日期：2026-04-19

## 汇报主线

这次汇报以**方法论文**为主线，顺序固定为：

1. 现有流程缺什么
2. 两个代表 related work 做了什么
3. 我们和它们的区别
4. 我们方法的价值与可行性
5. Transformer 主链上当前已经做了什么
6. 下一步怎么展开

这样安排的目的，是先证明问题真实存在，再证明我们的方法不是异想天开。

## 页面结构

### 第 1 页：标题页

- 标题：`从 Workload 到 Simulator 的结构化 Family 接口`
- 副标题：`导师汇报初版 / 方法论文主线 / mini_transformer_v4 原型`
- 作用：
  - 明确这不是采样优化汇报
  - 明确这是方法论汇报

### 第 2 页：问题定义

- 标题：`问题：现有流程仍缺少结构化接口`
- 核心点：
  - kernel 很多、phase 很杂、解释对象不显式
  - simulator 可以跑 workload，但 workload 很难被自然组织成 reasoning units
  - 当前缺的不是更多样本，而是 workload 到 simulator 的中间结构

### 第 3 页：相关工作一

- 标题：`相关工作一：PKA 做了什么`
- 放图：
  - PKA 论文方法图截图
- 讲法：
  - PKA 通过 representative kernel selection 和 projection 减少模拟量
  - 它压缩的是 `simulation samples`
- 本页结论：
  - `PKA 的目标是 sampled simulation，而不是 workload 结构化接口。`

### 第 4 页：相关工作二

- 标题：`相关工作二：Sieve 做了什么`
- 放图：
  - Sieve 论文流程图截图
- 讲法：
  - Sieve 通过 stratification 和 weighted prediction 提高 sampled simulation 稳定性
  - 它改进的是 sampling 质量，而不是结构接口
- 本页结论：
  - `Sieve 的目标仍然是 sampled simulation。`

### 第 5 页：我们的区别

- 标题：`我们的区别：不是采样技巧，而是结构化接口`
- 核心点：
  - PKA / Sieve 关注：`模哪些样本`
  - 我们关注：`按什么结构去解释和组织 workload`
- 建议形式：
  - 三行对比表
- 本页结论：
  - `已有工作考虑硬件，但普遍缺少一层 workload -> simulator reasoning 的结构层。`

### 第 6 页：方法价值与可行性

- 标题：`方法价值与可行性`
- 左侧讲价值：
  - 显式化 workload 映射过程
  - 把分析对象提升到 primitive / family 层
  - 为 future tuning 提供结构基础
- 右侧讲可行性：
  - 不按算子名分
  - 不按上层模块分
  - boundary case 已经逼出了 primitive 判据

### 第 7 页：Transformer 主链

- 标题：`Transformer 主计算路线：我们现在具体做了什么`
- 放两层内容：
  - 抽象方法链：`workload trace -> execution primitive -> family -> simulator reasoning lane`
  - 具体 transformer 路线：
    `QKV / projection -> attention_score -> softmax -> context_mul -> residual / norm / FFN`
- 本页结论：
  - `Transformer 主链已经足以承载第一版原型。`

### 第 8 页：当前原型

- 标题：`当前原型：Boundary Case 如何逼出判据`
- 案例：
  - `gemm_tiled vs attention_score`
  - `softmax_kernel vs context_mul`
- 讲法：
  - 前者说明不能只按算子名拆
  - 后者说明不能只按 attention 子模块并
- 本页结论：
  - `family 的核心不是语义标签，而是 execution primitive。`

### 第 9 页：下一步计划

- 标题：`下一步：先把方法做硬，再向 Simulator 对接`
- 三步：
  - 做硬 transformer 主链 primitive 判据
  - 写出 family selection / boundary protocol
  - 再向 simulator lane / tuning lane 对接

### 第 10 页：总结

- 标题：`总结`
- 三句总结：
  - 当前空缺真实存在
  - 我们的方法方向明确
  - Transformer 主链提供了第一版可落地原型

## 当前汇报最重要的一句话

`我们要解决的不是“如何少模拟几个 kernel”，而是“如何把 workload 结构化成 simulator 可以承接的硬件分析对象”。`
