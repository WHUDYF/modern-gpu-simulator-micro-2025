# GPU 模拟器相关工作与端到端缺口总结

日期：2026-04-14

## 目的

这份文档用于整理当前公开 GPU 模拟器相关工作的主流范式，并回答一个具体问题：

现有 GPU 模拟器通常如何被用于支持架构更新或设计决策，以及它们在缺少端到端 workload 闭环时，为什么容易导向“局部有效但外推不稳”的结论。

本文档不是某一个模拟器的使用说明，而是为后续论文写作和研究定位服务的 related work / motivation 备忘录。

---

## 一、当前主流 GPU 模拟器的共性

截至 2026 年 4 月，公开且仍有较强影响力的 GPU 模拟器路线大致包括：

- Accel-Sim / GPGPU-Sim：面向 NVIDIA GPU，强调 trace-driven + validated modeling
- gem5 GPU / GPUFS：面向 AMD GPU，强调 full-system GPU simulation
- MGPUSim：面向 AMD GCN / 多 GPU 系统，强调灵活性与精度
- NaviSim：面向 AMD RDNA，强调新架构建模与重新校准

这些工作的共同点不是“直接决定硬件如何更新”，而是：

1. 先构造一个尽可能可信的 baseline simulator
2. 再围绕少量架构点做 design space exploration
3. 最后将 case study 的结果解释为某类架构方向是否值得继续推进

因此，公开 GPU 模拟器的现实角色更接近：

**为架构设计提供方向筛选、优先级排序和机制解释**

而不是：

**单独作为最终硬件设计决策器**

---

## 二、现有工作通常如何支持架构更新

### 1. Baseline 校准先行

几乎所有可信的 GPU 模拟器工作，第一步都不是直接改架构，而是先做 baseline 校准。

常见方式包括：

- 使用 microbenchmark 提取时延、带宽、cache 行为
- 使用硬件计数器做 simulator vs hardware correlation
- 通过 tuner / correlator 自动或半自动调整参数
- 在少量代表 workload 上验证关键性能指标

典型例子是 Accel-Sim，它明确把 `Correlator + Tuner` 作为现代 GPU 建模流程的一部分，目标是把 simulator 拉到可以用于研究的可信起点。

**含义：** 当前社区已经默认接受一个前提：如果 baseline 不可靠，后续设计探索没有意义。

### 2. 在校准后的 baseline 上做局部 case study

完成 baseline 之后，主流工作通常会围绕少数架构点开展探索，例如：

- 内存调度策略是否值得更新
- cache policy 是否仍有收益
- 特定带宽 / 时序参数对某类 kernel 是否敏感
- 多 GPU 数据迁移、互连或 locality 机制是否有效

这些 case study 的特点是：

- 研究问题通常由研究者预先设定
- 参数扰动通常是人工决定的
- simulator 的主要作用是比较不同设计点的相对收益

**含义：** 现有工作善于回答“这个设计点值不值得”，但较少回答“复杂 workload 中到底哪些架构点最值得先被验证”。

### 3. 用 case study 支撑设计含义

模拟器论文通常不会声称“我们直接决定了硬件下一代怎么做”，而是做更稳妥的表述：

- 更准确的模型会改变对某类机制价值的判断
- 某类 workload 暴露了某个现有架构瓶颈
- 某类优化在给定建模假设下有稳定收益

这是一种合理且成熟的研究范式，但它默认了一个重要前提：

**研究者已经知道该调哪些参数、该观察哪些机制、该优先分析哪些 kernel。**

而这恰恰是复杂 AI workload 场景里最难的一步。

---

## 三、现有范式的关键缺口

### 缺口 1：Baseline 不准确会直接改变设计判断

Accel-Sim 2020 的一个核心贡献，不只是“模拟更准确了”，而是展示了：

**baseline 的准确性会改变研究者对架构优化价值的判断。**

论文中的案例显示，更现代的建模会重新评估某些内存系统设计点的收益；旧版 simulator 可能低估真正重要的机制，也可能高估已经不再关键的瓶颈。

这说明一个很强的结论：

**模拟器失真不只是数值偏差问题，而会改变研究者“该优化什么”的判断。**

因此，在缺少严密 baseline 校准时，architecture exploration 的结论很容易变成方向性错误，而不是简单地“不够精确”。

### 缺口 2：只调整公开参数并不足以支持可信研究

NaviSim 对这个问题说得更直接。

它明确指出，仅仅把公开参数填入 simulator，不足以建模一代新架构；如果没有重新建模和重新校准，甚至可能导向错误结论。

这意味着：

- “公开参数 + 原有 simulator” 并不自动等于可信 baseline
- 架构研究的难点不只在参数值，而在模型结构与 workload 表达是否匹配

换句话说，**参数抄对了，不代表研究问题就被建模对了。**

### 缺口 3：现代 GPU workload 的软件栈复杂度使得端到端表达变难

gem5 近年来对 GPUFS / full-system GPU simulation 的强调，本质上是在回应这个问题：

现代 GPU 应用越来越依赖复杂的软件栈、库函数和运行时环境，单纯的 syscall emulation 或简化执行模式，越来越难完整表达真实 workload。

这会带来一个直接后果：

**simulator 可能在一个过度简化的执行包络里给出结论，但这些结论未必能稳定外推到真实应用。**

因此，现有工作越来越重视 full-system，不是因为 full-system 更“优雅”，而是因为：

**缺少端到端 workload 表达时，simulation 结论的外推可信度会显著下降。**

### 缺口 4：即使是成熟的 trace-driven 模拟，也天然有边界

Accel-Sim 本身已经比传统 simulator 更接近现代 workload，但它也明确承认 trace-driven 有边界，例如：

- 依赖寄存器 / 数据值的机制难以仅靠 trace 研究
- 某些全局同步和运行时语义无法在 trace-only 中完整表达

这说明：

**即便已经有现代 tracer 和较高精度的前端，仍然需要一种方法帮助研究者区分：哪些信号可以进入 simulator 验证，哪些属于当前模型盲区。**

这不是 Accel-Sim 的缺点，而是整个 simulator literature 面临的共同边界。

---

## 四、为什么“缺少端到端闭环”会削弱可信度

这里需要一个更严格的表述。

我们不宜简单地说：

**现有 GPU 模拟器的结论都不可信。**

更准确的说法应该是：

**在缺少端到端闭环时，simulator 得出的架构结论通常只能被视为局部有效，其外推可信度显著下降。**

原因在于，缺少端到端闭环意味着下面至少有一个环节是断开的：

- workload 入口是否真实
- software stack 是否被合理表达
- kernel 之间的结构关系是否被系统识别
- 参数候选是否来自 workload 信号，而不是人工猜测
- simulator 的扰动结果是否回到原 workload 语境中解释

一旦这些链条断开，研究者仍然可以得到一个“看起来合理”的结论，但很可能出现以下问题：

1. 结论只对某个局部 microbenchmark 成立
2. 结论只对某个 kernel family 成立，但被错误外推到整个应用
3. 结论建立在错误的热点识别上
4. 结论依赖 simulator 的现有偏差，而非 workload 的真实需求

因此，端到端闭环的重要性不在于“看起来完整”，而在于：

**它提供了一种机制，把 architecture exploration 从“人工挑问题”转变为“由 workload 结构化地产生问题”。**

---

## 五、这对我们工作的定位意味着什么

我们的目标不是重复现有 simulator 工作已经做得很好的事情，比如：

- 再造一个新的 GPU simulator
- 单独提升某个 timing model 的精度
- 仅仅提供一个新的参数 tuner

我们真正补的缺口是：

**如何从复杂 workload 的执行行为中，系统地产生可验证的架构假设。**

这也是我们工作与现有 simulator literature 最核心的差异：

- 现有工作大多从 simulator 出发，再做人为设定的 case study
- 我们希望从 workload 出发，经由结构化中间层，再进入 simulator 验证

这条链路可以概括为：

`workload execution -> structured phases / families -> candidate architectural factors -> simulator perturbation -> validated / unsupported / model-blind conclusions`

如果这条链路成立，那么 simulator 不再只是“比较几个设计点的工具”，而成为端到端方法中的**验证后端**。

---

## 六、squash 与 batch 在这个框架里的角色

如果把整个故事讲成“从算法到模拟器的端到端视线”，那么 `squash` 和 `batch` 的价值都不能被简单看作辅助分析。

### `squash` 的角色

`squash` 的核心作用不是“把 kernel 分成几段”，而是：

- 将长执行流压缩为若干行为稳定的 phase
- 让复杂 workload 的时间结构变得可表示、可复用、可解释
- 为后续 trace 代表选择和 phase-level 验证提供基础

换句话说，`squash` 解决的是：

**算法执行流太长、太杂，无法直接映射到 simulator 假设**

### `batch` 的角色

`batch` 的核心作用不是“做了聚类”，而是：

- 识别共享行为的 kernel family
- 分离必须单独处理的 outlier
- 为后续 simulator 校准提供“哪些能共用解释、哪些必须单独建模”的结构

换句话说，`batch` 解决的是：

**不同 kernel 的行为差异无法被系统组织，导致后续验证成本过高、解释混乱**

### 两者共同的价值

如果没有 `squash` 和 `batch`，从 workload 到 simulator 的链路会出现两个直接问题：

1. 无法把算法执行流稳定压缩成少量可分析对象
2. 无法把 kernel 行为组织成“共性 vs 例外”的结构

因此，在我们的端到端故事里：

**`squash` 和 `batch` 不是最终贡献本身，但它们是让最终贡献成立的必要结构层。**

---

## 七、适合论文写作的 related work / motivation 论证框架

下面给出一套可直接复用的论证结构。

### 论点 A：现有 GPU 模拟器已经证明 simulation 对架构探索有效

可以先承认现有工作的价值：

- 高保真 simulator 已成为 architecture exploration 的重要工具
- 现代工作越来越重视 validated baseline、full-system 表达、以及更接近真实硬件的软件入口

这个开场是必要的，因为它表明我们不是在否定 simulator literature，而是在其基础上进一步推进。

### 论点 B：现有工作主要解决“如何更准确地模拟”，较少解决“如何从 workload 中系统地产生架构假设”

这是 related work 的分水岭。

我们需要指出：

- baseline 校准解决的是 simulator 本身的可信性
- case study 解决的是若干人工设定设计点的比较
- 但复杂 workload 到参数候选之间，仍缺少系统桥梁

因此，已有工作强于：

- simulator accuracy
- design-point comparison

弱于：

- workload-driven hypothesis generation
- end-to-end architectural factor discovery

### 论点 C：已有文献已经显示，baseline 失真或闭环缺失会改变甚至削弱架构结论

这里可以用三类证据串起来：

- Accel-Sim：baseline 更准确会改变对设计点价值的判断
- NaviSim：只调公开参数而不重新建模会导向错误结论
- gem5 GPUFS：现代 workload 需要更完整的软件栈表达，否则外推性下降

这个组合可以支撑一句很强但仍然严谨的话：

**已有工作已经表明，可信的 architecture exploration 不仅依赖 simulator 精度，还依赖 workload、模型与验证之间是否形成闭环。**

### 论点 D：我们的贡献是补上“workload 到 simulator 验证”之间的结构化桥梁

这就是我们的主定位：

我们不是替代已有 simulator，而是在 workload analysis 与 simulator validation 之间加入一个结构化中间层，通过：

- `squash` 抽取 phase
- `batch` 抽取 family / outlier
- `delta` 抽取候选架构因素
- Stage C 验证区分可模拟点与模型盲区

最终形成一条从 workload 到 simulator 的端到端视线。

---

## 八、最简收束版本

如果后续需要把整套论证压成最短形式，可以直接写成下面三句话：

1. 现有 GPU 模拟器已经证明了 simulation 对 architecture exploration 的有效性，但主流范式仍以 baseline 校准和局部 case study 为主。
2. 现有文献同时表明，baseline 失真、仅依赖公开参数建模、以及对现代 workload/software stack 的不完整表达，都会削弱架构结论的外推可信度。
3. 因此，我们的工作不是提出新的 simulator，而是提出一条从 workload 行为中系统地产生可验证架构假设，并最终通过 simulator 完成闭环验证的端到端流程。

---

## 参考资料

- Accel-Sim 官网：<https://accel-sim.github.io/>
- Khairy et al., "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling", ISCA 2020  
  <https://people.ece.ubc.ca/aamodt/publications/papers/accelsim.isca2020.pdf>
- Sun et al., "MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization", ISCA 2019  
  <https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf>
- MGPUSim GitHub：<https://github.com/sarchlab/mgpusim>
- gem5 GPUFS 文档：<https://www.gem5.org/documentation/general_docs/gpu_models/gpufs>
- gem5 博客："Modeling Modern GPU Applications in gem5"  
  <https://www.gem5.org/2020/05/27/modern-gpu-applications.html>
- gem5 博客："Moving to full system simulation of GPU applications"  
  <https://www.gem5.org/2023/02/13/moving-to-full-system-gpu.html>
- Patel et al., "NaviSim: A Modeling and Simulation Framework for AMD GPUs", PACT 2022  
  <https://bu-icsg.github.io/publications/2022/navisim_pact_2022.pdf>
