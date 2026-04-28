# Microbench Generation 相关工作阅读笔记

日期：2026-04-28

## 1. 目的

这份文档整理 microbench / synthetic benchmark generation 相关工作，重点回答三个问题：

1. 别人如何生成 microbench 或 synthetic benchmark；
2. 生成后的 microbench 被用来做什么；
3. 这些工作和我们计划中的 compression-guided microbench generation 有什么区别。

这里的 microbench 采用宽口径定义：只要它是一个小型、可控、用于复现或探测某类行为的 surrogate workload，都纳入讨论。

---

## 2. 总体观察

已有工作说明，microbench generation 本身不是空白方向。常见路线包括：

```text
真实 workload
  -> profile / static feature / control-flow representation
  -> synthetic code / miniature benchmark / generated input
  -> simulation / validation / modeling / training
```

这些工作给我们的启发是：

- microbench 不需要与真实 workload 语义等价；
- 关键是要在目标行为维度上相似；
- 生成过程通常需要明确 target features；
- 生成结果必须通过 profile / simulation / model accuracy 等指标验证。

因此，我们的工作不能只说“用 AI 生成 microbench”。更合理的差异点是：

> 使用 trace compression 暴露出的 execution-structure signature 作为 microbench generation 的 target / reward。

---

## 3. 主要工作

### 3.1 手写 GPU microbench：硬件机制探测

**代表工作：Demystifying GPU Microarchitecture through Microbenchmarking**

这类工作通常不从真实 workload 自动生成 microbench，而是手工设计小 kernel 来探测硬件机制，例如：

- memory latency / bandwidth；
- cache / coalescing；
- shared-memory bank conflict；
- warp divergence；
- atomic contention；
- instruction throughput。

使用方式：

- 用小而可控的 kernel 隔离某个硬件机制；
- 反推 GPU 微架构细节；
- 验证 simulator 对某个机制的建模是否合理。

对我们的启发：

- microbench 的价值首先来自可控性；
- 但手写 microbench 很难覆盖真实 workload 的复杂行为组合；
- 我们的方向可以看作把手写 microbench 的可控性和 target-driven synthesis 结合起来。

链接：

- 项目页面 / PDF 入口：https://www.stuffedcow.net/research/cudabmk

### 3.2 Performance Cloning：从 profile 生成 synthetic clone

**代表工作：Performance Cloning: A Technique for Disseminating Proprietary Applications as Benchmarks**

生成流程：

```text
real workload
  -> workload profiler
  -> microarchitecture-independent attributes
  -> workload synthesizer
  -> synthetic benchmark clone
```

提取的特征包括：

- instruction mix；
- basic block size；
- branch behavior；
- control-flow transition probability；
- memory locality / stride；
- dependency distance。

使用方式：

- 生成不暴露原始程序语义的 synthetic benchmark；
- 用于 architecture design、what-if analysis 和 workload sharing；
- 作为 proprietary workload 的可公开 proxy。

对我们的启发：

- 它证明“语义不同但行为相似”的 benchmark 是合理目标；
- 它也说明生成需要明确 behavior attributes；
- 我们的差异是把 target attributes 从 profiler / statistical features 扩展到 compression-derived trace structure。

PDF：

- https://iiswc.org/iiswc2006/IISWC2006P4.3.pdf

### 3.3 Miniature benchmark synthesis：合成更短的代表性程序

**代表工作：The Case for Automatic Synthesis of Miniature Benchmarks**

目标：

```text
long-running application
  -> representative miniature benchmark
  -> faster simulation / validation
```

使用方式：

- 生成比原程序短很多的 benchmark；
- 保留关键 performance characteristics；
- 降低 architecture simulation 成本。

对我们的启发：

- microbench / miniature benchmark 可以服务 simulation acceleration；
- “更小、更快、行为相似”是合理的研究目标；
- 但该路线主要依赖 general workload characteristics，不是 trace compression signature。

PDF：

- https://www.lca.ece.utexas.edu/pubs/bell-wmbs05.pdf

### 3.4 BenchMaker / synthetic benchmark knob space

**代表工作：The Return of Synthetic Benchmarks**

核心思想：

- synthetic benchmark 应该有可调 knob；
- 通过 knob 控制 instruction mix、memory behavior、branch behavior 等；
- 用来探索 workload behavior space。

使用方式：

- design-space exploration；
- 架构敏感性分析；
- 生成一组覆盖不同行为区域的 synthetic workloads。

对我们的启发：

- microbench generation 不应该只输出一个固定 kernel；
- 更好的形式是生成 code + parameters；
- compression signature 可以反过来定义这些 knobs 的目标方向。

PDF：

- https://lca.ece.utexas.edu/pubs/ajay-spec-workshop-08.pdf

### 3.5 GPGPU-MiniBench：GPU miniature workload generation

**代表工作：GPGPU-MiniBench: Accelerating GPGPU Micro-Architecture Simulation**

生成流程：

```text
GPGPU workload
  -> profile inherent execution behavior
  -> Divergence Flow Statistics Graph
  -> synthetic miniature GPGPU kernel
  -> faster architectural simulation
```

核心表示：

- Divergence Flow Statistics Graph；
- loops / branches / control-flow statistics；
- GPU kernel dynamic control behavior。

使用方式：

- 生成短小的 GPGPU miniature workload；
- 加速 GPU micro-architecture simulation；
- 保持关键性能趋势。

对我们的启发：

- 这是和我们最接近的 GPU 方向工作之一；
- 它证明 GPU miniature workload generation 是可行的；
- 它主要围绕 control-flow / divergence representation；
- 我们可以把 compression-derived signature 作为更广义的 trace-structure representation，覆盖 PC regularity、warp path、cross-TB memory regularity 和 fallback。

PDF：

- https://nilanjan.github.io/resources/GPGPU_MiniBench_NG_2015.pdf

### 3.6 MINIME-GPU：自动 GPU benchmark synthesis

**代表工作：MINIME-GPU: Multicore Benchmark Synthesizer for GPUs**

目标：

- 从真实 GPU 应用提取特征；
- 自动生成 synthetic GPU benchmark；
- 保留原 workload 的关键执行特征。

使用方式：

- GPU benchmark synthesis；
- 性能建模；
- architecture evaluation。

对我们的启发：

- GPU synthetic benchmark synthesis 已经存在；
- 因此我们的新意不能只是“GPU microbench generation”；
- 我们需要突出 target signal 的不同：compression-derived trace behavior signature。

论文链接：

- https://dl.acm.org/doi/pdf/10.1145/2818693

说明：这是 ACM publisher PDF，可能需要机构访问。

### 3.7 CLgen：用深度学习生成 OpenCL benchmark

**代表工作：Synthesizing Benchmarks for Predictive Modeling**

生成流程：

```text
OpenCL code corpus
  -> neural language model
  -> generated OpenCL kernels
  -> compile / execute / filter
  -> predictive model training data
```

使用方式：

- 扩展 benchmark feature space；
- 增加 compiler / performance model 的训练样本；
- 生成大量可编译 OpenCL kernels。

对我们的启发：

- AI / ML 生成 benchmark 已经存在；
- 生成结果需要编译和执行过滤；
- 它主要面向 code corpus / source-level generation，不直接面向 target workload trace。

PDF：

- https://www.pure.ed.ac.uk/ws/portalfiles/portal/29479104/2017_cgo_1.pdf

### 3.8 BenchPress：面向 feature space 的 active benchmark generation

**代表工作：BenchPress: A Deep Active Benchmark Generator**

生成流程：

```text
desired feature-space region
  -> deep benchmark generator
  -> OpenCL benchmark
  -> model evaluation / active learning
```

使用方式：

- 为 compiler / performance prediction 模型生成训练数据；
- 主动探索模型不确定区域；
- 提高 predictive model robustness。

对我们的启发：

- directed benchmark generation 是已有路线；
- target feature 对生成质量非常关键；
- 我们可以把 compression signature 定义为新的 target feature space。

PDF：

- https://chriscummins.cc/pub/2022-benchpress.pdf

### 3.9 BenchDirect：定向语言模型 benchmark generation

**代表工作：BenchDirect: A Directed Language Model for Compiler Benchmarks**

目标：

- 根据目标 feature 定向生成 compiler benchmark；
- 不是无目标采样，而是向指定 feature region 靠近。

使用方式：

- 生成满足目标特征的 OpenCL benchmarks；
- 支持 compiler heuristic / predictive model 训练；
- 减少随机生成带来的无效样本。

对我们的启发：

- 它强化了“生成必须有 target features”这一点；
- 我们的 trace compression signature 可以成为类似 target feature；
- 区别在于 BenchDirect 的 feature 更偏 source/compiler，而我们偏 dynamic trace structure。

PDF：

- https://www.foivos.co.uk/_files/ugd/ad4c78_e4f515e40f9e49f294cfb3e983cbb93b.pdf

### 3.10 Datamime：生成 representative dataset

**代表工作：Datamime: Generating Representative Benchmarks by Automatically Synthesizing Datasets**

生成流程：

```text
production workload behavior
  -> public application
  -> generated dataset
  -> benchmark behavior closer to production
```

它不直接生成 code，而是生成 input dataset，让已有应用表现得更像 target workload。

使用方式：

- 生成可公开、可复现的 representative benchmark；
- 避免直接公开 production workload；
- 让 benchmark 更贴近真实部署行为。

对我们的启发：

- 生成对象不一定只有 kernel code，也可以是 input / parameters；
- 对我们的 microbench 方向来说，可以考虑生成 code + parameters + data pattern；
- compression signature 可以评估最终行为，而不是只约束源码。

PDF：

- https://people.csail.mit.edu/hrlee/papers/micro22_datamime.pdf

### 3.11 DwarfCode：从 trace 生成压缩的代表性 benchmark

**代表工作：DwarfCode**

目标：

- 从应用 trace 中抽取重复 computation / communication pattern；
- 合并和压缩重复结构；
- 生成更短的 dwarf code benchmark。

使用方式：

- HPC / MPI workload 的 portable benchmark generation；
- 用短代码复现原程序关键行为；
- 支持性能预测和系统评估。

对我们的启发：

- 这是最接近“trace structure / compression -> benchmark generation”的非 GPU 工作；
- 它说明压缩 trace 中的重复结构可以直接服务 benchmark synthesis；
- 我们的差异是面向 GPU trace，并关注 warp / threadblock / memory address regularity。

论文链接：

- https://doi.org/10.1109/TC.2015.2417526

说明：这是 DOI 页面，是否能直接下载 PDF 取决于访问权限。

---

## 4. Microbench 的使用方式总结

已有工作中，microbench / synthetic benchmark 主要有五类用途：

1. **硬件机制理解**
   - 用小 kernel 探测具体硬件机制；
   - 典型例子是 GPU microarchitecture microbenchmarking。

2. **Simulator validation**
   - 用可控 microbench 验证 simulator 是否正确建模某类机制；
   - 比真实 workload 更容易定位问题。

3. **Architecture exploration**
   - 用短 benchmark 快速跑大量 architecture / parameter design points；
   - miniature benchmark synthesis 和 GPGPU-MiniBench 属于这一类。

4. **Proprietary workload proxy**
   - 不公开真实 workload；
   - 公开行为相似的 synthetic clone。

5. **ML / compiler training data augmentation**
   - 生成更多 benchmark 来覆盖 feature space；
   - 用于 predictive modeling、compiler heuristics 或 active learning。

---

## 5. 对我们工作的定位

从这些论文看，已有工作已经覆盖：

```text
profile / static / control-flow features
  -> benchmark generation
```

我们的工作应避免声称：

- 第一个自动生成 microbench；
- 第一个用 AI 生成 benchmark；
- 第一个用 synthetic benchmark 做 GPU simulation。

更稳的定位是：

```text
compressed GPU trace structure
  -> behavior signature
  -> microbench generation target / reward
```

也就是说，我们的重点不是“生成”本身，而是“生成目标的定义”。

---

## 6. 为什么 trace compression 之后再生成

和直接生成相比，compression-guided generation 多了三件事：

1. **明确目标**
   - 直接生成时，“像 target”很模糊；
   - compression signature 把“像”拆成 PC regularity、warp-path regularity、cross-TB memory regularity、address irregularity 等可解释目标。

2. **可计算反馈**
   - 生成后可以重新 trace / compress candidate；
   - 用 target signature 和 candidate signature 的 distance 判断是否更接近。

3. **可诊断修正**
   - 如果 candidate 不像，可以知道是 PC structure 不像、warp divergence 不像，还是 memory regularity 不像；
   - 这比只看 runtime 或 instruction count 更适合作为 agent feedback。

因此，我们不是“先压缩文件再喂给 AI”，而是：

> 用 trace compression 把 target kernel 的执行结构转化为 AI 可理解、可比较、可迭代优化的 behavior signature。

---

## 7. 建议优先阅读顺序

如果只读最关键的几篇，建议顺序如下：

1. **Performance Cloning**
2. **The Case for Automatic Synthesis of Miniature Benchmarks**
3. **GPGPU-MiniBench**
4. **CLgen**
5. **BenchPress**
6. **BenchDirect**
7. **Datamime**
8. **DwarfCode**

其中前三篇帮助理解 microbench / synthetic benchmark 的基本思想；CLgen、BenchPress、BenchDirect 说明 AI / directed generation 已经发展到哪里；Datamime 和 DwarfCode 帮助扩展“生成对象不一定只是代码”的视角。
