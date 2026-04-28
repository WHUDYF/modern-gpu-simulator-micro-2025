# Trace Compression 与 Microbench Generation 相关工作

日期：2026-04-28

## 1. 目的

这份 memo 总结与我们当前学术线最相关的论文方向，重点回答两个问题：

1. microbench / synthetic benchmark 是否已有成熟生成路线；
2. trace compression 的中间结构和退化模式，是否已有工作把它当作行为信号使用。

我们的目标不是罗列所有 benchmark 论文，而是找出一条对当前研究最有用的脉络。

---

## 2. 我们自己的问题设定

当前学术线的核心想法是：

```text
target kernel
  -> trace compression structure
  -> behavior signature
  -> generated microbench
```

这里的关键点是，压缩不是单纯为了减小 trace，而是为了提取 execution regularity、divergence、cross-threadblock similarity 等结构信号。

这和现有 representative selection / workload characterization 方法的关系在于：

- representative selection 负责选出值得关注的 target；
- 这条学术线负责解释 target 的 trace-level 行为，并用它指导 microbench synthesis；
- compression-side feature 不需要进入 upstream selector 才能发挥作用。

---

## 3. 最相关的工作分组

### 3.1 自动生成 synthetic benchmark / microbench

这类工作最接近“从真实 workload 生成可控替身程序”。

- **Performance Cloning**
  - 通过提取关键 performance attributes，合成与真实程序性能特征相似的 synthetic benchmark。
  - 对我们最有启发的是：benchmark 不需要保留原始语义，只需要保留目标行为特征。
  - 参考：Joshi et al., IISWC 2006.

- **The Return of Synthetic Benchmarks**
  - 强调 synthetic benchmark 可以通过可调 knob 覆盖 feature space。
  - 说明 benchmark 生成不必局限于“复制源码”，而是可以面向行为空间设计。
  - 参考：BenchMaker / SPEC Workshop 2008 相关工作。

- **MINIME-GPU**
  - 面向 GPU 的自动 benchmark synthesis。
  - 从真实 GPU 应用提取特征，再生成 synthetic GPU benchmark。
  - 和我们最接近的点是：它证明 GPU benchmark synthesis 本身是可行研究方向。

- **Thread-level synthetic benchmarks for multicore systems**
  - 从真实应用提取 thread-level 特征并生成 synthetic benchmarks。
  - 对我们“warp / threadblock-level signature”有方法学参考意义。

- **Benanza**
  - 用于生成 GPU micro-benchmarks，服务于 layer/model latency lower-bound 估计。
  - 更偏 DL/GPU 系统，但说明 microbench 可以作为更可控的分析工具。

### 3.2 AI / ML 驱动的 benchmark generation

这类工作和“AI agent 生成 microbench”最接近。

- **CLgen**
  - 用生成模型合成 OpenCL kernel，用于扩展 benchmark 特征空间。
  - 关键启发：合成 benchmark 可以覆盖真实 benchmark 稀疏覆盖不到的行为区域。

- **BenchPress / BenchDirect**
  - 面向 feature space 的定向 benchmark 生成。
  - 和我们的区别是：它们的 target 是 source / compiler / static features；我们的 target 是 compression-derived trace behavior signature。

### 3.3 GPU representative sampling / kernel selection

这类工作最接近 representative target selection。

- **PKA**
  - 选 representative kernels，说明真实 workload 太大时需要先做 representative selection。
  - 这为 target selection 提供方法论背景，但不构成本学术线的前提。

- **Sieve**
  - 关注 invocation-level stratification，说明同名 kernel 也可能有明显异质性。
  - 对我们后续做 target / cluster / group 的分层很有帮助。

- **Photon**
  - 使用 warp / basic-block 级执行结构做在线采样判断。
  - 它证明 execution-path structure 可以成为有效信号，但它不是 microbench synthesis 工作。

### 3.4 Trace compression / behavior characterization

这类工作直接支撑“压缩结构不只是为了省空间”。

- **Real-time compression of instruction and data address traces**
  - 讨论 instruction/data address trace 的压缩和 workload characterization。
  - 支撑工程线，也说明 trace compression 本身可以是观察行为的窗口。

- **Analysis of branch behavior via data compression / entropy-based program behavior**
  - 说明程序压缩性、熵、regularity 可以作为行为表征。
  - 对我们把 compression failure 视为 behavior complexity signal 很有帮助。

---

## 4. 对我们最有价值的交集

把这些工作放在一起后，可以得到一个比较清楚的空白区：

```text
existing work:
  workload features -> synthetic benchmark
  representative kernels -> simulator sampling
  trace compression -> smaller traces / behavior hints

our direction:
  trace compression structure -> behavior signature -> microbench generation
```

也就是说，已有工作分别覆盖了三块：

1. benchmark synthesis 已经存在；
2. representative kernel selection 已经存在；
3. trace compression 也已经存在。

但目前缺少的是：

> 将 compressed trace 的中间结构和退化模式系统性地当作 microbench generation 的目标信号。

这正是我们学术线可能成立的地方。

---

## 5. 我们可以借鉴的设计原则

### 5.1 语义不等价没关系

Performance cloning 和 synthetic benchmark 工作都说明：microbench 不需要等价于 target 的原始语义，只需要在目标行为维度上相似。

### 5.2 特征空间要可解释

BenchDirect / PKA / Photon 都说明，特征空间必须能解释为什么某个 candidate 更像 target。

### 5.3 结构信号比单点指标更强

trace compression 给我们的不是单个 runtime 数字，而是结构信号：

- run-length 分布；
- warp diff；
- cross-TB delta；
- address override；
- fallback 模式。

这些比单纯 instruction count 或 wall time 更适合作为 microbench target。

### 5.4 退化模式本身也是信息

如果某个 trace 很难压缩，这本身就是关于 workload irregularity 的强信号。

---

## 6. 研究缺口

目前最值得强调的 gap 是：

1. benchmark synthesis 论文大多用 performance counters、static features、profiling features；
2. GPU representative sampling 论文大多停留在“选哪些 kernel”；
3. trace compression 论文大多停留在“压缩 trace 或提取 characterization signal”；
4. 但还没有一条主线把 trace compression 的结构性输出直接变成 microbench synthesis 的 reward / matching target。

如果这条线做成，贡献可以表述为：

> We use compression-derived execution structure as a behavioral target for microbenchmark synthesis.

---

## 7. 下一步建议

如果要继续推进这条线，建议先做一个小的 feasibility probe：

1. 选一组已知行为差异明显的 target kernels；
2. 从它们的 trace 或 compressed trace 中抽取初始 signature；
3. 用简单距离度量验证 signature 能否区分 kernel 类型；
4. 再看能否用这些 signature 排序 candidate microbench。

如果第一步就不能区分行为类型，这条线需要重新定义 signature；如果能区分，再进入 microbench generation。

---

## 8. 参考脉络

最值得优先阅读的方向：

- performance cloning / synthetic benchmark generation；
- GPU benchmark synthesis；
- representative kernel selection / sampled simulation；
- trace compression as behavior characterization。

这四类工作合在一起，基本构成了我们学术线的文献坐标系。

---

## 9. 阅读索引

下面把正文中提到的工作整理成便于直接阅读的 PDF / 论文链接。个别论文如果没有稳定公开 PDF，我保留了 publisher / DOI 链接并标注说明。

| 工作 | 直接链接 |
|---|---|
| The Case for Automatic Synthesis of Miniature Benchmarks | https://www.lca.ece.utexas.edu/pubs/bell-wmbs05.pdf |
| Performance Cloning: A Technique for Disseminating Proprietary Applications as Benchmarks | https://iiswc.org/iiswc2006/IISWC2006P4.3.pdf |
| The Return of Synthetic Benchmarks | https://lca.ece.utexas.edu/pubs/ajay-spec-workshop-08.pdf |
| MINIME-GPU: Multicore Benchmark Synthesizer for GPUs | https://dl.acm.org/doi/pdf/10.1145/2818693 |
| GPGPU-MiniBench: Accelerating GPGPU Micro-Architecture Simulation | https://nilanjan.github.io/resources/GPGPU_MiniBench_NG_2015.pdf |
| Synthesizing Benchmarks for Predictive Modeling / CLgen | https://www.pure.ed.ac.uk/ws/portalfiles/portal/29479104/2017_cgo_1.pdf |
| BenchPress: A Deep Active Benchmark Generator | https://chriscummins.cc/pub/2022-benchpress.pdf |
| BenchDirect: A Directed Language Model for Compiler Benchmarks | https://www.foivos.co.uk/_files/ugd/ad4c78_e4f515e40f9e49f294cfb3e983cbb93b.pdf |
| Datamime: Generating Representative Benchmarks by Automatically Synthesizing Datasets | https://people.csail.mit.edu/hrlee/papers/micro22_datamime.pdf |
| Principal Kernel Analysis | https://mkhairy.github.io/Docs/PKA.pdf |
| Sieve: Stratified GPU-Compute Workload Sampling | https://users.elis.ugent.be/~leeckhou/papers/ispass-2023.pdf |
| Photon: A Fine-grained Sampled Simulation Methodology for GPU Workloads | https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf |
| Real-time compression of instruction and data address traces | https://userweb.cs.txstate.edu/~mb92/papers/dcc07b.pdf |
| Analysis of Branch Prediction via Data Compression | https://tnm.engin.umich.edu/wp-content/uploads/sites/353/2017/12/1996.10.Analysis-of-Branch-Prediction-via-Data-Compression.pdf |
| Introducing Entropies for Representing Program Behavior | https://www.usenix.org/events/expcs07/papers/17-yokota.pdf |

如果你想把阅读优先级再压缩一下，我建议按这个顺序看：

1. `Performance Cloning`
2. `The Return of Synthetic Benchmarks`
3. `CLgen`
4. `BenchPress`
5. `BenchDirect`
6. `PKA`
7. `Sieve`
8. `Photon`
9. `Real-time compression of instruction and data address traces`
10. `Introducing Entropies for Representing Program Behavior`
