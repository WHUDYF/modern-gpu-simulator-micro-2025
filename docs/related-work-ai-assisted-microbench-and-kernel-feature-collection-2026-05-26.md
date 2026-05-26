# Related Work - AI-Assisted Microbench Anchors and Kernel Feature Collection

日期：2026-05-26

本文档保存与 `mechanism_microbench_anchor_dataset` 相关的论文线索，重点回答两个问题：

1. 是否已有工作用 LLM / AI 生成训练数据来改进模型？
2. 是否已有工作通过自动化或 AI 方法收集 kernel 相关特征，从而减少人工判断？

结论：

```text
LLM-generated training data 是成熟范式；
GPU kernel 领域已有自动 counter/feature selection、trace graph embedding、performance-tool reasoning 等相关工作；
但尚未看到完整等价于“LLM 生成机制 microbench + 自动验证 + 图网络机制归因”的公开标准方案。
```

因此我们的方案有实现价值，但必须把 LLM 生成、counter-confirmation 和 knob-validation 分层。

---

## 1. LLM 生成训练数据

### Self-Instruct

链接：https://arxiv.org/abs/2212.10560

价值：

```text
证明可以用模型自生成 instruction / input / output 数据，再过滤后训练模型。
```

对我们的启发：

```text
LLM 可以生成 mechanism microbench 候选，但必须经过 filtering / validation。
```

### Stanford Alpaca

链接：https://crfm.stanford.edu/2023/03/13/alpaca

价值：

```text
用 text-davinci-003 生成 52K instruction-following demonstrations 来训练 LLaMA。
```

对我们的启发：

```text
LLM 生成数据可以形成可训练 corpus，但生成来源、成本、过滤和数据卡必须记录。
```

### WizardLM / Evol-Instruct

链接：https://arxiv.org/abs/2304.12244

价值：

```text
通过自动演化 instruction 的复杂度，生成更难、更丰富的训练样本。
```

对我们的启发：

```text
可以让 LLM 从简单机制 microbench 演化到组合机制 microbench，例如 memory + sync、tensor + shared memory。
```

### Orca

链接：https://arxiv.org/abs/2306.02707

价值：

```text
用 GPT-4 生成 explanation traces，帮助小模型学习复杂推理过程。
```

对我们的启发：

```text
microbench 数据不应只有 label，还应保存 expected signature、counter evidence 和 validation trace。
```

### Distilling Step-by-Step

链接：https://arxiv.org/abs/2305.02301

价值：

```text
用 LLM rationale 作为额外监督，降低训练数据需求。
```

对我们的启发：

```text
LLM rationale 可以保存在 audit 中，但第一版不建议直接输入模型，以免学到 annotator style。
```

### Textbooks Are All You Need / Phi-1

链接：https://arxiv.org/abs/2306.11644

价值：

```text
用高质量 synthetic textbook / exercise 数据训练小型代码模型。
```

对我们的启发：

```text
高质量、机制清晰的 microbench anchor 可能比大量真实但混杂的 kernel 更适合作为 prototype supervision。
```

### Magicoder / OSS-Instruct

链接：https://arxiv.org/abs/2312.02120

价值：

```text
利用开源代码片段引导 LLM 生成代码 instruction 数据。
```

对我们的启发：

```text
microbench 生成不应凭空开始，可以由已有 CUDA benchmark、validated seed 和 registry target 引导。
```

---

## 2. 自动 kernel 特征收集与特征选择

### Metric Selection for GPU Kernel Classification

链接：https://dl.acm.org/doi/10.1145/3295690

价值：

```text
研究 GPU kernel 分类中的 performance counter / metric selection，目标是减少手工选择指标的负担。
```

对我们的启发：

```text
我们可以把 counter / metric selection 作为自动特征收集模块，而不是人工决定哪些 NCU metrics 最重要。
```

### Utilizing GPU Performance Counters to Characterize GPU Kernels via Machine Learning

链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC7302272/

价值：

```text
使用 GPU performance counters 和机器学习来表征 kernel，使 kernel characterization 不完全依赖人工分析。
```

对我们的启发：

```text
counter vectors 可以作为自动采集的 raw evidence，并通过 ML 得到 kernel category / similarity。
```

### Searching CUDA Code Autotuning Spaces with Hardware Performance Counters

链接：https://arxiv.org/abs/2301.13297

价值：

```text
使用硬件 performance counters 来加速 CUDA autotuning 搜索。
```

对我们的启发：

```text
counter 不是最终标签，但可以作为 search / validation planning 的自动反馈信号。
```

### GPUscout: Locating Data Movement-related Bottlenecks on GPUs

链接：https://www.ce.cit.tum.de/fileadmin/w00cgn/caps/vanecek/sv_gpuscout.pdf

价值：

```text
结合 SASS static analysis、warp stalls 和 kernel performance metrics 来定位 GPU 数据移动瓶颈。
```

对我们的启发：

```text
可以把 SASS / stall / performance metric 组合成自动 evidence encoder，减少人工 bottleneck 判断。
```

### GCL-Sampler

链接：https://arxiv.org/abs/2603.00551

价值：

```text
将 GPU kernel trace 构造成异构关系图，用 R-GCN 和 contrastive learning 学习 kernel embedding，用于 kernel similarity 和 sampled simulation。
```

对我们的启发：

```text
说明从 trace graph 自动学习 kernel 表征是可行的。我们的图网络可以把 microbench anchors 和 real workload anchors 接到同一个 similarity / mechanism graph 中。
```

### Modeling Utilization to Identify Shared-Memory Atomic Bottlenecks for GPU Parallelization

链接：https://arxiv.org/abs/2502.03754

价值：

```text
用测量和建模识别 shared-memory atomic bottleneck。
```

对我们的启发：

```text
对于细粒度硬件机制，自动检测通常需要 counter + model，而不是简单 instruction ratio。
```

---

## 3. LLM / AI 与 GPU kernel 优化工具

### Integrating Performance Tools in Model Reasoning for GPU Kernel Optimization

链接：https://arxiv.org/abs/2510.17158

价值：

```text
把 performance tools 接入模型推理过程，用工具反馈增强 GPU kernel 优化。
```

对我们的启发：

```text
LLM 不应闭眼判断机制；它应该调用 profiler / compiler / simulator 工具，并基于工具输出生成下一步候选。
```

### GPU Kernel Scientist

链接：https://openreview.net/forum?id=2Rwl7a4MWc

价值：

```text
LLM-driven iterative GPU kernel optimization 框架，强调自动生成、执行和反馈循环。
```

对我们的启发：

```text
支持把 LLM 作为 microbench generator / experiment planner，而不是最终机制 oracle。
```

---

## 4. 对我们方案最关键的结论

### 4.1 可行性

已有工作支持以下事实：

```text
LLM 能生成训练数据；
performance counters 能自动表征 GPU kernels；
feature selection 可以减少人工 metric 选择；
graph contrastive learning 可以从 trace graph 学 kernel embedding；
LLM 可以与 performance tools 形成闭环。
```

因此：

```text
LLM-assisted mechanism microbench anchor generation
  + automatic counter / trace feature collection
  + graph-based mechanism attribution
```

不是空想，而是把已有路线组合到 GPU mechanism attribution 场景。

### 4.2 缺口

尚未看到完全等价的公开工作：

```text
LLM 生成机制 microbench
  -> 自动 profiling
  -> counter-confirmed labels
  -> simulator knob validation
  -> graph network mechanism attribution
```

这个缺口可以成为我们方法的研究贡献，但也意味着必须把 claim boundary 写清楚。

### 4.3 风险

最主要风险：

```text
design intent 被误当真值
counter evidence 不足以唯一证明机制
microbench 与真实 workload 分布差异过大
LLM 生成样本风格单一
模型学习到模板而不是硬件机制
```

对应防线：

```text
design_intent_only < counter_confirmed < knob_validated
profile / validation before training promotion
split-by-workload and split-by-template
dataset card records generator, prompt, tool versions
graph reasoner keeps abstain / boundary state
```

---

## 5. 推荐引用组合

如果后续写论文，建议把相关工作分成三组：

1. **Synthetic data and LLM distillation**
   - Self-Instruct
   - Alpaca
   - WizardLM
   - Orca
   - Phi-1
   - Magicoder

2. **Automatic GPU kernel characterization**
   - Metric Selection for GPU Kernel Classification
   - Utilizing GPU Performance Counters to Characterize GPU Kernels via Machine Learning
   - Searching CUDA Code Autotuning Spaces with Hardware Performance Counters
   - GPUscout

3. **Graph / tool-driven kernel reasoning**
   - GCL-Sampler
   - Integrating Performance Tools in Model Reasoning for GPU Kernel Optimization
   - GPU Kernel Scientist

我们的定位：

```text
We use LLM-assisted generation to create mechanism-focused microbenchmark anchors,
then rely on automated profiling and simulator validation to promote them from design intent
to counter-confirmed or knob-validated supervision for graph-based mechanism attribution.
```
