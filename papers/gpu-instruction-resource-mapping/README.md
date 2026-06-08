# GPU Instruction Resource Mapping Papers

日期：2026-06-06

这个目录保存与下面问题直接相关的论文和资料：

```text
能否从 SASS / trace instruction composition / graph motif
推断当前 kernel 更像使用了哪些 GPU hardware structures？
```

需要严格区分：

```text
instruction / motif -> hardware-structure candidate evidence
```

和：

```text
instruction / motif -> causal bottleneck truth
```

第一版只能使用前者。瓶颈和 knob 重要性仍然需要 profiler counter、microbench anchor 和 closed-loop validation 支撑。

## Local Files

| File | Paper / Source | Why It Matters |
| --- | --- | --- |
| `gpuscout-scw2023.pdf` | GPUscout: Locating Data Movement-related Bottlenecks on GPUs | 说明如何围绕 GPU 数据移动建立 bottleneck localization 证据。对我们的 `global_memory_path`、load-to-use motif、memory hierarchy evidence 很有参考价值。 |
| `instruction-roofline-gpu-pmbs2019.pdf` | An Instruction Roofline Model for GPUs | 把 GPU 执行建模到指令层 roofline，用 instruction categories 和 memory patterns 解释性能上限。适合作为“指令组合不是瓶颈真值，但可以形成资源候选证据”的理论来源。 |
| `sassi-flexible-software-profiling-gpu-architectures-isca2015.pdf` | SASSI: Flexible Software Profiling of GPU Architectures | 展示 SASS-level instrumentation 可以如何收集底层执行证据。对我们从 trace/SASS 构建 opcode evidence、edge evidence、dynamic instruction evidence 很关键。 |
| `nsight-compute-kernel-profiling-guide.pdf` | NVIDIA Nsight Compute Kernel Profiling Guide | NVIDIA 官方 profiler metric 语义来源。用于把 opcode / motif candidate 与真实 counter 名称、memory pipeline、scheduler、occupancy 等概念对齐。 |
| `gpu-performance-counters-kernel-characterization-iccs2020.pdf` | Utilizing GPU Performance Counters to Characterize GPU Kernels via Machine Learning | 使用 GPU performance counters 做 kernel characterization / classification。它不是 graph 方法，但可以作为 counter evidence 与 ML 分类的基线参考。 |
| `hong-dissertation-code-optimization-on-gpus-2019.pdf` | Code Optimization on GPUs, Changwan Hong, PhD dissertation, 2019 | 开放全文。第 2 章包含 `GPU Code Optimization Using Abstract Kernel Emulation and Sensitivity Analysis` 的扩展内容。对我们最重要的是 SAAKE：通过 abstract kernel emulation 和 sensitivity analysis 判断资源瓶颈，并把结果接入 OpenTuner。 |

## Not Downloaded

| Paper | Reason | Link |
| --- | --- | --- |
| GPU code optimization using abstract kernel emulation and sensitivity analysis, PLDI 2018 standalone paper PDF | ACM PDF 入口返回 HTTP 403，CORE unpaywalled PDF 链接返回 HTTP 522。为避免上传无效 HTML/损坏文件，本目录暂不保存 standalone paper PDF。已保存作者博士论文开放全文作为替代阅读材料，其中第 2 章包含该 PLDI 工作的扩展内容。 | https://www.pnnl.gov/publications/gpu-code-optimization-using-abstract-kernel-emulation-and-sensitivity-analysis |

## How To Use These Papers In Our Method

推荐将这些工作放在论文中的同一层：

```text
Hardware semantic binding / evidence grounding
```

它们不是 GCL / GNNExplainer 的替代，而是给 GCL cluster 和 explanation motif 提供硬件语义映射依据：

```text
GCL cluster
  -> GNNExplainer motif
  -> opcode + graph motif + counter evidence
  -> hardware-structure candidates
  -> family candidates
  -> registry-constrained knob candidates
```

可写成的低风险 claim：

```text
The instruction and counter evidence grounds the learned trace motifs in GPU hardware structures.
```

不要写成：

```text
The instruction mix directly identifies the bottleneck or causal knob importance.
```
