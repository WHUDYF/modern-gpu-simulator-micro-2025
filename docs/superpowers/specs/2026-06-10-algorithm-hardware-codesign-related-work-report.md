# Algorithm-Hardware Co-Design Related Work Report

Date: 2026-06-10

## 1. 核心问题

我们关心的问题是：

```text
有没有人从算法层面考虑硬件？
如果有，我们和他们的区别在哪里？
```

结论是：

```text
有，而且 hardware-aware NAS / algorithm-hardware co-design
已经是一个成熟方向。
```

但这些工作大多做的是：

```text
给定硬件，搜索或改造 neural network architecture
```

或者：

```text
同时搜索 neural architecture 和 specialized accelerator architecture
```

而我们希望做的是：

```text
固定 optimized AI workload
  -> 不改算法结构
  -> 从 kernel trace / graph / sensitivity 出发
  -> 推荐 GPU-like architecture configuration candidates
```

因此，我们不能 claim：

```text
首次从算法层面考虑硬件。
```

更安全的 claim 是：

```text
Unlike hardware-aware NAS that modifies the neural architecture,
we keep the optimized AI workload fixed and derive GPU configuration
candidates from trace-compressed representative kernel groups.
```

## 2. 已下载论文

论文放在：

```text
papers/algorithm-hardware-codesign/
```

| 文件 | 来源 | 类型 | 对我们的价值 |
| --- | --- | --- | --- |
| `mnasnet-platform-aware-nas-mobile-cvpr2019.pdf` | https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.html | Hardware-aware NAS | 证明 neural architecture search 可以直接把真实硬件 latency 放进目标。 |
| `proxylessnas-direct-nas-target-task-hardware-iclr2019.pdf` | https://arxiv.org/abs/1812.00332 | Hardware-aware NAS | 直接在 target task / target hardware 上搜索，不依赖 proxy task。 |
| `fbnet-hardware-aware-efficient-convnet-dnas-cvpr2019.pdf` | https://arxiv.org/abs/1812.03443 | Hardware-aware NAS | differentiable NAS + hardware latency objective。 |
| `once-for-all-train-one-network-efficient-deployment-iclr2020.pdf` | https://arxiv.org/abs/1908.09791 | Deployment-aware NAS | 训练一个 super-network，为不同硬件快速 specialize subnet。 |
| `nahas-neural-architecture-hardware-accelerator-search-arxiv2102.08619.pdf` | https://arxiv.org/abs/2102.08619 | NN + accelerator co-design | 同时考虑 neural architecture 和 hardware accelerator search。 |
| `naas-neural-accelerator-architecture-search-dac2021.pdf` | https://arxiv.org/abs/2105.13258 | Accelerator architecture search | 搜索 accelerator architecture / mapping，和 hardware-aware NAS 可结合。 |
| `towards-codesign-neural-networks-accelerators-mlsys2022.pdf` | https://proceedings.mlsys.org/paper_files/paper/2022/hash/4c430a4d0a7de11e85fa5b076e7f1895-Abstract.html | NN + accelerator co-design | 面向 Edge TPU 类硬件做网络和加速器联合优化。 |
| `co-exploration-neural-architectures-heterogeneous-asic-accelerators-dac2020.pdf` | https://arxiv.org/abs/2002.04116 | NN + ASIC co-exploration | 同时探索 neural architecture 和 heterogeneous ASIC accelerator design。 |
| `codebench-neural-architecture-hardware-accelerator-codesign-benchmark-tecs2022.pdf` | https://dl.acm.org/doi/10.1145/3575798 | Benchmark/framework | 为 neural architecture + hardware accelerator co-design 提供 benchmark/framework。 |
| `neural-architecture-search-survey-hardware-perspective-acmcsur2024.pdf` | https://arxiv.org/abs/2207.04785 | Survey | 从硬件视角总结 NAS 方向，帮助定位 related work。 |

## 3. 第一类：Hardware-Aware NAS

Hardware-aware NAS 的核心是：

```text
search neural network architecture
  under hardware-aware objectives
```

常见目标：

```text
accuracy
latency
energy
memory footprint
model size
throughput
```

### 3.1 MnasNet

MnasNet 是 platform-aware NAS 的代表工作。

它的思想是：

```text
在真实目标平台上测量 model latency，
把 accuracy 和 latency 一起放进 NAS reward，
搜索 mobile-friendly neural network architecture。
```

对我们的启发：

```text
硬件指标不能只用 FLOPs 代理。
真实硬件 latency / runtime 才是最终优化目标。
```

区别：

```text
MnasNet:
  改 neural network architecture 来适应硬件。

我们:
  固定 optimized AI workload，
  用它的 kernel behavior 反推 GPU configuration candidates。
```

### 3.2 ProxylessNAS

ProxylessNAS 的核心是：

```text
direct neural architecture search on target task and target hardware
```

它反对只在 proxy task / proxy dataset / proxy network 上搜索，因为 proxy 可能和真实部署目标不一致。

对我们的启发：

```text
如果我们的目标是 GPU configuration recommendation，
最好直接在目标 workload 和目标 GPU/simulator 上验证，
不要只依赖抽象 proxy。
```

区别：

```text
ProxylessNAS:
  搜索模型结构。

我们:
  搜索 GPU-like architecture knobs。
```

### 3.3 FBNet

FBNet 使用 differentiable NAS 做 hardware-aware efficient ConvNet design。

它的重要点是：

```text
把 hardware latency objective 融入可微搜索过程，
避免只用 FLOPs / parameter count 作为效率指标。
```

对我们的启发：

```text
硬件效率目标可以进入搜索算法本身，
而不是搜索结束后再做后处理。
```

对应到我们：

```text
PPA constraints / runtime speedup / simulator budget
应当进入 GPU configuration search 的目标函数。
```

### 3.4 Once-for-All

Once-for-All 训练一个 large super-network，然后针对不同 hardware constraints 快速选择 subnet。

它的思想是：

```text
一次训练，多硬件部署。
```

对我们的启发：

```text
同一 workload / model 在不同硬件预算下可以产生不同最优候选。
```

对应到我们：

```text
同一 AI workload 在 conservative / aggressive / exploratory PPA budgets 下，
应输出不同 GPU configuration candidates 和 Pareto frontier。
```

## 4. 第二类：Neural Architecture + Accelerator Co-Design

这类工作比 hardware-aware NAS 更接近我们，因为它们不仅改 neural network，也搜索 accelerator architecture。

### 4.1 NAHAS

NAHAS / Rethinking Co-design of Neural Architectures and Hardware Accelerators 关注：

```text
joint neural architecture and hardware accelerator search
```

它强调 network architecture 和 accelerator configuration 之间存在强耦合。

对我们的启发：

```text
算法结构和硬件配置不能完全分开看。
```

区别：

```text
NAHAS:
  联合搜索 neural architecture 和 accelerator。

我们:
  不改 neural architecture，
  只从 fixed optimized workload 中提取 kernel-level evidence。
```

### 4.2 NAAS

NAAS 是 neural accelerator architecture search。

它搜索：

```text
accelerator architecture
mapping
hardware resource allocation
```

并可与 hardware-aware NAS 组合。

对我们的启发：

```text
architecture search 需要同时考虑 mapping / execution behavior，
不能只看静态模型结构。
```

对应到我们：

```text
GCL kernel group compression
可以被看作对 optimized workload execution behavior 的压缩表示。
```

### 4.3 Towards the Co-design of Neural Networks and Accelerators

这篇 MLSys 2022 工作面向 Edge TPU 类硬件，联合优化 neural network 和 accelerator configuration。

它的核心意义是：

```text
co-design 需要真实系统约束，
不是纯模型结构搜索。
```

对我们的启发：

```text
我们的 GPU configuration recommendation 也必须保留合法配置空间和 PPA constraints。
```

### 4.4 Co-exploration of Neural Architectures and Heterogeneous ASIC Accelerator Designs

这篇 DAC 2020 工作同时探索：

```text
neural architectures
heterogeneous ASIC accelerator designs
```

对我们的启发：

```text
不同 neural architecture / workload component
可能适合不同硬件结构。
```

对应到我们：

```text
不同 kernel groups 对 GPU knobs 的 sensitivity 不同，
但最终配置必须是 global GPU configuration。
```

### 4.5 CODEBench

CODEBench 是 neural architecture and hardware accelerator co-design benchmark/framework。

对我们的价值：

```text
说明 co-design 领域已经意识到 benchmark / reproducibility 很重要。
```

如果我们希望快速投稿，也应至少准备：

```text
可复现 workload
固定 legal knob space
PPA budget
baseline search methods
output table schema
```

## 5. 第三类：Survey

`neural-architecture-search-survey-hardware-perspective` 这类 survey 的作用是帮助我们定位：

```text
hardware-aware NAS 已经很成熟，
我们不能把“考虑硬件”作为创新点。
```

我们应把创新点放在：

```text
fixed optimized AI workload
trace-compressed representative kernel groups
SAAKE-like sensitivity to GPU knob prior
PPA-constrained GPU configuration recommendation
```

## 6. 与我们工作的区别

现有 algorithm-hardware co-design 工作大多属于：

```text
A. 固定硬件，改模型结构。
B. 模型结构和专用 accelerator 一起搜索。
C. 为 DNN accelerator 搜 mapping / resource allocation。
```

我们的切口是：

```text
fixed optimized AI workload
  -> no neural architecture modification
  -> GPU kernel trace / graph compression
  -> representative kernel sensitivity
  -> workload-level GPU knob prior
  -> PPA-constrained GPU configuration candidates
```

可以这样写区别：

```text
Hardware-aware NAS changes the algorithm to meet hardware constraints.
Accelerator co-design jointly searches the algorithm and specialized accelerator.
In contrast, our method keeps the optimized AI workload fixed and uses its
kernel-level execution evidence to recommend GPU-like architecture
configuration candidates.
```

## 7. 对我们投稿策略的影响

这些工作说明：

```text
从算法层面考虑硬件是成熟方向。
```

所以我们不能写：

```text
首次提出 algorithm-to-hardware optimization。
```

但它们也说明：

```text
算法行为和硬件配置之间存在研究空间。
```

我们更适合投稿的定位是：

```text
simulation-budget-efficient GPU configuration recommendation
for fixed optimized AI workloads.
```

这比：

```text
end-to-end algorithm-hardware co-design
```

更窄，也更容易快速做出结果。

## 8. 推荐论文表述

推荐：

```text
Our work is inspired by hardware-aware NAS and accelerator co-design,
but targets a different problem: we do not modify the neural architecture.
Instead, we analyze fixed optimized AI workloads and derive GPU configuration
candidates from trace-compressed representative kernel groups.
```

推荐：

```text
The method bridges workload execution evidence and GPU architecture
configuration search under PPA constraints.
```

避免：

```text
We are the first to consider hardware from the algorithm level.
```

避免：

```text
We automatically co-design AI algorithms and GPUs.
```

## 9. 对下一步实验的建议

为了和这些工作拉开差异，第一版实验不要改模型结构。

明确固定：

```text
model architecture
software implementation
input shape
baseline GPU configuration
legal GPU knob space
PPA budget
```

只比较：

```text
不同 GPU configuration search methods
```

例如：

```text
random search
uniform grid search
instruction-ratio heuristic
QIG/Stargazer-like surrogate baseline
our trace-compressed sensitivity-guided search
```

这样我们的故事是：

```text
不是 hardware-aware NAS，
而是 workload-evidence-guided GPU configuration search。
```

这个定位更小，但更适合快速出结果。
