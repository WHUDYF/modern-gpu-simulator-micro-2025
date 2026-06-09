# GCL + SAAKE 到架构假设生成的相关工作阅读说明

Date: 2026-06-09

## 1. 我们现在真正要解决的问题

我们不能再把 GPU 架构迭代理解为简单的参数调优。更准确的表述应该是：

```text
optimized AI workload on an existing GPU
  -> trace / graph compression
  -> representative kernel groups
  -> sensitivity-based resource diagnosis
  -> network-level resource pressure profile
  -> architecture improvement hypothesis
```

也就是说，我们的目标不是直接证明“某个参数就是最优 GPU 架构”，而是用 GCL 压缩后的代表 kernel 和 SAAKE 类敏感性分析，快速找出优化后 AI workload 在现有架构上的主要资源压力，并把这些压力提升为可验证的架构假设。

安全 claim 应该是：

```text
trace-compressed, sensitivity-guided simulator calibration
and architecture-parameter prioritization for optimized AI workloads
```

如果要说架构设计，则应表述为：

```text
our tool diagnoses workload-specific resource pressure under optimized software,
then generates architecture improvement hypotheses.
```

不要 claim：

- GCL cluster 自动具有硬件语义。
- instruction ratio 可以直接证明 bottleneck 或 knob importance。
- SAAKE 类 sensitivity 可以直接证明真实硬件瓶颈。
- 我们找到了最优 GPU architecture。
- simulator knob tuning 等价于新架构发明。

## 2. 已下载论文

论文统一放在：

```text
papers/gpu-architecture-hypothesis/
```

| 文件名 | 原始地址 | 我们为什么需要 |
| --- | --- | --- |
| `delta-gpu-performance-model-deep-learning-ispass2019.pdf` | https://research.nvidia.com/publication/2019-04_delta-gpu-performance-model-deep-learning-applications-depth-memory-system | 学习如何把 DNN workload 的性能和 memory-system traffic 细分关联起来。 |
| `accel-sim-validated-gpu-modeling-isca2020.pdf` | https://engineering.purdue.edu/tgrogers/publication/khairy-isca-2020/ | 给 simulator-based GPU architecture study 提供可信基础。 |
| `modern-gpu-memory-system-design-accurate-modeling-arxiv1810.07269.pdf` | https://arxiv.org/abs/1810.07269 | 学习如何通过 accurate modeling 讨论现代 GPU memory system 设计问题。 |
| `flashattention2-parallelism-work-partitioning-arxiv2307.08691.pdf` | https://arxiv.org/abs/2307.08691 | 说明高度优化软件会改变 workload 对 GPU 资源的使用方式。 |
| `flashattention3-hopper-asynchrony-low-precision-arxiv2407.08608.pdf` | https://arxiv.org/abs/2407.08608 | 更直接体现 Hopper 异步机制和 low-precision 机制如何改变最优实现。 |
| `demystifying-gpu-microarchitecture-microbenchmarking-ispass2010.pdf` | https://www.stuffedcow.net/files/gpuarch-ispass2010.pdf | 学习 microbenchmark 如何反推 GPU 内部结构和资源属性。 |
| `dissecting-nvidia-volta-microbenchmarking-arxiv1804.06826.pdf` | https://arxiv.org/abs/1804.06826 | 学习对 Volta 架构做系统 microbenchmarking 的方法。 |
| `dissecting-nvidia-hopper-microbenchmarking-arxiv2501.12084.pdf` | https://arxiv.org/abs/2501.12084 | 学习新 GPU 架构如何通过 microbenchmark 归纳具体资源机制。 |
| `lumina-llm-guided-gpu-architecture-exploration-arxiv2603.05904.pdf` | https://arxiv.org/abs/2603.05904 | 作为前沿参考：LLM + bottleneck analysis 用于 GPU 架构探索。 |

## 3. 每篇工作的核心思想

### 3.1 DeLTA: GPU Performance Model for Deep Learning Applications

DeLTA 关注 DNN workload 在 GPU 上的性能建模，重点不是只看整体 runtime，而是更深入地分析 memory-system traffic。它的价值在于把深度学习模型运行中的数据移动、cache / memory traffic 和性能结果联系起来。

对我们的启发：

- 我们后续不能只说“这个 kernel 是 memory-bound”，而应该区分是哪一类 memory traffic 造成压力。
- 对 GCL cluster 的代表 kernel 做 sensitivity 时，应优先输出可解释的 traffic profile，例如 global memory、shared memory、cache、interconnect 或 data movement path。
- DeLTA 可以作为我们把 AI workload 和 GPU memory-system architecture 联系起来的核心参考。

不能过度使用的地方：

- DeLTA 是性能模型，不是图网络压缩方法。
- 它不能直接给出我们的 GCL cluster family。

### 3.2 Accel-Sim: Validated GPU Modeling

Accel-Sim 解决的是 GPU 模拟器可信度问题。它提供了可扩展、经过验证的 GPU simulation framework，让研究者能够在比较可信的环境中探索架构参数变化。

对我们的启发：

- 如果我们提出硬件参数调整建议，必须落到 validated simulator loop 中验证。
- 我们的输出最好是 `candidate knob priority`，然后通过 Accel-Sim / GPGPU-Sim 类工具做低预算验证。
- 论文里应强调 simulator validation 是 claim 升级的边界：没有验证之前只能叫 hypothesis。

不能过度使用的地方：

- Accel-Sim 不负责自动发现 kernel family。
- 它不是调参算法，而是验证平台。

### 3.3 Exploring Modern GPU Memory System Design Challenges through Accurate Modeling

这篇工作强调：现代 GPU memory system 的设计问题必须通过足够准确的建模来分析。它不是简单把 L2、DRAM、cache 参数逐个扫一遍，而是关注 memory system 中不同组件如何共同影响性能。

对我们的启发：

- 适合作为 memory-only MVP 的理论支撑。
- 如果第一版只做 memory resource family，仍然可以成立，因为 memory system 本身就是 GPU 架构设计中足够重要、足够复杂的一部分。
- 它支持我们从“调参数”转向“诊断 memory-system design pressure”。

不能过度使用的地方：

- 它主要是 memory-system design，不覆盖所有 compute pipeline。
- 如果我们的实验只做 memory family，就不要声称覆盖完整 GPU architecture。

### 3.4 FlashAttention-2

FlashAttention-2 展示了在现代 GPU 上，attention 算子的最优实现不仅取决于算法复杂度，还取决于并行划分、work partitioning、warp/block 组织和数据移动策略。

对我们的启发：

- 说明“软件已经高度优化”不是我们工作的威胁，而是我们工作的前提。
- 我们应该分析 optimized AI kernels 的资源压力，而不是用很差的 CUDA 代码制造虚假提升。
- GCL cluster 可以帮助我们从完整模型中找出真正代表优化后 workload 的 kernel group。

不能过度使用的地方：

- FlashAttention-2 是软件优化工作，不是硬件调参方法。
- 它不能直接证明某个硬件资源应该增加。

### 3.5 FlashAttention-3 on Hopper

FlashAttention-3 更直接体现了新架构机制如何改变软件最优实现。它利用 Hopper 的异步执行、Tensor Core、低精度能力等特性，让 attention kernel 的设计方式进一步变化。

对我们的启发：

- 这是“架构创新不是简单参数调大”的最好例子。
- 如果我们的 sensitivity 发现大量 optimized attention-like kernels 对 global-to-shared data movement 敏感，我们不能只说“加大带宽”，而应提出类似 TMA / async copy / pipeline overlap 的 architecture mechanism hypothesis。
- 它可以帮助我们解释：我们的工具输出的是下一代机制设计线索，不是最终架构方案。

不能过度使用的地方：

- 它证明 Hopper 机制对 FlashAttention 有价值，但不证明我们的 workload 一定需要同样机制。
- 我们仍然需要 simulator 或 microbenchmark validation。

### 3.6 Demystifying GPU Microarchitecture through Microbenchmarking

这类 microbenchmark 工作通过精心设计的小程序，反向推断 GPU 内部资源特征，例如 latency、throughput、cache 行为、bank 冲突、pipeline 特性等。

对我们的启发：

- 我们可以用 LLM 或人工设计的 microbench anchor 来给 GCL embedding space 提供硬件语义锚点。
- microbench anchor 不应当作为 ground truth，而应作为可复验的 semantic probe。
- 它支持我们前面讨论的路线：`real kernel graph + microbench anchor graph -> mechanism similarity / candidate attribution`。

不能过度使用的地方：

- microbenchmark 推断的是局部机制，不一定覆盖真实 AI workload 的组合行为。
- microbench label 需要校准，不能直接当成最终 family label。

### 3.7 Dissecting NVIDIA Volta via Microbenchmarking

这篇工作针对 Volta 做系统化 microbenchmarking，展示如何通过受控实验理解一个真实 GPU 架构的执行单元、memory hierarchy 和调度行为。

对我们的启发：

- 可以作为我们构建 microbench anchor library 的方法参考。
- 对每个 anchor，最好明确记录它想刺激的资源、预期 counter/sensitivity 行为、以及失败时如何解释。
- 它有助于让审稿人相信我们的 microbench 不是随意生成的。

不能过度使用的地方：

- 它主要是架构反推，不是 workload-level compression。
- 它不会自动给出 simulator knob importance。

### 3.8 Dissecting NVIDIA Hopper via Microbenchmarking

这篇工作延续 microbenchmarking 思路到更新的 Hopper 架构。它的价值在于说明：即使是新架构，也可以通过系统 microbench 去拆解关键资源机制。

对我们的启发：

- 对 Hopper / TMA / async pipeline 这类新机制，我们可以用 microbench anchor 作为机制解释层。
- 如果我们后续希望把 memory-only MVP 扩展到 async data movement 或 tensor pipeline，这篇工作值得优先读。

不能过度使用的地方：

- 它不是自动化 architecture search。
- 它无法代替我们的 GCL cluster selection 和 sensitivity aggregation。

### 3.9 LUMINA: LLM-Guided GPU Architecture Exploration

LUMINA 是非常新的方向，使用 LLM 引导 GPU architecture exploration，并围绕 bottleneck analysis 做设计空间探索。它和我们讨论的“LLM 辅助理解资源瓶颈、生成架构假设”方向接近。

对我们的启发：

- 可以作为 frontier related work，而不是核心依赖。
- 它支持我们把 LLM 放在 hypothesis generation / explanation 层，而不是把 LLM label 当成 ground truth。
- 如果要写论文，可以把我们的区别表述为：我们更强调 trace compression、representative kernel selection、simulator mismatch sensitivity，而不是直接让 LLM 做架构搜索。

不能过度使用的地方：

- LLM 输出必须被 simulator / microbenchmark / counter 约束。
- 不要把 LLM 解释当作可校准概率。

## 4. 这些论文如何拼成我们的路线

这些工作可以形成一条比较稳的论文逻辑：

```text
GCL / graph encoder
  -> 从完整 AI workload 中压缩出代表 kernel group

microbenchmarking papers
  -> 给 graph cluster 提供可复验的机制锚点

SAAKE-like sensitivity
  -> 对代表 kernel 做资源扰动，估计哪个资源解释性能差异

DeLTA / memory-system modeling
  -> 把 AI workload 的 memory traffic 和 GPU resource pressure 联系起来

Accel-Sim
  -> 提供 validated simulator loop，把 hypothesis 升级为 validated candidate

FlashAttention-2/3
  -> 说明 optimized software 与 architecture mechanism 会共同演化

LUMINA
  -> 说明 LLM-guided architecture exploration 是新趋势，但我们要用验证约束它
```

## 5. 对我们当前方案的建议

第一版不要试图覆盖所有硬件组件。更稳的 MVP 是：

```text
memory-focused GCL + sensitivity-guided hardware hypothesis generation
```

具体步骤：

1. 用 GCL 对真实 AI workload 的 trace graph 做 kernel cluster compression。
2. 每个 cluster 选择代表 kernel，而不是对所有 kernel 做昂贵闭环。
3. 对代表 kernel 做 SAAKE-like resource perturbation，先覆盖 global memory、shared memory、L2、DRAM、global-to-shared transfer path。
4. 聚合到 network-level resource pressure profile。
5. 输出候选架构假设，例如：

```json
{
  "architecture_hypothesis": "improve_global_to_shared_async_transfer",
  "supporting_clusters": ["cluster_A", "cluster_B", "cluster_D"],
  "network_weighted_support": 0.67,
  "candidate_mechanisms": ["TMA-like async copy", "wider copy path", "better overlap of copy and compute"],
  "claim_status": "architecture_hypothesis_not_validated_design"
}
```

6. 只对 top-k hypothesis 做 simulator validation。

## 6. 论文写作边界

推荐表述：

```text
We do not claim to automatically invent a new GPU architecture.
Instead, we use trace-compressed representative kernels and sensitivity analysis
to identify workload-specific resource pressure and generate validated
architecture-parameter priorities.
```

如果实验只验证了 memory family，应写成：

```text
This paper focuses on memory-system-oriented architecture hypotheses.
Compute-pipeline and scheduler-oriented extensions are left as future work.
```

这样比泛泛声称“端到端 GPU 架构自动调参”更容易站住。
