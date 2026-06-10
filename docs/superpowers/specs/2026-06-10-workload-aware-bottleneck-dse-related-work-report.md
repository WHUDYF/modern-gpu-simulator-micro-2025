# Workload-Aware Bottleneck-Guided DSE Related Work Report

Date: 2026-06-10

## 1. 背景问题

我们刚刚讨论的关键问题是：

```text
LUMINA 已经能从 simulator code + sensitivity study 出发做 GPU DSE。
那为什么我们还需要 workload-trace-centric 的入口？
```

结论应当收窄：

```text
workload-trace-centric 不是天然必要。
```

它只有在以下前提成立时才有价值：

```text
full-workload DSE / simulation 太贵，
不能对每个候选 GPU 配置都完整运行目标 AI workload。
```

此时：

```text
trace-compressed representative kernel groups
```

可以作为一种降低评估成本的方法。

因此，我们的准确定位不是：

```text
workload trace 比 simulator-code knowledge 更好。
```

而是：

```text
when full-workload simulation is expensive,
trace compression can reduce the number of expensive sensitivity and
simulation evaluations required for workload-aware GPU DSE.
```

## 2. 已下载论文和材料

新增目录：

```text
papers/workload-aware-dse/
```

| 文件 | 来源 | 类型 | 对我们的价值 |
| --- | --- | --- | --- |
| `archexplorer-microarchitecture-exploration-bottleneck-analysis-micro2023.pdf` | https://www.cse.cuhk.edu.hk/~byu/papers/C180-MICRO2023-ArchExplorer.pdf | MICRO 2023 paper | bottleneck-analysis-driven microarchitecture exploration 的直接参照。 |
| `explainable-dse-hwsw-codesign-bottleneck-analysis-asplos2024.pdf` | https://mpslab-asu.github.io/publications/papers/Dave2024ASPLOS.pdf | ASPLOS 2024 paper | DNN accelerator HW/SW co-design 中用 bottleneck analysis 指导 DSE，和我们最接近。 |
| `llmcompass-efficient-hardware-design-llm-inference-isca2024.pdf` | https://augustning.com/assets/papers/llmcompass-isca-2024.pdf | ISCA 2024 paper | LUMINA 使用的 LLM inference hardware design / simulator 相关基础。 |
| `dsdl-design-space-description-language-explainable-dse-latte2022.pdf` | https://capra.cs.cornell.edu/latte22/paper/5.pdf | LATTE 2022 paper | 用 design-space description language 系统表达 accelerator DSE 空间。 |
| `boom-explorer-riscv-boom-microarchitecture-dse-iccad2021-slides.pdf` | https://www.cse.cuhk.edu.hk/~byu/papers/C122-ICCAD2021-DSE-BOOM-slides.pdf | slides | BOOM-Explorer 的官方 slides；正文 PDF 链接不可用。 |

已有相关论文：

```text
papers/gpu-configuration-dse/lumina-llm-guided-gpu-architecture-exploration-arxiv2603.05904.pdf
```

## 3. LUMINA 的定位

LUMINA 是：

```text
LLM-guided GPU architecture DSE via bottleneck analysis.
```

它的关键链路是：

```text
simulator code
  -> Qualitative Engine
  -> architecture parameter to metric influence map

sensitivity study
  -> Quantitative Engine
  -> local PPA influence values

simulator feedback
  -> bottleneck / critical-path analysis
  -> Strategy Engine
  -> candidate parameter adjustment

evaluation results
  -> Pareto frontier
  -> refinement loop
```

它的核心思想可以概括为：

```text
qualitative mapping
  + quantitative sensitivity
  + bottleneck-guided search
  + simulator validation
```

这和我们讨论的 SAAKE / GCL 路线在结构上非常相似。

关键区别：

```text
LUMINA:
  knowledge source = simulator code + sensitivity studies

Our possible route:
  knowledge source = workload trace + representative kernel sensitivity
```

## 4. ArchExplorer

ArchExplorer 是 LUMINA 明确引用的 bottleneck-driven microarchitecture exploration 工作。

它的基本思想是：

```text
find bottleneck / critical path
  -> identify which architecture resources limit performance
  -> remove or mitigate bottleneck
  -> explore next design point
```

对我们的启发：

```text
DSE 不一定要完全 black-box。
如果能解释当前设计的 bottleneck，
就可以更高效地选择下一个 design point。
```

对应到我们：

```text
SAAKE-like sensitivity
  -> identifies representative kernel resource pressure
  -> guides GPU knob search
```

但区别也明显：

```text
ArchExplorer:
  microarchitecture-centric

我们:
  workload kernel group-centric
```

## 5. Explainable-DSE

Explainable-DSE 是当前与我们最像的方法论之一。

它处理的是：

```text
efficient HW/SW codesigns of deep learning accelerators
using bottleneck analysis
```

它的核心不是盲目搜索，而是：

```text
cost model / bottleneck model
  -> identify inefficient part of current design
  -> propose bottleneck-mitigating action
  -> evaluate candidate
  -> continue DSE
```

这和我们的目标高度相似：

```text
GCL/SAAKE:
  identify resource pressure from representative kernel groups

PPA-constrained search:
  propose bottleneck-mitigating GPU knob candidates

simulator validation:
  verify top candidates
```

区别：

```text
Explainable-DSE:
  DNN accelerator HW/SW co-design

Our route:
  GPU-like architecture configuration search
  with workload trace compression
```

它对我们的最大启发是：

```text
explainability 可以服务于 search efficiency，
而不是只服务于解释结果。
```

## 6. LLMCompass

LLMCompass 是面向 LLM inference hardware design 的建模 / 仿真基础。

它对我们的意义是：

```text
现代 AI workload 的硬件 DSE 可以建立 workload-aware simulator/model。
```

LUMINA 使用 LLMCompass 类工具来评估：

```text
TTFT
TPOT
area
```

这说明：

```text
如果有合适 workload-specific simulator，
simulator-centric DSE 是非常自然的。
```

也就是说，LLMCompass/LUMINA 反过来提醒我们：

```text
只有当 full workload evaluation 成本太高，
或者缺少完整 workload-level simulator 时，
trace compression 才更有必要。
```

## 7. DSDL

DSDL 的作用是：

```text
systematically describe design spaces for next-generation hardware accelerators.
```

对我们的启发：

```text
GPU configuration search 必须有 legal design space schema。
```

这和我们之前强调的边界一致：

```text
不能让模型自由生成非法 GPU 配置。
```

我们后续需要类似：

```json
{
  "knob": "num_sms",
  "level": "chip",
  "values": [0.0, 0.1, 0.2],
  "constraints": ["area_budget", "power_budget", "memory_partition_balance"]
}
```

而不是让 LLM / search algorithm 随便输出：

```text
single CTA uses 512 SMs
```

## 8. BOOM-Explorer

BOOM-Explorer 是 RISC-V BOOM microarchitecture DSE 工作。

这里下载的是官方 slides，因为论文 PDF 链接当前不可用。

它与我们不是同一目标架构，但说明：

```text
microarchitecture DSE 通常需要明确 design variables、
constraints、surrogate model 或 search strategy。
```

它可作为 broader microarchitecture DSE related work，而不是我们的核心对标。

## 9. 对我们路线的直接修正

我们不应强行说：

```text
我们比 LUMINA 更好，因为我们是 workload-trace-centric。
```

正确说法是：

```text
LUMINA demonstrates the effectiveness of simulator-code-centric,
bottleneck-guided GPU DSE.

Our work targets the complementary case where full workload evaluation
is expensive. We use trace-compressed representative kernel groups to reduce
the cost of sensitivity analysis and configuration validation.
```

因此，workload trace 的必要性来自：

```text
simulation budget pressure
```

而不是来自：

```text
trace graph 天然比 simulator code 更先进。
```

## 10. 推荐方法定位

推荐标题方向：

```text
Trace-Compressed Sensitivity-Guided GPU Configuration Search
for AI Workloads
```

或者：

```text
Simulation-Budget-Efficient GPU DSE via Representative Kernel Sensitivity
```

核心 claim：

```text
We reduce the number of expensive workload-level DSE evaluations by replacing
full-workload sensitivity analysis with trace-compressed representative kernel
groups, while preserving PPA-constrained configuration quality.
```

更保守 claim：

```text
We provide a workload-aware prior for GPU configuration search from
representative kernel sensitivity, and validate top candidates through
simulator feedback.
```

## 11. 不建议第一版做的事情

第一版不建议把系统做得过重：

```text
GCL + GNNExplainer + LLM + SAAKE + QIG + LUMINA-like refinement
```

这会导致：

```text
工程量过大
每个模块都可能被审稿人追问
核心贡献变模糊
```

第一版建议保留：

```text
representative kernel selection
SAAKE-like sensitivity
PPA-constrained small-grid search
top-k simulator validation
```

GNNExplainer / LLM 可以作为：

```text
interpretation layer
future extension
```

而不是 MVP 必需组件。

## 12. 最小实验建议

为了证明 workload-trace-centric 有必要，实验必须围绕：

```text
simulation cost reduction
```

而不是只报最终 speedup。

建议指标：

```text
number of full workload simulations saved
number of representative kernel evaluations
top-k hit rate
best speedup found under fixed simulation budget
PPA-valid candidate rate
Pareto frontier quality
```

关键对比：

```text
full workload DSE
random search
grid search
simulator-centric bottleneck baseline if available
our representative-kernel-guided search
```

如果我们的结果是：

```text
using <=30% evaluations to reach >=90% of best-found speedup
```

这就能支撑：

```text
trace compression has value.
```

否则，workload-trace-centric 入口的必要性不足。
