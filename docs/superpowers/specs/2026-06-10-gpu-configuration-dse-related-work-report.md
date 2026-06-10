# GPU Configuration DSE Related Work Report

Date: 2026-06-10

## 1. 问题背景

我们担心的问题是：

```text
如果我们想做 GPU 硬件配置调参 / 优化建议，
但相关工作很少，是否说明这条路线本身有问题？
```

检索后的结论是：

```text
这条路线有大量相关工作支撑。
```

已有工作已经覆盖：

```text
GPU design space exploration
architecture parameter importance analysis
performance / power / area modeling
surrogate-model-guided hardware search
DNN accelerator hardware resource assignment
mapping + hardware co-optimization
LLM-guided GPU architecture exploration
```

因此，我们的问题不是“没人做所以不可行”，而是：

```text
如何把我们的贡献边界写清楚。
```

推荐定位：

```text
Trace-compressed, sensitivity-guided, PPA-constrained GPU configuration search
for optimized AI workloads.
```

## 2. 已下载材料

论文和补充材料放在：

```text
papers/gpu-configuration-dse/
```

| 文件 | 类型 | 来源 | 对我们的价值 |
| --- | --- | --- | --- |
| `stargazer-automated-regression-gpu-dse-ispass2012.pdf` | paper | https://mrmgroup.cs.princeton.edu/papers/stargazer.pdf | GPU architecture design space exploration 的直接参考。 |
| `qig-importance-interaction-gpgpu-architecture-parameters-tcad2018.pdf` | paper | https://users.elis.ugent.be/~leeckhou/papers/tcad2018.pdf | 量化 GPGPU architecture parameter importance / interaction。 |
| `starchart-regression-trees-gpu-power-performance-pact2013.pdf` | paper | https://www.istc-cc.cmu.edu/publications/papers/2013/starchart.pdf | 用 regression tree 做 GPU power/performance tuning。 |
| `accelwattch-modern-gpu-power-modeling-micro2021.pdf` | paper | https://paragon.cs.northwestern.edu/papers/2021-MICRO-AccelWattch-Kandiah.pdf | modern GPU power modeling / validation 基础设施。 |
| `lumina-llm-guided-gpu-architecture-exploration-arxiv2603.05904.pdf` | paper | https://arxiv.org/abs/2603.05904 | LLM-guided GPU architecture exploration 的前沿参考。 |
| `confuciux-hardware-resource-assignment-dnn-accelerators-micro2020.pdf` | paper | https://arxiv.org/abs/2009.02010 | DNN accelerator hardware resource assignment under constraints。 |
| `maestro-data-centric-dnn-dataflow-cost-model-micro2019.pdf` | paper | https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2019_Maestro.pdf | DNN dataflow 的 performance / energy / hardware cost 建模。 |
| `digamma-hw-mapping-co-optimization-dnn-accelerators-date2022-official.pdf` | paper | https://d1qx31qr3h6wln.cloudfront.net/publications/DATE_2022_DiGamma.pdf | DNN accelerator hardware + mapping co-optimization。 |
| `gamma-mapping-space-exploration-tutorial-slides-micro2020.pdf` | tutorial slides | https://maestro.ece.gatech.edu/docs/build/html/GAMMA.html | GAMMA mapping-space exploration 的官方补充材料。 |

说明：

```text
GAMMA 正式论文 PDF 的 NSF 源下载速度过慢，作者旧链接不可用。
当前提交的是 MAESTRO 官方文档中的 GAMMA tutorial slides。
如后续需要正式论文 PDF，可单独从 ACM / IEEE / NSF 源补拉。
```

## 3. GPU 方向相关工作

### 3.1 Stargazer

Stargazer 是非常直接的 GPU architecture design space exploration 工作。

它的基本流程是：

```text
GPU architecture parameter space
  -> sparse random sampling
  -> GPGPU-Sim simulation
  -> stepwise regression model
  -> predict unexplored design points
  -> identify good GPU designs
```

它说明：

```text
通过少量采样 + surrogate model 来探索巨大 GPU 配置空间是可行的。
```

对我们的价值：

```text
我们的 PPA-constrained GPU configuration table 可以看作更小、更有先验的 DSE。
```

区别：

```text
Stargazer:
  直接在架构参数空间中采样。

我们:
  先用 GCL 压缩 workload kernel groups，
  再用 SAAKE-like sensitivity 生成 knob prior，
  最后只搜索更小的合法配置空间。
```

### 3.2 QIG

QIG 关注：

```text
GPGPU architecture parameters 的 importance 和 interaction。
```

它使用机器学习模型分析不同架构参数对性能的影响，并量化参数之间的交互。

这对我们非常关键，因为它证明：

```text
GPU architecture knob importance 不是不能学。
已有工作确实在量化哪些 GPU 参数重要，以及参数之间如何相互影响。
```

对我们的价值：

```text
支持我们输出 knob priority / resource pressure ranking 的合理性。
```

区别：

```text
QIG:
  从一批 architecture samples 中学习 parameter importance。

我们:
  从 kernel group sensitivity 出发，
  生成 workload-specific knob prior，
  再做 PPA-constrained search。
```

### 3.3 Starchart

Starchart 使用 regression trees 做 GPU hardware/software optimization。

它的核心思想是：

```text
GPU tuning space 非线性、分区明显，
可以用 recursive partitioning / regression trees
找不同区域的关键参数和优化策略。
```

对我们的价值：

```text
支持后续从 grid search 升级到 tree-based surrogate model。
```

它也提醒我们：

```text
单个线性 sensitivity 不能无限外推。
不同配置区域可能有不同 bottleneck。
```

这和我们前面对 SAAKE 的边界一致：

```text
sensitivity 是局部证据，
top-k 配置必须进入 simulator validation。
```

### 3.4 AccelWattch

AccelWattch 是 modern GPU power modeling framework。

它对我们的价值不是直接做配置搜索，而是提供：

```text
GPU power estimation / validation infrastructure
```

这对我们的 PPA-constrained search 很关键。

第一版我们可以使用 proxy cost：

```text
area_cost / power_cost as rough constraints
```

但更强版本应该是：

```text
top-k candidate configurations
  -> simulator validation
  -> AccelWattch-like power model
  -> stronger PPA evidence
```

论文边界：

```text
如果只用 proxy PPA，不能 claim accurate chip power.
如果接入 AccelWattch-like model，可以 claim stronger power-aware validation.
```

### 3.5 LUMINA

LUMINA 是非常新的 GPU architecture exploration 工作。

它的问题设定和我们很接近：

```text
GPU design space huge
simulation cost high
need performance / power / area trade-off
use bottleneck analysis / AI guidance to explore architecture
```

对我们的价值：

```text
说明 GPU architecture exploration + bottleneck analysis + AI guidance
是正在出现的新方向。
```

区别：

```text
LUMINA:
  更强调 LLM 从 simulator / architecture code 中提取知识，
  生成 DSE rules。

我们:
  更强调 trace-compressed representative kernel groups，
  SAAKE-like sensitivity prior，
  PPA-constrained workload-specific configuration table。
```

因此我们不是重复 LUMINA，而是更偏：

```text
workload trace compression + sensitivity-guided constrained search
```

## 4. DNN Accelerator 方向相关工作

这些工作不是通用 GPU，但与我们的目标高度相关，因为它们都在做：

```text
DNN workload
  -> hardware resource / mapping search
  -> performance / energy / area trade-off
```

### 4.1 MAESTRO

MAESTRO 是 DNN accelerator dataflow cost model。

它分析：

```text
data reuse
performance
energy
hardware cost
```

对我们的价值：

```text
支持 workload-specific cost modeling 的范式。
```

它说明：

```text
硬件配置好不好，必须结合 workload 的 data movement / reuse pattern。
```

这和我们的 GCL/SAAKE 目标一致：

```text
不是泛泛调 GPU 参数，
而是针对具体 AI workload 得到 resource pressure 和配置候选。
```

### 4.2 ConfuciuX

ConfuciuX 使用 RL 做 DNN accelerator hardware resource assignment。

它的问题形式非常接近：

```text
在 area / power constraints 下，
为 DNN accelerator 分配硬件资源，
优化 performance / energy。
```

对我们的价值：

```text
证明 hardware resource assignment under PPA constraints 是合理研究问题。
```

区别：

```text
ConfuciuX:
  主要面向 DNN accelerator。

我们:
  面向 GPU-like architecture configuration，
  并用 GCL/SAAKE 把 workload kernel groups 显式接到 resource sensitivity。
```

### 4.3 GAMMA / DiGamma

GAMMA / DiGamma 关注：

```text
DNN accelerator mapping + hardware co-optimization
```

它们用搜索方法探索：

```text
mapping choice
hardware resource allocation
performance / energy / cost trade-off
```

对我们的价值：

```text
说明 hardware configuration 不能脱离 workload mapping / execution structure。
```

对应到我们这里：

```text
GCL cluster = workload execution structure compression
SAAKE sensitivity = resource response evidence
PPA grid search = constrained hardware configuration search
```

因此，我们可以把这些工作放在 related work 中作为 accelerator DSE 的背景，而不是 GPU-specific baseline。

## 5. 我们路线的学术位置

已有工作证明以下方向是成立的：

```text
hardware design space exploration
parameter importance analysis
surrogate model
PPA-aware ranking
DNN workload-specific resource assignment
```

我们的新位置可以定义为：

```text
Trace-compressed, sensitivity-guided, PPA-constrained GPU configuration search.
```

方法链路：

```text
full AI workload
  -> GCL kernel group compression
  -> representative kernel selection
  -> SAAKE-like latency/gap sensitivity
  -> workload-level resource pressure ranking
  -> legal global GPU knob search
  -> proxy PPA filtering
  -> PPA table / Pareto frontier
  -> top-k simulator validation
```

与已有 GPU DSE 的区别：

```text
已有 GPU DSE:
  从巨大 architecture space 开始采样。

我们:
  从 workload trace structure 出发，
  用 representative kernel groups 和 sensitivity prior 缩小搜索空间。
```

与 DNN accelerator DSE 的区别：

```text
已有 accelerator DSE:
  多数面向专用 accelerator dataflow / mapping。

我们:
  面向 GPU-like configurable architecture，
  保留 GPU execution model 和合法配置边界。
```

## 6. 风险判断

这条路线的主要风险不是：

```text
没有相关工作。
```

真正风险是：

1. 搜索空间不合法，生成不符合 GPU execution model 的配置。
2. 把 SAAKE sensitivity 错写成最终硬件调参比例。
3. 只用 proxy PPA 却声称真实 chip area/power 最优。
4. 不做 simulator validation，只靠公式表格给最终结论。
5. 把 workload-specific candidate 夸大成 general next-generation GPU design。

因此必须写清楚：

```text
The output is the best candidate within the searched legal design space,
not the true optimal next-generation GPU.
```

## 7. 推荐论文表述

可以写：

```text
Prior GPU DSE methods explore large architecture spaces through simulation
sampling and surrogate modeling. In contrast, our method first compresses
optimized AI workloads into representative kernel groups, then uses
SAAKE-like sensitivity analysis to derive resource/knob priors before
performing PPA-constrained configuration search.
```

可以写：

```text
The proposed search outputs a PPA table and Pareto frontier for
workload-specific GPU configuration candidates under legal design constraints.
```

不要写：

```text
Our method automatically designs the next GPU.
```

不要写：

```text
SAAKE sensitivity directly gives the final hardware tuning ratio.
```

## 8. 对下一步实现的建议

第一版实现不要直接上复杂 DSE。

推荐先做：

```text
small-grid PPA-constrained search
```

最小 knob 集合：

```text
num_sms
fp16_throughput_per_sm
dram_bandwidth
l2_size
shared_memory_per_sm
```

最小输出：

```text
PPA table
valid / invalid status
Pareto frontier
top-k candidate configs
claim_status for each candidate
```

后续可以参考：

```text
Stargazer -> regression surrogate
QIG -> parameter importance / interaction
Starchart -> tree-based partitioning
AccelWattch -> stronger power validation
ConfuciuX / GAMMA / DiGamma -> constrained accelerator resource assignment
LUMINA -> AI-guided DSE rule generation
```

这个演进路径是合理的，而且有清晰相关工作支撑。
