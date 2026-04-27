# 训练 Workload 真实例子与精确周期仿真预算

日期：2026-04-26

## 1. 目的

这份文档回答两个问题：

1. 现代 GPU 训练 benchmark / workload 到底长什么样，主流对象有哪些；
2. 如果在 `modern-gpu-simulator-micro-2025` 这条链路上，真的想跑“训练 benchmark 的精确周期仿真”，大概要多久。

结论先写在前面：

- `PKA` / representative compression 能把海量 kernel 压到少量代表对象，但这不等于已经可以直接对完整训练 benchmark 做精确周期 tuning。
- 结合本仓库在 RTX 5090 上已经测到的仿真速度，**1 秒真实 GPU 执行时间**，在当前链路里大约对应 **2.99 到 50.78 天** 的仿真时间。
- 因此，对现代训练 workload，更现实的做法不是“完整 benchmark 端到端精确周期仿真”，而是：
  `phase-aware representative selection -> short-window trace -> sampled / capped simulation -> writeback`

---

## 2. 当前主流训练 benchmark / workload

下面优先列出当前最值得参考的一组对象：`MLCommons / MLPerf Training` 主线中的 benchmark。它们不是“所有训练 workload”，但它们是今天最接近“主流、公开、可复现、社区承认”的代表集。

### 2.1 当前主线对象

| Benchmark / Workload | 套件状态 | 算法类型 | 公开规模信号 | 数据集 / 任务 | 对模拟器最关键的压力 |
|---|---|---|---|---|---|
| `llama3.1_8b` | MLCommons Training v5.1 | decoder-only Transformer 预训练 | 8B 参数 | `C4` 预训练语料 | GEMM + attention + layernorm + 通信；kernel 数量大，phase 长 |
| `llama3.1_405b` | MLCommons Training v5.1 | 大模型 decoder-only Transformer 预训练 | 405B 参数 | `C4` 预训练语料 | 极高并行度、极强通信依赖；完整 trace / full-cycle 几乎不可做 |
| `llama2_70b_lora` | MLCommons Training v5.1 | decoder-only Transformer + LoRA 微调 | 70B 基座参数 | `SCROLLS GovReport` 长文总结 | 训练比预训练短，但 attention/MLP 结构仍重，context 敏感 |
| `retinanet` | MLCommons Training v5.1 | 一阶段目标检测 CNN/FPN | 37.7M 参数 | `OpenImages` 目标检测 | 卷积 backbone + FPN + detection head，多 kernel family 混合 |
| `dlrm_dcnv2` | MLCommons Training v5.1 | 推荐系统，embedding + MLP + DCNv2 | benchmark 名本身即 `3.5TB` 级数据对象 | `Criteo 1TB x 3.5TB` 风格推荐数据 | embedding lookup / irregular memory / all-to-all 是主要难点 |
| `rgat` | MLCommons Training v5.1 | 图神经网络，Relational GAT | benchmark README 使用 `IGBH-Full` | 异构大图节点分类 | 稀疏访问、scatter/gather、不规则邻接，trace 压缩效果通常差 |
| `flux.1` | MLCommons Training `master` / v6.0 方向 | 文生图生成模型 | 11.9B 参数 | `CC12M` 子集 | U-Net / Transformer / attention 混合，长 phase，显存与带宽压力大 |
| `deepseekv3` | MLCommons Training `master` / v6.0 方向 | MoE LLM 预训练 | 671B total / 37B activated | `C4` 预训练语料 | 专家路由带来更强 invocation heterogeneity 与通信复杂度 |
| `gpt_oss_20b` | MLCommons Training `master` / v6.0 方向 | LLM 预训练 | 21B total / 3.6B active | `Dolmino Mix-1124` | 仍是典型 Transformer/MoE 训练链，但比 400B 级更适合作为 bring-up 候选 |

### 2.2 这些 workload 的“真实感”来自哪里

它们不是随便挑的 toy case，而是覆盖了现代训练里最典型的几条算法主线：

| 算法主线 | 代表 benchmark | 你在 trace 里大概率会看到什么 |
|---|---|---|
| CNN / Vision | `retinanet` | conv / fused conv / norm / reduction / detection head |
| Dense Transformer | `llama3.1_8b`, `llama3.1_405b`, `llama2_70b_lora` | GEMM、attention、softmax、layernorm、residual、all-reduce |
| MoE Transformer | `deepseekv3`, `gpt_oss_20b` | 在 dense Transformer 基础上再叠加 expert routing 与更强异质性 |
| Recommender | `dlrm_dcnv2` | embedding lookup、MLP、cross features、访存不规则、通信重 |
| GNN | `rgat` | 稀疏 gather/scatter、邻接驱动的不规则访存 |
| Diffusion / Generative Image | `flux.1` | 卷积/attention 混合，长链路、显存与带宽开销大 |

---

## 3. 历史上仍然常用的“经典训练 benchmark”

如果你的目标是“先把 simulator 工作流跑通，再逐步上难度”，下面这些对象仍然很有价值。它们在 simulator 论文、调参论文、相关工作对照里出现频率很高。

| Workload | 典型算法 | 规模信号 | 为什么仍值得保留 |
|---|---|---|---|
| `ResNet-50` | CNN 图像分类 | 25.6M 参数级 | 规整、成熟、容易 bring-up，是最好的 vision 基线之一 |
| `BERT-Large` | encoder-only Transformer 预训练 | 340M 参数级 | 比 Llama 更早、更稳，是传统 NLP 训练基线 |
| `3D U-Net` | 3D 医学分割 | 医学体数据，kernel 形态与 2D vision 不同 | 能覆盖 3D 卷积 / 大张量 / 医学影像链路 |
| `Mask R-CNN` / `RetinaNet` | 目标检测 | 多 stage / 多 family | 适合检验 “family / regime” 是否真能分清 backbone 与 head |
| `DLRM` | 推荐系统 | embedding 主导 | 对 memory-side tuning 很有代表性 |
| `RNN-T` | 语音识别 | 序列建模 | 现在不如 LLM 主流，但能补一条不同于 Transformer 的时序链 |

这些 workload 的价值不在于“最前沿”，而在于：

- 社区已有大量基线；
- 结构相对成熟；
- 比 `405B`、`671B` 级对象更适合先做 trace-to-sim bring-up。

---

## 4. 数据和模型规模应如何理解

下面把几个最容易被问到的“大小”信号单独列一下。

| 对象 | 可直接引用的规模信号 | 备注 |
|---|---|---|
| `llama3.1_405b` | 405B 参数 | 这是今天公开 benchmark 中最典型的超大 dense LLM 之一 |
| `deepseekv3` | 671B total / 37B active | 说明 MoE workload 不能只看总参数，还要看 active parameters |
| `llama3.1_8b` | 8B 参数 | 如果真要上 LLM 训练 trace，这是比 70B/405B 更现实的入口 |
| `retinanet` | 37.7M 参数 | 相比 LLM 很小，但 kernel family 很杂，不是“简单 workload” |
| `dlrm_dcnv2` | `3.5TB` 数据集级信号 | 它的难点主要不是参数量，而是 embedding / memory / communication |
| `flux.1` | 11.9B 参数 | 生成模型已经不再是“小模型” |
| `C4` | 364,868,892 rows（HF dataset card） | 这是典型 web-scale 预训练语料规模 |
| `GovReport` | 19,466 reports | 数量不夸张，但属于长文档 summarization，sequence/context 压力高 |
| `OpenImages` | 约 9M images（官方描述） | vision benchmark 里非常常见的大规模检测数据集 |
| `CC12M` | 12M image-text pairs | diffusion / text-to-image 常见公开起点 |
| `IGBH-Full` | graph 节点 / 边规模远大于传统小图基准 | 对 GNN 仿真最关键的是不规则性，而不是只看参数量 |

---

## 5. 用本仓库数据估算“真的跑训练 benchmark 要多久”

### 5.1 本地已知事实

本仓库已经有两组可以直接拿来做预算的本地事实：

1. `SM120_RTX5090` 草案配置中的核心频率写的是 `2580 MHz`；
2. `docs/trace-benchmark-2026-04-03.md` 中，受控 `10k cycle` 窗口下的模拟速度大约在 `588` 到 `10000 cycle/s` 之间。

这意味着当前链路的大致换算是：

```text
仿真时间 ≈ 真实执行时间 × 2.58e9 / sim_cycle_per_sec
```

### 5.2 关键换算表

| 真实 GPU 执行时间 | 若仿真速度 = 10000 cycle/s | 若仿真速度 = 1000 cycle/s | 若仿真速度 = 588 cycle/s |
|---|---:|---:|---:|
| 1 秒 | 2.99 天 | 29.86 天 | 50.78 天 |
| 10 秒 | 29.86 天 | 0.82 年 | 1.39 年 |
| 1 分钟 | 0.49 年 | 4.9 年 | 8.3 年 |
| 10 分钟 | 4.9 年 | 49.1 年 | 83.5 年 |

### 5.3 这张表的真正含义

这张表还只是**单卡真实执行时间**到**单卡精确周期仿真时间**的换算。

它还没有把下面这些额外成本算进去：

- trace 导出时间；
- trace 存储体积；
- 多 GPU 通信建模；
- 长 benchmark 的 phase 切换与 kernel invocation heterogeneity；
- 失败重跑；
- 参数 sweep。

所以如果问题是：

> “我们能不能把一个真实训练 benchmark 从头到尾完整 trace，再拿精确周期模型完整模拟？”

那答案在当前阶段基本是：

**不能把它当常规实验流。**

更现实的理解是：

- `1~10 秒` 的真实 GPU 执行窗口，已经是需要非常谨慎选取的对象；
- `1 分钟` 级真实执行窗口，通常已经进入“年级别仿真时间”的量级；
- `10 分钟` 级真实执行窗口，对当前精确周期链路基本不可接受。

---

## 6. 把这些预算映射回具体 workload

下面不是“精确值”，而是**按 workload 结构做的工程级估算**。

| Workload | 如果你想做什么 | 现实可行性 | 原因 |
|---|---|---|---|
| `retinanet` | 抓一个短 phase，做 trace-to-sim bring-up | 高 | 模型不算大，但 family 丰富，适合作为“真实但还可控”的 vision benchmark |
| `dlrm_dcnv2` | 抓 embedding-heavy phase，验证 memory-side tuning | 中高 | 不规则 memory 很有研究价值，但 full benchmark 不适合完整周期模拟 |
| `BERT-Large` | 做传统 Transformer 训练基线 | 中高 | 比 Llama 更容易控制，仍能覆盖 GEMM/softmax/layernorm 主链 |
| `llama3.1_8b` | 抓单 step 或单 layer phase | 中 | 可以做代表 phase，但别把完整 benchmark 当目标 |
| `llama2_70b_lora` | 抓 LoRA fine-tune window | 中 | 比 full pretraining 短，但 context/attention 依然重 |
| `flux.1` | 做完整训练 benchmark | 低 | 生成模型链长，phase 杂，full trace 与 full-cycle 成本都高 |
| `rgat` | 做完整 benchmark | 低 | GNN 的不规则访问会显著削弱压缩和代表化效果 |
| `llama3.1_405b` / `deepseekv3` | 完整 benchmark 级精确周期模拟 | 极低 | 参数、通信、heterogeneity、kernel 数量都过高，只能做 sampling / representative phases |

---

## 7. 如果你现在真要选一个“训练 benchmark 起点”

按“最有研究价值”和“最不容易一上来就把链路拖死”之间的折中，我建议优先级如下：

1. `retinanet`
2. `dlrm_dcnv2`
3. `BERT-Large`（如果你愿意先走经典 benchmark）
4. `llama3.1_8b` 的单 step / 单层 phase

不建议一开始就把下面这些对象当成完整端到端精确周期目标：

- `llama3.1_405b`
- `deepseekv3`
- `flux.1`
- `rgat` full benchmark

更稳的计划应该是：

1. 先选一个真实 benchmark；
2. 只抓 `1~3` 个代表 phase；
3. 每个 phase 只保留短窗口 trace；
4. 用 `family / regime / importance ratio` 决定哪些对象真的进入后端精确模拟；
5. 把完整 benchmark 留给前端 compression / writeback，而不是留给 full-cycle。

---

## 8. 对你当前问题的直接回答

如果你问的是：

> “我们是不是已经可以开始跑训练 benchmark 的调参了？”

那么更准确的回答应该是：

- **可以开始做训练 workload 的代表 phase 选取和小窗口调参；**
- **还不能把完整训练 benchmark 的精确周期仿真当成常规 tuning 阶段。**

换句话说，当前更像是：

`进入 workload-aware tuning preparation`

而不是：

`已经进入 full benchmark exact-cycle tuning`

---

## 9. 参考来源

### 本仓库本地材料

- `docs/trace-benchmark-2026-04-03.md`
- `docs/5090-trace-to-sim.md`
- `docs/pka-to-family-interface-design-2026-04-20.md`
- `simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM120_RTX5090/gpgpusim.config`

### 外部公开来源

- MLCommons Training repository  
  https://github.com/mlcommons/training
- MLCommons Training v5.1 benchmark updates  
  https://github.com/mlcommons/training/releases
- C4 dataset card  
  https://huggingface.co/datasets/allenai/c4
- GovReport dataset page  
  https://gov-report-data.github.io/
- Open Images dataset  
  https://storage.googleapis.com/openimages/web/index.html
- Conceptual 12M paper / dataset description  
  https://arxiv.org/abs/2102.08981
- Illinois Graph Benchmark paper  
  https://arxiv.org/abs/2302.13522
- PKA (MICRO 2021)  
  https://engineering.purdue.edu/tgrogers/papers/baddouh.micro2021.pdf
- Accel-Sim (ISCA 2020)  
  https://engineering.purdue.edu/tgrogers/papers/khairy.isca2020.pdf
- STEM+ROOT (MICRO 2025)  
  https://ejchung0406.github.io/assets/pdf/STEM_micro25.pdf
