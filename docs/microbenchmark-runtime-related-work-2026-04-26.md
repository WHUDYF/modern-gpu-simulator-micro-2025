# Microbenchmark 运行时间与相关工作梳理

日期：2026-04-26

## 1. 直接结论

针对你现在的问题，结论可以先压缩成三句话：

1. **有相关工作明确测了 workload / benchmark 的运行时间、profiling 时间和 simulation 时间。**
2. **但我暂时没有看到公开论文把“从一组 microbenchmark 中，按运行时长预算 + 代表性 + 可模拟性自动筛选”做成标准方法。**
3. 因此，你提出的脚本方向是合理的，而且更像是在补一个“工程上非常需要、但文献里还没被标准化”的空位。

---

## 2. 哪些工作真的测了“时间”

这里先区分三种“时间”，因为论文里经常混用：

- **硬件原生运行时间**：benchmark 在真实 GPU 上跑多久。
- **profiling / tracing 时间**：采集 counters、BBV、trace、feature 要多久。
- **simulation 时间**：cycle-accurate / trace-driven simulator 跑多久。

对你后面要写的脚本来说，这三种时间都重要，但优先级不同：

- 如果你是要先筛选“适合 tracing 的 microbenchmark”，最先看的是 **硬件运行时间** 和 **trace 导出时间**。
- 如果你是要先筛选“适合后端精确模拟的对象”，最先看的是 **simulation 时间** 和 **trace 大小**。

---

## 3. 与你最相关的论文表

| 工作 | 年份 / 会议 | 主要对象 | 是否显式测时间 | 你最该记住的结论 | 对你脚本的启发 |
|---|---|---|---|---|---|
| `TBPoint: Reducing Simulation Time for Large-Scale GPGPU Kernels` | 2014 / IPDPS | 大 kernel 采样 | 是 | 直接比较了真实 GPU 时间和周期模拟时间；例如 `NB` 在 GPU 上约 `28557 ms`，模拟约 `3.78 weeks` | 说明“先看原生 runtime 再决定是否模拟”是合理的，而且很早就有需求 |
| `Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling` | 2020 / ISCA | 现代 NVIDIA GPU validated sim | 部分是 | 通过 targeted microbenchmark + tuner + correlator 建 baseline；microbenchmark 被明确设计成“kernel execution time dominated by target effect” | 说明 microbenchmark 不该只看短不短，还要看“机制纯度” |
| `Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator` | 2021 / HPCA | NVIDIA 内部 NVArchSim | 是，但偏系统级 | 明确提出 overly precise / overly slow model 会伤害架构师生产率；NVAS 用更高抽象层支撑 hundreds of workloads | 说明“不是所有东西都值得进精确周期模型”是工业界共识 |
| `Principal Kernel Analysis (PKA)` | 2021 / MICRO | 147 workloads + 7 MLPerf | 是 | MLPerf realistic workloads 在硬件上是秒级到分钟级，在 Accel-Sim 上可到 centuries；`BERT` offline inference 约 `10 minutes` in silicon | 说明 full workload 先压缩再谈模拟，不然根本不 tractable |
| `Sieve: Stratified GPU-Compute Workload Sampling` | 2023 / ISPASS | Cactus + MLPerf | 是 | full-workload PKS profiling 对某些 workload 估计要 `>3 weeks`；Sieve profiling 平均 `8×`、最高 `98×` 更快 | 说明 profiling 开销本身就可以成为筛选约束 |
| `Photon: A Fine-grained Sampled Simulation Methodology for GPU Workloads` | 2023 / MICRO | VGG / ResNet / PageRank 等 | 是 | `ResNet-152` batch 1 一次 inference 从 `7.05 days` 降到 `1.7 hours`；`VGG-16` 完整 workload 详细模拟约 `3.44 days` | 说明真实 DNN workload 的 full detailed sim 非常慢，时间预算必须进入选样逻辑 |
| `STEM+ROOT` | 2025 / MICRO | Rodinia + CASIO + HuggingFace | 是 | prior methods 的 profiling / processing overhead 可到 `78.68 days`；对 `GPT-2`，Photon 的 BBV 处理会遇到 `>50 million kernel invocations` 的复杂度问题 | 说明对大模型 workload，kernel invocation 数量本身就是一级筛选指标 |
| `Parallelizing a modern GPU simulator` | 2025 / CAMS / arXiv | Accel-Sim 并行化 | 是 | 单线程 Accel-Sim 跑某些 workload 可 `>5 days`；16 threads 平均 `5.8×`、最高 `14×` 加速 | 说明哪怕后端并行化了，时间预算仍然是硬约束 |

---

## 4. 每篇工作里，哪些“时间数据”最有价值

### 4.1 TBPoint（2014）

这是我看到的最直接把“真实 GPU 时间”和“模拟时间”并排摆出来的早期工作之一。

公开摘要和 PDF snippet 里给出的典型数据包括：

- `NB`: GPU `28557 ms`，simulation `3.78 weeks`
- `SP`: GPU `18779 ms`，simulation `2.48 weeks`
- `SSSP`: GPU `7067 ms`，simulation `6.54 days`
- `MM`: GPU `881 ms`，simulation `19.58 hours`

这篇工作的价值不在于它方法今天仍是最强，而在于它非常直接地说明：

**只要 kernel 稍微大一点，cycle-level simulation 时间就会立刻跨到“天/周”量级。**

对你现在的脚本最有启发的一点是：

**真实 GPU runtime 完全可以作为前置筛选条件。**

如果某个 microbenchmark 在硬件上已经长到秒级甚至十秒级，那它大概率不适合作为第一批 exact-cycle bring-up 对象。

---

### 4.2 Accel-Sim（2020）

Accel-Sim 不是在做“runtime budget 筛选器”，但它非常重要，因为它定义了现代 NVIDIA validated simulation 的主 baseline。

它和你问题最相关的点有两个：

1. 它明确用了一整套 **targeted microbenchmark suite** 去做参数调谐。
2. 它强调 microbenchmark 要设计成 **execution time dominated by target effect**。

也就是说，Accel-Sim 给你的启发不是“挑最短的 benchmark”，而是：

**挑那些既短、又纯、又能隔离目标机制的 benchmark。**

这对你后面写脚本很关键，因为脚本不能只按 runtime 排序，否则会偏向 trivial benchmark。

---

### 4.3 PKA（2021）

PKA 是这条线上的核心分水岭，因为它把问题明确升级成了：

**现代 realistic workload 太大，必须先压缩。**

它给出的几个非常重要的时间事实：

- realistic MLPerf workloads 在真实硬件上是 **several seconds**；
- 但 full simulation 可以到 **centuries**；
- `MLPerf SSD Training` 可以有 `5.3 million` kernel instances；
- `BERT` offline inference 在硬件上约 `10 minutes`。

它还给出总判断：

- 对 147 workloads，PKA 把 MLPerf 的 centuries-long simulation 压到 hours；
- 平均 cycle error 大约 `27%`。

这篇工作对你的启发是：

**不要假设“只要 workload 被压缩了，就已经进入 tuning 阶段”。**

更合理的理解是：

`runtime budget filter -> representative compression -> sampled simulation`

而不是：

`runtime budget filter -> exact full tuning`

---

### 4.4 Sieve（2023）

Sieve 和你的问题非常接近，因为它开始把 **profiling time** 当成一等问题。

它给出的关键信号包括：

- 对 long-running workloads，profiling 可能需要 **multiple days**，甚至 **several weeks**；
- 对 Cactus / MLPerf，full-workload PKS profiling 估计要 **more than three weeks**；
- Sieve 相比 PKS，profiling time 平均 `8×`、最高 `98×` 更快；
- 它列出了 MLPerf workload 的 kernel invocation 数：
  - `3d-unet`: `113,183`
  - `bert`: `141,964`
  - `resnet50`: `78,825`
  - `rnnt`: `205,440`
  - `ssd-mobilenet`: `64,138`
  - `ssd-resnet34`: `57,267`

这篇工作对你脚本的启发特别直接：

**除了 runtime，还应该把 `#kernel invocations` 放进筛选条件。**

因为 invocation 数量高，会同时推高：

- profiling 时间；
- trace 组织成本；
- 后端样本选择复杂度。

---

### 4.5 Photon（2023）

Photon 的价值在于，它给出了更贴近 DNN 的 wall-time 例子。

文中明确写到：

- `VGG-16` 完整 workload 详细模拟需要约 `3.44 days`
- `ResNet-152` batch size 1 一次 inference，从 `7.05 days` 降到 `1.7 hours`
- 对 `ResNet-152` 的 sampling error 约 `10.7%`

这说明：

**即便不是训练，而只是 inference，真实 DNN workload 的 full detailed sim 也已经很重。**

所以如果你想写一个“先挑一批适合测的 microbenchmark”的脚本，Photon 给你的启发是：

- 要优先找 **单 kernel 或少 kernel family** 的对象；
- 要避免一开始就抓多层深网的完整执行。

---

### 4.6 STEM+ROOT（2025）

这是目前最值得你认真看的前沿工作之一，因为它把问题推进到了：

**现代大 workload 不只是大，而且 invocation-level heterogeneity 很强。**

文中对时间开销的表述非常有力：

- prior methods 的 profiling / processing overhead 可到 `78.68 days`
- 对 HuggingFace workloads，这样的 overhead 是按 **per workload** 算的
- Photon 的 BBV 处理在 `GPT-2` 上会变得 infeasible，因为有 `over 50 million kernel invocations`
- 在 full workload simulation infeasible 时，他们直接退回用 machine profile cycle counts 估计 speedup / error

这篇工作的意义是：

**它已经明确承认：对现代 LLM workload，full sim 经常根本跑不动。**

对你脚本最直接的启发是：

除了 runtime，还要显式考虑：

- `#kernel invocations`
- invocation heterogeneity
- 处理流程本身的复杂度

---

### 4.7 Parallelizing a modern GPU simulator（2025）

这篇不是 sampling paper，但对现实预算很有价值。

它的核心信息很直接：

- 单线程 Accel-Sim 跑一些 GPGPU workloads 可 `>5 days`
- 16 线程平均 `5.8×` 加速，最高 `14×`
- 五天级 workload 可以压到 `<12 hours`

这说明一个很重要的现实：

**哪怕你后面把 simulator 并行化，时间预算仍然是必须前置建模的。**

也就是说，写脚本筛 microbenchmark 这件事，不会因为后端并行化而失去意义。

---

## 5. 我没有看到什么

我目前**没有看到**下面这种公开标准做法：

> 从一个已有 microbenchmark 池中，自动根据
> `硬件 runtime + trace size + 代表性 + 机制纯度 + 可模拟性`
> 选出一组最适合 trace-to-sim bring-up 的 benchmark。

现有工作更常见的做法是：

- 先拿一个 workload 跑起来；
- 再根据 profiling / feature / clustering / BBV / kernel similarity 做抽样；
- 最后减少 simulation cost。

也就是说，它们更多是在解决：

**“full workload 已经给定后，怎么压缩”**

而不是解决：

**“在进入 tracing 之前，先从候选 microbenchmark 池里挑哪些最值得跑”**

这正是你脚本可能补的位置。

---

## 6. 这对你后面脚本设计的直接启发

如果你后面真要写一个 benchmark 选择脚本，我建议至少收下面这些字段：

| 字段 | 为什么需要 |
|---|---|
| `native_runtime_ms` | 最基本的时间预算约束 |
| `trace_export_time_s` | tracing 开销本身可能比 workload runtime 更贵 |
| `trace_size_mb` | 决定存储成本和后端 IO 成本 |
| `sim_cycle_per_sec` 或短窗口仿真耗时 | 直接反映后端可模拟性 |
| `dynamic_inst_count` | 代表 workload 大小，比只看 runtime 更稳 |
| `kernel_count` / `invocation_count` | 代表前端复杂度 |
| `family_label` / `mechanism_label` | 保证不是全选同一类 benchmark |
| `stability_score` | 避免高度抖动 benchmark |
| `tail_effect` / partial wave 指标 | 避免因为 launch shape 造成假代表性 |
| `coverage_weight` | 保证选出来的一组 benchmark 真的覆盖目标 workload 空间 |

更进一步，一个合理的筛选目标可以写成：

```text
maximize(representativeness + family coverage + mechanism purity)
subject to
  native_runtime <= T1
  trace_export_time <= T2
  trace_size <= T3
  predicted_sim_cost <= T4
```

这和现有文献最大的区别在于：

你是在 **pre-tracing / pre-simulation** 阶段做预算控制。

---

## 7. 对你当前问题的明确回答

如果把你的问题翻成一句话：

> “相关工作有没有认真看过 benchmark/microbenchmark 的运行时间，还是大家都只谈代表性？”

答案是：

- **有，而且很多工作都非常重视时间成本；**
- 但他们更常测的是 **simulation time** 和 **profiling overhead**；
- **直接做 microbenchmark runtime-aware preselection 的公开方法很少，我暂时没看到成熟标准。**

所以你的方向不是重复别人，而更像是在把：

`runtime budget`

正式变成前端 benchmark 选择器的一部分。

---

## 8. 推荐你优先读的论文顺序

如果你的目标是“为 runtime-aware benchmark selection 脚本找依据”，我建议阅读顺序是：

1. `Accel-Sim`  
   先建立 validated GPU sim + microbenchmark tuner 的基线概念。
2. `PKA`  
   建立“为什么 realistic workload 太大，必须压缩”的大前提。
3. `Sieve`  
   看 profiling 时间如何成为一等约束。
4. `Photon`  
   看真实 DNN workload 的详细 simulation 时间到底有多重。
5. `STEM+ROOT`  
   看现代 HuggingFace / LLM workload 为什么连 profiling 都会炸。
6. `Parallelizing a modern GPU simulator`  
   看后端并行化最多能缓解多少，而不是幻想能完全解决问题。

如果你只想先抓一个最小集合：

- `Accel-Sim`
- `PKA`
- `Sieve`
- `STEM+ROOT`

这四篇已经足够支撑你写第一版脚本设计。

---

## 9. 参考来源

- TBPoint: Reducing Simulation Time for Large-Scale GPGPU Kernels  
  https://hsienhsinlee.github.io/MARS/pub/ipdps14.pdf
- Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling  
  https://engineering.purdue.edu/tgrogers/papers/khairy.isca2020.pdf
- Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator  
  https://d1qx31qr3h6wln.cloudfront.net/publications/HPCA_2021_NVArchSim.pdf
- Principal Kernel Analysis: A Tractable Methodology to Simulate Scaled GPU Workloads  
  https://engineering.purdue.edu/tgrogers/papers/baddouh.micro2021.pdf
- Sieve: Stratified GPU-Compute Workload Sampling  
  https://users.elis.ugent.be/~leeckhou/papers/ispass-2023.pdf
- Photon: A Fine-grained Sampled Simulation Methodology for GPU Workloads  
  https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf
- Swift and Trustworthy Large-Scale GPU Simulation with Fine-Grained Error Modeling and Hierarchical Clustering  
  https://ejchung0406.github.io/assets/pdf/STEM_micro25.pdf
- Parallelizing a modern GPU simulator  
  https://arxiv.org/abs/2502.14691

### 本仓库相关材料

- `docs/trace-benchmark-2026-04-03.md`
- `docs/related-work-summary-2026-04-22.md`
- `docs/training-workload-survey-and-sim-budget-2026-04-26.md`
