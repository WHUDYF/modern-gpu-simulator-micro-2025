# Trace Compression Engineering Bottleneck Map Design

> **工程线定位：** 本 spec 只定义“先验证瓶颈在哪里”的研究方法，不实现在线压缩系统，也不修改 L1 selector 或学术线的 behavior signature 逻辑。

**Goal:** 通过主流 GPU microbench / benchmark 套件建立一张可复用的 bottleneck map，判断当前工作应优先处理 trace 生成、trace 导出/读取、trace 解码、simulator replay，还是 benchmark 选择本身。

**Architecture:** 先把工业界和学术界常用的 GPU microbench 按行为类别分层，再用现有本地 trace / simulation 结果校准一个 cost table。表中每个 benchmark 都要同时记录 native runtime、trace 体积、trace 导出耗时、固定窗口模拟耗时和主要瓶颈标签。最后用这张表决定下一步是做 streaming trace compression、模拟器加速，还是先收缩 benchmark corpus。

**Tech Stack:** 现有 `trace-benchmark-2026-04-03.md` 数据、`experiments/baseline_diagnosis/results/*` 结果、Nsight Systems / NVBit / GPGPU-Sim、公开 benchmark suites（BabelStream, nvbandwidth, nvbench, CUTLASS profiler, Rodinia, Parboil, PolyBench/GPU）。

---

## 1. 这条线要解决什么

这条工程线不是先做 compression engine，而是先回答：

> 对我们当前的 GPU trace / simulator pipeline 来说，最先卡住的到底是哪一段？

如果不先回答这个问题，就会出现两种误判：

1. 把 `trace export`、`trace I/O`、`simulator replay`、`benchmark sweep` 混成一个“都很慢”的印象；
2. 直接进入实现，却不知道该优先优化哪一层，最后做成局部最优。

这条线的输出不是论文方法本体，而是一张**成本地图**：

```text
benchmark family
  -> native runtime
  -> trace size
  -> trace export time
  -> simulator window cost
  -> bottleneck label
```

这张表的作用是反推：

- 现在先优化 trace 生成还是 trace 压缩；
- 先做 simulator throughput 还是 trace format redesign；
- 哪些 benchmark 应该保留，哪些只该放 appendix；
- 多 GPU communication 是否应该独立成另一条线。

---

## 2. 非目标

这份 spec 不做下面这些事：

- 不实现 streaming compression 系统
- 不实现新的 parser 或 replay engine
- 不替代学术线的 behavior signature
- 不把 NCCL / OSU 这类通信微基准硬塞进当前单 GPU trace 表
- 不要求第一轮就给出全套精确运行时长

这里的重点是**判定瓶颈位置**，不是完成压缩系统本体。

---

## 3. 研究对象分层

### 3.1 主表对象：single-GPU microbench

主表只收对当前单 GPU trace-driven simulator 直接可用的 benchmark / microbench。

推荐分四类：

1. `memory bandwidth`
2. `memory latency / atomic`
3. `dense compute / GEMM`
4. `irregular / mixed control`

### 3.2 附录对象：multi-GPU communication

下面这些对象不进入当前主表，只作为 appendix：

- NCCL-tests
- OSU Micro-Benchmarks

原因很直接：

- 它们主要测试多 GPU / 通信链路；
- 当前主路径是单 GPU trace-driven pipeline；
- 把它们并入主表会污染瓶颈判断。

### 3.3 上界对象：full workload anchors

MLPerf Inference / Training 不作为 microbench 主表对象，但它们是上界锚点。

用途：

- 说明真实 workload 何时已经从“微基准选择问题”升级为“全 workload 不可直接模拟问题”；
- 为为什么要做 representative compression 提供上界背景。

---

## 4. 成本模型

### 4.1 总时间分解

我们把一次 benchmark 的端到端成本分成五段：

```text
T_total = T_native + T_capture + T_export + T_read + T_sim
```

其中：

- `T_native`：benchmark 在真实 GPU 上运行的原生时间
- `T_capture`：tracer / profiler 收集成本
- `T_export`：trace 落盘、序列化、目录组织、压缩或导出时间
- `T_read`：后续读取 trace 的时间
- `T_sim`：在 simulator 里 replay 的时间

### 4.2 当前阶段的主判据

本工程线第一阶段不追求 `T_total` 的绝对精度，而追求下面两个比较：

```text
T_export + T_read   vs   T_sim
T_native            vs   T_capture + T_export
```

这两个比较足够回答：

- 是 trace I/O 卡住了，还是 simulator 卡住了；
- 是 benchmark 本身太大，还是 tracing 结构太重；
- 是单个 kernel 慢，还是 sweep 数量一多就爆炸。

### 4.3 估计等级

因为并不是所有候选 benchmark 都已经在当前仓库里跑过，所以表格分成三种状态：

- `measured`：已有本地数据
- `estimated`：由结构相似度和公开资料推断
- `excluded`：当前不进入主表

---

## 5. 主表的字段

每一行至少包含下面这些字段：

| 字段 | 含义 |
|---|---|
| `suite` | benchmark 套件名 |
| `representative case` | 代表性 kernel / case |
| `category` | bandwidth / latency / compute / irregular |
| `native runtime class` | 原生运行时间级别 |
| `trace size class` | trace 体积级别 |
| `export cost class` | trace 导出耗时级别 |
| `simulator cost class` | simulator replay 耗时级别 |
| `dominant bottleneck` | 当前最可能的瓶颈 |
| `status` | measured / estimated / excluded |
| `evidence` | 数据或来源说明 |

建议的成本等级采用下面的离散分级：

- `sub-second`
- `seconds`
- `tens of seconds`
- `minutes`
- `hours`
- `infeasible`

---

## 6. 初始成本表

下面是工程线第一版建议使用的初始表。这个表不是最终结论，而是第一轮瓶颈地图的起点。

### 6.1 已有本地测量锚点

这里的 `sim proxy` 指的是现有 trace-benchmark 文档里的固定窗口模拟时间，主要是 10k-cycle 级别的 proxy，不是完整 workload 的全程 wall-clock。

| suite | representative case | category | trace size | export time | sim proxy (10k-cycle window) | dominant bottleneck | status |
|---|---|---|---:|---:|---:|---|---|
| `GPU_Microbenchmark` | `l1_bw_32f` | bandwidth | 2.962 MiB | 2.24 s | 1.46 s | balanced / small trace | measured |
| `GPU_Microbenchmark` | `mem_bw` | bandwidth | 47.975 MiB | 8.43 s | 14.92 s | simulator throughput | measured |
| `GPU_Microbenchmark` | `l2_bw_32f` | bandwidth | 568.192 MiB | 69.41 s | about 17 s | trace export / I/O | measured |
| `GPU_Microbenchmark` | `shared_bw` | bandwidth | 123.217 MiB | 34.55 s | 4.89 s | trace export / I/O | measured |
| `GPU_Microbenchmark` | `atomic_add_bw` | atomic / bandwidth | 5.423 MiB | 2.27 s | 10.13 s | simulator throughput | measured |
| `GPU_Microbenchmark` | `atomic_add_lat` | latency | 0.288 MiB | 2.11 s | 1.26 s | capture / fixed overhead | measured |
| `GPU_Microbenchmark` | `shared_lat` | latency | 0.340 MiB | 2.18 s | 1.29 s | capture / fixed overhead | measured |
| `GPU_Microbenchmark` | `MaxFlops` | compute | 8.715 MiB | 3.52 s | 1.53 s | balanced / compute light | measured |

### 6.2 当前 Rodinia / benchmark 结果的校准信号

| workload | representative kernel | trace / sim signal | implication |
|---|---|---|---|
| Rodinia | `backprop` | simulator logs show ~7-10 s windows for the relevant kernels | mixed control + compute, not purely bandwidth-bound |
| Rodinia | `nn` | 1-segment / uniform launch structure | launch regularity can hide interesting bottlenecks |
| Parboil | `sgemm`, `stencil`, `bfs` | local APE tables show these are standard benchmark kernels used in evaluation | good for mixed dense/irregular coverage |
| PolyBench/GPU | `gemm`, `3mm`, `3DConvolution`, `atax`, `bicg` | local APE tables show a stable regular-kernel family | good for regular dense coverage |

### 6.3 外部主流 microbench 套件的初始估计

| suite | representative case | category | estimated simulator cost class | dominant bottleneck | status | evidence |
|---|---|---|---|---|---|---|
| `BabelStream` | `copy/scale/add/triad/dot` | bandwidth | `seconds` per kernel, `tens of seconds` for larger sweeps | trace export or simulator depending on array size | estimated | official repo says it measures GPU memory transfer rates and omits PCIe transfer time |
| `nvbandwidth` | memcpy / link bandwidth patterns | bandwidth / link copy | `seconds` per pattern | export / I/O for large sweeps; communication path for multi-link modes | estimated | official repo says it measures bandwidth across links using copy engine or kernel copy methods; treat legacy `cuda-samples bandwidthTest` as a secondary reference |
| `nvbench` | runtime/compile-time parameter sweeps | generic kernel benchmark | `seconds` per configuration; `minutes+` for sweeps | benchmark sweep explosion | estimated | official repo says it benchmarks a single host-side critical region and supports parameter sweeps |
| `CUTLASS profiler` | GEMM / convolution configs | dense compute | `seconds` per config; `minutes+` across sweep grids | simulator throughput for large dense kernels; sweep explosion for parameter grids | estimated | official CUTLASS profiler docs show explicit kernel profiling and tuning sweeps |
| `Rodinia` | `nn`, `backprop`, `bfs`, `lud`, `nw` | irregular / mixed | `seconds` to `tens of seconds` per kernel window | mixed control / trace depth | measured/estimated | local trace + sim logs already exist |
| `Parboil` | `sgemm`, `stencil`, `cutcp`, `mri-q`, `histo`, `bfs` | mixed dense / irregular | `seconds` to `tens of seconds` | trace export + irregularity | estimated | local APE and validation docs exist |
| `PolyBench/GPU` | `gemm`, `3mm`, `3DConvolution`, `atax`, `bicg`, `syrk` | dense / regular | `seconds` | simulator throughput for compute-heavy configs | estimated | local APE tables show stable regular kernels |
| `NCCL-tests` | `all_reduce_perf` and friends | multi-GPU communication | `excluded` from current main table | different problem class | excluded | official repo is for correctness/performance of NCCL ops |
| `OSU micro-benchmarks` | MPI microbench suite | communication / network | `excluded` from current main table | different problem class | excluded | official repo is MPI microbenchmarks |
| `MLPerf Inference / Training` | BERT, ResNet, DLRM, Llama2, Mixtral | full workload anchor | `hours` to `infeasible` in full trace-driven simulation | full workload scale / trace explosion | appendix only | official MLCommons docs describe these as benchmark suites, not microbench |

### 6.4 What this table already tells us

从当前本地测量看，瓶颈并不统一：

- 小而规整的 kernel 里，`T_export` 和 `T_sim` 都很短，成本主要在固定开销；
- 中等规模 bandwidth kernel 里，`T_sim` 常常先上来；
- 大 trace 的 kernel 里，`T_export` 已经明显压过 `T_sim`；
- 这意味着**trace I/O / format / export** 已经是第一轮必须认真看的瓶颈，而不是只盯 simulator 本体。

---

## 7. 如何填满这张表

### 7.1 第一轮数据收集

优先对下列套件做统一采样：

- `BabelStream`
- `nvbandwidth`
- `nvbench`
- `CUTLASS profiler`
- `Rodinia`
- `Parboil`
- `PolyBench/GPU`

### 7.2 每个 benchmark 至少采这些量

- native wall time
- trace size
- trace export time
- `10k-cycle` simulator proxy time
- dynamic instruction count
- kernel / invocation count
- trace file count

### 7.3 估计规则

如果某个 suite 还没有本地测量数据，则按下面的顺序估计：

1. 找结构最接近的本地锚点；
2. 参考 trace size 和 file count 判断 export 负担；
3. 参考 dynamic instruction count 和 launch 数量判断 simulator 负担；
4. 给出成本等级和 confidence，不伪装成精确秒数。

---

## 8. 瓶颈判定规则

### 8.1 如果 export / I/O 先爆

如果大多数主表 benchmark 都满足：

```text
T_export + T_read >= T_sim
```

那么当前第一优先级不是 simulator 逻辑，而是：

- trace format redesign
- streaming compression
- file count reduction
- read path simplification

### 8.2 如果 simulator 先爆

如果大多数 benchmark 都满足：

```text
T_sim >> T_export + T_read
```

那么第一优先级是：

- simulator throughput
- replay 结构优化
- representative compression

### 8.3 如果 benchmark sweep 先爆

如果单个 kernel 还可以，但一旦参数 sweep / family sweep 就超预算，那么第一优先级是：

- benchmark selection
- family pruning
- knob 设计收敛

这时问题不是 simulator 本体，而是我们选的实验空间太大。

### 8.4 如果 multi-GPU 才是主需求

如果最终发现我们真正要研究的是 NCCL / MPI / collective path，那就应把它拆成独立工程线，不要和当前单 GPU trace pipeline 混在一起。

---

## 9. 与现有工作线的关系

### 9.1 和 L1 的关系

L1 负责代表对象选取。

工程线负责回答：

- 这些代表对象到底会把 pipeline 压到哪一层；
- 哪些对象适合继续留在主表；
- 哪些对象应该只作为 appendix；
- 当前的瓶颈是不是已经在 trace 读写层，而不是在 selector 或 simulator 逻辑层。

### 9.2 和学术线的关系

学术线负责把 trace 压缩结构当作 behavior signature。

工程线负责判断：

- 这种结构如果只是为了减文件，是不是还不够；
- streaming / format / IO 有没有足够大的工程价值；
- 这个方向是否值得单独做成系统工作。

这两条线共用 trace 相关底座，但评价目标不同。

---

## 10. 成功标准

这条工程线成立的最低标准是：

1. 形成一张可复用的 benchmark cost map；
2. 这张表能明确区分 trace export / I/O / decode / simulator 的相对压力；
3. 能把主流 microbench 分成主表、附录和上界锚点三层；
4. 能据此决定下一步先做 streaming compression、simulator 加速，还是 benchmark 收敛；
5. 不把 multi-GPU communication 和单 GPU trace-driven pipeline 混为一谈。

如果最后只得到一堆 benchmark 名字，而没有瓶颈判断，这条线就没有完成。

---

## 11. 现有证据来源

### 本地材料

- [trace-benchmark-2026-04-03.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/trace-benchmark-2026-04-03.md:1)
- [microbenchmark-runtime-related-work-2026-04-26.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/microbenchmark-runtime-related-work-2026-04-26.md:1)
- [a-line-kernel-validation-dataset-recommendation-2026-04-26.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/a-line-kernel-validation-dataset-recommendation-2026-04-26.md:1)
- [trace-compression-microbench-related-work-2026-04-28.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/trace-compression-microbench-related-work-2026-04-28.md:1)

### 公开项目

- [BabelStream](https://github.com/UoB-HPC/BabelStream)
- [nvbandwidth](https://github.com/NVIDIA/nvbandwidth)
- [nvbench](https://github.com/NVIDIA/nvbench)
- [CUTLASS performance profiling](https://github.com/NVIDIA/cutlass/wiki/Performance-Profiling)
- [cuda-samples](https://github.com/NVIDIA/cuda-samples)
- [NCCL-tests](https://github.com/NVIDIA/nccl-tests)
- [OSU micro-benchmarks](https://github.com/forresti/osu-micro-benchmarks)
- [MLPerf Inference docs](https://docs.mlcommons.org/inference/index_gh/)
- [MLPerf Inference working group](https://mlcommons.org/working-groups/benchmarks/inference/)
