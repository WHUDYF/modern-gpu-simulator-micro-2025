# A 线 Kernel 验证集推荐

日期：2026-04-26

## 1. 文档目的

这份文档用于回答一个当前已经非常关键的问题：

**如果 A 线要先复现 `PKA baseline`，并且希望后续压缩实验具有可验证性，那么前端压缩层应该优先使用什么样的 kernel 验证集。**

当前判断是：

- 只依赖 `mini_transformer_v4` 这类单一 workload 输入，数据过于原始；
- 只依赖少量 microbench，又会导致行为覆盖太窄；
- 因此 A 线需要建立一个：

**可验证、可扩展、足以支撑 compression experiment 的 kernel validation dataset。**

这份验证集的目标不是替代真实 workload，
而是作为 A 线的：

- correctness gate
- baseline reproduction gate
- feature audit gate
- regression gate

---

## 2. 推荐原则

当前 A 线的验证集不应只追求“更大”，而应同时满足下面四条：

### 2.1 行为轴清晰

数据集中应包含一批：

- memory-bound
- compute-bound
- divergence-sensitive
- atomic-heavy
- shared-memory-heavy

等主行为明确的 kernel。

### 2.2 样本规模足够

数据集不能只停留在 5~10 个 microbench 上，
否则无法真正验证 compression / clustering 的稳定性。

### 2.3 来源分层

数据集应同时覆盖：

- canonical microbench
- 真实 benchmark kernels
- 参数扫出的扩展 kernel

否则压缩实验只能说明一小段场景。

### 2.4 可回归

数据集中的对象应尽量来自：

- 公开 benchmark suite
- 可重复编译 / 运行的程序
- 能稳定采集 NCU 特征的 kernel

这样后续 A 线每次改 selector、feature extractor 或 schema，
都可以稳定回归。

---

## 3. 总体推荐：三层式验证集

当前最推荐的不是单一 benchmark，
而是一套三层式验证集：

1. `L0 canonical microbench layer`
2. `L1 benchmark kernel layer`
3. `L2 expansion layer`

这样可以兼顾：

- 行为可解释性
- 数据规模
- 压缩实验可信度

---

## 4. L0：Canonical Microbench Layer

### 4.1 目标

这一层用于验证：

- feature extraction 是否合理
- PKA 12 维信号是否能稳定反映主行为
- 同类 kernel 是否会在 behavior space 中靠近
- 异类 kernel 是否会被稳定分开

### 4.2 推荐来源

优先级从高到低：

1. 仓库现有 `GPU_Microbenchmark`
2. `SHOC`
3. `BabelStream`

### 4.3 推荐对象

建议至少覆盖下面这些 canonical 行为：

- `L1 bandwidth`
- `L2 bandwidth`
- `global memory latency`
- `shared memory bandwidth`
- `shared memory latency`
- `atomic throughput / latency`
- `max flops`

### 4.4 当前第一批建议对象

可以直接从现有仓库结果起步：

- `l1_bw_32f`
- `l1_bw_64f`
- `l2_bw_32f`
- `mem_bw`
- `mem_lat`
- `shared_bw`
- `shared_lat`
- `atomic_add_bw`
- `atomic_add_lat`
- `MaxFlops`

### 4.5 这一层的作用

这层不追求“真实 workload 代表性”，
而追求：

- feature sanity
- clustering sanity
- representative selection sanity

如果这一层都对不上，
A 线不应进入更大的 benchmark set。

---

## 5. L1：Benchmark Kernel Layer

### 5.1 目标

这一层用于验证：

- PKA baseline 在真实 benchmark kernel 上是否仍然稳定；
- compression 是否能在“非合成 kernel”上维持行为结构；
- representative selection 是否会因为噪声 metadata 跑偏。

### 5.2 最推荐的 benchmark 套件

当前最推荐优先组合为：

1. `Rodinia`
2. `Altis`
3. `Parboil`
4. `PolyBench/GPU`

### 5.3 推荐原因

#### Rodinia

优点：

- 经典、公开、社区熟悉；
- 行为类型丰富；
- 便于和既有 GPU simulation / sampled simulation 相关工作对话。

适合作为：

**A 线 benchmark kernel validation 的主干集合。**

#### Altis

优点：

- 比经典小套件更现代；
- 更偏 GPU 系统性能评测；
- 对今天的 GPU 环境更有说服力。

适合作为：

**Rodinia 之后的第一扩展集。**

#### Parboil

优点：

- 规模适中；
- 行为覆盖比单纯 microbench 更广；
- 适合作为额外 benchmark kernel 补充。

#### PolyBench/GPU

优点：

- 数值计算 kernel 较系统；
- 能为 dense / regular kernels 提供补点；
- 有助于形成更规则的 behavior subspace。

### 5.4 第一阶段推荐的最小 benchmark core

如果只选一小批，建议先从下面这些开始：

- Rodinia: `nn`, `backprop`, `bfs`, `lud`, `nw`
- Altis: 选 3~5 个能够稳定编译和 profile 的 kernel
- Parboil: 选 3~4 个行为差异明显的 kernel
- PolyBench/GPU: 选 3~4 个 dense / regular kernels

### 5.5 这一层的目标规模

建议第一阶段至少形成：

- `20 ~ 40` 个 benchmark kernels

如果进一步展开到 invocation 级或参数变体级，
可以自然扩展到：

- `100+` 个 compression objects

这已经足以支撑第一轮 compression experiment。

---

## 6. L2：Expansion Layer

### 6.1 目标

这一层用于解决一个现实问题：

**即使 benchmark kernels 已经不少，样本量仍可能不足以充分验证 compression space。**

因此需要一层能够“快速扩样本”的来源。

### 6.2 最推荐来源

当前推荐：

1. `CUTLASS Profiler`
2. `HeCBench`

### 6.3 CUTLASS Profiler 的角色

它最适合做的是：

- 参数 sweep
- 大量生成 dense compute kernels
- 快速得到成百上千个 kernel 样本

优点：

- 样本量增长快
- 参数可控
- 很适合做 compression stability test

缺点：

- 类型偏 dense linear algebra
- memory / irregular / atomic 行为覆盖不够广

因此它应被看作：

**扩样本层，而不是唯一主数据集。**

### 6.4 HeCBench 的角色

优点：

- 规模大
- 程序种类多
- 更适合做泛化测试

缺点：

- bring-up 成本更高
- 第一阶段不一定最划算

因此更适合作为：

**第二阶段的大规模扩展验证集。**

---

## 7. 不同数据源的推荐优先级

### 7.1 第一阶段必须集

这是当前最推荐的第一阶段组合：

- 仓库现有 `GPU_Microbenchmark`
- `Rodinia`
- `Altis`

理由：

- 行为轴清晰
- benchmark 真实性足够
- 带来的 bring-up 成本可控

### 7.2 第一阶段可选补充

- `Parboil`
- `PolyBench/GPU`

它们用于补：

- regular numeric kernels
- 中等规模 benchmark kernels

### 7.3 第二阶段扩展集

- `CUTLASS Profiler`
- `HeCBench`

这两者主要用于：

- 扩大样本规模
- 做泛化测试
- 做 compression robustness test

---

## 8. 第一版推荐数据集组合

如果当前就要定义一套足以进行 compression experiment 的验证集，
我建议如下组合：

### Tier A：Canonical kernels

从现有 microbench 中选择：

- `10 ~ 12` 个

代表：

- bandwidth
- latency
- shared
- atomic
- compute

### Tier B：Core benchmark kernels

从：

- `Rodinia`
- `Altis`
- `Parboil`
- `PolyBench/GPU`

中合计选出：

- `20 ~ 30` 个 kernel

### Tier C：Target AI kernels

保留你们自己的：

- `mini_transformer_v4`
- 后续其他 AI workload 中的 representative kernels

作为最终 target layer。

### 合计目标

第一版推荐目标规模为：

- `30 ~ 50` 个可验证 kernel 对象

如果再加入参数变体或 invocation variants，
很容易扩展到：

- `100 ~ 300` 个 compression objects

这个规模已经足以做第一轮 A 线压缩实验。

---

## 9. 为什么不建议第一步直接上 HeCBench 全量

虽然 HeCBench 很大，
但当前第一步并不建议直接全量引入，
原因是：

1. bring-up 成本高；
2. benchmark 行为杂，前期不利于 feature sanity；
3. 如果 baseline 本身还没稳，大数据集只会放大诊断难度。

因此更合理的顺序是：

1. 先用 `microbench + Rodinia + Altis` 建立 baseline；
2. 再用 `CUTLASS sweep` 扩样本；
3. 最后再用 `HeCBench` 做泛化。

---

## 10. 推荐的数据集构建流程

建议当前按下面顺序构建 A 线验证集：

### Step 1：建立 manifest

为每个候选 kernel 记录：

- 来源 benchmark
- 可执行路径
- 输入参数
- 是否已验证可稳定运行
- 是否已验证可稳定采集 NCU
- 预期行为标签

### Step 2：先采集 canonical microbench

目的：

- 先做 feature audit
- 检查 PKA 12 维信号是否可信

### Step 3：加入 benchmark kernels

目的：

- 验证 baseline clustering 在真实 kernel 上是否稳

### Step 4：加入 target AI kernels

目的：

- 验证 A 线最终要服务的 workload 是否也能进入同一 compression pipeline

### Step 5：如果样本量仍不足，再接 CUTLASS / HeCBench

目的：

- 增加样本量
- 提高 compression experiment 的统计稳定性

---

## 11. 当前最推荐的简短结论

如果把本文件压成最短形式，可以写成：

1. A 线不应只依赖原始 workload 输入，而应建立一套独立的 kernel 验证集。
2. 最合理的第一阶段验证集不是单一 microbench，而是：
   `GPU_Microbenchmark + Rodinia + Altis`
3. `Parboil` 与 `PolyBench/GPU` 适合作为第一阶段补充集。
4. `CUTLASS Profiler` 最适合快速扩大量级；`HeCBench` 更适合第二阶段泛化验证。
5. 第一版推荐目标规模为 `30 ~ 50` 个可验证 kernel；扩展后可达到 `100 ~ 300` 个 compression objects，足以支撑 A 线压缩实验。

---

## 12. 参考链接

- HeCBench: `https://github.com/ORNL/HeCBench`
- Altis: `https://utcs-scea.github.io/altis/`
- Rodinia: `https://rodinia.cs.virginia.edu/`
- gpu-rodinia mirror: `https://github.com/yuhc/gpu-rodinia`
- BabelStream: `https://github.com/UoB-HPC/BabelStream`
- CUTLASS Profiler: `https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html`
