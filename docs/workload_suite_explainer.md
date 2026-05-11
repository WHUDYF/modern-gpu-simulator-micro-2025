# M1.5 / Photon Workload Suite 说明

日期：2026-05-11

## 1. 这批 workload 的定位

这批 workload 不是一个普通的 benchmark 下载清单，而是后续 GPU workload trace corpus 的起点。它服务于三个目标：

1. 验证 Photon 对大量 kernel instances 的完整 workload trace 是否有压缩优势。
2. 扩展 PKA / M1 / M1.5 的 measured feature table，使 selector 不只依赖少量 microbench。
3. 为后续训练集提供可复用的数据来源，例如 kernel family classifier、trace compressibility predictor、cost model 和 failure predictor。

当前已经完成的是 `L0 source/harness` 和 `L1 workload registry draft`：

- `12` 个 source。
- `77` 个 workload candidates。
- 所有 build/run/input/license 状态仍是 `pending` 或 `needs_review`。

这意味着：我们已经有了候选池和目录索引，但还没有完成输入数据、模型权重、trace acquisition 或 claim-bearing measured evidence。

## 2. Source 层概览

| Source | 类型 | 当前角色 | 主要价值 |
|--------|------|----------|----------|
| `gpu-rodinia` | 传统 benchmark suite | control + candidate | 论文常用异构计算 benchmark，覆盖 BFS、CFD、Hotspot、LavaMD 等典型模式 |
| `gpu-parboil` | 传统 benchmark suite | control + candidate | throughput computing benchmark，覆盖 BFS、Histo、LBM、SGEMM、SpMV、Stencil 等 |
| `shoc` | GPU benchmark suite | candidate | 包含 GEMM、FFT、MD、Stencil 等 GPU kernel / mini-application 类型 |
| `altis` | modern GPU benchmark suite | candidate | 比传统 Rodinia / SHOC 更现代，适合作为补充候选 |
| `deepbench` | DNN primitive suite | candidate | GEMM、RNN、Convolution 等深度学习 primitive |
| `cutlass` | kernel generator | candidate | GEMM / Conv / Attention-like kernel 变体来源，适合 large-kernel 压力 |
| `mlperf-inference` | full-network suite | candidate | 完整网络 workload 的主要入口，如 BERT、ResNet50、DLRM、Stable Diffusion |
| `gunrock` | graph suite | candidate | GPU graph analytics，适合 irregular workload |
| `pannotia` | graph suite | candidate | 另一组 graph workload，适合和 Gunrock 交叉验证 |
| `hecbench` | heterogeneous suite | candidate | CUDA 版本 benchmark 的补充来源，已做候选筛选 |
| `lammps` | HPC full application | candidate | molecular dynamics full application，适合 full-step trace |
| `gromacs` | HPC full application | candidate | molecular dynamics full application，适合 full-step trace |

## 3. Workload 家族

### 3.1 Benchmark Kernel / Control Workloads

代表来源：

- `gpu-rodinia`
- `gpu-parboil`
- `shoc`
- `hecbench`

代表 workload：

- Rodinia：`bfs`、`cfd`、`hotspot`、`hotspot3d`、`lavamd`、`particlefilter`、`streamcluster`。
- Parboil：`bfs`、`histo`、`lbm`、`sgemm`、`spmv`、`stencil`。
- HeCBench：`bfs`、`spmv`、`hotspot`、`cfd`、`lavamd`。

作用：

- 作为工具链 control，验证 build、run、NCU capture、trace acquisition 能否稳定工作。
- 覆盖典型 GPU kernel 行为：规则计算、memory bandwidth、stencil、sparse、graph traversal。
- 提供可控输入规模，方便先做 smoke run。

局限：

- 大多数是单程序 / 少量 kernel，不足以单独证明 Photon 对完整 workload sequence 的优势。
- 论文中应把它们作为 calibration / control / component-level evidence，而不是完整应用主证据。

### 3.2 Full-Network Workloads

代表来源：

- `mlperf-inference`
- 后续可能补充 HuggingFace / PyTorch / TensorRT harness。

当前 registry 候选：

- `mlperf-inference_bert`
- `mlperf-inference_resnet50`
- `mlperf-inference_dlrm-v2`
- `mlperf-inference_retinanet`
- `mlperf-inference_3d-unet`
- `mlperf-inference_stable-diffusion`

作用：

- 这是 Photon stress suite 的核心。
- 完整网络会产生大量 kernel launches，能验证完整 pipeline 的 trace 压缩，而不是单个 kernel 的局部压缩。
- 网络内部通常包含多类 operator：GEMM、convolution、attention、normalization、activation、embedding、copy / transpose、reduction。
- 对后续训练集也最有价值，因为它能提供真实 kernel sequence、operator diversity 和 shape variation。

当前边界：

- `mlperf-inference` 当前是 sparse / partial clone。
- 还没有下载模型权重和数据集。
- 不能把 synthetic input 直接等同于真实 dataset run。

建议优先级：

1. `BERT`：attention-heavy，适合 sequence length grid。
2. `ResNet50`：经典 CNN，适合 image batch / convolution-heavy 对照。
3. `DLRM-v2`：embedding / recommendation，适合 irregular memory access。
4. `Stable Diffusion UNet`：kernel sequence 多、operator diversity 强。
5. `3D-UNet`：medical segmentation，3D tensor workload。
6. `RetinaNet`：detection pipeline，适合 vision workload 多阶段行为。

### 3.3 DNN Primitive / Kernel Generator

代表来源：

- `deepbench`
- `cutlass`

当前 registry 候选：

- DeepBench：`gemm`、`rnn`、`convolution`。
- CUTLASS：`gemm`、`conv`、`attention`。

作用：

- 用于构造 large-kernel 和 kernel-variant 压力。
- DeepBench 更偏深度学习 primitive benchmark。
- CUTLASS 适合系统地产生 GEMM / Conv / Attention-like kernel 变体。

为什么需要它们：

- 完整网络能提供真实 sequence，但不一定方便控制单个 kernel 的 shape。
- CUTLASS / DeepBench 可以帮助我们做 controlled ablation：固定 workload 类型，只改变 shape / precision / tile / layout。

局限：

- 它们不是完整网络。
- 不能用它们替代 full-network claim-bearing evidence。

### 3.4 Irregular / Graph Workloads

代表来源：

- `gunrock`
- `pannotia`
- Rodinia / Parboil / HeCBench 中的 BFS、SpMV、ParticleFilter、Streamcluster。

当前 registry 候选：

- Gunrock：`bfs`、`sssp`、`pagerank`、`connected-components`。
- Pannotia：`bfs`、`coloring`、`pagerank`。
- Parboil：`bfs`、`histo`、`spmv`。
- Rodinia：`bfs`、`particlefilter`、`streamcluster`。

作用：

- 用于验证 Photon 对不规则 workload 的优势。
- 这类 workload 的 kernel 和 trace 往往更难被简单重复模式压缩。
- 图结构、degree distribution、frontier size、sparse pattern 会显著影响 kernel behavior。

后续关键输入：

- graph size：small / medium / large。
- degree distribution：regular / power-law / skewed。
- real graph vs synthetic graph。
- sparse matrix shape 和 nnz distribution。

注意：

- 这类 workload 的源码大小不重要，真正重要的是图数据或稀疏矩阵数据的规模和结构。

### 3.5 HPC Full Applications

代表来源：

- `lammps`
- `gromacs`

当前 registry 候选：

- LAMMPS：`lj-small-step`、`eam-small-step`。
- GROMACS：`water-small-step`、`protein-small-step`。

作用：

- 验证真实科学应用的完整 step / timestep trace。
- 这类 workload 不是一个孤立 kernel，而是多个阶段组成的完整应用片段。
- 对 Photon 很重要，因为 full application trace 更接近论文中“真实 workload compression”的叙事。

后续关键输入：

- simulation input deck。
- system size。
- timestep count。
- precision。
- force field / pair style / solver mode。

当前边界：

- 仅源码已下载。
- 输入 deck、license、build configuration、GPU backend 还没有完成。
- claim-bearing run 必须使用登记过的 input deck，不能临时找一个未记录输入。

## 4. 按 Photon 压力类型理解

### High Kernel Count

目标：

- 验证 Photon 对大量 kernel launches 的完整 sequence 压缩能力。

优先 workload：

- `mlperf-inference_bert`
- `mlperf-inference_stable-diffusion`
- `mlperf-inference_dlrm-v2`
- `lammps_lj-small-step`
- `gromacs_protein-small-step`

关键指标：

- kernel launch count。
- unique kernel count。
- operator / phase diversity。
- per-workload trace size。
- per-workload compression ratio。

### Large Kernel

目标：

- 验证单个 kernel 动态 trace 很大时 Photon 是否仍然有效。

优先 workload：

- `gpu-parboil_sgemm`
- `gpu-parboil_lbm`
- `gpu-parboil_stencil`
- `gpu-rodinia_cfd`
- `gpu-rodinia_lavamd`
- `cutlass_gemm`
- `cutlass_conv`
- `cutlass_attention`

关键指标：

- dynamic instruction count。
- memory transaction count。
- trace size per invocation。
- compression ratio per kernel。

### Irregular Workload

目标：

- 验证 Photon 对不规则访存、不规则 control flow、非平稳 sequence 的适应能力。

优先 workload：

- `gunrock_bfs`
- `gunrock_sssp`
- `gunrock_pagerank`
- `pannotia_bfs`
- `gpu-parboil_spmv`
- `gpu-parboil_histo`
- `gpu-rodinia_bfs`
- `gpu-rodinia_streamcluster`
- `mlperf-inference_dlrm-v2`

关键指标：

- branch / divergence feature。
- memory coalescing feature。
- atomic / scatter / gather 特征。
- compression ratio vs graph / sparse input shape。

## 5. 建议的第一批 Smoke Workloads

第一批不要直接选择最重的完整网络或 HPC full app。建议按“能快速发现工具链问题，又覆盖三类压力”的原则选：

| 目的 | Workload | 原因 |
|------|----------|------|
| control | `gpu-rodinia_nn` | 已有历史上下文，适合验证 harness |
| irregular | `gpu-parboil_spmv` | 稀疏矩阵访问，适合验证 irregular path |
| large kernel | `gpu-parboil_sgemm` | 规则大计算 kernel，适合 large-kernel control |
| graph | `gunrock_bfs` | graph traversal，适合 irregular graph |
| full network small | `mlperf-inference_resnet50` 或 `mlperf-inference_bert` | 完整网络入口，但先用 small synthetic input |
| HPC full app | `lammps_lj-small-step` | 比 GROMACS 更容易先做 small input deck |

这批 smoke workload 的目标不是产生论文结论，而是验证：

- 能否 build。
- 能否 run。
- 能否采集 launch metadata。
- 能否采集 NCU measured feature。
- 能否生成小规模 trace。

## 6. 后续 Gate C2/C3 要补的内容

### Gate C2：Input / Weight Plan

需要为每个 claim-bearing workload 明确：

- 权重来源。
- 数据集来源或 synthetic input policy。
- 输入大小。
- license 状态。
- checksum。
- 是否允许进入 claim-bearing run。

### Gate C3：Trace Acquisition Smoke

需要验证：

- build command。
- run command。
- small input。
- launch metadata。
- NCU capture。
- trace export。
- trace parser 是否接受目录结构。

只有完成 Gate C2/C3 后，这批 workload 才能进入真正的 measured trace corpus。

## 7. 当前最重要的判断

当前 registry 里的 `77` 个 workload 是候选，不是最终训练集。

对后续论文最关键的是三条线：

1. 完整网络线：证明 Photon 不只对单 kernel 有效。
2. 不规则线：证明 Photon 对 graph / sparse / embedding 这类复杂行为有优势。
3. HPC full app 线：证明 Photon 能处理真实科学应用的完整 step。

因此下一步应该优先写 `input_asset_plan.md`，并选择少量 smoke workload 验证 build/run/trace，再逐步扩大到完整 shape grid。

