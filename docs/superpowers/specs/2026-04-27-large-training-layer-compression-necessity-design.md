# 大训练单层压缩必要性实验设计

日期：2026-04-27

## 1. 目标

这份 spec 定义第一轮大训练 workload 实验，用来证明为什么在进入 GPU 精确周期仿真之前必须先做 representative compression。

这个实验的目标不是训练完整模型，不是复现 MLPerf 提交结果，也不是立刻完成完整 simulator validation。它的目标是构造一个可控但真实的规模证据：

```text
一个真实大模型训练层 -> kernel timeline -> invocation 规模 -> 压缩空间 -> projected simulation cost
```

核心论点是：

> 对现代训练 workload 来说，即使只是一个大模型单层，也可能产生足够多的 kernel invocation、运行时间和 trace 体积，使得直接走 full-trace exact-cycle simulation 不适合作为默认路径。因此 representative compression 不是单纯优化，而是后端精确仿真的前置条件。

## 2. 背景

当前仓库已经有一条可运行的小规模方法链：

```text
frontend anchor -> middle structure -> backend planning -> execution bridge -> result summary -> writeback
```

这条链路目前主要在 `mini_transformer_v4`、microbenchmarks 和经典 benchmark kernels 上跑通过。这些输入适合做 correctness gate、schema 稳定性检查和接口 bring-up，但它们太小，无法证明 compression 的必要性。

本实验把证据目标从“方法链能不能跑”转成“为什么现代训练 workload 必须先压缩”。

## 3. 实验单元

第一轮 workload 单元是一个 Llama-style decoder block 的训练 step：

```text
单个 decoder layer，随机合成 tokens / activations，forward + backward
```

这一层应包含现代 decoder-only Transformer block 的主要结构：

- RMSNorm 或 LayerNorm
- QKV projection
- attention score computation
- softmax
- attention value / context computation
- output projection
- MLP up / gate / down projections
- activation
- residual paths
- loss proxy
- backward pass

第一轮目标形状如下：

| 参数 | 数值 |
|---|---:|
| batch size | 1 |
| sequence length | 2048 |
| hidden size | 4096 |
| intermediate size | 14336 |
| attention heads | 32 |
| dtype | 优先 bf16，fallback 为 fp16 |
| device | CUDA GPU |
| measured region | warmup 之后的一次 forward + backward |

这个形状足够接近 8B 级 decoder layer 的结构和规模，但又不需要加载完整 pretrained model，因此应能在 32GB RTX 5090 上运行。

## 4. 非目标

这个实验刻意不做以下事情：

- 加载完整 Llama-8B 或更大的 pretrained weights；
- 跑完整模型所有层的 training step；
- 跑完整 dataset、epoch 或 MLPerf benchmark；
- 在第一轮就采集完整 NCU measured PKA features；
- 把所有生成的 kernel 直接送入 exact-cycle simulation；
- 证明最终 simulator accuracy。

这些属于后续阶段。第一轮只证明规模压力和 compression necessity。

## 5. 本地环境假设

2026-04-27 已观察到的本机事实：

- GPU：两张 NVIDIA GeForce RTX 5090，每张约 32GB 显存。
- 工具：`nsys`、`ncu`、`nvcc` 均可用。
- 当前 base Python 没有 PyTorch。
- 当前 `trace_gen` 环境有 CPU-only PyTorch 和 `transformers`。
- NCU performance counters 当前受 `ERR_NVGPUCTRPERM` 限制。

因此第一轮 implementation 应创建或使用 CUDA-enabled PyTorch 环境，并优先使用 Nsight Systems。Nsight Compute measured feature collection 延后到 performance counter 权限可用之后。

## 6. 数据流

实验数据流如下：

```text
large layer harness
  -> nsys profile
  -> exported kernel timeline
  -> invocation table
  -> compression summary
  -> scale proof report
```

### 6.1 Harness 输出

harness 应打印一份小型 machine-readable run summary，至少包括：

- model unit name
- batch size
- sequence length
- hidden size
- intermediate size
- number of heads
- dtype
- warmup count
- profiled iteration count
- forward / backward wall time
- peak GPU memory if available

### 6.2 Timeline 输出

Nsight Systems 输出应转换成 kernel invocation table，至少包含：

- invocation id
- kernel name
- start timestamp
- duration
- grid dimensions
- block dimensions
- stream id if available
- source profiler path

这张表是大规模 scale-proof 版本的现有 small-workload invocation table。

### 6.3 Compression 输出

第一轮 compression summary 可以先基于 name-and-shape grouping，因为第一目标是证明规模压力。它应报告：

- total kernel invocations
- unique kernel names
- unique kernel name plus launch-shape groups
- top groups by runtime coverage
- representative count at several coverage thresholds
- compression ratio under each grouping policy

等 NCU counter access 可用后，同一张 invocation table 可以继续扩展为 PKA measured feature records。

## 7. 输出产物

第一轮 implementation 应把实验产物写到：

```text
experiments/large_training_layer/
```

建议新增文件：

- `run_llama_layer_train.py`
- `run_nsys_layer.sh`
- `parse_nsys_kernels.py`
- `summarize_compression_scale.py`
- `results/llama_layer_b1_s2048_h4096/`

建议生成结果文件：

- `run_summary.json`
- `nsys_report.nsys-rep`
- `nsys_kernel_stats.csv`
- `kernel_invocation_table.json`
- `compression_scale_summary.json`
- `scale_proof_report.md`

## 8. 成功标准

实验成功的标准是产出一份 report，其中包含：

| 证据 | 含义 |
|---|---|
| total kernel invocations | 单层训练已经不是 trivial input |
| unique kernel groups | 存在明显 heterogeneity 和 grouping structure |
| top runtime coverage | 少数 groups 可能主导执行时间 |
| trace / report size | full multi-layer tracing 会明显放大成本 |
| forward / backward wall time | 可以推导 exact-cycle simulation budget |
| compressed representative count | representative compression 能减少 backend candidate count |

第一轮不需要证明最终 simulator speedup。它只需要证明未压缩输入规模使直接后端路径不合理。

## 9. 风险控制

### 9.1 Trace 过大

如果第一轮 `seq_len=2048` 的 run 太大，则把 sequence length 降到 `1024`，同时保持 hidden size 不变。报告中必须记录 fallback shape。

### 9.2 显存不足

如果 `bf16/fp16` training with backward 放不下，优先使用 gradient checkpointing 或降低 sequence length，然后才考虑降低 hidden size。目标是尽量保留 large-layer structure。

### 9.3 NCU Counter 权限

第一轮实验不应阻塞在 NCU 上。先用 `nsys` 生成 timeline 和 scale evidence。PKA measured features 标记为后续 acquisition step。

### 9.4 Synthetic Data 质疑

实验使用 synthetic activations / tokens，但 layer computation 是真实的。报告中必须明确这一点：本实验的 proof target 是真实训练层形状下的 GPU kernel scale 和 trace burden，不是模型质量或数据集收敛。

## 10. 后续路径

第一份 report 生成后，后续阶段是：

1. 当 NCU 权限可用后，为 representative kernel groups 补 measured PKA features。
2. 对比 `seq_len=1024`、`2048`，以及可能的 `4096` 下的 compression summary。
3. 加入一个非 Transformer training layer，例如 DLRM-style embedding / MLP 或 RetinaNet-style vision training，用来观察不同 workload family 下 compression difficulty 是否变化。
4. 只把 selected representatives 接入 backend planning path。

## 11. 验收 Gate

implementation 开始前，应审阅这份 spec 中的以下问题：

- single-layer Llama-style target 是否适合作为第一轮 workload；
- `seq_len=2048, hidden=4096` 是否适合作为第一轮形状；
- `nsys`-first evidence 是否足够支撑第一轮 scale proof；
- 输出产物是否足以接回 A 线 frontend compression。
