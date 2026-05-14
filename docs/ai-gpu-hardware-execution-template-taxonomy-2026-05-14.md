# AI GPU Hardware Execution Template Taxonomy

日期：2026-05-14

## 1. 文档目的

本文档用于扩展 B 线中的两层结构：

```text
Route Primitive
Hardware Execution Template
```

目标是回答三个问题：

1. `Route Primitive` 如何从主流 AI workload 的算法路径中提出；
2. `Hardware Execution Template` 为什么是必要的分组层；
3. 在 GPU 上部署的主流 AI 算法中，哪些常见计算角色可以映射到哪些硬件执行模板。

本文档不是最终完备 taxonomy。它是第一版详细候选集，用于后续 PKA -> B/C 线实现时避免只按 kernel name、operator string 或 PKA cluster 做分组。

边界声明：

- 本文档依据公开的 NVIDIA library 文档、FlashAttention 论文、CUTLASS/cuDNN/cuBLAS/cuSPARSE/CUB/NCCL/HugeCTR 等实现资料，以及当前项目 mini_transformer_v4 的已有 B 线协议归纳；
- 它不是对所有框架、所有 kernel、所有厂商后端的穷尽枚举；
- 它的目标不是证明某个算法只能属于某一类，而是给 B/C 线提供一组可解释、可扩展、可验证的硬件执行骨架；
- 后续如果真实 trace、kernel profiling 或 vendor implementation 显示不同执行路径，应以 measured evidence 覆盖本文档的先验 hint。

## 2. 核心区分

### 2.1 Route Primitive

`Route Primitive` 回答：

```text
这个 kernel / anchor 在算法主路线中承担什么功能角色？
```

它靠近 workload / algorithm path，不等于 kernel name，也不等于网络模块名称。

例如：

```text
attention block
  -> Dense Projection/Transform
  -> Pairwise Score
  -> Reduction / Normalize
  -> Weighted Aggregation
  -> Dense Projection/Transform
```

这里的 `attention block` 是上层模块名，不是 primitive。`softmax_kernel` 是 kernel name，也不是 primitive。`Reduction / Normalize` 才是 route primitive。

### 2.2 Hardware Execution Template

`Hardware Execution Template` 回答：

```text
这个算法角色在 GPU 上主要通过什么执行骨架实现？
```

它靠近 GPU 执行方式、资源压力和 simulator 参数方向。

例如：

```text
Dense Projection/Transform
  -> Dense Tiled Tensor-Core Compute

Reduction / Normalize
  -> Reduction / Scan / Normalize Template

Weighted Aggregation
  -> Streaming Gather / Aggregation Template
```

### 2.3 Family 与 Template 的关系

在 B 线中：

```text
phase
  -> family
  -> regime
  -> lane
```

`family` 是共享机制组织层。`Hardware Execution Template` 是 family 判断的重要证据之一，但二者不完全等同。

更精确地说：

```text
Hardware Execution Template = GPU 上的执行骨架
Family = 在某个 phase 内可共享 simulator reasoning 的机制组
Regime = phase + family + route primitive + template + shape/resource 后的 C-line 对象
```

## 3. 为什么需要 Hardware Execution Template

只看 `Route Primitive` 会丢掉 GPU 执行方式。

例如：

- `Dense Projection/Transform` 通常落到 GEMM / tiled matrix compute；
- 但某些 transform 也可能通过 convolution、grouped GEMM 或 fused custom kernel 实现；
- 同一个 primitive 的不同实现会暴露不同资源压力。

只看 kernel name 也不稳定。

例如：

- 多个 `gemm` kernel 可能分别承担 projection、attention score、FFN expansion；
- 多个 fused kernel 名称可能隐藏 reduction、elementwise、layout conversion；
- vendor library kernel name 不一定稳定，也不一定表达算法角色。

只看 PKA cluster 也不够。

PKA cluster 说明 feature-space 近似，但不保证：

- phase 相同；
- route primitive 相同；
- hardware template 相同；
- shape / size regime 相近；
- resource signature 相容。

因此需要 `Hardware Execution Template` 作为中间抽象，把算法角色和 GPU 执行机制连接起来。

### 3.1 Template 的提出规则

一个候选 template 只有满足下面条件，才应进入 B 线 taxonomy：

1. 它在多个主流 AI workload 中反复出现，或者在单一 workload 中承担高影响路径；
2. 它对应的 GPU 执行骨架会改变 simulator-side reasoning，例如 Tensor Core tile、shared memory、reduction tree、irregular memory、collective topology；
3. 它不能被已有 template 充分解释，否则只作为已有 template 下的 regime/shape/resource 子类；
4. 它能帮助判断 family，但不直接等于 family；
5. 它必须保留 provenance 和 confidence，不能把网络结构先验冒充成 measured runtime。

因此本文档的抽取流程是：

```text
mainstream AI workload
  -> algorithm path role
  -> route primitive
  -> observed / documented GPU implementation skeleton
  -> hardware execution template
  -> phase-local family / regime / lane candidate
```

这个流程也说明了为什么它不是“按算子阶段分 regime”。`Route Primitive` 来自算法路径角色，`Hardware Execution Template` 来自 GPU 执行骨架；真正的 `regime` 仍然要在 squash 输出的 phase 内，结合 shape/resource/signature 后生成。

### 3.2 公开资料支持到哪一层

| 资料来源 | 能支持的结论 | 不应过度推出的结论 |
|---|---|---|
| cuDNN | DNN 中常见 primitives 包括 scaled dot-product attention、convolution、matmul、normalization、softmax、pooling、pointwise 和 fusion | 不代表所有框架都会用同一个 cuDNN kernel |
| cuBLAS / cuBLASLt | GEMM、batched GEMM、low/mixed precision 和 Tensor Core 优化是 dense AI compute 的主路径之一 | 不代表所有 dense transform 都是同一 regime |
| CUTLASS | GEMM 和 implicit GEMM convolution 可由 tile、threadblock、iterator、layout、fused epilogue 等机制表达 | 不代表所有 convolution 都应该归入 dense GEMM family |
| FlashAttention | attention 可以通过 IO-aware tiling、online softmax、避免完整 score materialization 形成融合执行骨架 | 不代表所有 attention kernel 都是 flash/fused template |
| CUB | reduction、scan、sort、histogram、top-k 等并行 primitives 是 GPU selection/reduction/routing 的基础构件 | 不代表 CUB API 名称就是 B 线 primitive 名称 |
| cuSPARSE | SpMV、SpMM、sparse-dense 等 sparse linear algebra 路径在 GPU 上有独立 API 和算法选择 | 不代表 GNN/sparse workload 都由 cuSPARSE 实现 |
| HugeCTR / Merlin | 推荐系统 embedding table lookup、slot 内 reduction、slot 间 concat、model-parallel embedding table 是重要 GPU 路径 | 不代表 token embedding 与 recommender embedding 完全同一 regime |
| NCCL | allreduce、allgather、reduce-scatter、broadcast、send/recv 是分布式 AI 的核心通信 primitive | 不代表单 GPU C 线必须立即建模 collectives |

## 4. 主流 AI Workload 覆盖矩阵

这张表先从算法 workload 角度列出“为什么我们需要这些分组”。它不是直接生成 regime 的表，而是说明 route primitive 和 hardware template 的来源。

| Workload / 算法结构 | 常见算法路径角色 | 常见 Hardware Execution Template |
|---|---|---|
| Transformer / LLM prefill | QKV/O projection、QK score、softmax、PV aggregation、FFN、norm、residual、RoPE | HET-1, HET-3, HET-4, HET-5, HET-6 |
| LLM decode / serving | small-batch projection、KV-cache read/write、attention aggregation、sampling/top-k/top-p、tensor-parallel collectives | HET-1, HET-3, HET-6, HET-9, HET-10, HET-11 |
| CNN / vision backbone | convolution、depthwise/grouped convolution、pooling、batchnorm、activation、layout transform | HET-2, HET-4, HET-5, HET-10 |
| Diffusion U-Net / vision generative model | convolution blocks、attention blocks、normalization、upsample/downsample、elementwise conditioning | HET-2, HET-3, HET-4, HET-5, HET-10 |
| Recommender / DLRM-like model | embedding lookup、embedding pooling/reduction、dense MLP、feature interaction、multi-GPU table sharding | HET-7, HET-4, HET-1, HET-11 |
| GNN / sparse graph model | neighbor gather、edge traversal、scatter/update、SpMM-like propagation、segment reduction | HET-8, HET-6, HET-4 |
| Sparse attention / retrieval | sparse index traversal、pairwise score、top-k selection、weighted aggregation | HET-8, HET-1, HET-6, HET-9 |
| MoE / conditional computation | gating score、top-k expert selection、token dispatch/pack、grouped expert GEMM、expert result combine、expert-parallel exchange | HET-9, HET-10, HET-1, HET-6, HET-11 |
| Distributed training / inference | gradient allreduce、tensor-parallel allreduce/allgather/reduce-scatter、pipeline/expert communication | HET-11 |

## 5. 主流 AI Workload 的 Route Primitive 候选

下面是第一版候选 Route Primitive。它们来自 Transformer/LLM、CNN/Diffusion、Recommender、GNN/Sparse、MoE 和分布式训练/推理等常见 GPU workload。

| Route Primitive | 算法路径含义 | 常见 workload |
|---|---|---|
| `Dense Projection/Transform` | 线性投影、MLP、FFN、dense transform | Transformer、LLM、MLP、推荐模型 |
| `Pairwise Score` | 两两关系分数、相似度、QK score | Attention、retrieval、matching |
| `Reduction / Normalize` | sum/max/mean、softmax、layernorm、RMSNorm、batchnorm | Transformer、CNN、GNN |
| `Weighted Aggregation` | 根据权重聚合 value / feature | Attention、GNN、推荐模型 |
| `Convolution / Stencil Transform` | 局部窗口卷积、stencil-like transform | CNN、Diffusion U-Net、vision backbone |
| `Embedding / Table Lookup` | sparse/categorical id 到 dense vector | Recommender、NLP embedding、token embedding |
| `Sparse Gather / Scatter` | sparse edge / index 驱动的数据收集与写回 | GNN、sparse attention、recommendation |
| `Routing / Dispatch` | MoE gating、expert dispatch、top-k routing | MoE LLM、conditional computation |
| `Elementwise Fusion` | activation、bias、residual、mask、scale、dropout | 几乎所有 AI workload |
| `Layout / Pack / Quantize` | transpose、reshape、pack、quant/dequant、KV-cache update | LLM inference、TensorRT/cuDNN graph、quantized inference |
| `Selection / Sampling` | top-k、top-p、argmax、sampling、beam step | LLM decoding、retrieval |
| `Collective Communication` | allreduce、allgather、reduce-scatter、broadcast | distributed training / inference |

这张表不是封闭列表。它的作用是提供 B 线第一版可扩展词表。

## 6. Hardware Execution Template 候选

### HET-1: Dense Tiled Tensor-Core Compute

核心模式：

```text
tiled matrix multiply / batched matrix multiply / grouped GEMM
```

典型 route primitive：

- `Dense Projection/Transform`
- `Pairwise Score`
- MLP / FFN blocks
- MoE expert GEMM

典型 GPU 行为：

- tile 化加载；
- shared memory 或寄存器复用；
- Tensor Core / MMA 指令；
- occupancy、register pressure、tile size、memory alignment 影响明显。

为什么需要这个 template：

主流 AI workload 中大量 compute-heavy 路径最终落到 GEMM 或 batched GEMM。cuBLAS 文档说明 Tensor Cores 会显著加速矩阵乘，cuBLAS 也会在合适条件下选择 Tensor Core 实现。CUTLASS 也围绕高性能 GEMM 提供 CUDA C++ 抽象。

B 线意义：

```text
这是 dense_compute family 的核心执行骨架。
```

但它不等于单一 regime。比如 projection、pairwise score、FFN expansion 都可能共享这个 template，但 phase、route primitive、shape 和 resource signature 不同。

常见 C-line lane：

- `occupancy_sensitive`
- `register_sensitive`
- `compute_pipeline_sensitive`

### HET-2: Convolution / Stencil Tiled Compute

核心模式：

```text
local-window convolution / implicit GEMM convolution / Winograd or transform-based convolution
```

典型 route primitive：

- `Convolution / Stencil Transform`
- patch embedding convolution
- depthwise / grouped convolution
- U-Net / CNN feature transform

典型 GPU 行为：

- filter / activation tile reuse；
- implicit GEMM lowering 或 direct convolution；
- layout sensitivity；
- memory reuse 与 tensor core utilization 共同影响性能。

为什么需要这个 template：

CNN 和 diffusion U-Net 中的卷积不是普通 dense projection。NVIDIA cuDNN 文档把 convolution、pooling、normalization、attention、matmul 等列为 DNN primitives；NVIDIA convolution performance guide 也说明 cuDNN convolution 有 implicit-GEMM 和 transform-based 实现路径。

B 线意义：

```text
它避免把 CNN / diffusion workload 全部误塞进 Dense Projection/Transform。
```

常见 C-line lane：

- `tensor_layout_sensitive`
- `cache_sensitive`
- `occupancy_sensitive`

### HET-3: IO-Aware Attention Tile

核心模式：

```text
QK score + online softmax + PV aggregation fused / tiled over SRAM-HBM hierarchy
```

典型 route primitive：

- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`

典型 GPU 行为：

- attention score 不完全物化；
- online softmax；
- tile K/V；
- 减少 HBM read/write；
- register 和 shared memory 压力明显。

为什么需要这个 template：

FlashAttention 论文将 attention 实现为 IO-aware exact attention，通过 tiling 减少 HBM 与 SRAM 之间的数据移动。cuDNN 也把 scaled dot-product attention / fused flash attention 纳入 DNN primitive / graph API 能力。

B 线意义：

```text
标准 attention 可拆为多个 route primitives；
但 fused/flash attention 在硬件执行上形成新的 template。
```

因此 B 线不能只按 `Pairwise Score`、`Reduction / Normalize`、`Weighted Aggregation` 分开，也要能表达 fused attention template。

常见 C-line lane：

- `memory_hierarchy_sensitive`
- `shared_memory_sensitive`
- `fused_attention_sensitive`

### HET-4: Reduction / Scan / Normalize Template

核心模式：

```text
parallel reduction / prefix scan / normalization / softmax / pooling
```

典型 route primitive：

- `Reduction / Normalize`
- pooling
- loss reduction
- layernorm / RMSNorm / batchnorm
- softmax

典型 GPU 行为：

- warp/block reduction；
- shared memory 或 warp shuffle；
- synchronization；
- numerical stability path；
- memory bandwidth 与 reduction tree 共同影响性能。

为什么需要这个 template：

CUB 提供 warp/block/device 层级的 reduction、scan、sort、histogram 等并行 primitives。cuDNN 也将 normalization、softmax、pooling 作为深度学习基础 primitives。

B 线意义：

```text
Reduction / Normalize 是算法路径角色；
Reduction Template 是 GPU 执行骨架。
```

同为 reduction template 的 softmax、layernorm、pooling 仍可因 phase、shape、resource signature 不同拆成不同 regimes。

常见 C-line lane：

- `reduction_path_sensitive`
- `shared_memory_sensitive`
- `cache_sensitive`

### HET-5: Elementwise / Pointwise Fusion Template

核心模式：

```text
map-style elementwise operations, often fused with bias / activation / residual / mask / scale
```

典型 route primitive：

- `Elementwise Fusion`
- activation
- bias add
- residual add
- dropout / mask
- RoPE / scale / clamp

典型 GPU 行为：

- memory bandwidth dominated；
- simple arithmetic intensity；
- kernel launch overhead may matter；
- fusion reduces intermediate memory traffic。

为什么需要这个 template：

Elementwise operations are everywhere in AI workloads. Treating them as residual noise is unsafe because they can dominate small-batch inference, decoding, or memory-bound phases. But they usually should not be confused with compute-heavy GEMM regimes.

B 线意义：

```text
Elementwise template often maps to constraint/regression or fusion-sensitive regimes.
```

常见 C-line lane：

- `constraint_regression`
- `memory_bandwidth_sensitive`
- `fusion_sensitive`

### HET-6: Streaming Gather / Weighted Aggregation Template

核心模式：

```text
read weights / indices / values, stream through memory, accumulate output
```

典型 route primitive：

- `Weighted Aggregation`
- attention value aggregation
- graph neighbor aggregation
- feature pooling / weighted sum

典型 GPU 行为：

- memory coalescing sensitivity；
- locality / cache behavior；
- accumulation order；
- possible scatter/gather irregularity；
- L1/L2/DRAM pressure。

为什么需要这个 template：

Weighted aggregation often looks simple at the algorithm level, but GPU performance depends strongly on locality and memory access pattern. It should not be merged into reduction or dense compute solely because it is inside the same attention or graph module.

B 线意义：

```text
It separates attention readout / aggregation-side behavior from score compute and normalization.
```

常见 C-line lane：

- `memory_coalescing_sensitive`
- `cache_sensitive`
- `locality_sensitive`

### HET-7: Embedding / Table Lookup Template

核心模式：

```text
categorical id lookup -> dense embedding vectors, often with pooling / reduction
```

典型 route primitive：

- `Embedding / Table Lookup`
- feature lookup
- embedding bag
- categorical sparse feature expansion

典型 GPU 行为：

- random or semi-random memory access；
- huge table footprint；
- cache miss sensitivity；
- model parallel table sharding；
- reduction / concatenation across slots。

为什么需要这个 template：

NVIDIA Merlin HugeCTR is designed for GPU-accelerated recommender systems and supports model-parallel embedding tables. Its documentation describes table lookup, weight reduction within slots, and concatenation across slots as embedding-layer operations.

B 线意义：

```text
Embedding lookup should not be treated as generic streaming memory only.
```

Its table size, cache behavior, sharding, and sparse categorical access make it a distinct template for recommender and retrieval workloads.

常见 C-line lane：

- `embedding_cache_sensitive`
- `memory_random_access_sensitive`
- `collective_or_sharding_sensitive`

### HET-8: Sparse / Irregular Matrix-Graph Template

核心模式：

```text
SpMV / SpMM / sparse gather-scatter / graph edge traversal
```

典型 route primitive：

- `Sparse Gather / Scatter`
- sparse attention
- graph propagation
- sparse matrix transform

典型 GPU 行为：

- irregular memory access；
- load imbalance；
- atomics or segmented reductions；
- sparse format sensitivity；
- divergence。

为什么需要这个 template：

cuSPARSE provides sparse linear algebra APIs such as SpMV and SpMM, with support for different sparse formats and algorithms. GNN and sparse workloads commonly reduce to sparse matrix dense vector/matrix operations or scatter/gather aggregation.

B 线意义：

```text
Sparse/irregular behavior is not just low arithmetic intensity;
it has distinct control, indexing, format, and load-balance behavior.
```

常见 C-line lane：

- `irregular_control_sensitive`
- `memory_coalescing_sensitive`
- `atomic_or_segmented_reduction_sensitive`

### HET-9: Selection / Sort / Routing Template

核心模式：

```text
top-k / top-p / sort / histogram / route dispatch / expert selection
```

典型 route primitive：

- `Selection / Sampling`
- `Routing / Dispatch`
- top-k gating
- beam search step
- token sampling

典型 GPU 行为：

- partial sort / select；
- prefix scan；
- histogram or bucketization；
- scatter / gather dispatch；
- small batch latency sensitivity。

为什么需要这个 template：

LLM decoding and MoE routing increasingly expose selection/routing kernels. CUB provides GPU parallel primitives including sort, prefix scan, reduction, histogram, and DeviceTopK support for built-in numeric types and half/bfloat16.

B 线意义：

```text
Selection/routing kernels are not dense compute and not plain elementwise.
```

They often control downstream work distribution, so they can have high decision importance even if raw FLOPs are low.

常见 C-line lane：

- `routing_dispatch_sensitive`
- `scan_sort_sensitive`
- `latency_sensitive`

### HET-10: Layout / Pack / Quantize / Cache Update Template

核心模式：

```text
transpose / reshape / pack / unpack / quantize / dequantize / KV-cache update
```

典型 route primitive：

- `Layout / Pack / Quantize`
- tensor format conversion
- KV cache write/read/update
- prefill/decode cache movement

典型 GPU 行为：

- bandwidth dominated；
- strided memory access；
- coalescing sensitivity；
- alignment and vectorization；
- sometimes fused with matmul or attention。

为什么需要这个 template：

Modern inference systems spend meaningful time in layout conversion, quantization/dequantization, and KV-cache movement, especially in low-batch decoding. These kernels often do not map cleanly to algorithmic operators, but they strongly affect GPU runtime and simulator behavior.

B 线意义：

```text
This template prevents data-movement infrastructure kernels from being hidden under nearby compute operators.
```

常见 C-line lane：

- `memory_coalescing_sensitive`
- `layout_transform_sensitive`
- `decode_latency_sensitive`

### HET-11: Collective Communication Template

核心模式：

```text
allreduce / allgather / reduce-scatter / broadcast / send-recv
```

典型 route primitive：

- `Collective Communication`
- tensor parallel allreduce
- data parallel gradient allreduce
- pipeline / expert parallel exchange

典型 GPU 行为：

- interconnect bandwidth / latency；
- GPU-GPU synchronization；
- overlap with compute；
- message size and topology sensitivity。

为什么需要这个 template：

NCCL implements GPU-optimized collective and point-to-point communication primitives, including allreduce, broadcast, reduce-scatter, allgather and send/recv patterns. These operations are core to distributed AI training and inference.

B 线意义：

```text
Collectives are not local kernel compute templates, but they are critical backend validation objects.
```

They should be modeled separately when the B/C line expands beyond single-GPU local kernels.

常见 C-line lane：

- `collective_bandwidth_sensitive`
- `overlap_sensitive`
- `multi_gpu_scaling_sensitive`

## 7. Route Primitive 到 Hardware Template 的映射

| Route Primitive | Primary Hardware Templates | Notes |
|---|---|---|
| `Dense Projection/Transform` | HET-1, HET-2 | Linear layers usually map to GEMM; conv-like transforms map to convolution templates. |
| `Pairwise Score` | HET-1, HET-3 | Standard attention score may be GEMM; flash/fused attention changes the execution template. |
| `Reduction / Normalize` | HET-4, HET-3 | Standalone softmax/norm is reduction template; fused attention may absorb it. |
| `Weighted Aggregation` | HET-6, HET-3, HET-8 | Attention PV aggregation may be streaming/fused; graph aggregation may be sparse. |
| `Convolution / Stencil Transform` | HET-2 | CNN/diffusion conv-like routes. |
| `Embedding / Table Lookup` | HET-7 | Recommender and token embedding lookup. |
| `Sparse Gather / Scatter` | HET-8, HET-6 | GNN and sparse attention. |
| `Routing / Dispatch` | HET-9, HET-10 | MoE routing and expert dispatch. |
| `Elementwise Fusion` | HET-5, HET-10 | Fused pointwise and layout-adjacent elementwise kernels. |
| `Layout / Pack / Quantize` | HET-10 | Data movement and format conversion. |
| `Selection / Sampling` | HET-9 | LLM decoding and retrieval selection. |
| `Collective Communication` | HET-11 | Distributed training/inference. |

## 8. 为什么这些 template 能支撑 family 分组

Family 的目标不是复述算法语义，而是组织 simulator-side reasoning。

因此 family 分组需要能解释：

1. 主要资源压力是什么；
2. 哪些 simulator 参数方向可能影响它；
3. 哪些对象可以共享验证结果；
4. 哪些对象不能合并。

Hardware Execution Template 提供这层解释。

例如：

| Template | 主要资源压力 | 可能 family | 不应混并的对象 |
|---|---|---|---|
| Dense Tiled Tensor-Core Compute | Tensor Core, register, occupancy, tile reuse | `dense_compute` | reduction-only, embedding lookup, sparse traversal |
| Reduction / Scan / Normalize | synchronization, shared memory, bandwidth | `reduction_normalization` | dense GEMM, streaming aggregation |
| Streaming Gather / Weighted Aggregation | locality, coalescing, DRAM/cache | `streaming_memory` | compute-heavy GEMM, pure reduction |
| Embedding / Table Lookup | random access, huge table, cache/sharding | `embedding_memory` | regular streaming copy |
| Sparse / Irregular Matrix-Graph | divergence, sparse format, atomics | `irregular_traversal` | dense memory coalesced paths |
| Selection / Routing | scan/sort/top-k, scatter dispatch | `routing_selection` | pure elementwise |
| Collective Communication | interconnect bandwidth/latency | `collective_communication` | local compute kernels |

## 9. 对现有 mini_transformer_v4 的回填

当前已有最小路线：

| Existing object | Route Primitive | Hardware Template |
|---|---|---|
| `gemm_tiled` projection | `Dense Projection/Transform` | HET-1 Dense Tiled Tensor-Core Compute |
| `attention_score` | `Pairwise Score` | HET-1 Dense Tiled Tensor-Core Compute, possible HET-3 if fused |
| `softmax_kernel` | `Reduction / Normalize` | HET-4 Reduction / Scan / Normalize |
| `context_mul` | `Weighted Aggregation` | HET-6 Streaming Gather / Weighted Aggregation |
| `layernorm_kernel` | `Reduction / Normalize` | HET-4 Reduction / Scan / Normalize |
| `residual_add` | `Elementwise Fusion` | HET-5 Elementwise / Pointwise Fusion |

这说明原来的五类 primitive 是合理的，但还不够覆盖 CNN、recommender、GNN、MoE、distributed training 和 LLM decoding。

## 10. B/C 线实现建议

### 10.1 输入字段

未来 `NetworkStructureContext` 或 route annotation 应提供：

| 字段 | 含义 |
|---|---|
| `phase_id` | squash 输出的 phase |
| `route_primitive` | 算法路径角色 |
| `hardware_template_hint` | 可选硬件模板 hint |
| `shape_signature` | M/N/K、seq length、batch、head dim、table size、sparse format 等 |
| `resource_signature_hint` | register、shared、DRAM、cache、irregular、collective 等 |
| `source` | annotation / parser / model graph / manual card |
| `confidence` | low / medium / high |

### 10.2 Builder 规则

1. 先按 `phase_id` 展开；
2. 在 phase 内按 family 判断 shared mechanism；
3. 使用 `route_primitive` 保留算法路径角色；
4. 使用 `hardware_template` 判断 GPU 执行骨架；
5. 使用 shape/resource 决定 regime；
6. 禁止 raw kernel name 直接生成 stable regime；
7. 禁止 PKA cluster 直接生成 stable regime。

### 10.3 Boundary 策略

如果一个 anchor 同时符合多个 template：

- fused attention：允许 HET-3，并记录 absorbed primitives；
- MoE route + GEMM：route/dispatch 和 expert GEMM 应拆 regime；
- embedding lookup + reduction：可以先作为 HET-7，但记录 reduction subpath；
- sparse aggregation：如果是 dense-like batched matmul，走 HET-1；如果是 index/edge driven，走 HET-8。

## 11. 当前结论

这些 Hardware Execution Templates 不是任意分类，而是从主流 AI GPU workload 中反复出现的执行骨架抽象出来的：

1. dense matrix/tensor compute；
2. convolution/stencil compute；
3. IO-aware fused attention；
4. reduction/scan/normalize；
5. elementwise fusion；
6. streaming / weighted aggregation；
7. embedding / table lookup；
8. sparse / irregular graph-matrix behavior；
9. selection / routing；
10. layout / quantize / cache movement；
11. collective communication。

它们的价值在于把算法角色和 GPU 执行机制拆开：

```text
Route Primitive: 算法路径中做什么
Hardware Execution Template: GPU 上如何执行
Family: 在 phase 内共享什么硬件机制
Regime: 哪一段执行工作区间可以给 C 线验证
Lane: 沿哪个 backend 参数方向验证
```

## 12. Sources

- NVIDIA cuDNN documentation: https://docs.nvidia.com/cudnn/index.html
- NVIDIA cuDNN latest documentation: https://docs.nvidia.com/deeplearning/cudnn/latest/
- NVIDIA cuBLAS documentation: https://docs.nvidia.com/cuda/archive/12.9.1/pdf/CUBLAS_Library.pdf
- NVIDIA cuBLAS developer page: https://developer.nvidia.com/cuBLAS
- NVIDIA cuSPARSE documentation: https://docs.nvidia.com/cuda/archive/11.8.0/cusparse/index.html
- NVIDIA cuSPARSE developer page: https://developer.nvidia.com/cusparse
- NVIDIA CUB documentation: https://docs.nvidia.com/cuda/cub/index.html
- NVIDIA CUTLASS documentation: https://docs.nvidia.com/cutlass/4.3.5/index.html
- NVIDIA CUTLASS implicit GEMM convolution: https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/implicit_gemm_convolution.html
- NVIDIA Deep Learning Performance convolution guide: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
- NVIDIA Merlin HugeCTR documentation: https://nvidia-merlin.github.io/HugeCTR/v23.12.00/hugectr_user_guide.html
- NVIDIA Merlin HugeCTR overview: https://developer.nvidia.com/blog/introducing-merlin-hugectr-training-framework-dedicated-to-recommender-systems
- NVIDIA NCCL documentation: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
- NVIDIA NCCL developer page: https://developer.nvidia.com/nccl
- FlashAttention paper: https://arxiv.org/abs/2205.14135
