# Instruction Motif To GPU Hardware Structure Design Spec

日期：2026-06-06

## 1. 核心问题

我们现在缺少的是这条桥：

```text
GCL / GNNExplainer learned trace structure
  -> GPU hardware structure candidate
  -> mechanism family candidate
  -> legal tuning knob candidate
```

如果没有这条桥，GCL 只能回答：

```text
哪些 kernel 在 trace graph embedding space 中相似？
```

但不能回答：

```text
这个 cluster 为什么像 global memory path？
为什么应该优先验证 dram latency / cache / coalescing 相关 knob？
```

因此本 spec 的目标不是直接预测瓶颈，也不是直接预测 knob importance ratio，而是建立一个低风险的证据绑定层：

```text
instruction + graph motif + counter + static metadata
  -> hardware_structure_candidates
```

## 2. Claim 边界

必须避免的说法：

```text
instruction ratio proves memory-bound / compute-bound.
GNNExplainer motif proves FP64 pressure.
cluster assignment is a calibrated hardware probability.
knob score is causal importance.
```

第一版允许的说法：

```text
instruction composition and explanation motifs provide candidate evidence
for GPU hardware structures.
```

也就是说：

```text
LDG/STG + load-to-use motif + memory counter support
```

可以支持：

```text
global_memory_path candidate
```

但不能单独推出：

```text
the kernel is memory-bound
dram_latency is the causal best knob
```

瓶颈和调参结论必须由后续 validation 或 counter agreement 提升。

## 3. 输入

第一版输入来自五类证据。

### 3.1 Opcode Evidence

来源：

```text
SASS / PTX / trace instruction records
```

例子：

```text
LDG, STG, LDS, STS
FFMA, DFMA, HMMA, MMA
BAR.SYNC
BRA, SSY, SYNC
register move / spill-related local memory access
```

它回答：

```text
trace 中出现了哪些可能触发特定硬件路径的指令？
```

### 3.2 Graph Motif Evidence

来源：

```text
GCL trace graph
GNNExplainer compact explanation subgraph
edge mask
node mask
feature mask
```

例子：

```text
load-to-use dependency edge
long compute dependency chain
barrier fan-in / fan-out
branch divergent path
shared-memory access neighborhood
high fanout value dependency
```

它回答：

```text
GNN 的 cluster / prototype 判断依赖哪些局部结构？
```

### 3.3 Static Resource Metadata

来源：

```text
register count
shared memory bytes
local memory bytes / spill signal
block size
occupancy estimate
launch configuration
```

它回答：

```text
该 kernel 是否有资源占用层面的支持证据？
```

### 3.4 Profiler / Counter Evidence

来源：

```text
Nsight Compute counters
PKA-style summarized counters
hardware performance counters
```

例子：

```text
memory throughput / latency / replay
warp stall reason
SM issue utilization
tensor pipe utilization
occupancy
shared memory bank conflict
branch efficiency
```

它回答：

```text
硬件实际观测是否支持该候选结构？
```

### 3.5 Microbench Anchor Evidence

来源：

```text
LLM / human-generated microbench anchors
validated synthetic kernels
known-mechanism fixtures
```

它回答：

```text
当前 motif 是否接近某个已知机制的 anchor？
```

## 4. 第一版硬件结构枚举

先不要把 family 设计得太细。第一版建议只支持：

```text
global_memory_path
shared_memory_path
fp32_fp64_compute_pipeline
tensor_core_pipeline
register_file_occupancy
sync_barrier
branch_control
```

这些结构足够覆盖我们目前最关心的调参方向，并且容易从指令、motif、counter 中找到对应证据。

## 5. Rule-Based Hardware Semantic Binding

第一版不建议直接用 LLM 解释，也不建议让 FC head 黑盒输出 hardware family。

推荐使用固定 taxonomy matcher：

```text
evidence pack
  -> rule-based hardware structure matcher
  -> candidate support score
  -> claim_status
```

原因：

1. 审稿风险更低；
2. 每个候选都有可追溯 evidence；
3. 后续可以用 validation data 替换或校准 rule weight；
4. 不会把 LLM judgment 或 instruction ratio 伪装成 ground truth。

## 6. Mapping Table

| Evidence Pattern | Graph / Motif Signal | Hardware Structure Candidate | Notes |
| --- | --- | --- | --- |
| `LDG` / `STG` 比例高，且 explanation motif 中包含 load-to-use edge | load node 到 dependent compute node 的 edge mask 高 | `global_memory_path` | 只能说明 global memory path evidence 强，不能单独说明 memory-bound。 |
| `LDS` / `STS` 明显，shared address pattern 或 bank-conflict counter 支持 | shared-memory neighborhood 在 motif 中稳定出现 | `shared_memory_path` | 需要 counter 或 microbench anchor 区分 bandwidth、bank conflict、latency。 |
| `FFMA` / FP32 arithmetic chain 在 motif 中稳定出现 | compute dependency chain 长，memory edge 弱 | `fp32_fp64_compute_pipeline` | 需要 dtype / opcode 子类区分 FP32 与 FP64。 |
| `DFMA` / FP64 arithmetic chain 在 motif 中稳定出现 | long dependent arithmetic chain | `fp32_fp64_compute_pipeline` | 可以进一步标记 `fp64_pipeline_like` subtype。 |
| `HMMA` / `MMA` 出现在 motif 中 | tensor op node/edge mask 高 | `tensor_core_pipeline` | 需要 shape/layout metadata 或 counter 支撑。 |
| register live range、fanout、spill/local memory 信号强 | high fanout value dependency 或 spill-related local load/store | `register_file_occupancy` | 不能只靠 register count；需要 occupancy/spill/counter 支持。 |
| `BAR.SYNC`、warp/block barrier 相关指令 | barrier fan-in / fan-out motif | `sync_barrier` | 后续调参可能关联 block size、CTA scheduling、shared-memory tiling。 |
| `BRA`、`SSY`、`SYNC`、predicate-heavy path | branch split / merge motif | `branch_control` | 需要 branch efficiency 或 divergence counter 支持。 |

## 7. Candidate Support Score

第一版 support score 只表示候选支持强度，不表示 calibrated probability。

推荐形式：

```text
support(structure)
  = normalized_sum(
      opcode_support,
      motif_support,
      static_support,
      counter_support,
      anchor_support
    )
```

其中每个子分数都要保留来源：

```json
{
  "structure": "global_memory_path",
  "support": 0.78,
  "support_breakdown": {
    "opcode_support": 0.70,
    "motif_support": 0.84,
    "static_support": 0.20,
    "counter_support": 0.88,
    "anchor_support": 0.71
  },
  "claim_status": "candidate_not_validated"
}
```

这里的 `0.78` 不是：

```text
78% memory bottleneck
```

而是：

```text
当前证据包对 global_memory_path 这个候选结构的支持强度较高。
```

## 8. 输出 Schema

建议输出：

```json
{
  "cluster_id": "cluster_03",
  "prototype_id": "proto_03",
  "hardware_structure_candidates": [
    {
      "structure": "global_memory_path",
      "support": 0.78,
      "support_breakdown": {
        "opcode_support": 0.70,
        "motif_support": 0.84,
        "static_support": 0.20,
        "counter_support": 0.88,
        "anchor_support": 0.71
      },
      "evidence": [
        "LDG/STG appear in explanation motif",
        "load-to-use edge has high explanation mask",
        "memory counter support is available"
      ],
      "claim_status": "candidate_not_validated"
    }
  ],
  "family_candidates": [
    {
      "family": "memory_latency_like",
      "support": 0.71,
      "source_structures": ["global_memory_path"],
      "claim_status": "weak_family_prior"
    }
  ],
  "knob_candidates": [
    {
      "knob": "dram_latency",
      "score": 0.66,
      "source_family": "memory_latency_like",
      "claim_status": "registry_constrained_prior_not_validated"
    }
  ]
}
```

## 9. 与 GCL / GNNExplainer 的关系

GCL 提供：

```text
kernel -> prototype / cluster
```

GNNExplainer 提供：

```text
prototype / cluster -> compact explanation motif + feature mask
```

本 spec 提供：

```text
compact motif + opcode/counter/static/anchor evidence
  -> hardware structure candidate
```

因此完整链路是：

```text
raw trace
  -> warp graph / kernel graph
  -> GCL encoder
  -> prototype assignment
  -> GNNExplainer motif
  -> hardware semantic binding
  -> family prior
  -> knob prior
  -> validation planner
```

## 10. 第一版实现步骤

### Step 1: 定义 Evidence Pack

为每个 kernel / cluster 生成：

```json
{
  "record_id": "kernel_or_cluster_id",
  "opcode_summary": {},
  "motif_summary": {},
  "static_resource_summary": {},
  "counter_summary": {},
  "anchor_similarity": {}
}
```

### Step 2: 实现 Fixed Taxonomy Matcher

建立一个可审计规则表：

```text
evidence_pattern -> hardware_structure_candidate
```

每条规则需要：

```text
rule_id
required_evidence
optional_evidence
support_weight
failure_reason
```

### Step 3: 生成 Hardware Structure Candidates

输出：

```text
candidate structure
support breakdown
evidence trace
claim status
```

### Step 4: 映射到 Family / Knob Registry

使用固定 registry：

```text
hardware structure -> mechanism family -> legal knob candidates
```

例如：

```text
global_memory_path -> memory_latency_like -> dram_latency/cache/coalescing knobs
register_file_occupancy -> occupancy_like -> register/warp/block-size knobs
```

### Step 5: 只验证 Top Candidates

由于一次 kernel validation 可能很慢，第一版只做：

```text
top-1 / top-3 candidate validation
```

低成本证据用于排序，昂贵实验只用于提升或否决候选。

## 11. 相关论文和资料

本 spec 对应论文保存在：

```text
papers/gpu-instruction-resource-mapping/
```

推荐阅读顺序：

1. `instruction-roofline-gpu-pmbs2019.pdf`

   先看这篇，因为它最直接说明如何把 GPU 性能分析下沉到 instruction categories 和 memory patterns。它能帮助我们建立“instruction-level evidence 可以支持资源候选，但不是 causal bottleneck truth”的论文表述。

2. `sassi-flexible-software-profiling-gpu-architectures-isca2015.pdf`

   这篇说明 SASS-level instrumentation 如何收集动态指令证据。它支撑我们的 raw trace / opcode evidence 来源。

3. `gpuscout-scw2023.pdf`

   这篇聚焦 data movement bottleneck localization。它适合作为 global-memory / data-movement family 的相关工作。

4. `nsight-compute-kernel-profiling-guide.pdf`

   这是 NVIDIA 官方 profiling metric 语义来源。我们的 counter evidence 名称和解释应该尽量与它对齐。

5. `gpu-performance-counters-kernel-characterization-iccs2020.pdf`

   这篇展示了用 performance counters 做 kernel characterization / classification 的传统路线。它可以作为我们区别于纯 counter-vector classifier 的 baseline。

另有一篇未保存 PDF：

```text
GPU code optimization using abstract kernel emulation and sensitivity analysis
```

它与 sensitivity-based knob reasoning 相关，但本次公开 PDF 下载返回 HTTP 522。可在后续通过 ACM/PNNL/机构访问补充。

## 12. 论文写法建议

可以写：

```text
We ground learned trace motifs with instruction-level and counter-level evidence,
and produce hardware-structure candidates under an explicit claim-status boundary.
```

不要写：

```text
Our method infers the exact hardware bottleneck from instruction ratios.
```

更稳妥的论文定位是：

```text
GCL learns structural similarity.
GNNExplainer extracts compact motif evidence.
Hardware semantic binding maps motif evidence to hardware-structure candidates.
Validation promotes or rejects top-ranked tuning candidates.
```
