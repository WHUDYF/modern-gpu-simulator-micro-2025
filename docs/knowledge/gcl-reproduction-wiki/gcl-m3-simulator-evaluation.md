# GCL-M3 Simulator Evaluation

GCL-M3 是 simulator and cross-architecture evaluation stage。

它接收 [[gcl-m2-rgcn-embedding-and-selector]] 产出的 representative anchors，并评估这些代表点是否能保持 full workload 的 simulator-relevant metrics。

## 核心问题

M3 回答的问题是：

```text
GCL-selected representatives 能否用较少 simulation points 重建 full workload metrics？
```

这和 M0/M1/M2 的 structural validation 不同。M3 才允许讨论 sampled simulation accuracy。

## 输入

M3 输入：

```text
gcl_representative_anchor_table_l1.json
full workload metric table
sampled representative metric table
optional cross-GPU metric tables
```

## 输出

M3 输出：

```text
gcl_simulator_accuracy_l1.json
gcl_microarchitectural_metric_error_l1.json
gcl_cross_architecture_transfer_l1.json
```

## Evaluation

M3 应比较 sampled reconstruction 与 full workload metrics。

重点指标包括：

```text
cycles
IPC
cache metrics
occupancy metrics
selected microarchitectural metrics
```

M3 必须区分：

```text
structural compression ratio
measured simulator speedup
sampled reconstruction error
```

不能把 structural compression ratio 直接说成 measured speedup。

## 与 PKA 的比较

当 PKA 和 GCL 都可用时，M3 应在同一 workload set 上比较：

```text
PKA selected representatives
GCL selected representatives
full workload metrics
sampled reconstructed metrics
```

M3 是最终回答 GCL 是否优于 PKA 的地方。这个 claim 不能在 [[gcl-m0-offline-embedding-selector]]、[[gcl-m1-trace-graph-construction]] 或 [[gcl-m2-rgcn-embedding-and-selector]] 中提前提出。

