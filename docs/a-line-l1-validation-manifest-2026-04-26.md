# A 线 L1 基础验证集 Manifest（第一版）

日期：2026-04-26

## 1. 目的

这份 manifest 用于定义：

**A 线 / B 线当前最先应落地的一批 `L1` 基础验证对象。**

L1 的角色不是做大规模 compression benchmark，
而是做：

- functionality gate
- feature sanity gate
- downstream interface gate

因此 L1 必须满足：

- 行为明确
- 路径稳定
- 人工可检查
- 尽量优先使用仓库内已有结果

---

## 2. 推荐规模

第一版建议控制在：

- `12 ~ 18` 个对象

这样既足够覆盖主行为轴，
又不会让第一轮 bring-up 失控。

---

## 3. L1 对象清单

| ID | 来源 | 对象 | 本地路径 / 来源路径 | 优先级 | 面向线路 | 预期行为轴 | 主要作用 | 当前状态 |
|---|---|---|---|---|---|---|---|---|
| `L1_MB_01` | microbench | `l1_bw_32f` | [l1_bw_32f.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/microbench/l1_bw_32f.json) | `P0` | `A+B` | `L1 bandwidth / coalesced load-heavy` | feature sanity, anchor sanity | `ready_local` |
| `L1_MB_02` | microbench | `l2_bw_32f` | [l2_bw_32f.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/microbench/l2_bw_32f.json) | `P0` | `A+B` | `L2 / global-memory bandwidth` | feature sanity, anchor sanity | `ready_local` |
| `L1_MB_03` | microbench | `mem_bw` | [mem_bw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/microbench/mem_bw.json) | `P0` | `A+B` | `global-memory bandwidth` | memory-axis sanity | `ready_local` |
| `L1_MB_04` | microbench | `mem_lat` | [mem_lat.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/microbench/mem_lat.json) | `P0` | `A+B` | `global-memory latency` | memory-axis sanity | `ready_local` |
| `L1_MB_05` | microbench | `shared_bw` | [shared_bw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/microbench/shared_bw.json) | `P0` | `A+B` | `shared-memory throughput` | shared-axis sanity | `ready_local` |
| `L1_MB_06` | microbench | `shared_lat` | [shared_lat.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/microbench/shared_lat.json) | `P1` | `A+B` | `shared-memory latency` | shared-axis sanity | `ready_local` |
| `L1_MB_07` | microbench | `atomic_add_bw` | [atomic_add_bw.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/microbench/atomic_add_bw.json) | `P1` | `A+B` | `atomic-heavy / serialization-sensitive` | atomic axis sanity | `ready_local` |
| `L1_MB_08` | microbench | `atomic_add_lat` | [atomic_add_lat.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/microbench/atomic_add_lat.json) | `P1` | `A+B` | `atomic latency / contention-sensitive` | atomic axis sanity | `ready_local` |
| `L1_MB_09` | microbench | `MaxFlops` | [MaxFlops.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/microbench/MaxFlops.json) | `P0` | `A+B` | `compute-bound` | compute axis sanity | `ready_local` |
| `L1_RD_01` | Rodinia | `nn` | [nn_trace.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/rodinia/nn_trace.json) | `P0` | `A+B` | `distance / memory-sensitive / possible uncoalesced global access` | real-kernel sanity | `ready_local` |
| `L1_RD_02` | Rodinia | `backprop` | [backprop_4096_prescription_v1.md](/home/dyf/modern-gpu-simulator-micro-2025/experiments/baseline_diagnosis/results/rodinia/backprop_4096_prescription_v1.md) | `P1` | `A+B` | `dense numeric / low-divergence` | real-kernel sanity | `ready_local` |
| `L1_AI_01` | AI workload | `gemm_tiled` | [mini_transformer_v4_full.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_full.json) | `P0` | `A+B` | `dense compute backbone` | target-kernel sanity | `ready_local` |
| `L1_AI_02` | AI workload | `attention_score` | [mini_transformer_v4_full.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_full.json) | `P0` | `A+B` | `pairwise score / dense compute` | target-kernel sanity | `ready_local` |
| `L1_AI_03` | AI workload | `softmax_kernel` | [mini_transformer_v4_full.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_full.json) | `P0` | `A+B` | `reduction / normalize` | target-kernel sanity | `ready_local` |
| `L1_AI_04` | AI workload | `context_mul` | [mini_transformer_v4_full.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_full.json) | `P1` | `A+B` | `streaming aggregation` | target-kernel sanity | `ready_local` |
| `L1_AI_05` | AI workload | `layernorm_kernel` | [mini_transformer_v4_full.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_full.json) | `P1` | `A+B` | `reduction / normalize` | target-kernel sanity | `ready_local` |
| `L1_AI_06` | AI workload | `residual_add` | [mini_transformer_v4_full.json](/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_full.json) | `P1` | `A+B` | `elementwise / lightweight` | target-kernel sanity | `ready_local` |

---

## 4. L1 使用方式

### 4.1 A 线

L1 用于检查：

- PKA feature extraction 是否稳定
- baseline grouping 是否明显依赖错误字段
- representative object 是否可解释

### 4.2 B 线

L1 用于检查：

- anchor 输出是否可被 family / regime builder 稳定消费
- downstream interface 是否完整
- writeback 所需最小字段是否齐备

---

## 5. L1 验收重点

L1 的重点不是：

- 压缩率最大化

而是：

- 输出结构稳定
- 行为标签与特征空间基本一致
- downstream consumption 无断裂
- 改代码后能快速发现语义回归

---

## 6. 当前最建议的执行顺序

1. 先对 `L1_MB_01 ~ L1_MB_09` 做 feature audit
2. 再把 `L1_RD_01 ~ L1_RD_02` 接进 `pka_baseline`
3. 最后用 `L1_AI_01 ~ L1_AI_06` 验证目标 workload 对齐情况

---

## 7. 备注

- 如果 L1 都无法稳定通过，不应继续扩大 L2。
- L1 是 `functionality gate`，不是 `compression quality gate`。
