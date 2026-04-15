# Boundary Case: `gemm_tiled` vs `attention_score`

## Case Goal

这份边界 case 文档用于回答一个核心问题：

**`gemm_tiled` 与 `attention_score` 是否应被视为共享同一架构解释的 family，还是应因为 shared memory 等实现差异被拆开。**

本轮分析采用的原则是：

- 同时记录共享点与区分点
- 以区分点分析为主
- 结论采用分级判断，而不是直接做绝对并类裁决

## Shared Points

### 1. 二者都明显属于 compute-heavy 区域

来自 [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md) 的 kernel 指标表：

- `gemm_tiled`: `compute=90.9%`, `warp_cyc=36.3`
- `attention_score`: `compute=95.2%`, `warp_cyc=34.0`

这说明二者在基线观测上都落在高计算吞吐、低等待的区域，而不是 memory-heavy 区域。

### 2. 二者共享最严格的寄存器限制信号

同样来自 [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md)：

- `gemm_tiled`: `block_limit_registers=6`
- `attention_score`: `block_limit_registers=6`

在同一张表中，这也是当前六类 kernel 中最严格的一档寄存器限制信号。

### 3. 多机制综合解释指向同一主机制

[E4_full.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E4_full.md) 的“发现 C-1”明确给出：

- 两者共享计算瓶颈
- 三机制一致收敛到**寄存器文件大小**是主瓶颈
- 对应 Stage C 处方都优先指向寄存器 / occupancy 相关方向

因此，从多机制综合解释看，二者存在共享同一架构解释的强证据。

## Distinguishing Points

### 1. `attention_score` 仍然是 batch 里的良性 outlier

[E2_batch.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E2_batch.md) 明确指出：

- `attention_score` 没有并入 `gemm_tiled` 聚类
- 原因不是软件问题，而是良性架构异质
- 其显著差异来自：
  - `shmem=8256B`
  - `waves=25.6`

这意味着：

**即便它和 `gemm_tiled` 共享主解释，它仍然具有不能被忽略的实现层差异。**

### 2. shared memory 特征可能改变次级解释路径

在 [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md) 中：

- `gemm_tiled`: `shmem(B)=2048`
- `attention_score`: `shmem(B)=8256`

而 [E4_full.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E4_full.md) 里也明确说：

- `attention_score` 的处方在共享寄存器主线之外，还要额外考虑 shared memory bank 配置

因此，这个差异不一定推翻 shared family，但至少说明：

**`attention_score` 不能被粗暴地当成纯 GEMM 的等价副本。**

### 3. Batch 与 E4 在这里形成“共享主解释 + 保留实现异质”的张力

这是这一对 case 最有价值的地方：

- `E4_full` 支持二者共享主解释
- `E2_batch` 支持二者保留异质边界

所以这不是一个“完全能并类”或“完全不能并类”的简单问题，而是一个典型的：

**共享主机制，但次级实现特征仍值得保留。**

## Graded Conclusion

**结论等级：弱共享**

理由：

- 共享点足够强：执行模式和主导资源解释都高度重合
- 但区分点也足够强：`attention_score` 的 shared memory / waves 特征不能在第一版被抹平

因此，第一轮不宜把它们视为“完全同质”的强共享 family 样本，但可以视为：

**共享同一主架构机制、但保留次级实现异质的弱共享样本。**

## Current Execution Advice

### Family 划分建议

- 当前阶段可将两者放在同一候选 family 的边界讨论中
- 该候选 family 的主解释可暂定为：
  - `compute-heavy -> register-limited`
- 但 `attention_score` 必须在 analysis card 中保留 shared-memory-coupled 的次级说明

### 验证组织建议

- 当前阶段可优先共享寄存器 / occupancy 主线的验证思路
- 但不要因此删除 `attention_score` 的 shared memory 差异记录
- 若后续 family 判据继续成熟，才决定是否把它完全并入 `gemm_tiled` 代表的同一条验证主线

## Evidence References

- [E0_baseline.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md): “每 Kernel 关键指标（v4，6 层均值）”
- [E2_batch.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E2_batch.md): “Batch 输出：3 聚类 + 3 outlier”
- [E4_full.md](../../../../experiments/baseline_diagnosis/results/mini_transformer_v4/E4_full.md): “发现 C-1：gemm_tiled + attention_score 共享计算瓶颈”
