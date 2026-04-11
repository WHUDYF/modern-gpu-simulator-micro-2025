# 诊断报告：mini-transformer [E0_baseline]

**日期：** 2026-04-10
**硬件：** RTX 3080 Ti (SM_86)
**输入特征：** mini_transformer_full.json（NVBit trace + NCU 硬件统计）
**启用机制：** 无
**Kernel 数量：** 14（1 层 Transformer：注意力 + FFN）

---

## Stage A：软件利用率检查

### 利用率指标

| Kernel（代表） | waves_per_sm | achieved_occupancy | 网格配置 |
|--------------|--------------|---------------------|---------|
| gemm_tiled（k01-04, k08, k11-12） | 3.2 | 89.0% | 48×32×1 |
| attention_score（k05） | 25.6 | 95.5% | 32×32×12 |
| softmax（k06） | 0.1 | 16.6% | 24×1×1 |
| context_mul（k07） | 3.2 | 90.3% | 32×32×12 |
| residual_add（k09, k13） | 3.2 | 75.1% | — |
| layernorm（k10, k14） | 1.1 | 80.1% | 512×1×1 |

### Stage A 结论

- **softmax（k06）：不通过** — waves_per_sm = 0.1，occupancy = 16.6%。
  网格 = ceil(NUM_HEADS × SEQ_LEN / 256) = ceil(6144 / 256) = 24 个 block。
  在 80 个 SM 的 GPU 上，大部分 SM 分配到 0 个 block。这是启动配置缺陷，
  不是架构瓶颈。
- 其余 kernel：通过（waves ≥ 1.0，occupancy ≥ 75%）。

**Class A 处方（softmax）：** 改为每行一个线程启动（6144 个线程，按 256 分块
= 24 × 256），或使用 warp 级并行归约，网格 = (NUM_HEADS × SEQ_LEN) × 1 × 1。

---

## Stage B：架构瓶颈分析

### 各 Kernel 瓶颈概览

| Kernel | dram% | compute% | l1命中% | ipc | warp周期 | 主要瓶颈 |
|--------|-------|----------|--------|-----|---------|---------|
| gemm_tiled | 15.6 | 90.3 | 2.4 | 1.18 | 36.3 | 计算密集（寄存器限制） |
| attention_score | 1.6 | 22.4 | 97.2 | 0.26 | 174.7 | **RAW 依赖链** |
| softmax | 11.2 | 1.9 | 91.1 | 0.16 | 51.4 | （Class A：利用率不足） |
| context_mul | 7.4 | 89.4 | 88.5 | 1.37 | 31.6 | 计算密集（近最优） |
| residual_add | 60.4 | 14.9 | 33.3 | 0.38 | 87.0 | HBM 带宽瓶颈 |
| layernorm | 21.9 | 47.6 | 75.0 | 1.47 | 25.7 | 混合（近最优） |

---

### 发现 1：attention_score — RAW 依赖链（HIGH）

**信号：** ipc = 0.26，warp_cycles = 174.7，L1 命中率 = 97.2%，compute = 22.4%。

高 L1 命中率（数据已从缓存取出）与极低 IPC 及超长 warp 停顿周期（174.7）并存，
排除了内存延迟作为原因。瓶颈是**寄存器级 RAW（先写后读）依赖链**。

内层循环计算如下：
```c
for (int d = 0; d < head_dim; d++)    // head_dim = 64
    sum += Q[offset + d] * K[offset + d];
```

单累加器（`sum`）使得每次 FMA 都依赖上一次的结果。CUDA FMA 有多周期执行延迟
（通常 4–8 周期）。没有多个独立累加器时，64 次迭代完全串行。
warp_cycles = 174.7 ≈ head_dim（64）× FMA 延迟 / 发射宽度，与串行依赖一致。

**处方 B-1（HIGH 置信度）：**
- 使用 4–8 个独立累加器展开内层循环以打破依赖链：
  ```c
  float s0=0, s1=0, s2=0, s3=0;
  for (int d = 0; d < head_dim; d += 4) {
      s0 += Q[..+d]   * K[..+d];
      s1 += Q[..+d+1] * K[..+d+1];
      s2 += Q[..+d+2] * K[..+d+2];
      s3 += Q[..+d+3] * K[..+d+3];
  }
  sum = s0 + s1 + s2 + s3;
  ```
  预期效果：IPC 从 0.26 提升至 ~1.0+；warp_cycles 从 174.7 降至 ~40。

---

### 发现 2：softmax — 启动配置不足（HIGH，Class A）

已在 Stage A 诊断。24 个 block 对应 80 个 SM。

**处方 A-1（HIGH 置信度）：** 改为每行一个 block，网格 = 6144。
预期：waves 从 0.1 提升至 ~77，occupancy 从 16.6% 提升至 60%+。

---

### 发现 3：gemm_tiled — 寄存器限制 occupancy（MEDIUM）

**信号：** compute = 90.3%，block_limit_registers = 6，block_limit_warps = 6，
achieved_occupancy = 89.0%，warp_cycles = 36.3。

寄存器和 warp 限制均报告 block_limit = 6，achieved_occupancy 已达理论上限的 89%。
warp_cycles = 36.3 对于使用 shared memory tile 的 GEMM 偏高。tile buffer 占用
2 × TILE_SIZE² × 4 = 2 KB per block，远小于 SM 容量。全局内存 L1 命中率仅 2.4%
（符合预期：每个 tile 仅从全局内存加载一次，之后在 shared memory 中复用）。

**处方 B-2（MEDIUM 置信度）：**
- 将 TILE_SIZE 从 16 增至 32，提升每字节全局内存加载对应的算术强度。
  代价：shared memory 从 2 KB 增至 8 KB per block，可能减少每 SM 的 block 数。
- 备选：寄存器 tiling（2D 寄存器累加数组）提升 shared memory 加载的计算复用。

---

### 发现 4：residual_add — HBM 带宽瓶颈（LOW）

**信号：** DRAM = 60.4%，compute = 14.9%，warp_cycles = 87.0。

对两个 1.5 MB 张量（SEQ_LEN × HIDDEN_DIM = 393216 个 float）做元素级加法，
本质上是带宽受限操作。60.4% HBM 利用率接近流式访问的可达效率。
warp_cycles = 87.0 对应 HBM 延迟（~200–300 周期）被 3.2 waves 的 warp 部分掩盖。

**处方：** 不需要单独优化。可考虑将 residual_add 与 layernorm 融合，减少一次内存往返。

---

### 发现 5：context_mul — 近最优（LOW）

**信号：** compute = 89.4%，L1 命中率 = 88.5%，ipc = 1.37，warp_cycles = 31.6。

scores × V 矩阵乘法，得益于 scores 矩阵（12 × 512 × 512 × 4B = 12 MB，驻留 L2）
的良好数据复用。IPC = 1.37、compute = 89.4%，已接近计算上限，无需优化。

---

## 处方汇总

| ID | Kernel | 类别 | 修改内容 | 置信度 | 预期效果 |
|----|--------|------|---------|--------|---------|
| A-1 | softmax | A | 每行一个 block，grid = 6144 | HIGH | waves 0.1→77，occupancy 17%→60%+ |
| B-1 | attention_score | B | 4-wide 累加器展开 | HIGH | warp_cycles 174→40，IPC 0.26→1.0+ |
| B-2 | gemm_tiled | B | TILE_SIZE 16→32 | MEDIUM | 算术强度 ×2，stall 减少 |
| — | residual_add | — | 与 layernorm 融合（可选） | LOW | 减少一次内存往返 |

---

## E0 局限性（机制待填补的盲区）

1. 无法量化**哪些 kernel 是结构性 outlier**，哪些属于主导 cluster — 需要 Batch。
2. 无法识别**计算阶段边界**（不依赖 kernel 命名） — 需要 Squash。
3. L1 命中率与计算吞吐的**反相关规律**逐行可见但不能跨 kernel 系统化呈现 — 需要 Delta。
4. `uses_shared_memory` 字段对 GEMM 记录为 0，尽管 GEMM 使用了 2 KB static shared memory — 特征提取 bug，可由 Delta cold-field 分析捕获。
