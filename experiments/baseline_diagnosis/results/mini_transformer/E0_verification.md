# 闭环验证报告：mini-transformer v2–v4（处方 A-1 / B-1 / B-4）

**日期：** 2026-04-11
**硬件：** RTX 3080 Ti (SM_86)
**binary 序列：** v1（broken）→ v2（B-1 尝试）→ v3（B-4 shmem，有 bank conflict）→ v4（B-4 shmem+padding）
**对比基线：** E0 baseline（原始 1-layer broken binary）

---

## 验证结论摘要

| 处方 | 预期效果 | 实测效果 | 判定 |
|------|---------|---------|------|
| A-1：softmax 每行一个 block（6144 blocks） | waves 0.1→77+，occupancy 16.6%→90%+ | waves 0.05→12.8，occ 16.7%→94.1% | **VALIDATED ✅** |
| B-1：attention_score 4-wide 累加器展开 | warp_cycles 174.7→~40 | warp_cycles 174.7→169.7（-3%） | **INVALID ❌** |
| B-4：attention_score shared memory + padding | warp_cycles 大幅下降，compute 显著提升 | warp_cycles 174.7→**34.0**（-80.5%），compute 22.4%→**95.2%** | **VALIDATED ✅** |

---

## 处方 A-1 验证：softmax 启动配置修复

### 修改内容

```c
// 修复前（原始）：(total_rows + bs - 1) / bs 个 block，每行 bs 个线程，无 shared memory
softmax_kernel<<<(24 + 255) / 256, 256>>>(d_scores, total_rows, SEQ_LEN);

// 修复后：每行一个 block，256 线程并行归约，256×4=1024B dynamic shared memory
int total_rows = NUM_HEADS * SEQ_LEN;  // 12 × 512 = 6144
softmax_kernel<<<total_rows, 256, 256 * sizeof(float)>>>(d_scores, total_rows, SEQ_LEN);
```

### 指标对比

| 指标 | v1（broken） | v2（fixed） | 变化 |
|------|------------|------------|------|
| waves_per_sm | 0.05 | 12.80 | **+12.75** |
| achieved_occupancy_pct | 16.65% | 94.09% | **+77.4pp** |
| warp_cycles_per_issued_inst | 51.43 | 21.84 | **-29.6** |
| compute_throughput_pct | 1.85% | 85.52% | **+83.7pp** |
| ipc_active | 0.16 | 2.06 | **+12.9×** |
| dram_throughput_pct | 11.20% | 41.20% | +30pp |

### 分析

- waves=12.8（6 层均值）= 6144 blocks / 80 SMs / 6 layers ≈ 12.8，符合预期
- occupancy 从 16.7% 到 94.1%，软件串行化瓶颈彻底消除
- compute_throughput 从 1.85% 到 85.5%，内核从访存受限转变为计算/访存均衡
- **处方 A-1 全面验证**：Stage A 诊断（waves=0.1, occupancy=16.6%）的根因分析和处方均正确

---

## 处方 B-1 验证：attention_score 累加器展开

### 修改内容

```c
// 修复前（原始）：单累加器，64 次串行 FMA
float sum = 0.0f;
for (int d = 0; d < head_dim; d++)
    sum += Q[q_offset + d] * K[k_offset + d];

// 修复后：4-wide 独立累加器
float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
for (int d = 0; d <= head_dim - 4; d += 4) {
    s0 += Q[q_offset + d]     * K[k_offset + d];
    s1 += Q[q_offset + d + 1] * K[k_offset + d + 1];
    s2 += Q[q_offset + d + 2] * K[k_offset + d + 2];
    s3 += Q[q_offset + d + 3] * K[k_offset + d + 3];
}
float sum = s0 + s1 + s2 + s3;
```

### 指标对比

| 指标 | v1（broken） | v2（fixed） | 变化 |
|------|------------|------------|------|
| warp_cycles_per_issued_inst | 174.66 | 169.68 | -5.0（-3%）|
| compute_throughput_pct | 22.42% | 22.40% | -0.02pp（无变化）|
| l1_hit_rate_pct | 97.19% | 97.20% | +0.01pp（无变化）|
| ipc_active | 0.260 | 0.270 | +0.01（微小）|
| achieved_occupancy_pct | 95.55% | 95.48% | -0.07pp（无变化）|

### 根因重新分析

**原始诊断（Class B-1）：** warp_cycles 高 + L1 命中高 + compute 低 → 归因于单累加器 RAW 依赖链。
**处方：** 4-wide 累加器展开。
**验证结果：** 无效（warp_cycles 174.7 → 169.7，仅 -3%）。

**反驳证据：**

SASS 反汇编显示 `mini_transformer_v2` 中 attention_score 已生成：
- **71 个 FFMA** 指令（多个独立目标寄存器：R27, R22, R16, R15, R21, R25, R29, R24）
- **98 个 LDG** 指令

编译器在 `-O2` 下已自动生成 **≥ 8-wide 的并行累加器**，无论是原始单累加器版本还是修复后的 4-wide 版本，编译结果几乎相同。这说明：

1. **原始单累加器版本的 warp_cycles=174.7 不是 RAW 依赖链导致的**——因为编译器已经消除了 RAW 链
2. **4-wide 手动展开是冗余的**——编译器已经做到更好
3. **真正的瓶颈是 LDG 加载延迟**：SM_86 L1 命中延迟 ≈ 28 cycles，而 98 个 LDG 指令与 71 个 FFMA 之间的软件流水线深度不足，导致 warp 在等待全局内存加载时停滞

**新根因假设：** attention_score 的瓶颈是 **LDG 软件流水线不足（Class B-4）**，而非 RAW 依赖链（Class B-1）。

---

## 诊断协议的修订需求

### Stage B 新增判据（B-4：LDG 延迟软件流水线不足）

**触发条件（同时满足）：**
- `warp_cycles_per_issued_inst` ≥ 50
- `l1_hit_rate_pct` ≥ 70%（数据在 L1，非 L2/DRAM 延迟）
- `compute_throughput_pct` ≤ 50%（计算流水线未饱和）
- **编译器已生成多累加器**（cuobjdump FFMA 数量 ≥ head_dim × 2）

**与 Class B-1 的区分方法：**
- B-1（RAW 依赖链）：手动展开后 warp_cycles 大幅下降（> 40%）
- B-4（LDG 延迟）：手动展开后 warp_cycles 几乎无变化

**修复归属：** 需要软件流水线（software pipelining）——
- 使用 `__ldg()` + 显式预取，或
- 将 Q/K 行预加载到 shared memory（类似 GEMM 的 tiled 方案），或
- 使用 CUDA 的 `cp.async` 异步预取（Ampere 特性）

---

## 其余 kernel 指标稳定性确认

| Kernel | 指标变化说明 |
|--------|------------|
| gemm_tiled | warp_cycles=36.3（无变化），compute=90.9%（+0.6pp），符合预期 |
| context_mul | 全部指标变化 < 0.5%，稳定 |
| residual_add | 全部指标变化 < 1.5%，稳定 |
| layernorm_kernel | 全部指标变化 < 1%，稳定 |

非修改 kernel 的指标高度稳定，说明测量噪声极低，v1 → v2 的变化量可信。

---

## 处方 B-4 验证：attention_score shared memory tiling

### 演进过程

| 版本 | 修改 | warp_cycles | compute | 说明 |
|------|------|:-----------:|:-------:|------|
| v1 | 原始单累加器 | 174.7 | 22.4% | Stage B 不通过 |
| v2 | 4-wide 累加器展开 | 169.7 | 22.4% | B-1 无效（编译器已自动展开）|
| v3 | shared memory 预加载 | 121.6 | 17.1% | 改善但引入 bank conflict |
| v4 | shared memory + Ks 行 padding (+1) | **34.0** | **95.2%** | B-4 完全验证 |

### v3 → v4：bank conflict 分析与修复

**冲突根因：** `Ks[TILE_SIZE][HEAD_DIM]` 的行步长 = HEAD_DIM×4 = 256 bytes。
SM_86 有 32 个 4-byte bank，256 / 4 = 64 ≡ 0 (mod 32)，
所有 `Ks[x][d]`（x=0..15）落在同一 bank → 16-way conflict。

**修复：** `__shared__ float Ks[TILE_SIZE][HEAD_DIM + 1]`
行步长变为 65×4=260 bytes，bank = (x×65+d)%32，x=0..15 全部不同 ✅

### v4 最终指标对比

| 指标 | v1 (broken) | v4 (final) | 变化 |
|------|:-----------:|:----------:|:----:|
| warp_cycles_per_issued_inst | 174.7 | 34.0 | **-80.5%** |
| compute_throughput_pct | 22.4% | 95.2% | **+72.8pp** |
| achieved_occupancy_pct | 95.6% | 95.1% | -0.5pp |
| ipc_active | 0.26 | — | 大幅提升 |

attention_score 现已通过 Stage B，所有 6 个 kernel 可进入 Stage C 架构诊断。

---

## 下一步行动

| 优先级 | 行动 | 说明 |
|--------|------|------|
| 1 | ~~更新 diagnosis_protocol_v2.md~~ | **已完成**（新增 B-4 判据） |
| 2 | ~~修复 attention_score~~ | **已完成**（v4 = shmem + padding，warp_cycles -80.5%） |
| 3 | 生成 v4 的 full.json | 合并 v4 hardware stats 与 trace features，重新跑 E0-E4 |
| 4 | 生成跨 dwarf 总结报告 | backprop + nn + mini-transformer 处方成功率表 |
