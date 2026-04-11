# 诊断报告：mini-transformer [E4_full]

**日期：** 2026-04-10
**硬件：** RTX 3080 Ti (SM_86)
**输入特征：** mini_transformer_full.json + squash.json + batch.json + delta.json
**启用机制：** Squash + Batch + Delta

---

## 多机制综合分析

三个机制各自提供独立的证据流，多个机制同时指向同一发现时置信度最高。

---

## 发现 1：attention_score 是计算串行化瓶颈（三重收敛 — HIGH）

**E0：** IPC = 0.26，warp_cycles = 174.7，L1 命中率 = 97.2%，compute = 22.4% → 内层循环 RAW 依赖链。

**Squash 补充：** attention_score 位于段 1，凝聚度 = 0.850（所有段中最低）。
attention_score 与 softmax 之间的行为差异——尽管它们在执行顺序上相邻——
得到机器化确认。Squash 提供了"相位边界"证据：该 kernel 从 GEMM 聚类中急剧切换行为。

**Batch 补充：** attention_score 是 outlier kernel——它与 GEMM（计算特征不同）、
residual_add（内存特征不同）和 layernorm（occupancy 特征不同）均无法聚类。
这机器化地将其认证为首要调查目标。

**Delta 补充：** `l1_hit_rate ↔ compute_throughput` 反相关（-0.646）将
attention_score 标注为"L1 命中高、计算吞吐低"规律的最极端代表。
Delta 使因果关系显式：数据已就绪（L1 命中），但计算流水线无法消化（串行化）。

**综合结论：** 三个独立机制分别将 attention_score 标记为结构性异常。
三重收敛排除了测量噪声作为替代解释的可能，为处方 B-1 提供了最高可能的置信度。

**处方 B-1（HIGH，三重确认）：** 使用 4-wide 独立累加器展开内层循环。
预期：IPC 0.26 → 1.0+，warp_cycles 174.7 → ~40。

---

## 发现 2：softmax 启动配置不足（E0 + Batch — HIGH）

**E0：** occupancy = 16.6%，waves = 0.1 → 80 个 SM 上仅 24 个 block。

**Batch 补充：** softmax 是 outlier kernel，部分原因在于其极低的 occupancy
（其余 kernel 均 ≥ 75%）。Batch 的 outlier 标注无需扫描全部 14 行就能
直接引导调查到 softmax。

**Squash 补充：** softmax 出现在段 1 与 attention_score 并列（凝聚度 0.850）。
段内的异质性确认 softmax 与 attention_score 行为截然不同，需要独立处方。

**综合结论：** E0 + Batch 一致。置信度高。

**处方 A-1（HIGH）：** 每行一个 block，grid = 6144。

---

## 发现 3：GEMM 是计算密集型 + 寄存器限制 occupancy（E0 + Batch — MEDIUM）

**Batch 补充：** 全部 7 个 GEMM 实例形成凝聚度 = 1.0 的单一聚类。
一个处方覆盖全部 7 个实例。

**Squash 补充：** GEMM 实例分布在三个独立段（段 0：4 个 QKV GEMM，
段 2：输出 GEMM 与 context_mul 并列，段 5：2 个 FFN GEMM）。
分段揭示出 GEMM 服务于 Transformer 中不同角色（QKV / 输出 / FFN），
即便它们的行为特征完全相同。这意味着模拟复用（用单一 GEMM trace 代表全部 7 个）
是有效的。

**Delta 补充：** `l1_throughput ↔ dram_throughput` 强反相关（-0.940）将 GEMM
定位在"低 DRAM，中等 L2"区间——数据驻留在 L2（tile 加载一次，后续计算全在
shared memory 和寄存器中）。这确认了 GEMM tile 策略有效，瓶颈在计算流水线
吞吐而非内存带宽。

**综合结论：** MEDIUM 置信度。处方（TILE_SIZE 16→32）需要模拟器验证实际 IPC 提升。

**处方 B-2（MEDIUM）：** TILE_SIZE 16 → 32。

---

## 发现 4：特征提取 bug — uses_shared_memory 恒为 0（Delta — HIGH，正确性）

**Delta：** COLD 字段，温度 = 0.0，但 GEMM 和 layernorm 均使用了 static shared memory。
根因：特征提取只检查 `dynamic_shmem_per_block`，未考虑 `static_shmem_per_block`。

此 bug 影响未来对 GEMM warp stall 的诊断：AI 不知道 shared memory bank conflict
是潜在成因。

**行动（非处方）：** 修复 `extract_trace_features.py`，在所有现有特征 JSON 重新
提取前，不建议对 GEMM 的 warp stall 问题做进一步诊断。

---

## 发现 5：三种内存访问机制（Delta — 信息性）

| 机制 | Kernel | L1 吞吐 | DRAM 吞吐 | 策略含义 |
|------|--------|---------|---------|---------|
| L1 驻留 | attention_score, context_mul | 高 | 低 | 缓存复用，计算密集 |
| L2 驻留 | gemm_tiled | 中等 | 低 | Shared memory tiling |
| HBM 流式 | residual_add | 低 | 高 | 带宽受限，可考虑融合 |
| 混合 | layernorm, softmax | — | — | 归约 + 计算 |

**对模拟器的启示：** 模拟器需要对这三种机制分别精确建模。
正确建模 GEMM（L2 带宽 + shared memory）的配置，可能错误预测 residual_add
（HBM 带宽）的性能——如果 HBM 带宽模型不准确。这提供了一个结构化的模拟器校准测试套件。

---

## 完整处方表（E4）

| ID | Kernel | 类别 | 修改内容 | 置信度 | 证据来源 |
|----|--------|------|---------|--------|---------|
| B-1 | attention_score | B | 4-wide 累加器展开 | **HIGH** | E0 + Squash + Batch + Delta |
| A-1 | softmax | A | 每行一个 block | **HIGH** | E0 + Batch + Squash |
| B-2 | gemm_tiled | B | TILE_SIZE 16 → 32 | MEDIUM | E0 + Batch + Delta |
| 修复 | 特征提取脚本 | 系统 | uses_shared_memory = static + dynamic | HIGH | Delta |
| — | residual_add | — | 与 layernorm 融合（可选） | LOW | E0 |

---

## 跨机制一致性矩阵

| 发现 | E0 | Squash | Batch | Delta |
|------|----|--------|-------|-------|
| attention_score RAW 依赖链 | ✅ | ✅（低凝聚度） | ✅（outlier） | ✅（反相关规律） |
| softmax 利用率不足 | ✅ | ✅（段 1 分裂） | ✅（outlier） | — |
| GEMM 寄存器限制 | ✅ | ✅（均匀段） | ✅（聚类） | ✅（L2 驻留） |
| uses_shared_memory bug | — | — | — | ✅ |
| 内存三态分解 | 部分 | — | 部分 | ✅ |
