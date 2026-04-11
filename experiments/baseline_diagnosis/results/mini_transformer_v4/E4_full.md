# 诊断报告：mini-transformer v4 [E4_full]

**日期：** 2026-04-11
**硬件：** RTX 3080 Ti (SM_86)
**启用机制：** Squash + Batch + Delta（三机制完整）

---

## 多机制综合 Stage C 发现

### 发现 C-1：gemm_tiled + attention_score 共享计算瓶颈（三重收敛 — HIGH）

**E0：** 两者 compute≈90%+，warp_cycles≈34-36，block_limit_registers=6。

**Squash 补充：** 两者同处段 0（cohesion=0.933），行为特征已高度收敛——
这在 v1 中不可能，因为 v1 的 attention_score 是异常段的制造者。
段 0 的高凝聚度确认了 attention_score v4 与 GEMM 在执行特征上已等价。

**Batch 补充：** attention_score 仍是 outlier（因 shmem=8256B 和 waves=25.6 与 GEMM 不同），
但现在是"良性计算异质"，不再是"问题 kernel"。
同一 outlier 在 v1 vs v4 中的成因完全不同——这是 Batch 区分软件/架构问题的直接证明。

**Delta 补充：** `block_limit_registers` 是 v4 中与 compute/L1 相关性最强的字段（-0.946/-0.964）。
两者均为 block_limit_registers=6，是该规律的最极端代表。

**综合结论：** 三机制一致指向——gemm 和 attention_score 的瓶颈是**寄存器文件大小**，
模拟器的 `gpgpu_shader_registers` 配置必须精确才能正确预测这两个 kernel 的 occupancy 和 IPC。

**处方 C-1（HIGH）：** 验证并校准模拟器寄存器文件配置；FP32 initiation interval。

---

### 发现 C-2：residual_add 是孤立的 HBM 带宽瓶颈（E0 + Delta + Batch — HIGH）

**E0：** DRAM=58.3%，compute=14.7%，warp_cycles=87.6。

**Batch 补充：** residual_add 形成独立聚类（n=2，凝聚度=1.0），与其余所有 kernel
完全不同——这在 v1 和 v4 中保持一致，是唯一在软件修复前后均稳定的聚类特征。

**Delta 补充：** `dram_throughput ↔ block_limit_registers = +0.957`
（DRAM 受限 kernel 的寄存器限制反而宽松=16）。residual_add 是该规律唯一的极端代表。
该相关性是 v4 新出现的，v1 中被 attention_score 的噪声掩盖。

**综合结论：** residual_add 的 HBM 带宽敏感性是 mini-transformer 中
唯一不受任何软件变化影响的稳定架构信号，是 HBM 带宽模型校准的理想基准。

**处方 C-2（HIGH）：** 以 residual_add 为基准校准模拟器 HBM 带宽模型（`gpgpu_n_mem`，时序参数）。

---

### 发现 C-3：softmax 揭示 L2 cache 容量限制（E0 + Delta — MEDIUM）

**E0：** DRAM=41.2%（异常高，远超其他计算 kernel），L1_hit=79.9%。

**Delta 补充：** softmax 的 `dynamic_shmem_per_block` 温度=0.957（最高 HOT 字段），
使 softmax 在 Delta 中是最独特的 kernel。但该字段本身是实现特性，
真正的架构信号在于其高 DRAM 使用率。

**解读：** attention scores 矩阵 = 12×512×512×4B = 12MB > L2 容量（RTX 3080 Ti L2 = 6MB），
softmax 的归约需要反复读写 HBM，导致 DRAM 使用率异常高。

**综合结论：** softmax 是 L2 cache 容量敏感性的标志 kernel。
模拟器若将 L2 配置过大，softmax 的 DRAM 利用率会被低估。

**处方 C-3（MEDIUM）：** 验证模拟器 L2 cache 容量配置（`gpgpu_cache:dl2`）。

---

### 发现 C-4：v1 的伪相关消失是最强的方法论验证（Delta 元信号 — 方法论贡献）

**v1 Delta：** l1_hit ↔ compute = -0.646（反相关，被解读为"高 L1 命中但计算低"是异常信号）。
**v4 Delta：** 该反相关消失。

这个"消失"本身是一个重要发现：
- **在软件未清洗的 workload 上**，Delta 产出的相关性可能是软件缺陷的投影，而非架构特性
- **在软件清洗后**，Delta 的相关性才反映真实的架构信号（如 block_limit_registers 中心地位）
- 这为"三层过滤框架"的必要性提供了实证支持：同一个 Delta 机制，在软件干净 vs 不干净的 workload 上，产出完全不同质量的洞察

---

## 完整处方表（E4 最终）

| ID | Kernel | 类别 | 处方内容 | 置信度 | 证据来源 |
|----|--------|------|---------|:------:|---------|
| C-1a | gemm_tiled | C | 寄存器文件大小 + FP32 initiation interval | HIGH | E0 + Squash + Batch + Delta |
| C-1b | attention_score | C | 同 C-1a + shared memory bank 配置 | HIGH | E0 + Squash + Batch + Delta |
| C-2 | residual_add | C | HBM 带宽模型（内存控制器 + 时序） | HIGH | E0 + Batch + Delta |
| C-3 | softmax | C | L2 cache 容量配置 | MEDIUM | E0 + Delta |
| C-4 | context_mul | C | L1 cache 容量 + 替换策略 | MEDIUM | E0 + Delta |
| — | layernorm | — | 无紧迫架构问题 | LOW | E0 |

---

## 跨机制一致性矩阵（v4）

| 发现 | E0 | Squash | Batch | Delta |
|------|----|--------|-------|-------|
| gemm + attention 寄存器限制 | ✅ | ✅（同段） | ✅（良性 outlier） | ✅（block_limit 中心） |
| residual_add HBM 瓶颈 | ✅ | ✅（独立段） | ✅（稳定聚类） | ✅（dram 极端代表） |
| softmax L2 溢出 | ✅ | — | ✅（outlier） | ✅（DRAM 异常） |
| v1 伪相关消失 | — | — | — | ✅（元信号） |
| 三态内存结构 | ✅ | — | ✅（三类聚类） | ✅（l1↔dram 反相关） |
