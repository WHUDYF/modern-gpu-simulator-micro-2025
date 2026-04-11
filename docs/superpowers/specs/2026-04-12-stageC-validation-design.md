# Stage C 闭环验证设计文档

**日期：** 2026-04-12
**目标：** 验证 mini-transformer v4 的三条 Stage C 架构处方（C-1/C-2/C-3）
**硬件：** RTX 3080 Ti (SM_86)
**模拟器：** GPGPU-Sim 4.2，SM86_RTX3080_TI 配置

---

## 背景

mini-transformer v4 完成软件层清洗后，Delta 机制发现
`block_limit_registers` 是中心约束字段，生成了三条 Stage C 处方：

| 处方 | 目标参数 | 目标 Kernel | 置信度 |
|------|---------|------------|:------:|
| C-1 | `gpgpu_shader_registers` | gemm_tiled、attention_score | HIGH |
| C-2 | `gpgpu_n_mem` + HBM 时序 | residual_add | HIGH |
| C-3 | `gpgpu_cache:dl2` | softmax | MEDIUM |

本文档设计如何在模拟器上验证这三条处方。

---

## 整体流程

```
Step 1：录制 trace（RTX 3080 Ti）
  编译 mini_transformer_v4
  NVBit tracer 录制完整运行
  提取 6 个代表 kernel 的 trace 文件

        ↓

Step 2：基准模拟（Baseline）
  SM86_RTX3080_TI 默认配置跑 6 个 kernel
  收集模拟器输出指标
  对比 NCU 实测数据，计算 baseline APE

        ↓

Step 3：处方验证（逐一修改参数）
  C-1：调整 gpgpu_shader_registers
  C-2：调整 gpgpu_n_mem
  C-3：调整 gpgpu_cache:dl2

        ↓

Step 4：汇总报告（E5_stageC_validation.md）
```

---

## 6 个代表 Kernel（来自 Batch 输出）

| 来源 | Kernel | 代表什么 |
|------|--------|---------|
| 聚类 A 代表 | gemm_tiled_1 | 计算密集类（7 次 launch）|
| 聚类 B 代表 | residual_add_9 | HBM 流式类（2 次 launch）|
| 聚类 C 代表 | layernorm_10 | 混合归约类（2 次 launch）|
| Outlier | attention_score | 高 shmem 计算密集 |
| Outlier | softmax_kernel | L2 溢出混合访存 |
| Outlier | context_mul | L1 驻留计算 |

---

## 测量指标

每个 kernel 对比以下 5 个指标：

| 指标 | 模拟器字段 | 对应处方 |
|------|----------|---------|
| `achieved_occupancy_pct` | `gpu_occ` | C-1 |
| `compute_throughput_pct` | `gpu_ipc` 换算 | C-1 |
| `dram_throughput_pct` | `dram_bw_util` | C-2 |
| `warp_cycles_per_issued_inst` | `gpu_ipc` 换算 | C-2/C-3 |
| `l1_hit_rate_pct` | `L1D_total_cache_hit_rate` | C-3 |

APE 计算：
```
APE = |模拟器值 - NCU实测值| / NCU实测值 × 100%
```

---

## 处方验证设计

### C-1：寄存器文件配置

**当前值：** `gpgpu_shader_registers 65536`

**验证逻辑：**
- 目标 kernel：gemm_tiled、attention_score
- 关键指标：occupancy、compute_throughput
- 对照 kernel：residual_add（APE 变化应 < 2%）

**参数调整方向：**
baseline 配置已是 65536（SM_86 真实值），
验证目的是确认模拟器能否正确预测 block_limit_registers=6 带来的 occupancy 限制。
若 baseline APE < 10% 则 C-1 直接通过；
若 APE > 10% 则需要同时检查 `trace_opcode_latency_initiation_sp`。

### C-2：HBM 带宽模型

**当前值：** `gpgpu_n_mem 24`

**验证逻辑：**
- 目标 kernel：residual_add
- 关键指标：dram_throughput、warp_cycles
- 对照 kernel：gemm_tiled（APE 变化应 < 2%）

**参数调整方向：**
若 baseline dram_throughput APE > 10%，
尝试调整 `gpgpu_n_mem`（±4）或 HBM 时序参数（tCL、tRCD）。

### C-3：L2 Cache 容量

**当前值：** `gpgpu_cache:dl2 S:64:128:16,L:B:m:L:P,A:192:96,32:0,32`

**验证逻辑：**
- 目标 kernel：softmax
- 关键指标：dram_throughput、l1_hit_rate
- 对照 kernel：context_mul（APE 变化应 < 2%）

**参数调整方向：**
softmax working set = 12MB > L2 6MB，
若模拟器 L2 配置过大会低估 DRAM 利用率。
尝试调整 L2 大小参数，观察 dram_throughput APE 变化。

---

## 成功标准

**处方有效判定（同时满足）：**
1. 目标 kernel 关键指标 APE 下降
2. 对照 kernel APE 变化 < 2%
3. APE 变化量 > 5%（超过测量噪声）

**APE 分级：**

| APE | 含义 |
|-----|------|
| < 10% | 模拟器建模准确 |
| 10% ~ 30% | 有偏差，处方提供改善方向 |
| > 30% | 建模存在根本性问题 |

**整体判定：**
- C-1/C-2（HIGH）：APE 必须下降，否则重新分析根因
- C-3（MEDIUM）：APE 下降即成功；无变化记录为"模拟器限制"

---

## 输出文件结构

```
results/mini_transformer_v4/
├── E5_stageC_validation.md
└── configs/
    ├── baseline/
    │   └── trace.config + gpgpusim.config
    ├── rx_C1/
    │   └── gpgpusim.config（调整 shader_registers）
    ├── rx_C2/
    │   └── gpgpusim.config（调整 n_mem）
    └── rx_C3/
        └── gpgpusim.config（调整 dl2）
```

---

## 实施前置条件

1. RTX 3080 Ti 可连接（已确认）
2. NVBit tracer 已配置（已确认）
3. mini_transformer_v4 binary 存在（已确认）
4. GPGPU-Sim 编译正常（参考 backprop 闭环验证先例）
