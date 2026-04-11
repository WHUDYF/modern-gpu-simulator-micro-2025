# 设计规格：Squash / Batch / Delta 语义机制 + 多 Dwarf 验证

日期：2026-04-08
基于：`2026-04-06-trace-semantic-ai-diagnosis-design.md`（已完成的 baseline 闭环）

## 1. 背景与定位

backprop 闭环验证（v2 报告 + closed_loop_validation_report）已经证明：
- 基于 trace 静态特征 + NCU 动态指标 + 模拟器闭环的处方诊断方法论可行
- AI 能给出可验证的、非平凡的架构修改处方（处方 B.2.1 在模拟器上让
  adjust_weights kernel 加速 44%，对照 forward kernel 零变化）

但 backprop 只是一个数据点，且当前用的是基础特征。需要回答两个问题：

- **Q1（广度）**：方法论在多个 dwarf 类别上的泛化性如何？
- **Q2（深度）**：Squash / Batch / Delta 这三种语义机制能否提升 AI 诊断质量？

本 spec 设计的就是回答这两个问题的工作。

## 2. 研究目标与定位

### 2.1 核心命题

> 通过结构化的语义提取机制（Squash / Batch / Delta），从 GPU trace 中产出
> AI 诊断需要的高层语义特征，让 AI agent 能在多个 dwarf 类别上稳定地产出
> 可验证的架构修改处方。

### 2.2 三层目标

```
最终目标（不在本次范围）：
   AI agent 能指导 GPU 部署 AI 模型的可行性

  ↑ 由以下问题支撑

直接目标（本次设计范围）：
   AI agent 能否基于 trace 语义特征对真实 workload 产出有效处方？
   这种能力能否跨 dwarf 类别泛化？

  ↑ 由以下三个子问题支撑

子问题 1（Track 1）：
   方法论在多个 dwarf 上的泛化性如何？

子问题 2（Track 2）：
   Squash / Batch / Delta 这三种语义提取机制各自能否
   提升 AI 诊断的质量？

子问题 3（联合）：
   这三种机制在不同 dwarf 上的有效性是否一致？
   失败的场景为什么失败？
```

### 2.3 不做的事

- 不做 GPT-2 或其他 PyTorch workload（NVBit 兼容性问题，留作扩展）
- 不改 GPGPU-Sim 的 trace 格式（不动 protobuf schema）
- 不做真正的 trace 压缩（这次只做语义提取，压缩率不是评估目标）
- 不追求"最优处方"（追求"有根据的处方"）
- 不全跑 Rodinia 10 个 benchmark（只做 3-4 个 AI 相关的）

### 2.4 评估的两个维度

| 维度 | 主目标 | 类型 | 数据来源 |
|------|--------|------|---------|
| 1. AI 诊断质量 | ★主要 | 人工评估 + 闭环加速比 | 处方报告 + 模拟器实测 |
| 2. 模拟时间节省 | ☆机会性 | 自动测量 | 模拟器运行时间日志 |

机会性评估的含义：不主动为模拟时间节省而设计机制，但 Squash 和 Batch
天然产出"复用提示"信息，跑实验时顺便记录模拟时间。如果数据显示出节省效果
就加入论文，否则放弃。

## 3. 三个机制的概念定义

### 3.1 Squash —— 时序分段

**输入**：Per-TB 行为特征向量序列，按 TB launch 顺序排列。

每个 TB 的特征向量来自三个来源：
- **静态指令分布**：top opcode 比例（FFMA、LDG、LDS、DFMA、IADD3 等的占比）
- **动态行为指标**：来自 NCU 的 per-kernel 指标（如果该 TB 所属 kernel 的
  NCU 数据可用），或来自 trace 的 per-TB 统计（指令数、warp 数）
- **压缩特征**：v8 cross-TB delta 中该 TB 的 delta 模式（地址 override
  数量、是否 full encoding）

形式上，每个 TB 表示为一个约 20 维的特征向量。

**机制**：滑动窗口的相似度聚类。

```
对相邻的 TB 计算特征向量的余弦相似度
如果相似度 > 阈值 τ_similar，认为属于同一段
否则在这里产生一个分段边界
```

起步用最简单的滑动窗口算法，确保语义清晰、可解释。复杂的 change point
detection 算法作为可选升级路径。

**输出**：

```json
{
  "squash_segments": [
    {
      "segment_id": 0,
      "tb_range": [0, 127],
      "tb_count": 128,
      "dominant_opcodes": ["FFMA", "LDS", "STS"],
      "cohesion_score": 0.93,
      "representative_tb": 12,
      "behavior_summary": "tile-based matmul, stable phase"
    },
    {
      "segment_id": 1,
      "tb_range": [128, 255],
      "tb_count": 128,
      "dominant_opcodes": ["DMUL", "DFMA", "F2F"],
      "cohesion_score": 0.97,
      "representative_tb": 192,
      "behavior_summary": "FP64 weight update phase"
    }
  ],
  "boundary_count": 1,
  "total_tbs": 256,
  "max_segment_size_pct": 50.0,
  "_simulation_reuse_hint": {
    "representative_tbs": [12, 192],
    "expected_speedup": "~2x if simulator can reuse representative results"
  }
}
```

**AI 用它回答的问题**：
> "这个 workload 有几个明显的行为阶段？每个阶段主导计算/访存模式是什么？
> 瓶颈在哪个阶段？"

对应的处方风格：
> "Phase 1（tile matmul）的瓶颈是 X，Phase 2（FP64 update）的瓶颈是 Y，
> 建议针对 Phase 2 的 DP 单元做修改"

**机会性副产品**：`_simulation_reuse_hint` 字段告诉模拟器"如果你只跑代表 TB
的模拟，其他 TB 可以推算"。这是模拟时间节省的钩子。

### 3.2 Batch —— 空间同质性识别

**输入**：全部 TB 的特征向量集合（不按顺序）。和 Squash 用同样的特征向量。

**机制**：全局聚类 + 离群点检测。

```
对所有 TB 跑 k-means 或 DBSCAN
- k-means 的 k 通过 silhouette score 自动选择
- DBSCAN 自动识别离群点
输出：
- 每个 TB 的 cluster id
- 每个 cluster 的大小、中心、半径
- 离群点列表
```

**和 Squash 的关键区别**：Squash 看时序顺序（TB launch 的时间维度），Batch
看全局集合（不看时序）。Squash 答"有几段"，Batch 答"有几类"。

**输出**：

```json
{
  "batch_clusters": [
    {
      "cluster_id": 0,
      "tb_count": 240,
      "tb_pct": 93.75,
      "centroid_summary": {
        "dominant_opcodes": ["FFMA", "LDS"],
        "avg_inst_per_warp": 450
      },
      "cohesion": 0.95
    },
    {
      "cluster_id": 1,
      "tb_count": 14,
      "tb_pct": 5.47,
      "centroid_summary": {
        "dominant_opcodes": ["FFMA", "LDS"],
        "avg_inst_per_warp": 520
      },
      "cohesion": 0.88,
      "deviation_from_main_cluster": "20% more instructions per warp"
    }
  ],
  "outlier_tbs": [255, 254, 253],
  "outlier_count": 3,
  "homogeneity_score": 0.94,
  "_simulation_reuse_hint": {
    "main_cluster_representative": 12,
    "secondary_cluster_representative": 200,
    "outliers_must_simulate": [255, 254, 253]
  }
}
```

**AI 用它回答的问题**：
> "workload 的并行单元行为有多一致？哪些 TB/warp 是离群的？这些离群点
> 的行为差异是什么？"

对应的处方风格：
> "96% 的 TB 是同质的，但 4% 的 TB（边界 TB）有 20% 的额外指令——这是
> 因为 sequence length 不被 tile size 整除。建议：要么 padding 到整除
> （kernel 层），要么增加 scheduler 给离群 TB 隐藏延迟（架构层）"

### 3.3 Delta —— 字段级变化模式

**输入**：相邻 TB（按 launch 顺序）之间的特征向量差。Delta 不看 TB 自己
的特征，只看相邻 TB 之间的变化。

**机制**：字段级变化频率统计 + 关联分析。

```
对所有相邻 TB 对 (TB_i, TB_{i+1}):
  diff_i = TB_{i+1}.features - TB_i.features

对每个字段 f:
  hot_score(f) = 平均 |diff(f)| / std(f.values)

字段相关性:
  对每对字段 (f1, f2):
    correlation_score = 共同变化的 TB 对数 / 总 TB 对数
```

**输出**：

```json
{
  "field_temperature": {
    "active_mask_dist": 0.02,
    "instruction_count": 0.05,
    "address_pattern_hash": 0.85,
    "stall_count_avg": 0.15,
    "register_pressure": 0.03
  },
  "hot_fields": ["address_pattern_hash"],
  "cold_fields": ["active_mask_dist", "instruction_count", "register_pressure"],
  "field_correlations": [
    {
      "fields": ["address_pattern_hash", "stall_count_avg"],
      "correlation": 0.72,
      "interpretation": "address pattern changes correlate with stall changes"
    }
  ],
  "outlier_diffs": [
    {
      "tb_pair": [127, 128],
      "magnitude": 5.2,
      "dominant_changing_fields": ["dominant_opcode"],
      "interpretation": "phase transition between TB 127 and 128"
    }
  ]
}
```

**AI 用它回答的问题**：
> "workload 的什么维度在变化、什么维度恒定？变化的字段之间有什么关联？"

对应的处方风格：
> "address_pattern 变化和 stall 变化高度相关（0.72），说明 stall 主要由
> 访存模式决定——这指向 memory subsystem 优化处方而不是 compute 处方"

### 3.4 操作层级（重要补充）

三个机制都必须在**两个层级**上独立操作：

- **Kernel-level**：把 workload 看作 kernel 序列。例如 backprop 的
  workload 是 [forward, adjust_weights] 两个 kernel
- **TB-level**：把单个 kernel 看作 TB 序列。例如 forward kernel 内部的
  256 个 threadblock

**为什么必须两层都做：**

考虑 backprop 的实际情况：
- forward 的 256 TB 高度规则（cross_tb_offset_coverage = 1.0）
- adjust_weights 的 256 TB 同样高度规则
- **如果 Squash 只在 TB 层操作**：每个 kernel 内部的 TB 都很相似，每个
  kernel 只产出 1 个 trivial 段，没有信息量
- **如果 Squash 在 kernel 层操作**：能识别 forward 和 adjust_weights 的
  行为差异（FFMA vs DFMA, FP32 vs FP64），自然产出 v2 报告手动得出的
  "两阶段瓶颈"结论

类似地：
- Batch 在 kernel-level 把整个 workload 的 kernel 集合做聚类，识别"哪些
  kernel 是同质的"
- Batch 在 TB-level 把单 kernel 内部的 TB 做聚类，识别"哪些 TB 是离群的"
- Delta 在 kernel-level 看相邻 kernel 之间字段的变化
- Delta 在 TB-level 看相邻 TB 之间字段的变化

**输出 JSON 结构**：每个机制的输出 JSON 包含 `kernel_level` 和 `tb_level`
两个 sub-section：

```json
{
  "kernel_level": {
    "squash_segments": [...]   // 整个 workload 的 kernel 序列分段
  },
  "tb_level": {
    "kernel_1_squash_segments": [...],  // forward kernel 的 TB 分段
    "kernel_2_squash_segments": [...]   // adjust_weights kernel 的 TB 分段
  }
}
```

**适用场景对比**：

| 层级 | 适用 workload | 例子 |
|------|--------------|------|
| Kernel-level | 多 kernel workload，每个 kernel 内行为均匀 | backprop, GPT-2 decode |
| TB-level | 长 kernel，单 kernel 内有内部相变 | FlashAttention prefill, 大型 GEMM |
| 两层都用 | 多 kernel + 部分 kernel 有内部相变 | 大多数真实 AI workload |

实现上**两层级共享同一个算法**（聚类、差分等），只是输入数据集不同，所以
工作量大致是 1.5x 而不是 2x。

### 3.5 三个机制的正交性

| 机制 | 看什么维度 | 答什么问题 |
|------|----------|----------|
| Squash | 时间维度 | 有几段（行为阶段） |
| Batch | 空间维度 | 有几类（并行单元类别） |
| Delta | 变化维度 | 什么在变、什么不变、变化间的关联 |

三个机制是正交的视角，不互相替代。在 backprop 这种行为高度规则的 workload
上 Squash 和 Batch 可能给出相似结果，但在 GPT-2 这种有 prefill/decode 阶段
切换 + 边界 batch 处理的 workload 上会截然不同。

### 3.6 共同接口约束

为了让消融实验、跨 dwarf 比较、最终分析都能机械化，三个机制必须遵守同一个
接口约束：

1. **输入端**：都接受同一个标准化的 `per_tb_features.json`（已有的特征
   提取脚本产出）
2. **输出端**：都产出一个独立的 JSON 子包（`squash_features.json`、
   `batch_features.json`、`delta_features.json`）
3. **配置端**：每个机制都有可调阈值（τ_similar、k 上限、hot 阈值等），
   都写在一个 `mechanism_config.json` 里
4. **AI 接入端**：诊断 prompt 里可以独立开关每个子包——这是消融实验的关键

接口约束是后续工作的基础，必须在 Phase 0 就固化。

## 4. 执行流程：五个 Phase

### 4.1 Phase 总览

```
Phase 0: 概念设计与接口约定（不写实现代码）
            ↓
Phase 1: backprop 上完成 Track 2 的三个机制原型
            ↓
Phase 2: backprop 上消融实验 + 错误分析（第一个 checkpoint）
            ↓ 决策点：是否调整机制
Phase 3: 第二个 dwarf（nn）上跑全套（第二个 checkpoint）
            ↓ 决策点：跨 dwarf 一致性
Phase 4: 第三个 dwarf（lud）上跑全套
            ↓ 决策点：是否扩展第四个 dwarf
Phase 5: 跨 dwarf 差异分析 + 最终方法论总结
```

每个 Phase 之间是强决策点，不是机械推进。

### 4.2 Phase 0：概念设计与接口约定

**目标**：把 Section 3 的内容固化成可执行的接口规范文档，不写实现代码。

**产出**：

1. `per_tb_features_schema.json`：定义标准化的 per-TB 特征向量长什么样
   （字段名、类型、来源）
2. `squash_output_schema.json` / `batch_output_schema.json` /
   `delta_output_schema.json`：三个机制的输出 JSON schema
3. `mechanism_config.json`：所有可调参数的清单（阈值、k 上限等）
4. `ablation_protocol.md`：消融实验的标准化流程（如何独立开关每个机制、
   如何对比、如何记录）
5. **`diagnose-workload.md` skill 定义**：放在 `~/.claude/skills/`，
   包含 prompt 模板、参数 schema、固定诊断协议
6. **`diagnosis_template.md` 报告模板**：所有诊断报告（手动和 skill 调用）
   必须遵循的 markdown 结构

**为什么必须先做**：
- 没有 schema，三个机制可能产出格式不兼容的 JSON
- 没有 ablation protocol，后续无法做"有 vs 没有"的公平对比
- 这些文档跨 Phase 都要用，提前定义避免返工

**估计工作量**：1-2 小时

### 4.3 Phase 1：backprop 三个机制原型实现

**目标**：在 backprop 上实现 Squash/Batch/Delta 的最小可行版本，每个能
独立产出符合 schema 的输出。

**产出**：

1. `extract_squash_features.py`：
   - 输入：`backprop_4096_full.json`（已有的 per-TB 特征）
   - 输出：`backprop_squash.json`
   - 实现：滑动窗口聚类（最简单版本）

2. `extract_batch_features.py`：
   - 输入：同上
   - 输出：`backprop_batch.json`
   - 实现：k-means + 离群检测

3. `extract_delta_features.py`：
   - 输入：同上
   - 输出：`backprop_delta.json`
   - 实现：相邻 TB 差分 + 字段温度统计

每个脚本支持 `--config mechanism_config.json` 来调阈值。

**关键约束**：
- 三个脚本完全独立——不互相依赖
- 每个脚本只用现有 `backprop_4096_full.json`，不需要重跑 trace 提取
- 不修改 simulator、tracer、protobuf

**估计工作量**：每个机制 1-2 小时，三个共计 4-6 小时

### 4.4 Phase 2：backprop 消融实验 + 错误分析

**目标**：跑出"四种诊断对比"的实验数据，判断每个机制是否真有效。

**实验矩阵**：

| 实验编号 | 输入特征 | Squash | Batch | Delta |
|---------|---------|--------|-------|-------|
| E0 (baseline) | 基础特征 | ❌ | ❌ | ❌ |
| E1 | 基础 + Squash | ✅ | ❌ | ❌ |
| E2 | 基础 + Batch | ❌ | ✅ | ❌ |
| E3 | 基础 + Delta | ❌ | ❌ | ✅ |
| E4 | 基础 + 全套 | ✅ | ✅ | ✅ |

**每个实验的产出**：
- 跑一次 AI 诊断，产出一份处方报告
- 记录每个新增机制带来的"新发现"
- 记录每个新增机制引入的"误诊"
- 把效果显著的处方拿到模拟器上做闭环验证

**AI 诊断的运行方式（混合 C —— skill 模式）**：

为了控制工作量并保持质量，AI 诊断采用混合方式。**不使用独立的 Claude
API**，而是把诊断任务包装成一个 superpowers 风格的 Claude Code skill，
让同一个 Claude 会话作为 "agent runtime" 反复执行诊断协议。

- **关键诊断（手动在对话中）**：每个 dwarf 的 E0（baseline）和 E4
  （全套机制）由 Claude 在对话中直接完成
  - 优点：质量最高，能立即发现问题
  - 用途：作为该 dwarf 的"金标"诊断，其他实验的对照

- **重复消融（通过 skill batch 模式自动化）**：不是每个实验单独调用 skill，
  而是**一个 plan.md 文件描述所有要跑的实验**，一次 skill 调用完成所有。
  - 实现形式：`/superpowers:diagnose-workload` skill 支持两种模式：
    - **Single mode**：`--features ... --enable-mechanism ... --output ...`
      用于单次诊断（调试用）
    - **Batch mode**：`--plan plan.md`
      从 plan.md 中读取多个实验定义，依次执行
  - 批量调用方式示例：
    ```
    /superpowers:diagnose-workload --plan experiments/.../backprop_ablation_plan.md
    ```
    或跨 dwarf 的 mega-plan：
    ```
    /superpowers:diagnose-workload --plan experiments/.../full_ablation_plan.md
    ```
  - 优点：
    - **1 次 dispatch 完成所有实验**（5 次或 15 次诊断都只需要 1 次命令）
    - **不需要 API key 和计费管理**
    - 和手动诊断使用同一个 Claude 会话 → **模型行为完全一致**，无需
      "手动 vs 自动一致性检查"
    - skill 执行过程中 Claude 可以调用其他工具（Read / Bash / Grep）
      来验证数据
    - **plan.md 本身是可复现的实验定义**（可 commit 到 git）
    - 失败时可以直接追问调试
  - 限制：
    - 需要 skill 实现上下文管理（见下方"遗忘协议"），否则 15 次诊断会
      爆 context window

### 4.4.1 Plan.md 文件格式

Batch 模式的核心输入是 plan.md。它必须包含：

```markdown
# <Experiment Name> Plan

## Feature Inputs
<每个 dwarf 的特征 JSON 路径>

## Mechanism Inputs (per dwarf)
<每个 dwarf 的 squash/batch/delta JSON 路径>

## Experiments
<实验编号 → 使用哪些机制的映射>

## Output
<输出目录路径>
```

**单 dwarf 示例**（只跑 backprop 的 E0-E4）：

```markdown
# Backprop Ablation Plan

## Feature Inputs
- workload: backprop
- base_features: experiments/baseline_diagnosis/results/rodinia/backprop_4096_full.json

## Mechanism Inputs
- squash: experiments/baseline_diagnosis/results/rodinia/backprop_squash.json
- batch: experiments/baseline_diagnosis/results/rodinia/backprop_batch.json
- delta: experiments/baseline_diagnosis/results/rodinia/backprop_delta.json

## Experiments
- E0_baseline: []
- E1_squash: [squash]
- E2_batch: [batch]
- E3_delta: [delta]
- E4_full: [squash, batch, delta]

## Output
- directory: experiments/baseline_diagnosis/results/rodinia/backprop_ablation/
- naming: {experiment_id}.md
- summary: _summary.md
```

**跨 dwarf mega-plan 示例**（一次跑 backprop + nn + lud 的全部 15 次诊断）：

```markdown
# Multi-Dwarf Full Ablation Plan

## Dwarfs
- backprop:
    base_features: .../backprop_4096_full.json
    mechanism_squash: .../backprop_squash.json
    mechanism_batch: .../backprop_batch.json
    mechanism_delta: .../backprop_delta.json
- nn:
    base_features: .../nn_full.json
    mechanism_squash: .../nn_squash.json
    mechanism_batch: .../nn_batch.json
    mechanism_delta: .../nn_delta.json
- lud:
    base_features: .../lud_full.json
    mechanism_squash: .../lud_squash.json
    mechanism_batch: .../lud_batch.json
    mechanism_delta: .../lud_delta.json

## Experiments (applied to each dwarf)
- E0_baseline: []
- E1_squash: [squash]
- E2_batch: [batch]
- E3_delta: [delta]
- E4_full: [squash, batch, delta]

## Output
- directory: experiments/baseline_diagnosis/results/ablation/
- naming: {dwarf}/{experiment_id}.md
- summary: _cross_dwarf_summary.md
```

### 4.4.2 Skill 内部协议（含上下文管理）

`/superpowers:diagnose-workload` 在 batch 模式下遵循以下协议：

1. **读 plan.md**，解析成实验清单（dwarf × experiment 的笛卡尔积）

2. **对每个实验（顺序执行）**：

   a. 读取当前实验需要的 base_features 和 mechanism JSON
      （**只读当前需要的**，不要一次性加载所有 dwarf 的所有特征）

   b. 执行诊断流程：
      - Stage A 检查：`waves_per_sm`, `achieved_occupancy`
      - Stage B 分析：distance to roof, top bottleneck, 交叉推理
      - 处方生成：每条处方包含 { 修改, 依据, 预期, 验证, 置信度 }

   c. 按固定 markdown 模板写入输出文件路径

   d. **在对话中只回报一行**："E1_squash done, 2 prescriptions written to <path>"

   e. **主动丢弃当前实验的详细内容**（对话里不保留完整诊断文字）——
      只保留 "已完成实验列表" 这个 metadata 进入下一次迭代

3. **所有实验完成后**，生成 summary：
   - **从磁盘重新读取**每个实验的输出文件（不依赖对话内存里的内容）
   - 提取关键信息（主要处方、置信度、发现的瓶颈）
   - 写入 `_summary.md`（如果 plan 里指定了的话）

4. **执行单模式时**（`--features --enable-mechanism --output`）：
   直接执行上述第 2 步一次，跳过第 1 和第 3 步

### 4.4.3 上下文管理的具体约束

由于 batch 模式可能要跑 15 次诊断，必须显式约束上下文消耗：

- **每次诊断的峰值 context** ≈ {plan metadata (~2K)} + {当前 features + mechanism
  JSON (~5K)} + {当前诊断报告 (~4K)} ≈ **11K tokens**
- **历史实验只保留 metadata**（1 行/实验 ≈ 50 tokens × 15 = 750 tokens）
- **总峰值** ≈ 11K + 750 ≈ 12K tokens（远低于 context window 上限）

**实现要求**：
1. Skill 定义放在 `~/.claude/skills/diagnose-workload.md`
2. 诊断报告统一使用 `experiments/baseline_diagnosis/diagnosis_template.md`
   作为模板
3. Skill 诊断输出**仍需人工审阅**才能进入闭环验证（防止 hallucination
   导致的错误处方进入模拟器）
4. Plan.md 是**可 commit 到 git 的可复现实验定义**，一个 plan.md 对应一次
   完整的消融实验
5. 由于手动诊断和 skill 诊断共用同一个 Claude 会话，**无需做一致性
   校准**——它们本来就是一样的

**错误分析**（D-级深度的核心）：
- Squash 没能识别的相变在哪里？为什么？
- Batch 误判的离群点在哪里？为什么？
- Delta 没能捕捉的字段关联在哪里？为什么？

**Checkpoint 1 决策**：

- **如果 E1-E4 中至少 1 个实验显示出非平凡的诊断改进**（找到 E0 没找到
  的真实瓶颈）→ 进入 Phase 3
- **如果 E1-E4 都和 E0 没有质的差异**（机制对 backprop 没用）→ 暂停
  - 是机制设计错了？回 Phase 0 修订接口
  - 还是 backprop 太规则没有发挥空间？跳到 Phase 3，看 nn 上是否不同
  - 还是机制本身不适合这类 workload？记录为 negative result，重新设计

不是机械推进，是真正的决策点。

**估计工作量**：1-2 天

### 4.5 Phase 3：nn 上跑全套（第一次跨 dwarf 合流）

**目标**：把 Phase 1 实现好的 Track 2 全套机制应用到 nn-rodinia-2.0-ft 上。

**步骤**：

1. **数据采集**（基础流程，已有脚本）：
   - 使用 rodinia2Ampere 已解压的 nn trace
   - 编译 nn 二进制（如果需要 NCU 数据）
   - 跑 NCU
   - 用现有 `extract_trace_features.py` 提取基础特征

2. **机制应用**：
   - 用 Phase 1 的三个脚本直接处理 nn 的特征 JSON
   - 产出 nn 的 squash/batch/delta JSON

3. **诊断 + 闭环**：
   - 重复 Phase 2 的实验矩阵 E0-E4
   - 在模拟器上验证至少 1 条 high-confidence 处方

4. **跨 dwarf 对比**：
   - 关键产出：`backprop_vs_nn_comparison.md`
   - 内容：三个机制在两个 dwarf 上的有效性对比表

**Checkpoint 2 决策**：

- **如果机制在 nn 上效果和 backprop 一致**（都有效或都无效）→ 直接进入
  Phase 4
- **如果机制在 nn 上表现不同**（比如 Squash 在 backprop 上没用但在 nn 上
  有用，或反之）→ 停下分析
  - 这正是迭代合流的价值——发现 workload 特异性
  - 可能要修订机制设计或参数
  - 输出："机制 X 在 dwarf 类别 Y 上有效，类别 Z 上无效"的精细化结论
- **如果机制在 nn 上严重失败**（产出错误处方）→ 回 Phase 0 重新设计

**估计工作量**：2-3 天

### 4.6 Phase 4：lud 上跑全套（第二次合流）

**目标**：跑第三个数据点，让"跨 dwarf 一致性"或"跨 dwarf 差异"的结论
有 3 个数据点支撑。

和 Phase 3 几乎相同，只是换 dwarf 为 lud。

**Checkpoint 3 决策**：

- 三个 dwarf 都跑完后，判断：
  - 三个机制各自在 3 个 dwarf 上的成功率
  - 跨 dwarf 的 pattern：哪些机制是"普适的"，哪些是"workload-specific 的"
  - 是否需要扩展第 4 个 dwarf（比如 hotspot 这种和 AI 不相关的，作为
    "对照组"）

**估计工作量**：2-3 天

### 4.7 Phase 5：跨 dwarf 差异分析与最终总结

**目标**：把所有实验数据整合成一份完整的方法论报告，可以作为论文的核心
实验章节草稿。

**产出**：

1. **跨 dwarf 矩阵表**：

| Dwarf | Squash 有效？| Batch 有效？| Delta 有效？| 处方是否经过闭环验证？|
|-------|-------------|------------|------------|---------------------|
| backprop | ? | ? | ? | ? |
| nn | ? | ? | ? | ? |
| lud | ? | ? | ? | ? |

2. **每个机制的"适用边界"分析**：
   - Squash 适合什么类型的 workload？
   - Batch 适合什么类型的 workload？
   - Delta 适合什么类型的 workload？

3. **失败案例汇总**：所有出现失败的场景及原因分析

4. **机会性长尾数据**：如果有值得报告的模拟时间节省，整理成附录

5. **方法论总结**：
   - 哪些 Track 2 机制是真创新点？
   - 哪些只是工程实现？
   - 论文的可发表 contribution 清单

**估计工作量**：1-2 天

### 4.8 总时间估算与里程碑

| Phase | 工作量 | 累计天数 |
|-------|-------|---------|
| Phase 0 | 1-2 小时 | 0.5 天 |
| Phase 1 | 4-6 小时 | 1.5 天 |
| Phase 2 + Checkpoint 1 | 1-2 天 | 3 天 |
| Phase 3 + Checkpoint 2 | 2-3 天 | 6 天 |
| Phase 4 + Checkpoint 3 | 2-3 天 | 9 天 |
| Phase 5 | 1-2 天 | 11 天 |

**总计：约 11 天的有效工作时间**（按持续投入计算，实际日历时间可能 2-3 周）。

**关键里程碑**：

- **Day 3**：Checkpoint 1 决定要不要继续——这是最重要的早期判断
- **Day 6**：Checkpoint 2 决定 Track 2 的工作是否泛化
- **Day 9**：Checkpoint 3 决定是否需要扩展第 4 个 dwarf
- **Day 11**：方法论报告完成

## 5. 风险与应对

| 风险 | 可能性 | 应对 |
|------|-------|------|
| Phase 2 后机制全部无效 | 中 | 不强行继续，回 Phase 0 重新设计或承认负面结果 |
| 跨 dwarf 表现差异极大 | 中-高 | 这本来就是研究产出之一，写进论文 |
| nn/lud 的 trace 数据有问题 | 低 | rodinia2Ampere 是论文作者预生成的，应该没问题 |
| AI 诊断质量不稳定 | 中 | 用统一 prompt 模板 + 同一对话窗口降低方差 |
| 时间超出预期 | 高 | 如果 Phase 4 后时间不够，省略 Phase 5 的"机制适用边界分析"，但保留矩阵表 |

## 6. 与现有工作的关系

### 6.1 继承关系

- **基于**：`2026-04-06-trace-semantic-ai-diagnosis-design.md` 的方法论
  原则（包括 §2.5 的 Class A/B 分离、交叉推理、闭环验证、空对照原则）
- **基于**：`backprop_prescription_v2.md` 和 `closed_loop_validation_report.md`
  作为 baseline，本次工作的所有改进都相对于此基线衡量

### 6.2 扩展点

本 spec 是对原 spec **第三步（Squash/Delta 语义增强）的具体化**，原 spec
里这一步是"条件性"任务，本 spec 把它升级为主要任务。

### 6.3 不替代

- 不替代原有的特征提取脚本（`extract_trace_features.py`、
  `parse_ncu_metrics_v2.py`），新机制是它们的下游消费者
- 不替代闭环验证流程，新机制提出的处方仍然按现有流程在模拟器上验证

## 7. 与同门工作的协同空间

同门正在做的工作：AI 自动生成 microbench 用于硬件表征。这个工作的产出
（hardware fact sheet）可以作为本 spec 的输入：

- 让 AI 诊断时引用同门测量的硬件常数（如 FP64 throughput、L1 bank
  conflict penalty 等），而不是依赖 NCU 间接推测
- 在 Phase 5 的方法论总结里讨论"如果有同门的 hardware fact sheet 作为
  上游输入，本方法论的诊断准确度能进一步提升"

但本 spec 不依赖同门工作的完成。两个工作互相独立但接口兼容。

## 8. 待决问题

无。所有方法论决策已通过 7 + 4 个澄清问题确定：

1. ~~两个 track 的关系~~ → 完全并行（B）
2. ~~Squash/Batch/Delta 优先级~~ → 三者同时（D）
3. ~~评估标准~~ → A + B 双评估（D）
4. ~~多 dwarf 广度~~ → AI 导向 3-4 个 + 条件性扩展（D）
5. ~~验证深度~~ → 消融 + 错误分析 + 跨 dwarf 差异（D）
6. ~~合流方式~~ → 迭代合流（C）
7. ~~起点~~ → 概念设计先行（D → A）+ 方案 1（语义优先）
8. ~~机会性目标~~ → 模拟时间节省作为副产品（C）
9. ~~Squash/Batch/Delta 操作层级~~ → 两层级都做（kernel-level + TB-level）
10. ~~AI 诊断方式~~ → 混合 C（关键手动 + 重复用 superpowers skill）
11. ~~时间预算~~ → 11 天完整方案，过程中按 Checkpoint 决策是否继续
12. ~~Skill vs API~~ → 用 Claude Code skill 替代 API，无额外基础设施依赖
