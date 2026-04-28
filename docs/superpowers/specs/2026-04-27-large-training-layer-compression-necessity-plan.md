# 大训练单层压缩必要性证明——实施计划

## 目标描述

构造一个可控、可复现的大训练 workload 实验，使用单个 Llama-style decoder block 训练 step，证明在大训练 workload 进入后端 planning 之前，representative compression 是必要的前置条件。

实验通过以下链路生成规模证据：

```text
单个 decoder layer 训练 step
  → nsys kernel timeline
  → invocation table
  → compression summary（多种分组策略）
  → projected simulation cost
  → necessity claim（accepted 或 rejected）
```

本计划将要证明的主张已从原始草稿中较宽的声明中收窄：本实验证明的是 **representative compression 在后端 planning 之前是必要的**——而非更强的"端到端 exact-cycle simulation 之前必要"主张。后者需要多层/全模型证据，推迟到后续阶段。

核心论点：

> 即使只是一个现代 decoder-only Transformer 单层，也会产生足够多的 kernel invocation、运行时间和 trace 体积，使得将所有 invocation 直接送入后端 planning 不可行。Representative compression 将候选集缩小到可管理规模，使后端 planning 可行。

## 验收标准

遵循 TDD 理念，每个标准包含正向测试和负向测试以实现确定性验证。

### 执行契约

以下所有标准均假设已建立以下固定执行契约，必须在接受任何证据之前验证：

- PyTorch >= 2.6 with CUDA 12.8
- 注意力实现：eager SDPA（不使用 FlashAttention）
- 融合策略：不使用 `torch.compile`，不使用 Inductor/Triton 生成的内核
- 确定性：`torch.backends.cudnn.deterministic=True`，`CUBLAS_WORKSPACE_CONFIG=:4096:8`
- cuBLASLt 出现：记录为观察但不视为契约违规（待定用户决策 DEC-1）
- 设备：单张 NVIDIA RTX 5090（32 GB）

### 标准

- AC-1: 训练 harness 环境门控通过，且 harness 产生稳定、可复现的 kernel timeline
  - 正向测试（预期通过）：
    - 检测到 CUDA-enabled PyTorch，版本正确且 GPU 可见
    - `nsys` CLI 已找到、可执行，版本已记录
    - GPU 驱动版本、VRAM 总量、可用磁盘空间（>= 10 GB 用于 trace 产物）已记录在门控报告中
    - 三次连续 profiled run 产生的总 kernel 数量彼此差异在 5% 以内
    - 三次 run 之间 kernel 名称分布的 Jensen-Shannon divergence < 0.05
    - 三次 run 之间 Top-5 kernel coverage 份额在 +/- 3 个百分点以内
    - 所有 run 中每个 stream 内的 kernel 顺序保持一致
  - 负向测试（预期失败）：
    - 当无 CUDA GPU 可用时，harness 以清晰的错误消息退出
    - Harness 拒绝不支持的 dtype（如 fp64）
    - 当 `nsys` 未找到时，harness 以清晰消息退出
    - 任何前置检查失败时不生成门控报告
    - 跨 stream 的全局交错顺序不需要完全相同（并发 stream 在不同 run 之间可能以不同方式交错；这不视为失败）
  - AC-1.1: 从生成的产物中验证执行契约合规性
    - 正向：Harness 后检查确认 timeline 中零个 FlashAttention kernel 名称
    - 正向：Harness 后检查确认零个 `torch.compile` / Inductor / Triton 生成的 kernel 名称模式
    - 正向：cuBLASLt kernel 出现已记录，含数量和名称
    - 负向：如果检测到 FlashAttention 或 compile/Inductor/Triton 模式，契约合规性检查失败

- AC-2: Kernel invocation table 将 nsys 输出映射为具有明确 stream 排序语义的结构化记录
  - 正向测试：
    - 每条记录包含：invocation_id、kernel_name、start_timestamp、duration、grid_dim、block_dim、stream_id、source_profiler_path
    - 全局时间戳顺序为主排序键
    - Per-stream 顺序作为辅助一致性检查进行验证
    - Stream 来源保留（每条 invocation 记录 stream_id）
    - 重复 invocation（相同 kernel_name + 相同 start_timestamp）被检测并拒绝
  - 负向测试：
    - Parser 对无法识别或损坏的 nsys 导出格式报错
    - Parser 拒绝缺少必需列的输入
    - Parser 拒绝单个 stream 内时间戳单调性违规的输入

- AC-3: Compression summary 使用至少三种分组策略量化规模压力
  - 正向测试：
    - 报告：total invocations、unique kernel names、unique (name + launch-shape) groups、top-k runtime coverage、在 >= 3 个 coverage 阈值下的 representative counts（如 80%、90%、95%）
    - 每种分组策略计算 compression ratio：name-only、name+shape、approximated-hybrid（使用可用字段，不含 NCU 特征）
    - 组级别统计包括成员数、总运行时间份额和异质性指示器
  - 负向测试：
    - Summary 标记退化情况：所有 kernel 在同一组（过于粗糙）或每个 kernel 都是独立组（过于精细）
    - 无效分组配置（如负 coverage 阈值）报错

- AC-4: Scale proof report 使用显式投影模型接受或拒绝必要性主张
  - 正向测试：
    - 报告包含投影公式：`single_layer_cost × N_layers × safety_factor` 及文档化参数
    - 默认投影：N_layers = 32（Llama-8B 参考），safety_factor = 1.5x（多层交互开销）
    - 敏感性分析包含除默认值外至少两个 safety factors（如 1.2x 和 2.0x）
    - 预先声明的不可行性阈值：projected uncompressed backend candidates > 500 或 projected trace size > 100 GB
    - 报告根据证据与阈值的比较，明确声明主张为 ACCEPTED 或 REJECTED
    - Primary 证据和 fallback 证据在单独章节中呈现，附清晰 deltas
  - 负向测试：
    - 报告不得将 fallback 证据呈现为与 primary 证据等价
    - 如果 primary tier 失败或不可用，报告不得接受主张
    - 如果所有 tier 均失败，实验记录为 BLOCKED（含根本原因），而非负面证据

- AC-5: 通过 degraded-mode adapter 使输出产物与现有 A-line frontend anchor schema 兼容
  - 正向测试：
    - Kernel invocation table JSON 通过 A-line KernelInvocationRecord schema 验证（所有必需字段存在）
    - 缺失的 NCU 派生字段（`dynamic_inst_count`、`feature_vector`、squash 字段）用显式 `"absent"` 标记表示，而非省略
    - Schema 来源标签区分 nsys 派生的记录和未来的 ncu 丰富记录
    - Adapter 验证测试确认往返：nsys CSV → invocation table → schema 检查通过
  - 负向测试：
    - 当必需的身份字段（kernel_name、grid_dim、block_dim）缺失时，schema 验证失败（含清晰消息）
    - Adapter 拒绝使用与 A-line 约定不同的硬编码字段名的输入

- AC-6: Evidence tier 通过来源、deltas 和具有约束力的决策规则明确区分
  - 正向测试：
    - 每个 evidence tier 标记其与 primary config 的差异
    - Primary 与每个 fallback tier 之间的 deltas（kernel 数量、运行时间、组分布）被计算并并排呈现
    - 决策规则明确声明：primary tier 证据是接受主张所必需的；fallback tier 仅用于表征敏感性
  - 负向测试：
    - 静默 fallback（primary 失败，fallback 结果无来源标签呈现）视为错误
    - 仅 fallback 证据不能触发主张接受
    - Tier 失败无根本原因文档视为不完整

## 路径边界

路径边界定义了实施质量和选择的可接受范围。

### 上界（最大可接受范围）

四个脚本（training harness、nsys runner、kernel parser、compression summarizer），具有完整错误处理和 CLI 接口。`seq_len=2048` 和 `seq_len=1024` 均作为计划实验点（非一个作为另一个的 fallback）。Primary 与所有 fallback tier 的并排比较及计算出的 deltas。使用三个 safety factors 进行敏感性分析的投影模型。生成 A-line 兼容的 KernelInvocationRecord 且具有显式 feature-absence 标记、schema 验证测试和来源标签的 Degraded-mode adapter。独立于 profiling 脚本的契约合规性验证脚本。环境门控报告以机器可读 JSON 格式输出。

### 下界（最低可接受范围）

单个 training harness 脚本，运行一次 forward+backward pass，包含 nsys profiling 和 NVTX ROI 标记。将 nsys CSV 导出转换为遵循 A-line KernelInvocationRecord 字段名的结构化 invocation 记录的 parser（所有必需的身份字段均存在）。最少计算以下内容的 compression summarizer：total invocations、unique kernel names、unique (name + launch-shape) groups、name-only 和 name+shape 策略的 compression ratios。包含定义的投影模型和明确声明接受/拒绝的 scale proof report。用 `"absent"` 标记表示缺失 NCU 字段的 degraded-mode adapter。契约合规性验证集成到 profiling 脚本中（非独立脚本）。裸 CSV 不可接受作为唯一的 invocation table 输出；结构化 JSON 记录是最低要求。

### 允许的选择

- 可以使用：PyTorch >= 2.6 with CUDA 12.8、Nsight Systems CLI、Python 3.10+、JSON 和 CSV 用于中间数据、name-only 和 name+shape 分组策略、eager SDPA attention、cuBLAS matmul
- 不能使用：Nsight Compute（counter 权限被阻止）、FlashAttention、`torch.compile` / Inductor / Triton、任何运行时需要互联网访问的框架、任何会加载 pretrained weights 的框架
- 草稿已固定：实验输出目录为 `experiments/large_training_layer/`；主要形状为 `seq_len=2048, hidden=4096`；输出脚本和结果子目录如草稿第 7 节所述
- 执行契约已固定：PyTorch + CUDA 版本、确定性标志、融合策略如上所述

## 可行性提示与建议

> **注意**：本节仅供参考和理解。这些是概念性建议，而非规定性要求。

### 概念方案

```text
+-------------------+     +------------------+     +-------------------+
| Environment Gate  |     | Training Harness |     | Nsight Systems    |
| - PyTorch check   |---->| - 构造 decoder   |---->| - NVTX ROI 标记   |
| - nsys check      |     |   layer          |     | - nsys profile    |
| - GPU/disk check  |     | - 合成数据       |     | - 导出 CSV        |
+-------------------+     | - Forward+back   |     +-------------------+
                          +------------------+              |
                                                            v
+-------------------+     +------------------+     +-------------------+
| Scale Proof       |     | Compression      |     | Kernel Parser     |
| Report            |<----| Summarizer       |<----| - 解析 nsys CSV   |
| - 投影模型         |     | - 3 种策略       |     | - 构造 records    |
| - Accept/reject   |     | - 比率/coverage  |     | - Stream 排序     |
| - Tier deltas     |     | - 组统计         |     | - 验证 schema     |
+-------------------+     +------------------+     +-------------------+
         |                         |                        |
         v                         v                        v
+-------------------+     +------------------+     +-------------------+
| Evidence Tiers    |     | Schema Adapter   |     | A-line Frontend   |
| - Primary         |     | - 缺失字段标记    |     | Anchor 检查       |
| - Fallback A/B/C  |     | - 来源标签       |     | - Schema 兼容性   |
+-------------------+     +------------------+     +-------------------+
```

Harness 使用 eager SDPA attention（不用 FlashAttention）构造单个 Llama-style decoder block。使用随机合成 tokens 和 activations——计算图是真实的，但数据是随机的。这将 GPU kernel 种群与模型质量问题隔离。

NVTX 范围标记 ROI，排除初始化、allocator churn 和不相关的 CUDA runtime 活动。nsys 分析 warmup + 一次测量的 forward+backward pass。

Parser 将 nsys CSV 导出转换为 A-line 兼容的 KernelInvocationRecord 格式。Stream 来源保留。缺失的 NCU 派生字段用显式 `"absent"` 值标记。

Compression summarizer 应用 name-only、name+shape 和 approximated-hybrid 分组策略。Approximated-hybrid 策略使用可用字段（grid_dim、block_dim、duration、kernel_name）来近似完整 hybrid 策略在有 NCU 特征时的效果。

Scale proof report 应用投影模型并声明主张是接受还是拒绝。

### 相关参考

- `experiments/baseline_diagnosis/frontend_anchor/invocation_table.py` — KernelInvocationRecord schema 定义和 `build_records_from_dual_sources()`。大层 adapter 必须生成与此 schema 兼容的记录。
- `experiments/baseline_diagnosis/frontend_anchor/selector.py` — 三种分组模式（name-only、pka-like-coarse、hybrid）和锚点选择策略。大层 compression summarizer 应反映这些分组策略。
- `experiments/baseline_diagnosis/frontend_anchor/exporter.py` — RepresentativeAnchorRecord schema 和验证约束。定义了 A-line 集成的目标格式。
- `experiments/backend_pipeline/execution_bridge.py` — `build_run_specs()` 和 `execute_run_specs()` 展示 run spec 如何流转到仿真。有助于理解"后端 planning"消费什么。
- `experiments/backend_pipeline/workload_profiles.py` — Workload profile 结构，包括路径、环境变量、CLI 参数和场景覆盖。展示后端 run 需要什么配置。
- `experiments/baseline_diagnosis/schemas/` — 用于 batch、delta、squash 和 per-tb 特征输出的 JSON schema 文件。项目内 schema 设计模式的参考。
- `experiments/mini_transformer/` — 现有 mini_transformer workload，包括 CUDA 源码、trace 和 NCU 指标。小 workload 如何构建的参考。
- `experiments/gpt2_decode/run_decode.py` — GPT-2 decode workload 的现有 Python harness。项目内 harness 模式的参考。

## 依赖关系与顺序

### 里程碑

1. Milestone 1: 环境门控与 Harness
   - Phase A: 验证 CUDA PyTorch、nsys、GPU 和磁盘；生成门控报告
   - Phase B: 实现带执行契约的单 decoder layer 训练 harness
   - Phase C: 实现带 NVTX ROI 标记和契约合规性检查的 nsys profiling 脚本

2. Milestone 2: Schema 审查与数据提取
   - Step 1: 审查 A-line schema 兼容性；识别所有必需字段、约束和 degraded-mode 需求
   - Step 2: 实现生成结构化 invocation 记录的 nsys CSV parser
   - Step 3: 实现带验证测试和来源标签的 degraded-mode schema adapter

3. Milestone 3: 分析与报告
   - Step 1: 实现至少三种分组策略的 compression summarizer
   - Step 2: 生成带投影模型、敏感性分析和主张决定的 scale proof report
   - Step 3: 执行 evidence tier 矩阵（seq_len 变体、fp16、gradient checkpointing），附来源标签

Milestone 1 必须在 Milestone 2 之前完成。Milestone 2 Step 1（schema 审查）必须在 Milestone 2 Step 2-3 之前完成。Milestone 2 必须在 Milestone 3 之前完成。

## 任务分解

每个任务必须恰好包含一个路由标签：
- `coding`: 由 Claude 实现
- `analyze`: 通过 Codex 执行（`/humanize:ask-codex`）

| Task ID | 描述 | 目标 AC | 标签 (`coding`/`analyze`) | 依赖 |
|---------|------|---------|----------------------------|------|
| task1 | 验证 CUDA PyTorch 环境、nsys、GPU、磁盘；生成环境门控报告或记录阻塞原因 | AC-1 | coding | - |
| task2 | 实现带执行契约的单个 Llama-style decoder block 训练 harness（eager SDPA、无 compile、确定性标志） | AC-1, AC-1.1 | coding | task1 |
| task3 | 实现带 NVTX ROI 标记的 nsys profiling 脚本；集成契约合规性检查（无 FlashAttention、无 compile/Inductor/Triton 模式） | AC-1, AC-1.1, AC-2 | coding | task2 |
| task4 | 审查 A-line frontend anchor schema 兼容性；识别必需 vs 可选字段、约束以及 nsys-only 记录的 degraded-mode 映射 | AC-5 | analyze | - |
| task5 | 实现 nsys CSV parser，生成具有明确 stream 排序语义和重复检测的结构化 invocation 记录 | AC-2 | coding | task3, task4 |
| task6 | 实现 degraded-mode schema adapter，具有显式 NCU 特征缺失标记、来源标签和验证测试 | AC-5 | coding | task4, task5 |
| task7 | 实现 compression summarizer：name-only、name+shape、approximated-hybrid 分组；compression ratios；组统计；边缘情况检测 | AC-3 | coding | task5 |
| task8 | 生成 scale proof report：投影模型（32 层、1.5x safety、>= 2 个敏感性因子）、不可行性阈值、主张决定、primary/fallback deltas | AC-4 | coding | task7 |
| task9 | 执行 evidence tier 矩阵：seq_len=1024、fp16、gradient checkpointing；用来源标记每个 tier；计算与 primary 的 deltas | AC-6 | coding | task3, task8 |

## Claude-Codex 审议

### 共识

- **主张范围必须收窄**：Claude 和 Codex 一致认为原始草稿的"compression 在 exact-cycle simulation 之前必要"主张对于单层证据来说过于宽泛。已收窄为"在后端 planning 之前必要"——这可以用单层数据来辩护。
- **执行契约至关重要**：双方一致认为，不固定 PyTorch/CUDA 版本、注意力实现和融合策略，kernel 种群就不可复现，实验就不可解释。
- **环境是门控，不是任务**：双方一致认为 CUDA PyTorch 可用性是一个前置检查，而非例行的实施步骤。如果不可用，实验将被阻塞。
- **Degraded-mode adapter 必须具体**：双方一致认为 nsys-only 路径需要一个具体的 schema adapter，带有显式的缺失标记和验证测试——而非文档注释。
- **Fallback 不能替代 primary**：双方一致认为 evidence tier 必须标记，如果 primary tier 失败，fallback 证据不能接受主张。
- **A-line schema 审查在 parser 工作之前**：双方一致认为集成约束应塑造 parser 和 adapter，而非推迟到后期审查任务。
- **Stream 排序已定义**：双方一致认为 per-stream 排序必须保留，而跨 stream 的全局交错在不同 run 之间可以合法变化。
- **定量阈值已声明**：双方一致认为不可行性阈值（> 500 backend candidates，> 100 GB projected trace）必须在证据收集前声明。

### 已解决的分歧

- **可复现性范围**：Claude 最初仅提出 5% kernel 数量方差。Codex 认为这过于狭窄。已解决：AC-1 现包含 kernel 名称 J-S divergence < 0.05、top-5 coverage 稳定性 +/- 3pp、per-stream 排序保留——不仅仅是总数。
- **下界底线**：Claude 最初允许裸 CSV 作为最低 invocation table。Codex 认为这与 AC-2/AC-5 冲突。已解决：下界现要求结构化 JSON 记录（带 A-line 字段名）加 degraded-mode adapter。裸 CSV 不可接受。
- **Name+shape 分组的充分性**：Codex 认为 name+shape 对训练 kernel 过于薄弱——相同形状可能隐藏不同的内存行为。Claude 指出没有 NCU 特征就无法更精细分组。已解决：计划增加使用可用字段的 approximated-hybrid 策略，记录其局限性，并将完整 hybrid 模式推迟到 NCU 可用的未来阶段。
- **任务排序**：Claude 将 A-line 可行性审查（前 task9）放在最后。Codex 认为必须在 parser/adapter 工作之前。已解决：task4（schema 审查）现在位于 task5-6（parser 和 adapter）之前，且 task4 独立于 task1-3（不依赖 harness 完成）。
- **投影模型时机**：Claude 将投影模型推迟到 report 任务。Codex 要求它在 task8 之前定义。已解决：投影模型参数（32 层、1.5x safety、>= 2 个敏感性因子、阈值）在 AC-4 中定义，必须在 task8 中实现。

### 收敛状态

- 最终状态：`converged`
- 执行轮次：3
- 第 1 轮：识别出 10 个 REQUIRED_CHANGES
- 第 2 轮：识别出 4 个 REQUIRED_CHANGES
- 第 3 轮：0 个 REQUIRED_CHANGES，0 个 DISAGREE——收敛达成

## 待定用户决策

- DEC-1: 执行契约中的 cuBLASLt 出现策略
  - Claude 立场：cuBLASLt kernel 出现应记录（kernel 名称和数量），但不视为契约违规。理由：cuBLASLt 可能在固定的 PyTorch/CUDA 栈上对 matmul 操作出现，即使没有显式用户请求，完全禁止在 RTX 5090 上可能不可行。记录保留透明度而不阻塞实验。
  - Codex 立场：cuBLASLt 出现应该明确允许（有文档化理由）或明确禁止（指定替代 matmul 路径）。保留"只记录不失败"是模糊的，使契约合规性意义减弱。
  - 权衡总结：禁止 cuBLASLt 给出更清晰的 kernel 分类，但在可用硬件/驱动栈上可能不可能。允许它改变 kernel 种群，但反映真实训练行为。计划目前默认采用 Claude 的立场（仅记录，不失败），以避免在可能不可避免的问题上阻塞实验。
  - 决策状态：`PENDING`

## 实施说明

### 代码风格要求
- 实施代码和注释不得包含计划特定术语，如"AC-"、"Milestone"、"Step"、"Phase"或类似的工作流标记
- 这些术语仅用于计划文档，不用于生成的代码库
- 在代码中使用描述性、领域适当的命名

### Evidence Tier 矩阵

实验在四个 evidence tier 中产生结果：

| Tier | 配置 | 角色 |
|------|------|------|
| Primary | bf16, seq_len=2048, 无 checkpointing | 主张接受所必需 |
| Fallback-A | bf16, seq_len=1024, 无 checkpointing | 敏感性表征 |
| Fallback-B | fp16, seq_len=2048, 无 checkpointing | 敏感性表征 |
| Fallback-C | bf16, seq_len=2048, gradient checkpointing | 敏感性表征 |

决策规则：主张接受要求 primary tier 成功。Fallback tier 表征敏感性，但不能替代 primary 证据。如果 primary tier 失败，实验不是"负面证据"——它是 BLOCKED 状态，并记录根本原因。

### 草稿完整性审计

原始草稿的所有章节均保留在本计划末尾。关键草稿要求及其计划处理：

- 草稿第 3 节（实验单元）：完全纳入——单 decoder layer、形状参数、组件列表，全部保留
- 草稿第 4 节（非目标）：完全保留——无完整模型、无 MLPerf、无 pretrained weights、第一轮无 NCU
- 草稿第 5 节（本地环境）：纳入环境门控（task1）和执行契约
- 草稿第 6 节（数据流）：纳入任务管线（task2 → task3 → task5 → task7 → task8）
- 草稿第 7 节（输出产物）：所有指定文件和目录保留；schema adapter 作为额外产物添加
- 草稿第 8 节（成功标准）：全部六个证据项映射到 AC-3 和 AC-4
- 草稿第 9 节（风险控制）：全部四个风险映射到 evidence tier（Fallback-A 对应 trace 过大、Fallback-C 对应 OOM、NCU 推迟、合成数据声明在报告中）
- 草稿第 10 节（后续路径）：作为上下文保留；不在本次实施范围内
- 草稿第 11 节（验收 Gate）：四个审查问题通过计划细化和 Claude-Codex 收敛得到处理

--- Original Design Draft Start ---

# 大训练单层压缩必要性实验设计

日期：2026-04-27

## 1. 目标

这份 spec 定义第一轮大训练 workload 实验，用来证明为什么在进入 GPU 精确周期仿真之前必须先做 representative compression。

这个实验的目标不是训练完整模型，不是复现 MLPerf 提交结果，也不是立刻完成完整 simulator validation。它的目标是构造一个可控但真实的规模证据：

```text
一个真实大模型训练层 -> kernel timeline -> invocation 规模 -> 压缩空间 -> projected simulation cost
```

核心论点是：

> 对现代训练 workload 来说，即使只是一个大模型单层，也可能产生足够多的 kernel invocation、运行时间和 trace 体积，使得直接走 full-trace exact-cycle simulation 不适合作为默认路径。因此 representative compression 不是单纯优化，而是后端精确仿真的前置条件。

## 2. 背景

当前仓库已经有一条可运行的小规模方法链：

```text
frontend anchor -> middle structure -> backend planning -> execution bridge -> result summary -> writeback
```

这条链路目前主要在 `mini_transformer_v4`、microbenchmarks 和经典 benchmark kernels 上跑通过。这些输入适合做 correctness gate、schema 稳定性检查和接口 bring-up，但它们太小，无法证明 compression 的必要性。

本实验把证据目标从“方法链能不能跑”转成“为什么现代训练 workload 必须先压缩”。

## 3. 实验单元

第一轮 workload 单元是一个 Llama-style decoder block 的训练 step：

```text
单个 decoder layer，随机合成 tokens / activations，forward + backward
```

这一层应包含现代 decoder-only Transformer block 的主要结构：

- RMSNorm 或 LayerNorm
- QKV projection
- attention score computation
- softmax
- attention value / context computation
- output projection
- MLP up / gate / down projections
- activation
- residual paths
- loss proxy
- backward pass

第一轮目标形状如下：

| 参数 | 数值 |
|---|---:|
| batch size | 1 |
| sequence length | 2048 |
| hidden size | 4096 |
| intermediate size | 14336 |
| attention heads | 32 |
| dtype | 优先 bf16，fallback 为 fp16 |
| device | CUDA GPU |
| measured region | warmup 之后的一次 forward + backward |

这个形状足够接近 8B 级 decoder layer 的结构和规模，但又不需要加载完整 pretrained model，因此应能在 32GB RTX 5090 上运行。

## 4. 非目标

这个实验刻意不做以下事情：

- 加载完整 Llama-8B 或更大的 pretrained weights；
- 跑完整模型所有层的 training step；
- 跑完整 dataset、epoch 或 MLPerf benchmark；
- 在第一轮就采集完整 NCU measured PKA features；
- 把所有生成的 kernel 直接送入 exact-cycle simulation；
- 证明最终 simulator accuracy。

这些属于后续阶段。第一轮只证明规模压力和 compression necessity。

## 5. 本地环境假设

2026-04-27 已观察到的本机事实：

- GPU：两张 NVIDIA GeForce RTX 5090，每张约 32GB 显存。
- 工具：`nsys`、`ncu`、`nvcc` 均可用。
- 当前 base Python 没有 PyTorch。
- 当前 `trace_gen` 环境有 CPU-only PyTorch 和 `transformers`。
- NCU performance counters 当前受 `ERR_NVGPUCTRPERM` 限制。

因此第一轮 implementation 应创建或使用 CUDA-enabled PyTorch 环境，并优先使用 Nsight Systems。Nsight Compute measured feature collection 延后到 performance counter 权限可用之后。

## 6. 数据流

实验数据流如下：

```text
large layer harness
  -> nsys profile
  -> exported kernel timeline
  -> invocation table
  -> compression summary
  -> scale proof report
```

### 6.1 Harness 输出

harness 应打印一份小型 machine-readable run summary，至少包括：

- model unit name
- batch size
- sequence length
- hidden size
- intermediate size
- number of heads
- dtype
- warmup count
- profiled iteration count
- forward / backward wall time
- peak GPU memory if available

### 6.2 Timeline 输出

Nsight Systems 输出应转换成 kernel invocation table，至少包含：

- invocation id
- kernel name
- start timestamp
- duration
- grid dimensions
- block dimensions
- stream id if available
- source profiler path

这张表是大规模 scale-proof 版本的现有 small-workload invocation table。

### 6.3 Compression 输出

第一轮 compression summary 可以先基于 name-and-shape grouping，因为第一目标是证明规模压力。它应报告：

- total kernel invocations
- unique kernel names
- unique kernel name plus launch-shape groups
- top groups by runtime coverage
- representative count at several coverage thresholds
- compression ratio under each grouping policy

等 NCU counter access 可用后，同一张 invocation table 可以继续扩展为 PKA measured feature records。

## 7. 输出产物

第一轮 implementation 应把实验产物写到：

```text
experiments/large_training_layer/
```

建议新增文件：

- `run_llama_layer_train.py`
- `run_nsys_layer.sh`
- `parse_nsys_kernels.py`
- `summarize_compression_scale.py`
- `results/llama_layer_b1_s2048_h4096/`

建议生成结果文件：

- `run_summary.json`
- `nsys_report.nsys-rep`
- `nsys_kernel_stats.csv`
- `kernel_invocation_table.json`
- `compression_scale_summary.json`
- `scale_proof_report.md`

## 8. 成功标准

实验成功的标准是产出一份 report，其中包含：

| 证据 | 含义 |
|---|---|
| total kernel invocations | 单层训练已经不是 trivial input |
| unique kernel groups | 存在明显 heterogeneity 和 grouping structure |
| top runtime coverage | 少数 groups 可能主导执行时间 |
| trace / report size | full multi-layer tracing 会明显放大成本 |
| forward / backward wall time | 可以推导 exact-cycle simulation budget |
| compressed representative count | representative compression 能减少 backend candidate count |

第一轮不需要证明最终 simulator speedup。它只需要证明未压缩输入规模使直接后端路径不合理。

## 9. 风险控制

### 9.1 Trace 过大

如果第一轮 `seq_len=2048` 的 run 太大，则把 sequence length 降到 `1024`，同时保持 hidden size 不变。报告中必须记录 fallback shape。

### 9.2 显存不足

如果 `bf16/fp16` training with backward 放不下，优先使用 gradient checkpointing 或降低 sequence length，然后才考虑降低 hidden size。目标是尽量保留 large-layer structure。

### 9.3 NCU Counter 权限

第一轮实验不应阻塞在 NCU 上。先用 `nsys` 生成 timeline 和 scale evidence。PKA measured features 标记为后续 acquisition step。

### 9.4 Synthetic Data 质疑

实验使用 synthetic activations / tokens，但 layer computation 是真实的。报告中必须明确这一点：本实验的 proof target 是真实训练层形状下的 GPU kernel scale 和 trace burden，不是模型质量或数据集收敛。

## 10. 后续路径

第一份 report 生成后，后续阶段是：

1. 当 NCU 权限可用后，为 representative kernel groups 补 measured PKA features。
2. 对比 `seq_len=1024`、`2048`，以及可能的 `4096` 下的 compression summary。
3. 加入一个非 Transformer training layer，例如 DLRM-style embedding / MLP 或 RetinaNet-style vision training，用来观察不同 workload family 下 compression difficulty 是否变化。
4. 只把 selected representatives 接入 backend planning path。

## 11. 验收 Gate

implementation 开始前，应审阅这份 spec 中的以下问题：

- single-layer Llama-style target 是否适合作为第一轮 workload；
- `seq_len=2048, hidden=4096` 是否适合作为第一轮形状；
- `nsys`-first evidence 是否足够支撑第一轮 scale proof；
- 输出产物是否足以接回 A 线 frontend compression。

--- Original Design Draft End ---
