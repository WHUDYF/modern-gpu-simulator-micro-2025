# GPU Trace Frontend 必要性研究计划

## 执行摘要

这份计划是自包含的。执行者不需要预先知道 DiffTest 是什么。

本项目研究一个 trace-driven GPU simulator。当前要回答的不是“能不能加速整个 simulator”，而是：

```text
当 kernel 或 workload trace 已经存在后，
把 trace 数据转换成 simulator frontend 可消费输入到底花了多少时间？
```

计划必须产出证据，而不是假设。第一阶段交付物是一条可证伪的测量与建模流水线，用来判断 `T_trace_to_sim` 是否足够大，从而值得实现 frontend 优化原型。

## 这里的 DiffTest 风格是什么意思

DiffTest 是香山项目使用的 RISC-V 协同仿真与检查框架。香山文档中和本项目相关的部分，不是 RISC-V checker 本身，而是一个方法：高频硬件/软件通信可能成为瓶颈，因此通信路径可以通过 batch、state fusion 或 delta handling、non-blocking transfer、replay 等方式优化。

在本 GPU simulator 项目里，"DiffTest-style" 只表示 frontend input restructuring 的类比：

- batch：把大量小 trace event 聚合成更大的 threadblock、CTA 或 warp chunk 后再交给 simulator 消费；
- delta/cache：避免反复 decode 或 bind 相同的静态元数据，例如 `(unique_function_id, pc)`；
- validate/filter：规范化 simulator frontend 真正需要的字段，并提前拒绝格式错误的记录；
- replay：让 parser-to-frontend chunk 可重放，便于调试和性能回归隔离。

不要移植 RISC-V DiffTest checker。不要比较 RISC-V 架构状态。不要 squash 动态 GPU instruction event。不要改变 SM backend timing semantics。本计划的安全边界是：

```text
trace-parser -> trace-driven frontend -> shader core input
```

core timing model、scoreboard 行为、warp issue 顺序和 memory pipeline timing 都不在范围内。

## 目标描述

建立一条可复现的证据流水线，用来判断 trace frontend input restructuring 是否对 trace-driven GPU simulator 来说既必要又可行。

这份计划把设计 spec 转成可执行步骤：测量 `T_trace_to_sim`，构建面向 AI 训练的 workload 证据，估算 frontend-structuring 降耗空间，并把安全的原型边界限定在 `trace-parser` 与 `trace-driven` 的接口处。

spec 中的定量阈值，包括单次运行 30-60 秒、一次 sweep 10 分钟到 1 小时、15% / 30% / 50% 的降幅情景，以及 `P_trace_to_sim` 分档，都是建模和规划阈值，不是硬性的性能承诺。后续实测数据可以对它们进行校准。

## 已确认的研究范围

第一轮研究使用两个计量单位：

```text
primary unit: workload slice
secondary unit: training step
```

主要的早期指标是：

```text
P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done

T_kernel_to_sim_done =
  T_kernel_or_trace_export
+ T_trace_to_sim
+ T_sim_backend_execution
+ T_result_analysis
```

`T_kernel_or_trace_export` 包含 NVBit trace generation 或等价 trace export 时间。`T_result_analysis` 也包含在内，因为目标是完整流程：从获得 kernel 或 trace，到 simulator 执行完，再到结果处理完成。

第一阶段 go/no-go 规则刻意保持宽松：

```text
P_trace_to_sim_slice > 15%
OR
P_trace_to_sim_step > 15%
```

只要 slice 层面或 step 层面任意一个比例超过 15%，就认为 frontend preparation path 值得进入原型研究。这个 15% 阈值只是早期工程门槛，不是最终论文主张阈值。

第一轮 workload 固定为：

- T1 baseline: `BERT-base encoder layer slice`
- T1 baseline: `BERT-base pretraining full step`
- T2 representative: `Llama 3.1 8B decoder layer slice`
- T2 nice-to-have: `Llama 3.1 8B full step`，只在 RLCR 末尾尝试

`GPT-2 small` 不作为 fallback workload，因为它对当前证据线来说太小。

对于 `BERT-base pretraining full step`，batch size 从小 batch 开始，并逐步放大，直到触及已确认的资源上限：

```text
per-GPU memory: <= 28 GiB
trace + artifact size per workload unit: <= 500 GiB
single complete iteration time: <= 2 hours
```

## 验收标准

遵循 TDD 思路，每个标准都包含可确定性验证的正反测试。

- AC-1: workload 目录覆盖 AI 训练规模分层，并区分控制组和主张组。
  - 正向测试（应通过）:
    - workload 表包含必需的 T1 行：`BERT-base encoder layer slice` 和 `BERT-base pretraining full step`。
    - workload 表包含必需的 T2 行：`Llama 3.1 8B decoder layer slice`。
    - workload 表把 `Llama 3.1 8B full step` 记录为非阻塞的 nice-to-have 尾部尝试。
    - 每一行都记录模型或 slice 名称、近似规模、trace 粒度、预期 trace 大小区间，以及它在论证中的角色。
    - 现有 microbenchmark 和 export-dominated case 被列为 control 或附录材料，而不是 AI 训练的主证据。
  - 反向测试（应失败）:
    - 只包含 microbenchmark、没有 AI 训练或训练相关 trace 的目录会被拒绝。
    - 用 `GPT-2 small` 作为主 fallback、替代已确认的 BERT-base 与 Llama 3.1 8B 证据线的 workload 目录会被拒绝。
    - 用 export-dominated workload 去主张 simulator 侧前端加速的表格会被拒绝。

- AC-2: 每个 workload 的 trace-to-simulator 时间分解可以被测量，或被明确建模。
  - 正向测试（应通过）:
    - 每次测量运行都能报告 `trace_read_s`、`parse_pb_s`、`static_bind_s`、`tb_load_s`、`warp_trace_build_s`、`get_next_inst_s`、`core_cycle_s`、`total_sim_wall_s` 和 `frontend_share`。
    - 研究按 trace read、protobuf parse、static binding、threadblock / warp loading、frontend instruction delivery preparation 的总和计算 `T_trace_to_sim`。
    - 如果某个大型 workload 在第一轮无法直接测量，报告会把该值标为 modeled，并记录模型输入。
  - 反向测试（应失败）:
    - 只给出总 simulator wall time、没有前端分解的报告不合格。
    - 把 backend core timing 混进 `T_trace_to_sim` 却不标明的报告不合格。

- AC-3: trace-size 规划公式作为透明的 calculator artifact 实现。
  - 正向测试（应通过）:
    - calculator 实现 `T_trace_to_sim ~= C_fixed + S_trace_GiB / R_frontend_GiBps`。
    - calculator 支持 fast、expected 和 pessimistic 三种情景，且 `R_frontend_GiBps` 与 `C_fixed` 可配置。
    - 生成的 Markdown 和 JSON 输出包含从本地规模 trace 到至少 1 TiB scale-anchor trace 的大小行。
  - 反向测试（应失败）:
    - 只硬编码一种 trace 大小的 calculator 会被拒绝。
    - 不能在 expected 情景下复现 `T_trace_to_sim ~= 5 + 10 * S_trace_GiB seconds` 的 calculator 会被拒绝。

- AC-4: 对 slice 和 training-step 两个单位计算完整流程 burden ratio。
  - 正向测试（应通过）:
    - 证据流水线计算 `P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done`。
    - `T_kernel_to_sim_done` 包含 `T_kernel_or_trace_export`、`T_trace_to_sim`、`T_sim_backend_execution` 和 `T_result_analysis`。
    - 在对应测量可用时，报告同时计算 `P_trace_to_sim_slice` 和 `P_trace_to_sim_step`。
    - 早期 go/no-go 规则接受 `P_trace_to_sim_slice > 15%` 或 `P_trace_to_sim_step > 15%` 任一成立。
    - 报告在比例之外保留绝对时间和 sweep 级累计时间。
  - 反向测试（应失败）:
    - 要求前端优化必须先胜过 backend simulation acceleration 才算有用的研究会被拒绝。
    - 如果没有标注 alternate denominator，却把 trace export 或 result analysis 排除在完整流程分母之外，研究会被拒绝。
    - 只报百分比、不报绝对时间的研究会被拒绝。

- AC-5: redundancy profiling 用来判断 frontend caching 和 chunking 是否有本地机会。
  - 正向测试（应通过）:
    - profile 报告 dynamic instruction count、unique `(unique_function_id, pc)` count、static-info lookup count、threadblock count、warp trace count、metadata object construction count，以及可获得时的 frontend allocation count。
    - profile 计算 `static_reuse_ratio`、`tb_metadata_reuse_ratio` 和 `frontend_allocation_density`。
    - 至少一个 AI 训练 workload 或 model slice 能显示重复 static binding 是否足够大，从而值得做 frontend cache 原型。
  - 反向测试（应失败）:
    - 不测 unique static identifiers 却假设存在大量重复的 profile 会被拒绝。
    - 把 dynamic instruction squash 当作第一阶段允许优化的 profile 会被拒绝。

- AC-6: DiffTest-style 降幅模型只作用于 `T_trace_to_sim`。
  - 正向测试（应通过）:
    - 模型分别计算保守 15%、预期 30%、乐观 50% 的 `T_trace_to_sim` 降幅。
    - 输出包含 reduced `T_trace_to_sim`、每次运行节省时间，以及每次 sweep 节省时间。
    - 报告把该降幅模型标为 planning evidence，直到原型测量可以替换它。
  - 反向测试（应失败）:
    - 把降幅率直接套到 total simulator wall time 上、却没有正当理由的模型会被拒绝。
    - 把 15% / 30% / 50% 情景当作已实现速度提升来表述的模型会被拒绝。

- AC-7: 中央证据表把 workload 大小、前端成本、建模节省和论文论点连接起来。
  - 正向测试（应通过）:
    - 证据表包含 workload、measurement unit、model slice 或 step type、trace size、kernel count、threadblock 或 warp count、`T_trace_to_sim`、`T_kernel_to_sim_done`、`P_trace_to_sim`、估计的 frontend-structuring 降幅、reduced `T_trace_to_sim` 和完整流程影响。
    - 每一行区分 measured 值和 modeled 值。
    - 这张表既能支撑正面结论，也能支撑负面结论。
  - 反向测试（应失败）:
    - 缺少 trace size 或 `T_trace_to_sim` 的表会被拒绝。
    - 把 modeled 值伪装成 measured 值的表会被拒绝。

- AC-8: 原型边界安全，且 simulator timing semantics 保持不变。
  - 正向测试（应通过）:
    - 允许的原型范围仅限于 decoded static-info cache、metadata normalization cache、threadblock chunk staging，以及 parser / trace-driven 边界上的 local replay。
    - 等价性检查比较 `sim_cycle`、`sim_insn`、IPC、cache stats、kernel 顺序、每个 kernel 的 instruction count、warning 和 fatal 条件。
    - 任何 future semantic compression 或 dynamic instruction squash 都推迟到单独 spec。
  - 反向测试（应失败）:
    - 改变 scoreboard 依赖、warp issue 顺序、memory pipeline timing 或 SM backend timing 状态的原型会被拒绝。
    - 前端更快但 simulator 输出指标变化了的原型会被拒绝。

- AC-9: artifact 布局稳定且便于审阅。
  - 正向测试（应通过）:
    - 研究结果以 Markdown 和 JSON 形式写入 `artifacts/gpu_trace_frontend_difftest_necessity/`。
    - 必需 artifact 包括 workload evidence、trace-to-sim 公式、complete-flow burden ratio、frontend timing breakdown、redundancy profile、DiffTest reduction model、prototype equivalence report 和 paper argument matrix。
    - JSON artifact 便于机器读取，Markdown artifact 适合论文或 thesis 讨论。
  - 反向测试（应失败）:
    - 只存在于临时 console 输出里的结果会被拒绝。
    - 缺少足够元数据、无法复现实验假设的 artifact 会被拒绝。

- AC-10: 可选的 Llama 3.1 8B full-step validation 只在必需证据线完成后尝试。
  - 正向测试（应通过）:
    - 在可选 full-step 尝试开始之前，必须保留 `BERT-base pretraining full step`、`Llama 3.1 8B decoder layer slice`、公式建模、完整流程 burden ratio 和中央证据表等主线交付物。
    - 可选的 `Llama 3.1 8B full step` 尝试需要记录尝试次数、失败原因、部分 artifact，以及该结果是 measured 还是 abandoned。
    - 可选尝试多次失败时，不会使已经完成的必需 artifact 失效，也不会覆盖它们。
  - 反向测试（应失败）:
    - 把必需证据表阻塞在 `Llama 3.1 8B full step` 成功上的执行计划会被拒绝。
    - 用可选 full-step 的部分输出或失败输出覆盖早先完整结果的尝试会被拒绝。

- AC-11: batch scaling 遵守已确认的资源上限。
  - 正向测试（应通过）:
    - `BERT-base pretraining full step` 从小 batch 开始，并且只在运行保持在资源上限内时继续放大。
    - 资源上限记录为 per-GPU memory `<= 28 GiB`、trace plus artifact size per workload unit `<= 500 GiB`、single complete iteration time `<= 2 hours`。
    - 如果因为某个限制停止放大，报告会记录具体是哪一个限制触发了停止。
  - 反向测试（应失败）:
    - 未经用户明确批准就超过已确认资源上限的 batch-scaling run 会被拒绝。
    - 改变 batch size 却不记录资源使用情况的报告会被拒绝。

## 路径边界

### 上界（最大可接受范围）

实现可以包括完整的测量与报告流水线、workload 目录、trace-size calculator、complete-flow burden ratio calculator、redundancy profiler、DiffTest 风格降幅模型，以及一个带等价性报告的最小无语义原型。只要改动仍然保持在 trace parser 和 trace-driven frontend 边界内，还可以增加 simulator 侧 timing counters 和 artifact 生成工具。

### 下界（最小可接受范围）

最小可接受实现至少要产出 workload 目录、trace-size 公式 calculator、complete-flow burden ratio 报告、DiffTest 风格降幅表，以及带清晰 measured / modeled 标记的中央证据表。即使原型还没做出来，它也必须足以判断 frontend 重构是否值得进入原型阶段。

### 允许的选择

- 可以使用：现有 NVBit trace artifacts、现有 trace bottleneck map 输出、simulator timing counters、JSON artifacts、Markdown 摘要、仓库中已有的 shell 或 Python 报告脚本，以及 `trace-parser` 或 `trace-driven` 中的 C++ instrumentation。
- 可以使用：当第一轮无法拿到完整大规模 trace 时，为 T2 和 T3 scale anchor 使用明确说明假设的 modeled 值。
- 可以使用：DiffTest 仅作为“在软件消费前对高频事件进行结构化传递”的方法类比。
- 不可以使用：直接移植 RISC-V DiffTest checker、动态 GPU instruction squash、SM backend timing semantics 改动、scoreboard 或 memory pipeline 改动、export-time 优化主张，以及没有标注的外推性能结论。

## 可行性提示与建议

### 概念路径

先建立证据线，再优化 simulator：

1. 将 workload 元数据归一化到一个小目录中。
2. 用公式模型生成 trace-size 的时间估计。
3. 在 simulator-side trace frontend 上复用或增加 timing counters。
4. 输出包含 measured 和 modeled 值的中央证据表。
5. 计算完整流程 burden ratio 和 sweep 级累计成本。
6. 只对 `T_trace_to_sim` 计算 DiffTest 风格节省。
7. 再根据证据决定是否实现最小 frontend 原型。

核心计算应保持简单且可审计：

```text
T_trace_to_sim =
  T_trace_read
+ T_protobuf_parse
+ T_static_bind
+ T_threadblock_warp_load
+ T_frontend_instruction_delivery_preparation

P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done
```

### 相关参考

定义类比所需的外部参考：

- 香山 DiffTest 文档：<https://docs.xiangshan.cc/zh-cn/latest/tools/difftest/>。这里只借鉴通信优化思想：batch、delta/state fusion、non-blocking transfer 和 replay。
- Accel-Sim 项目页：<https://accel-sim.github.io/>。用于说明 trace-driven GPU simulation 是主流方法。
- Accel-Sim framework 仓库：<https://github.com/accel-sim/accel-sim-framework>。用于参考 GPU trace generation 和 trace-driven simulation workflow。
- gem5 TraceCPU 文档：<https://www.gem5.org/documentation/general_docs/cpu_models/TraceCPU>。用于说明 replayable trace representation 是体系结构模拟中的常见技术。
- ChampSim 仓库：<https://github.com/ChampSim/ChampSim>。用于说明 trace input format 和 trace consumption 是 simulator 的一等边界。
- SMARTS 概览：<https://users.ece.cmu.edu/~jhoe/doku/doku.php?id=smarts_simulation_sampling>。用于说明模拟成本可以通过代表性子集测量来降低；本计划不实现 SMARTS。
- SimPoint 项目页：<https://cseweb.ucsd.edu/~calder/simpoint/>。用于背景论证：体系结构模拟经常通过代表性执行来降低成本；本计划不实现 SimPoint。

本地参考：

- `docs/superpowers/specs/2026-05-03-gpu-trace-frontend-difftest-style-optimization-necessity-design.md` - 本计划的源 spec。
- `docs/superpowers/specs/2026-04-28-trace-compression-engineering-bottleneck-map-design.md` - 之前的 bottleneck map 叙事基础。
- `artifacts/trace_bottleneck_map/benchmark_cost_map.json` - 现有 cost map 输入。
- `artifacts/trace_bottleneck_map/benchmark_cost_map.md` - 现有 cost map 摘要。
- `docs/trace-benchmark-2026-04-03.md` - 之前的 trace benchmark 记录。
- `simulator-remodeled/gpu-simulator/trace-parser` - 预期的 parser 侧 instrumentation 边界。
- `simulator-remodeled/gpu-simulator/trace-driven` - 预期的 frontend 消费与 replay 边界。

## 依赖关系与执行顺序

### 里程碑

1. workload 目录与控制组
   - 定义 `BERT-base encoder layer slice`、`BERT-base pretraining full step` 和 `Llama 3.1 8B decoder layer slice` 的必需行。
   - 记录 `Llama 3.1 8B full step` 为非阻塞 nice-to-have 尾部尝试。
   - 第一轮证据线不使用 `GPT-2 small` 作为 fallback workload。
   - 标记 microbenchmark 和 export-dominated case 为 control。
   - 记录 trace 大小分层和测量可行性。

2. 公式与 burden 建模
   - 实现 trace-size 到 `T_trace_to_sim` 的 calculator。
   - 生成 fast、expected 和 pessimistic 三种情景。
   - 使用完整流程分母计算 `P_trace_to_sim_slice`、`P_trace_to_sim_step` 和 sweep 级累计成本。

3. BERT-base batch scaling guardrail
   - `BERT-base pretraining full step` 从小 batch 开始。
   - 逐步增大 batch size，直到 per-GPU memory、trace plus artifact size 或 single-iteration time 触及已确认资源上限。
   - 为每个 batch size 保留资源使用记录。

4. timing 分解仪表化
   - 找到现有 parser 和 trace-driven frontend 的边界。
   - 在 read、parse、bind、load、warp trace build、frontend delivery 和 core cycle timing 周围加低开销计时器和计数器。
   - 输出每次运行的 JSON 记录。

5. redundancy profiling
   - 统计 unique static identifiers、dynamic instructions、threadblocks、warp traces、metadata constructions 和 frontend allocations。
   - 计算 reuse 和 allocation-density 比率。
   - 对比 AI 训练 trace 和 control workload。

6. 证据表与论文论点矩阵
   - 合并 workload 目录、timing breakdown、公式估计、burden ratio、redundancy metrics 和降幅估计。
   - 生成 Markdown 和 JSON 报告。
   - 把 XiangShan DiffTest、Accel-Sim、gem5 TraceCPU、ChampSim、SMARTS 和 SimPoint 与本地证据需求对应起来。

7. 最小无语义原型决策
   - 只有在证据满足必要性标准时才继续。
   - 原型仅限于 decoded static-info cache、metadata normalization cache、threadblock chunk staging 和 local replay。
   - 在做任何性能主张之前先跑等价性检查。

8. nice-to-have 的 Llama 3.1 8B full-step validation
   - 只有在必需的 slice 和 local-step 证据完成后，才尝试 `Llama 3.1 8B full training-step`。
   - 这项工作作为 RLCR 尾部任务处理，不阻塞主线结果。
   - 如果因为 trace export、存储、simulator runtime 或基础设施限制导致多次尝试失败，就保留已经完成的必需 artifact 作为最终可用结果，并把失败证据单独记录。

## 任务拆解

将下面任务作为执行单元。输出必须确定、可复现，并以 artifact 为中心。

| 任务 ID | 描述 | 目标 AC | 依赖 |
|---------|------|---------|------|
| task1 | 创建 workload 目录 schema，并为 BERT-base slice、BERT-base pretraining full step、Llama 3.1 8B decoder layer slice、可选 Llama 3.1 8B full step 和 control workload 填充种子行 | AC-1, AC-10 | - |
| task2 | 检查现有 bottleneck map artifact，并把可复用字段映射到新的证据 schema | AC-1, AC-7 | task1 |
| task3 | 实现 trace-size 公式 calculator，并生成 Markdown / JSON 规划表 | AC-3 | task1 |
| task4 | 为 slice 和 step 单位实现使用明确 export、frontend、backend 和 analysis 字段的完整流程 burden ratio calculator | AC-4 | task3 |
| task5 | 找出 parser 和 trace-driven 中适合做 timing 分解的插桩点 | AC-2 | task1 |
| task6 | 增加或接入低开销计时器，并输出每次运行的 frontend timing JSON | AC-2, AC-9 | task5 |
| task7 | 找出可用于 redundancy profiling 的计数器或插入点 | AC-5 | task5 |
| task8 | 输出 static reuse、threadblock metadata reuse 和 frontend allocation density 的 redundancy profile | AC-5, AC-9 | task7 |
| task9 | 实现只作用于 `T_trace_to_sim` 的 DiffTest-style 降幅模型 | AC-6 | task3, task4 |
| task10 | 构建带 measured / modeled 标记的中央证据表生成器 | AC-7, AC-9 | task2, task4, task6, task8, task9 |
| task11 | 起草论文论点矩阵，把外部例子与本地 GPU simulator 证据连接起来 | AC-7, AC-9 | task10 |
| task12 | 定义最小无语义原型的门控条件和等价性报告检查清单 | AC-8 | task10, task11 |
| task13 | 在已确认资源上限下增加 BERT-base batch-scaling 记录和停止条件报告 | AC-11 | task1, task4 |
| task14 | 在必需 artifact 完成后尝试可选的 Llama 3.1 8B full-step validation；如果失败，则保留回退结果 | AC-10 | task10, task12, task13 |

## 决策记录

- 第一阶段应该先证明必要性和可行性，再实施 simulator 优化。
- DiffTest 类比应限制在结构化事件传递、缓存、批处理、验证和 replay。
- 本地优化边界应保持在 `trace-parser -> trace-driven -> shader core`，不要提前进入 backend timing semantics。
- 定量阈值适合作为规划阈值，并应通过测量校准。
- 15% 的 `P_trace_to_sim` 阈值是早期 go/no-go 门槛，不是最终论文主张阈值。
- 第一阶段只要 slice-level 或 step-level `P_trace_to_sim` 任一超过 15%，就足以进入原型研究。
- 第一轮证据线聚焦 BERT-base 和 Llama 3.1 8B；`GPT-2 small` 太小，不适合作为有意义的 fallback。
- Llama 3.1 8B full-step validation 对规模证据有帮助，但它应该作为必需证据线完成后的尾部尝试。
- 前端主导性 vs 设计循环阻塞：最终选择是证明 `T_trace_to_sim` 大到足以阻塞端到端迭代，而不是证明它压过所有 simulator 瓶颈。
- 只接受实测 vs 允许规模锚点建模：最终选择是要求本地 workload 必须实测，同时允许对 T2 和 T3 大规模锚点使用明确标注的 modeled 值。
- workload selection：最终选择是 `BERT-base encoder layer slice`、`BERT-base pretraining full step` 和 `Llama 3.1 8B decoder layer slice`，不使用 `GPT-2 small` fallback。
- Llama 3.1 8B full step vs Llama 3.1 8B layer slice：最终选择是要求 BERT-base full-step 测量和 Llama 3.1 8B layer-slice 证据，然后在 RLCR 末尾把 Llama 3.1 8B full-step validation 作为非阻塞 nice-to-have 尝试。
- BERT-base batch sizing：最终选择是从小 pretraining batch 开始，逐步放大到已确认资源上限。

## 待用户确认事项

- 当前没有待用户确认事项。这个计划把定量阈值视为规划和建模阈值，而不是硬性的性能保证。

## 实现说明

### 代码风格要求

- 实现代码和注释中不得包含 `AC-`、`Milestone`、`Step`、`Phase` 之类的计划术语。
- 代码和 artifact 中应使用描述性、符合领域习惯的命名。
- 计量开销要尽量低；如果测量开销变得可见，需要报告出来。
- 在做任何性能主张之前，必须保持 baseline simulator 输出语义不变。
- 每个生成的报告都要清楚标注 measured、modeled、extrapolated 和 control 值。
- 把 `P_trace_to_sim > 15%` 视为第一阶段工程门槛；更严格的论文主张阈值等实测数据出来后再确定。

### Artifact 约定

主要生成的 artifact 应放在：

```text
artifacts/gpu_trace_frontend_difftest_necessity/
```

预期文件如下：

- `workload_evidence_table.md`
- `workload_evidence_table.json`
- `trace_to_sim_formula.md`
- `trace_to_sim_formula.json`
- `complete_flow_burden_ratio.md`
- `complete_flow_burden_ratio.json`
- `frontend_timing_breakdown.md`
- `frontend_timing_breakdown.json`
- `redundancy_profile.md`
- `redundancy_profile.json`
- `difftest_reduction_model.md`
- `difftest_reduction_model.json`
- `prototype_equivalence_report.md`
- `prototype_equivalence_report.json`
- `paper_argument_matrix.md`
- `resource_bound_config.md`
- `resource_bound_config.json`
- `llama8b_full_step_attempt.md`（可选）
- `llama8b_full_step_attempt.json`（可选）
