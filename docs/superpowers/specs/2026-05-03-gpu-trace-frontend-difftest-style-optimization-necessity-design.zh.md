# GPU Trace 前端 DiffTest 式优化必要性设计

> 工程 / 研究定位：这份 spec 用来判断，trace-driven GPU simulator 里是否真的需要、也是否适合做 DiffTest 式的前端输入重构。它不实现新的 trace 格式，不改变模拟器时序语义，也不主张直接把原始 RISC-V DiffTest checker 搬到 GPU 里。

**目标：** 判断 trace-to-simulator 前端准备时间是否已经大到足以阻碍“算法/优化方法 -> simulator 验证”的端到端设计迭代；以及 DiffTest 风格的 preprocess、validate、delta/cache、batch/chunk、replay 思路，是否能在不改变模拟结果的前提下降低这部分成本。

**架构：** 把 trace-to-simulator 成本拆成 trace read、protobuf parse、static binding、threadblock/warp trace loading、frontend instruction delivery，以及 core timing model 的上下文成本。针对代表性 AI training trace 建立 workload evidence table，然后在 `T_trace_to_sim` 上估计 conservative / expected / optimistic 三档 DiffTest-style 降低幅度。如果值得推进，优化点只放在 `trace-parser` 和 `trace-driven` 边界：decoded static-info cache、threadblock chunk staging、metadata-level squash、local replay。

**技术栈：** 现有 NVBit 生成的 trace artifact、`simulator-remodeled/gpu-simulator/trace-parser`、`simulator-remodeled/gpu-simulator/trace-driven`、已有的 trace bottleneck cost map artifact、C++ timing counter、JSON/Markdown 测量报告，以及对 simulator 输出指标的本地回归检查。

---

## 1. 动机

前一条 trace compression 工程线已经说明，pipeline 的瓶颈并不是单一类别。有些 GPU microbenchmark 主要被 simulator throughput 限制，有些主要被 trace export / I/O 限制，还有一些是 balanced 或 fixed-overhead 主导。

这份 spec 只聚焦 `T_sim` 里面更窄的一个问题：

```text
当 trace 已经落盘以后，
把 trace artifact 变成 simulator-ready 的前端输入，到底要花多少时间？
```

这就是 DiffTest 真正有价值的那部分思想在 GPU 场景里的对应物。关键迁移点不是 CPU checker 本身，而是高频硬件事件不应作为零散 raw event 直接喂给软件消费端，而应先做标准化、过滤、缓存、批处理和可回放化。

本地已有的映射文档已经明确了目标边界：

```text
trace-parser -> trace-driven -> shader core
```

并且明确不从 `sm.cc`、`subcore.cc`、`ldst_unit_sm.cc`、scoreboard 逻辑或 memory timing 语义开始。

### 1.1 聚焦 AI training workload

这项研究有意把目标 workload 收束到 AI training 以及 training-adjacent trace 上。

原因是：

- AI training step 通常包含很多 kernel，而不是一个孤立 kernel。
- 多层网络会在 forward、backward、update 阶段反复出现相似执行结构。
- kernel、threadblock、warp trace 数量足够大，更容易让 frontend overhead 暴露出来。
- 静态指令形态往往高度重复，因此 static binding 和 metadata normalization 更可能被 cache 复用。

因此我们可以提出一个更强的假设：

> 相比小型 microbenchmark，AI training workload 更容易暴露 DiffTest 式的前端输入压力，因为它同时具备高事件数量和高结构重复。

后续测量可以按下面几层组织代表性 workload slice：

- mini-transformer 或 toy transformer trace
- GPT-style decode 或小规模 training step
- 大模型训练 trace 里的代表性 layer slice

目标不是声称所有大 workload 都一定慢，而是验证 AI training workload 是否会系统性放大 DiffTest 在另一个场景里解决的那类 frontend-input pattern。

### 1.2 端到端设计闭环阻碍目标

这项研究不需要证明 frontend restructuring 比 simulator backend 加速、减少 kernel 数量或 benchmark pruning 更重要。

更窄的目标已经足够：

> trace-to-simulator preparation time 是端到端 workflow 的实质阻碍；这个 workflow 指的是从算法或优化想法出发，生成 trace，加载到 simulator，并评估设计。

因此主指标不只是 `frontend_share`，而是：

```text
T_trace_to_sim =
  T_trace_read
+ T_protobuf_parse
+ T_static_bind
+ T_threadblock_warp_load
+ T_frontend_instruction_delivery_preparation
```

`T_sim_total` 仍然是有用上下文，但论证不要求 `T_trace_to_sim` 压过所有其他优化机会。只要它的绝对成本足够高，或者在 workload sweep 中累计成本足够高，就足以支撑这条线。

## 2. 核心主张

要验证的主张是：

> 如果 trace event 数量大、碎片化严重，并且在到达 core timing model 之前要反复和静态元数据绑定，那么 trace-driven GPU simulation 就会出现一种 DiffTest 式的前端输入问题。

这个主张是可证伪的。如果大多数 simulator wall time 都花在 core timing model 里面，而 parser / trace-driven frontend 只占很小一部分，那它就是假的。如果前端输入处理在 simulator 时间里占有显著比例，或者随着 threadblock 数、warp trace 数、文件数、dynamic instruction 数扩张而变差，那它就足够支持优化动机。

对这条工程线来说，即使 `frontend_share` 中等，只要 `T_trace_to_sim` 的绝对成本或累计成本足够高，这个主张也仍然成立。比如 frontend 只占 10%，但每轮设计 sweep 要处理大量 AI training trace 或反复跑多个 model slice，它仍然可能成为严重阻碍。

## 3. 论文动机用的主流例子

### 3.1 XiangShan DiffTest

XiangShan 的 DiffTest 是最接近的方法论锚点。它把硬件到软件的验证看成高频事件传输问题，而不是单纯的 checker 问题。关键原则是：事件要先组织好，再进入软件侧消费。

来源：

- XiangShan DiffTest 文档：<https://docs.xiangshan.cc/zh-cn/latest/tools/difftest/>
- XiangShan 项目文档：<https://docs.xiangshan.cc/>

用于我们的论证：

- DiffTest 说明 raw event 和 software-side consumption 之间应该有结构化边界。
- 我们不照搬 RISC-V checker 语义。
- 我们照搬的是 pipeline 思路：preprocess、validate、delta/cache、batch、replay。

### 3.2 Accel-Sim / GPGPU-Sim 的 trace-driven simulation

Accel-Sim 是直接相关的 GPU simulation 背景。它提供经过验证的 trace-driven GPU simulation，并消费 SASS 级别 trace。这个仓库本身就是围绕 tracer 生成 trace、再由 simulator 消费 trace 这一类问题组织起来的。

来源：

- Accel-Sim 项目：<https://accel-sim.github.io/>
- Accel-Sim 框架仓库：<https://github.com/accel-sim/accel-sim-framework>
- Accel-Sim 论文页：<https://accel-sim.github.io/accel-sim_website/>

用于我们的论证：

- GPU trace-driven simulation 本身就是主流方法。
- trace generation / trace consumption 边界本身就是自然的优化边界。
- 我们的贡献不是“trace-driven GPU simulation 存在”，而是测量并重构前端消费路径。

### 3.3 gem5 TraceCPU / Elastic Trace

gem5 的 TraceCPU 和 elastic trace 工作展示了另一种成熟模式：把执行捕获和 replay 分开，并构造一个可回放的中间表示，只保留对有用模拟足够的依赖信息。

来源：

- gem5 TraceCPU 文档：<https://www.gem5.org/documentation/general_docs/cpu_models/TraceCPU>
- gem5 文档：<https://www.gem5.org/documentation/>

用于我们的论证：

- 可回放的中间 trace 是一种被广泛接受的模拟器架构技术。
- 中间表示不只是存储格式，它决定 simulator 高效消费什么。
- 这支持把 local replay 做成 parser 和 trace-driven frontend 之间的一等能力。

### 3.4 ChampSim

ChampSim 是广泛使用的 trace-based simulator，常用于 cache 和 branch prediction 研究。它的流程本身就是围绕 trace 作为 simulator 输入组织的。

来源：

- ChampSim 仓库：<https://github.com/ChampSim/ChampSim>
- ChampSim 文档：<https://champsim.github.io/ChampSim/>

用于我们的论证：

- trace format 和 trace consumption 是主流 simulator 里的一级设计边界。
- 把 simulator input-path overhead 单独拿出来评估，是合理的。

### 3.5 SMARTS / SimPoint / sampled simulation

SMARTS 和 SimPoint 不是 frontend parser 优化，但它们说明架构模拟里常常要用代表性执行、采样或 phase selection 来降低模拟成本，同时保留足够精度。

来源：

- SMARTS 论文页：<https://dl.acm.org/doi/10.1145/605397.605403>
- SimPoint 项目页：<https://cseweb.ucsd.edu/~calder/simpoint/>

用于我们的论证：

- 更广泛的 simulation 社区早就接受：完整、无结构的事件消费往往太贵。
- 我们是在 GPU trace frontend 上施加同样的压力，只是层次更低。

## 4. 本地证据锚点

这份 spec 应该建立在现有本地 artifact 上，而不是凭直觉起步。

主要本地锚点：

- `docs/superpowers/specs/2026-04-28-trace-compression-engineering-bottleneck-map-design.md`
- `artifacts/trace_bottleneck_map/benchmark_cost_map.json`
- `artifacts/trace_bottleneck_map/benchmark_cost_map.md`
- `docs/difftest-optimization-mapping.md`（在 `difftest-doc` worktree 里）
- `docs/trace-benchmark-2026-04-03.md`

一个重要本地观察是：

现有 cost map 会把已测样本分成 `simulator throughput`、`trace export / I/O`、`balanced / mixed`、`capture / fixed overhead` 几类。这意味着前端输入研究只适用于 simulator-side 时间足够显著的那一部分，不应拿来解释 export 主导的案例，比如大型 trace 写出。

## 5. 研究问题

### RQ1：trace-to-simulator preparation 是否造成了实质设计迭代成本？

测量 parser 和 trace-driven frontend 的工作，是否占 `T_sim` 的足够比例，或者是否在端到端设计闭环中造成足够大的绝对 / 累计延迟。

目标拆解：

```text
T_sim =
  T_read_pb
+ T_parse_pb
+ T_static_bind
+ T_threadblock_load
+ T_warp_trace_build
+ T_get_next_inst_frontend
+ T_core_cycle_model
```

### RQ2：是否存在足够多的重复结构可以利用？

测量 dynamic trace event 是否在反复对应一个远小于动态事件数的静态标识集合。

需要统计的计数器：

- dynamic instruction count
- unique `(unique_function_id, pc)` count
- static-info lookup count
- static-info cacheable-hit opportunity count
- threadblock count
- warp trace count
- metadata object construction count
- frontend trace loading 里的 map/vector allocation count

### RQ3：优化是否能保持模拟语义不变？

最小原型必须保持：

- `sim_cycle`
- `sim_insn`
- IPC
- cache miss statistics
- kernel launch 和 completion order
- per-kernel instruction count
- simulator warning 和 fatal condition

### RQ4：哪些 DiffTest 风格思想可以安全迁移？

第一阶段可以安全迁移的：

- preprocess 的静态元数据绑定
- validate / filter 不用的 frontend 字段
- delta 作为 cache，而不是有损压缩
- batch 作为 threadblock / CTA / warp chunk staging
- parser 和 trace-driven 边界的 replay
- 只针对重复元数据构造做 squash

第一阶段不能做的：

- squash 动态指令事件
- 改变 fetch/decode 可见性
- 改变 scoreboard 依赖
- 改变 warp issue 顺序
- 改变 memory pipeline 时序
- 把新状态深度塞进 SM backend 结构

## 6. 拟议架构

### 6.1 测量层

在现有边界上加低开销 timing 和 counter。

主要边界：

- protobuf / trace file read
- protobuf parse
- static instruction binding
- address normalization 或 decompression
- threadblock trace loading
- warp trace map/vector construction
- `trace_shader_core_ctx::init_traces()`
- `trace_shader_core_ctx::get_next_inst()`
- `g_the_gpu->cycle()` 外层计时

输出：

- 每个 benchmark run 一条 JSON 记录
- 一张 Markdown 汇总表
- frontend 占 simulator wall time 的比例
- redundancy ratio

### 6.2 前端中间表示

如果 RQ1 和 RQ2 证明值得优化，就引入一个只存在于 simulator 前端内部的表示：

```text
TraceFrontendChunk
  kernel_id
  cta_id / threadblock_id
  warp_chunks[]
  decoded_static_refs[]
  metadata_refs[]
```

这在 v1 不替换磁盘上的 trace 格式。它只是 parser 和 trace-driven frontend 之间的内部表示。

### 6.3 Delta / Cache 层

加入只作用于安全语义边界的 cache：

- per-kernel decoded static-info cache
- `(unique_function_id, pc)` decoded instruction cache
- per-threadblock metadata normalization cache

cache 里存的是已经解码好的 metadata，而不是动态时序状态。

### 6.4 Batch / Chunk 层

按 threadblock / CTA / warp chunk 来加载 trace，而不是散乱地往 frontend 写入。

目标：

- 减少小对象构造
- 减少重复 map 插入
- 为后续 prefetch / double-buffer 打基础
- 保持 CTA launch 和 completion ordering

### 6.5 Replay 层

在 parser / trace-driven 边界增加 local replay 模式。

Replay 目标：

- 一个 kernel
- 一个 CTA / threadblock
- 一个 frontend chunk

Replay 主要用于调试和性能回归隔离，不是新的 timing model。

## 7. 实验设计

### Phase 0：基准选择

使用当前 trace bottleneck cost map 作为校准基线，但主要 evidence table 转向 AI training 和 training-adjacent trace。

推荐 workload 类别：

- mini-transformer 或 toy transformer training trace
- GPT-2 small training 或 decode trace
- BERT / transformer encoder layer trace
- Llama-style decoder-only layer slice
- MLPerf Training-style reference anchor

现有 cost map 里的 microbenchmark 控制组仍然放在 appendix：

- simulator-throughput 样本：`atomic_add_bw`、`atomic_add_bw_conflict`、`mem_bw`、`mem_lat`
- export-dominated 对照样本：`l2_bw_32f`、`shared_bw`
- balanced 样本：`l2_bw_128`、`l1_bw_32f`

export-dominated 和 microbenchmark 样本只是控制组。它们不应用来过度声称 AI-training frontend 优化，但可以帮助说明这项研究有意聚焦 trace-to-simulator frontend cost。

### Phase 1：时间拆解

对每个选定 trace，用和现有 bottleneck map 相同的 simulator 配置和固定 cycle window 跑。

每个 workload 必须输出：

| metric | meaning |
|---|---|
| `total_sim_wall_s` | simulator 进程总 wall time |
| `trace_read_s` | trace file / protobuf read 时间 |
| `parse_pb_s` | protobuf parse 时间 |
| `static_bind_s` | static metadata binding 时间 |
| `tb_load_s` | threadblock trace loading 时间 |
| `warp_trace_build_s` | warp trace 结构构造时间 |
| `get_next_inst_s` | frontend instruction delivery 时间 |
| `core_cycle_s` | 剩余 core cycle model 时间 |
| `frontend_share` | frontend 各项时间占 total sim wall time 的比例 |

### Phase 1.5：Workload Evidence Table

建立一张直接支撑论文论点的表：

| workload | model slice | trace size | kernel count | TB / warp count | `T_trace_to_sim` | `T_sim_total` | frontend share | estimated DiffTest-style reduction | reduced `T_trace_to_sim` | E2E impact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mini-transformer | full toy step | measured | measured | measured | measured | measured | measured | modelled | modelled | measured/modelled |
| GPT-style | decode or small train step | measured/modelled | measured/modelled | measured/modelled | measured/modelled | measured/modelled | measured/modelled | modelled | modelled | modelled |
| BERT-style | encoder layer | measured/modelled | measured/modelled | measured/modelled | measured/modelled | measured/modelled | measured/modelled | modelled | modelled | modelled |
| Llama-style | decoder layer slice | modelled | modelled | modelled | modelled | modelled | modelled | modelled | modelled | modelled |
| MLPerf-style | training reference anchor | scale anchor | scale anchor | scale anchor | scale anchor | scale anchor | scale anchor | modelled | modelled | scale argument |

这张表是核心 artifact。它证明 trace-to-simulator 时间是否已经成为端到端设计迭代中的实际阻碍。`T_sim_total` 只作为上下文，不是 frontend 优化必须打败的对手。

### Phase 2：冗余测量

统计冗余指标：

```text
static_reuse_ratio = dynamic_instruction_count / unique_function_pc_count
tb_metadata_reuse_ratio = threadblock_count / unique_tb_metadata_shape_count
frontend_allocation_density = frontend_allocations / dynamic_instruction_count
```

最低有效信号：

- static reuse ratio 明显大于 1
- frontend allocation density 不小
- frontend share 足够高，值得关注

### Phase 2.5：DiffTest-Style Reduction Model

只在 `T_trace_to_sim` 上估计收益，不在整个 simulator wall time 上估计收益。

使用三档显式降低幅度：

| scenario | reduction applied to `T_trace_to_sim` | meaning |
|---|---:|---|
| conservative | 15% | 保守的 cache / metadata reuse 下界收益 |
| expected | 30% | cache 加 chunking 在重复 training trace 上的预期收益 |
| optimistic | 50% | cache、batch、replay-locality 都较强时的乐观收益 |

对每个 workload：

```text
reduced_T_trace_to_sim = T_trace_to_sim * (1 - reduction_rate)
saved_time_per_run = T_trace_to_sim - reduced_T_trace_to_sim
saved_time_per_sweep = saved_time_per_run * number_of_design_runs
```

这只是规划模型，不是性能声明。后续原型实现后，必须用实测降低幅度替换这些估计。

### Phase 3：最小无语义原型

只做最安全的变换：

1. decoded static-info cache
2. metadata normalization cache
3. threadblock chunk staging
4. local replay harness

这一阶段不要做 dynamic instruction squash。

### Phase 4：正确性和性能评估

对比 baseline 和优化后结果。

正确性表：

| metric | required relation |
|---|---|
| `sim_cycle` | 一致，除非已有文档说明 simulator 存在非确定性 |
| `sim_insn` | 一致 |
| IPC | 一致，或者能解释为相同分子 / 分母 |
| cache stats | 一致 |
| kernel order | 一致 |
| warning / fatal 输出 | 不应出现新 warning 或 fatal |

性能表：

| metric | desired direction |
|---|---|
| frontend wall time | 更低 |
| total sim wall time | 更低或不变 |
| static bind time | 更低 |
| threadblock load time | 更低 |
| map/vector allocation count | 更低 |

## 8. 成功标准

如果满足下面所有条件，这条研究方向就算站得住：

1. evidence table 至少包含三个代表性 AI training / training-adjacent workload slice。
2. 至少一个 workload 的 `T_trace_to_sim` 实测或合理建模后超过实际单次运行阈值，比如 30-60 秒。
3. 多 workload 或多配置 sweep 中，累计 `T_trace_to_sim` 已经足以拖慢设计迭代，比如达到 10 分钟到 1 小时。
4. 至少一个 workload 的 static reuse ratio 很高，说明很多 dynamic instruction 其实对应很少的 `(unique_function_id, pc)`。
5. conservative / expected / optimistic reduction table 显示 trace-to-simulator 部分有明确节省时间。
6. export-dominated 样本只作为控制组，不拿来过度声称 simulator-side 加速。

如果出现下面这些情况，这个方向就不够强，或者不值得继续：

- `T_trace_to_sim` 的绝对成本和累计成本都可以忽略
- 重复 static binding 几乎不存在
- AI training trace 没有比 microbenchmark 控制组表现出更强的 frontend pressure
- 只改 frontend 后正确性测试不稳定

## 9. 非目标

- 第一轮不替换磁盘上的 trace format。
- 这份 spec 不实现 streaming trace compression。
- 不优化 NVBit trace export 时间。
- 不修改 SM backend 的 timing semantics。
- 不移植 RISC-V DiffTest checker。
- 不 squash dynamic instruction events。
- 只有 simulator-side measurement 支持时，才声称 export-dominated workload 有性能收益。

## 10. 风险

### 10.1 Squash 误解

最大风险是把 DiffTest 的 `Squash` 误解成“可以压扁 GPU 动态指令流”。这很可能破坏 fetch/decode 时序、scoreboard 依赖、warp issue 顺序和 memory pipeline 可见性。

缓解方式：

- 第一阶段只允许 metadata-level squash；
- 要求输出指标等价；
- 未来如果要做语义压缩，单独写另一份 spec。

### 10.2 前端 / 后端边界漂移

如果优化状态过早泄漏到 SM backend 结构里，会很难理解，也很难验证。

缓解方式：

- v1 变化只放在 parser 和 trace-driven frontend 之间；
- 向消费者暴露不可变的 chunk/cache 记录；
- 不要在 cache 里加入可变 timing state。

### 10.3 过度类比 DiffTest

DiffTest 和 GPU trace-driven simulation 不是同一个系统。

缓解方式：

- 只把 DiffTest 当方法论类比；
- 必须靠本地 timing 和 redundancy evidence 才能声明必要性；
- 一定要包含 export-dominated 控制组。

## 11. 预期产物

建议的 artifact 路径：

```text
artifacts/gpu_trace_frontend_difftest_necessity/
```

预期文件：

- `workload_evidence_table.json`
- `workload_evidence_table.md`
- `frontend_timing_breakdown.json`
- `frontend_timing_breakdown.md`
- `redundancy_profile.json`
- `redundancy_profile.md`
- `difftest_reduction_model.json`
- `difftest_reduction_model.md`
- `prototype_equivalence_report.json`
- `prototype_equivalence_report.md`
- `paper_argument_matrix.md`

`paper_argument_matrix.md` 应该明确把每个外部例子和本地 GPU simulator 论点对应起来：

| external example | transferable idea | local GPU analogue | evidence needed |
|---|---|---|---|
| XiangShan DiffTest | structured event transfer | trace frontend staging | frontend share + repeated events |
| Accel-Sim | trace-driven GPU simulation | SASS trace consumption | simulator-side timing breakdown |
| gem5 TraceCPU | replayable trace representation | local CTA / chunk replay | reproducible frontend replay |
| ChampSim | trace as simulator input boundary | trace parser / consumer boundary | parser cost and format pressure |
| SMARTS / SimPoint | reduce full event consumption | representative frontend chunks | future extension, not v1 proof |

## 12. 决策规则

在 Phase 1 和 Phase 2 之后：

- 如果 `T_trace_to_sim` 的绝对成本高：即使 `frontend_share` 中等，也进入最小原型。
- 如果单次 `T_trace_to_sim` 中等，但 sweep-level 累计成本高：进入最小原型。
- 如果 `T_trace_to_sim` 低，累计成本也低：不优先推进这条线。
- 如果 trace 到达 simulator 之前 export 已经主导：把它报告为 export / I/O pressure，而不是 frontend input pressure。
- 如果 backend simulation 主导，但 `T_trace_to_sim` 仍然大到阻碍迭代：保留这条线作为独立 frontend 优化，不声称它替代 backend acceleration。

在 Phase 3 和 Phase 4 之后：

- 如果正确性成立且 frontend 时间下降：继续做 chunking / prefetch 设计。
- 如果正确性失败：停止并缩小 cache 范围。
- 如果性能没有改善：记录负结果，不要过拟合。

## 13. 一句话主张

DiffTest 说明高频硬件事件在进入软件消费之前应该先结构化；这项研究要验证 GPU trace-driven simulation 是否也有同样的前端输入问题，以及 parser / trace-driven 边界重构能否在不改变时序语义的前提下降低 simulator 成本。
