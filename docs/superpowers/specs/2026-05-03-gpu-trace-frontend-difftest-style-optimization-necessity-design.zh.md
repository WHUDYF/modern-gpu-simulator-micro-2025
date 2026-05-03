# GPU Trace 前端 DiffTest 式优化必要性设计

> 工程 / 研究定位：这份 spec 用来判断，trace-driven GPU simulator 里是否真的需要、也是否适合做 DiffTest 式的前端输入重构。它不实现新的 trace 格式，不改变模拟器时序语义，也不主张直接把原始 RISC-V DiffTest checker 搬到 GPU 里。

**目标：** 判断当前 GPU trace-driven simulator 在 trace artifact 和 SM timing model 之间，是否存在可测的前端输入瓶颈；以及 DiffTest 风格的 preprocess、validate、delta/cache、batch/chunk、replay 思路，是否能在不改变模拟结果的前提下缓解这个瓶颈。

**架构：** 把现有 simulator wall time 拆成 trace read、protobuf parse、static binding、threadblock/warp trace loading、frontend instruction delivery 和 core timing model 几个阶段。用这组测量决定是否值得做一个最小前端优化原型。如果值得，优化点只放在 `trace-parser` 和 `trace-driven` 边界：decoded static-info cache、threadblock chunk staging、metadata-level squash、local replay。

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

## 2. 核心主张

要验证的主张是：

> 如果 trace event 数量大、碎片化严重，并且在到达 core timing model 之前要反复和静态元数据绑定，那么 trace-driven GPU simulation 就会出现一种 DiffTest 式的前端输入问题。

这个主张是可证伪的。如果大多数 simulator wall time 都花在 core timing model 里面，而 parser / trace-driven frontend 只占很小一部分，那它就是假的。如果前端输入处理在 simulator 时间里占有显著比例，或者随着 threadblock 数、warp trace 数、文件数、dynamic instruction 数扩张而变差，那它就足够支持优化动机。

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

### RQ1：trace frontend 是否占了 simulator 时间里显著的一部分？

测量 parser 和 trace-driven frontend 的工作，是否占 `T_sim` 的足够比例。

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

从 trace bottleneck cost map 里选已测 workload：

- simulator-throughput 样本：`atomic_add_bw`、`atomic_add_bw_conflict`、`mem_bw`、`mem_lat`
- export-dominated 对照样本：`l2_bw_32f`、`shared_bw`
- balanced 样本：`l2_bw_128`、`l1_bw_32f`

export-dominated 样本只作为控制组使用，不应拿来过度声称 simulator 前端优化。

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

1. 至少一个 simulator-throughput 或 balanced workload 的 `frontend_share >= 0.20`。
2. 至少一个 workload 的 static reuse ratio 很高，说明很多 dynamic instruction 其实对应很少的 `(unique_function_id, pc)`。
3. 最小无语义原型在选定 trace 上保持 simulator 输出指标不变。
4. 原型在一个或多个 frontend-heavy workload 上至少减少 15% 的 frontend 时间。
5. export-dominated 样本只作为控制组，不拿来过度声称 simulator-side 加速。

如果出现下面这些情况，这个方向就不够强，或者不值得继续：

- frontend share 一直低于 5%
- 重复 static binding 几乎不存在
- 大部分时间都花在 SM backend 的 timing structure 里
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

- `frontend_timing_breakdown.json`
- `frontend_timing_breakdown.md`
- `redundancy_profile.json`
- `redundancy_profile.md`
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

- 如果 frontend share 和 redundancy 都高：进入最小原型。
- 如果 frontend share 低但 export 高：回到 trace export / I/O compression 线。
- 如果 frontend share 低但 core cycle 高：优先 simulator backend throughput。
- 如果 benchmark sweep 是主要成本：优先 benchmark selection 和 family pruning。

在 Phase 3 和 Phase 4 之后：

- 如果正确性成立且 frontend 时间下降：继续做 chunking / prefetch 设计。
- 如果正确性失败：停止并缩小 cache 范围。
- 如果性能没有改善：记录负结果，不要过拟合。

## 13. 一句话主张

DiffTest 说明高频硬件事件在进入软件消费之前应该先结构化；这项研究要验证 GPU trace-driven simulation 是否也有同样的前端输入问题，以及 parser / trace-driven 边界重构能否在不改变时序语义的前提下降低 simulator 成本。
