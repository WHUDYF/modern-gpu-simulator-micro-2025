# Trace Compression Behavior Signature for Microbench Generation

日期：2026-04-28
状态：draft v1

## 1. 问题陈述

现有 GPU workload characterization、representative sampling 和 benchmark synthesis 方法通常依赖 flat profile counters、kernel-level summaries 或 static features。这些特征能帮助选择或描述 target kernel，但难以直接表达 trace 内部的 execution regularity、warp-path divergence 和 cross-threadblock memory similarity。

这类方法主要回答的是：

> 哪些真实 kernel 值得作为 backend planning / tuning target？

但它们没有充分回答另一个问题：

> 对一个 target kernel，我们能否构造一个更小、更可控、更透明的 microbench，复现它的关键执行结构？

本 spec 定义一条独立的学术线：将 trace compression 产生的中间结构和退化模式视为 workload behavior signature，用它描述 GPU target 的 trace-level 行为，并指导 AI agent 生成 microbench。

核心主张是：

> Trace compression is not only a storage reduction technique. Its intermediate structures and failure modes expose execution regularity, divergence, and cross-threadblock similarity. These signals can become a behavioral target for microbench generation.

## 2. 与现有流水线和工程线的边界

### 2.1 与代表性选择流水线的关系

本学术线可以消费任何来源的 target trace：

```text
target source
  -> target kernel / invocation
  -> trace or compressed trace
  -> behavior signature
  -> generated microbench target
```

可能的 target source 包括：

- 代表性选择流水线输出的 kernel；
- 现有 benchmark suite 中手工挑选的 kernel；
- microbenchmark / synthetic benchmark；
- 真实 workload 的 trace 中抽取的 kernel；
- 后续任意代表性选择实验的输出。

因此，这条学术线不需要绑定某个特定 selector，也不需要以某个特定 baseline 作为前提。

### 2.2 与工程线的关系

工程线关注 streaming trace compression：

```text
online trace read
  -> streaming compression
  -> smaller trace / faster transfer / replay support
```

本学术线关注 compression-derived behavior signal：

```text
trace compression artifacts
  -> signature extraction
  -> behavior comparison
  -> microbench generation target
```

两条线可以共享 compressed trace representation，但评价标准不同。

| 方向 | 主要问题 | 主要评价 |
|---|---|---|
| 工程线 | trace 能否在线压缩、传输、读取得更快 | compression ratio、I/O speedup、transfer speedup、replay correctness |
| 学术线 | 压缩结构能否表达 kernel 行为并指导 microbench | separability、stability、matching quality、diagnostic value |

## 3. 目标与非目标

### 3.1 目标

1. 定义一组 trace-compression behavior signature，用于描述 kernel execution structure。
2. 证明这些 signature 能区分不同 kernel 行为类型，例如 GEMM-like、reduction、elementwise、irregular memory、control-heavy。
3. 证明 compression failure / fallback 不是纯失败，而是 behavior complexity signal。
4. 建立 target kernel 与 generated microbench 之间的 signature matching 方法。
5. 形成一条从一般 GPU target trace 到 microbench synthesis target 的研究路径。

### 3.2 非目标

- 不实现完整 online / streaming trace compression 系统；
- 不把 compression-side feature 直接混入 flat-profile selector 的 grouping 字段；
- 不声称 generated microbench 与真实 workload 语义等价；
- 不要求第一阶段完成自动 agent search；
- 不要求第一阶段完成 backend simulator 调参闭环。

## 4. 核心概念

### 4.1 Target kernel

Target kernel 是真实 workload 中被选为研究对象的 kernel invocation。它可以来自：

- 代表性选择流水线；
- mini-transformer trace；
- microbenchmark trace；
- benchmark suite；
- 后续任意真实 workload trace。

Target kernel 是 ground truth observation，不是我们要改写的对象。

### 4.2 Microbench

Microbench 是人为编写或 AI agent 生成的小 CUDA 程序 / kernel。它不要求实现 target workload 的真实语义，只要求复现 target 的硬件行为结构。

例如：

- tiled dense compute；
- block reduction；
- shared-memory bank conflict；
- warp divergence；
- atomic contention；
- irregular gather / scatter；
- data-dependent indexing。

Microbench 的角色是 surrogate workload：小、可控、可重复、可调参。

### 4.3 Compression signature

Compression signature 是从 trace compression 过程中提取的结构化行为特征。它关注的不是最终文件大小，而是：

- 哪些部分能被 run-length / delta / shared-sequence 压缩；
- 哪些部分需要 override；
- 哪些部分退化为 full encoding；
- 这些模式在 warp、threadblock、kernel 之间如何分布。

## 5. 候选 Signature 维度

第一阶段不要求所有字段都一次实现。字段分为四组，按可获得性逐步推进。

### 5.1 Instruction regularity

用于描述 PC 序列和指令流是否规则。

| 字段 | 含义 |
|---|---|
| `instruction_run_coverage` | 可被 instruction run 覆盖的指令比例 |
| `instruction_run_length_p50` / `p90` | run length 分布 |
| `pc_delta_regular_rate` | PC delta 稳定的比例 |
| `shared_pc_sequence_coverage` | warp 间共享 PC 序列覆盖率 |

解释：

- 高覆盖率通常对应规则循环、dense compute、低控制流复杂度；
- 低覆盖率可能对应控制流复杂、频繁分叉或 trace 本身高度不规则。

### 5.2 Warp-path regularity

用于描述同一 threadblock 内 warp 之间的执行路径差异。

| 字段 | 含义 |
|---|---|
| `warp_pc_override_density` | warp 相对 shared PC sequence 的 override 密度 |
| `active_mask_override_density` | active mask 非默认值密度 |
| `predicate_mask_override_density` | predicate mask 非默认值密度 |
| `warp_diff_density` | warp-level diff 项密度 |

解释：

- 高 override 通常说明 warp divergence 或 predication 更明显；
- 低 override 通常说明 SIMT 路径高度一致。

### 5.3 Cross-threadblock regularity

用于描述不同 threadblock 是否共享同一行为模板。

| 字段 | 含义 |
|---|---|
| `cross_tb_delta_coverage` | 可由 base + delta 表示的 threadblock 比例 |
| `global_address_offset_coverage` | 可由统一地址偏移解释的访问比例 |
| `address_override_density` | 必须单独记录的地址 override 密度 |
| `full_encoding_fallback_rate` | 退化为完整编码的 threadblock 比例 |

解释：

- 高 `global_address_offset_coverage` 通常对应规则 tiling / stencil / GEMM-like 访问；
- 高 `address_override_density` 或 `full_encoding_fallback_rate` 通常对应 irregular memory、data-dependent access 或高度异质的 TB 行为。

### 5.4 Signature confidence

用于记录 signature 本身是否可信。

| 字段 | 含义 |
|---|---|
| `trace_segment_coverage` | signature 覆盖的 trace 片段比例 |
| `compressed_format_version` | 使用的 compressed trace 格式版本 |
| `fallback_reason_counts` | fallback 原因计数 |
| `missing_signal_fields` | 缺失字段列表 |

解释：

- 第一阶段可以接受不完整 signature；
- 但必须显式标记缺失字段，不能用默认值伪装完整观测。

## 6. 研究假设

### H1：Signature separability

Compression signature 能把已知行为类别分开。

例如：

- dense compute target 应表现为高 run coverage、低 warp diff、低 address override；
- reduction target 应表现为较强 shared-memory / synchronization pattern，可能有阶段性控制结构；
- irregular gather / scatter target 应表现为低 cross-TB delta coverage、高 address override；
- control-heavy target 应表现为较高 warp override 或 PC sequence irregularity。

### H2：Compression failure is diagnostic

压缩退化不是单纯的实现失败，而是行为复杂度信号。

例如：

- full encoding fallback 高，说明当前压缩模型无法用共享结构解释该 trace；
- address override 高，说明访存地址不符合简单 base + delta 模型；
- warp PC override 高，说明执行路径不能被单一 shared sequence 表达。

### H3：Signature matching improves microbench selection

如果一个 generated microbench 的 compression signature 比 unrelated microbench 更接近 target，则它更可能复现 target 的硬件行为结构。

第一阶段只要求证明：

- matching 能排除明显错误的 microbench；
- matching 能把手工挑选的相似 microbench 排在不相似 microbench 前面；
- matching 不只复制 runtime 或 instruction count。

### H4：Representative targets are useful inputs

来自任何代表性选择或手工挑选流程的 target kernels 都是有用输入，但本学术线不依赖某个特定 pipeline 才成立。

## 7. 数据流

### 7.1 第一阶段：离线可行性

```text
existing trace / compressed trace
  -> signature extractor
  -> signature table
  -> behavior separability report
```

第一阶段可以使用已有 trace，不要求在线压缩。

### 7.2 第二阶段：Target-to-microbench matching

```text
target kernel trace
  -> target signature

candidate microbench traces
  -> candidate signatures

signature distance / ranking
  -> matching report
```

这一阶段可以先使用手写 microbench 或已有 microbench，不要求 agent 自动生成。

### 7.3 第三阶段：Agent-guided generation

```text
target signature
  -> agent prompt / constraints
  -> generated microbench
  -> trace + compression signature
  -> similarity score
  -> agent revision
```

这一阶段才进入 AI agent microbench generation。

## 8. Signature Matching

第一版 matching 不追求复杂模型，优先使用可解释距离。

建议分三类距离：

1. **Regularity distance**
   - `instruction_run_coverage`
   - `shared_pc_sequence_coverage`
   - `pc_delta_regular_rate`

2. **Divergence distance**
   - `warp_pc_override_density`
   - `active_mask_override_density`
   - `warp_diff_density`

3. **Memory-structure distance**
   - `cross_tb_delta_coverage`
   - `global_address_offset_coverage`
   - `address_override_density`
   - `full_encoding_fallback_rate`

最终 score 可以先采用归一化加权距离：

```text
signature_distance =
    w_regular * D_regular
  + w_divergence * D_divergence
  + w_memory * D_memory
```

第一阶段权重不作为研究结论，只作为报告参数，必须进行 sensitivity analysis。

## 9. 验证设计

### 9.1 Positive controls

选择行为差异明显的 kernel / microbench：

- regular dense compute；
- reduction；
- elementwise；
- irregular gather / scatter；
- branch-heavy / divergence-heavy。

期望 signature 能表现出可解释差异。

### 9.2 Negative controls

构造 runtime 或 instruction count 接近、但行为结构不同的 pairs。

例如：

- instruction count 接近的 regular memory kernel 与 random gather kernel；
- runtime 接近的 dense compute kernel 与 branch-heavy kernel。

期望 compression signature 能把它们区分开。

### 9.3 Stability checks

同一 kernel 多次运行，signature 应在合理范围内稳定。

建议报告：

- per-field relative difference；
- signature distance across repeated runs；
- behavior category 是否保持不变。

### 9.4 Matching checks

给定一个 target，candidate microbench 排名应满足：

- 手工相似 candidate 排名高于明显不相关 candidate；
- 只匹配 runtime / instruction count 但结构错误的 candidate 排名不能过高；
- 缺失字段过多的 candidate 被标记为 low-confidence，而不是被当作高质量匹配。

## 10. 产物

本学术线建议产出以下文档和数据产物：

```text
experiments/trace_compression_behavior/
  signatures/
    target_signature_table.json
    candidate_microbench_signature_table.json
  reports/
    behavior_separability_report.md
    behavior_separability_report.json
    microbench_matching_report.md
    microbench_matching_report.json
```

后续如果进入实施计划，再定义具体 schema 和脚本路径。

## 11. 成功标准

第一阶段成功不要求自动生成 microbench。成功条件是：

1. 至少提取三类 signature 维度中的两类；
2. signature 能区分至少三种已知 kernel 行为类型；
3. 至少一个 negative control 被 signature 正确区分；
4. repeated runs 的 signature 稳定性被报告；
5. 所有缺失字段和 fallback 均显式记录；
6. 报告明确声明 signature 不作为 flat-profile selector 的隐式替代字段。

第二阶段成功条件是：

1. 对至少一个来自真实 workload 或 benchmark suite 的 target 生成 target signature；
2. 对至少三个 candidate microbench 生成 candidate signatures；
3. matching report 能给出可解释 ranking；
4. 排名结果能解释为什么某个 candidate 更像 target；
5. 至少一个 runtime / instruction-count 近似但结构错误的 candidate 被降权。

第三阶段成功条件是：

1. AI agent 能根据 target signature 生成 microbench 草案；
2. 生成 microbench 可编译、可运行、可 trace；
3. 至少一次 agent revision 能降低 signature distance；
4. 报告保留失败样例，说明哪些 signature 难以被生成代码复现。

## 12. 风险与控制

### 12.1 Signature 不等价于性能

Compression signature 只表达 trace structure，不保证 cycle count 或 cache behavior 完全一致。

控制：

- 把 signature matching 定位为 microbench candidate screening；
- 后续仍需 simulator statistics 或 hardware profiling 做二次验证。

### 12.2 压缩格式偏置

现有压缩格式能看到的行为，受格式设计影响。

控制：

- 报告 signature coverage 和 missing fields；
- 不把单一压缩率当作全部行为相似度；
- 将 fallback 视为诊断信号，而不是丢弃样本。

### 12.3 与 flat-profile selector 混淆

Compression-side feature 如果进入 flat-profile selector，会破坏 profile-based target selection 的可解释性。

控制：

- 本学术线只消费 target trace；
- 不向 profile-based selector 回写 compression feature；
- 所有报告必须声明 `scope: trace-compression-behavior-analysis`。

### 12.4 Microbench 过拟合 signature

Agent 可能生成只匹配某些 signature 字段、但缺少真实机制相似性的 kernel。

控制：

- 使用多组 signature；
- 加入 negative controls；
- 后续引入 simulator / profiler validation；
- 保留人工解释环节。

## 13. 推荐下一步

下一步不应直接实现 agent 自动生成 microbench，而应先做一个小范围 feasibility plan：

1. 盘点现有 trace / compressed trace 是否足以提取初始 signature；
2. 选择 5 到 8 个行为差异明显的 target / microbench；
3. 实现最小 signature extractor；
4. 生成 separability report；
5. 决定是否进入 target-to-microbench matching 阶段。

如果第一阶段 separability 不成立，本学术线应终止或重新定义 signature；如果成立，再进入 microbench synthesis。
