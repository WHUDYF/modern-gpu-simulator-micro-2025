# LLM-Assisted Mechanism Microbench Anchor Dataset Design

日期：2026-05-26

## 1. 目标

本文档定义第一版 `mechanism_microbench_anchor_dataset` 的生成与使用方式。目标不是让 LLM 直接生成 ground truth 标签，而是让 LLM 协助生成**机制清晰、可验证、可复用**的 microbench anchor，并将这些 anchor 通过 profiling、counter collection 和 simulator validation 转换成训练图网络所需的弱监督数据。

核心链路：

```text
registry-driven mechanism target
  -> LLM-assisted microbench generation
  -> compile / run / profile
  -> automatic feature collection
  -> counter-confirmed mechanism anchors
  -> optional knob validation
  -> graph training / candidate ranking
```

第一版的核心价值是：把“人工判断某个 kernel 属于哪类硬件资源消耗”的问题，拆成“先构造单机制 anchor，再由自动采集的特征与验证结果决定它是否真能代表该机制”。

---

## 2. 非目标

第一版不做：

1. 不把 LLM 输出当作最终标签；
2. 不把 design intent 直接当作 validated mechanism；
3. 不要求所有 microbench 都能覆盖真实 workload 的混合行为；
4. 不要求人工逐条审查每个 kernel 的特征；
5. 不绕过 profiling / validation 直接训练图网络；
6. 不把 instruction ratio 直接当作机制重要性或性能贡献；
7. 不把 proxy feature 伪装成 measured feature。

LLM 只负责生成候选 microbench 设计与说明，真正可训练的证据必须来自编译、运行、profile 和闭环验证。

---

## 3. 数据集定义

### 3.1 数据集目标

`mechanism_microbench_anchor_dataset` 由三类样本组成：

```text
design-intent anchors
counter-confirmed anchors
knob-validated anchors
```

它们分别表示：

1. 这个 microbench 的设计目标是什么；
2. 它运行时是否真的激活了预期的硬件资源压力；
3. 对应 simulator knob 是否在 target/control 比较中产生了可验证变化。

### 3.2 样本粒度

第一版建议以 `microbench_anchor` 为最小单元，而不是单个 kernel invocation。

原因：

1. microbench 本身通常是少量、结构清晰的 kernel；
2. anchor 粒度更容易与 B-Line graph reasoner 对接；
3. anchor 可以天然携带 design intent、counter summary 和 validation state；
4. anchor 更适合作为 prototype supervision 的节点。

### 3.3 建议字段

每个 anchor 记录至少包含：

```json
{
  "anchor_id": "microbench_shared_memory_bank_conflict_v1",
  "anchor_type": "mechanism_microbench",
  "target_family": "streaming_memory",
  "target_subtype": "shared_memory_bank_path",
  "target_knob": "shared_memory_latency_or_bank_conflict_related_knob",
  "design_intent": "stress shared memory bank conflict path",
  "generator": {
    "type": "llm",
    "model": "claude-code",
    "prompt_version": "microbench_anchor_v0.1"
  },
  "compile_status": "success",
  "profile_status": "success",
  "feature_status": "auto_collected",
  "validation_status": "not_validated",
  "claim_status": "design_intent_only"
}
```

---

## 4. 生成流程

### 4.1 Registry-driven target selection

第一步不是让 LLM 自由生成，而是先由 registry 决定要覆盖哪些机制。

输入：

```text
canonical family enum
canonical subtype registry
subtype-to-knob map
known validated seeds
existing proxy anchors
```

输出：

```text
mechanism target list
```

例如：

```text
dense_compute / fp64_dp_pipeline_compute
streaming_memory / shared_memory_bank_path
memory_hierarchy / global_memory_latency_path
control_divergence / branch_heavy_path
occupancy_limited / register_pressure_path
atomic_contention / shared_memory_atomic_path
tensor_core / mma_path
```

### 4.2 LLM-assisted microbench synthesis

LLM 的任务是为每个 target 生成候选 microbench 方案，包括：

```text
kernel shape
loop structure
memory access pattern
thread/block configuration
expected stress mechanism
expected negative control
```

LLM 输出必须是结构化 JSON，而不是自然语言散文。每个候选都要带：

```text
target_mechanism
implementation sketch
expected profiling signature
abstain if uncertain
```

### 4.3 Compile and run

候选 microbench 进入自动化执行链：

```text
source generation
  -> compile
  -> execute
  -> collect runtime / counter / trace outputs
```

任何编译失败、运行失败、数值不稳定或 profile 缺失的样本都不能进入 counter-confirmed 层，只能留在 design-intent 层或被丢弃。

### 4.4 Automatic feature collection

feature collection 必须尽量自动完成，不依赖人工逐条判断。

可采集的特征来源：

```text
hardware performance counters
trace / SASS / warp stall summaries
derived ratio features
roofline-style quantities
resource axis weights
graph embeddings
```

第一版将 feature collection 分成两条线：

1. **计数器线**：从 NCU / CUPTI / profiler 采集原始 counters 和 derived metrics；
2. **结构线**：从 trace graph / SASS / dependency graph 自动生成 embedding 或 graph summary。

人工只处理两类例外：

```text
compile / run / profile failure
counter ambiguity requiring annotation
```

### 4.5 Validation and promotion

如果 microbench 有对应 knob，可进一步进入 closed-loop validation：

```text
baseline run
modified run
target/control comparison
```

只有通过这一层的 anchor 才能进入 `knob-validated`。

---

## 5. 标签层级

### 5.1 Design intent label

由 LLM 或 human design 给出，表示“这个 microbench 想测什么”。

示例：

```json
{
  "label_type": "design_intent",
  "label": "shared_memory_bank_path",
  "confidence": 0.95,
  "source": "llm_generation"
}
```

### 5.2 Counter-confirmed label

由自动 profiling 结果决定，表示“这个 microbench 实际激活了什么”。

示例：

```json
{
  "label_type": "counter_confirmed",
  "label": "shared_memory_pressure",
  "evidence": [
    "shared_memory_pressure_proxy",
    "stall_reason_shared_dependency",
    "bank_conflict_indicator"
  ],
  "confidence": 0.84
}
```

### 5.3 Knob-validated label

由 before/after validation 决定，表示“这个 microbench 对应的 simulator knob 确实可被验证”。

示例：

```json
{
  "label_type": "knob_validated",
  "knob": "trace_opcode_latency_initiation_dp",
  "target_delta": -0.43572,
  "control_stable": true,
  "claim_status": "simulator_internal_closed_loop"
}
```

---

## 6. 自动特征收集策略

### 6.1 原始特征

优先保留原始 counters，而不是只保留 ratio：

```text
instruction counts
memory counts
stall counts
cache metrics
occupancy metrics
atomic metrics
warp sampling metrics
```

### 6.2 派生特征

确定性公式可用于生成描述性特征，但这些特征只能作为输入，不应被写成最终归因：

```text
memory_ratio
compute_to_memory_ratio
branch_ratio
sync_ratio
global_memory_pressure_proxy
shared_memory_pressure_proxy
compute_instruction_pressure_proxy
```

### 6.3 图表示征

对于结构线，建议自动构造：

```text
instruction graph
dependency graph
basic-block graph
trace graph
resource-axis graph
```

这些图可以输入 GNN 或 graph contrastive model，用于自动学习 kernel 表征，而不依赖手工设计的全部特征。

---

## 7. 训练用途

`mechanism_microbench_anchor_dataset` 主要服务两类训练目标：

1. **图网络的 prototype supervision**
   - microbench anchor 作为已知机制原型；
   - 训练图网络学会 anchor 间相似性；
   - 训练后迁移到真实 workload kernel。

2. **候选排序和 validation planning**
   - microbench 的 counter-confirmed/validated 结果作为高质量 seed；
   - 帮助 B-Line graph reasoner 给真实 kernel 做 family/subtype/knob 排序。

---

## 8. 风险边界

### 8.1 不能把 design intent 当 ground truth

LLM 生成的 microbench 设计意图不等于真实机制。

### 8.2 不能只靠指令数判断资源消耗

instruction count 和 ratio 只能提供弱描述，不能单独说明 compute-bound / memory-bound / latency-bound。

### 8.3 不能忽略真实 workload 与 microbench 的分布差

microbench 太干净，真实 kernel 太混合。训练必须包含：

```text
prototype anchors
proxy anchors
validated seed anchors
boundary / mixed anchors
```

### 8.4 不能把 LLM 输出写成真值标签

LLM 只能生成候选。标签升级必须依赖自动 profiling 和 validation。

---

## 9. 验收标准

第一版实现完成后，必须满足：

1. 能按 registry 批量生成 mechanism target list；
2. 能用 LLM 生成结构化 microbench 候选；
3. 能自动编译、运行并采集 profiling 特征；
4. 能区分 design-intent、counter-confirmed、knob-validated 三类标签；
5. 能导出 anchor-level dataset card；
6. 能把 microbench anchor 接到 B-Line graph reasoner；
7. 能在没有人工逐条判断的情况下完成大部分 feature collection；
8. 只有失败样本和歧义样本才需要人工介入。

---

## 10. 后续接口

第一版完成后，后续可直接接入：

```text
raw microbench anchors
  -> evidence signature builder
  -> graph reasoner
  -> validation planner
  -> weak training target builder
```

这使 microbench 不只是“测试样例”，而是可复用的机制原型数据资产。
