# A 线 GCL-M1/M2 Graph Embedding Roadmap Design Spec

日期：2026-05-29

## 1. 定位

这份 spec 是 GCL-M1/M2 graph embedding 路线的主入口。它不再承载所有实现细节，而是定义：

- GCL-M1/M2 的总体目标；
- Phase A/B/C 的执行顺序；
- 每个阶段的 artifact 边界；
- 与 M0 selector、M1 trace graph construction 和后续 M2 implementation plan 的关系；
- 哪些问题必须先做，哪些问题必须后置。

详细实现被拆到三个副 spec：

```text
Phase A: semantic end-to-end GCL
  -> docs/superpowers/specs/2026-06-01-a-line-gcl-m1-m2-phase-a-semantic-e2e-design.md

Phase B: scalable graph embedding path
  -> docs/superpowers/specs/2026-06-01-a-line-gcl-m1-m2-phase-b-scalable-graph-design.md

Phase C: compression / abstraction path
  -> docs/superpowers/specs/2026-06-01-a-line-gcl-m1-m2-phase-c-compression-abstraction-design.md
```

这份主 spec 专门回答一个问题：

```text
我们应该按什么顺序，把 GCL 从小规模语义闭环推进到真实 trace 可扩展训练，
最后再推进到 Photon-inspired compression / abstraction？
```

## 2. 背景问题

GCL 的目标不是把 PKA 的 selector 换成另一个手写规则，而是让 kernel invocation 先经过图表示学习：

```text
trace records
  -> canonical graph artifacts
  -> graph encoder
  -> kernel embedding
  -> M0-compatible selector
```

直接把真实 kernel 的全部动态 trace 展开成一个巨型 graph 不是合理默认路径。原因是：

- 一个 kernel invocation 内部动态指令数可能达到上万甚至更多；
- instruction nodes、variable nodes 和 data-flow edges 会进一步放大 graph size；
- RGCN 训练显存、batch 组织和 replay 成本会失控；
- 如果 M1 不记录 graph scope 和 graph size，M2 无法判断某个 graph 是否适合训练；
- 如果直接对 full graph 做 pooling，warp-level SIMT 结构会被过早抹平；
- 如果第一版直接引入 stream dedup / abstraction，定位问题会变得困难。

因此，GCL-M1/M2 必须先做语义闭环，再做规模化，最后做压缩抽象。

## 3. 总体路线

执行顺序固定为：

```text
Phase A: semantic end-to-end GCL
  -> 先证明 trace -> graph -> embedding -> M0 selector 语义通路能闭合

Phase B: scalable graph embedding path
  -> 再处理 trace scope、per-warp graph、graph size audit、hierarchical pooling

Phase C: compression / abstraction path
  -> 最后引入 instruction stream dedup、stream_weight、weighted pooling
```

Phase A/B/C 不是三个互相替代的方案，而是逐步增加难度的实现层次。

### 3.1 Phase A: Semantic End-to-End GCL

Phase A 的目标是先在 small controlled trace 上打通最小语义闭环：

```text
small controlled trace
  -> canonical graph
  -> tensorization
  -> minimal RGCN contrastive training
  -> kernel embedding table
  -> M0 selector
  -> cluster / representative anchor / evaluation artifacts
```

Phase A 不关心大规模 trace、不证明 embedding quality、不做 instruction stream compression。它只回答：

```text
GCL 的所有必要部件能否串起来，并生成 M0 可以消费的 embedding table？
```

详细设计见：

```text
docs/superpowers/specs/2026-06-01-a-line-gcl-m1-m2-phase-a-semantic-e2e-design.md
```

### 3.2 Phase B: Scalable Graph Embedding Path

Phase B 在 Phase A 闭环成立后，加入真实 trace 规模约束：

```text
trace scope
representative SM policy audit
selected warps / bounded instruction windows
per-warp graph construction
kernel graph union with warp_partitions
graph size audit
training eligibility
hierarchical readout
augmentation manifest
```

Phase B 的目标是让 M2 不再假设 full-kernel dynamic graph 一定可以直接训练。它仍然保持 Phase A 的语义通路，只是把输入规模、图结构和训练资格显式化。

详细设计见：

```text
docs/superpowers/specs/2026-06-01-a-line-gcl-m1-m2-phase-b-scalable-graph-design.md
```

### 3.3 Phase C: Compression / Abstraction Path

Phase C 是后续优化阶段，用来引入 Photon-inspired instruction stream compression：

```text
warp instruction stream
  -> stable stream hash
  -> unique stream representative graph
  -> stream_weight
  -> weighted pooling
  -> kernel embedding
```

Phase C 的目标是减少重复动态 instruction stream 对 graph size 和 training cost 的影响。它必须以 Phase A/B 的未压缩或轻压缩结果作为对照，不能直接把压缩后的结果当成 simulator accuracy claim。

详细设计见：

```text
docs/superpowers/specs/2026-06-01-a-line-gcl-m1-m2-phase-c-compression-abstraction-design.md
```

## 4. 跨阶段 Artifact 边界

M1 负责从 trace 生成 canonical graph artifacts：

```text
trace records
  -> normalized trace entries
  -> scoped trace
  -> per-warp graph records
  -> kernel graph artifact
  -> graph audit
```

M2 负责从 canonical graph 生成 kernel embedding：

```text
canonical graph artifact
  -> tensorization
  -> augmented graph views for training
  -> RGCN encoder
  -> node embeddings
  -> warp / stream embeddings
  -> kernel embedding
  -> M0-compatible embedding table
```

M0 只消费最终 embedding table：

```text
gcl_embedding_table_l1.json
  -> z-score normalization
  -> silhouette_k / deterministic_fixed_k
  -> deterministic K-Means
  -> representative anchors
  -> structural evaluation artifacts
```

M0 不关心 embedding 是 fixture embedding、controlled encoder embedding、RGCN embedding，还是 weighted stream embedding。差异必须通过 `representation_mode`、`source_graph_hash`、`encoder_manifest_hash` 和 `embedding_hash` 记录。

## 5. 阶段排序规则

实现时必须遵守以下排序：

1. 没有 Phase A 闭环，不进入 Phase B 的真实 trace 规模问题；
2. 没有 Phase B 的 graph size audit 和 training eligibility，不进入 Phase C 的 compression claim；
3. Phase C 的压缩结果必须和 Phase A/B 的非压缩路径做稳定性对照；
4. 任何阶段都不得跳过 M0 embedding table contract；
5. 任何阶段都不得静默截断 trace 或静默丢弃 oversized graph；
6. 任何阶段都不得用 augmented graph 覆盖 canonical graph；
7. selector embedding 必须来自 canonical non-augmented graph，不来自 contrastive projection head output。

## 6. 非目标

这份主 spec 不做：

- 重新定义 M0 selector；
- 替代 M1 trace graph construction spec；
- 替代 Phase A/B/C 的副 spec；
- 定义具体代码模块和测试步骤；
- 证明 RGCN embedding quality；
- 证明 GCL 比 PKA 更准；
- 证明 sampled simulation accuracy；
- 定义真实 NVBit deployment 权限和集群 orchestration；
- 要求第一版支持 full-GPU full-kernel trace；
- 要求第一版实现 instruction stream dedup。

## 7. 与现有 Spec 的关系

```text
GCL-M0 selector interface spec
  定义 embedding table -> selector -> anchors/evaluation

GCL-M1 trace graph construction spec
  定义 trace records -> canonical graph artifacts

本 spec
  定义 GCL-M1/M2 的阶段路线和 artifact 边界

Phase A/B/C 副 spec
  分别定义语义闭环、规模化图训练、压缩抽象的具体设计

未来 GCL-M2 implementation plan
  应把 Phase A 副 spec 优先转成可执行任务
```

因此，当前优先级是：

```text
Phase A semantic end-to-end GCL
  -> Phase B scalable graph embedding
  -> Phase C compression / abstraction
```
