# A 线 GCL-Sampler RGCN Contrastive Learning Explainer Spec

日期：2026-06-02

## 1. 定位

这份 spec 是解释型文档，用来说明 GCL-Sampler 中从 trace graph 到 kernel embedding 的完整学习链路。

它专门回答几个问题：

- trace 为什么要先变成 graph；
- canonical graph artifact 如何进入 tensorization；
- `feature_width = 64` 为什么不要求每一维都有固定人工语义；
- RGCN 如何利用 control/data flow 传播上下文；
- contrastive learning 如何训练这些 learned embeddings；
- projection head 为什么训练时使用、采样时不用；
- K-Means 和 silhouette 如何把 kernel embedding 变成 representative simulation points；
- 这条链路如何对应我们拆分出的 M1/M2/M0。

这份 spec 不替代 Phase A/B/C 设计文档。它是帮助理解 GCL 内部学习机制的说明文档。

参考论文：

```text
GCL-Sampler: Graph Contrastive Learning based Sampler for GPU Architecture Simulation
https://arxiv.org/pdf/2603.00551
```

## 2. 总体路径

GCL-Sampler 的核心路径可以概括为：

```text
kernel trace
  -> heterogeneous relation graph
  -> node feature initialization
  -> graph augmentation
  -> RGCN encoder
  -> node embeddings
  -> warp embedding
  -> kernel embedding
  -> K-Means clustering
  -> representative kernel selection
```

对应到我们的 M0/M1/M2 拆分：

```text
M1:
  trace records
    -> canonical graph artifact

M2:
  canonical graph artifact
    -> tensorization
    -> RGCN contrastive learning
    -> M0-compatible embedding table

M0:
  embedding table
    -> silhouette_k
    -> deterministic K-Means
    -> representative anchors
```

论文主要描述方法；我们的 spec 额外要求 artifact 可审计、可复现、可 hash。

## 3. Trace 为什么要变成 Graph

原始 trace 是顺序日志：

```text
instruction_0
instruction_1
instruction_2
...
```

这种形式能表示执行顺序，但不方便表达数据依赖。例如：

```text
I1: R4 = LDG [R2]
I2: R5 = FADD R4, R6
```

这里 `I2` 使用了 `I1` 产生的 `R4`。如果只看 instruction list，这个依赖需要额外解析；如果变成 graph，就可以显式表示：

```text
R2 -> I1 -> R4 -> I2 -> R5
```

GCL-Sampler 使用 heterogeneous relation graph 来同时表达：

```text
control_flow:
  谁在执行顺序上接着谁

data_flow:
  哪个值被哪个 instruction 使用或产生
```

因此，graph 的目的不是增加形式复杂度，而是把 kernel 行为中的结构关系显式暴露给 RGCN。

## 4. Graph Node 和 Edge

GCL-Sampler 的 graph 中主要有三类 node。

`Instruction Node`：

- 表示一条 SASS instruction；
- 例如 `LDG`、`STG`、`IMAD`、`FADD`、`BRA`；
- 初始 feature 来自 opcode token 和 normalized PC。

`Variable Node`：

- 表示执行中的动态值；
- 例如 register version、memory value、input variable；
- 每次 write 产生新的 variable node；
- 后续 read 连接到最近可见的 producer version。

`Pseudo Node`：

- 表示单条 instruction 内部需要显式建模的语义操作；
- 例如 `mem_ref`；
- 作用是让 memory reference 等中间语义也能进入 graph learning。

主要 edge 类型：

```text
control_flow edge:
  consecutive executed instructions

data_flow edge:
  source value / source node -> consumer or result node
```

RGCN 之所以适合这里，是因为它可以为不同 relation type 使用不同的 message passing 参数。

## 5. Node Feature Initialization

Graph 中的 node 不能直接用字符串喂给模型。Tensorization 需要把它们变成数字向量。

GCL-Sampler 中所有 node 最终统一到：

```text
feature_width = 64
node_features.shape = [node_count, 64]
```

论文中的初始化方式可以概括为：

```text
instruction node:
  dense embedding(opcode token ID)
  + positional encoding(normalized PC)
  -> 64-dimensional vector

variable node:
  32-dimensional token ID embedding
  + 8-dimensional dynamic value statistics
  = 40-dimensional vector
  -> zero-pad to 64

pseudo node:
  16-dimensional token ID embedding
  -> zero-pad to 64
```

Variable node 的 8 维 dynamic value statistics 为：

```text
mean
standard_deviation
median
minimum
maximum
percentile_25
percentile_75
skewness
```

这些 feature 有两类。

第一类是 fixed numeric features：

```text
normalized PC
mean
standard deviation
median
minimum
maximum
active mask density
fan-in / fan-out summary
```

这些维度有明确人工含义。

第二类是 learned embedding features：

```text
opcode token embedding
variable token embedding
pseudo token embedding
instruction class embedding
operand shape embedding
```

这些 block 内部的单个维度通常没有固定人工语义。

## 6. 为什么没有固定人工语义也能训练

关键区别是：

```text
block 有语义；
block 内部的 learned embedding 维度不一定有人工语义。
```

例如：

```text
opcode token embedding block
```

这个 block 的来源很明确：它来自 opcode token。但 block 内部的每一维不应该解释成：

```text
第 0 维 = memory intensity
第 1 维 = branch behavior
第 2 维 = compute intensity
```

learned embedding 的作用不是提供人工可读的逐维解释，而是提供可训练的表示空间。

如果直接把 opcode 编码成整数：

```text
LDG = 1
STG = 2
IMAD = 3
```

模型会看到错误的数值关系，例如 `IMAD` 比 `LDG` 大。Opcode 是类别，不是连续数值。

因此更合理的做法是：

```text
LDG  -> learned vector
STG  -> learned vector
IMAD -> learned vector
```

训练会调整这些向量，使它们在最终任务中有用。例如 memory 相关 opcode 可能在某些方向上更接近，arithmetic 相关 opcode 可能在另一些方向上更接近。

所以，“没有固定人工语义”不是说输入不可控，而是说：

```text
我们知道某个 block 来自什么 trace 信息；
但 block 内部如何编码由训练自动学习。
```

这也是 GCL 试图取代 hand-crafted feature 的核心原因：人工只定义 graph schema、token source、relation type 和训练目标；具体行为表示由模型学习。

## 7. Graph Augmentation

Contrastive learning 需要为同一个 graph 构造两个不同 view。

GCL-Sampler 使用的 augmentation pool 包括：

```text
node dropping:
  randomly remove 15% nodes and incident edges

edge dropping:
  randomly remove 15% edges

feature noise injection:
  add Gaussian noise with sigma = 0.01 to node features
```

对同一个 kernel graph：

```text
canonical graph G
  -> augmented view G_a
  -> augmented view G_b
```

`G_a` 和 `G_b` 来自同一个 kernel，因此构成 positive pair。Batch 中其他 kernel 的 views 构成 negative pairs。

我们工程实现中必须保留一个边界：

```text
augmentation 只属于 training；
canonical graph artifact 不得被 augmentation 覆盖；
selector embedding 必须来自 canonical non-augmented graph。
```

## 8. Contrastive Learning 如何推动表示变化

GCL-Sampler 不需要人工标签，例如：

```text
kernel A = memory-bound
kernel B = compute-bound
```

它使用 self-supervised contrastive learning。

训练目标是：

```text
same kernel, different augmented views:
  embeddings should be close

different kernels:
  embeddings should be far apart
```

对于 batch 中的每个 kernel：

```text
view A embedding: z'_a
view B embedding: z'_b
```

同一个 kernel 的 `(z'_a, z'_b)` 是 positive pair；不同 kernel 之间是 negative pairs。

InfoNCE loss 会推动：

```text
positive pair cosine similarity increase
negative pair cosine similarity decrease
```

反向传播会更新：

```text
opcode embedding table
variable embedding table
pseudo embedding table
RGCN weights
projection head weights
```

因此，虽然 learned embedding 的单维没有人工语义，但它们会被训练目标塑造成对区分 kernel graph 有用的表示。

## 9. RGCN Message Passing

RGCN 的作用是让 node embedding 吸收邻居上下文，并区分不同 relation type。

初始状态下，一个 instruction node 可能只知道：

```text
I = LDG
PC = normalized_pc
```

但在 graph 中，它还连接到：

```text
source variable
destination variable
previous instruction
next instruction
memory reference pseudo node
```

RGCN 每一层大致做：

```text
new node embedding =
  transformed self embedding
  + messages from control_flow neighbors
  + messages from data_flow neighbors
  + messages from other relation-specific neighbors
```

不同 relation type 使用不同参数，因此：

```text
control_flow message != data_flow message
```

经过多层后，node 不再只是局部 token，而是带有上下文的行为表示。

例如：

```text
R2 -> LDG -> R4 -> FADD -> R5
```

经过 RGCN 后，`FADD` 的 embedding 可以融合：

- 它消费了由 `LDG` 产生的 `R4`；
- 它位于某段 load 后的 arithmetic pattern；
- 它的结果后续被其他 instruction 使用；
- 它属于某个 warp 的执行上下文。

GCL-Sampler 使用三层 RGCN：

```text
input dimension = 64
hidden dimension = 128
output dimension = 256
```

每层使用 basis decomposition 控制参数量，并配合 LayerNorm、ReLU、Dropout。最后一层不使用 Dropout，以保留完整表示。

## 10. Readout：从 Node 到 Kernel

RGCN 输出的是 node-level embeddings：

```text
node_0 -> 256-dimensional embedding
node_1 -> 256-dimensional embedding
node_2 -> 256-dimensional embedding
...
```

但 selector 需要的是每个 kernel invocation 一个 embedding。

GCL-Sampler 的 readout 路径是：

```text
node embeddings
  -> mean pooling within warp
  -> warp embeddings
  -> average pooling across warps
  -> kernel embedding
```

也就是：

```text
node -> warp -> kernel
```

最终得到：

```text
kernel_embedding z_k in R^256
```

这个 256 维向量是 kernel 的 behavioral signature。它不是人工统计项列表，而是 RGCN 从 graph structure、node features 和 relation types 中学习得到的整体行为表示。

## 11. Projection Head

训练时，GCL-Sampler 不直接用 256 维 kernel embedding 计算 InfoNCE，而是接一个 projection head：

```text
kernel embedding z_k, 256-dimensional
  -> MLP hidden layer, 128-dimensional
  -> projection output z'_k, 64-dimensional
```

InfoNCE loss 使用：

```text
z'_k
```

采样时使用：

```text
z_k
```

也就是 projection head 之前的 256 维 kernel embedding。

原因是：

```text
RGCN encoder 学通用 kernel representation；
projection head 专门适配 contrastive training loss；
下游 clustering 使用 encoder readout embedding。
```

因此，我们的 M2 embedding export 必须使用 canonical graph 经过 encoder readout 得到的 256 维 embedding，不应使用 projection head output。

## 12. K-Means 和 Silhouette

训练结束后，每个 kernel invocation 都有一个 256 维 embedding：

```text
kernel_0 -> e_0
kernel_1 -> e_1
kernel_2 -> e_2
...
```

GCL-Sampler 使用 K-Means 对这些 embedding 聚类：

```text
embedding table
  -> K-Means
  -> clusters
```

K-Means 需要指定 cluster count：

```text
K = ?
```

GCL-Sampler 使用 silhouette coefficient 选择 K。它尝试多个候选 K，并评估：

```text
cluster 内部是否紧凑；
cluster 之间是否分离。
```

如果多个 K 的 silhouette score 接近，论文倾向于选择更小的 K，因为更小 K 意味着更少 representative kernels，也就是更高 sampling speedup。

确定 K 后，每个 cluster 选择一个 representative kernel invocation。论文描述的策略是选择 cluster 中的第一个 kernel invocation。

在我们的 M0 中，这一段对应：

```text
embedding table
  -> z-score normalization
  -> silhouette_k
  -> deterministic K-Means
  -> representative anchors
```

## 13. 与 M0/M1/M2 的工程边界

这条解释链路在我们的实现中应被拆成清晰 artifact 边界。

M1 负责：

```text
trace records
  -> canonical graph artifact
```

关键产物：

```text
graph_id
kernel_invocation_id
nodes
edges
warp_partitions
graph_summary
graph_hash
```

M2 负责：

```text
canonical graph artifact
  -> tensorization
  -> augmentation for training
  -> RGCN contrastive learning
  -> canonical graph embedding export
```

关键产物：

```text
node_feature_schema
tensorizer_version
tensor_hash
augmentation_manifest_hash
training_config_hash
checkpoint_hash
encoder_manifest_hash
embedding_hash
```

M0 负责：

```text
M0-compatible embedding table
  -> cluster selection
  -> representative anchors
```

关键产物：

```text
cluster_manifest
silhouette_report
representative_anchor_table
structural_evaluation_artifacts
```

## 14. 实现时必须保持的语义边界

1. `canonical graph artifact` 是 M1 的正式输出，不得被 augmentation 覆盖；
2. `node_features.shape = [node_count, 64]` 是 tensorization 结果，不是 graph artifact 本身；
3. learned embedding block 内部单维不要求人工可解释，但 block source 必须可审计；
4. `node_feature_schema` 必须记录每个 block 的 index range、source fields 和 trainable 状态；
5. contrastive training 使用 augmented views；
6. embedding export 使用 canonical non-augmented graph；
7. training loss 使用 projection head output；
8. selector 使用 projection head 之前的 256 维 kernel embedding；
9. M0 不关心 embedding 如何训练，但必须能通过 manifest 回溯来源；
10. silhouette 选 K 是 selector 侧逻辑，不属于 RGCN training。

## 15. 非目标

这份 spec 不做：

- 定义 Phase A/B/C 的具体任务拆分；
- 替代 GCL-M1 trace graph construction spec；
- 替代 GCL-M2 implementation plan；
- 证明 GCL 比 PKA 更准；
- 证明 simulator accuracy；
- 要求 learned embedding 的每一维都有人工解释；
- 要求第一版实现 instruction stream compression；
- 要求第一版支持 full-GPU full-kernel trace。

## 16. 成功标准

读完这份 spec 后，应该能回答：

1. 为什么 trace 要先变成 graph；
2. graph 中 instruction、variable、pseudo node 分别代表什么；
3. 为什么 `feature_width = 64` 不等于 64 个手写统计项；
4. learned embedding block 为什么可以没有逐维人工语义；
5. contrastive learning 如何通过 positive / negative pairs 更新 embedding；
6. RGCN 如何沿 control/data flow 传播上下文；
7. node embeddings 如何通过 readout 变成 kernel embedding；
8. projection head 为什么训练时使用、采样时不用；
9. K-Means 和 silhouette 如何选 representative kernels；
10. 这条链路如何落到我们的 M1/M2/M0 artifact 边界。
