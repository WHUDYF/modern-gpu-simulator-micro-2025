# A 线 GCL RGCN Message Passing and Readout Explainer Spec

日期：2026-06-02

## 1. 定位

这份 spec 解释 GCL-Sampler 中 RGCN 如何把 node feature 转换成 kernel embedding。

它覆盖：

- node embedding 初始只包含局部信息；
- RGCN message passing 如何沿 control-flow / data-flow 传播上下文；
- relation type 为什么需要不同参数；
- 三层 RGCN 的信息传播范围；
- 64 -> 128 -> 256 的维度变化；
- basis decomposition、LayerNorm、ReLU、Dropout 的作用；
- readout / pooling 如何从 node embedding 得到 kernel embedding；
- projection head 为什么训练时使用、采样时不用；
- 最终 M2 如何导出 M0-compatible kernel embedding table。

这份 spec 是解释型文档，不替代 Phase A/B/C implementation spec。

## 2. 输入与输出

RGCN 的输入来自 M2 tensorization：

```text
node_features
edge_index
edge_type
warp_partitions
graph_batch_metadata
```

其中：

```text
node_features.shape = [node_count, 64]
```

RGCN encoder 输出 node-level embeddings：

```text
node_embeddings.shape = [node_count, 256]
```

经过 readout 后得到 kernel-level embedding：

```text
kernel_embedding.shape = [256]
```

训练时还会经过 projection head：

```text
kernel_embedding, 256
  -> projection head
  -> projection_output, 64
```

最终 selector 使用的是：

```text
kernel_embedding, 256
```

不是 projection output。

## 3. 初始 Node Embedding 只包含局部信息

在进入 RGCN 前，每个 node 的 feature 只表示该 node 自己的初始信息。

例如 instruction node：

```text
I = LDG
PC = normalized_pc
```

它一开始只知道：

```text
我是一个 LDG instruction；
我位于 kernel code 的某个 normalized PC 位置。
```

它还不知道：

```text
我读取了哪个 source variable；
我产生了哪个 destination variable；
我前后执行了哪些 instruction；
我的结果后来被谁使用；
我是否处在某段 memory-load-to-arithmetic pattern 中。
```

这些上下文信息在 graph edges 和邻居 nodes 中。

RGCN 的作用就是把这些邻居上下文逐层传播进 node embedding。

## 4. Message Passing 的直观含义

Message passing 可以理解为：

```text
每个 node 从邻居 node 接收消息，并用这些消息更新自己的 embedding。
```

对于一个简单 data-flow：

```text
R2 -> LDG -> R4 -> FADD -> R5
```

`FADD` 的邻居可能包括：

```text
R4:
  source variable

R5:
  destination variable

previous instruction:
  control-flow predecessor

next instruction:
  control-flow successor
```

一层 RGCN 后，`FADD` 不再只表示：

```text
我是 FADD
```

而是融合了：

```text
我是 FADD；
我消费了 R4；
我产生了 R5；
我处在某个 control-flow context 中。
```

## 5. Relation Type 的作用

GCL graph 中至少有两类核心 relation：

```text
control_flow
data_flow
```

它们语义不同。

`control_flow` 表示：

```text
谁在执行顺序上接着谁
```

`data_flow` 表示：

```text
哪个值流向哪个 instruction 或 result node
```

如果普通 GCN 不区分 edge type，就会把这两类信息混在一起。

RGCN 的 `Relational` 含义是：

```text
不同 relation type 使用不同 message transformation。
```

直观公式：

```text
new_node_embedding =
  self_transform(current_node_embedding)
  + control_flow_messages
  + data_flow_messages
  + other_relation_specific_messages
```

因此：

```text
control_flow message != data_flow message
```

这让模型可以分别学习：

```text
执行顺序上下文如何影响 node 表示；
数据依赖上下文如何影响 node 表示。
```

## 6. 多层 RGCN 的传播范围

一层 RGCN 主要聚合一跳邻居。

对于：

```text
R2 -> LDG -> R4 -> FADD -> R5
```

对 `FADD` 来说：

第一层后，它能直接融合：

```text
R4
R5
control-flow predecessor / successor
```

第二层后，它可以间接感知：

```text
LDG
```

因为路径是：

```text
LDG -> R4 -> FADD
```

第三层后，它可以进一步感知：

```text
R2
```

因为路径是：

```text
R2 -> LDG -> R4 -> FADD
```

所以，三层 RGCN 的意义不是简单加深模型，而是让 node embedding 可以融合更远的局部执行上下文。

经过三层后，`FADD` 的 embedding 可以表示：

- 它是一个 arithmetic instruction；
- 它消费的 `R4` 来自上游 `LDG`；
- 该 `LDG` 的地址或 source 与 `R2` 有关；
- 它处于 load 后的 arithmetic pattern；
- 它的 result 后续可能继续流向其他 instruction。

## 7. RGCN 维度变化

GCL-Sampler 使用三层 RGCN：

```text
input dimension = 64
hidden dimension = 128
output dimension = 256
```

含义是：

```text
initial node feature:
  64-dimensional

hidden node embedding:
  128-dimensional

final node embedding:
  256-dimensional
```

最终每个 node 都有一个 256 维上下文表示：

```text
node_i -> h_i in R^256
```

这些 node embeddings 仍然是 node-level，不是 kernel-level。

## 8. Basis Decomposition

RGCN 通常需要为每种 relation type 学一组参数。

如果 relation type 较多，直接为每种 relation 存完整权重矩阵会增加参数量：

```text
control_flow -> W_control
data_flow -> W_data
memory_source -> W_memory
predicate_source -> W_predicate
...
```

Basis decomposition 的思想是：

```text
不用为每个 relation 存完整独立矩阵；
用少量 basis matrices 线性组合出 relation-specific weights。
```

直观形式：

```text
W_relation =
  a_1 * B_1
  + a_2 * B_2
  + ...
  + a_k * B_k
```

其中：

```text
B_i:
  shared basis matrix

a_i:
  relation-specific coefficient
```

作用：

- 降低参数量；
- 降低过拟合风险；
- 让多个 relation type 共享一部分结构信息；
- 保持 relation-specific message passing 能力。

## 9. LayerNorm / ReLU / Dropout

GCL-Sampler 在 RGCN layer 后使用：

```text
LayerNorm
ReLU
Dropout
```

`LayerNorm`：

```text
稳定每层输出的数值分布，让训练更平稳。
```

`ReLU`：

```text
引入非线性，使模型不只是线性组合邻居信息。
```

`Dropout`：

```text
训练时随机置零一部分 hidden features，降低过拟合。
```

最后一层 RGCN 不使用 Dropout。原因是最后一层输出要作为完整 node representation，再进入 pooling。如果最终表示被随机丢掉一部分，会损害导出的 node / graph representation 稳定性。

## 10. 为什么不能直接用 Node Embedding 做 Selector

RGCN 输出的是：

```text
node_0 -> 256-dimensional embedding
node_1 -> 256-dimensional embedding
node_2 -> 256-dimensional embedding
...
```

但 GCL-Sampler 的目标是选择 representative kernel invocation。

M0 selector 需要的是：

```text
kernel_0 -> embedding
kernel_1 -> embedding
kernel_2 -> embedding
...
```

因此，M2 必须把 node-level embeddings 汇总为 kernel-level embedding。

这一步叫：

```text
readout / pooling
```

## 11. Node To Warp Pooling

GCL-Sampler 先把同一个 warp 内的 node embeddings 做 mean pooling：

```text
warp_embedding =
  mean(node_embeddings in this warp)
```

例如：

```text
warp_0 nodes:
  h_R2
  h_LDG
  h_R4
  h_FADD
  h_R5
```

则：

```text
w_0 = mean(h_R2, h_LDG, h_R4, h_FADD, h_R5)
```

结果：

```text
w_0 in R^256
```

这一步把 node-level behavior 汇总成 warp-level behavior。

## 12. Warp To Kernel Pooling

一个 kernel invocation 可能包含多个 warp：

```text
warp_0
warp_1
warp_2
...
```

每个 warp 先得到一个 256 维 embedding：

```text
w_0
w_1
w_2
...
```

然后做 average pooling：

```text
kernel_embedding =
  average(w_0, w_1, w_2, ...)
```

结果：

```text
z_k in R^256
```

这就是 kernel invocation 的 behavioral signature。

## 13. 为什么使用 Node -> Warp -> Kernel

GPU kernel 的执行天然有 warp 结构。

如果直接把所有 nodes 平均：

```text
all node embeddings -> kernel embedding
```

会过早抹掉 warp-level 层次。

GCL-Sampler 的路径：

```text
node -> warp -> kernel
```

至少保留了两层结构：

```text
node-level local execution behavior
warp-level execution behavior
kernel-level aggregate behavior
```

这也是我们在 M1 canonical graph artifact 中保留 `warp_partitions` 的原因。

## 14. Projection Head

训练时，GCL-Sampler 不直接用 256 维 kernel embedding 计算 InfoNCE。

它使用 projection head：

```text
kernel_embedding z_k, 256
  -> MLP hidden layer, 128
  -> projection_output z'_k, 64
```

InfoNCE loss 使用：

```text
z'_k
```

Projection head 是训练用模块。它把 encoder readout embedding 映射到 contrastive training space。

## 15. 为什么训练用 Projection Output，采样用 Kernel Embedding

Contrastive learning 中常见做法是：

```text
encoder representation:
  保留更通用的信息

projection output:
  专门适配 contrastive loss
```

Projection head 可能会丢掉一些对下游 clustering 有用、但对 InfoNCE 优化不直接关键的信息。

因此 GCL-Sampler 训练时使用：

```text
projection_output z'_k, 64
```

采样时使用：

```text
kernel_embedding z_k, 256
```

这也是 M2 embedding export 的边界：

```text
selector embedding must come from encoder readout,
not from projection head output.
```

## 16. Training Update Path

InfoNCE loss 的梯度路径是：

```text
InfoNCE loss
  -> projection output
  -> projection head weights
  -> kernel embedding
  -> pooling
  -> node embeddings
  -> RGCN weights
  -> input node feature embedding tables
```

训练会更新：

```text
opcode dense embedding
variable token embedding
pseudo token embedding
RGCN relation weights
basis decomposition coefficients
projection head weights
```

训练不会更新：

```text
canonical graph artifact
edge_index
edge_type
normalized PC raw value
dynamic value statistics raw value
warp_partitions
```

这些是输入事实或派生数值，不是模型参数。

## 17. Embedding Export

训练完成后，M2 不再使用 augmented graph 生成 selector embedding。

正确导出路径：

```text
canonical graph
  -> tensorization
  -> trained RGCN encoder
  -> node embeddings
  -> node-to-warp pooling
  -> warp-to-kernel pooling
  -> 256-dimensional kernel embedding
  -> M0-compatible embedding table
```

导出 row 至少包含：

```text
record_id
kernel_invocation_id
representation_mode
embedding_dim = 256
embedding
source_graph_hash
encoder_manifest_hash
embedding_hash
weight_input
```

M0 selector 后续使用这个 embedding table：

```text
embedding table
  -> normalization
  -> silhouette_k
  -> K-Means
  -> representative anchors
```

## 18. 必须保持的边界

1. RGCN 输入是 tensorized canonical graph 或 training augmented views；
2. Training loss 使用 augmented views 和 projection output；
3. Canonical graph artifact 不得被 RGCN training 覆盖；
4. Selector embedding 必须来自 canonical non-augmented graph；
5. Selector embedding 必须使用 projection head 之前的 256 维 kernel embedding；
6. `warp_partitions` 必须保留，否则无法执行 node -> warp -> kernel readout；
7. RGCN learned parameters 和 canonical graph artifacts 必须通过 manifest 区分；
8. `encoder_manifest_hash` 必须覆盖 RGCN config、checkpoint、tensorizer config 和 projection head config；
9. `embedding_hash` 必须覆盖导出的 kernel embedding 内容；
10. M0 不关心 RGCN 内部训练细节，但必须能通过 manifest 回溯 embedding 来源。

## 19. 成功标准

读完这份 spec 后，应该能回答：

1. 为什么 node 初始 feature 只有局部信息；
2. RGCN message passing 如何吸收邻居上下文；
3. 为什么 control-flow 和 data-flow 需要不同 relation parameters；
4. 三层 RGCN 如何扩大 node 的上下文范围；
5. 64、128、256 三个维度分别代表什么；
6. basis decomposition 为什么能减少 relation-specific 参数量；
7. LayerNorm、ReLU、Dropout 分别在训练中起什么作用；
8. 为什么 selector 不能直接使用 node embeddings；
9. node -> warp -> kernel readout 如何生成 256 维 kernel embedding；
10. projection head 为什么训练时用、采样时不用；
11. contrastive loss 会更新哪些参数；
12. M2 应如何从 canonical graph 导出 M0-compatible embedding table。
