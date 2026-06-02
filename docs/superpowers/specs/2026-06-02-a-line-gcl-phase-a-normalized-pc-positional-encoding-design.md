# A 线 GCL Phase A Normalized PC Positional Encoding Design Spec

日期：2026-06-02

## 1. 定位

这份 spec 定义 GCL Phase A 中 instruction node 的 `normalized_pc_positional_encoding` 如何生成。

它对应 GCL-Sampler 论文中这句话的工程解释：

```text
dense embedding of opcode token ID
+ positional encoding derived from normalized PC value
-> 64-dimensional instruction node feature
```

GCL-Sampler 论文没有规定 positional encoding 在 64 维 instruction feature 内部的固定 index range。因此 Phase A strict paper reproduction 不默认使用 `[16:24)` 这类固定 block。

这份 spec 不改变 GCL-Sampler 的整体流程，也不替代 Phase A spec。它只定义 normalized PC 如何产生，以及 positional encoding config 如何记录。

## 2. 背景问题

只使用 opcode dense embedding 会丢失位置信息。

例如两条指令都是：

```text
LDG
```

如果只看 opcode，它们会查到同一个 opcode embedding：

```text
embedding(LDG)
```

但它们可能位于 kernel code 的不同区域：

```text
PC = 0x100: LDG R4, [R2]
PC = 0x8f0: LDG R20, [R18]
```

这两条 `LDG` 的上下文可能不同：

- 前者可能位于 kernel 初始化阶段；
- 后者可能位于循环体或 epilogue；
- 它们连接到的 data-flow / control-flow pattern 也可能不同。

因此，instruction node feature 需要同时包含：

```text
opcode identity
code position signal
```

GCL-Sampler 论文使用的是：

```text
dense embedding of opcode token ID
+ positional encoding derived from normalized PC value
```

Phase A strict paper reproduction 不固定该 encoding 在 64 维 instruction feature 内部的 index range。具体 combine 方式必须通过 manifest 记录。

## 3. PC 与 Normalized PC

`pc` 表示 program counter，也就是 instruction 在 kernel 机器码中的地址或偏移。

原始 PC 不能直接作为模型输入，原因是：

- 不同 kernel 的 PC 地址范围不同；
- absolute PC 可能受到编译、加载地址、trace 工具输出格式影响；
- 直接输入大整数会制造没有意义的数值尺度。

因此，Phase A 默认使用 per-kernel normalized PC：

```text
normalized_pc = (pc - min_pc) / max(max_pc - min_pc, 1)
```

其中：

```text
min_pc = 当前 kernel invocation trace 中最小 PC
max_pc = 当前 kernel invocation trace 中最大 PC
```

结果范围：

```text
0.0 <= normalized_pc <= 1.0
```

如果 `max_pc == min_pc`，则：

```text
normalized_pc = 0.0
```

并在 manifest 中记录：

```text
pc_range_degenerate = true
```

## 4. 默认 Encoding 方法

GCL-Sampler 论文没有明确给出 positional encoding 的具体公式。Phase A 可以使用 deterministic sinusoidal positional encoding 作为工程默认，但必须把它标记为 implementation choice，而不是论文原文 claim。

输入：

```text
normalized_pc in [0, 1]
encoding_dim = implementation-defined
```

若实现选择 8 维 sinusoidal encoding，频率可以定义为：

```text
frequency_j = 2^j * 2*pi
j = 0, 1, 2, 3
```

输出：

```text
pe[0] = sin(normalized_pc * frequency_0)
pe[1] = cos(normalized_pc * frequency_0)
pe[2] = sin(normalized_pc * frequency_1)
pe[3] = cos(normalized_pc * frequency_1)
pe[4] = sin(normalized_pc * frequency_2)
pe[5] = cos(normalized_pc * frequency_2)
pe[6] = sin(normalized_pc * frequency_3)
pe[7] = cos(normalized_pc * frequency_3)
```

默认 block 属性：

```text
block_name = normalized_pc_positional_encoding
block_kind = fixed_numeric
trainable = false
encoding_method = sinusoidal_normalized_pc_v1
paper_defined_formula = false
```

## 5. 为什么使用 Sinusoidal Encoding

直接输入一个 scalar：

```text
normalized_pc = 0.42
```

也可以提供位置信息，但表达能力较弱。Sinusoidal encoding 把一个位置映射到多个频率尺度：

```text
low frequency:
  表示粗粒度前部 / 中部 / 后部

high frequency:
  表示更细的位置差异
```

这样模型可以更容易学习：

- 两条 instruction 是否处在相近代码区域；
- 某些 opcode pattern 是否集中出现在 kernel 前段、循环体或后段；
- 相同 opcode 在不同 PC 区域是否承担不同角色。

它是 deterministic fixed numeric feature，不是 learned embedding。训练会更新后续 RGCN 权重，但不会更新 positional encoding 本身。

## 6. PC 与 Trace Index 的区别

`pc` 和 `trace_index` 不表达同一件事。

`pc`：

```text
static code position
```

表示 instruction 属于 kernel code 的哪个位置。

`trace_index`：

```text
dynamic execution order
```

表示 instruction 在这次运行 trace 中第几个发生。

例如 loop 中同一条静态 instruction 可能执行多次：

```text
pc = 0x200
trace_index = 10
trace_index = 80
trace_index = 150
```

如果使用 normalized PC，这些动态实例共享同一个静态 code position signal。
如果使用 trace index，它们会获得不同的 dynamic time signal。

GCL-Sampler 论文描述的是基于 normalized PC 的 positional encoding，因此 Phase A 默认不使用 trace index encoding。

`trace_index` 可以作为后续 ablation：

```text
trace_index_positional_encoding
```

但不得替代默认的 normalized PC positional encoding，除非 manifest 中明确记录 `position_source`。

## 7. 输入字段

生成 positional encoding 至少需要：

```text
kernel_invocation_id
node_id
node_type = instruction
pc
min_pc
max_pc
```

可选字段：

```text
trace_index
sequence_index
basic_block_id
source_entry_hash
```

如果 `pc` 缺失，Phase A 不得静默生成随机 position feature。必须使用明确 fallback：

```text
position_source = missing_pc_zero_fill
normalized_pc = 0.0
pc_missing = true
```

并在 `missing_value_policy` 中记录。

## 8. Manifest 要求

Tensorization 必须在 `node_feature_schema` 或 tensor manifest 中记录：

```text
positional_encoding_config:
  block_name
  index_range
  position_source
  normalization_scope
  encoding_method
  encoding_dim
  frequency_schedule
  min_pc
  max_pc
  pc_range_degenerate
  missing_value_policy
```

Phase A 默认：

```text
block_name = normalized_pc_positional_encoding
index_range = not_fixed_by_paper
position_source = pc
normalization_scope = per_kernel_invocation
encoding_method = sinusoidal_normalized_pc_v1
encoding_dim = implementation_defined
frequency_schedule = powers_of_two_times_2pi
```

这些字段必须参与：

```text
tensor_hash
encoder_manifest_hash
```

否则后续无法判断两个 embedding 是否使用了同一套 position encoding。

## 9. 可选模式

第一版 strict paper reproduction 只要求：

```text
position_source = normalized_pc
encoding_method is recorded
paper_defined_formula = false unless a paper-defined formula is implemented
```

允许的 implementation choices：

```text
sinusoidal_normalized_pc_v1
normalized_pc_scalar_plus_padding
learned_pc_bucket_embedding
trace_index_sinusoidal_encoding
no_positional_encoding
```

若使用，必须记录：

```text
position_source
encoding_method
ablation_reason
```

## 10. 与 64 维 Instruction Feature 的关系

Instruction node 的 Phase A layout 是：

```text
dense embedding(opcode token ID)
  + positional encoding(normalized PC)
  -> 64-dimensional instruction node feature
```

其中：

```text
opcode_token_embedding:
  learned embedding，训练中更新

normalized_pc_positional_encoding:
  deterministic input feature，训练中不更新；
  在 64 维 feature 中的 index range 不由论文固定

RGCN weights:
  训练中更新，并学习如何使用 position signal
```

因此，positional encoding 自身不是“学出来的位置向量”。它是确定性输入特征。模型学的是如何结合：

```text
opcode identity
position signal
graph neighborhood
edge relation type
```

来生成最终 kernel embedding。

## 11. 成功标准

Phase A positional encoding 完成标准：

1. 每个 instruction node 都能得到由 normalized PC 派生的 positional encoding；
2. normalized PC 使用 per-kernel invocation 的 `min_pc` / `max_pc`；
3. `max_pc == min_pc` 时结果可复现，并记录 `pc_range_degenerate = true`；
4. PC 缺失时不得静默生成随机值；
5. `node_feature_schema` 记录 encoding method、position source 和 trainable 状态；
6. `node_feature_schema` 不把固定 index range 伪装成论文要求；
7. `tensor_hash` 和 `encoder_manifest_hash` 能反映 positional encoding config 的变化；
8. `sinusoidal_normalized_pc_v1` 被标记为 implementation choice；
9. trace index encoding 只作为 ablation，不作为默认模式；
10. 该 encoding 与 Phase A `feature_width = 64` 的 instruction node schema 一致。
