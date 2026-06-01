# A 线 GCL-M1/M2 Phase C Compression Abstraction Design Spec

日期：2026-06-01

## 1. 定位

Phase C 是 GCL-M1/M2 的后续优化阶段。它引入 Photon-inspired instruction stream compression / abstraction，用来减少重复动态指令流对 graph size 和 training cost 的影响。

Phase C 不替代 Phase A/B：

```text
Phase A proves semantic GCL path
Phase B makes graph embedding scalable and auditable
Phase C compresses repeated instruction streams after A/B are available
```

Phase C 的核心思想是：

```text
不要让 GCL 反复看完全相同或高度重复的动态 instruction stream；
让 GCL 看到 unique streams，并用 stream_weight 表示重复次数或权重。
```

## 2. 输入和输出

输入来自 Phase B：

```text
scoped trace
per-warp ordered trace entries
kernel graph artifact
warp_partitions
graph size audit
```

Phase C 新增派生产物：

```text
stream_manifest
unique_stream_graphs
stream_weight_table
weighted_pooling_manifest
compressed_embedding_table
```

输出仍然必须满足 M0-compatible embedding table：

```text
record_id
kernel_invocation_id
representation_mode
embedding_dim
embedding
source_graph_hash
encoder_manifest_hash
embedding_hash
weight_input
```

推荐：

```text
representation_mode = gcl_m2_weighted_stream_embedding
```

## 3. Instruction Stream Definition

第一版 stream 以 warp 为基本单位：

```text
warp instruction stream
  = ordered normalized instruction sequence inside one warp scope
```

用于 stream hash 的字段应尽量稳定：

```text
opcode
operand_shape
predicate_shape
memory_access_class
control_flow_marker
```

不建议第一版把以下动态值直接纳入默认 stream hash：

```text
raw register id
raw memory address
raw immediate value
cycle timestamp
```

原因是这些字段会让等价或近似等价的 instruction stream 被过度拆分。

如果确实需要保留动态值，应通过 `stream_hash_mode` 显式声明：

```text
semantic_shape_hash
semantic_shape_with_immediates_hash
exact_trace_hash
```

默认推荐：

```text
stream_hash_mode = semantic_shape_hash
```

## 4. Stream Hash and Dedup

M1 或 Phase C preprocessor 必须为每个 warp stream 生成 stable hash：

```text
stream_id
kernel_invocation_id
warp_id
stream_hash_mode
stream_hash
instruction_count
source_trace_hash
```

Dedup 过程：

```text
warp streams
  -> group by stream_hash
  -> choose representative stream
  -> build or reuse representative graph
  -> count duplicate streams
  -> assign stream_weight
```

`stream_weight` 第一版可以定义为：

```text
stream_weight = duplicate_stream_count
```

后续可以扩展为：

```text
instruction_count_weight
active_lane_weight
cta_weight
profile_weight
```

但第一版必须保持简单、可 replay。

## 5. Unique Stream Representative Graph

Phase C 不为每个重复 warp 都重新构建完整 graph。

它构建：

```text
unique stream representative graph
```

必要字段：

```text
stream_graph_id
stream_hash
stream_hash_mode
representative_kernel_invocation_id
representative_warp_id
nodes
edges
graph_summary
stream_graph_hash
stream_weight
duplicate_warp_refs
```

`duplicate_warp_refs` 至少记录：

```text
kernel_invocation_id
warp_id
cta_id
source_trace_hash
```

这样 Phase C 既能减少训练图数量，也能回溯每个 compressed stream 代表了哪些原始 warp。

## 6. Weighted Pooling

Phase C 的 pooling 路径：

```text
unique stream graph
  -> RGCN encoder
  -> stream embedding
  -> weighted stream pooling
  -> kernel embedding
```

Weighted pooling：

```text
kernel_embedding =
  sum(stream_embedding_i * stream_weight_i) / sum(stream_weight_i)
```

M2 必须记录：

```text
pooling_method = "weighted_average"
stream_count_before_dedup
unique_stream_count
total_stream_weight
weight_definition
weighted_pooling_hash
```

如果 `total_stream_weight <= 0`，必须报错，不得生成 fallback embedding。

## 7. 与 Phase A/B 的对照

Phase C 必须和未压缩或轻压缩路径做对照。

最小对照项：

```text
embedding cosine similarity
cluster assignment stability
representative anchor stability
graph size reduction ratio
training cost proxy
```

推荐记录：

```text
baseline_representation_mode
compressed_representation_mode
baseline_embedding_hash
compressed_embedding_hash
baseline_cluster_id
compressed_cluster_id
anchor_changed
node_count_before_compression
node_count_after_compression
edge_count_before_compression
edge_count_after_compression
unique_stream_ratio
```

Phase C 不能只报告 compression ratio。它必须说明压缩是否改变了 selector 看到的 embedding / cluster / anchor。

## 8. 非目标

Phase C 不做：

- 替代 Phase A 的 semantic end-to-end GCL；
- 替代 Phase B 的 graph size audit；
- 证明 simulator accuracy；
- 证明 Photon 方法本身；
- 保证 compressed embedding 一定更优；
- 默认压缩跨 kernel 的所有 stream；
- 默认把 raw memory address 作为 stream identity。

Phase C 也不允许：

- 没有 `stream_hash_mode` 就生成 stream hash；
- 没有 `stream_weight` 就做 weighted pooling；
- 压缩后丢失原始 warp refs；
- 把 compression result 直接当成 sampled simulation accuracy claim；
- 覆盖 Phase B canonical graph artifact。

## 9. 成功标准

Phase C 完成标准：

1. 能对 warp instruction stream 生成 stable hash；
2. 每个 stream hash 都记录 `stream_hash_mode`；
3. 能把重复 stream 合并为 unique stream representative；
4. 能记录 `stream_weight` 和 `duplicate_warp_refs`；
5. 能构建 unique stream representative graph；
6. 能使用 weighted pooling 生成 kernel-level embedding；
7. embedding table 满足 M0 输入契约；
8. 能和 Phase A/B 的未压缩或轻压缩结果做 embedding / cluster / anchor stability 对照；
9. 能报告 graph size reduction ratio 和 unique stream ratio；
10. 不把 Photon-inspired compression 结果直接当成 simulator accuracy claim。
