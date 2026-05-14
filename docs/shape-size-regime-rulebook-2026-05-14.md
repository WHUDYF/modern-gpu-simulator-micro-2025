# Shape / Size Regime Rulebook

日期：2026-05-14

## 1. 文档目的

本文档专门定义 B 线 `shape_size_regime` 的判定规则。

它承接现有 regime guardrail：

```text
squash / phase
  -> per-phase family
  -> Route Primitive
  -> Hardware Execution Template compatibility check
  -> shape / size regime
  -> resource signature
  -> weights / validation role
```

本文档回答：

```text
在同一个 phase + family + route primitive + hardware template 内，
两个 anchors / records 的 shape 和 size 什么时候可以认为相近，
什么时候必须拆开，
什么时候只能标 boundary / provisional？
```

`shape_size_regime` 不是 raw shape，也不是 grid/block 数值本身。它是把 raw shape、model shape、launch metadata 和 template-specific 规模因素归一化后得到的执行区间。

## 2. 非目标

本文档不做以下事情：

- 不按 kernel name / operator string 建 regime；
- 不用单个 M/N/K、seq_len、grid/block 数字直接生成 stable regime；
- 不替代 family 判断；
- 不替代 hardware template compatibility check；
- 不替代 resource signature 检查；
- 不给所有硬件和所有后端定死一组永久阈值。

本文档提供第一版规则。后续真实 trace、profiling、C-line 验证或 vendor backend 差异可以覆盖默认先验。

## 3. 核心定义

### 3.1 Raw Shape

`raw_shape` 是直接观测到的形状字段，例如：

```text
M, N, K
batch
sequence length
head dimension
channels
height / width
nnz
embedding dimension
grid / block
```

它是输入证据，不是最终 regime。

### 3.2 Shape Signature

`shape_signature` 是把 raw shape 转成 template-specific 执行描述后的结构化对象。

例如 dense GEMM：

```text
raw: M=1, N=4096, K=4096
signature:
  template = Dense Tiled Tensor-Core Compute
  role = small_batch_decode_gemm
  reuse = low_m_reuse
  tile_fit = tensor_core_friendly
  latency_profile = launch_or_tile_underfill_sensitive
```

### 3.3 Shape / Size Regime

`shape_size_regime` 是一组 shape signatures 的稳定区间。

例如：

```text
dense_prefill_projection_large_m
dense_decode_small_m
attention_prefill_long_seq
attention_decode_single_query
rowwise_norm_large_hidden
embedding_random_lookup_high_hotness
sparse_graph_powerlaw_high_nnz
```

它的作用是：

```text
既不把所有同 template 对象粗暴合并，
也不把每个具体 shape 都拆成单独 regime。
```

## 4. 输入字段

### 4.1 必需字段

每个 shape/size 判定至少需要：

| 字段 | 含义 |
|---|---|
| `phase_id` / `phase_context` | 已由 squash 提供的 phase |
| `family_id` | phase 内 family |
| `route_primitive` | 算法路径角色 |
| `hardware_template` | 已通过 Step 4 检查的执行骨架 |
| `raw_shape` | 原始 shape / size 字段 |
| `shape_source` | shape 来自 trace、graph parser、manual annotation、fixture 还是 prior |
| `shape_confidence` | low / medium / high |

### 4.2 推荐字段

如果存在，builder 应使用：

| 字段 | 用途 |
|---|---|
| `dtype` | 判断 tensor-core eligibility、bandwidth footprint |
| `layout` | 判断 coalescing、contiguous、stride、NHWC/NCHW 等 |
| `batch_size` | 区分 prefill / decode / throughput / latency 区间 |
| `sequence_length` | 区分 short/medium/long sequence |
| `hidden_dim` / `head_dim` | 判断 tile fit、alignment、register pressure |
| `grid_dim` / `block_dim` | 只能作为辅助 evidence，不能单独建 regime |
| `fused_ops` | 判断 shape 是否被 fused template 吸收 |
| `sparse_format` | CSR/COO/CSC/blocked sparse 等 |
| `nnz` / `degree_stats` | sparse / graph 的核心 size evidence |
| `table_size` / `lookup_count` / `hotness` | embedding lookup 的核心 size evidence |
| `message_size` / `world_size` | collective communication 的核心 size evidence |

### 4.3 缺失字段处理

缺失字段按下面规则处理：

| 情况 | 处理 |
|---|---|
| 缺 `hardware_template` | 不能进入 shape/size stable 判定 |
| 缺 template-specific 关键 shape 字段 | `blocked_missing_shape_signature` |
| 只有 kernel name / operator name | `blocked_no_shape_basis` |
| 只有 grid/block，没有 model shape 或 launch semantics | `provisional_grid_only_shape` |
| shape 来自人工 annotation 且无 provenance | `boundary_untrusted_shape_prior` |
| fixture shape 无 claim-bearing evidence | `fixture_non_claim_bearing` |

## 5. 通用判定流程

Shape/size rulebook 的执行顺序固定为：

```text
1. 读取 phase + family + route primitive + hardware template
2. 按 hardware template 选择 shape interpreter
3. 从 raw shape 生成 shape_signature
4. 将 shape_signature 映射到 candidate shape_size_regime
5. 做 merge / split / boundary 判断
6. 输出 shape_size_regime + reason + confidence
7. 交给 resource signature 做最后检查
```

重要约束：

```text
shape/size 只能在同 phase + family + route primitive + compatible template 内比较。
```

不同 template 的 shape 含义不同，不能直接横向比较。例如：

```text
GEMM 的 M/N/K
attention 的 seq_q/seq_kv/head_dim
embedding 的 table_size/hotness
sparse 的 nnz/degree_distribution
collective 的 message_size/world_size
```

这些不是同一种 shape 空间。

## 6. 通用 Merge 规则

两个 anchors 可以进入同一个 `shape_size_regime`，必须满足：

1. 已经处在同一 `phase_id` 或明确相容的 phase context；
2. 已经处在同一 `family_id`；
3. `route_primitive` 相同或显式相容；
4. `hardware_template` 相同或显式相容；
5. template-specific 关键维度落在同一执行区间；
6. tile utilization / memory reuse / access pattern 的 qualitative profile 相近；
7. shape source 可信，且 confidence 不强烈冲突；
8. 没有已知 resource signature 冲突。

推荐输出：

```text
shape_size_regime = <stable label>
shape_merge_reason = same template, same role, compatible size class, compatible reuse profile
shape_confidence = high | medium | low
```

## 7. 通用 Split 规则

出现以下情况时，shape/size 层默认拆分：

| Split 因素 | 说明 |
|---|---|
| small-batch vs large-batch | latency-sensitive 与 throughput-oriented 执行不同 |
| prefill vs decode | LLM 中 sequence behavior、KV cache 和 batch 形态不同 |
| short-seq vs long-seq | attention/reduction/memory behavior 改变 |
| projection-like vs expansion-like | dense shape 的 N/K 比例和 output size 不同 |
| tile-friendly vs tile-fringe | tensor core tile 对齐、尾块、underfill 明显不同 |
| cache-resident vs DRAM-streaming | working set 是否超出 cache 级别 |
| regular dense vs irregular sparse | access pattern 和 load balance 不同 |
| row-wise vs column-wise vs full-tensor reduction | reduction axis 改变同步和访存模式 |
| low hotness vs high hotness lookup | embedding pooling/reduction 行为不同 |
| small message vs large message collective | communication latency/bandwidth 主导项不同 |

推荐输出：

```text
shape_split_reason = same template, but size class or execution profile differs
```

## 8. Boundary 规则

出现以下情况时，不允许输出 stable shape/size regime：

| 情况 | boundary status |
|---|---|
| 关键维度缺失 | `blocked_missing_shape_signature` |
| shape 字段互相矛盾 | `boundary_conflicting_shape_evidence` |
| raw shape 可见但算法角色不明 | `boundary_unknown_shape_role` |
| 同一候选 regime 内混入多个 size class | `boundary_mixed_size_class` |
| shape 相近但 resource signature 强冲突 | `boundary_shape_resource_conflict` |
| 只有 fixture / synthetic shape | `fixture_non_claim_bearing` |
| 只有 network prior，无 measured/proxy evidence | `provisional_shape_prior_only` |

Boundary 对象可以进入 review/report，但不能作为 claim-bearing C-line stable validation target。

## 9. HET-1: Dense Tiled Tensor-Core Compute

### 9.1 关键字段

| 字段 | 作用 |
|---|---|
| `M` | row / token / batch-expanded dimension |
| `N` | output channel / hidden / projection dimension |
| `K` | reduction dimension |
| `batch_count` | batched GEMM / grouped GEMM |
| `dtype` | tensor-core eligibility |
| `layout` | contiguous / stride / transpose |
| `alignment` | tile / vectorization friendliness |
| `route_primitive` | projection、pairwise score、FFN、expert GEMM 等 |

### 9.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `dense_decode_small_m` | M 很小，常见于 decode 或 tiny batch |
| `dense_prefill_large_m` | M 由 batch * seq 展开，吞吐型 |
| `dense_projection_balanced` | projection-like M/N/K 较规整 |
| `dense_ffn_expansion` | N 相对 hidden 显著扩张 |
| `dense_ffn_contraction` | K 或 N 表现为 contraction path |
| `dense_pairwise_score` | QK score 类，M/N 由 query/key length 决定 |
| `dense_grouped_expert` | MoE grouped/batched expert GEMM |
| `dense_tile_fringe` | 关键维度不对齐或尾块明显 |

### 9.3 Merge 规则

Dense 对象可以 merge 的条件：

- 同 route primitive 或显式相容，例如同为 projection-like；
- M/N/K 所在 size class 相同；
- dtype 和 tensor-core eligibility 相容；
- alignment / tile fit 没有明显差异；
- batch/grouped 行为相同或相容；
- 不混合 decode small-M 和 prefill large-M。

### 9.4 Split 规则

必须拆分：

- `dense_decode_small_m` vs `dense_prefill_large_m`；
- `dense_projection_balanced` vs `dense_ffn_expansion`；
- `dense_pairwise_score` vs `dense_projection_balanced`；
- regular GEMM vs grouped expert GEMM；
- tensor-core-friendly vs tile-fringe / alignment-poor；
- batched GEMM vs single large GEMM，除非 profiling 证明行为相容。

## 10. HET-2: Convolution / Stencil Tiled Compute

### 10.1 关键字段

| 字段 | 作用 |
|---|---|
| `N` | batch |
| `C_in` / `C_out` | channel size |
| `H` / `W` | spatial size |
| `kernel_h` / `kernel_w` | convolution window |
| `stride` / `padding` / `dilation` | output shape 和 reuse |
| `groups` | regular / grouped / depthwise |
| `layout` | NCHW / NHWC |
| `algorithm_hint` | direct / implicit GEMM / Winograd / transform-based |

### 10.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `conv_1x1_projection` | 1x1 conv，接近 dense projection |
| `conv_spatial_3x3_regular` | 常规 spatial conv |
| `conv_depthwise_channelwise` | depthwise / high groups |
| `conv_grouped` | grouped conv |
| `conv_large_spatial` | H/W 大，activation reuse 明显 |
| `conv_small_spatial_large_channel` | spatial 小但 channel 大 |
| `conv_layout_sensitive` | layout/transpose 影响明显 |

### 10.3 Split 规则

必须拆分：

- 1x1 conv vs spatial conv；
- regular conv vs depthwise/grouped conv；
- large spatial vs small spatial；
- layout-sensitive path vs layout-stable path；
- direct/implicit-GEMM-like vs transform-based path，除非 template compatibility 已证明相容。

## 11. HET-3: IO-Aware Attention Tile

### 11.1 关键字段

| 字段 | 作用 |
|---|---|
| `batch` | batch |
| `num_heads` | head count |
| `seq_q` | query length |
| `seq_kv` | key/value length |
| `head_dim` | per-head dimension |
| `causal` | causal / bidirectional |
| `mask_type` | dense mask / sparse mask / none |
| `phase_role` | prefill / decode / mixed |
| `kv_cache_state` | read/write/update behavior |

### 11.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `attention_prefill_short_seq` | prefill，短序列 |
| `attention_prefill_medium_seq` | prefill，中等序列 |
| `attention_prefill_long_seq` | prefill，长序列，IO pressure 高 |
| `attention_decode_single_query` | decode，seq_q 通常很小 |
| `attention_cross_attention` | seq_q 和 seq_kv 来源不同 |
| `attention_head_dim_small` | head_dim 小，tile/fragment 行为不同 |
| `attention_head_dim_large` | head_dim 大，register/shared pressure 高 |
| `attention_mask_heavy` | mask 影响执行路径 |

### 11.3 Merge 规则

可以 merge：

- 同为 prefill 或同为 decode；
- seq_q / seq_kv 落在同一 size class；
- head_dim size class 相同；
- causal/mask 行为相容；
- KV cache 行为相容；
- fused primitives 一致或 absorbed primitive 列表相容。

### 11.4 Split 规则

必须拆分：

- prefill vs decode；
- short-seq vs long-seq；
- causal vs non-causal 且影响 execution path；
- dense attention vs sparse/masked-heavy attention；
- head_dim small vs large 且造成 register/shared pressure 明显不同；
- standalone attention primitives vs fused flash/IO-aware template。

## 12. HET-4: Reduction / Scan / Normalize Template

### 12.1 关键字段

| 字段 | 作用 |
|---|---|
| `reduction_axis` | row-wise / column-wise / full tensor / segment |
| `axis_length` | 被 reduction 的长度 |
| `outer_size` | reduction 外层并行度 |
| `op_type` | sum/max/mean/softmax/layernorm/RMSNorm/batchnorm |
| `dtype` | accumulation precision |
| `need_index` | argmax/topk-like 是否返回 index |
| `numerical_path` | stable softmax、variance path 等 |

### 12.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `reduction_rowwise_short` | row-wise 短轴 |
| `reduction_rowwise_long` | row-wise 长轴 |
| `reduction_full_tensor` | 全局 reduction |
| `normalization_hidden_small` | hidden dim 小 |
| `normalization_hidden_large` | hidden dim 大 |
| `softmax_seq_short` | softmax 短序列 |
| `softmax_seq_long` | softmax 长序列 |
| `segmented_reduction_irregular` | segment 长度不规则 |

### 12.3 Split 规则

必须拆分：

- row-wise vs full tensor reduction；
- short axis vs long axis；
- normalization vs pooling/loss reduction；
- softmax numerical path vs simple sum/max；
- regular reduction vs segmented/irregular reduction。

## 13. HET-5: Elementwise / Pointwise Fusion Template

### 13.1 关键字段

| 字段 | 作用 |
|---|---|
| `num_elements` | 总元素数量 |
| `vector_width` | vectorized load/store |
| `fused_op_count` | 融合操作数量 |
| `broadcast_pattern` | scalar / row / column / tensor broadcast |
| `memory_layout` | contiguous / strided |
| `inplace` | 是否 in-place |
| `dtype` | bandwidth footprint |

### 13.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `elementwise_tiny_latency` | 元素数很小，launch/latency 敏感 |
| `elementwise_bandwidth_large` | 大 tensor，bandwidth 主导 |
| `elementwise_broadcast_rowwise` | row-wise broadcast |
| `elementwise_broadcast_channelwise` | channel/channel-like broadcast |
| `elementwise_fused_activation_bias` | activation/bias/residual 融合 |
| `elementwise_strided_layout` | stride/layout 影响 coalescing |

### 13.3 Split 规则

必须拆分：

- tiny latency-sensitive vs large bandwidth-dominated；
- contiguous vs strided；
- simple unary/binary vs multi-op fused；
- scalar broadcast vs row/channel/tensor broadcast；
- in-place vs out-of-place 且 memory traffic 不同。

## 14. HET-6: Streaming Gather / Weighted Aggregation Template

### 14.1 关键字段

| 字段 | 作用 |
|---|---|
| `input_count` | 被聚合对象数量 |
| `feature_dim` | 每个 value / feature 的维度 |
| `weight_shape` | 权重维度和 broadcast 行为 |
| `gather_pattern` | contiguous / strided / indexed |
| `reuse_distance` | locality / cache reuse |
| `output_count` | 输出数量 |
| `accumulation_axis` | 聚合轴 |

### 14.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `aggregation_contiguous_streaming` | 连续 streaming aggregation |
| `aggregation_indexed_gather` | index-driven gather |
| `aggregation_high_reuse` | value/cache reuse 强 |
| `aggregation_low_reuse` | DRAM streaming 主导 |
| `aggregation_small_feature_dim` | feature dim 小 |
| `aggregation_large_feature_dim` | feature dim 大 |

### 14.3 Split 规则

必须拆分：

- contiguous streaming vs indexed gather；
- high reuse vs low reuse；
- small feature dim vs large feature dim；
- attention PV-like aggregation vs graph edge-driven aggregation；
- deterministic accumulation vs atomic/scatter-heavy accumulation。

## 15. HET-7: Embedding / Table Lookup Template

### 15.1 关键字段

| 字段 | 作用 |
|---|---|
| `table_size` | embedding table footprint |
| `embedding_dim` | vector width |
| `lookup_count` | lookup 数量 |
| `hotness` | 每个 slot 的 lookup 个数 |
| `pooling_type` | sum/mean/concat/none |
| `num_tables` | table 数量 |
| `sharding` | model parallel / table parallel |
| `id_distribution` | uniform / skewed / cached hot ids |

### 15.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `embedding_small_table_cacheable` | table 较小，cache 命中可能高 |
| `embedding_large_table_random` | table 大，随机访问主导 |
| `embedding_high_hotness_pooling` | slot 内 lookup/reduction 多 |
| `embedding_low_hotness_lookup` | lookup 少，pooling 弱 |
| `embedding_multi_table_concat` | 多表 concat |
| `embedding_sharded_parallel` | table/model parallel |

### 15.3 Split 规则

必须拆分：

- small cacheable table vs huge random table；
- high hotness pooling vs simple lookup；
- single table vs multi-table concat；
- local embedding vs sharded embedding；
- skewed hot-id distribution vs uniform random distribution。

## 16. HET-8: Sparse / Irregular Matrix-Graph Template

### 16.1 关键字段

| 字段 | 作用 |
|---|---|
| `nnz` | non-zero 数量 |
| `rows` / `cols` | sparse matrix size |
| `sparse_format` | CSR/COO/CSC/blocked |
| `degree_distribution` | graph degree 分布 |
| `segment_length_stats` | segment min/mean/max |
| `feature_dim` | dense feature 维度 |
| `atomic_usage` | 是否需要 atomic/scatter |
| `load_balance_hint` | row/edge load balance |

### 16.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `sparse_regular_blocked` | block sparse 或规则 sparse |
| `sparse_csr_balanced` | CSR 行长度较均衡 |
| `sparse_csr_powerlaw` | power-law degree / load imbalance |
| `sparse_low_nnz` | nnz 小，latency/overhead 明显 |
| `sparse_high_nnz_streaming` | nnz 大，memory streaming 明显 |
| `graph_small_feature_dim` | graph feature dim 小 |
| `graph_large_feature_dim` | graph feature dim 大 |

### 16.3 Split 规则

必须拆分：

- balanced sparse vs power-law / skewed sparse；
- blocked sparse vs unstructured sparse；
- low nnz vs high nnz；
- gather-only vs scatter/atomic；
- small feature dim vs large feature dim；
- sparse matrix-like path vs graph traversal-like path。

## 17. HET-9: Selection / Sort / Routing Template

### 17.1 关键字段

| 字段 | 作用 |
|---|---|
| `candidate_count` | 候选数量 |
| `k` | top-k 的 k |
| `batch` | batch / token count |
| `sort_required` | full sort / partial select |
| `histogram_bins` | bucketization / histogram |
| `routing_targets` | experts / beams / buckets |
| `dispatch_shape` | dispatch 后 shape |

### 17.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `selection_top1_small_candidate` | top-1，小候选集 |
| `selection_topk_medium_candidate` | top-k，中等候选 |
| `selection_sort_large_candidate` | 大候选集排序 |
| `routing_moe_low_expert_count` | MoE expert 少 |
| `routing_moe_high_expert_count` | MoE expert 多 |
| `routing_dispatch_imbalanced` | token-expert 分布不均 |
| `sampling_decode_small_batch` | decode sampling，小 batch |

### 17.3 Split 规则

必须拆分：

- top-1 vs top-k vs full sort；
- small candidate vs large candidate；
- routing selection vs sampling selection；
- balanced dispatch vs imbalanced dispatch；
- selection-only vs selection + pack/dispatch fused path。

## 18. HET-10: Layout / Pack / Quantize / Cache Update Template

### 18.1 关键字段

| 字段 | 作用 |
|---|---|
| `num_elements` | 数据规模 |
| `source_layout` / `target_layout` | layout transform |
| `stride_pattern` | coalescing |
| `pack_factor` | pack/unpack |
| `quant_dtype` | int8/fp8/fp16 等 |
| `scale_shape` | per-tensor/per-channel/per-group scale |
| `cache_shape` | KV cache block/page shape |
| `update_pattern` | append/read/update |

### 18.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `layout_contiguous_copy` | contiguous copy/reshape-like |
| `layout_transpose_strided` | transpose/strided movement |
| `pack_vectorized_aligned` | aligned pack/unpack |
| `pack_unaligned_fringe` | unaligned/fringe pack |
| `quant_per_tensor` | per-tensor quant/dequant |
| `quant_per_channel_or_group` | per-channel/per-group quant |
| `kv_cache_decode_update` | decode KV cache update |
| `kv_cache_prefill_bulk_write` | prefill bulk cache write |

### 18.3 Split 规则

必须拆分：

- contiguous copy vs transpose/strided；
- aligned pack vs unaligned fringe；
- per-tensor quant vs per-channel/per-group quant；
- prefill bulk cache write vs decode incremental update；
- pure layout movement vs fused layout + compute。

## 19. HET-11: Collective Communication Template

### 19.1 关键字段

| 字段 | 作用 |
|---|---|
| `collective_type` | allreduce/allgather/reduce-scatter/broadcast/send-recv |
| `message_size` | 每次通信大小 |
| `world_size` | 参与 GPU 数量 |
| `topology_hint` | NVLink/PCIe/IB 等 |
| `overlap_hint` | 是否与 compute overlap |
| `dtype` | payload footprint |
| `frequency` | 调用频率 |

### 19.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `collective_small_latency` | 小消息，latency 主导 |
| `collective_large_bandwidth` | 大消息，bandwidth 主导 |
| `collective_allreduce_gradient` | gradient allreduce |
| `collective_tensor_parallel_allreduce` | tensor parallel allreduce |
| `collective_reduce_scatter` | reduce-scatter |
| `collective_allgather` | allgather |
| `collective_overlap_sensitive` | overlap 决定性强 |

### 19.3 Split 规则

必须拆分：

- small message vs large message；
- allreduce vs allgather vs reduce-scatter；
- local intra-node vs inter-node；
- overlapped vs non-overlapped；
- high-frequency small collectives vs low-frequency large collectives。

## 20. 输出字段

Shape/size rulebook 应输出以下字段给 regime builder：

| 字段 | 含义 |
|---|---|
| `shape_size_regime` | 稳定 label |
| `shape_signature` | 归一化后的结构化 shape 描述 |
| `shape_size_class` | small/medium/large 或 template-specific class |
| `shape_role` | projection-like、decode-like、rowwise、random-lookup 等 |
| `shape_merge_reason` | 为什么可 merge |
| `shape_split_reason` | 为什么拆开 |
| `shape_boundary_reason` | 为什么只能 boundary/provisional |
| `shape_confidence` | low / medium / high |
| `shape_source` | trace / parser / manual / fixture / prior |

示例：

```text
shape_size_regime = dense_decode_small_m
shape_signature = {
  template: HET-1,
  m_class: small,
  n_class: hidden_sized,
  k_class: hidden_sized,
  role: decode_projection,
  tile_fit: tensor_core_friendly,
  reuse_profile: low_m_reuse
}
shape_confidence = high
shape_source = trace_metadata
```

## 21. 与 Resource Signature 的边界

Shape/size regime 只判断规模和形状区间，不直接最终决定 C-line lane。

例如：

```text
dense_decode_small_m
```

可能暗示 latency、tile underfill、occupancy 等风险，但最终是否沿 `occupancy_sensitive`、`register_sensitive` 或 `memory_bandwidth_sensitive` 验证，要由 resource signature 决定。

因此：

```text
shape/size regime = 候选执行区间
resource signature = 后端响应检查
lane = C 线验证方向
```

如果 shape/size 相近但 resource signature 不相容，必须拆 regime 或标 boundary。

## 22. 实现 Guardrails

未来实现 shape/size builder 时必须满足：

1. 不允许 raw grid/block 单独生成 stable shape regime；
2. 不允许 raw M/N/K 单独生成 stable regime label，必须经过 template-specific interpreter；
3. 不同 hardware template 使用不同 shape interpreter；
4. shape rule 只能在 phase + family + route primitive + compatible template 内运行；
5. 缺关键字段必须输出 blocker/boundary，不得猜 stable；
6. shape labels 必须稳定、可复现；
7. 所有 merge/split/boundary 都必须记录 reason；
8. fixture-only shape 不能 claim-bearing；
9. shape/size regime 不直接决定 lane，必须交给 resource signature；
10. 测试必须覆盖 dense、attention、reduction、embedding、sparse、layout/quantize 至少六类 template。

## 23. Acceptance Criteria

### AC-1: Phase / Family / Template 前置

Shape/size builder 只能在已有 phase、family、route primitive 和 compatible hardware template 后运行。

### AC-2: Template-Specific Interpretation

实现必须为不同 HET 使用不同 shape interpreter，不能用一套全局 numeric threshold 处理所有对象。

### AC-3: Stable Label 不来自单个 Raw Field

任何 stable `shape_size_regime` 都不能只由单个 raw shape、grid/block 或 kernel name 决定。

### AC-4: Boundary on Missing Evidence

缺关键 shape 字段、shape 冲突、或只有 fixture/prior evidence 时，必须输出 boundary/provisional/blocker。

### AC-5: Merge/Split Reason

每个 shape/size merge 或 split 都必须有 reason 字段，能解释具体使用了哪些 shape signature。

### AC-6: Resource Signature 后置检查

实现必须允许 resource signature 推翻 shape/size merge。shape/size 相近不等于最终 stable regime。

## 24. 当前结论

`shape_size_regime` 是 regime builder 中最容易做错的一步。

它应该被理解为：

```text
raw shape
  -> template-specific shape signature
  -> stable shape/size execution interval
  -> resource signature compatibility check
```

它的价值在于：

- 防止同 family + same template 的对象被过度合并；
- 防止每个具体 shape 都变成一个 regime；
- 保留 prefill/decode、small/large、regular/irregular、cache/DRAM、tile-friendly/fringe 等关键执行差异；
- 给 C 线提供更清晰的 validation target。
