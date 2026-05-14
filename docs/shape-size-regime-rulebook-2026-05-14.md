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

### 2.1 2026-05-14 简化修正：PKA-aware first

本 rulebook 不应把 B 线 shape/size 判断做成一套全量专家系统。更稳的设计是先尊重 A 线 PKA-style compression 已经保留的 measured scale evidence，然后只在必要时做 B 线 refinement。

依据现有 A 线 PKA spec：

- PKA 12D 中 `num_instructions` 是 work-size signal；
- PKA 12D 中 `num_thread_blocks` 对应 launch grid size，是 kernel 规模信号；
- memory operation counts、shared/global/local access、atomics、divergence 也会反映 shape/size 带来的执行行为差异；
- `grid_dim`、`block_dim`、`shape_hint` 不应进入 PKA 主 grouping，只能作为 metadata / audit / constrained refinement。

因此本文档采用两层判断：

```text
Stage A: PKA cluster shape consistency
  先看同一个 PKA cluster 内 measured scale/work features 是否紧凑。

Stage B: B-line constrained shape refinement
  只在 PKA cluster 内部出现 shape/size 混合、或 B/C 线需要更明确标签时，
  才使用 route/template-specific shape fields 做有限 refinement。
```

这意味着：

```text
shape/size regime 不从零开始重建 clustering；
它先继承 PKA 的 measured behavior/scale evidence，
再检查这个 cluster 是否需要被 B 线拆分或标 boundary。
```

后文列出的 HET-specific labels 是 refinement vocabulary，不是第一版必须全部实现的分类器。

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

### 3.4 PKA Shape Consistency

`pka_shape_consistency` 是 B 线优先使用的 shape/size evidence。

它回答：

```text
同一个 PKA cluster 的成员，在 measured scale/work behavior 上是否已经足够一致？
```

第一版主要看：

| PKA evidence | 作用 |
|---|---|
| `num_instructions` | work-size / dynamic instruction scale |
| `num_thread_blocks` | launch grid size / thread-block scale |
| global/local/shared memory operation counts | shape 引起的 memory footprint / access behavior |
| `thread_global_atomics` | sparse/scatter/irregular scale signal |
| `divergence_efficiency` | irregular / branch behavior signal |
| cluster feature variance | cluster 内部是否混入多种规模行为 |

如果这些 measured features 在 cluster 内紧凑，B 线不应重新用复杂 shape rules 拆分它。此时 shape/size 层只生成一个 coarse label，例如：

```text
pka_cluster_shape_consistent
pka_cluster_small_work
pka_cluster_medium_work
pka_cluster_large_work
pka_cluster_grid_limited
```

如果这些 measured features 显示 cluster 内部有多个规模模式，B 线再进入 constrained refinement。

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
| 缺 PKA measured scale/work summary | `provisional_missing_pka_shape_summary` 或 `blocked_missing_pka_shape_basis` |
| 缺 template-specific 关键 shape 字段 | Stage A 不受阻；Stage B refinement 输出 `blocked_missing_shape_signature` |
| 只有 kernel name / operator name | `blocked_no_shape_basis` |
| 只有 grid/block，没有 model shape 或 launch semantics | `provisional_grid_only_shape` |
| shape 来自人工 annotation 且无 provenance | `boundary_untrusted_shape_prior` |
| fixture shape 无 claim-bearing evidence | `fixture_non_claim_bearing` |

## 5. 通用判定流程

Shape/size rulebook 的执行顺序固定为 PKA-aware two-stage flow：

```text
1. 读取 phase + family + route primitive + hardware template
2. 读取 PKA cluster membership 和 12D measured feature summary
3. 先做 PKA cluster shape consistency check
4. 如果 PKA scale/work features 紧凑，直接输出 coarse shape_size_regime
5. 如果 PKA cluster 内部规模混合，再启用 template-specific constrained refinement
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

### 5.1 Stage A: PKA Cluster Shape Consistency Check

这是第一版实现的默认主路径。

输入：

| 字段 | 作用 |
|---|---|
| `source_cluster_id` | 只作为 provenance 和 membership 分组入口 |
| `member_record_ids` | cluster 成员 |
| `pka_feature_summary` | 12D measured feature 的 min/median/max/variance |
| `num_instructions_summary` | work-size 一致性 |
| `num_thread_blocks_summary` | launch size 一致性 |
| `memory_ops_summary` | memory footprint / access behavior 一致性 |
| `atomic_divergence_summary` | irregular behavior 一致性 |

判断：

```text
如果 cluster 内 measured scale/work features 紧凑：
  shape_size_regime = pka_cluster_shape_consistent 或 coarse size label
  不进入复杂 HET-specific split

如果 cluster 内 measured scale/work features 呈多峰或 outlier 明显：
  shape_size_regime = boundary_mixed_pka_shape_scale
  进入 Stage B constrained refinement
```

这里的“紧凑”不在 spec 中定死全局阈值。实现应先输出 summary 和 reason，阈值由实验配置或 calibration 决定。

### 5.2 Stage B: B-line Constrained Shape Refinement

只有在下面情况才启用 Stage B：

1. 同一 PKA cluster 内 `num_instructions` 或 `num_thread_blocks` 明显分裂；
2. memory/atomic/divergence features 暗示同 cluster 内存在不同 shape-driven behavior；
3. B 线 phase/route/template 显示同一 PKA cluster 混入不同 execution context；
4. C 线需要更明确的 validation target label。

Stage B 只能做 constrained refinement：

```text
不能重新替代 PKA cluster；
不能用 raw shape 重新全局聚类；
不能把 template-specific labels 当作必须穷尽的 taxonomy。
```

Stage B 的输出通常是：

```text
pka_cluster_shape_consistent
pka_cluster_shape_split_by_work_size
pka_cluster_shape_split_by_launch_size
pka_cluster_shape_split_by_memory_behavior
template_refined_dense_decode_small_m
template_refined_attention_long_seq
boundary_mixed_pka_shape_scale
```

## 6. 通用 Merge 规则

两个 anchors 可以进入同一个 `shape_size_regime`，必须满足：

1. 已经处在同一 `phase_id` 或明确相容的 phase context；
2. 已经处在同一 `family_id`；
3. `route_primitive` 相同或显式相容；
4. `hardware_template` 相同或显式相容；
5. PKA measured scale/work features 相容，或有明确 reason 解释为什么可以跨 PKA cluster merge；
6. 如果启用了 Stage B，template-specific 关键维度落在同一执行区间；
7. shape source 可信，且 confidence 不强烈冲突；
8. 没有已知 resource signature 冲突。

推荐输出：

```text
shape_size_regime = <stable label>
shape_merge_reason = pka cluster scale/work features compact; optional template refinement compatible
shape_confidence = high | medium | low
```

## 7. 通用 Split 规则

出现以下情况时，shape/size 层默认拆分：

| Split 因素 | 说明 |
|---|---|
| PKA scale/work feature 多峰 | 同一 cluster 内 measured behavior 已经显示多种规模 |
| PKA `num_instructions` 差异显著 | work-size 不同 |
| PKA `num_thread_blocks` 差异显著 | launch size / grid scale 不同 |
| PKA memory/atomic/divergence 差异显著 | shape 可能导致不同 memory/irregular behavior |
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
shape_split_reason = PKA measured scale/work features split, optionally refined by template-specific shape fields
```

## 8. Boundary 规则

出现以下情况时，不允许输出 stable shape/size regime：

| 情况 | boundary status |
|---|---|
| 关键维度缺失 | `blocked_missing_shape_signature` |
| shape 字段互相矛盾 | `boundary_conflicting_shape_evidence` |
| raw shape 可见但算法角色不明 | `boundary_unknown_shape_role` |
| PKA cluster 内 measured scale/work features 多峰 | `boundary_mixed_pka_shape_scale` |
| PKA cluster 内 `num_thread_blocks` 或 `num_instructions` outlier 明显 | `boundary_pka_scale_outlier` |
| 同一候选 regime 内混入多个 size class | `boundary_mixed_size_class` |
| shape 相近但 resource signature 强冲突 | `boundary_shape_resource_conflict` |
| 只有 fixture / synthetic shape | `fixture_non_claim_bearing` |
| 只有 network prior，无 measured/proxy evidence | `provisional_shape_prior_only` |

Boundary 对象可以进入 review/report，但不能作为 claim-bearing C-line stable validation target。

## 9. HET-Specific Refinement Vocabulary

从本节开始的 HET-specific 规则是第二阶段 constrained refinement vocabulary。

第一版实现不需要一次性完整实现全部 HET labels。推荐先实现：

1. PKA cluster shape consistency check；
2. dense / attention / reduction 三类常见 template 的少量 refinement；
3. boundary 输出；
4. reason 和 confidence 记录。

只有当 PKA cluster 的 measured scale/work evidence 不足以决定 shape/size 相容性时，才进入下面的 template-specific 判断。

## 10. HET-1: Dense Tiled Tensor-Core Compute

### 10.1 关键字段

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

### 10.2 推荐 shape labels

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

### 10.3 Merge 规则

Dense 对象可以 merge 的条件：

- PKA scale/work features 已经相容，或 Stage B 有明确 split/merge reason；
- 同 route primitive 或显式相容，例如同为 projection-like；
- M/N/K 所在 size class 相同；
- dtype 和 tensor-core eligibility 相容；
- alignment / tile fit 没有明显差异；
- batch/grouped 行为相同或相容；
- 不混合 decode small-M 和 prefill large-M。

### 10.4 Split 规则

必须拆分：

- `dense_decode_small_m` vs `dense_prefill_large_m`；
- `dense_projection_balanced` vs `dense_ffn_expansion`；
- `dense_pairwise_score` vs `dense_projection_balanced`；
- regular GEMM vs grouped expert GEMM；
- tensor-core-friendly vs tile-fringe / alignment-poor；
- batched GEMM vs single large GEMM，除非 profiling 证明行为相容。

## 11. HET-2: Convolution / Stencil Tiled Compute

### 11.1 关键字段

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

### 11.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `conv_1x1_projection` | 1x1 conv，接近 dense projection |
| `conv_spatial_3x3_regular` | 常规 spatial conv |
| `conv_depthwise_channelwise` | depthwise / high groups |
| `conv_grouped` | grouped conv |
| `conv_large_spatial` | H/W 大，activation reuse 明显 |
| `conv_small_spatial_large_channel` | spatial 小但 channel 大 |
| `conv_layout_sensitive` | layout/transpose 影响明显 |

### 11.3 Split 规则

必须拆分：

- 1x1 conv vs spatial conv；
- regular conv vs depthwise/grouped conv；
- large spatial vs small spatial；
- layout-sensitive path vs layout-stable path；
- direct/implicit-GEMM-like vs transform-based path，除非 template compatibility 已证明相容。

## 12. HET-3: IO-Aware Attention Tile

### 12.1 关键字段

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

### 12.2 推荐 shape labels

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

### 12.3 Merge 规则

可以 merge：

- PKA scale/work features 已经相容，或 Stage B 有明确 split/merge reason；
- 同为 prefill 或同为 decode；
- seq_q / seq_kv 落在同一 size class；
- head_dim size class 相同；
- causal/mask 行为相容；
- KV cache 行为相容；
- fused primitives 一致或 absorbed primitive 列表相容。

### 12.4 Split 规则

必须拆分：

- prefill vs decode；
- short-seq vs long-seq；
- causal vs non-causal 且影响 execution path；
- dense attention vs sparse/masked-heavy attention；
- head_dim small vs large 且造成 register/shared pressure 明显不同；
- standalone attention primitives vs fused flash/IO-aware template。

## 13. HET-4: Reduction / Scan / Normalize Template

### 13.1 关键字段

| 字段 | 作用 |
|---|---|
| `reduction_axis` | row-wise / column-wise / full tensor / segment |
| `axis_length` | 被 reduction 的长度 |
| `outer_size` | reduction 外层并行度 |
| `op_type` | sum/max/mean/softmax/layernorm/RMSNorm/batchnorm |
| `dtype` | accumulation precision |
| `need_index` | argmax/topk-like 是否返回 index |
| `numerical_path` | stable softmax、variance path 等 |

### 13.2 推荐 shape labels

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

### 13.3 Split 规则

必须拆分：

- row-wise vs full tensor reduction；
- short axis vs long axis；
- normalization vs pooling/loss reduction；
- softmax numerical path vs simple sum/max；
- regular reduction vs segmented/irregular reduction。

## 14. HET-5: Elementwise / Pointwise Fusion Template

### 14.1 关键字段

| 字段 | 作用 |
|---|---|
| `num_elements` | 总元素数量 |
| `vector_width` | vectorized load/store |
| `fused_op_count` | 融合操作数量 |
| `broadcast_pattern` | scalar / row / column / tensor broadcast |
| `memory_layout` | contiguous / strided |
| `inplace` | 是否 in-place |
| `dtype` | bandwidth footprint |

### 14.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `elementwise_tiny_latency` | 元素数很小，launch/latency 敏感 |
| `elementwise_bandwidth_large` | 大 tensor，bandwidth 主导 |
| `elementwise_broadcast_rowwise` | row-wise broadcast |
| `elementwise_broadcast_channelwise` | channel/channel-like broadcast |
| `elementwise_fused_activation_bias` | activation/bias/residual 融合 |
| `elementwise_strided_layout` | stride/layout 影响 coalescing |

### 14.3 Split 规则

必须拆分：

- tiny latency-sensitive vs large bandwidth-dominated；
- contiguous vs strided；
- simple unary/binary vs multi-op fused；
- scalar broadcast vs row/channel/tensor broadcast；
- in-place vs out-of-place 且 memory traffic 不同。

## 15. HET-6: Streaming Gather / Weighted Aggregation Template

### 15.1 关键字段

| 字段 | 作用 |
|---|---|
| `input_count` | 被聚合对象数量 |
| `feature_dim` | 每个 value / feature 的维度 |
| `weight_shape` | 权重维度和 broadcast 行为 |
| `gather_pattern` | contiguous / strided / indexed |
| `reuse_distance` | locality / cache reuse |
| `output_count` | 输出数量 |
| `accumulation_axis` | 聚合轴 |

### 15.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `aggregation_contiguous_streaming` | 连续 streaming aggregation |
| `aggregation_indexed_gather` | index-driven gather |
| `aggregation_high_reuse` | value/cache reuse 强 |
| `aggregation_low_reuse` | DRAM streaming 主导 |
| `aggregation_small_feature_dim` | feature dim 小 |
| `aggregation_large_feature_dim` | feature dim 大 |

### 15.3 Split 规则

必须拆分：

- contiguous streaming vs indexed gather；
- high reuse vs low reuse；
- small feature dim vs large feature dim；
- attention PV-like aggregation vs graph edge-driven aggregation；
- deterministic accumulation vs atomic/scatter-heavy accumulation。

## 16. HET-7: Embedding / Table Lookup Template

### 16.1 关键字段

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

### 16.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `embedding_small_table_cacheable` | table 较小，cache 命中可能高 |
| `embedding_large_table_random` | table 大，随机访问主导 |
| `embedding_high_hotness_pooling` | slot 内 lookup/reduction 多 |
| `embedding_low_hotness_lookup` | lookup 少，pooling 弱 |
| `embedding_multi_table_concat` | 多表 concat |
| `embedding_sharded_parallel` | table/model parallel |

### 16.3 Split 规则

必须拆分：

- small cacheable table vs huge random table；
- high hotness pooling vs simple lookup；
- single table vs multi-table concat；
- local embedding vs sharded embedding；
- skewed hot-id distribution vs uniform random distribution。

## 17. HET-8: Sparse / Irregular Matrix-Graph Template

### 17.1 关键字段

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

### 17.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `sparse_regular_blocked` | block sparse 或规则 sparse |
| `sparse_csr_balanced` | CSR 行长度较均衡 |
| `sparse_csr_powerlaw` | power-law degree / load imbalance |
| `sparse_low_nnz` | nnz 小，latency/overhead 明显 |
| `sparse_high_nnz_streaming` | nnz 大，memory streaming 明显 |
| `graph_small_feature_dim` | graph feature dim 小 |
| `graph_large_feature_dim` | graph feature dim 大 |

### 17.3 Split 规则

必须拆分：

- balanced sparse vs power-law / skewed sparse；
- blocked sparse vs unstructured sparse；
- low nnz vs high nnz；
- gather-only vs scatter/atomic；
- small feature dim vs large feature dim；
- sparse matrix-like path vs graph traversal-like path。

## 18. HET-9: Selection / Sort / Routing Template

### 18.1 关键字段

| 字段 | 作用 |
|---|---|
| `candidate_count` | 候选数量 |
| `k` | top-k 的 k |
| `batch` | batch / token count |
| `sort_required` | full sort / partial select |
| `histogram_bins` | bucketization / histogram |
| `routing_targets` | experts / beams / buckets |
| `dispatch_shape` | dispatch 后 shape |

### 18.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `selection_top1_small_candidate` | top-1，小候选集 |
| `selection_topk_medium_candidate` | top-k，中等候选 |
| `selection_sort_large_candidate` | 大候选集排序 |
| `routing_moe_low_expert_count` | MoE expert 少 |
| `routing_moe_high_expert_count` | MoE expert 多 |
| `routing_dispatch_imbalanced` | token-expert 分布不均 |
| `sampling_decode_small_batch` | decode sampling，小 batch |

### 18.3 Split 规则

必须拆分：

- top-1 vs top-k vs full sort；
- small candidate vs large candidate；
- routing selection vs sampling selection；
- balanced dispatch vs imbalanced dispatch；
- selection-only vs selection + pack/dispatch fused path。

## 19. HET-10: Layout / Pack / Quantize / Cache Update Template

### 19.1 关键字段

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

### 19.2 推荐 shape labels

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

### 19.3 Split 规则

必须拆分：

- contiguous copy vs transpose/strided；
- aligned pack vs unaligned fringe；
- per-tensor quant vs per-channel/per-group quant；
- prefill bulk cache write vs decode incremental update；
- pure layout movement vs fused layout + compute。

## 20. HET-11: Collective Communication Template

### 20.1 关键字段

| 字段 | 作用 |
|---|---|
| `collective_type` | allreduce/allgather/reduce-scatter/broadcast/send-recv |
| `message_size` | 每次通信大小 |
| `world_size` | 参与 GPU 数量 |
| `topology_hint` | NVLink/PCIe/IB 等 |
| `overlap_hint` | 是否与 compute overlap |
| `dtype` | payload footprint |
| `frequency` | 调用频率 |

### 20.2 推荐 shape labels

| Label | 典型含义 |
|---|---|
| `collective_small_latency` | 小消息，latency 主导 |
| `collective_large_bandwidth` | 大消息，bandwidth 主导 |
| `collective_allreduce_gradient` | gradient allreduce |
| `collective_tensor_parallel_allreduce` | tensor parallel allreduce |
| `collective_reduce_scatter` | reduce-scatter |
| `collective_allgather` | allgather |
| `collective_overlap_sensitive` | overlap 决定性强 |

### 20.3 Split 规则

必须拆分：

- small message vs large message；
- allreduce vs allgather vs reduce-scatter；
- local intra-node vs inter-node；
- overlapped vs non-overlapped；
- high-frequency small collectives vs low-frequency large collectives。

## 21. 输出字段

Shape/size rulebook 应输出以下字段给 regime builder：

| 字段 | 含义 |
|---|---|
| `shape_size_regime` | 稳定 label |
| `shape_signature` | 归一化后的结构化 shape 描述 |
| `pka_shape_consistency` | PKA cluster 内 measured scale/work features 是否紧凑 |
| `pka_scale_summary` | `num_instructions`、`num_thread_blocks` 等摘要 |
| `shape_size_class` | small/medium/large 或 template-specific class |
| `shape_role` | projection-like、decode-like、rowwise、random-lookup 等 |
| `shape_merge_reason` | 为什么可 merge |
| `shape_split_reason` | 为什么拆开 |
| `shape_boundary_reason` | 为什么只能 boundary/provisional |
| `shape_confidence` | low / medium / high |
| `shape_source` | trace / parser / manual / fixture / prior |

示例：

```text
shape_size_regime = pka_cluster_shape_consistent
pka_shape_consistency = compact
pka_scale_summary = {
  num_instructions_class: medium,
  num_thread_blocks_class: small,
  memory_behavior_class: coalesced_memory_light,
  variance_status: compact
}
shape_confidence = high
shape_source = pka_12d_measured_summary
```

Stage B refinement 示例：

```text
shape_size_regime = template_refined_dense_decode_small_m
pka_shape_consistency = mixed_scale_requires_refinement
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

## 22. 与 Resource Signature 的边界

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

## 23. 实现 Guardrails

未来实现 shape/size builder 时必须满足：

1. 优先使用 PKA cluster membership 和 12D measured scale/work features 做 shape consistency check；
2. 不允许 raw grid/block 单独生成 stable shape regime；
3. 不允许 raw M/N/K 单独生成 stable regime label；
4. template-specific interpreter 只能作为 Stage B constrained refinement；
5. shape rule 只能在 phase + family + route primitive + compatible template 内运行；
6. 缺 PKA measured scale/work summary 时，必须降级为 provisional/boundary；
7. 缺关键 template-specific 字段时，不能阻塞 Stage A，但会阻塞 Stage B stable refinement；
8. shape labels 必须稳定、可复现；
9. 所有 merge/split/boundary 都必须记录 reason；
10. fixture-only shape 不能 claim-bearing；
11. shape/size regime 不直接决定 lane，必须交给 resource signature；
12. 第一版测试优先覆盖 PKA compact cluster、PKA mixed-scale cluster、PKA scale outlier、dense refinement、attention refinement、boundary 六类行为。

## 24. Acceptance Criteria

### AC-1: Phase / Family / Template 前置

Shape/size builder 只能在已有 phase、family、route primitive 和 compatible hardware template 后运行。

### AC-2: PKA-Aware First

实现必须先读取 PKA cluster membership 和 12D measured scale/work summary。若 PKA cluster 内部 measured behavior 紧凑，不能强制进入复杂 template-specific split。

### AC-3: Stable Label 不来自单个 Raw Field

任何 stable `shape_size_regime` 都不能只由单个 raw shape、grid/block 或 kernel name 决定。

### AC-4: Template-Specific Interpretation Is Refinement

实现可以为不同 HET 使用不同 shape interpreter，但这些 interpreter 是 Stage B refinement，不是替代 PKA cluster consistency 的主路径。

### AC-5: Boundary on Missing Evidence

缺 PKA measured scale/work summary、关键 shape 字段冲突、或只有 fixture/prior evidence 时，必须输出 boundary/provisional/blocker。

### AC-6: Merge/Split Reason

每个 shape/size merge 或 split 都必须有 reason 字段，能解释具体使用了哪些 shape signature。

### AC-7: Resource Signature 后置检查

实现必须允许 resource signature 推翻 shape/size merge。shape/size 相近不等于最终 stable regime。

## 25. 当前结论

`shape_size_regime` 是 regime builder 中最容易做错的一步。

它应该被理解为：

```text
PKA cluster membership + 12D measured scale/work evidence
  -> PKA cluster shape consistency check
  -> optional B-line constrained shape refinement
  -> stable shape/size execution interval or boundary
  -> resource signature compatibility check
```

它的价值在于：

- 复用 PKA compression 已经提供的 measured scale/work evidence；
- 防止同 family + same template 的对象被过度合并；
- 防止每个具体 shape 都变成一个 regime；
- 只在必要时保留 prefill/decode、small/large、regular/irregular、cache/DRAM、tile-friendly/fringe 等关键执行差异；
- 给 C 线提供更清晰的 validation target。
