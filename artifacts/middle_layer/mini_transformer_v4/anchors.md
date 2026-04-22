# mini_transformer_v4 Middle Layer Artifacts

- workload: `mini_transformer_v4`
- builder: `experiments/baseline_diagnosis/build_middle_layer.py`
- importance formula: `0.3*coverage_label + 0.4*time_label + 0.3*decision_label`
- regime priority formula: `0.35*family_importance + 0.25*coverage_label + 0.25*time_label + 0.15*local_decision_label`

## Anchors

| anchor_id | phase_id | context_scope | member_invocations | observed_coverage_ratio | observed_time_ratio | coverage_label | time_label | route_hint | template_hint |
|---|---|---|---|---|---|---|---|---|---|
| A1_qkv_projection_dense_48x32 | Phase A | Q/K/V projection path | [1, 2, 3, 4] | 0.2857 | 0.5011 | High | High | Dense Projection/Transform | Dense Tiled Compute |
| A2_attention_score_dense_32x32x12 | Phase B | attention score path | [5] | 0.0714 | 0.0422 | Medium | Medium-High | Pairwise Score | Dense Tiled Compute |
| A3_softmax_reduce_24x1 | Phase B | attention normalization path | [6] | 0.0714 | 0.015 | Medium | Medium-High | Reduction / Normalize | Reduction Template |
| A4_context_stream_4x32x12 | Phase B | attention aggregation path | [7] | 0.0714 | 0.0579 | Medium | Medium | Weighted Aggregation | Streaming Aggregation Template |
| A5_output_projection_dense_48x32 | Phase B_to_C | attention output projection path | [8] | 0.0714 | 0.1253 | Medium | Medium | Dense Projection/Transform | Dense Tiled Compute |
| A6_residual_elementwise_1536 | Phase C | residual path | [9, 13] | 0.1429 | 0.0031 | High | Low | Elementwise Fusion | Elementwise Template |
| A7_layernorm_reduce_512 | Phase C | normalization path | [10, 14] | 0.1429 | 0.0048 | Medium | Medium | Reduction / Normalize | Reduction Template |
| A8_ffn_expand_dense_192x32 | Phase C | FFN expansion path | [11] | 0.0714 | 0.1253 | Medium | High | Dense Projection/Transform | Dense Tiled Compute |
| A9_ffn_contract_dense_48x32 | Phase C | FFN contraction path | [12] | 0.0714 | 0.1253 | Medium | Medium | Dense Projection/Transform | Dense Tiled Compute |

