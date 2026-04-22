# mini_transformer_v4 Middle Layer Artifacts

- workload: `mini_transformer_v4`
- builder: `experiments/baseline_diagnosis/build_middle_layer.py`
- importance formula: `0.3*coverage_label + 0.4*time_label + 0.3*decision_label`
- regime priority formula: `0.35*family_importance + 0.25*coverage_label + 0.25*time_label + 0.15*local_decision_label`

## Regimes

| regime_id | family_id | source_anchor_ids | observed_coverage_ratio | observed_time_ratio | local_decision_label | regime_priority_score | validation_status |
|---|---|---|---|---|---|---|---|
| R1_qkv_projection_dense | F1_dense_tiled_backbone | ['A1_qkv_projection_dense_48x32'] | 0.2857 | 0.5011 | High | 0.9 | pending |
| R2_attention_score_dense | F1_dense_tiled_backbone | ['A2_attention_score_dense_32x32x12'] | 0.0714 | 0.0422 | High | 0.7875 | pending |
| R3_output_projection_dense | F1_dense_tiled_backbone | ['A5_output_projection_dense_48x32'] | 0.0714 | 0.1253 | Medium | 0.705 | pending |
| R4_ffn_expand_dense | F1_dense_tiled_backbone | ['A8_ffn_expand_dense_192x32'] | 0.0714 | 0.1253 | High | 0.825 | pending |
| R5_ffn_contract_dense | F1_dense_tiled_backbone | ['A9_ffn_contract_dense_48x32'] | 0.0714 | 0.1253 | Medium | 0.705 | pending |
| R6_softmax_reduction | F2_reduction_normalize | ['A3_softmax_reduce_24x1'] | 0.0714 | 0.015 | High | 0.735 | pending |
| R7_layernorm_reduction | F2_reduction_normalize | ['A7_layernorm_reduce_512'] | 0.1429 | 0.0048 | Medium | 0.6525 | pending |
| R8_context_streaming | F3_streaming_aggregation | ['A4_context_stream_4x32x12'] | 0.0714 | 0.0579 | Medium-High | 0.6382 | pending |
| R9_residual_elementwise | F4_elementwise_residual | ['A6_residual_elementwise_1536'] | 0.1429 | 0.0031 | Low | 0.5288 | pending |

