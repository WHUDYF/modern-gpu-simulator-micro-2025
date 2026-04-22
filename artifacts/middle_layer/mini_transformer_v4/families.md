# mini_transformer_v4 Middle Layer Artifacts

- workload: `mini_transformer_v4`
- builder: `experiments/baseline_diagnosis/build_middle_layer.py`
- importance formula: `0.3*coverage_label + 0.4*time_label + 0.3*decision_label`
- regime priority formula: `0.35*family_importance + 0.25*coverage_label + 0.25*time_label + 0.15*local_decision_label`

## Families

| family_id | input_anchor_ids | observed_coverage_ratio | observed_time_ratio | coverage_label | time_label | decision_label | importance_score | priority_class |
|---|---|---|---|---|---|---|---|---|
| F1_dense_tiled_backbone | ['A1_qkv_projection_dense_48x32', 'A2_attention_score_dense_32x32x12', 'A5_output_projection_dense_48x32', 'A8_ffn_expand_dense_192x32', 'A9_ffn_contract_dense_48x32'] | 0.5714 | 0.9192 | High | High | High | 0.9 | High |
| F2_reduction_normalize | ['A3_softmax_reduce_24x1', 'A7_layernorm_reduce_512'] | 0.2143 | 0.0198 | Medium | Medium-High | High | 0.75 | High |
| F3_streaming_aggregation | ['A4_context_stream_4x32x12'] | 0.0714 | 0.0579 | Medium | Medium | Medium-High | 0.645 | Medium |
| F4_elementwise_residual | ['A6_residual_elementwise_1536'] | 0.1429 | 0.0031 | High | Low | Low-Medium | 0.525 | Low |

