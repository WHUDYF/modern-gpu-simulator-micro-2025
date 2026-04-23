# mini_transformer_v4 Middle Layer Artifacts

- workload: `mini_transformer_v4`
- builder: `experiments/baseline_diagnosis/build_middle_layer.py`
- rule config: `docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml`
- rule config version: `v1`
- importance formula: `0.3*coverage_label + 0.4*time_label + 0.3*decision_label`
- regime priority formula: `0.35*family_importance + 0.25*coverage_label + 0.25*time_label + 0.15*local_decision_label`

## Importance Scoring Sheet

| object_level | object_id | parent_family_id | coverage_weight | time_weight | decision_weight | family_importance_score | local_decision_weight | regime_priority_score | importance_score |
|---|---|---|---|---|---|---|---|---|---|
| family | F1_dense_tiled_backbone | F1_dense_tiled_backbone | High | High | High | 0.9 | None | None | 0.9 |
| family | F2_reduction_normalize | F2_reduction_normalize | Medium | Medium-High | High | 0.75 | None | None | 0.75 |
| family | F3_streaming_aggregation | F3_streaming_aggregation | Medium | Medium | Medium-High | 0.645 | None | None | 0.645 |
| family | F4_elementwise_residual | F4_elementwise_residual | High | Low | Low-Medium | 0.525 | None | None | 0.525 |
| regime | R1_qkv_projection_dense | F1_dense_tiled_backbone | High | High | High | 0.9 | High | 0.9 | 0.9 |
| regime | R2_attention_score_dense | F1_dense_tiled_backbone | Medium | Medium-High | High | 0.9 | High | 0.7875 | 0.7875 |
| regime | R3_output_projection_dense | F1_dense_tiled_backbone | Medium | Medium | Medium | 0.9 | Medium | 0.705 | 0.705 |
| regime | R4_ffn_expand_dense | F1_dense_tiled_backbone | Medium | High | High | 0.9 | High | 0.825 | 0.825 |
| regime | R5_ffn_contract_dense | F1_dense_tiled_backbone | Medium | Medium | Medium | 0.9 | Medium | 0.705 | 0.705 |
| regime | R6_softmax_reduction | F2_reduction_normalize | Medium | Medium-High | High | 0.75 | High | 0.735 | 0.735 |
| regime | R7_layernorm_reduction | F2_reduction_normalize | Medium | Medium | Medium | 0.75 | Medium | 0.6525 | 0.6525 |
| regime | R8_context_streaming | F3_streaming_aggregation | Medium | Medium | Medium-High | 0.645 | Medium-High | 0.6382 | 0.6382 |
| regime | R9_residual_elementwise | F4_elementwise_residual | High | Low | Low | 0.525 | Low | 0.5288 | 0.5288 |

