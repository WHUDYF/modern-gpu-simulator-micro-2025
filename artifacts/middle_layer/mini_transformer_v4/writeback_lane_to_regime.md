# mini_transformer_v4 Middle Layer Artifacts

- workload: `mini_transformer_v4`
- builder: `experiments/baseline_diagnosis/build_middle_layer.py`
- rule config: `docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml`
- rule config version: `v1`
- importance formula: `0.3*coverage_label + 0.4*time_label + 0.3*decision_label`
- regime priority formula: `0.35*family_importance + 0.25*coverage_label + 0.25*time_label + 0.15*local_decision_label`

## Writeback Lane To Regime

| lane_id | target_regime_id | target_family_id | writeback_target | parameter_direction |
|---|---|---|---|---|
| L1_dense_projection | R1_qkv_projection_dense | F1_dense_tiled_backbone | R1 -> F1 -> dense backbone summary | register-sensitive, occupancy-sensitive |
| L2_attention_score | R2_attention_score_dense | F1_dense_tiled_backbone | R2 -> F1 boundary refinement | shared-memory-coupled, register-sensitive |
| L3_output_projection | R3_output_projection_dense | F1_dense_tiled_backbone | R3 -> F1 dense reuse note | register-sensitive, projection-path reuse |
| L4_ffn_expand | R4_ffn_expand_dense | F1_dense_tiled_backbone | R4 -> F1 FFN summary | register-sensitive, large-shape dense compute |
| L5_ffn_contract | R5_ffn_contract_dense | F1_dense_tiled_backbone | R5 -> F1 secondary regime note | dense contraction reuse, occupancy-sensitive |
| L6_softmax | R6_softmax_reduction | F2_reduction_normalize | R6 -> F2 reduction summary | cache-sensitive, reduction-sensitive |
| L7_layernorm | R7_layernorm_reduction | F2_reduction_normalize | R7 -> F2 normalization note | reduction-sensitive, normalization-path validation |
| L8_context_streaming | R8_context_streaming | F3_streaming_aggregation | R8 -> F3 streaming summary | locality-sensitive, L1-sensitive |
| L9_residual_regression | R9_residual_elementwise | F4_elementwise_residual | R9 -> F4 residual constraint note | lightweight memory-side, regression-check |

