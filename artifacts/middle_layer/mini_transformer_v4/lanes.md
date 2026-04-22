# mini_transformer_v4 Middle Layer Artifacts

- workload: `mini_transformer_v4`
- builder: `experiments/baseline_diagnosis/build_middle_layer.py`
- rule config: `docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml`
- rule config version: `v1`
- importance formula: `0.3*coverage_label + 0.4*time_label + 0.3*decision_label`
- regime priority formula: `0.35*family_importance + 0.25*coverage_label + 0.25*time_label + 0.15*local_decision_label`

## Lanes

| lane_id | target_regime_id | target_family_id | parameter_direction | baseline_type | validation_metric |
|---|---|---|---|---|---|
| L1_dense_projection | R1_qkv_projection_dense | F1_dense_tiled_backbone | register-sensitive, occupancy-sensitive | importance-guided vs time-only | cycles delta, occupancy response, top-k coverage gain |
| L2_attention_score | R2_attention_score_dense | F1_dense_tiled_backbone | shared-memory-coupled, register-sensitive | importance-guided vs manual | cycles delta, cache behavior shift, shmem-coupled response |
| L3_output_projection | R3_output_projection_dense | F1_dense_tiled_backbone | register-sensitive, projection-path reuse | importance-guided vs family-shared baseline | cycles delta, reuse consistency, lane overlap |
| L4_ffn_expand | R4_ffn_expand_dense | F1_dense_tiled_backbone | register-sensitive, large-shape dense compute | importance-guided vs time-only | cycles delta, priority rank gain, sensitivity concentration |
| L5_ffn_contract | R5_ffn_contract_dense | F1_dense_tiled_backbone | dense contraction reuse, occupancy-sensitive | importance-guided vs no-priority | cycles delta, marginal gain |
| L6_softmax | R6_softmax_reduction | F2_reduction_normalize | cache-sensitive, reduction-sensitive | importance-guided vs time-only | cycles delta, dram throughput response, cache behavior response |
| L7_layernorm | R7_layernorm_reduction | F2_reduction_normalize | reduction-sensitive, normalization-path validation | importance-guided vs family-shared baseline | cycles delta, normalization consistency |
| L8_context_streaming | R8_context_streaming | F3_streaming_aggregation | locality-sensitive, L1-sensitive | importance-guided vs manual | cycles delta, l1 hit response, locality concentration |
| L9_residual_regression | R9_residual_elementwise | F4_elementwise_residual | lightweight memory-side, regression-check | no-priority baseline | correctness-preserving delta, regression stability |

