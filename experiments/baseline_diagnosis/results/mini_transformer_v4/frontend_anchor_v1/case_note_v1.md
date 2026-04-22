# Frontend Anchor Case Note

_Output role: evidence_only_

## Representative split cases

### _Z10gemm_tiledPKfS0_Pfiii
- `name-only` groups: 1; `PKA-like coarse` groups: 2; `hybrid` groups: 3
- hybrid cluster `hybrid-1` members: _Z10gemm_tiledPKfS0_Pfiii#1, _Z10gemm_tiledPKfS0_Pfiii#2, _Z10gemm_tiledPKfS0_Pfiii#3, _Z10gemm_tiledPKfS0_Pfiii#4, _Z10gemm_tiledPKfS0_Pfiii#8
  - evidence: grid_dim=['48x32x1'], dynamic_inst_count=[98832384]
- hybrid cluster `hybrid-7` members: _Z10gemm_tiledPKfS0_Pfiii#11
  - evidence: grid_dim=['192x32x1'], dynamic_inst_count=[395329536]
- hybrid cluster `hybrid-8` members: _Z10gemm_tiledPKfS0_Pfiii#12
  - evidence: grid_dim=['48x32x1'], dynamic_inst_count=[392564736]
  - interpretation: hybrid introduces an extra boundary only when the bucket-internal execution signature changes enough to justify splitting anchors within the same coarse group.
