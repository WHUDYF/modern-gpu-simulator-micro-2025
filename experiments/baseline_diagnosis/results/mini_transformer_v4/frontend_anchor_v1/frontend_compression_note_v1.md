# Frontend Compression Note

- Anchors are generated from a v1 full.json shortcut input and synthetic kernel_invocation_id values.
- member_invocations are emitted as full lists in this v1 pass.
- coverage_weight is derived from member counts.
- time_weight is derived from exec_time (preferring duration_ns, then elapsed_cycles).
- Current likely bias sources include premerged input shortcuts, synthetic invocation IDs, and provisional squash summaries.
