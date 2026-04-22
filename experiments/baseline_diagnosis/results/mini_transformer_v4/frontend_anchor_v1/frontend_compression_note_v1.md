# Frontend Compression Note

## Anchor generation
- Anchors are generated from an explicit dual-source CLI path in which the current v1 workflow uses the same premerged `full.json` file as both identity/context source and feature source.
- `kernel_invocation_id` is synthetic in v1 and follows `<kernel_name>#<trace_order>`.
- `member_invocations` are emitted as full lists in this v1 pass.

## Field status
- `coverage_weight`: derived from member counts.
- `time_weight`: derived from exec_time, preferring `duration_ns` and falling back to `elapsed_cycles`.
- `grid_dim` / `block_dim`: measured from the committed input data.
- `kernel_squash_*` / `tb_squash_*`: derived from squash summaries and used as context/guardrail support.
- `shape_hint_summary`: placeholder (`null`) in this v1 pass.

## Output boundary
- `Representative Anchor Table` is the mainline A-line output.
- `Comparison Table` and `Case Note` are evidence-only outputs and must not be treated as downstream mainline input tables.

## Bias sources
- Current likely bias sources include the premerged `full.json` shortcut path, synthetic invocation IDs, and still-lightweight squash guardrail integration.
