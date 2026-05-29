# GCL Reproduction Wiki Instructions

This directory is a lightweight knowledge wiki for understanding the staged GCL-Sampler reproduction plan in the A-line worktree.

Use this wiki to answer:

- What does each GCL stage do?
- Which artifacts are produced by each stage?
- Which claims are allowed at each stage?
- How does the pipeline move from fixture embeddings to trace graphs, RGCN embeddings, and simulator evaluation?

Conventions:

- Keep stage names as `GCL-M0`, `GCL-M1`, `GCL-M2`, and `GCL-M3`.
- Keep artifact names and JSON keys in English.
- Write explanatory prose in Chinese.
- Use wikilinks like `[[gcl-m1-trace-graph-construction]]` for relationships between articles.
- Do not mix B-line semantic metadata or simulator outcome fields into selector-side stage definitions.

