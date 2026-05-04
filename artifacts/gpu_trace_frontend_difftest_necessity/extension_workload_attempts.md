# Round 1 Extension Workload Attempt Evidence

**Generated**: 2026-05-04T23:15:00+08:00

## task-E1: BERT-base Pretraining Full Step

Status: blocked before trace export.

Hypothesis: a BERT-base full pretraining-step harness may exist locally and can be run from the smallest allowed batch under the 28 GiB per-GPU ceiling.

Commands checked:

- `rg -n "bert-base-pretraining-full-step|pretraining full" -S .`
- `find . -maxdepth 4 -type f \( -name '*bert*' -o -name '*trace*' \) | sort`
- `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader`

Result: no BERT full-step training harness exists in this repository. The only BERT harness found is `docs/bert-inference.py`, which runs a single `BertLayer` forward+backward slice. The available RTX 5090 GPUs have 32607 MiB total memory, so the plan's <=28 GiB per-GPU execution ceiling is compatible with the existing slice but does not supply the missing full-step harness.

Next action: implement or provide a BERT full-step trace harness, then run batch size 1 under NVBit and replay it through the instrumented simulator.

## task-E2: Llama 3.1 8B Decoder-Layer Slice

Status: blocked before trace export.

Hypothesis: a Llama 3.1 8B decoder-layer slice harness may exist locally and can be run as extension evidence after the measured BERT slice path.

Commands checked:

- `rg -n "llama3.1-8b-decoder-layer-slice|decoder-layer|llama" -S .`
- `find . -maxdepth 4 -type f \( -name '*llama*' -o -name '*trace*' \) | sort`
- `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader`

Result: no Llama decoder-layer harness exists in this repository. Existing Llama references are modeled/planning rows only. The local RTX 5090 GPUs have 32607 MiB total memory, below the 48 GiB infrastructure requirement recorded for Llama full-step work and too close to the 28 GiB per-GPU plan ceiling to synthesize an 8B slice trace without a validated harness.

Next action: provide or implement a bounded Llama decoder-layer harness with explicit memory controls, then run trace export and simulator replay as task-E2.

## bitlesson-selector

`bitlesson-selector` was not found on PATH or under `/home/dyf`. Round 1 applied `BL-20260503-repo-structure` manually by verifying both file discovery and artifact inventory before declaring harness absence.
