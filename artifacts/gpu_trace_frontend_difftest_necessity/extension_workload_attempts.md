# Round 2 Extension Workload Attempt Evidence

**Generated**: 2026-05-05T00:18:00+08:00

## task-E1: BERT-base Pretraining Full Step

Status: trace export passed; bounded simulator replay blocked.

Evidence: `bert_full_step_attempt.json`.

The new `docs/bert-full-step-training.py` harness validates without NVBit at batch 1 and sequence length 16, peaking at 1042.7 MiB. NVBit export completed and produced `/tmp/bert_full_step_trace_r3/traces` with 20.0 GB of trace artifacts. Bounded replay entered `accel-sim.out` but failed deterministically on unsupported Blackwell opcode `UF2I.FTZ.U32.TRUNC.NTZ`.

Next action: extend simulator trace opcode support for Blackwell `UF2I` before BERT full-step timing/redundancy can be measured.

## task-E2: Llama 3.1 8B Decoder-Layer Slice

Status: trace export and bounded simulator replay passed.

Evidence: `llama_decoder_layer_attempt.json`.

The new `docs/llama-decoder-layer-training.py` harness validates without NVBit at batch 1, sequence length 8, hidden size 4096, and intermediate size 14336, peaking at 864.0 MiB. NVBit export completed and produced `/tmp/llama_decoder_layer_trace_r2/traces` with 6.4 GB of trace artifacts. Bounded replay reached the 50000-cycle cap with `gpu_tot_sim_insn = 12330848` and `gpgpu_simulation_time = 205s`.

## bitlesson-selector

`bitlesson-selector` was not found on PATH or under `/home/dyf`. Round 2 applied `BL-20260503-repo-structure` manually by verifying filesystem paths and trace artifacts directly.
