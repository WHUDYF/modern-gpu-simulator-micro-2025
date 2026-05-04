# Round 3 Extension Workload Attempt Evidence

**Generated**: 2026-05-05T00:55:00+08:00

## task-E1: BERT-base Pretraining Full Step

Status: trace export and bounded simulator replay passed.

Evidence: `bert_full_step_attempt.json`.

The new `docs/bert-full-step-training.py` harness validates without NVBit at batch 1 and sequence length 16, peaking at 1042.7 MiB. NVBit export completed and produced `/tmp/bert_full_step_trace_r3/traces` with 20.0 GB of trace artifacts. The first bounded replay failed on unsupported Blackwell opcode `UF2I.FTZ.U32.TRUNC.NTZ`; after adding `UF2I` to the simulator opcode mapping and forcing the shared trace utility to rebuild, bounded replay reached the 50000-cycle cap and emitted canonical frontend timing and redundancy JSON.

Canonical artifacts: `frontend_timing_breakdown_bert-base-pretraining-full-step.json` and `redundancy_profile_bert-base-pretraining-full-step.json`.

## task-E2: Llama 3.1 8B Decoder-Layer Slice

Status: trace export and bounded simulator replay passed.

Evidence: `llama_decoder_layer_attempt.json`.

The new `docs/llama-decoder-layer-training.py` harness validates without NVBit at batch 1, sequence length 8, hidden size 4096, and intermediate size 14336, peaking at 864.0 MiB. NVBit export completed and produced `/tmp/llama_decoder_layer_trace_r2/traces` with 6.4 GB of trace artifacts. Bounded replay reached the 50000-cycle cap with `gpu_tot_sim_insn = 12330848` and `gpgpu_simulation_time = 205s`.

## bitlesson-selector

`bitlesson-selector` was not found on PATH or under `/home/dyf`. Round 3 applied `BL-20260503-repo-structure` manually by verifying filesystem paths and trace artifacts directly.
