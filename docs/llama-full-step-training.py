#!/usr/bin/env python3
"""Llama 3.1 8B synthetic full-step harness for resource-bound validation."""
import os

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


def getenv_int(name, default):
    return int(os.environ.get(name, str(default)))


def main():
    torch.manual_seed(20260505)
    torch.cuda.init()

    batch = getenv_int("LLAMA_FULL_TRACE_BATCH", 1)
    seq_len = getenv_int("LLAMA_FULL_TRACE_SEQ_LEN", 8)
    vocab_size = getenv_int("LLAMA_FULL_TRACE_VOCAB", 128256)

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=max(seq_len, 64),
        rms_norm_eps=1e-5,
        use_cache=False,
    )

    model = LlamaForCausalLM(config).cuda().to(dtype=torch.float16).train()
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")
    labels = input_ids.clone()

    output = model(input_ids=input_ids, labels=labels, use_cache=False)
    loss = output.loss
    loss.backward()
    torch.cuda.synchronize()

    peak_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(
        "Llama full-step training complete: "
        f"batch={batch} seq_len={seq_len} loss={loss.item():.6f} "
        f"peak_alloc_mib={peak_mib:.1f}"
    )


if __name__ == "__main__":
    main()
