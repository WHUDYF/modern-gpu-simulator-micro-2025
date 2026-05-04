#!/usr/bin/env python3
"""Llama 3.1 8B decoder-layer synthetic training harness for NVBit tracing."""
import os

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding


def getenv_int(name, default):
    return int(os.environ.get(name, str(default)))


def main():
    torch.manual_seed(20260504)
    torch.cuda.init()

    batch = getenv_int("LLAMA_TRACE_BATCH", 1)
    seq_len = getenv_int("LLAMA_TRACE_SEQ_LEN", 8)
    hidden_size = getenv_int("LLAMA_TRACE_HIDDEN", 4096)
    intermediate_size = getenv_int("LLAMA_TRACE_INTERMEDIATE", 14336)

    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=max(seq_len, 64),
        rms_norm_eps=1e-5,
        use_cache=False,
    )

    dtype = torch.float16
    layer = LlamaDecoderLayer(config, layer_idx=0).cuda().to(dtype=dtype).train()
    rotary = LlamaRotaryEmbedding(config).cuda()

    hidden_states = torch.randn(
        batch, seq_len, hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    position_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(hidden_states, position_ids)

    output = layer(
        hidden_states,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        use_cache=False,
    )
    if isinstance(output, tuple):
        output = output[0]
    loss = output.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()

    peak_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(
        "Llama decoder-layer training complete: "
        f"batch={batch} seq_len={seq_len} hidden={hidden_size} "
        f"loss={loss.item():.6f} peak_alloc_mib={peak_mib:.1f}"
    )


if __name__ == "__main__":
    main()
