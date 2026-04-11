#!/usr/bin/env python3
import argparse
import ctypes
import time

import torch
from transformers import AutoModelForCausalLM


def load_cudart():
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("libcudart.so not found")


def profiler_start(cudart_lib):
    rc = cudart_lib.cudaProfilerStart()
    if rc != 0:
        raise RuntimeError(f"cudaProfilerStart failed: {rc}")


def profiler_stop(cudart_lib):
    rc = cudart_lib.cudaProfilerStop()
    if rc != 0:
        raise RuntimeError(f"cudaProfilerStop failed: {rc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--gen-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).cuda().eval()

    vocab_size = model.config.vocab_size
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, args.context_len),
        device="cuda",
        dtype=torch.long,
    )
    next_token = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, 1),
        device="cuda",
        dtype=torch.long,
    )

    cudart_lib = load_cudart()
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        _ = model(input_ids=next_token, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    torch.cuda.synchronize()

    t0 = time.time()
    profiler_start(cudart_lib)
    for _ in range(args.gen_tokens):
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    profiler_stop(cudart_lib)
    t1 = time.time()

    print(f"model={args.model}")
    print(f"context_len={args.context_len}")
    print(f"gen_tokens={args.gen_tokens}")
    print(f"decode_time_s={t1 - t0:.6f}")


if __name__ == "__main__":
    main()
