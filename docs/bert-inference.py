#!/usr/bin/env python3
"""BERT-base encoder layer training harness for NVBit trace generation.

Runs one encoder layer with forward+backward pass on random input,
simulating a pretraining step for claim-bearing AI-training trace acquisition.
Runs under NVBit via LD_PRELOAD to generate .pb trace files.
"""
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.configuration_bert import BertConfig

def run_bert_encoder_layer_training():
    config = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    layer = BertLayer(config).cuda().train()

    dummy_input = torch.randn(1, 128, 768, device="cuda", requires_grad=True)
    attention_mask = torch.ones(1, 1, 1, 128, device="cuda")

    output = layer(dummy_input, attention_mask)
    loss = output[0].sum()
    loss.backward()

    print(f"Training loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_bert_encoder_layer_training()
