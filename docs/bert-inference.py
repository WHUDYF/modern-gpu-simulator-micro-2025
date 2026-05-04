#!/usr/bin/env python3
"""BERT-base encoder inference harness for NVBit trace generation.

Used for claim-bearing AI-training trace acquisition on RTX 5090.
Run under NVBit via LD_PRELOAD to generate .pb trace files.
"""
import torch
from transformers import BertModel

def run_bert():
    model = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
    dummy_input = torch.randint(0, 30522, (1, 128)).cuda()
    attention_mask = torch.ones(1, 128, dtype=torch.long).cuda()

    with torch.no_grad():
        output = model(input_ids=dummy_input, attention_mask=attention_mask)

    print(f"BERT-base output shape: {output.last_hidden_state.shape}")

if __name__ == "__main__":
    run_bert()
