#!/usr/bin/env python3
"""BERT-base synthetic pretraining full-step harness for NVBit tracing."""
import os

import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForPreTraining


def getenv_int(name, default):
    return int(os.environ.get(name, str(default)))


def main():
    torch.manual_seed(20260504)
    torch.cuda.init()

    batch = getenv_int("BERT_TRACE_BATCH", 1)
    seq_len = getenv_int("BERT_TRACE_SEQ_LEN", 16)
    vocab_size = getenv_int("BERT_TRACE_VOCAB", 30522)

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    model = BertForPreTraining(config).cuda().train()

    input_ids = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long, device="cuda")
    token_type_ids = torch.zeros((batch, seq_len), dtype=torch.long, device="cuda")
    labels = input_ids.clone()
    next_sentence_label = torch.zeros((batch,), dtype=torch.long, device="cuda")

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels,
        next_sentence_label=next_sentence_label,
    )
    loss = output.loss
    loss.backward()
    torch.cuda.synchronize()

    peak_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(
        "BERT full-step training complete: "
        f"batch={batch} seq_len={seq_len} loss={loss.item():.6f} "
        f"peak_alloc_mib={peak_mib:.1f}"
    )


if __name__ == "__main__":
    main()
