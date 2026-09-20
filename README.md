# K-GPT local inference test

This folder contains the first local inference layer for K-GPT.

Expected checkpoint:
`../model/weights/gpt2-medium355M-sft.pth`

Files:
- `model/model.py`: GPT-2 Medium architecture matching the training checkpoint
- `model/generate.py`: generation/token conversion utilities
- `model/tokenizer.py`: GPT-2 tokenizer
- `run_local.py`: loads the checkpoint on CUDA when available and generates a response

Run from the K-GPT project root:

    python run_local.py

The model configuration is GPT-2 Medium:
- vocab_size: 50257
- context_length: 1024
- emb_dim: 1024
- n_layers: 24
- n_heads: 16
- qkv_bias: True
