# 01 - Base Transformer (Toy, from scratch)

This module implements a minimal **decoder-only** Transformer language model using the core 2017 Transformer components:

- Multi-head self-attention
- Position-wise feed-forward network
- Residual connections + LayerNorm
- Sinusoidal positional encoding

Why decoder-only? The original paper introduced an encoder-decoder model for translation, but decoder-only is the simplest setting for language modeling and still teaches the main architecture.

## Files

- `model.py`: all architecture components.
- `train.py`: tiny character-level language model training script.

## Architecture map (paper -> code)

1. Token embeddings + positional encoding
   - `TinyTransformerLM.token_embed`
   - `SinusoidalPositionalEncoding`
2. Repeated Transformer block
   - `TransformerBlock`
   - Self-attention: `MultiHeadSelfAttention`
   - FFN: `PositionwiseFFN`
3. Final projection to vocabulary logits
   - `TinyTransformerLM.lm_head`

## Block equations (Post-LN)

For hidden states `x`:

1. `x = LayerNorm(x + MHA(x))`
2. `x = LayerNorm(x + FFN(x))`

In this implementation, that is exactly `TransformerBlock.forward`.

## Run

From this folder:

```bash
python train.py
```

Optional quick run:

```bash
python train.py --steps 200 --eval-interval 50
```

By default it reads `../../nanoGPT/data/input.txt` (tiny Shakespeare text).

## What to inspect first

1. `MultiHeadSelfAttention.forward` in `model.py`
2. The causal mask (`torch.tril`) and where it is applied
3. `train.py` batching (`get_batch`) and next-token loss

## Suggested exercises

1. Remove the causal mask and observe behavior (it should leak future tokens).
2. Switch to learned positional embeddings and compare.
3. Change `n_heads`, `d_model`, and `n_layers` to see parameter/performance tradeoffs.
4. Add a learning-rate warmup schedule.
