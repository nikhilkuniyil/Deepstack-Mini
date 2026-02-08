import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.1
    use_gqa: bool = False
    n_kv_heads: int | None = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Take even-indexed features from the last dimension.
    x1 = x[..., ::2]
    # Take odd-indexed features from the last dimension.
    x2 = x[..., 1::2]
    # Convert each pair (a, b) -> (-b, a), a 90-degree 2D rotation.
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh), cos/sin: (1, 1, T, Dh)
    # RoPE formula: x_rot = x * cos(theta) + rotate_half(x) * sin(theta).
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 block_size: int, 
                 dropout: float,
                 use_gqa: bool = False,
                 n_kv_heads: int | None = None,):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_gqa = use_gqa
        self.n_q_heads = n_heads
        self.n_kv_heads = n_kv_heads if (use_gqa and n_kv_heads is not None) else n_heads

        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads for GQA")

        self.kv_group_size = self.n_q_heads // self.n_kv_heads

        # RoPE rotates 2-feature pairs, so per-head dimension must be even.
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        # Position indices: 0..T-1.
        pos = torch.arange(block_size, dtype=torch.float32)  # (T,)
        # One frequency bucket per feature pair.
        freq = torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        # Inverse frequencies define how fast each pair rotates with position.
        inv_freq = 1.0 / (10000 ** freq)  # (Dh/2,)

        # Base angles for each (position, feature-pair).
        angles = torch.outer(pos, inv_freq)  # (T, Dh/2)
        # Duplicate each pair-angle so tensor width matches Dh.
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)  # (T, Dh)

        # Cache cos/sin terms for fast lookup in forward.
        self.register_buffer("rope_cos", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("rope_sin", angles.sin()[None, None, :, :], persistent=False)

        # Queries keep full head count (Hq) for expressivity
        self.q_proj = nn.Linear(d_model, d_model)

        # Keys use fewr heads (Hkv) in GQA, else same as HQ
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim)

        # Values use the same reduced head count as keys
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim)

        # Output proejction always maps back to model width D.
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape

        # Project Q to full model width D, then split into Hq heads
        # Shape: (B, T, D) -> (B, T, Hq, Dh) -> (B, Hq, T, Dh)
        q = self.q_proj(x).view(bsz, seq_len, self.n_q_heads, self.head_dim).transpose(1, 2)

        # Project K, V to reduced KV width (Hkv * Dh), then split into Hkv heads
        # Shape: (B, T, Hkv*Dh) -> (B, T, Hkv, Dh) -> (B, Hkv, T, Dh)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Slice precomputed RoPE angles to current sequence length.
        cos = self.rope_cos[:, :, :seq_len, :]
        sin = self.rope_sin[:, :, :seq_len, :]
        # Apply RoPE only to Q and K (not V), before attention score computation.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Attention logits: (B, H, T, T) = QK^T / sqrt(Dh).
        att = (q @ k.transpose(-2, -1)) * self.scale
        # Causal mask blocks attention to future positions (j > i).
        att = att.masked_fill(~self.causal_mask[:seq_len, :seq_len], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum over values, then merge heads back to (B, T, D).
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        y = self.out_proj(y)
        return self.resid_dropout(y)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # Independent MLP applied at each position.
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # Buld attention with optional GQA behavior
        self.attn = MultiHeadSelfAttention(
            # Model width D
            d_model=config.d_model,
            # Query head count Hq
            n_heads=config.n_heads,
            # Max context length for mask/RoPE cache
            block_size=config.block_size,
            # Dropout used in attention probs and residual projection
            dropout=config.dropout,
            # Feature flag: switch GQA on/off
            use_gqa=config.use_gqa,
            # KV head count when GQA is on (ignored when off)
            n_kv_heads=config.n_kv_heads,
        )
        self.ffn = PositionwiseFFN(config.d_model, config.d_ff, config.dropout)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-LN: normalize after each residual add.
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if idx.size(1) > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        # Token ids (B, T) -> token embeddings (B, T, D).
        # Positional information now comes from RoPE inside attention.
        x = self.token_embed(idx)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        # Vocabulary logits per position: (B, T, V).
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Next-token objective over all time steps in the batch.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Keep only the most recent context that fits the model window.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
