import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DenseTransformerConfig:
    vocab_size: int = 8192
    # Parameter-matched to MoEUT (~42M) per the MoEUT paper (arXiv:2405.16039, §3):
    # the dense model is the anchor; MoEUT sets H_moeut = H/4, head_dim_moeut = 2*head_dim.
    n_layers: int = 8
    model_dim: int = 512
    n_heads: int = 16
    head_dim: int = 64          # n_heads*head_dim = 1024 attention inner dim
    ff_dim: int = 2048          # 4 * model_dim (standard Transformer ratio)
    max_seq_len: int = 2048
    rope_base: int = 10_000


BASELINE_CONFIG = DenseTransformerConfig()

DEBUG_CONFIG = DenseTransformerConfig(
    n_layers=2,
    model_dim=128,
    n_heads=2,
    head_dim=64,
    ff_dim=256,
    max_seq_len=256,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.size(-1) // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10_000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)            # (max_seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)     # (max_seq_len, head_dim)
        self.register_buffer("sin_table", emb.sin())
        self.register_buffer("cos_table", emb.cos())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, T, head_dim)
        T = x.size(2)
        sin = self.sin_table[:T]
        cos = self.cos_table[:T]
        return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DenseTransformerConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        inner = cfg.n_heads * cfg.head_dim
        self.qkv = nn.Linear(cfg.model_dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, cfg.model_dim, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.n_heads * self.head_dim, dim=-1)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).contiguous().view(B, T, -1))


class FeedForward(nn.Module):
    def __init__(self, cfg: DenseTransformerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.model_dim, cfg.ff_dim, bias=False)
        self.fc2 = nn.Linear(cfg.ff_dim, cfg.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DenseTransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.model_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.model_dim)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DenseTransformer(nn.Module):
    def __init__(self, cfg: DenseTransformerConfig = BASELINE_CONFIG) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.model_dim)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) → logits: (B, T, vocab_size)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.lm_head.weight.numel()
