from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.source import (
    AdaptiveMoEUTLM,
    AdaptiveMoEUTOutput,
    AttentionMask,
    MultilayerKVCache
)

@dataclass
class AdaptiveMoEUTConfig():
    vocab_size: int = 8192
    max_loops: int = 4          # the network will be interrupted at max_loops even if not all tokens have halted
                                # 4 loops x group_size=2 layers = 8 layer-passes = exactly the MoEUT n_layers=8
                                # baseline's depth (compute-matched), and ~2.5x faster than max_loops=10.
    group_size: int = 2         # unique layers
    model_dim: int = 512
    n_heads: int = 4
    head_dim: int = 128
    # SwitchHead attention MoE
    att_n_experts: int = 3      # N_A  (param-matched; ~12.5% of non-embed params in attention)
    att_k: int = 2              # K_A  (active attention experts per head)
    # SigmaMoE feed-forward
    ff_n_experts: int = 112     # N_E  (param-matched to the dense baseline)
    ff_expert_size: int = 128   # d_expert
    ff_k: int = 8               # K = 2 * model_dim / ff_expert_size
    # regularization (paper: minimal entropy reg stabilizes MoE training)
    entropy_reg: float = 0.01
    att_entropy_reg: float = 0.001
    dropout: float = 0.0
    max_seq_len: int = 2048
    # halting params
    init_halt_bias: float = -3  # initial bias in the halting head (deep start to let it explore longer depths); -3 -> ~5% halting prob
    default_halt_thresh: float = 0.999  # default cumulative halt prob threshold above which the token will be immediately halted

ADAPTIVE_MOEUT_CONFIG = AdaptiveMoEUTConfig()

DEBUG_CONFIG = AdaptiveMoEUTConfig(
    group_size=1, model_dim=64, n_heads=4, head_dim=16,
    att_n_experts=2, ff_n_experts=2, ff_expert_size=64, ff_k=1, att_k=1,
    max_loops=5
)

class AdaptiveMoEUT(nn.Module):
  
    def __init__(self, cfg: AdaptiveMoEUTConfig = ADAPTIVE_MOEUT_CONFIG):
        super().__init__()

        self.net = AdaptiveMoEUTLM(
            n_tokens=cfg.vocab_size,
            d_model=cfg.model_dim,
            max_loops=cfg.max_loops,
            n_heads=cfg.n_heads,
            ff_n_experts=cfg.ff_n_experts,
            att_n_experts=cfg.att_n_experts,
            d_head=cfg.head_dim,
            group_size=cfg.group_size,
            ff_k=cfg.ff_k,
            att_k=cfg.att_k,
            ff_expert_size=cfg.ff_expert_size,
            dropout=cfg.dropout,
            entropy_reg=cfg.entropy_reg,
            att_entropy_reg=cfg.att_entropy_reg,
            init_halt_bias=cfg.init_halt_bias,
            default_halt_thresh=cfg.default_halt_thresh
        )
        self.net.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        self.net.lm_head.weight = self.net.embedding.weight
        self.last_reg_loss: torch.Tensor = torch.zeros(())

        self.max_loops = cfg.max_loops

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None, kv_cache: MultilayerKVCache = None) -> tuple[torch.Tensor, AdaptiveMoEUTOutput]:
        out = self.net(x, kv_cache)
        self.last_reg_loss = out.reg_loss.squeeze()
        return out.outputs, out