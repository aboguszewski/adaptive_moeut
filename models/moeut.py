"""Parameter-matched MoEUT model.

Thin wrapper around the official MoEUT implementation (utils/moute_code.py +
utils/cvmm.py from https://github.com/RobertCsordas/moeut) that:
  - fixes the architecture to the dense-baseline-matched config (~37.9M params,
    see utils/param_match.py and the MoEUT paper arXiv:2405.16039 §3), and
  - exposes the same interface as DenseTransformer: forward(x) -> logits, with
    the MoE entropy regularization stashed on `.last_reg_loss` for the training
    loop to add to the cross-entropy loss.

NOTE: the forward pass uses a Triton kernel (cvmm) and therefore requires a GPU.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from moeut_code import MoEUTLM


@dataclass
class MoEUTConfig:
    vocab_size: int = 8192
    n_layers: int = 8           # total layers (= group_size * n_repeats)
    group_size: int = 2         # unique layers; shared/repeated n_layers//group_size times
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


# Parameter-matched to DenseTransformer BASELINE_CONFIG (~37.9M). See utils/param_match.py.
MOEUT_CONFIG = MoEUTConfig()

# Tiny config for sanity checks.
DEBUG_CONFIG = MoEUTConfig(
    n_layers=2, group_size=1, model_dim=128, n_heads=2, head_dim=64,
    att_n_experts=2, ff_n_experts=8, ff_expert_size=64, ff_k=2,
)


class MoEUT(nn.Module):
    def __init__(self, cfg: MoEUTConfig = MOEUT_CONFIG) -> None:
        super().__init__()
        self.net = MoEUTLM(
            n_tokens=cfg.vocab_size,
            d_model=cfg.model_dim,
            n_layers=cfg.n_layers,
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
        )
        # Tie lm_head to the embedding and drop its bias, matching DenseTransformer
        # so the two models stay parameter-matched.
        self.net.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        self.net.lm_head.weight = self.net.embedding.weight

        self.last_reg_loss: torch.Tensor = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # reg_loss is a 1-element tensor (entropy reg summed over layers); 0 in eval.
        self.last_reg_loss = out.reg_loss.squeeze()
        return out.outputs
