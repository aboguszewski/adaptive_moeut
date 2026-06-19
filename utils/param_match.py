# Parameter-matching between the dense baseline and MoEUT, per MoEUT paper §3
# (arXiv:2405.16039). The DENSE model is the anchor; MoEUT is constructed to
# match its total parameter count.
#
# Protocol:
#   - MoEUT uses the same d_model and n_layers as the dense baseline.
#   - H_moeut = H_dense / 4,  head_dim_moeut = 2 * head_dim_dense,  K_A = 2.
#   - d_expert = 128,  K = 2 * d_model / d_expert.
#   - N_E (FFN experts) and N_A (attention experts) are chosen so total params
#     match the dense baseline, with ~10-15% of non-embedding params in attention.
#
# Run:  python utils/param_match.py
# It reads the dense BASELINE_CONFIG and reports the matched MoEUT config, and
# also verifies that the constants currently in mouet_math.py are matched.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.dense_transformer import BASELINE_CONFIG

VOCAB = BASELINE_CONFIG.vocab_size


def dense_params(d, L, H, dh, ff, vocab=VOCAB):
    inner = H * dh
    per = 2 * d + 4 * d * inner + 2 * d + 2 * d * ff   # ln1, QKVO, ln2, FFN
    return L * per + vocab * d + 2 * d   # tied embedding/lm_head (see DenseTransformer)              # embed + lm_head, out_norm


def moeut_params(d, L, G, H, dh, NE, dexp, NA, vocab=VOCAB):
    per = (2 * d                       # ln1
           + d * H * dh                # Q
           + d * H * dh                # K
           + H * NA * d * dh           # V (per-head experts)
           + H * NA * dh * d           # O
           + H * NA * d                # sel_v
           + H * NA * d                # sel_o
           + NE * d * dexp             # FFN expert keys
           + NE * dexp * d             # FFN expert values
           + NE * d                    # expert selection
           + 2 * d)                    # ln2
    return G * per + vocab * d + 2 * d   # tied embedding/lm_head (see DenseTransformer)


def attn_fraction(d, L, G, H, dh, NE, dexp, NA):
    attn = 2 * d * H * dh + 2 * H * NA * d * dh + 2 * H * NA * d
    ff = 2 * NE * d * dexp + NE * d
    return attn / (attn + ff)


def match(dense_cfg, group_size, dexp=128, attn_target=(0.10, 0.15)):
    d, L = dense_cfg.model_dim, dense_cfg.n_layers
    target = dense_params(d, L, dense_cfg.n_heads, dense_cfg.head_dim, dense_cfg.ff_dim)

    H_m = dense_cfg.n_heads // 4
    dh_m = dense_cfg.head_dim * 2
    K = 2 * d // dexp
    assert H_m >= 1, "dense n_heads must be a multiple of 4 (MoEUT uses H/4 heads)"

    # Pick N_A whose attention fraction lands mid-range, then N_E to match params.
    best = None
    for NA in range(1, 16):
        NE = min(
            range(1, 8192),
            key=lambda n: abs(moeut_params(d, L, group_size, H_m, dh_m, n, dexp, NA) - target),
        )
        frac = attn_fraction(d, L, group_size, H_m, dh_m, NE, dexp, NA)
        score = 0 if attn_target[0] <= frac <= attn_target[1] else min(
            abs(frac - attn_target[0]), abs(frac - attn_target[1])
        )
        if best is None or score < best[0]:
            best = (score, NA, NE, frac)
    _, NA, NE, frac = best
    p = moeut_params(d, L, group_size, H_m, dh_m, NE, dexp, NA)
    return dict(d=d, L=L, G=group_size, H=H_m, head_dim=dh_m, K_A=2,
                d_expert=dexp, K=K, N_A=NA, N_E=NE,
                params=p, target=target, attn_frac=frac)


if __name__ == "__main__":
    c = BASELINE_CONFIG
    G = 2 if dense_params(c.model_dim, c.n_layers, c.n_heads, c.head_dim, c.ff_dim) < 300e6 else 4
    r = match(c, group_size=G)

    print("Dense baseline (anchor):")
    print(f"  d={c.model_dim} L={c.n_layers} H={c.n_heads} head_dim={c.head_dim} "
          f"ff={c.ff_dim}  ->  {r['target']/1e6:.2f}M params\n")
    print("Parameter-matched MoEUT:")
    print(f"  d={r['d']} L={r['L']} G={r['G']} H={r['H']} head_dim={r['head_dim']} "
          f"K_A={r['K_A']} d_expert={r['d_expert']} K={r['K']}")
    print(f"  N_A (ATT_N_EXPERTS)={r['N_A']}  N_E (FF_N_EXPERTS)={r['N_E']}")
    print(f"  -> {r['params']/1e6:.2f}M params "
          f"({(r['params']-r['target'])/r['target']*100:+.1f}% vs dense), "
          f"attention = {r['attn_frac']*100:.1f}% of non-embedding params\n")

    # Cross-check the constants currently committed in mouet_math.py
    import utils.mouet_math as m
    cur = moeut_params(m.MODEL_DIM, m.N_LAYERS, m.GROUP_SIZE, m.N_HEADS, m.HEAD_DIM,
                       m.FF_N_EXPERTS, m.FF_EXPERT_SIZE, m.ATT_N_EXPERTS)
    print(f"mouet_math.py current constants -> {cur/1e6:.2f}M params "
          f"({(cur-r['target'])/r['target']*100:+.1f}% vs dense baseline)")
