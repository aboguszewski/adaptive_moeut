# Finds dense transformer configurations whose forward-pass FLOP cost matches MoEUT (iso-FLOP baseline).
#
# Strategy: fix HEAD_DIM and the FEEDFORWARD_DIM:MODEL_DIM ratio, then binary-search over MODEL_DIM
# (restricted to multiples of HEAD_DIM so ATTENTION_HEADS = MODEL_DIM // HEAD_DIM stays integer)
# for each candidate number of transformer blocks.

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import mouet_math as moeut

# --- Parametric dense transformer costs ---
# Op costs and flags are taken from mouet_math to keep the comparison apples-to-apples.

E_EXP_COST = moeut.E_EXP_COST
SQRT_COST = moeut.SQRT_COST
PRECOMP_MODEL_DIM_INVERSE = moeut.PRECOMP_MODEL_DIM_INVERSE
PRECOMP_HEAD_DIM_SQRT_INVERSE = moeut.PRECOMP_HEAD_DIM_SQRT_INVERSE
PRECOMP_ROPE_ANGLES = moeut.PRECOMP_ROPE_ANGLES

HEAD_DIM = moeut.HEAD_DIM      # fixed; ATTENTION_HEADS = MODEL_DIM // HEAD_DIM
FF_RATIO = 2                    # FEEDFORWARD_DIM = FF_RATIO * MODEL_DIM (matches transformer_math.py default)
VOCAB_SIZE = moeut.VOCAB_SIZE


def _layernorm(seq_len: int, model_dim: int) -> int:
    mean_cost = model_dim + (not PRECOMP_MODEL_DIM_INVERSE)
    variance_cost = 3 * model_dim + (not PRECOMP_MODEL_DIM_INVERSE)
    normalization_cost = model_dim + SQRT_COST + 2
    scaling_cost = 2 * model_dim
    return seq_len * (mean_cost + variance_cost + normalization_cost + scaling_cost)


def _feedforward(seq_len: int, model_dim: int) -> int:
    ff_dim = FF_RATIO * model_dim
    in_proj = seq_len * ff_dim * (2 * model_dim - 1)
    relu = seq_len * ff_dim
    out_proj = seq_len * model_dim * (2 * ff_dim - 1)
    return in_proj + relu + out_proj


def _singlehead_attention(seq_len: int, model_dim: int) -> int:
    qkv_proj = 3 * seq_len * HEAD_DIM * (2 * model_dim - 1)

    # Full-rotation RoPE (rotate_fraction=1.0, matching transformer_math.py)
    rope_rotation_cost = 3 * seq_len * HEAD_DIM + seq_len * (HEAD_DIM // 2)
    rope_cost = 2 * rope_rotation_cost if PRECOMP_ROPE_ANGLES else (
        seq_len * (HEAD_DIM // 2) + 2 * rope_rotation_cost
    )

    qk_t = seq_len**2 * (2 * HEAD_DIM - 1)
    qk_scale = seq_len**2
    softmax = seq_len * (seq_len * (E_EXP_COST + 4) - 1)
    values = seq_len * HEAD_DIM * (2 * seq_len - 1)

    return qkv_proj + rope_cost + qk_t + qk_scale + softmax + values


def _multihead_attention(seq_len: int, model_dim: int) -> int:
    n_heads = model_dim // HEAD_DIM
    out_proj = seq_len * model_dim * (2 * n_heads * HEAD_DIM - 1)
    return n_heads * _singlehead_attention(seq_len, model_dim) + out_proj


def _block(seq_len: int, model_dim: int) -> int:
    return (_layernorm(seq_len, model_dim)
            + _multihead_attention(seq_len, model_dim)
            + seq_len * model_dim                       # residual
            + _layernorm(seq_len, model_dim)
            + _feedforward(seq_len, model_dim)
            + seq_len * model_dim)                      # residual


def dense_forward_pass_cost(seq_len: int, model_dim: int, n_blocks: int) -> int:
    # Include out_norm LayerNorm before lm_head to match MoEUT's architecture.
    out_norm = _layernorm(seq_len, model_dim)
    lm_head = seq_len * VOCAB_SIZE * (2 * model_dim - 1)
    return n_blocks * _block(seq_len, model_dim) + out_norm + lm_head


def _find_model_dim(seq_len: int, n_blocks: int, target: int) -> int:
    """Return the multiple of HEAD_DIM whose forward-pass cost is closest to target."""
    lo, hi = 1, 1024  # search range in units of HEAD_DIM

    # Find the largest multiple still under target
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if dense_forward_pass_cost(seq_len, mid * HEAD_DIM, n_blocks) < target:
            lo = mid
        else:
            hi = mid

    # Pick whichever neighbour is closer
    cost_lo = dense_forward_pass_cost(seq_len, lo * HEAD_DIM, n_blocks)
    cost_hi = dense_forward_pass_cost(seq_len, hi * HEAD_DIM, n_blocks)
    return (lo if abs(cost_lo - target) <= abs(cost_hi - target) else hi) * HEAD_DIM


def main(seq_len: int) -> None:
    target = moeut.forward_pass_cost(seq_len)
    print(f"MoEUT target  ({moeut.N_LAYERS} layers, d={moeut.MODEL_DIM}, "
          f"{moeut.FF_N_EXPERTS} experts, k={moeut.FF_K}): {moeut.format_flops(target)}")
    print()
    print(f"{'Blocks':>8}  {'MODEL_DIM':>10}  {'N_HEADS':>8}  {'FF_DIM':>8}  "
          f"{'FLOPs':>18}  {'Δ vs target':>12}")
    print("-" * 78)

    for n_blocks in [2, 4, 6, 8, 12, 16]:
        dim = _find_model_dim(seq_len, n_blocks, target)
        cost = dense_forward_pass_cost(seq_len, dim, n_blocks)
        n_heads = dim // HEAD_DIM
        err_pct = (cost - target) / target * 100
        print(f"{n_blocks:>8}  {dim:>10}  {n_heads:>8}  {FF_RATIO * dim:>8}  "
              f"{moeut.format_flops(cost):>18}  {err_pct:>+11.1f}%")


# GPU specs: (label, total_fp16_tflops, mfu)
# Titan V / Titan RTX have no NVLink; multi-GPU runs over PCIe, so MFU drops from ~35% to ~20%.
GPU_CONFIGS = [
    ("1× Titan V",    110,       0.35),
    ("1× Titan RTX",  130,       0.35),
    ("2× Titan V",    2 * 110,   0.20),
    ("2× Titan RTX",  2 * 130,   0.20),
    ("4× Titan V",    4 * 110,   0.20),
    ("4× Titan RTX",  4 * 130,   0.20),
    ("8× Titan V",    8 * 110,   0.20),
    ("8× Titan RTX",  8 * 130,   0.20),
]


def training_time_table(seq_len: int) -> None:
    # Training FLOPs ≈ 3× forward pass (forward + 2× backward, no activation checkpointing).
    # Wall-clock time = total_flops / (effective_tflops × 1e12)
    target = moeut.forward_pass_cost(seq_len)
    models = [
        ("MoEUT",          target),
        ("Dense 4-block",  dense_forward_pass_cost(seq_len, _find_model_dim(seq_len, 4, target), 4)),
        ("Dense 12-block", dense_forward_pass_cost(seq_len, _find_model_dim(seq_len, 12, target), 12)),
    ]

    flops_per_token = {name: 3 * fwd / seq_len for name, fwd in models}

    print(f"\nHours per billion training tokens  (seq_len={seq_len:,}, 3× forward, no activation checkpointing)\n")
    col = 16
    header = f"{'Model':<16}" + "".join(f"{g[0]:>{col}}" for g in GPU_CONFIGS)
    print(header)
    print("-" * (16 + col * len(GPU_CONFIGS)))

    for name, _ in models:
        row = f"{name:<16}"
        for _, tflops, mfu in GPU_CONFIGS:
            effective = tflops * mfu * 1e12  # FLOPs/s
            h_per_bt = (flops_per_token[name] * 1e9) / effective / 3600
            row += f"{h_per_bt:>{col}.1f}h"
        print(row)

    print(f"\nMulti-GPU MFU set to 20% (PCIe, no NVLink); single-GPU 35%.")
    print(f"Multiply by your token budget in billions to get total wall-clock hours.")


# ---- Parameter counts ----

def _params_dense(model_dim: int, n_blocks: int) -> int:
    ff_dim = FF_RATIO * model_dim
    per_block = (2 * model_dim            # ln1 weight + bias
                 + 4 * model_dim**2       # Q, K, V, O projections (no bias)
                 + 2 * model_dim          # ln2 weight + bias
                 + model_dim * ff_dim     # FFN in (no bias)
                 + ff_dim * model_dim)    # FFN out (no bias)
    return (n_blocks * per_block
            + 2 * VOCAB_SIZE * model_dim + VOCAB_SIZE   # embedding + lm_head weight + bias
            + 2 * model_dim)                            # out_norm weight + bias


def _params_moeut() -> int:
    m = moeut
    per_layer = (
        2 * m.MODEL_DIM                                                     # ln1
        + m.MODEL_DIM * m.N_HEADS * m.HEAD_DIM                             # Q (no bias)
        + m.MODEL_DIM * m.N_HEADS * m.HEAD_DIM                             # K (no bias)
        + m.N_HEADS * m.ATT_N_EXPERTS * m.MODEL_DIM * m.HEAD_DIM          # v
        + m.N_HEADS * m.ATT_N_EXPERTS * m.HEAD_DIM * m.MODEL_DIM          # o
        + m.N_HEADS * m.ATT_N_EXPERTS * m.MODEL_DIM                        # sel_v
        + m.N_HEADS * m.ATT_N_EXPERTS * m.MODEL_DIM                        # sel_o
        + m.FF_N_EXPERTS * m.MODEL_DIM * m.FF_EXPERT_SIZE                  # keys
        + m.FF_N_EXPERTS * m.FF_EXPERT_SIZE * m.MODEL_DIM                  # values
        + m.FF_N_EXPERTS * m.MODEL_DIM                                      # expert_sel
        + 2 * m.MODEL_DIM                                                   # ln2
    )
    return (m.GROUP_SIZE * per_layer
            + 2 * VOCAB_SIZE * m.MODEL_DIM + VOCAB_SIZE  # embedding + lm_head weight + bias
            + 2 * m.MODEL_DIM)                           # out_norm weight + bias


# ---- Memory estimates ----

def _static_memory_gb(n_params: int) -> float:
    # Mixed-precision Adam: fp16 params (2) + fp16 grads (2) + fp32 master (4) + fp32 m (4) + fp32 v (4) = 16 bytes
    return n_params * 16 / 1e9


def _activation_gb_per_sample(seq_len: int, model_dim: int, n_heads: int, n_layers: int) -> float:
    # Per layer (fp16, no activation checkpointing, no FlashAttention):
    #   ~9 copies of the residual stream (input, LN outputs, Q/K/V, attn out, FFN in/out)
    #   + attention matrix (n_heads × L²)
    per_layer = 9 * seq_len * model_dim * 2 + n_heads * seq_len**2 * 2
    return n_layers * per_layer / 1e9


def memory_table(seq_len: int, available_gb: float) -> None:
    target = moeut.forward_pass_cost(seq_len)
    d4  = _find_model_dim(seq_len, 4, target)
    d12 = _find_model_dim(seq_len, 12, target)

    models = [
        ("MoEUT",          _params_moeut(),       moeut.MODEL_DIM, moeut.N_HEADS, moeut.N_LAYERS),
        ("Dense 4-block",  _params_dense(d4, 4),  d4,  d4  // HEAD_DIM, 4),
        ("Dense 12-block", _params_dense(d12, 12), d12, d12 // HEAD_DIM, 12),
    ]

    print(f"\nMemory breakdown  (seq_len={seq_len:,}, available={available_gb:.0f} GB)\n")
    print(f"{'Model':<18}  {'Params':>10}  {'Static (GB)':>12}  "
          f"{'Activ/sample (GB)':>18}  {'Max batch':>10}")
    print("-" * 76)

    for name, n_params, model_dim, n_heads, n_layers in models:
        static = _static_memory_gb(n_params)
        activ  = _activation_gb_per_sample(seq_len, model_dim, n_heads, n_layers)
        max_bs = max(1, int((available_gb - static) / activ))
        print(f"{name:<18}  {n_params/1e6:>8.1f}M  {static:>12.2f}  {activ:>18.2f}  {max_bs:>10}")

    print(f"\nActivation memory assumes fp16, no checkpointing, no FlashAttention.")
    print(f"FlashAttention removes the O(L²) term; checkpointing reduces residual activations.")


if __name__ == "__main__":
    main(8_000)
    memory_table(8_000, available_gb=60)
    training_time_table(8_000)
