# Full Model Architecture:
#   Tokenization --> Embedding Layer --> N_LAYERS MoEUT Blocks --> LayerNorm --> LM Head (De-Embedding)
#
#   N_LAYERS = N_REPEATS * GROUP_SIZE, where GROUP_SIZE unique layers are repeated N_REPEATS times (weight sharing).
#
#   MoEUT Block (MoEUTLayer):
#       Input -------Residual--------> (+) -------Residual--------> (+) --> Output
#             --> LayerNorm (ln1) --> SwitchHeadRope Attention ----/
#                                         --> LayerNorm (ln2) --> SigmaMoE FFN --/
#
#   SwitchHeadRope Attention (ATT_N_EXPERTS > 1):
#       q_src (ln1(x)) --> Q projection --> scale --> RoPE (partial) --> \
#       k_src (ln1(x)) --> K projection --> scale --> RoPE (partial) --> Scaled Dot-Product Attention --> cvmm O projection
#       k_src          --> V selection (linear + sigmoid + topk) --> cvmm V projection ---------> /
#       q_src          --> O selection (linear + sigmoid + topk) --> (weights scale O projection output)
#
#   SigmaMoE FFN:
#       sel_input (ln2(x)) --> expert selection (linear + sigmoid + topk)
#       input (x) -----------> cvmm up-projection --> ReLU --> cvmm down-projection (weighted by selection scores)

# Assumptions:
# 1. Memory operations (concat, transpose, reshape, view) require 0 FLOPs.
# 2. All element-wise divisions are done by computing the inverse and multiplying.
# 3. Softmax uses the numerically stable version (same as transformer_math.py).
# 4. Attention is bidirectional (no causal masking FLOPs).
# 5. ReLU(x) where x is a scalar requires 1 FLOP.
# 6. TopK requires 0 FLOPs.
# 7. ATT_N_EXPERTS > 1 (MoE attention path in SwitchHeadCore).
# 8. Dropout is training-only; inference cost is 0.
# 9. cvmm down-projections weight each expert's output by the sigmoid selection score (1 multiply per output element per expert).

# Precomputation flags:
PRECOMP_MODEL_DIM_INVERSE = True      # 1/model_dim
PRECOMP_HEAD_DIM_SQRT = True          # sqrt(head_dim)
PRECOMP_HEAD_DIM_SQRT_INVERSE = True  # 1/sqrt(head_dim)

PRECOMP_ROPE_ANGLES = True            # RoPE angles R^{max_seq_len, n_rotate//2} precomputed
PRECOMP_ROPE_SIN = True               # sin(angles) precomputed (requires PRECOMP_ROPE_ANGLES)
PRECOMP_ROPE_COS = True               # cos(angles) precomputed (requires PRECOMP_ROPE_ANGLES)
PRECOMP_MAX_SEQ_LEN = 256_000         # Required if any PRECOMP_ROPE_* is True

# Op costs:
SIN_COST = 1
COS_COST = 1
E_EXP_COST = 1
SQRT_COST = 1
ARBITRARY_EXP_COST = 1

# Model architecture:
VOCAB_SIZE = 8192
N_LAYERS = 8           # Total transformer layers (= N_REPEATS * GROUP_SIZE)
GROUP_SIZE = 2          # Unique layers; weights shared and repeated N_LAYERS // GROUP_SIZE times
MODEL_DIM = 512         # d_model
N_HEADS = 4             # Attention heads (n_heads in SwitchHead)
HEAD_DIM = 128          # Projection size per head
ROTATE_FRACTION = 0.5   # Fraction of HEAD_DIM rotated by RoPE (n_rotate = int(ROTATE_FRACTION * HEAD_DIM))

# SwitchHead attention MoE:
ATT_N_EXPERTS = 3       # N_A: tuned so total params match the dense baseline (~42M) and
                        #      ~12.5% of non-embedding params sit in attention (paper target 10-15%)
ATT_K = 2               # Active experts per head per token (moe_k); K_A=2 per paper

# SigmaMoE FFN:
FF_N_EXPERTS = 112      # N_E: tuned for parameter-match with the dense baseline (see utils/param_match.py)
FF_EXPERT_SIZE = 128    # Hidden size per expert (d_expert=128 per paper)
FF_K = 8                # Active experts per token; K = 2*MODEL_DIM/FF_EXPERT_SIZE = 8


def layernorm_cost(seq_len: int) -> int:
    mean_cost = MODEL_DIM + (not PRECOMP_MODEL_DIM_INVERSE)
    variance_cost = 3 * MODEL_DIM + (not PRECOMP_MODEL_DIM_INVERSE)
    normalization_cost = MODEL_DIM + SQRT_COST + 2
    scaling_cost = 2 * MODEL_DIM
    return seq_len * (mean_cost + variance_cost + normalization_cost + scaling_cost)


def residual_addition_cost(seq_len: int) -> int:
    return seq_len * MODEL_DIM


def sigma_moe_cost(seq_len: int) -> int:
    # Expert selection: F.linear(sel_input, expert_sel), expert_sel: [FF_N_EXPERTS, MODEL_DIM]
    sel_linear_cost = seq_len * FF_N_EXPERTS * (2 * MODEL_DIM - 1)
    sel_sigmoid_cost = seq_len * FF_N_EXPERTS

    # Up-projection: cvmm(input, sel_indices, keys), keys: [FF_N_EXPERTS, MODEL_DIM, FF_EXPERT_SIZE]
    # FF_K active experts per token
    up_proj_cost = seq_len * FF_K * FF_EXPERT_SIZE * (2 * MODEL_DIM - 1)
    relu_cost = seq_len * FF_K * FF_EXPERT_SIZE

    # Down-projection: cvmm(scores, sel_indices, values), values: [FF_N_EXPERTS, FF_EXPERT_SIZE, MODEL_DIM]
    # FF_K active experts per token, each output scaled by its selection weight (reduction_weight = sel_val)
    down_proj_cost = seq_len * FF_K * MODEL_DIM * (2 * FF_EXPERT_SIZE - 1)
    sel_weight_scale_cost = seq_len * FF_K * MODEL_DIM

    return sel_linear_cost + sel_sigmoid_cost + up_proj_cost + relu_cost + down_proj_cost + sel_weight_scale_cost


def switchhead_rope_cost(seq_len: int) -> int:
    n_rotate = int(ROTATE_FRACTION * HEAD_DIM)

    # Q and K projections: nn.Linear(MODEL_DIM, N_HEADS * HEAD_DIM), no bias
    qk_proj_cost = 2 * seq_len * N_HEADS * HEAD_DIM * (2 * MODEL_DIM - 1)

    # Scale Q and K by a scalar derived from 1/sqrt(head_dim) (applied element-wise before attention)
    # scale = self.scale.sqrt() is one extra SQRT unless both sqrt and its inverse are precomputed
    scale_computation_cost = (not PRECOMP_HEAD_DIM_SQRT_INVERSE) * (SQRT_COST + 1) + SQRT_COST
    qk_scale_cost = 2 * seq_len * N_HEADS * HEAD_DIM + scale_computation_cost

    # V selection: F.linear(k_src, sel_v), sel_v: [N_HEADS * ATT_N_EXPERTS, MODEL_DIM]
    v_sel_linear_cost = seq_len * N_HEADS * ATT_N_EXPERTS * (2 * MODEL_DIM - 1)
    v_sel_sigmoid_cost = seq_len * N_HEADS * ATT_N_EXPERTS

    # O selection: F.linear(q_src, sel_o), sel_o: [N_HEADS * ATT_N_EXPERTS, MODEL_DIM]
    o_sel_linear_cost = seq_len * N_HEADS * ATT_N_EXPERTS * (2 * MODEL_DIM - 1)
    o_sel_sigmoid_cost = seq_len * N_HEADS * ATT_N_EXPERTS

    # V projection: cvmm(v_src, v_sel, self.v), v: [N_HEADS * ATT_N_EXPERTS, MODEL_DIM, HEAD_DIM]
    # ATT_K active experts per head -> N_HEADS * ATT_K active (head, expert) pairs per token
    v_proj_cost = seq_len * N_HEADS * ATT_K * HEAD_DIM * (2 * MODEL_DIM - 1)

    # RoPE: applied to first n_rotate dims of Q and K via apply_rot
    # rotate_half(x) negates second half: seq_len * N_HEADS * (n_rotate // 2) FLOPs
    # new_x = x * cos + rotate_half(x) * sin: 3 multiplications and 1 addition per element
    rope_rotation_cost = 3 * seq_len * N_HEADS * n_rotate + seq_len * N_HEADS * (n_rotate // 2)
    if PRECOMP_ROPE_ANGLES:
        rope_cost = 2 * rope_rotation_cost  # applied to both Q and K
    else:
        # angles: einsum("i,j->ij", t, inv_freq) -> [seq_len, n_rotate // 2]
        rope_angles_cost = seq_len * (n_rotate // 2)
        rope_sin_cost = (not PRECOMP_ROPE_SIN) * seq_len * n_rotate * SIN_COST
        rope_cos_cost = (not PRECOMP_ROPE_COS) * seq_len * n_rotate * COS_COST
        rope_cost = rope_angles_cost + rope_sin_cost + rope_cos_cost + 2 * rope_rotation_cost

    # Scaled dot-product attention (N_HEADS heads; scaling already absorbed into Q, K above)
    # QK^T: [N_HEADS, seq_len, HEAD_DIM] x [N_HEADS, HEAD_DIM, seq_len]
    qk_t_cost = N_HEADS * seq_len**2 * (2 * HEAD_DIM - 1)
    # softmax(R^{seq_len}) cost = seq_len * (E_EXP_COST + 4) - 1; applied per head per row
    softmax_cost = N_HEADS * seq_len * (seq_len * (E_EXP_COST + 4) - 1)
    # softmax(QK^T) x V: [N_HEADS, seq_len, seq_len] x [N_HEADS, seq_len, HEAD_DIM]
    values_cost = N_HEADS * seq_len * HEAD_DIM * (2 * seq_len - 1)

    # O projection: cvmm(res, o_sel, self.o), o: [N_HEADS * ATT_N_EXPERTS, HEAD_DIM, MODEL_DIM]
    # ATT_K active experts per head; selection weight scales each expert output (reduction_weight = sel_val)
    o_proj_cost = seq_len * N_HEADS * ATT_K * MODEL_DIM * (2 * HEAD_DIM - 1)
    o_weight_scale_cost = seq_len * N_HEADS * ATT_K * MODEL_DIM

    return (qk_proj_cost + qk_scale_cost
            + v_sel_linear_cost + v_sel_sigmoid_cost
            + o_sel_linear_cost + o_sel_sigmoid_cost
            + v_proj_cost
            + rope_cost
            + qk_t_cost + softmax_cost + values_cost
            + o_proj_cost + o_weight_scale_cost)


def moeut_layer_cost(seq_len: int) -> int:
    return (layernorm_cost(seq_len)           # ln1, used as Q/K input to attention
            + switchhead_rope_cost(seq_len)    # SwitchHeadRope attention
            + residual_addition_cost(seq_len)  # post-attention residual
            + layernorm_cost(seq_len)           # ln2, used as selection input to SigmaMoE
            + sigma_moe_cost(seq_len)           # SigmaMoE FFN
            + residual_addition_cost(seq_len))  # post-FFN residual


def deembedding_cost(seq_len: int) -> int:
    # out_norm LayerNorm before lm_head (MoEUTLM.forward)
    out_norm_cost = layernorm_cost(seq_len)
    # lm_head: Linear(MODEL_DIM, VOCAB_SIZE), no bias
    lm_head_cost = seq_len * VOCAB_SIZE * (2 * MODEL_DIM - 1)
    return out_norm_cost + lm_head_cost


def forward_pass_cost(seq_len: int) -> int:
    # Embedding: 0 FLOPs (table lookup)
    # N_LAYERS total layers (GROUP_SIZE unique layers repeated N_LAYERS // GROUP_SIZE times)
    return N_LAYERS * moeut_layer_cost(seq_len) + deembedding_cost(seq_len)


def precompute_cost() -> int:
    n_rotate = int(ROTATE_FRACTION * HEAD_DIM)

    # GROUP_SIZE unique SwitchHeadRope layers, each with its own RotaryPosEncoding object
    rope_cost_per_layer = 0

    if PRECOMP_ROPE_ANGLES:
        # inv_freq: base^{-2i/n_rotate} for i in {0, ..., n_rotate//2 - 1}
        inv_freq_cost = (n_rotate // 2) * ARBITRARY_EXP_COST
        # angles = einsum("i,j->ij", t, inv_freq): [PRECOMP_MAX_SEQ_LEN, n_rotate // 2]
        angles_cost = PRECOMP_MAX_SEQ_LEN * (n_rotate // 2)
        rope_cost_per_layer += inv_freq_cost + angles_cost

    if PRECOMP_ROPE_SIN:
        # sin over emb = cat(angles, angles): [PRECOMP_MAX_SEQ_LEN, n_rotate]
        rope_cost_per_layer += PRECOMP_MAX_SEQ_LEN * n_rotate * SIN_COST

    if PRECOMP_ROPE_COS:
        rope_cost_per_layer += PRECOMP_MAX_SEQ_LEN * n_rotate * COS_COST

    return GROUP_SIZE * rope_cost_per_layer


def format_flops(flops: float | int, decimal_points: int = 2) -> str:
    divisor = 1000.0
    prefixes = ["", "Kilo", "Mega", "Giga", "Tera", "Peta", "Exa", "Zetta", "Yotta"]

    magnitude = 0
    while abs(flops) >= divisor and magnitude < len(prefixes) - 1:
        flops /= divisor
        magnitude += 1

    prefix = prefixes[magnitude]
    if prefix:
        return f"{flops:.{decimal_points}f} {prefix}FLOPs"
    else:
        return f"{int(flops)} FLOPs"


def main(seq_len: int) -> None:
    precompute = format_flops(precompute_cost())
    forward_pass = format_flops(forward_pass_cost(seq_len))

    print("--- FLOPs ---")
    print(f"Precompute Cost: {precompute}")
    print(f"Forward Pass Cost: {forward_pass}")


if __name__ == "__main__":
    main(8_000)

#   precompute_cost() — RoPE sin/cos tables for each of the GROUP_SIZE unique SwitchHeadRope layers. Each has its own RotaryPosEncoding
#   object caching [max_seq_len, n_rotate] tables.

#   forward_pass_cost(seq_len) — N_LAYERS × moeut_layer_cost + de-embedding, where each layer comprises:

#   - sigma_moe_cost — selection linear + sigmoid, cvmm up-projection + ReLU, cvmm down-projection with selection-weight scaling (sel_val
#   scales each expert's output by 1 multiply/element)
#   - switchhead_rope_cost — Q/K dense projections + scaling; V/O MoE selection (linear + sigmoid per head×expert); cvmm V-projection
#   (N_HEADS × ATT_K active experts/token); partial RoPE on n_rotate = ROTATE_FRACTION × HEAD_DIM dims; standard scaled dot-product
#   attention; cvmm O-projection with selection-weight scaling
#   - Two layernorm_cost calls (ln1 for attention input, ln2 for FFN selection input) and two residual additions

#   scales each expert's output by 1 multiply/element)
#   - switchhead_rope_cost — Q/K dense projections + scaling; V/O MoE selection (linear + sigmoid per head×expert); cvmm V-projection
#   (N_HEADS × ATT_K active experts/token); partial RoPE on n_rotate = ROTATE_FRACTION × HEAD_DIM dims; standard scaled dot-product
#   attention; cvmm O-projection with selection-weight scaling
#   - Two layernorm_cost calls (ln1 for attention input, ln2 for FFN selection input) and two residual additions

#   Key MoEUT differences from the dense transformer: the FFN is entirely replaced by SigmaMoE (sparse; only FF_K of FF_N_EXPERTS activated
#    per token), and the attention V/O projections are MoE (ATT_K of ATT_N_EXPERTS per head). The weight sharing (GROUP_SIZE) doesn't
#   reduce FLOPs — only parameter count.


    
