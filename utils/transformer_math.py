# Full Model Architecture:
#   Tokenization --> Embedding Layer --> B Transformer Blocks --> De-Embedding Layer
#   
#   Transformer Block:
#       Input ------------------ Residual Stream --------------> (+) --------- Residual Stream ----> (+) --> Output
#             --> LayerNorm --> MultiHead Attention (with RoPE) --/  --> LayerNorm --> FeedForward --/

# Assumptions:
# 1. Memory operations such as concat, duplicate, transpose etc. require 0 FLOPs.
# 2. All element-wise divisions are done by computing the inverse of the scalar and multiplying by that inverse.
#    E.g. instead of QK^T / sqrt(head_dim), 1/(sqrt(head_dim)) * QK^T. This seems to be realistic for GPU math.
# 3. Softmax in the attention mechanism is defined in its numerically stable version, so as:
#    softmax(x_i) = exp(x_i - max(x)) / (sum from j=1 to L: exp(x_j - max(x)), where x is a vector of length L and x_i is its i-th element.
#    And computed as follows:
#    a. compute max(x)
#    b. compute x - max(x)
#    c. compute exp(x_i - max(x)) for all i in {1, ..., L}
#    d. compute sum from i=1 to L: exp(x_j - max(x))
#    e. compute 1 / (sum from i=1 to L: exp(x_j - max(x)))
#    f. do multiplication of exp(x_i - max(x)) for all i in {1, ..., L} by 1 / (sum from i=1 to L: exp(x_j - max(x)))
#       which yields softmax(x)
# 4. The attention mechanism doesn't include masking. The compute cost is calculated as if the attention was bidirectional.
# 5. ReLU(x) where x is a scalar requires 1 FLOP.

# Notation:
# 1. R^{w, h} -> Matrix of Real numbers of width w and height h.
# 2. x * M -> multiplying vector x by matrix M, but x is a column vector (R^{1, length}).
#    The multiplication is column by rows.
# This opposite convention was chosen to achieve a natural input, output sequence of dimensions.
# When x is the input and W is the weight matrix, x is R^input and W is R^{input, output}.
# x * W matches the control flow, from left to right, input to output.

# Precomputation:
PRECOMP_MODEL_DIM_INVERSE = True # 1/model_dim
PRECOMP_HEAD_DIM_INVERSE = True # 1/head_dim
PRECOMP_HEAD_DIM_SQRT = True # sqrt(head_dim)
PRECOMP_HEAD_DIM_SQRT_INVERSE = True # 1/sqrt(head_dim)

PRECOMP_ROPE_SIN = True # RoPE required vector of sin(i) for i in {0, ... , head_dim/2 -1}
PRECOMP_ROPE_COS = True # RoPE required vector of cos(i) for i in {0, ... , head_dim/2 -1}

PRECOMP_ROPE_ANGLES = True # RoPE required matrix of rotation angles, R^{seq_len, head_dim/2}. Will precompute R^{max_seq_len, head_dim/2}
PRECOMP_MAX_SEQ_LEN = 256_000 # Can't be None if PRECOMP_ROPE_ANGLES=True

# Constants:
VOCAB_SIZE = 8192

TRANSFORMER_BLOCKS = 4

MODEL_DIM = 512
HEAD_DIM = 128
ATTENTION_HEADS = 4
FEEDFORWARD_DIM = 1024

SIN_COST = 1
COS_COST = 1
E_EXP_COST = 1
SQRT_COST = 1
ARBITRARY_EXP_COST = 1

def deembedding_cost(seq_len: int) -> int:
    # R^{seq_len, model_dim} * R^{model_dim, vocab_size}
    return seq_len * VOCAB_SIZE * (2*MODEL_DIM -1)

def residual_addition_cost(seq_len: int) -> int:
    # R^{seq_len, model_dim} + R^{seq_len, model_dim}
    return seq_len * MODEL_DIM

def feedforward_cost(seq_len: int) -> int:
    # R^{seq_len, model_dim} * R^{model_dim, feedforward_dim}
    in_projection_cost = seq_len * FEEDFORWARD_DIM * (2*MODEL_DIM - 1)

    # ReLU(R^{seq_len, feedforward_dim})
    relu_cost = seq_len * FEEDFORWARD_DIM

    # R^{seq_len, feedforward_dim} * R^{feedforward_dim, model_dim}
    out_projection_cost = seq_len * MODEL_DIM * (2*FEEDFORWARD_DIM - 1)
    
    return in_projection_cost + relu_cost + out_projection_cost 

def singlehead_attention_cost(seq_len: int) -> int:
    # --- input -> Q, K, V ---
    # 3x [R^{seq_len, model_dim} * R^{model_dim, head_dim}]
    qkv_projection_cost = 3 * (seq_len * HEAD_DIM * (2*MODEL_DIM - 1))

    # --- RoPE ---
    
    rope_angles_cost = 0
    if not PRECOMP_ROPE_ANGLES:
        # R^{seq_len, 1} * R^{1, head_dim/2}
        rope_angles_cost = seq_len * (HEAD_DIM//2)
    
    rope_sin_cost = 0
    if not PRECOMP_ROPE_SIN:
        # sin(torch.arange(0, head_dim/2))
        rope_sin_cost = seq_len * (HEAD_DIM//2) * SIN_COST

    rope_cos_cost = 0
    if not PRECOMP_ROPE_COS:
        # cos(torch.arange(0, head_dim/2))
        rope_cos_cost = seq_len * (HEAD_DIM//2) * COS_COST
    
    # If x is RoPE's input, then
    # new_x = x (*) cos + rotate_half(x) (*) sin
    # R^{seq_len, head_dim} (*) R^{head_dim} + R^{seq_len, head_dim} (*) R^{head_dim}
    # but additionally rotate_half(x) involves -1 * R^{seq_len, head_dim/2}
    rope_rotation_cost = 3 * (seq_len * HEAD_DIM) + seq_len * (HEAD_DIM//2)

    # RoPE is applied to both Q and K, so 2x rotation.
    rope_cost = rope_angles_cost \
        + rope_sin_cost + rope_cos_cost \
        + 2 * rope_rotation_cost
    
    # --- Dot-product Attention ---
    
    # R^{seq_len, head_dim} * R^{head_dim, seq_len}
    qk_t_cost = seq_len**2 * (2*HEAD_DIM - 1)

    # sqrt(head_dim), 1/sqrt(head_dim)
    # scalar * R^{seq_len, seq_len}
    qk_t_normalization_cost = (not PRECOMP_HEAD_DIM_SQRT) * SQRT_COST \
        + (not PRECOMP_HEAD_DIM_SQRT_INVERSE) * 1 \
        + seq_len**2

    # general softmax(x) cost for x in R^L is L(exp_cost + 4)-1
    # softmax(R^{seq_len, seq_len}) -> seq_len * softmax(R^{seq_len})
    softmax_cost = seq_len * (seq_len * (E_EXP_COST + 4) - 1)

    # R^{seq_len, seq_len} * R^{seq_len, head_dim}
    values_cost = seq_len * HEAD_DIM * (2*seq_len - 1)

    return qkv_projection_cost + rope_cost \
        + qk_t_cost + qk_t_normalization_cost \
        + softmax_cost + values_cost

def multihead_attention_cost(seq_len: int) -> int:
    # concat(attention_heads x R^{seq_len, head_dim}) -> R^{seq_len, attention_heads * head_dim}
    # R^{seq_len, attention_heads * head_dim} * R^{attention_heads * head_dim, model_dim}
    out_projection_cost = seq_len * MODEL_DIM * (2*ATTENTION_HEADS*HEAD_DIM - 1)

    return ATTENTION_HEADS * singlehead_attention_cost(seq_len) + out_projection_cost

def layernorm_cost(seq_len: int) -> int:
    mean_cost = MODEL_DIM + (not PRECOMP_MODEL_DIM_INVERSE)
    variance_cost = 3*MODEL_DIM + (not PRECOMP_MODEL_DIM_INVERSE)
    normalization_cost = MODEL_DIM + SQRT_COST + 2
    scaling_cost = 2*MODEL_DIM
    
    single_embedding_cost = mean_cost + variance_cost \
        + normalization_cost + scaling_cost
    
    return seq_len * single_embedding_cost

def transformer_block_cost(seq_len: int) -> int:
    return layernorm_cost(seq_len) + multihead_attention_cost(seq_len) \
        + residual_addition_cost(seq_len) + layernorm_cost(seq_len) \
        + feedforward_cost(seq_len) + residual_addition_cost(seq_len)

def forward_pass_cost(seq_len: int) -> int:    
    # Tokenization and Embedding require 0 FLOPs.
    return TRANSFORMER_BLOCKS * transformer_block_cost(seq_len) + deembedding_cost(seq_len)

def precompute_cost() -> int:
    scalars_cost = PRECOMP_MODEL_DIM_INVERSE * 1 \
    + PRECOMP_HEAD_DIM_INVERSE * 1 \
    + PRECOMP_HEAD_DIM_SQRT * SQRT_COST \
    + PRECOMP_HEAD_DIM_SQRT_INVERSE * 1
    
    rope_cost = 0

    if PRECOMP_ROPE_ANGLES:
        # RoPE has a 'base' parameter.
        # freqs, R^{head_dim/2}; freqs_i = base^{-2i/head_dim}
        # positions, R^{max_seq_len, 1}; positions = torch.arange(0, max_seq_len)
        # angles, R^{max_seq_len, head_dim/2} = positions * freqs
        rope_cost += (HEAD_DIM//2) * ARBITRARY_EXP_COST + PRECOMP_MAX_SEQ_LEN * (HEAD_DIM//2)
    
    if PRECOMP_ROPE_SIN:
        # sin(torch.arange(0, head_dim/2))
        rope_cost += PRECOMP_MAX_SEQ_LEN * (HEAD_DIM//2) * SIN_COST

    if PRECOMP_ROPE_COS:
        # cos(torch.arange(0, head_dim/2))
        rope_cost += PRECOMP_MAX_SEQ_LEN * (HEAD_DIM//2) * COS_COST

    return scalars_cost + rope_cost

def format_flops(flops: float | int, decimal_points: int = 2) -> str:
    """
    Converts a raw number of FLOPs into a human-readable string using standard SI prefixes.
    """
    # Hardware compute traditionally uses base-10 (1000), not base-2 (1024)
    divisor = 1000.0
    prefixes = ["", "Kilo", "Mega", "Giga", "Tera", "Peta", "Exa", "Zetta", "Yotta"]
    
    magnitude = 0
    # Keep dividing by 1000 until the number is readable or we run out of prefixes
    while abs(flops) >= divisor and magnitude < len(prefixes) - 1:
        flops /= divisor
        magnitude += 1
        
    prefix = prefixes[magnitude]
    
    # Format the string with the requested precision
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