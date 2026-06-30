# Zero-shot LAMBADA evaluation for autoregressive language models.
import argparse
import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset


@torch.no_grad()
def evaluate_lambada(
    model: nn.Module,
    encode: Callable[[str], list[int]],
    device: torch.device,
    max_seq_len: int = 2048,
    max_examples: int | None = None,
) -> dict:
    """
    Args:
        model:        autoregressive LM — forward(x: LongTensor[B,T]) → logits[B,T,V]
        encode:       tokenizer callable, e.g. tok.encode
        device:       torch device
        max_seq_len:  model's context window; long examples are truncated from the left
        max_examples: evaluate only the first N examples (None = full 5153-example test set)

    Returns dict with keys: accuracy, perplexity, n_evaluated, n_skipped
    """
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    model.eval()
    n_correct = 0
    total_nll = 0.0
    total_target_tokens = 0
    skipped = 0

    for example in ds:
        text = example["text"]

        last_space = text.rfind(" ")
        if last_space == -1:
            skipped += 1
            continue

        context_str = text[:last_space]
        full_ids    = encode(text)
        context_ids = encode(context_str)
        target_ids  = full_ids[len(context_ids):]

        if not target_ids:
            skipped += 1
            continue

        # Truncate context from the left if the sequence would exceed max_seq_len.
        max_ctx = max_seq_len - len(target_ids)
        if max_ctx < 1:
            skipped += 1
            continue
        context_ids = context_ids[-max_ctx:]

        ids = context_ids + target_ids
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        lbl = torch.tensor([ids[1:]], dtype=torch.long, device=device)

        logits = model(inp)  # (1, seq_len, vocab_size)

        # Target predictions start at position len(context_ids)-1 in the output.
        t_start = len(context_ids) - 1
        t_end   = t_start + len(target_ids)
        t_logits = logits[0, t_start:t_end]   # (n_target, vocab)
        t_labels = lbl[0,   t_start:t_end]    # (n_target,)

        if (t_logits.argmax(dim=-1) == t_labels).all():
            n_correct += 1

        total_nll          += F.cross_entropy(t_logits, t_labels, reduction="sum").item()
        total_target_tokens += len(target_ids)

    n_evaluated = len(ds) - skipped
    return {
        "accuracy":    n_correct / n_evaluated if n_evaluated > 0 else 0.0,
        "perplexity":  math.exp(total_nll / total_target_tokens) if total_target_tokens > 0 else float("inf"),
        "n_evaluated": n_evaluated,
        "n_skipped":   skipped,
    }


if __name__ == "__main__":
    import sentencepiece as spm
    
    # 1. Import both model architectures and their respective configurations
    from models.dense_transformer import DenseTransformer, DEBUG_CONFIG, BASELINE_CONFIG
    from models.moeut import MoEUT, MOEUT_CONFIG # <-- Verified MoEUT module imports

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--tokenizer_path", required=True)
    # 2. Added "moeut" as an eligible choice configuration
    parser.add_argument("--model_config",   choices=["debug", "baseline", "moeut"], default="baseline")
    parser.add_argument("--max_examples",   type=int, default=None,
                        help="Evaluate only the first N examples (default: all 5153)")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tok    = spm.SentencePieceProcessor(args.tokenizer_path)

    if args.model_config == "debug":
        cfg = DEBUG_CONFIG
        model = DenseTransformer(cfg).to(device)
    elif args.model_config == "baseline":
        cfg = BASELINE_CONFIG
        model = DenseTransformer(cfg).to(device)
    elif args.model_config == "moeut":
        cfg = MOEUT_CONFIG
        model = MoEUT(cfg).to(device) 
    else:
        raise ValueError(f"Unknown model configuration: {args.model_config}")

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

    results = evaluate_lambada(
        model,
        encode=tok.encode,
        device=device,
        max_seq_len=cfg.max_seq_len,
        max_examples=args.max_examples,
    )

    print(f"\nLAMBADA zero-shot ({args.model_config.upper()})")
    print(f"  accuracy:    {results['accuracy']:.4f}  ({results['accuracy']*100:.1f}%)")
    print(f"  perplexity:  {results['perplexity']:.2f}  (last-word tokens only)")
    print(f"  evaluated:   {results['n_evaluated']}  /  skipped: {results['n_skipped']}")