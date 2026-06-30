# Zero-shot LAMBADA and Hellaswag evaluation for autoregressive language models.
import argparse
import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


@torch.no_grad()
def evaluate_lambada(
    model: nn.Module,
    encode: Callable[[str], list[int]],
    device: torch.device,
    max_seq_len: int = 2048,
    max_examples: int | None = None,
) -> dict:
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

        max_ctx = max_seq_len - len(target_ids)
        if max_ctx < 1:
            skipped += 1
            continue
        context_ids = context_ids[-max_ctx:]

        ids = context_ids + target_ids
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        lbl = torch.tensor([ids[1:]], dtype=torch.long, device=device)

        logits = model(inp)

        t_start = len(context_ids) - 1
        t_end   = t_start + len(target_ids)
        t_logits = logits[0, t_start:t_end]   
        t_labels = lbl[0,   t_start:t_end]    

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


@torch.no_grad()
def evaluate_hellaswag(
    model: nn.Module,
    encode: Callable[[str], list[int]],
    device: torch.device,
    max_seq_len: int = 2048,
    max_examples: int | None = None,
) -> dict:
    """
    Evaluates the model on Hellaswag multiple-choice completion.
    Selects the ending with the lowest average cross-entropy loss per token.
    """
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    model.eval()
    n_correct = 0
    n_evaluated = len(ds)

    for example in tqdm(ds, desc="Evaluating Hellaswag"):
        context = example["ctx"]
        endings = example["endings"]  # List of 4 strings
        label = int(example["label"])  # Correct index (0-3)

        context_ids = encode(context)
        ending_losses = []

        for ending in endings:
            # Explicitly space-separate the ending from the context
            ending_ids = encode(" " + ending) 
            
            # Truncate context from left if it exceeds max context
            max_ctx = max_seq_len - len(ending_ids)
            ctx_ids_trunc = context_ids[-max_ctx:] if max_ctx > 0 else []
            
            ids = ctx_ids_trunc + ending_ids
            
            inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            lbl = torch.tensor([ids[1:]], dtype=torch.long, device=device)

            logits = model(inp)

            # Evaluate loss only on the ending tokens
            t_start = len(ctx_ids_trunc) - 1
            t_logits = logits[0, t_start:] 
            t_labels = lbl[0, t_start:]

            # Calculate mean loss for this specific ending
            loss = F.cross_entropy(t_logits, t_labels, reduction="mean").item()
            ending_losses.append(loss)

        # The model "chooses" the ending with the lowest loss (highest probability)
        prediction = ending_losses.index(min(ending_losses))
        if prediction == label:
            n_correct += 1

    return {
        "accuracy": n_correct / n_evaluated if n_evaluated > 0 else 0.0,
        "n_evaluated": n_evaluated,
    }


if __name__ == "__main__":
    import sentencepiece as spm
    from models.dense_transformer import DenseTransformer, DEBUG_CONFIG, BASELINE_CONFIG
    from models.moeut import MoEUT, MOEUT_CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--model_config",   choices=["debug", "baseline", "moeut"], default="baseline")
    parser.add_argument("--max_examples",   type=int, default=None,
                        help="Evaluate only the first N examples")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task",           choices=["lambada", "hellaswag", "both"], default="both",
                        help="Which evaluation task to run")
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
    print(f"Loaded {args.model_config.upper()} checkpoint from step {ckpt.get('step', '?')}")

    # Run LAMBADA
    if args.task in ["lambada", "both"]:
        print("\n--- Running LAMBADA ---")
        lambada_res = evaluate_lambada(
            model,
            encode=tok.encode,
            device=device,
            max_seq_len=cfg.max_seq_len,
            max_examples=args.max_examples,
        )
        print(f"  accuracy:    {lambada_res['accuracy']:.4f}  ({lambada_res['accuracy']*100:.1f}%)")
        print(f"  perplexity:  {lambada_res['perplexity']:.2f}")
        print(f"  evaluated:   {lambada_res['n_evaluated']}  /  skipped: {lambada_res['n_skipped']}")

    # Run Hellaswag
    if args.task in ["hellaswag", "both"]:
        print("\n--- Running HELLASWAG ---")
        hellaswag_res = evaluate_hellaswag(
            model,
            encode=tok.encode,
            device=device,
            max_seq_len=cfg.max_seq_len,
            max_examples=args.max_examples,
        )
        print(f"  accuracy:    {hellaswag_res['accuracy']:.4f}  ({hellaswag_res['accuracy']*100:.1f}%)")
        print(f"  evaluated:   {hellaswag_res['n_evaluated']}")