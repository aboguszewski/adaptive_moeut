"""Benchmark inference latency of the dense baseline vs. MoEUT, fairly.

Runs both models on the SAME GPU at the SAME precision, discards warmup
iterations, and times steady-state forward passes with proper CUDA
synchronization. Reports per-forward latency and tokens/sec, plus the MoEUT/
dense ratio.

IMPORTANT precision note: on sm_<=7 GPUs (Volta V100, Turing, Pascal) the cvmm
kernel returns garbage under fp16/bf16 autocast, so MoEUT can only be timed in
fp32 there. This script refuses fp16/bf16 on such GPUs. On a V100 use
--precision fp32 for BOTH models (the only apples-to-apples comparison).

Usage:
    python inference_bench.py \
        --dense-ckpt runs/dense/ckpt_0100000.pt \
        --moeut-ckpt runs/moeut/ckpt_0100000.pt \
        --precision fp32 --batch-size 8 --seq-len 512
"""
import argparse
import time

import torch

from models.dense_transformer import DenseTransformer, BASELINE_CONFIG
from models.moeut import MoEUT, MOEUT_CONFIG


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load(arch: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = DenseTransformer(BASELINE_CONFIG) if arch == "dense" else MoEUT(MOEUT_CONFIG)
    model = model.to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    return model


@torch.no_grad()
def bench(model, x, device, autocast_dtype, warmup: int, iters: int) -> dict:
    use_autocast = autocast_dtype != torch.float32 and device.type == "cuda"

    def run_once():
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=use_autocast):
            return model(x)

    # Warmup — triggers Triton compile/autotune for MoEUT; not timed.
    for _ in range(warmup):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_tokens = x.numel() * iters
    return {
        "ms_per_forward": 1e3 * elapsed / iters,
        "tokens_per_sec": n_tokens / elapsed,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense-ckpt", required=True)
    p.add_argument("--moeut-ckpt", required=True)
    p.add_argument("--precision", choices=DTYPES.keys(), default="fp32")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    autocast_dtype = DTYPES[args.precision]

    # Guard the V100/Volta trap: cvmm is broken under fp16/bf16 on sm_<=7.
    if device.type == "cuda" and args.precision != "fp32":
        major = torch.cuda.get_device_properties(device).major
        if major <= 7:
            raise SystemExit(
                f"Refusing to benchmark in {args.precision} on sm_{major}x "
                f"({torch.cuda.get_device_name(device)}): the cvmm kernel returns "
                f"garbage under autocast on Volta/Turing/Pascal. Use --precision "
                f"fp32 here (for BOTH models), or run on an Ampere+ (sm_80+) GPU."
            )

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"Device: {gpu}  |  precision: {args.precision}  |  "
          f"batch {args.batch_size} × seq {args.seq_len}  |  "
          f"warmup {args.warmup}, iters {args.iters}\n")

    # Same input tokens for both models, so the comparison is identical work.
    vocab = BASELINE_CONFIG.vocab_size
    x = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)

    results = {}
    for arch, ckpt in [("dense", args.dense_ckpt), ("moeut", args.moeut_ckpt)]:
        model = load(arch, ckpt, device)
        results[arch] = bench(model, x, device, autocast_dtype, args.warmup, args.iters)
        r = results[arch]
        print(f"{arch:>6}: {r['ms_per_forward']:8.2f} ms/fwd   "
              f"{r['tokens_per_sec']:12,.0f} tok/s")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ratio = results["moeut"]["ms_per_forward"] / results["dense"]["ms_per_forward"]
    print(f"\nMoEUT / dense latency ratio: {ratio:.2f}×")


if __name__ == "__main__":
    main()
