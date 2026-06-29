import itertools
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import make_dataloader
from act_stats_tracker import ACTStatsTracker

@dataclass
class TrainConfig:
    # --- paths ---
    shard_dir: str = "filtered_c4"
    tokenizer_path: str = "tokenizer/c4_bpe8k.model"
    output_dir: str = "runs/run"
    resume: str | None = None       # path to a checkpoint to resume training from

    # --- data ---
    seq_len: int = 512
    batch_size: int = 32
    grad_accum_steps: int = 1       # effective batch = batch_size × grad_accum_steps
    num_workers: int = 4

    # --- optimisation ---
    max_steps: int = 100_000
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 1_000

    # --- logging / checkpointing ---
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 2_000
    eval_steps: int = 50            # validation batches per eval call

    # --- spike detection ---
    spike_threshold: float = 1.5    # warn if loss > threshold × EMA loss
    loss_ema_beta: float = 0.98     # EMA decay; ~50-step window at beta=0.98

    # --- hardware ---
    device: str = "cuda"
    mixed_precision: bool = True


# Quick config for sanity checks: 200 steps, tiny batches, frequent logging.
DEBUG_TRAIN_CONFIG = TrainConfig(
    seq_len=128,
    batch_size=8,
    grad_accum_steps=1,
    num_workers=0,
    max_steps=200,
    lr=3e-4,
    warmup_steps=20,
    log_every=10,
    eval_every=50,
    save_every=100,
    eval_steps=10,
    mixed_precision=torch.cuda.is_available(),
)


def _cosine_with_warmup(warmup_steps: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _infinite(loader):
    """Yield batches forever by re-iterating the loader each epoch.

    Unlike itertools.cycle, this does NOT cache batches — cycle would hold an
    entire epoch of tensors in RAM and eventually exhaust the node's memory.
    """
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, cfg: TrainConfig, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in itertools.islice(val_loader, cfg.eval_steps):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=cfg.mixed_precision):
            logits = model(x)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        n += 1
    model.train()
    return total / max(n, 1)


def _save(out_dir: Path, model: nn.Module, optimizer, scheduler, step: int) -> None:
    path = out_dir / f"ckpt_{step:07d}.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print(f"  checkpoint → {path}")


def train(model: nn.Module, cfg: TrainConfig = TrainConfig()) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # The MoEUT cvmm Triton kernel silently returns zeros under AMP on GPUs with
    # compute capability <= 7 (Volta/Turing/Pascal). Detect that and fall back to
    # fp32 so training is correct; newer GPUs (cc >= 8) keep mixed precision.
    uses_cvmm = getattr(model, "last_reg_loss", None) is not None
    if uses_cvmm and device.type == "cuda" and cfg.mixed_precision:
        major = torch.cuda.get_device_properties(device).major
        if major <= 7:
            print(f"WARNING: cvmm is broken under AMP on sm_{major}x GPUs; "
                  f"disabling mixed precision for this run (fp32).")
            cfg.mixed_precision = False

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    tok = spm.SentencePieceProcessor(cfg.tokenizer_path)

    train_loader = make_dataloader(
        shard_dir=Path(cfg.shard_dir) / "train",
        encode=tok.encode,
        eos_id=tok.eos_id(),
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = make_dataloader(
        shard_dir=Path(cfg.shard_dir) / "validation",
        encode=tok.encode,
        eos_id=tok.eos_id(),
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _cosine_with_warmup(cfg.warmup_steps, cfg.max_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    tracker = ACTStatsTracker(model.max_loops, device)

    # --- resume ---
    start_step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {cfg.resume} at step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {device}  |  mixed precision: {cfg.mixed_precision}")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum_steps} sequences "
          f"× {cfg.seq_len} tokens = "
          f"{cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len:,} tokens\n")

    model.train()
    optimizer.zero_grad()

    # Re-iterate the dataloader forever (no batch caching; see _infinite).
    data_iter = _infinite(train_loader)
    step = start_step
    accumulated_loss = 0.0
    accumulated_reg = 0.0
    grad_norm = 0.0
    loss_ema: float | None = None   # initialised on first step
    t0 = time.perf_counter()

    while step < cfg.max_steps:
        # --- gradient accumulation micro-loop ---
        for _ in range(cfg.grad_accum_steps):
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=cfg.mixed_precision):
                logits, output = model(x)
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                # MoEUT stashes its expert-selection entropy regularization here;
                # dense models have no such attribute, so reg is 0.
                reg = getattr(model, "last_reg_loss", None)
                loss = (ce + reg if reg is not None else ce) / cfg.grad_accum_steps
            scaler.scale(loss).backward()

            tracker.update(output)

            accumulated_loss += ce.item() / cfg.grad_accum_steps   # log CE for comparable ppl
            if reg is not None:
                accumulated_reg += float(reg) / cfg.grad_accum_steps

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        # --- spike detection ---
        loss_ema = (
            accumulated_loss if loss_ema is None
            else cfg.loss_ema_beta * loss_ema + (1 - cfg.loss_ema_beta) * accumulated_loss
        )
        if step > cfg.warmup_steps and accumulated_loss > cfg.spike_threshold * loss_ema:
            print(
                f"\n  *** LOSS SPIKE at step {step}: "
                f"loss={accumulated_loss:.4f}  EMA={loss_ema:.4f}  "
                f"ratio={accumulated_loss/loss_ema:.2f}x ***"
            )
            writer.add_scalar("train/spike", accumulated_loss, step)
            spike_path = out_dir / f"ckpt_{step:07d}_SPIKE.pt"
            torch.save({"step": step, "model": model.state_dict()}, spike_path)
            print(f"  Emergency checkpoint → {spike_path}\n")

        # --- logging ---
        if step % cfg.log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = cfg.log_every * cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len / dt
            lr = scheduler.get_last_lr()[0]
            print(
                f"step {step:6d} | loss {accumulated_loss:.4f} | ppl {math.exp(accumulated_loss):.1f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | {tok_per_sec:.0f} tok/s"
            )
            writer.add_scalar("train/loss", accumulated_loss, step)
            writer.add_scalar("train/ppl", math.exp(accumulated_loss), step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", grad_norm, step)
            if accumulated_reg:
                writer.add_scalar("train/reg_loss", accumulated_reg, step)
            
            tracker.log_to_tensorboard(writer, step)
            
            t0 = time.perf_counter()

        accumulated_loss = 0.0
        accumulated_reg = 0.0

        # --- eval ---
        if step % cfg.eval_every == 0 and step > 0:
            val_loss = evaluate(model, val_loader, cfg, device)
            print(f"  [eval]  step {step:6d} | val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.1f}")
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/ppl", math.exp(val_loss), step)

        # --- checkpoint ---
        if step % cfg.save_every == 0 and step > 0:
            _save(out_dir, model, optimizer, scheduler, step)

        step += 1

    # final eval + checkpoint
    val_loss = evaluate(model, val_loader, cfg, device)
    print(f"\n[final] val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.1f}")
    _save(out_dir, model, optimizer, scheduler, step)
    writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dense", "moeut"], default="dense",
                        help="Which architecture to train (parameter-matched)")
    parser.add_argument("--debug", action="store_true",
                        help="Tiny model + 200 steps for a quick sanity check")
    parser.add_argument("--shard_dir", default="filtered_c4")
    parser.add_argument("--tokenizer_path", default="tokenizer/c4_bpe8k.model")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint (.pt) to resume training from")
    args = parser.parse_args()

    if args.model == "moeut":
        from models.moeut import MoEUT, MOEUT_CONFIG, DEBUG_CONFIG
        build_model = lambda debug: MoEUT(DEBUG_CONFIG if debug else MOEUT_CONFIG)
        default_out = "runs/debug_moeut" if args.debug else "runs/moeut"
    else:
        from models.dense_transformer import DenseTransformer, DEBUG_CONFIG, BASELINE_CONFIG
        build_model = lambda debug: DenseTransformer(DEBUG_CONFIG if debug else BASELINE_CONFIG)
        default_out = "runs/debug" if args.debug else "runs/baseline_dense"

    if args.debug:
        cfg = DEBUG_TRAIN_CONFIG
        cfg.shard_dir = args.shard_dir
        cfg.tokenizer_path = args.tokenizer_path
        cfg.output_dir = args.output_dir or default_out
        cfg.resume = args.resume
    else:
        cfg = TrainConfig(
            shard_dir=args.shard_dir,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir or default_out,
            resume=args.resume,
        )

    train(build_model(args.debug), cfg)
