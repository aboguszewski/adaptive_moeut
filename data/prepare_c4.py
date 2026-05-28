#filters C4 dataset to remove documents contaminated with LAMBADA and saves a capped subset for training.
#method: 13-gram word-level fingerprinting.
#any C4 document sharing a 13-gram with a LAMBADA test or validation passage is dropped.

import re
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

NGRAM_SIZE = 13
AVG_TOKENS_PER_DOC = 430


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def _ngrams(tokens: list[str], n: int):
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def build_lambada_fingerprints(ngram_size: int = NGRAM_SIZE) -> frozenset[str]:
    passages = []
    for split in ("test", "validation"):
        try:
            ds = load_dataset("EleutherAI/lambada_openai", split=split)
            passages.extend(ex["text"] for ex in ds)
            print(f"  {split}: {len(ds)} passages")
        except Exception as e:
            print(f"  {split}: skipped ({e})")

    fingerprints: set[str] = set()
    for text in passages:
        tokens = _normalize(text)
        fingerprints.update(_ngrams(tokens, ngram_size))

    print(f"  → {len(passages)} passages, {len(fingerprints):,} unique {ngram_size}-grams\n")
    return frozenset(fingerprints)


def is_contaminated(text: str, fingerprints: frozenset[str], ngram_size: int) -> bool:
    tokens = _normalize(text)
    if len(tokens) < ngram_size:
        return False
    return any(ng in fingerprints for ng in _ngrams(tokens, ngram_size))


def filter_c4(output_dir: str, max_train_docs: int, ngram_size: int = NGRAM_SIZE) -> None:
    fingerprints = build_lambada_fingerprints(ngram_size)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    #training is capped
    split_limits = {"train": max_train_docs, "validation": None}

    for split, max_docs in split_limits.items():
        stats = {"kept": 0, "removed": 0}
        print(f"Filtering C4 {split} split (streaming) ...")
        if max_docs is not None:
            est_tokens = max_docs * AVG_TOKENS_PER_DOC
            print(f"  Capped at {max_docs:,} docs (~{est_tokens/1e9:.1f}B tokens)")

        c4 = load_dataset("allenai/c4", "en", split=split, streaming=True)
        out = output_path / split
        out.mkdir(parents=True, exist_ok=True)

        SHARD_SIZE = 1_000_000
        shard_idx = 0
        buffer: list[dict] = []

        def flush(buf: list[dict], idx: int) -> None:
            from datasets import Dataset
            Dataset.from_list(buf).save_to_disk(str(out / f"shard_{idx:05d}"))

        for example in tqdm(c4, desc=f"  {split}"):
            if max_docs is not None and stats["kept"] >= max_docs:
                break
            if not is_contaminated(example["text"], fingerprints, ngram_size):
                buffer.append(example)
                stats["kept"] += 1
                if len(buffer) >= SHARD_SIZE:
                    flush(buffer, shard_idx)
                    shard_idx += 1
                    buffer = []
            else:
                stats["removed"] += 1

        if buffer:
            flush(buffer, shard_idx)

        total = stats["kept"] + stats["removed"]
        pct = stats["removed"] / total * 100 if total > 0 else 0
        est = stats["kept"] * AVG_TOKENS_PER_DOC
        print(f"  Shards: {shard_idx + 1} | "
              f"Kept {stats['kept']:,} (~{est/1e9:.2f}B tokens) | "
              f"Removed {stats['removed']:,} ({pct:.3f}%)\n")

    print("Filtered C4 saved to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_docs", type=int, default=2_500_000)
    parser.add_argument("--ngram_size", type=int, default=NGRAM_SIZE)
    args = parser.parse_args()

    filter_c4(args.output_dir, args.max_train_docs, args.ngram_size)
