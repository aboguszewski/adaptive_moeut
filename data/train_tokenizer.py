# Trains a SentencePiece BPE tokenizer on filtered C4 shards.

# Special token layout:
#   0 <pad>   1 <unk>   2 <s>   3 </s>

import argparse
from pathlib import Path

import sentencepiece as spm
from datasets import load_from_disk

VOCAB_SIZE = 8192
EOS_ID = 3


def _text_iter(shard_dir: Path, max_docs: int | None):
    total = 0
    for shard_path in sorted(shard_dir.iterdir()):
        shard = load_from_disk(str(shard_path))
        for example in shard:
            yield example["text"]
            total += 1
            if total % 50_000 == 0:
                print(f"  loaded {total:,} docs ...", flush=True)
            if max_docs is not None and total >= max_docs:
                return


def train(shard_dir: str, output_prefix: str, vocab_size: int, max_docs: int | None) -> None:
    shard_dir = Path(shard_dir)
    n_docs = f"{max_docs:,}" if max_docs is not None else "all"
    print(f"Training SentencePiece BPE (vocab_size={vocab_size}) on {n_docs} docs ...")

    spm.SentencePieceTrainer.train(
        sentence_iterator=_text_iter(shard_dir, max_docs),
        model_prefix=str(Path(output_prefix)),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        pad_id=0,  pad_piece="<pad>",
        unk_id=1,  unk_piece="<unk>",
        bos_id=2,  bos_piece="<s>",
        eos_id=3,  eos_piece="</s>",
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        num_threads=16,
    )

    print(f"Saved: {output_prefix}.model  {output_prefix}.vocab")
    print(f"Load:  import sentencepiece as spm; tok = spm.SentencePieceProcessor('{output_prefix}.model')")
    print(f"Use:   encode=tok.encode, eos_id={EOS_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", required=True,
                        help="Train split of filtered C4, e.g. filtered_c4/train")
    parser.add_argument("--output_prefix", required=True,
                        help="Output path prefix, e.g. tokenizer/c4_bpe8k")
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE,
                        help=f"Vocabulary size (default: {VOCAB_SIZE})")
    parser.add_argument("--max_docs", type=int, default=1_000_000,
                        help="Documents to train on (default: 1_000_000; 0 = all)")
    args = parser.parse_args()

    max_docs = args.max_docs if args.max_docs > 0 else None
    train(args.shard_dir, args.output_prefix, args.vocab_size, max_docs)
