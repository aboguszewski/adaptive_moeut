import random
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
import torch.utils.data
from datasets import load_from_disk
from torch.utils.data import DataLoader, IterableDataset


class PackedC4Dataset(IterableDataset):
    def __init__(
        self,
        shard_dir: str | Path,
        encode: Callable[[str], list[int]],
        eos_id: int,
        seq_len: int,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.encode = encode
        self.eos_id = eos_id
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.shards = sorted(self.shard_dir.iterdir())

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        shards = [s for i, s in enumerate(self.shards) if i % num_workers == worker_id]

        rng = random.Random(self.seed + worker_id)
        if self.shuffle:
            rng.shuffle(shards)

        buf: list[int] = []
        need = self.seq_len + 1

        for shard_path in shards:
            shard = load_from_disk(str(shard_path))
            indices = list(range(len(shard)))
            if self.shuffle:
                rng.shuffle(indices)
            for i in indices:
                buf.extend(self.encode(shard[i]["text"]))
                buf.append(self.eos_id)
                while len(buf) >= need:
                    chunk = torch.tensor(buf[:need], dtype=torch.long)
                    yield chunk[:-1], chunk[1:]
                    buf = buf[self.seq_len:]


def make_dataloader(
    shard_dir: str | Path,
    encode: Callable[[str], list[int]],
    eos_id: int,
    seq_len: int,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Args:
        shard_dir:   path to a split directory, e.g. <output_dir>/train
        encode:      callable mapping a string to a list of token ids
        eos_id:      token id inserted between documents
        seq_len:     context length (tokens per sample)
        batch_size:  sequences per batch
        shuffle:     randomize shard and document order
        seed:        base RNG seed (each worker gets seed + worker_id)
        num_workers: DataLoader worker processes
        pin_memory:  pin CPU tensors for faster GPU transfer
    """
    dataset = PackedC4Dataset(shard_dir, encode, eos_id, seq_len, shuffle=shuffle, seed=seed)

    # With an IterableDataset, each worker owns a disjoint subset of shards
    # (round-robin in __iter__). More workers than shards leaves some idle, so
    # cap it at the shard count.
    num_workers = min(num_workers, len(dataset.shards))

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    if num_workers > 0:
        # Keep workers alive across epochs (avoids respawn deadlocks on NFS) and
        # fail loudly instead of hanging forever if a worker stalls.
        kwargs.update(persistent_workers=True, prefetch_factor=2, timeout=120)
    return DataLoader(dataset, **kwargs)
