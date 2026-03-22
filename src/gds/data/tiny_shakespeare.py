"""TinyShakespeare dataset for character-level language modelling.

Downloads the dataset from Karpathy's GitHub, splits into fixed-length
chunks as "samples", and returns (input_ids, target_ids, sample_id) tuples
suitable for the data-selection pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gds.common.io import ensure_dir, read_json, write_json


_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(data_dir: Path) -> str:
    """Download TinyShakespeare text and return it as a string."""
    out_path = data_dir / "tiny_shakespeare" / "input.txt"
    if out_path.exists():
        return out_path.read_text(encoding="utf-8")

    ensure_dir(out_path.parent)
    import urllib.request
    urllib.request.urlretrieve(_TINY_SHAKESPEARE_URL, str(out_path))
    return out_path.read_text(encoding="utf-8")


def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Build character-level vocabulary from text."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode_text(text: str, stoi: dict[str, int]) -> np.ndarray:
    """Encode text to integer array."""
    return np.array([stoi[ch] for ch in text], dtype=np.int64)


@dataclass(frozen=True)
class TextDatasetSplit:
    train_ids: list[int]
    val_ids: list[int]
    split_seed: int


def chunk_into_samples(
    data: np.ndarray, block_size: int
) -> list[np.ndarray]:
    """Split encoded text into non-overlapping chunks of block_size + 1.

    Each chunk has block_size input tokens and 1 target token (next char).
    """
    chunk_len = block_size + 1
    n_chunks = len(data) // chunk_len
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_len
        chunks.append(data[start : start + chunk_len])
    return chunks


def load_or_create_text_split(
    data_dir: Path,
    split_file: Path,
    block_size: int,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[TextDatasetSplit, list[np.ndarray], dict[str, int], dict[int, str]]:
    """Load or create a TinyShakespeare split.

    Returns (split, chunks, stoi, itos).
    """
    text = download_tiny_shakespeare(data_dir)
    stoi, itos = build_char_vocab(text)
    encoded = encode_text(text, stoi)
    chunks = chunk_into_samples(encoded, block_size)

    if split_file.exists():
        payload = read_json(split_file)
        split = TextDatasetSplit(
            train_ids=list(payload["train_ids"]),
            val_ids=list(payload["val_ids"]),
            split_seed=int(payload["split_seed"]),
        )
        return split, chunks, stoi, itos

    ensure_dir(split_file.parent)
    n = len(chunks)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_size = max(1, int(n * val_fraction))
    val_ids = sorted(perm[:val_size].tolist())
    train_ids = sorted(perm[val_size:].tolist())

    payload = {"train_ids": train_ids, "val_ids": val_ids, "split_seed": seed}
    write_json(split_file, payload)
    split = TextDatasetSplit(train_ids=train_ids, val_ids=val_ids, split_seed=seed)
    return split, chunks, stoi, itos


class TinyShakespeareDataset(Dataset):
    """Character-level dataset returning (input_ids, target_ids, sample_id).

    input_ids:  chunk[:-1] (block_size tokens)
    target_ids: chunk[1:]  (block_size tokens, shifted by 1)
    sample_id:  integer index for the data-selection pipeline
    """

    def __init__(self, chunks: list[np.ndarray], indices: Sequence[int]) -> None:
        self.chunks = chunks
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample_id = self.indices[idx]
        chunk = self.chunks[sample_id]
        x = torch.from_numpy(chunk[:-1].copy())  # input
        y = torch.from_numpy(chunk[1:].copy())    # target
        return x, y, sample_id


def build_text_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )
