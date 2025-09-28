"""Synthetic datasets for ranking/recommendation benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


@dataclass
class SyntheticDatasetConfig:
    num_samples: int = 10_000
    sequence_length: int = 32
    dense_dim: int = 128
    sparse_features: int = 16
    vocab_size: int = 50_000
    num_dense_features: int = 8
    seed: int = 13


class SyntheticRankingDataset(Dataset):
    """Deterministic synthetic dataset producing ranking features."""

    def __init__(self, cfg: SyntheticDatasetConfig) -> None:
        self.cfg = cfg
        self._base_generator = torch.Generator().manual_seed(cfg.seed)

    def __len__(self) -> int:
        return self.cfg.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.cfg.seed + index)
        seq_len = self.cfg.sequence_length

        dense = torch.randn(
            (seq_len, self.cfg.num_dense_features, self.cfg.dense_dim), generator=generator
        )
        categorical = torch.randint(
            0,
            self.cfg.vocab_size,
            (seq_len, self.cfg.sparse_features),
            generator=generator,
            dtype=torch.int64,
        )
        target = torch.rand(seq_len, generator=generator)
        importance = torch.rand(seq_len, generator=generator)

        return {
            "dense": dense,
            "categorical": categorical,
            "target": target,
            "importance": importance,
        }


def _collate(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    dense = torch.stack([item["dense"] for item in batch], dim=0)
    categorical = torch.stack([item["categorical"] for item in batch], dim=0)
    target = torch.stack([item["target"] for item in batch], dim=0)
    importance = torch.stack([item["importance"] for item in batch], dim=0)
    return {
        "dense": dense,
        "categorical": categorical,
        "target": target,
        "importance": importance,
    }


def create_dataloader(
    cfg: SyntheticDatasetConfig,
    *,
    batch_size: int,
    world_size: int,
    rank: int,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = SyntheticRankingDataset(cfg)
    sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


__all__ = ["SyntheticRankingDataset", "SyntheticDatasetConfig", "create_dataloader"]
