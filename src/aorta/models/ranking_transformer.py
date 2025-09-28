"""Transformer-based recommendation model for overlap benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 50_000
    embedding_dim: int = 128
    num_dense_features: int = 8
    dense_dim: int = 128
    model_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    mlp_hidden_dim: int = 1024


class RankingTransformerModel(nn.Module):
    """Large transformer encoder tailored for ranking signals."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        dense_in = cfg.num_dense_features * cfg.dense_dim
        self.dense_projection = nn.Linear(dense_in, cfg.model_dim)
        self.embedding_projection = nn.Linear(cfg.embedding_dim, cfg.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.mlp_hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.norm = nn.LayerNorm(cfg.model_dim)
        self.scorer = nn.Sequential(
            nn.Linear(cfg.model_dim, cfg.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        dense = batch["dense"]  # [B, T, F, D]
        categorical = batch["categorical"]  # [B, T, S]

        batch_size, sequence_length, _, _ = dense.shape

        dense_features = dense.flatten(start_dim=2)  # [B, T, F*D]
        dense_repr = self.dense_projection(dense_features)  # [B, T, model_dim]

        embeddings = self.embedding(categorical)  # [B, T, S, embed_dim]
        pooled_embed = embeddings.mean(dim=2)  # [B, T, embed_dim]
        embed_repr = self.embedding_projection(pooled_embed)

        combined = dense_repr + embed_repr

        encoded = self.encoder(combined)  # [B, T, model_dim]
        encoded = self.norm(encoded)

        scores = self.scorer(encoded).squeeze(-1)  # [B, T]
        return scores


__all__ = ["ModelConfig", "RankingTransformerModel"]
