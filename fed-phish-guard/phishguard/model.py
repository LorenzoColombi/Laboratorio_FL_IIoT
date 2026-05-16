"""
Phishing URL Classification - Lightweight CNN Model

Architecture:
- Byte embedding
- Single Conv1d (k=3) + ReLU + BatchNorm
- Global max pool
- Dropout → Linear → logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from phishguard.data import PAD_IDX, VOCAB_SIZE


class PhishingCNN(nn.Module):
    """Lightweight single-conv model for phishing URL classification.

    Architecture:
        Input:          (batch, max_len) byte indices
        Embedding:      (batch, max_len, embed_dim)
        Conv1d k=3:     (batch, num_filters, max_len)
        Global max pool:(batch, num_filters)
        Dropout + Linear → (batch, 1) logits
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        num_filters: int = 128,
        dropout: float = 0.3,
        padding_idx: int = PAD_IDX,
        **kwargs,  # absorb unused params (e.g. kernel_sizes) for compatibility
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(num_filters)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids).transpose(1, 2)  # (batch, embed_dim, max_len)
        x = F.relu(self.bn(self.conv(x)))  # (batch, num_filters, max_len)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (batch, num_filters)
        return self.head(x)  # (batch, 1)
