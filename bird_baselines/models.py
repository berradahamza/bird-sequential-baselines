"""Explicit BIRD adaptations, implemented independently from the cited papers.

See docs/METHODS.md: full-catalogue CE on the last position of every prefix.
These are not the original authors' software or exact benchmark reproductions.
"""
from __future__ import annotations

import math
import torch
from torch import nn


LABELS = {"gru4rec": "GRU4Rec-CE-prefix (BIRD adaptation)",
          "sasrec": "SASRec-CE-prefix (BIRD adaptation)"}


class GRU4RecPrefix(nn.Module):
    """Learned item embeddings, unidirectional GRU, untied linear score head."""
    def __init__(self, n_items, hidden=64, dropout=0.2, **unused):
        super().__init__()
        self.items = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_items)
        nn.init.normal_(self.items.weight, std=0.02)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def encode_steps(self, ids):
        out, _ = self.gru(self.dropout(self.items(ids)))
        return out

    def encode(self, ids, lengths):
        steps = self.encode_steps(ids)
        return self.dropout(steps[torch.arange(len(ids), device=ids.device), lengths - 1])

    def forward(self, ids, lengths):
        return self.head(self.encode(ids, lengths))


class CausalBlock(nn.Module):
    """Pre-normalized causal self-attention and position-wise feed-forward block."""
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, eps=1e-8)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, eps=1e-8)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden, hidden), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding, causal):
        q = self.norm1(x)
        # SASRec-style normalized query, sequence keys/values, query residual.
        message, _ = self.attention(q, x, x, attn_mask=causal,
                                    key_padding_mask=padding, need_weights=False)
        x = q + self.dropout(message)
        y = self.norm2(x)
        x = y + self.ff(y)
        return x.masked_fill(padding.unsqueeze(-1), 0.0)


class SASRecPrefix(nn.Module):
    """Causal attention, absolute positions, last real position, tied item scoring."""
    def __init__(self, n_items, hidden=64, dropout=0.2, max_history=32,
                 blocks=2, heads=2, **unused):
        super().__init__()
        self.n_items, self.hidden = n_items, hidden
        self.items = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.positions = nn.Embedding(max_history, hidden)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([CausalBlock(hidden, heads, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden, eps=1e-8)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def encode_steps(self, ids):
        padding = ids.eq(0)
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.items(ids) * math.sqrt(self.hidden) + self.positions(positions)[None]
        x = self.dropout(x).masked_fill(padding.unsqueeze(-1), 0.0)
        causal = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device).triu(1)
        for block in self.blocks:
            x = block(x, padding, causal)
        return self.norm(x).masked_fill(padding.unsqueeze(-1), 0.0)

    def encode(self, ids, lengths):
        x = self.encode_steps(ids)
        return x[torch.arange(len(ids), device=ids.device), lengths - 1]

    def forward(self, ids, lengths):
        # PAD and MASK are excluded from both CE and all ranking metrics.
        return self.encode(ids, lengths) @ self.items.weight[1:self.n_items + 1].T


def build_model(config, n_items):
    common = dict(n_items=n_items, hidden=config["hidden"], dropout=config["dropout"],
                  max_history=config["max_history"], blocks=config["blocks"], heads=config["heads"])
    if config["model"] == "gru4rec":
        return GRU4RecPrefix(**common)
    if config["model"] == "sasrec":
        return SASRecPrefix(**common)
    raise ValueError("Unknown model")


