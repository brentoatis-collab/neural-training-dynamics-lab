"""
model.py — 1-layer transformer for modular addition (Experiment 3).

Architecture (matches Power et al. 2022 style):
  Shared token embedding:   nn.Embedding(p, d_model)
  Positional embedding:     nn.Embedding(2, d_model)  [position 0=a, 1=b]
  Transformer layer:        nn.TransformerEncoderLayer(d_model, n_heads, d_ff,
                                activation='gelu', batch_first=True, norm_first=True)
  Mean pooling over sequence positions.
  Output projection:        nn.Linear(d_model, p)

Input convention:
  a, b — LongTensor of shape (N,) with values in [0, p).
  Tokens stacked to shape (N, 2) before embedding.
"""

import math

import torch
import torch.nn as nn


class GrokTransformer(nn.Module):
    def __init__(self, p: int = 97, d_model: int = 128, n_heads: int = 4, d_ff: int = 512):
        super().__init__()
        self.p       = p
        self.d_model = d_model

        self.embed     = nn.Embedding(p, d_model)
        self.pos_embed = nn.Embedding(2, d_model)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= d_ff,
            activation     = "gelu",
            batch_first    = True,
            norm_first     = True,   # pre-norm (more stable training)
            dropout        = 0.0,
        )
        self.out = nn.Linear(d_model, p)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: LongTensor (N,) — first operand
            b: LongTensor (N,) — second operand
        Returns:
            logits: FloatTensor (N, p)
        """
        tokens = torch.stack([a, b], dim=1)           # (N, 2)
        pos    = torch.arange(2, device=a.device)     # [0, 1]

        x = self.embed(tokens) + self.pos_embed(pos)  # (N, 2, d_model)
        x = self.transformer_layer(x)                 # (N, 2, d_model)
        x = x.mean(dim=1)                             # (N, d_model)  — mean pooling
        return self.out(x)                            # (N, p)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def l2_norm(self) -> float:
        return math.sqrt(sum(p.pow(2).sum().item() for p in self.parameters()))
