"""
model.py — PyTorch single-hidden-layer MLP for the double-descent sweep.

Architecture: Linear(D_in, H) -> ReLU -> Linear(H, D_out)

Design choices:
  - No batch normalization, no dropout: either would add implicit regularization
    and alter the effective capacity as a function of width, confounding H*_seed.
  - PyTorch default (Kaiming uniform) weight initialization for ReLU layers.
  - Output is raw logits; cross-entropy loss is applied externally.
  - Parameter count: D_in*H + H + H*D_out + D_out  =  (D_in + D_out + 1)*H + D_out
    For D_in=64, D_out=10: n_params = 75*H + 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Two-weight-matrix MLP (one hidden layer).

    Parameters
    ----------
    D_in  : int — input dimension (64 for sklearn digits)
    H     : int — hidden width (swept across [4, 8, 16, 32, 64, 128, 256, 512])
    D_out : int — number of output classes (10 for digits)
    """

    def __init__(self, D_in: int, H: int, D_out: int):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

    def n_params(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters())

    def l2_norm(self) -> float:
        """L2 norm of all parameters concatenated into a single vector."""
        return torch.sqrt(
            sum(p.pow(2).sum() for p in self.parameters())
        ).item()
