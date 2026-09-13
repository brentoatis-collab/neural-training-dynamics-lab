"""
mlp.py — Hand-written two-layer MLP with explicit forward and backward passes.

Architecture:
    X  →  Linear(D_in → H)  →  ReLU  →  Linear(H → D_out)  →  MSE loss

No autograd is used anywhere in this file. Every gradient is derived from
first principles and computed by explicit tensor operations.

Loss function:
    L = mean((z2 - y)^2)

where mean() divides by N * D_out (total number of scalar predictions).
MSE is chosen over cross-entropy because the gradient of MSE w.r.t. the
pre-output activations is a single closed-form expression with no softmax
numerics, making the backward pass fully transparent.
"""

import torch


class TwoLayerMLP:
    """
    Two-layer MLP with hand-written backward pass.

    Parameters are stored as plain torch.Tensors with requires_grad=False.
    The forward() method caches all intermediate values needed for backward().
    The backward() method computes gradients purely by chain rule.

    Attributes
    ----------
    W1 : Tensor (D_in, H)
    b1 : Tensor (H,)
    W2 : Tensor (H, D_out)
    b2 : Tensor (D_out,)
    dtype : torch.dtype  — must be float64 for gradient verification
    """

    def __init__(self, D_in: int, H: int, D_out: int,
                 dtype: torch.dtype = torch.float64,
                 seed: int = 42):
        torch.manual_seed(seed)

        # Xavier-style initialization: scale by 1/sqrt(fan_in) so that
        # activations are neither too large nor too small at the start.
        self.W1 = torch.randn(D_in, H, dtype=dtype) / D_in ** 0.5
        self.b1 = torch.zeros(H, dtype=dtype)
        self.W2 = torch.randn(H, D_out, dtype=dtype) / H ** 0.5
        self.b2 = torch.zeros(D_out, dtype=dtype)

        self.dtype = dtype
        self._cache: dict = {}   # populated by forward(); consumed by backward()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def parameters(self) -> list[tuple[str, torch.Tensor]]:
        """Return (name, tensor) pairs in a consistent order."""
        return [("W1", self.W1), ("b1", self.b1),
                ("W2", self.W2), ("b2", self.b2)]

    def total_params(self) -> int:
        return sum(t.numel() for _, t in self.parameters())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the MSE loss for inputs X and targets y.

        Caches all intermediate tensors required by backward().

        Args
        ----
        X : (N, D_in)  — input batch
        y : (N, D_out) — target batch

        Returns
        -------
        loss : scalar Tensor — mean squared error over all N * D_out predictions
        """
        # --- Layer 1: linear transform ---
        # z1[n, h] = sum_i( X[n, i] * W1[i, h] ) + b1[h]
        z1 = X @ self.W1 + self.b1          # (N, H)

        # --- ReLU activation ---
        # a1[n, h] = max(0, z1[n, h])
        a1 = torch.clamp(z1, min=0.0)       # (N, H)

        # --- Layer 2: linear transform ---
        # z2[n, k] = sum_h( a1[n, h] * W2[h, k] ) + b2[k]
        z2 = a1 @ self.W2 + self.b2         # (N, D_out)

        # --- MSE loss ---
        # L = (1 / N*D_out) * sum_{n,k} (z2[n,k] - y[n,k])^2
        # torch.mean() on a 2-D tensor divides by N * D_out.
        diff = z2 - y                        # (N, D_out)
        loss = (diff ** 2).mean()            # scalar

        # Cache everything backward() will need.
        # Stored without detach so that tests can inspect them,
        # but none of these tensors have requires_grad=True.
        self._cache = {
            "X": X, "y": y,
            "z1": z1, "a1": a1,
            "z2": z2, "diff": diff,
            "N": X.shape[0],
            "D_out": z2.shape[1],
        }

        return loss

    # ------------------------------------------------------------------
    # Backward pass — the part that is manually verified
    # ------------------------------------------------------------------

    def backward(self) -> dict[str, torch.Tensor]:
        """
        Compute gradients of the MSE loss w.r.t. all four parameters.

        Must be called immediately after forward() on the same (X, y).

        Returns
        -------
        grads : dict mapping param name → gradient tensor (same shape as param)

        Chain-rule derivation
        ---------------------
        Let N = batch size, D_out = output dimension.

        SECTION A — gradient at the output:
            L = mean(diff^2) where diff = z2 - y
            dL/dz2 = 2 * diff / (N * D_out)          shape: (N, D_out)

        SECTION B — gradients for W2 and b2:
            z2 = a1 @ W2 + b2
            dL/dW2 = a1.T  @ dL/dz2                  shape: (H, D_out)
            dL/db2 = sum over batch dim of dL/dz2     shape: (D_out,)

        SECTION C — backprop through layer 2 into a1:
            dL/da1 = dL/dz2 @ W2.T                   shape: (N, H)

        SECTION D — backprop through ReLU:
            a1 = clamp(z1, 0)  →  da1/dz1 = 1 if z1 > 0, else 0
            dL/dz1 = dL/da1 * (z1 > 0)               shape: (N, H)
            Note: the subgradient at z1 == 0 is defined as 0 here,
            consistent with PyTorch's convention.

        SECTION E — gradients for W1 and b1:
            z1 = X @ W1 + b1
            dL/dW1 = X.T  @ dL/dz1                   shape: (D_in, H)
            dL/db1 = sum over batch dim of dL/dz1     shape: (H,)
        """
        c = self._cache
        if not c:
            raise RuntimeError("backward() called before forward()")

        X     = c["X"]
        z1    = c["z1"]
        a1    = c["a1"]
        diff  = c["diff"]
        N     = c["N"]
        D_out = c["D_out"]

        # SECTION A: gradient of loss w.r.t. z2
        d_z2 = (2.0 / (N * D_out)) * diff          # (N, D_out)

        # SECTION B: gradients for output-layer parameters
        d_W2 = a1.T @ d_z2                          # (H, D_out)
        d_b2 = d_z2.sum(dim=0)                      # (D_out,)

        # SECTION C: propagate gradient back through layer 2
        d_a1 = d_z2 @ self.W2.T                     # (N, H)

        # SECTION D: gradient through ReLU
        # Cast the boolean mask to the same dtype as d_a1 so multiplication
        # is well-defined and stays in float64.
        relu_mask = (z1 > 0).to(self.dtype)         # (N, H), 1.0 or 0.0
        d_z1 = d_a1 * relu_mask                     # (N, H)

        # SECTION E: gradients for first-layer parameters
        d_W1 = X.T @ d_z1                           # (D_in, H)
        d_b1 = d_z1.sum(dim=0)                      # (H,)

        return {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
