from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn


class SurgicallyPatchedModel(nn.Module):
    """PyTorch wrapper implementing surgical model repair.

    Augments a frozen base encoder f_theta(X) in R^d with a linear projection
    onto the residual subspace B_res in R^{p x k}:
        Z_repaired = [ f_theta(X), B_res^T X ] in R^{d + k}

    This augmented representation is then fed into a lightweight prediction head.
    """

    def __init__(
        self,
        base_encoder: Optional[nn.Module],
        p_dim: int,
        d_dim: int,
        k_dim: int,
        B_res: Union[np.ndarray, torch.Tensor],
        out_dim: int = 1,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.p_dim = p_dim
        self.d_dim = d_dim
        self.k_dim = k_dim

        # Base encoder (frozen by default)
        self.base_encoder = base_encoder
        if self.base_encoder is not None and freeze_base:
            for p in self.base_encoder.parameters():
                p.requires_grad = False

        # Residual projection layer (fixed or fine-tunable)
        if isinstance(B_res, np.ndarray):
            B_tensor = torch.tensor(B_res, dtype=torch.float32)
        else:
            B_tensor = B_res.clone().detach().to(torch.float32)

        self.register_buffer("B_res", B_tensor)

        # Lightweight prediction head on concatenated representation [Z, B_res^T X]
        self.head = nn.Sequential(
            nn.Linear(d_dim + k_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(
        self,
        X: torch.Tensor,
        Z_precomputed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the surgically patched model.

        Args:
            X: Raw input tensor of shape (batch_size, p_dim).
            Z_precomputed: Optional precomputed embeddings of shape (batch_size, d_dim)
                if base_encoder is None or skipped.

        Returns:
            Output prediction tensor of shape (batch_size, out_dim).
        """
        if Z_precomputed is not None:
            Z = Z_precomputed
        elif self.base_encoder is not None:
            with torch.no_grad():
                Z = self.base_encoder(X)
        else:
            raise ValueError("Either base_encoder or Z_precomputed must be provided.")

        # Compute residual features B_res^T X
        Z_res = torch.matmul(X, self.B_res)

        # Concatenate: [Z, Z_res]
        Z_augmented = torch.cat([Z, Z_res], dim=-1)

        return self.head(Z_augmented)
