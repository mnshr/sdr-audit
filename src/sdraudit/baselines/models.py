from typing import List, Optional
import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Multi-Layer Perceptron feature extractor mapping raw inputs to latent embeddings."""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 4,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        curr_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h

        layers.append(nn.Linear(curr_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
