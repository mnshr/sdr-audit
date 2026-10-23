import numpy as np
import pytest
import torch

from sdraudit.repair.residual_subspace import ResidualSubspaceEstimator
from sdraudit.repair.patched_model import SurgicallyPatchedModel
from sdraudit.data.generator import SemiSyntheticGenerator


def test_residual_subspace_estimator():
    gen = SemiSyntheticGenerator(p=15, d=2, k_res=1, random_state=42)
    X, Y, Z = gen.generate(n_samples=120, setting="mean_shift")

    estimator = ResidualSubspaceEstimator(n_components=1)
    B_res = estimator.fit(X, Y, Z)

    assert B_res.shape == (15, 1)
    # Check orthonormality
    assert np.allclose(B_res.T @ B_res, np.eye(1), atol=1e-5)

    # Check transform
    Z_res = estimator.transform(X)
    assert Z_res.shape == (120, 1)


def test_surgically_patched_model_forward():
    p, d, k = 20, 3, 2
    n = 16

    B_res = np.random.randn(p, k)
    model = SurgicallyPatchedModel(
        base_encoder=None,
        p_dim=p,
        d_dim=d,
        k_dim=k,
        B_res=B_res,
        out_dim=1,
    )

    X = torch.randn(n, p)
    Z = torch.randn(n, d)

    out = model(X, Z_precomputed=Z)
    assert out.shape == (n, 1)
