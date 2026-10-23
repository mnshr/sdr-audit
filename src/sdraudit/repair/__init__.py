"""Model repair subpackage for estimating residual subspaces and surgical patching."""

from sdraudit.repair.residual_subspace import ResidualSubspaceEstimator
from sdraudit.repair.patched_model import SurgicallyPatchedModel

__all__ = [
    "ResidualSubspaceEstimator",
    "SurgicallyPatchedModel",
]
