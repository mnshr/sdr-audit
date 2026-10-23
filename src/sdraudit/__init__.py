"""
sdraudit: Sufficient Dimension Reduction for Pretrained Model Auditing,
Residual Subspace Repair, and Uncertainty Quantification.
"""

__version__ = "0.1.0"

from sdraudit.audit.base import AuditResult, BaseAuditor
from sdraudit.audit.kernel_ci import KernelCIAuditor
from sdraudit.audit.dcov import DistanceCovarianceAuditor
from sdraudit.repair.residual_subspace import ResidualSubspaceEstimator
from sdraudit.repair.patched_model import SurgicallyPatchedModel
from sdraudit.uncertainty.variance_subspace import CentralVarianceAuditor
from sdraudit.uncertainty.tail_audit import TailSufficiencyAuditor
from sdraudit.data.generator import SemiSyntheticGenerator
from sdraudit.data.gtex_loader import GTExDataLoader

__all__ = [
    "AuditResult",
    "BaseAuditor",
    "KernelCIAuditor",
    "DistanceCovarianceAuditor",
    "ResidualSubspaceEstimator",
    "SurgicallyPatchedModel",
    "CentralVarianceAuditor",
    "TailSufficiencyAuditor",
    "SemiSyntheticGenerator",
    "GTExDataLoader",
]
