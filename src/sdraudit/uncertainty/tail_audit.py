from typing import Optional, Union
import numpy as np
import torch

from sdraudit.audit.base import AuditResult, BaseAuditor
from sdraudit.audit.kernel_ci import KernelCIAuditor


class TailSufficiencyAuditor(BaseAuditor):
    """Audits representation sufficiency in extreme response tails.

    Tests whether a representation Z preserves information governing extreme
    quantiles of Y (e.g., top 10% or bottom 10%), extending Tail Inverse
    Regression (TIREX) to representation evaluation.

    References:
        - Aghbalou, Portier, Sabourin & Zhou (2024), Tail inverse regression:
          Dimension reduction for prediction of extremes (Bernoulli).
    """

    def __init__(
        self,
        quantile: float = 0.90,
        alpha: float = 0.05,
        n_permutations: int = 200,
        random_state: Optional[int] = None,
    ):
        super().__init__(alpha=alpha, random_state=random_state)
        self.quantile = quantile
        self.n_permutations = n_permutations
        self.kci = KernelCIAuditor(
            alpha=alpha,
            n_permutations=n_permutations,
            random_state=random_state,
        )

    def audit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        Z: Union[np.ndarray, torch.Tensor],
    ) -> AuditResult:
        X_arr = self._to_numpy(X)
        Y_arr = self._to_numpy(Y).ravel()
        Z_arr = self._to_numpy(Z)

        # Compute extreme quantile threshold
        threshold = np.quantile(Y_arr, self.quantile)
        tail_indicator = (Y_arr >= threshold).astype(float).reshape(-1, 1)

        # Audit conditional independence of the tail event: H0: I(Y >= q) ⫫ X | Z
        kci_res = self.kci.audit(X_arr, tail_indicator, Z_arr)

        details = kci_res.details.copy()
        details["quantile"] = self.quantile
        details["threshold"] = float(threshold)
        details["tail_count"] = int(np.sum(tail_indicator))

        return AuditResult(
            statistic=kci_res.statistic,
            p_value=kci_res.p_value,
            reject_h0=kci_res.reject_h0,
            alpha=self.alpha,
            details=details,
        )
