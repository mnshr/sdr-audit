from typing import Optional, Union
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import torch

from sdraudit.audit.base import AuditResult, BaseAuditor
from sdraudit.audit.kernel_ci import KernelCIAuditor


class CentralVarianceAuditor(BaseAuditor):
    """Auditor for Central Variance Subspace (CVS) preservation.

    Tests whether a representation Z captures the conditional variance
    Var(Y | X) in addition to the conditional mean E(Y | X).

    Theoretical formulation:
        Let e^2 = (Y - E(Y | Z))^2 be the squared prediction residual.
        If Z is sufficient for the conditional variance, then:
            H0: e^2 ⫫ X | Z

    Rejection of H0 indicates that X carries additional information about
    heteroscedasticity / prediction uncertainty that Z has discarded,
    invalidating downstream confidence intervals and conformal prediction sets.

    References:
        - Cook (2002), Dimension reduction for conditional variance in summary.
        - Fertl & Bura (2022), Conditional variance estimator for SDR (Bernoulli).
        - Fertl & Bura (2022), The ensemble conditional variance estimator (EJS).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_permutations: int = 200,
        random_state: Optional[int] = None,
    ):
        super().__init__(alpha=alpha, random_state=random_state)
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
        Y_arr = self._to_numpy(Y)
        Z_arr = self._to_numpy(Z)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)

        # 1. Estimate conditional mean E(Y | Z) via non-linear ridge regression
        kr = KernelRidge(kernel="rbf", alpha=1e-2)
        y_hat = kr.fit(Z_arr, Y_arr).predict(Z_arr)

        # 2. Form squared residual e^2 = (Y - y_hat)^2 reflecting conditional variance
        e_sq = (Y_arr - y_hat) ** 2

        # 3. Test conditional independence: H0: e^2 ⫫ X | Z
        kci_res = self.kci.audit(X_arr, e_sq, Z_arr)

        details = kci_res.details.copy()
        details["mean_residual_var"] = float(np.mean(e_sq))
        details["std_residual_var"] = float(np.std(e_sq))

        return AuditResult(
            statistic=kci_res.statistic,
            p_value=kci_res.p_value,
            reject_h0=kci_res.reject_h0,
            alpha=self.alpha,
            details=details,
        )
