from typing import Optional, Union
import numpy as np
from scipy.spatial.distance import cdist
import torch

from sdraudit.audit.base import AuditResult, BaseAuditor


class DistanceCovarianceAuditor(BaseAuditor):
    """Distance Covariance (dCov) representation sufficiency auditor.

    Audits H0: Y ⫫ X | Z by evaluating distance covariance between the
    non-linear residuals of X and Y given Z.

    References:
        - Székely, Rizzo & Bakirov (2007), Measuring and testing dependence by correlation of distances.
        - Wang et al. (2015), Conditional distance correlation.
        - Huang, Yu & Zhang (2025), Robust SDR via α-Distance Covariance.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_permutations: int = 200,
        random_state: Optional[int] = None,
    ):
        super().__init__(alpha=alpha, random_state=random_state)
        self.n_permutations = n_permutations

    @staticmethod
    def _double_center_distance(D: np.ndarray) -> np.ndarray:
        n = D.shape[0]
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        total_mean = D.mean()
        return D - row_mean - col_mean + total_mean

    @staticmethod
    def _distance_matrix(data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return cdist(data, data, metric="euclidean")

    @classmethod
    def _dcov_sq(cls, A: np.ndarray, B: np.ndarray) -> float:
        n = A.shape[0]
        return float(np.sum(A * B) / (n * n))

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

        n = len(X_arr)

        # Regress X and Y on Z via kernel ridge / local projections to obtain residuals
        from sklearn.kernel_ridge import KernelRidge

        kr_x = KernelRidge(kernel="rbf", alpha=1e-2)
        res_X = X_arr - kr_x.fit(Z_arr, X_arr).predict(Z_arr)

        kr_y = KernelRidge(kernel="rbf", alpha=1e-2)
        res_Y = Y_arr - kr_y.fit(Z_arr, Y_arr).predict(Z_arr)

        A = self._double_center_distance(self._distance_matrix(res_X))
        B = self._double_center_distance(self._distance_matrix(res_Y))

        observed_stat = self._dcov_sq(A, B)

        rng = np.random.default_rng(self.random_state)
        null_stats = []
        for _ in range(self.n_permutations):
            perm = rng.permutation(n)
            A_perm = A[np.ix_(perm, perm)]
            null_stats.append(self._dcov_sq(A_perm, B))

        null_stats = np.array(null_stats)
        p_val = float(np.mean(null_stats >= observed_stat))
        reject = bool(p_val < self.alpha)

        return AuditResult(
            statistic=observed_stat,
            p_value=p_val,
            reject_h0=reject,
            alpha=self.alpha,
            details={
                "null_mean": float(np.mean(null_stats)),
                "null_std": float(np.std(null_stats)),
            },
        )
