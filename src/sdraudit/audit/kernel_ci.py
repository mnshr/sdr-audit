from typing import Optional, Union
import numpy as np
from scipy.spatial.distance import cdist
import torch

from sdraudit.audit.base import AuditResult, BaseAuditor


class KernelCIAuditor(BaseAuditor):
    """Kernel Conditional Independence (KCI) representation auditor.

    Tests the hypothesis H0: Y ⫫ X | Z using regularized kernel conditioning
    operators in Reproducing Kernel Hilbert Spaces (RKHS).

    References:
        - Fukumizu et al. (2008), Kernel measures of conditional dependence.
        - Zhang et al. (2011), Kernel-based conditional independence test.
        - Tang & Li (2026), RKHS test for conditional independence.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        regularization: float = 1e-3,
        n_permutations: int = 200,
        random_state: Optional[int] = None,
    ):
        super().__init__(alpha=alpha, random_state=random_state)
        self.regularization = regularization
        self.n_permutations = n_permutations

    @staticmethod
    def _median_bandwidth(data: np.ndarray) -> float:
        """Computes RBF bandwidth using the median pairwise Euclidean distance heuristic."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        subsample = data[:min(500, len(data))]
        dists = cdist(subsample, subsample, metric="euclidean")
        median_dist = np.median(dists[dists > 0])
        return float(median_dist) if median_dist > 0 else 1.0

    @staticmethod
    def _rbf_kernel(data: np.ndarray, sigma: float) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        dists_sq = cdist(data, data, metric="sqeuclidean")
        gamma = 1.0 / (2.0 * (sigma ** 2))
        return np.exp(-gamma * dists_sq)

    @staticmethod
    def _center_kernel(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def _compute_statistic(
        self, K_X: np.ndarray, K_Y: np.ndarray, K_Z: np.ndarray
    ) -> float:
        """Computes Tr(K_{X|Z} K_{Y|Z}) using regularized inversion on Z."""
        n = K_X.shape[0]
        R_Z = self.regularization * np.linalg.pinv(
            K_Z + self.regularization * np.eye(n)
        )
        K_X_given_Z = R_Z @ K_X @ R_Z
        K_Y_given_Z = R_Z @ K_Y @ R_Z
        stat = np.trace(K_X_given_Z @ K_Y_given_Z) / n
        return float(stat)

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
        sigma_x = self._median_bandwidth(X_arr)
        sigma_y = self._median_bandwidth(Y_arr)
        sigma_z = self._median_bandwidth(Z_arr)

        K_X = self._center_kernel(self._rbf_kernel(X_arr, sigma_x))
        K_Y = self._center_kernel(self._rbf_kernel(Y_arr, sigma_y))
        K_Z = self._rbf_kernel(Z_arr, sigma_z)

        observed_stat = self._compute_statistic(K_X, K_Y, K_Z)

        rng = np.random.default_rng(self.random_state)
        null_stats = []
        for _ in range(self.n_permutations):
            perm_idx = rng.permutation(n)
            K_X_perm = K_X[np.ix_(perm_idx, perm_idx)]
            null_stats.append(self._compute_statistic(K_X_perm, K_Y, K_Z))

        null_stats = np.array(null_stats)
        p_val = float(np.mean(null_stats >= observed_stat))
        reject = bool(p_val < self.alpha)

        return AuditResult(
            statistic=observed_stat,
            p_value=p_val,
            reject_h0=reject,
            alpha=self.alpha,
            details={
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "sigma_z": sigma_z,
                "null_mean": float(np.mean(null_stats)),
                "null_std": float(np.std(null_stats)),
            },
        )
