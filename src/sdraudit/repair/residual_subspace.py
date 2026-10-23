from typing import Optional, Union
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import torch


class ResidualSubspaceEstimator:
    """Estimator for the uncaptured residual sufficient subspace S_res.

    Identifies a k-dimensional basis B_res in R^{p x k} such that:
        Y ⫫ X | (Z, B_res^T X)

    The algorithm:
    1. Computes non-linear residuals X_res = X - E(X|Z) and Y_res = Y - E(Y|Z).
    2. Identifies the leading SDR directions connecting X_res to Y_res using
       inverse regression / covariance operator decomposition.
    """

    def __init__(self, n_components: int = 1, alpha_ridge: float = 1e-2):
        self.n_components = n_components
        self.alpha_ridge = alpha_ridge
        self.basis_: Optional[np.ndarray] = None

    @staticmethod
    def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        Z: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Fits the residual subspace basis B_res.

        Args:
            X: Raw predictors of shape (n_samples, p_dim).
            Y: Response of shape (n_samples, q_dim) or (n_samples,).
            Z: Pretrained embeddings of shape (n_samples, d_dim).

        Returns:
            Orthonormal basis matrix B_res of shape (p_dim, n_components).
        """
        X_arr = self._to_numpy(X)
        Y_arr = self._to_numpy(Y)
        Z_arr = self._to_numpy(Z)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)

        # Step 1: Project out effect of Z from both X and Y
        kr_x = KernelRidge(kernel="rbf", alpha=self.alpha_ridge)
        X_res = X_arr - kr_x.fit(Z_arr, X_arr).predict(Z_arr)

        kr_y = KernelRidge(kernel="rbf", alpha=self.alpha_ridge)
        Y_res = Y_arr - kr_y.fit(Z_arr, Y_arr).predict(Z_arr)

        # Step 2: Cross-covariance decomposition on residuals
        n, p = X_res.shape
        Sigma_xx = (X_res.T @ X_res) / n + 1e-4 * np.eye(p)
        Sigma_xy = (X_res.T @ Y_res) / n

        # Generalized inverse regression target: Sigma_xx^{-1} Sigma_xy Sigma_yx
        M = np.linalg.pinv(Sigma_xx) @ (Sigma_xy @ Sigma_xy.T)

        eigvals, eigvecs = np.linalg.eigh(M)
        top_indices = np.argsort(eigvals)[::-1][: self.n_components]
        B_raw = eigvecs[:, top_indices]

        # Orthonormalize via QR
        Q, _ = np.linalg.qr(B_raw)
        self.basis_ = Q[:, : self.n_components]
        return self.basis_

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Projects predictors onto the estimated residual subspace."""
        if self.basis_ is None:
            raise ValueError("Estimator has not been fitted. Call fit() first.")
        X_arr = self._to_numpy(X)
        return X_arr @ self.basis_
