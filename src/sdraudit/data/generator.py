from typing import Literal, Optional, Tuple, Union
import numpy as np


class SemiSyntheticGenerator:
    """Semi-synthetic data generator for controlled representation auditing experiments.

    Supports three experimental configurations:
    1. 'null': Target data shares the identical sufficient subspace S_0.
       The representation Z is fully sufficient: Y ⫫ X | Z.
    2. 'mean_shift': Target data activates an uncaptured residual direction B_res.
       The representation Z is deficient in conditional mean: Y ̸⫫ X | Z.
    3. 'variance_shift': Target data conditional mean is preserved, but conditional
       variance Var(Y|X) is modulated by an orthogonal direction B_var.
       The representation Z satisfies mean sufficiency but fails variance sufficiency.
    """

    def __init__(
        self,
        p: int = 50,
        d: int = 2,
        k_res: int = 1,
        noise_std: float = 0.5,
        random_state: Optional[int] = None,
    ):
        self.p = p
        self.d = d
        self.k_res = k_res
        self.noise_std = noise_std
        self.random_state = random_state

        rng = np.random.default_rng(self.random_state)

        # Draw orthogonal true subspace directions
        full_basis, _ = np.linalg.qr(rng.standard_normal((p, d + k_res + 1)))
        self.B_0 = full_basis[:, :d]                  # Shared sufficient directions
        self.B_res = full_basis[:, d : d + k_res]     # Uncaptured residual direction
        self.B_var = full_basis[:, [d + k_res]]       # Variance-modulating direction

    def generate(
        self,
        n_samples: int = 500,
        setting: Literal["null", "mean_shift", "variance_shift"] = "null",
        X_empirical: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates (X, Y, Z) under the chosen experimental setting.

        Args:
            n_samples: Number of observations to generate (if X_empirical is None).
            setting: One of 'null', 'mean_shift', or 'variance_shift'.
            X_empirical: Optional real predictor matrix (e.g., from GTEx).

        Returns:
            X: Predictor matrix (n_samples, p).
            Y: Response vector (n_samples, 1).
            Z: Evaluated representation (n_samples, d), mapping X via B_0.
        """
        rng = np.random.default_rng(self.random_state)

        if X_empirical is not None:
            X = np.asarray(X_empirical)
            n_samples, p_empirical = X.shape
            if p_empirical != self.p:
                raise ValueError(f"X_empirical dimension {p_empirical} != generator p {self.p}")
        else:
            # Correlated non-Gaussian synthetic predictors
            cov = np.fromfunction(
                lambda i, j: 0.5 ** np.abs(i - j), (self.p, self.p)
            )
            X = rng.multivariate_normal(np.zeros(self.p), cov, size=n_samples)

        # Frozen representation captures B_0: Z = X @ B_0
        Z = X @ self.B_0

        eps = rng.normal(0, self.noise_std, size=(n_samples, 1))

        # Base link function on Z: e.g. rational + interaction
        z1 = Z[:, [0]]
        z2 = Z[:, [1]] if self.d > 1 else z1
        f_base = z1 / (0.5 + (z2 + 1.0) ** 2)

        if setting == "null":
            # Y depends strictly on Z
            Y = f_base + eps

        elif setting == "mean_shift":
            # Y depends on both Z and uncaptured direction B_res
            z_res = X @ self.B_res
            f_shift = 1.5 * np.sin(z_res)
            Y = f_base + f_shift + eps

        elif setting == "variance_shift":
            # Conditional mean depends strictly on Z, but variance depends on B_var
            z_var = X @ self.B_var
            hetero_scale = np.exp(0.8 * np.tanh(z_var))
            Y = f_base + hetero_scale * eps

        else:
            raise ValueError(f"Unknown setting '{setting}'. Must be 'null', 'mean_shift', or 'variance_shift'.")

        return X, Y, Z
