from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import numpy as np
import torch


@dataclass
class AuditResult:
    """Container for sufficiency audit test outcomes.

    Attributes:
        statistic: Computed test statistic value.
        p_value: Empirical or asymptotic p-value under H0: Y ⫫ X | Z.
        reject_h0: Boolean flag indicating whether H0 is rejected at the chosen alpha level.
        alpha: Significance level (default: 0.05).
        details: Additional diagnostic metadata (e.g., null distribution values, bandwidths).
    """
    statistic: float
    p_value: float
    reject_h0: bool
    alpha: float = 0.05
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "REJECTED (Insufficient)" if self.reject_h0 else "ACCEPTED (Sufficient)"
        return (
            f"Sufficiency Audit Result:\n"
            f"  Status:         {status}\n"
            f"  Test Statistic: {self.statistic:.4f}\n"
            f"  p-value:        {self.p_value:.4f} (alpha = {self.alpha})\n"
            f"  Null:           H0: Y ⫫ X | Z"
        )


class BaseAuditor(ABC):
    """Abstract base class for conditional independence representation auditors."""

    def __init__(self, alpha: float = 0.05, random_state: Optional[int] = None):
        self.alpha = alpha
        self.random_state = random_state

    @staticmethod
    def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @abstractmethod
    def audit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        Z: Union[np.ndarray, torch.Tensor],
    ) -> AuditResult:
        """Audits the hypothesis H0: Y ⫫ X | Z.

        Args:
            X: Raw predictor matrix of shape (n_samples, p_dim).
            Y: Response vector/matrix of shape (n_samples, q_dim) or (n_samples,).
            Z: Pretrained representation of shape (n_samples, d_dim).

        Returns:
            AuditResult instance containing test statistic, p-value, and decision.
        """
        pass
