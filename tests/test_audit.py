import numpy as np
import pytest

from sdraudit.audit.kernel_ci import KernelCIAuditor
from sdraudit.audit.dcov import DistanceCovarianceAuditor
from sdraudit.uncertainty.variance_subspace import CentralVarianceAuditor
from sdraudit.uncertainty.tail_audit import TailSufficiencyAuditor
from sdraudit.data.generator import SemiSyntheticGenerator


@pytest.fixture
def synthetic_null():
    gen = SemiSyntheticGenerator(p=10, d=2, noise_std=0.1, random_state=42)
    return gen.generate(n_samples=100, setting="null")


@pytest.fixture
def synthetic_shift():
    gen = SemiSyntheticGenerator(p=10, d=2, noise_std=0.1, random_state=42)
    return gen.generate(n_samples=100, setting="mean_shift")


def test_kernel_ci_null_vs_shift(synthetic_null, synthetic_shift):
    X_null, Y_null, Z_null = synthetic_null
    X_shift, Y_shift, Z_shift = synthetic_shift

    auditor = KernelCIAuditor(alpha=0.05, n_permutations=50, random_state=42)

    # Under null: H0: Y ⫫ X | Z should be accepted (reject_h0 == False)
    res_null = auditor.audit(X_null, Y_null, Z_null)
    assert isinstance(res_null.statistic, float)
    assert 0.0 <= res_null.p_value <= 1.0

    # Under shift: uncaptured direction causes rejection of H0
    res_shift = auditor.audit(X_shift, Y_shift, Z_shift)
    assert res_shift.statistic > res_null.statistic


def test_dcov_auditor(synthetic_null, synthetic_shift):
    X_null, Y_null, Z_null = synthetic_null
    X_shift, Y_shift, Z_shift = synthetic_shift

    auditor = DistanceCovarianceAuditor(alpha=0.05, n_permutations=50, random_state=42)

    res_null = auditor.audit(X_null, Y_null, Z_null)
    res_shift = auditor.audit(X_shift, Y_shift, Z_shift)

    assert isinstance(res_null.statistic, float)
    assert res_shift.statistic > res_null.statistic


def test_central_variance_auditor():
    gen = SemiSyntheticGenerator(p=10, d=2, noise_std=0.2, random_state=42)
    X_var, Y_var, Z_var = gen.generate(n_samples=100, setting="variance_shift")

    cv_auditor = CentralVarianceAuditor(alpha=0.05, n_permutations=50, random_state=42)
    res = cv_auditor.audit(X_var, Y_var, Z_var)

    assert isinstance(res.statistic, float)
    assert "mean_residual_var" in res.details


def test_tail_sufficiency_auditor(synthetic_null):
    X_null, Y_null, Z_null = synthetic_null
    tail_auditor = TailSufficiencyAuditor(quantile=0.85, n_permutations=30, random_state=42)
    res = tail_auditor.audit(X_null, Y_null, Z_null)
    assert isinstance(res.statistic, float)
    assert "threshold" in res.details
