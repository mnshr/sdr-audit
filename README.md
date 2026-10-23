# `sdr-audit`: Sufficient Dimension Reduction for Pretrained Model Auditing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**`sdr-audit`** is a Python framework that bridges **Sufficient Dimension Reduction (SDR)** and **deep transfer learning**. It provides non-parametric statistical tools to audit frozen representations, detect uncaptured residual subspaces, surgically repair models without full retraining, and certify uncertainty preservation.

---

## 1. Problem Formulation & Motivation

In modern transfer learning, a foundation or pretrained network maps raw inputs $\mathbf{X} \in \mathbb{R}^p$ to a latent representation $\mathbf{Z} = f_ {\boldsymbol{\theta}}(\mathbf{X}) \in \mathbb{R}^d$. When deploying to a new environment $(\mathbf{X}_ {\mathrm{new}}, Y_ {\mathrm{new}})$, practitioners typically evaluate point-prediction error (e.g., test MSE). If performance drops, the standard reaction is brute-force retraining or full fine-tuning.

This paradigm exhibits three major statistical failure modes:
1. **Prediction Error $\neq$ Statistical Sufficiency**: A representation can achieve low point-prediction error while discarding critical non-linear dependencies, violating conditional independence $Y \perp\!\!\perp \mathbf{X} \mid \mathbf{Z}$.
2. **Wastefulness of Full Retraining**: Deficiencies on new domains are often confined to a low-dimensional **residual subspace** $\mathcal{S}_ {\mathrm{res}} \subset \mathbb{R}^p$. Detecting $\mathcal{S}_ {\mathrm{res}}$ allows **surgical model repair** $(\mathbf{Z}_ {\mathrm{repaired}} = [\mathbf{Z}, \mathbf{B}_ {\mathrm{res}}^\top \mathbf{X}])$ at minimal compute cost.
3. **The Stability / Uncertainty Gap**: Optimizing point prediction captures only the conditional mean $E(Y \mid \mathbf{X})$ (the Central Mean Subspace). It frequently discards feature directions governing **conditional variance $\mathrm{Var}(Y \mid \mathbf{X})$ (heteroscedasticity)** or extreme tail risk, rendering prediction intervals uncalibrated.

---

## 2. Core Methodological Pillars

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  sdr-audit Framework                                   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            ▼                               ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│        Phase 1        │       │        Phase 2        │       │        Phase 3        │
│  Sufficiency Auditing │  ──▶  │    Residual Repair    │  ──▶  │   UQ & Variance SDR   │
│                       │       │                       │       │                       │
│ Test: Y ⫫ X | Z       │       │ Detect S_res in R^p   │       │ Central Variance Sub. │
│ (Kernel CI / dCov)    │       │ Augment: [Z, B_res X] │       │ & Tail Preservation   │
└───────────────────────┘       └───────────────────────┘       └───────────────────────┘
```

### Phase 1: Statistical Sufficiency Auditing

Tests whether the frozen embedding $\mathbf{Z} = f_ {\boldsymbol{\theta}}(\mathbf{X})$ preserves all information that $\mathbf{X}$ carries about $Y$:

$$
H_0: Y \perp\!\!\perp \mathbf{X} \mid \mathbf{Z} \quad \text{versus} \quad H_1: Y \not\perp\!\!\perp \mathbf{X} \mid \mathbf{Z}
$$

Implemented via model-free Kernel Conditional Independence (`KernelCIAuditor`) and Distance Covariance (`DistanceCovarianceAuditor`).

### Phase 2: Residual Subspace Detection & Surgical Model Repair

When $H_0$ is rejected, detects the missing $k$-dimensional subspace $\mathcal{S}_ {\mathrm{res}} \subset \mathbb{R}^p$ such that:

$$
Y \perp\!\!\perp \mathbf{X} \mid \left( \mathbf{Z}, \; \mathbf{P}_ {\mathcal{S}_ {\mathrm{res}}} \mathbf{X} \right)
$$

Wraps the frozen trunk and low-rank projection $\mathbf{B}_ {\mathrm{res}}^\top \mathbf{X}$ into `SurgicallyPatchedModel`, avoiding complete backpropagation.

### Phase 3: Higher-Order Moments & Uncertainty Preservation (UQ)
Audits whether representations preserve the **Central Variance Subspace (CVS)** $\mathcal{S}_ {\mathrm{Var}(Y \mid \mathbf{X})}$ or tail behavior for calibrated prediction intervals and conformal inference.

---

## 3. Directory Layout

```text
sdr-audit/
├── pyproject.toml              # Build & dependency specifications
├── README.md                   # Project overview & quickstart
├── LICENSE                     # MIT License
├── src/
│   └── sdraudit/
│       ├── audit/              # Phase 1: Conditional independence / sufficiency testing
│       │   ├── base.py         # Abstract BaseAuditor interface
│       │   ├── kernel_ci.py    # Kernel Conditional Independence (KCI) test
│       │   └── dcov.py         # Conditional distance covariance test
│       ├── repair/             # Phase 2: Residual subspace detection & model repair
│       │   ├── residual_subspace.py  # S_res estimator
│       │   └── patched_model.py      # SurgicallyPatchedModel wrapper (PyTorch)
│       ├── uncertainty/        # Phase 3: Central Variance Subspace & tail sufficiency
│       │   ├── variance_subspace.py  # Conditional variance estimator (CVE)
│       │   └── tail_audit.py         # Tail & extreme value sufficiency (TIREX)
│       ├── data/               # GTEx gene expression & semi-synthetic generator
│       │   ├── gtex_loader.py  # GTEx transcriptomics loader
│       │   └── generator.py    # Semi-synthetic null / shift / variance generator
│       └── baselines/          # Feature extractors and benchmark architectures
├── tests/                      # Pytest suite
├── docs/                       # Research literature review and key mathematical terms
└── refs/                       # Curated BibTeX references
```

---

## 4. Quickstart

### Installation

```bash
git clone https://github.com/mnshr/sdr-audit.git
cd sdr-audit
pip install -e ".[dev]"
```

### Auditing a Pretrained Representation

```python
import torch
from sdraudit.audit import KernelCIAuditor
from sdraudit.data import SemiSyntheticGenerator

# 1. Generate semi-synthetic data (e.g. from empirical transcriptomic predictors)
gen = SemiSyntheticGenerator(p=50, d=2, random_state=42)
X, Y, Z_suff = gen.generate(n_samples=500, setting="null")

# 2. Audit whether representation Z is statistically sufficient
auditor = KernelCIAuditor(alpha=0.05, n_permutations=200)
result = auditor.audit(X, Y, Z_suff)

print(f"Test Statistic: {result.statistic:.4f}")
print(f"p-value:        {result.p_value:.4f}")
print(f"Is Sufficient:  {not result.reject_h0}")
```

### Surgically Repairing a Representation

```python
from sdraudit.repair import ResidualSubspaceEstimator, SurgicallyPatchedModel

# If H0 is rejected on a shifted target dataset:
X_target, Y_target, Z_target = gen.generate(n_samples=500, setting="mean_shift")

# Estimate the residual subspace of rank k=1
repairer = ResidualSubspaceEstimator(n_components=1)
B_res = repairer.fit(X_target, Y_target, Z_target)

# Wrap into a surgically augmented PyTorch model
patched_model = SurgicallyPatchedModel(
    base_encoder=None,  # Or pass an existing nn.Module
    p_dim=50,
    d_dim=2,
    k_dim=1,
    B_res=B_res
)
```

---

## 5. References & Theoretical Foundations

This repository implements and builds upon key methodologies from statistical dimension reduction and representation learning:

- **Sufficiency Auditing**: Tang & Li (*JMVA*, 2026); Liu, Zhang & Li (*Stat Comput*, 2026); Dong et al. (*FlowSDR*, 2026).
- **Transfer & Representation**: Ge, Zhou & Huang (*TESR*, 2025); Chen et al. (*DDR*, 2024); Zheng et al. (2022); Jo et al. (*MEASURE*, 2025).
- **Uncertainty & Variance Subspaces**: Cook (*CVS*, 2002); Fertl & Bura (*Bernoulli*, 2022); Aghbalou et al. (*Bernoulli*, 2024); Huang et al. (*J Nonparametr Stat*, 2025).

See [`docs/literature_review.md`](docs/literature_review.md) for full citations and detailed taxonomy.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
