# Literature Review & Screening: Representation Auditing, SDR & Uncertainty

This document synthesizes the systematic screening of 17 foundational and state-of-the-art papers relevant to **pretrained representation auditing**, **residual subspace repair**, and **uncertainty preservation via SDR**.

---

## 1. Classification & Methodological Taxonomies

```
Total Validated Papers: 17
├── Pillar 1: Statistical Sufficiency Auditing (5)
│   ├── Tang & Li (2026) — JMVA
│   ├── Liu, Zhang & Li (2026) — Statistics and Computing
│   ├── Dong, McAlinn, Airoldi & Li (2026) — FlowSDR (arXiv)
│   ├── Mai, Shao, Wang & Zhang (2025) — Statistica Sinica
│   └── Li, Jones & Artemiou (2025) — Relative Universality (arXiv)
├── Pillar 2: Transfer Learning & Representation Repair (5)
│   ├── Ge, Zhou & Huang (2025) — TESR (arXiv)
│   ├── Jiao, Lin, Luo & Yang (2024) — Deep Transfer Learning (arXiv)
│   ├── Zheng, Lin & Huang (2022) — MSRL (arXiv)
│   ├── Jo, Lee, Shin & Ro (2025) — MEASURE (arXiv / OpenReview)
│   └── Xu, Yu & Huang (2025) — GenSDR (arXiv)
└── Pillar 3: Uncertainty Quantification & Variance Subspaces (7)
    ├── Fertl & Bura (2022a) — CVE (Bernoulli)
    ├── Fertl & Bura (2022b) — ECVE (EJS)
    ├── Aghbalou, Portier, Sabourin & Zhou (2024) — TIREX (Bernoulli)
    ├── Huang, Yu & Zhang (2025) — α-dCov SDR (J Nonparametr Stat)
    ├── Hong, Kim & Song (2024) — Hellinger SDR (ICML PMLR 235)
    ├── Soale & Dong (2022) — PALS (J Nonparametr Stat)
    └── Soale & Dong (2021) — EA-SIR (JSPI)
```

---

## 2. Pillar 1: Statistical Sufficiency Auditing

Focus: Testing the hypothesis $H_ {0}: Y \perp\!\!\perp \mathbf{X} \mid \mathbf{Z}$ where $\mathbf{Z} = f_ {\boldsymbol{\theta}}(\mathbf{X})$.

| Paper Title | Authors | Venue / Status | Key Contribution |
|---|---|---|---|
| *A Reproducing-Kernel-Based Nonparametric Test for Conditional Independence of Functional Data* | Tang & Li (2026) | *Journal of Multivariate Analysis*<br>[DOI: 10.1016/j.jmva.2026.105709](https://doi.org/10.1016/j.jmva.2026.105709) | RKHS conditional independence operator test with asymptotic Gaussian null distributions. |
| *Sufficient Dimension Reduction via Inverse Conditional Mean or Variance Independence* | Liu, Zhang & Li (2026) | *Statistics and Computing*<br>[DOI: 10.1007/s11222-026-10943-4](https://doi.org/10.1007/s11222-026-10943-4) | Inverse conditioning criteria for mean and variance subspaces under relaxed distributional assumptions. |
| *FlowSDR: Sufficient Dimension Reduction via Conditional Normalizing Flows* | Dong, McAlinn, Airoldi & Li (2026) | arXiv:2606.01346 | Exact conditional density modeling via continuous flows; likelihood-ratio auditing of representation sufficiency. |
| *Slicing-free Inverse Regression in High-dimensional Sufficient Dimension Reduction* | Mai, Shao, Wang & Zhang (2025) | *Statistica Sinica* 35(1): 1–23<br>[DOI: 10.5705/ss.202022.0112](https://doi.org/10.5705/ss.202022.0112) | High-dimensional slicing-free SDR resolving slice-selection arbitrariness. |
| *On Relative Universality, Regression Operator, and Conditional Independence* | Li, Jones & Artemiou (2025) | arXiv:2504.11044 | Universal regression operators characterizing conditional independence in infinite-dimensional RKHS. |

---

## 3. Pillar 2: Transfer Learning & Representation Repair

Focus: Transferable SDR, bounding domain shift error, and residual subspace estimation.

| Paper Title | Authors | Venue / Status | Key Contribution |
|---|---|---|---|
| *Transfer Learning through Enhanced Sufficient Representation (TESR)* | Ge, Zhou & Huang (2025) | arXiv:2502.20414 | Decomposes predictor spaces into shared transferable directions and task-specific residual directions. |
| *Deep Transfer Learning: Model Framework and Error Analysis* | Jiao, Lin, Luo & Yang (2024) | arXiv:2410.09383 | Non-asymptotic generalization bounds for deep transfer representations under structural discrepancy. |
| *Deep Sufficient Representation Learning via Mutual Information* | Zheng, Lin & Huang (2022) | arXiv:2207.10772 | Variational mutual information lower bounds enforcing minimal representation sufficiency. |
| *MEASURE: Multi-scale Minimal Sufficient Representation Learning* | Jo, Lee, Shin & Ro (2025) | arXiv:2510.12070 / OpenReview | Multi-scale information-theoretic representations discarding task-irrelevant nuisance factors. |
| *On Conditional Stochastic Interpolation for Generative Nonlinear SDR* | Xu, Yu & Huang (2025) | arXiv:2512.18971 | Generative flow-based nonlinear SDR mapping arbitrary conditioning variables to standard priors. |

---

## 4. Pillar 3: Uncertainty Quantification & Variance Subspaces

Focus: Central Variance Subspaces $\mathcal{S}_ {\mathrm{Var}(Y \mid \mathbf{X})}$, heteroscedasticity, and extreme tail sufficiency.

| Paper Title | Authors | Venue / Status | Key Contribution |
|---|---|---|---|
| *Conditional Variance Estimator for Sufficient Dimension Reduction* | Fertl & Bura (2022a) | *Bernoulli* 28(3): 1957–1983<br>[DOI: 10.3150/21-BEJ1402](https://doi.org/10.3150/21-BEJ1402) | Model-free estimator for the Central Variance Subspace without slicing or linearity conditions. |
| *The Ensemble Conditional Variance Estimator for SDR* | Fertl & Bura (2022b) | *Electronic Journal of Statistics* 16(1): 2405–2436<br>[DOI: 10.1214/22-EJS1994](https://doi.org/10.1214/22-EJS1994) | Ensemble bootstrap aggregation for variance subspace recovery in small-sample regimes. |
| *Tail Inverse Regression: Dimension Reduction for Prediction of Extremes* | Aghbalou, Portier, Sabourin & Zhou (2024) | *Bernoulli* 30(4): 2977–3008<br>[DOI: 10.3150/23-BEJ1686](https://doi.org/10.3150/23-BEJ1686) | Semi-parametric dimension reduction targeting extreme value conditional tails. |
| *Robust Sufficient Dimension Reduction via α-Distance Covariance* | Huang, Yu & Zhang (2025) | *Journal of Nonparametric Statistics* 37(1): 1–25<br>[DOI: 10.1080/10485252.2024.2313137](https://doi.org/10.1080/10485252.2024.2313137) | Heavy-tail robust SDR using generalized alpha-powers of distance matrices. |
| *Sufficient Dimension Reduction with Hellinger Correlation* | Hong, Kim & Song (2024) | *ICML 2024*<br>[PMLR 235: 18634–18647](https://proceedings.mlr.press/v235/hong24a.html) | Bounded affinity measure for SDR resilient to outliers and severe heavy tails. |
| *Principal Asymmetric Least Squares Regression* | Soale & Dong (2022) | *Journal of Nonparametric Statistics* 34(3): 586–613<br>[DOI: 10.1080/10485252.2021.2025237](https://doi.org/10.1080/10485252.2021.2025237) | Expectile dimension reduction capturing asymmetric heteroscedastic spread. |
| *Expectile-Assisted Sliced Inverse Regression* | Soale & Dong (2021) | *JSPI* 214: 88–101<br>[DOI: 10.1016/j.jspi.2020.12.006](https://doi.org/10.1016/j.jspi.2020.12.006) | Joint recovery of central mean and central variance subspaces via expectile slicing. |
