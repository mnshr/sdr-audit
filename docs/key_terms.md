# Key Mathematical Concepts & Definitions

This reference document defines the core mathematical apparatus underpinning `sdr-audit`.

---

## 1. Statistical Sufficiency & Conditional Independence

### Definition 1.1 (Sufficient Representation)
Let $\mathbf{X} \in \mathbb{R}^p$ denote raw predictors, $Y \in \mathbb{R}^q$ a response variable, and $f_ {\boldsymbol{\theta}}: \mathbb{R}^p \to \mathbb{R}^d$ a deterministic or stochastic encoder yielding latent representation $\mathbf{Z} = f_ {\boldsymbol{\theta}}(\mathbf{X})$. The representation $\mathbf{Z}$ is **statistically sufficient** for $Y$ with respect to $\mathbf{X}$ if:

$$
Y \perp\!\!\perp \mathbf{X} \mid \mathbf{Z}
$$

Equivalently, the conditional distribution satisfies $P(Y \mid \mathbf{X}, \mathbf{Z}) = P(Y \mid \mathbf{Z})$ almost surely.

### Definition 1.2 (Central Subspace & Central Mean Subspace)
- **Central Subspace (CS)** $\mathcal{S}_ {Y \mid \mathbf{X}}$: The minimal linear subspace $\mathrm{span}(\mathbf{B}) \subset \mathbb{R}^p$ such that $Y \perp\!\!\perp \mathbf{X} \mid \mathbf{B}^\top \mathbf{X}$.
- **Central Mean Subspace (CMS)** $\mathcal{S}_ {E(Y \mid \mathbf{X})}$: The minimal linear subspace such that $E(Y \mid \mathbf{X}) = E(Y \mid \mathbf{B}_ {\mathrm{mean}}^\top \mathbf{X})$. Always satisfies $\mathcal{S}_ {E(Y \mid \mathbf{X})} \subseteq \mathcal{S}_ {Y \mid \mathbf{X}}$.

---

## 2. Residual Subspace & Surgical Model Repair

### Definition 2.1 (Residual Sufficient Subspace)
Let $\mathbf{Z} \in \mathbb{R}^d$ be a frozen, incomplete representation such that $Y \not\perp\!\!\perp \mathbf{X} \mid \mathbf{Z}$. A subspace $\mathcal{S}_ {\mathrm{res}} = \mathrm{span}(\mathbf{B}_ {\mathrm{res}}) \subset \mathbb{R}^p$ of dimension $k \ll p$ is a **residual sufficient subspace** if:

$$
Y \perp\!\!\perp \mathbf{X} \mid \left( \mathbf{Z}, \; \mathbf{B}_ {\mathrm{res}}^\top \mathbf{X} \right)
$$

### Definition 2.2 (Surgical Augmentation)
Instead of backpropagating gradients through the parameters $\boldsymbol{\theta}$ of the base encoder $f_ {\boldsymbol{\theta}}(\mathbf{X})$, surgical repair concatenates the linear projection $\mathbf{B}_ {\mathrm{res}}^\top \mathbf{X}$:

$$
\mathbf{Z}_ {\mathrm{repaired}} = \begin{bmatrix} f_ {\boldsymbol{\theta}}(\mathbf{X}) \\ \mathbf{B}_ {\mathrm{res}}^\top \mathbf{X} \end{bmatrix} \in \mathbb{R}^{d + k}
$$

and fits only a lightweight prediction head $g_ {\boldsymbol{\phi}}: \mathbb{R}^{d+k} \to \mathcal{Y}$.

---

## 3. Second-Order Moments & Uncertainty Quantification

### Definition 3.1 (Central Variance Subspace - CVS)
The **Central Variance Subspace** $\mathcal{S}_ {\mathrm{Var}(Y \mid \mathbf{X})}$ (Cook, 2002; Fertl & Bura, 2022) is the intersection of all subspaces $\mathrm{span}(\mathbf{B}_ {\mathrm{var}}) \subset \mathbb{R}^p$ satisfying:

$$
\mathrm{Var}(Y \mid \mathbf{X}) = \mathrm{Var}\left( Y \mid \mathbf{B}_ {\mathrm{var}}^\top \mathbf{X} \right)
$$

In general, $\mathcal{S}_ {\mathrm{Var}(Y \mid \mathbf{X})} \not\subseteq \mathcal{S}_ {E(Y \mid \mathbf{X})}$. Consequently, a deep neural network that perfectly recovers the CMS may completely discard the CVS, creating miscalibrated prediction intervals.

### Definition 3.2 (Variance Sufficiency Audit)
A representation $\mathbf{Z}$ satisfies **second-order variance sufficiency** if the squared prediction residuals $e^2 = (Y - \hat{E}(Y \mid \mathbf{Z}))^2$ are conditionally independent of $\mathbf{X}$ given $\mathbf{Z}$:

$$
e^2 \perp\!\!\perp \mathbf{X} \mid \mathbf{Z}
$$

---

## 4. Kernel Operators in RKHS

### Definition 4.1 (Cross-Covariance Operator)
Let $\mathcal{H}_{\mathcal{X}}$ and $\mathcal{H}_{\mathcal{Y}}$ be Reproducing Kernel Hilbert Spaces with feature maps $\phi(\mathbf{x})$ and $\psi(y)$. The cross-covariance operator $\Sigma_{YX}: \mathcal{H}_{\mathcal{X}} \to \mathcal{H}_{\mathcal{Y}}$ satisfies:

$$
\langle g, \Sigma_{YX} f \rangle_{\mathcal{H}_{\mathcal{Y}}} = \mathrm{Cov}(f(\mathbf{X}), g(Y)) \quad \forall f \in \mathcal{H}_{\mathcal{X}}, g \in \mathcal{H}_{\mathcal{Y}}
$$

### Definition 4.2 (Normalized Conditional Cross-Covariance Operator)
Conditioning on $\mathbf{Z}$ via ridge regularization parameter $\epsilon > 0$:

$$
\Sigma_{YX \mid Z} = \Sigma_{YX} - \Sigma_{YZ} (\Sigma_{ZZ} + \epsilon I)^{-1} \Sigma_{ZX}
$$

The Hilbert-Schmidt norm $\|\Sigma_{YX \mid Z}\|_{\text{HS}}^2 = 0$ if and only if $Y \perp\!\!\perp \mathbf{X} \mid \mathbf{Z}$ under universal kernels.
