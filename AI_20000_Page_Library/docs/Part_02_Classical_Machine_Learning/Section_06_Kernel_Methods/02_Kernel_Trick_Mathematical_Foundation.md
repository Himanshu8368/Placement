# Kernel Trick: Mathematical Foundation

## Table of Contents
1. [From Dot Products to Kernels](#from-dot-products-to-kernels)
2. [Feature Maps and Hilbert Spaces](#feature-maps-and-hilbert-spaces)
3. [The Kernel Trick: Formal Derivation](#the-kernel-trick-formal-derivation)
4. [Mercer's Theorem](#mercers-theorem)
5. [Positive Semi-Definite Matrices](#positive-semi-definite-matrices)
6. [Reproducing Kernel Hilbert Spaces (RKHS)](#reproducing-kernel-hilbert-spaces)
7. [Kernel Construction Rules](#kernel-construction-rules)
8. [From Scratch: Verifying Valid Kernels](#from-scratch-verifying-valid-kernels)
9. [Summary and Next Steps](#summary-and-next-steps)

---

## From Dot Products to Kernels

### Why Dot Products Matter in ML

Every linear algorithm ultimately relies on **dot products** between data points. This is the key that unlocks the kernel trick.

**Linear regression** uses:
```
ŷ = wᵀx = Σᵢ wᵢxᵢ
```

**Ridge regression** solution uses:
```
w = (XᵀX + λI)⁻¹ Xᵀy
```
Both XᵀX and Xᵀy are built entirely from dot products xᵢᵀxⱼ.

**Logistic regression** decision boundary uses:
```
f(x) = wᵀx + b  →  only dot products
```

**SVM** margin and decision function use:
```
f(x) = Σᵢ αᵢ yᵢ (xᵢᵀx) + b  →  only dot products with training points
```

**The key observation:**

> If we replace every dot product xᵢᵀxⱼ with a kernel K(xᵢ, xⱼ),
> the algorithm now implicitly operates in the feature space defined by K —
> without ever computing the transformation explicitly.

### The Substitution Principle

```
Original algorithm:    uses  xᵢᵀxⱼ
Kernelized algorithm:  uses  K(xᵢ, xⱼ)

That's it. That's the trick.
```

```python
import numpy as np

# ============================================================
# SHOW THE SUBSTITUTION: Ridge Regression → Kernel Ridge
# ============================================================

# --- STANDARD RIDGE REGRESSION ---
def ridge_regression(X, y, lam):
    """
    Standard Ridge: w = (XᵀX + λI)⁻¹ Xᵀy
    Prediction:     ŷ = Xw = X(XᵀX + λI)⁻¹Xᵀy
    """
    n, d = X.shape
    I = np.eye(d)
    w = np.linalg.solve(X.T @ X + lam * I, X.T @ y)
    return w

def ridge_predict(X_train, w):
    return X_train @ w


# --- KERNEL RIDGE REGRESSION ---
def rbf_kernel_matrix(X1, X2, gamma=1.0):
    """Compute K[i,j] = exp(-gamma * ||X1[i] - X2[j]||²)"""
    # Efficient vectorized computation
    # ||x - x'||² = ||x||² - 2xᵀx' + ||x'||²
    sq1 = np.sum(X1**2, axis=1, keepdims=True)   # (n1, 1)
    sq2 = np.sum(X2**2, axis=1, keepdims=True).T  # (1, n2)
    cross = X1 @ X2.T                              # (n1, n2)
    dists_sq = sq1 - 2 * cross + sq2
    return np.exp(-gamma * dists_sq)

def kernel_ridge_regression(X_train, y, lam, gamma=1.0):
    """
    Kernel Ridge: α = (K + λI)⁻¹ y
    where K[i,j] = K(xᵢ, xⱼ)
    """
    K = rbf_kernel_matrix(X_train, X_train, gamma)
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    return alpha

def kernel_ridge_predict(X_train, X_test, alpha, gamma=1.0):
    """
    Prediction: ŷ = K_test · α
    K_test[i,j] = K(x_test_i, x_train_j)
    """
    K_test = rbf_kernel_matrix(X_test, X_train, gamma)
    return K_test @ alpha


# ============================================================
# DEMO: Compare on a non-linear regression task
# ============================================================

np.random.seed(42)
n = 60
X_train = np.sort(np.random.uniform(0, 6, n)).reshape(-1, 1)
y_train  = np.sin(X_train.ravel()) + np.random.randn(n) * 0.15

X_test = np.linspace(0, 6, 200).reshape(-1, 1)
y_true = np.sin(X_test.ravel())

# Standard Ridge
w = ridge_regression(X_train, y_train, lam=0.1)
y_ridge = ridge_predict(X_test, w)

# Kernel Ridge (RBF)
alpha = kernel_ridge_regression(X_train, y_train, lam=0.1, gamma=1.0)
y_kridge = kernel_ridge_predict(X_train, X_test, alpha, gamma=1.0)

# Compare
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, c='gray', s=20, alpha=0.7, label='Data')
plt.plot(X_test, y_true,   'g-', linewidth=2, label='True sin(x)')
plt.plot(X_test, y_ridge,  'r--', linewidth=2, label='Ridge (linear)')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Standard Ridge\n(cannot fit sin curve)')
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, c='gray', s=20, alpha=0.7, label='Data')
plt.plot(X_test, y_true,   'g-', linewidth=2, label='True sin(x)')
plt.plot(X_test, y_kridge, 'b-', linewidth=2, label='Kernel Ridge (RBF)')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Kernel Ridge\n(fits sin curve perfectly!)')
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ridge_vs_kernel_ridge.png', dpi=150, bbox_inches='tight')
print("Saved: ridge_vs_kernel_ridge.png")

ridge_mse  = np.mean((y_ridge  - y_true)**2)
kridge_mse = np.mean((y_kridge - y_true)**2)
print(f"\nStandard Ridge MSE:  {ridge_mse:.4f}")
print(f"Kernel Ridge MSE:    {kridge_mse:.4f}")
print(f"Improvement:         {ridge_mse/kridge_mse:.1f}x better")
```

---

## Feature Maps and Hilbert Spaces

### What Is a Feature Map?

A **feature map** φ is a function that transforms input data into a (possibly much higher-dimensional) feature space.

```
φ: X → H

Input space X:     Raw data  (e.g., ℝ²)
Feature space H:   Transformed data  (e.g., ℝ⁵ or even ℝ^∞)
```

### Concrete Example: Degree-2 Polynomial Feature Map

For input x = (x₁, x₂) ∈ ℝ²:

```
φ(x) = (x₁², √2 x₁x₂, x₂²)  ∈ ℝ³

Why √2?  To make the math work out exactly.
We'll verify this below.
```

```python
import numpy as np

# ============================================================
# FEATURE MAP: Explicit vs Implicit
# ============================================================

def phi_degree2(x):
    """
    Explicit degree-2 feature map for x ∈ ℝ²
    φ(x) = (x₁², √2·x₁x₂, x₂²)
    """
    x1, x2 = x[0], x[1]
    return np.array([x1**2, np.sqrt(2)*x1*x2, x2**2])

def poly_kernel(x, xp, degree=2, c=0):
    """
    Polynomial kernel K(x,x') = (xᵀx' + c)^degree
    """
    return (np.dot(x, xp) + c) ** degree


# ============================================================
# VERIFY: K(x, x') = ⟨φ(x), φ(x')⟩
# ============================================================

print("="*60)
print("VERIFICATION: K(x, x') = ⟨φ(x), φ(x')⟩")
print("="*60)

test_pairs = [
    (np.array([1.0, 2.0]), np.array([3.0, 4.0])),
    (np.array([0.5, 1.5]), np.array([2.0, 0.0])),
    (np.array([-1.0, 3.0]), np.array([2.0, -1.0])),
]

for x, xp in test_pairs:
    phi_x  = phi_degree2(x)
    phi_xp = phi_degree2(xp)

    explicit = np.dot(phi_x, phi_xp)
    kernel   = poly_kernel(x, xp, degree=2, c=0)

    match = "✓" if np.isclose(explicit, kernel) else "✗"
    print(f"\n  x={x}, x'={xp}")
    print(f"  φ(x)  = {phi_x}")
    print(f"  φ(x') = {phi_xp}")
    print(f"  ⟨φ(x),φ(x')⟩  = {explicit:.4f}")
    print(f"  K(x,x')        = {kernel:.4f}    {match} MATCH")


# ============================================================
# COMPUTATIONAL SAVINGS: count operations
# ============================================================

print("\n" + "="*60)
print("COMPUTATIONAL SAVINGS")
print("="*60)

import time

def time_explicit(X, n_trials=1000):
    """Time the explicit feature map + inner product approach"""
    start = time.perf_counter()
    for _ in range(n_trials):
        for i in range(len(X)):
            for j in range(len(X)):
                phi_i = phi_degree2(X[i])
                phi_j = phi_degree2(X[j])
                _ = np.dot(phi_i, phi_j)
    return (time.perf_counter() - start) / n_trials

def time_kernel(X, n_trials=1000):
    """Time the kernel function approach"""
    start = time.perf_counter()
    for _ in range(n_trials):
        for i in range(len(X)):
            for j in range(len(X)):
                _ = poly_kernel(X[i], X[j], degree=2)
    return (time.perf_counter() - start) / n_trials

X_sample = np.random.randn(20, 2)

t_explicit = time_explicit(X_sample, n_trials=100)
t_kernel   = time_kernel(X_sample,   n_trials=100)

print(f"Explicit (φ then inner product): {t_explicit*1000:.2f} ms per trial")
print(f"Kernel (direct computation):     {t_kernel*1000:.2f} ms per trial")
print(f"Speedup: {t_explicit/t_kernel:.1f}x")
print(f"\nThis gap EXPLODES with higher degrees and dimensions!")
```

### What Is a Hilbert Space?

A **Hilbert space** H is a complete inner product space. For our purposes, it has three key properties:

```
1. INNER PRODUCT ⟨·,·⟩:
   Maps two elements to a real number.
   For vectors: ⟨u, v⟩ = uᵀv

2. NORM:
   ||f||_H = √⟨f, f⟩_H

3. COMPLETENESS:
   Every Cauchy sequence converges within H.
   (Ensures infinite-dimensional spaces are well-defined)
```

**Why Hilbert spaces?** They generalize ℝⁿ to infinite dimensions while keeping the geometry we need — especially the inner product — intact.

### The Representer Theorem (Preview)

A critical result that makes kernels practical:

> **Representer Theorem**: For any kernel method minimizing a regularized empirical risk, the optimal solution lies in the span of the training kernel evaluations.

```
Optimal solution:  f*(x) = Σᵢ αᵢ K(xᵢ, x)

Even though H may be infinite-dimensional,
the solution is determined by n coefficients α₁, ..., αn.
This is what makes kernel methods computationally tractable!
```

```python
import numpy as np

# ============================================================
# REPRESENTER THEOREM IN ACTION
# ============================================================

print("="*60)
print("REPRESENTER THEOREM DEMONSTRATION")
print("="*60)

np.random.seed(42)
n = 10
X_train = np.sort(np.random.uniform(0, 5, n)).reshape(-1, 1)
y_train  = np.sin(X_train.ravel())

# The solution α to kernel ridge
lam   = 0.01
gamma = 1.0
K     = rbf_kernel_matrix(X_train, X_train, gamma)
alpha = np.linalg.solve(K + lam * np.eye(n), y_train)

print(f"\nTraining set size:  n = {n}")
print(f"Feature space dim:  ∞  (RBF kernel)")
print(f"Solution dimension: {len(alpha)}  (one α per training point)")
print(f"\nα values: {np.round(alpha, 4)}")

print("""
Despite working in infinite-dimensional space,
the solution is described by just n=10 numbers!

Prediction at any new point x:
  f(x) = Σᵢ αᵢ · K(xᵢ, x)
       = weighted sum of kernel evaluations
""")

# Predict
X_test_pt = np.array([[2.5]])
K_test    = rbf_kernel_matrix(X_test_pt, X_train, gamma)
pred      = K_test @ alpha
print(f"Prediction at x=2.5:   {pred[0]:.4f}")
print(f"True value sin(2.5):   {np.sin(2.5):.4f}")
```

---

## The Kernel Trick: Formal Derivation

### Step-by-Step Derivation for SVM

The SVM optimization problem in **primal form** is:

```
Minimize:    ½||w||²
Subject to:  yᵢ(wᵀxᵢ + b) ≥ 1  ∀i
```

Using Lagrange multipliers, we get the **dual form**:

```
Maximize:    Σᵢ αᵢ  -  ½ Σᵢ Σⱼ αᵢαⱼ yᵢyⱼ (xᵢᵀxⱼ)
Subject to:  Σᵢ αᵢyᵢ = 0,   αᵢ ≥ 0  ∀i
```

Notice: The data only appears as dot products **xᵢᵀxⱼ**!

**Applying the kernel trick** — replace xᵢᵀxⱼ with K(xᵢ, xⱼ):

```
Maximize:    Σᵢ αᵢ  -  ½ Σᵢ Σⱼ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ)
Subject to:  Σᵢ αᵢyᵢ = 0,   αᵢ ≥ 0  ∀i
```

**Decision function** after kernelization:

```
f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b
```

That's the complete kernel trick applied to SVM!

```python
import numpy as np

# ============================================================
# KERNEL SVM DUAL: FROM SCRATCH
# ============================================================

class KernelSVMScratch:
    """
    Binary SVM using the dual formulation with kernel.
    Solved via gradient ascent on the dual objective.
    (In practice, use quadratic programming; this is for understanding.)
    """

    def __init__(self, kernel_fn, C=1.0, lr=0.01, n_iter=500):
        self.kernel_fn = kernel_fn
        self.C         = C
        self.lr        = lr
        self.n_iter    = n_iter
        self.alpha     = None
        self.b         = 0.0
        self.X_train   = None
        self.y_train   = None

    def _kernel_matrix(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel_fn(X[i], X[j])
        return K

    def fit(self, X, y):
        """Maximize dual objective via projected gradient ascent"""
        n = len(y)
        self.X_train = X.copy()
        self.y_train = y.copy()

        K = self._kernel_matrix(X)
        self.alpha = np.zeros(n)

        for iteration in range(self.n_iter):
            # Gradient of dual objective w.r.t. alpha
            # d/dα [ Σαᵢ - ½ΣΣ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ) ]
            # = 1 - (y ⊙ α) ⊙ (K · (y ⊙ α)) ... simplified:
            yKya = (y * self.alpha) @ K * y
            grad = np.ones(n) - yKya

            # Gradient ascent step
            self.alpha += self.lr * grad

            # Project onto constraints: 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0
            self.alpha = np.clip(self.alpha, 0, self.C)

            # Enforce Σαᵢyᵢ = 0 (approximate)
            pos = self.alpha[y == 1].sum()
            neg = self.alpha[y == -1].sum()
            if pos > 0 and neg > 0:
                scale = min(pos, neg) / max(pos, neg)
                if pos > neg:
                    self.alpha[y == 1] *= scale
                else:
                    self.alpha[y == -1] *= scale

        # Compute bias from support vectors
        sv_mask = self.alpha > 1e-4
        if sv_mask.sum() > 0:
            decisions = np.array([
                sum(self.alpha[i] * y[i] * self.kernel_fn(X[i], x)
                    for i in range(n))
                for x in X[sv_mask]
            ])
            self.b = np.mean(y[sv_mask] - decisions)

        print(f"Support vectors found: {sv_mask.sum()}")
        return self

    def decision_function(self, X_test):
        n = len(self.X_train)
        scores = np.array([
            sum(self.alpha[i] * self.y_train[i] *
                self.kernel_fn(self.X_train[i], x)
                for i in range(n)) + self.b
            for x in X_test
        ])
        return scores

    def predict(self, X_test):
        return np.sign(self.decision_function(X_test)).astype(int)


# ============================================================
# DEMO
# ============================================================

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X, y = make_moons(n_samples=50, noise=0.15, random_state=42)
X = StandardScaler().fit_transform(X)
y_svm = np.where(y == 0, -1, 1)   # SVM needs {-1, +1} labels

# Define RBF kernel
def rbf(x1, x2, gamma=0.5):
    return np.exp(-gamma * np.sum((x1 - x2)**2))

# Train
svm_scratch = KernelSVMScratch(kernel_fn=rbf, C=1.0, lr=0.005, n_iter=300)
svm_scratch.fit(X, y_svm)

# Evaluate
y_pred = svm_scratch.predict(X)
acc = np.mean(y_pred == y_svm)
print(f"Scratch Kernel SVM accuracy: {acc*100:.1f}%")

# Compare with sklearn
from sklearn.svm import SVC
sklearn_svm = SVC(kernel='rbf', gamma=0.5, C=1.0)
sklearn_svm.fit(X, y_svm)
print(f"Sklearn Kernel SVM accuracy: {sklearn_svm.score(X, y_svm)*100:.1f}%")
```

### The Dual Representation: Why It Works

```python
import numpy as np

# ============================================================
# VISUALIZE: What the dual coefficients represent
# ============================================================

print("="*60)
print("DUAL COEFFICIENTS: What do they mean?")
print("="*60)

from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
X    = StandardScaler().fit_transform(X)
y_sv = np.where(y == 0, -1, 1)

svm = SVC(kernel='rbf', C=1.0, gamma=1.0)
svm.fit(X, y_sv)

alpha_times_y = svm.dual_coef_.ravel()  # αᵢ * yᵢ for support vectors
sv_indices    = svm.support_            # indices of support vectors

print(f"\nTotal training points: {len(X)}")
print(f"Support vectors:       {len(sv_indices)}")
print(f"  ({len(sv_indices)/len(X)*100:.0f}% of training data)")
print(f"\n|αᵢ| range: [{np.abs(alpha_times_y).min():.4f}, "
      f"{np.abs(alpha_times_y).max():.4f}]")

print("""
Key insight:
  - Most training points have αᵢ = 0
    → They don't influence the decision boundary AT ALL
  - Only support vectors (near the boundary) have αᵢ > 0
    → The model is SPARSE: defined by a small subset of points
  - This sparsity is what makes SVMs memory-efficient
    and resistant to overfitting
""")

# Visualize support vectors
import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.linspace(-3, 3, 200),
                     np.linspace(-3, 3, 200))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlBu')
plt.contour(xx, yy, Z, colors='k', linewidths=1)

# All training points
plt.scatter(X[y_sv==-1, 0], X[y_sv==-1, 1], c='royalblue',
            s=25, edgecolors='k', linewidths=0.4, alpha=0.6, label='Class -1')
plt.scatter(X[y_sv== 1, 0], X[y_sv== 1, 1], c='tomato',
            s=25, marker='s', edgecolors='k', linewidths=0.4, alpha=0.6, label='Class +1')

# Highlight support vectors
plt.scatter(X[sv_indices, 0], X[sv_indices, 1],
            s=150, facecolors='none', edgecolors='gold',
            linewidths=2.5, label='Support Vectors')

plt.title(f'SVM Decision Boundary\n'
          f'{len(sv_indices)}/{len(X)} support vectors (circled)')
plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
plt.legend(fontsize=9); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('support_vectors_visualization.png', dpi=150, bbox_inches='tight')
print("Saved: support_vectors_visualization.png")
```

---

## Mercer's Theorem

### The Central Question

Given a function K(x, x'), **when is it a valid kernel?**

Meaning: when does there exist a feature map φ such that K(x, x') = ⟨φ(x), φ(x')⟩?

### Mercer's Theorem (Statement)

> **Mercer's Theorem**: A continuous, symmetric function K: X × X → ℝ is a valid kernel if and only if for any finite set of points {x₁, ..., xₙ}, the **kernel matrix** (Gram matrix) K where K[i,j] = K(xᵢ, xⱼ) is **positive semi-definite**.

```
K is a valid kernel
⟺
∀ finite {x₁,...,xₙ} ⊂ X:
  K (the n×n matrix) is positive semi-definite
⟺
∀ vector c ∈ ℝⁿ:
  Σᵢ Σⱼ cᵢcⱼ K(xᵢ,xⱼ) ≥ 0
⟺
All eigenvalues of K are ≥ 0
```

### Why Positive Semi-Definite?

Inner products in Hilbert spaces satisfy:

```
⟨f, f⟩ ≥ 0  for all f ∈ H    (non-negative)
⟨f, f⟩ = 0  iff f = 0         (definite)

This means:
  Σᵢ Σⱼ cᵢcⱼ ⟨φ(xᵢ), φ(xⱼ)⟩
= ⟨Σᵢ cᵢφ(xᵢ), Σⱼ cⱼφ(xⱼ)⟩
= ||Σᵢ cᵢφ(xᵢ)||²
≥ 0

Since K(xᵢ,xⱼ) = ⟨φ(xᵢ),φ(xⱼ)⟩, the kernel matrix must be PSD.
```

```python
import numpy as np

# ============================================================
# MERCER'S THEOREM: CHECKING VALIDITY
# ============================================================

def is_valid_kernel(kernel_fn, X, tol=1e-8):
    """
    Check if a kernel function is valid (PSD) on dataset X.
    Uses Mercer's theorem: check if kernel matrix is PSD.
    """
    n = len(X)

    # Build kernel matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_fn(X[i], X[j])

    # Symmetry check
    is_symmetric = np.allclose(K, K.T)

    # Eigenvalue check (PSD = all eigenvalues ≥ 0)
    eigenvalues = np.linalg.eigvalsh(K)
    min_eigenvalue = eigenvalues.min()
    is_psd = min_eigenvalue >= -tol

    print(f"  Symmetric:           {is_symmetric}")
    print(f"  Min eigenvalue:      {min_eigenvalue:.6f}")
    print(f"  All eigenvalues ≥ 0: {is_psd}")
    print(f"  → Valid kernel:      {is_symmetric and is_psd}")

    return is_symmetric and is_psd


# Test data
np.random.seed(42)
X_check = np.random.randn(20, 2)

print("="*60)
print("MERCER'S THEOREM: KERNEL VALIDITY CHECKS")
print("="*60)

# Valid kernels
valid_kernels = {
    'Linear K(x,x\') = xᵀx\'':
        lambda x, xp: np.dot(x, xp),

    'Polynomial K(x,x\') = (xᵀx\' + 1)²':
        lambda x, xp: (np.dot(x, xp) + 1)**2,

    'RBF K(x,x\') = exp(-||x-x\'||²)':
        lambda x, xp: np.exp(-np.sum((x - xp)**2)),
}

# Invalid kernels
invalid_kernels = {
    'Signed K(x,x\') = xᵀx\' - 10  (shifted, may go negative)':
        lambda x, xp: np.dot(x, xp) - 10,

    'Asymmetric K(x,x\') = xᵀx\' + ||x||  (not symmetric)':
        lambda x, xp: np.dot(x, xp) + np.linalg.norm(x),
}

print("\n--- VALID KERNELS ---")
for name, kfn in valid_kernels.items():
    print(f"\n{name}:")
    is_valid_kernel(kfn, X_check)

print("\n--- POTENTIALLY INVALID KERNELS ---")
for name, kfn in invalid_kernels.items():
    print(f"\n{name}:")
    is_valid_kernel(kfn, X_check)
```

### Mercer's Theorem: Integral Form

For continuous kernels on compact domains, Mercer's theorem gives an eigenfunction expansion:

```
K(x, x') = Σₖ λₖ φₖ(x) φₖ(x')

Where:
  λₖ ≥ 0     are eigenvalues (non-negative by PSD)
  φₖ          are orthonormal eigenfunctions

This is the explicit connection between the kernel
and its feature map:
  φ(x) = (√λ₁ φ₁(x),  √λ₂ φ₂(x),  √λ₃ φ₃(x), ...)
```

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# MERCER EXPANSION: APPROXIMATE WITH FINITE TERMS
# ============================================================

def mercer_approximation(kernel_fn, X, n_components=5):
    """
    Approximate a kernel using its Mercer eigendecomposition.
    K(x,x') ≈ Σₖ₌₁ᴷ λₖ φₖ(x) φₖ(x')
    """
    n = len(X)

    # Build kernel matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_fn(X[i], X[j])

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep only positive eigenvalues
    pos = eigenvalues > 1e-10
    eigenvalues  = eigenvalues[pos]
    eigenvectors = eigenvectors[:, pos]

    # Approximate kernel matrix using top-K components
    K_approx = np.zeros((n, n))
    for k in range(min(n_components, len(eigenvalues))):
        phi_k = np.sqrt(eigenvalues[k]) * eigenvectors[:, k]
        K_approx += np.outer(phi_k, phi_k) / eigenvalues[k] * eigenvalues[k]

    return K, K_approx[:n_components], eigenvalues


# Test
np.random.seed(42)
X_1d = np.sort(np.random.uniform(0, 5, 30)).reshape(-1, 1)

rbf_1d = lambda x, xp: np.exp(-0.5 * np.sum((x - xp)**2))

K_exact, _, eigenvalues = mercer_approximation(rbf_1d, X_1d, n_components=10)

# Plot eigenvalue decay
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(eigenvalues[:15], 'o-', linewidth=2, markersize=8)
plt.xlabel('Component index k')
plt.ylabel('Eigenvalue λₖ')
plt.title('Mercer Eigenvalue Decay\n(RBF Kernel)')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
# Show kernel matrix
import matplotlib
im = plt.imshow(K_exact, cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title('RBF Kernel Matrix K[i,j]')
plt.xlabel('Point index j')
plt.ylabel('Point index i')

plt.tight_layout()
plt.savefig('mercer_decomposition.png', dpi=150, bbox_inches='tight')
print("Saved: mercer_decomposition.png")

print(f"\nTop 5 eigenvalues: {eigenvalues[:5].round(3)}")
print(f"Eigenvalue sum (total variance): {eigenvalues.sum():.3f}")
print(f"Top-3 explain: {eigenvalues[:3].sum()/eigenvalues.sum()*100:.1f}% of variance")
```

---

## Positive Semi-Definite Matrices

### Definition

A symmetric matrix M is **positive semi-definite (PSD)** if:

```
∀ v ∈ ℝⁿ:   vᵀMv ≥ 0

Equivalently:
  - All eigenvalues λᵢ ≥ 0
  - All principal minors have determinant ≥ 0
  - M = AᵀA for some matrix A
```

### From Scratch: PSD Check and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PSD MATRICES: PROPERTIES AND CHECKS
# ============================================================

def check_psd(M, name="Matrix"):
    """Check if matrix is PSD and report properties"""
    print(f"\n--- {name} ---")
    print(f"Matrix:\n{np.round(M, 3)}")

    # Symmetry
    is_sym = np.allclose(M, M.T)
    print(f"Symmetric: {is_sym}")

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(M)
    print(f"Eigenvalues: {np.round(eigenvalues, 4)}")
    is_psd = all(ev >= -1e-10 for ev in eigenvalues)
    print(f"PSD (all eigenvalues ≥ 0): {is_psd}")

    # Quadratic form check
    n = M.shape[0]
    qf_values = []
    for _ in range(100):
        v = np.random.randn(n)
        qf_values.append(v @ M @ v)
    print(f"Quadratic form vᵀMv — min: {min(qf_values):.4f}, "
          f"max: {max(qf_values):.4f}")
    print(f"All vᵀMv ≥ 0: {min(qf_values) >= -1e-8}")

    return is_psd


# Example 1: Valid kernel matrix (PSD)
np.random.seed(42)
A = np.random.randn(4, 4)
M_psd = A.T @ A   # Always PSD: M = AᵀA
check_psd(M_psd, "PSD Matrix (M = AᵀA)")

# Example 2: RBF kernel matrix (should be PSD)
X_small = np.array([[0.0], [1.0], [2.0], [3.0]])
K_rbf_small = np.array([
    [np.exp(-0.5*(x-xp)**2) for xp in X_small.ravel()]
    for x in X_small.ravel()
])
check_psd(K_rbf_small, "RBF Kernel Matrix")

# Example 3: NOT PSD
M_not_psd = np.array([
    [ 1.0, -3.0],
    [-3.0,  1.0]
])
check_psd(M_not_psd, "NOT PSD Matrix")

# ============================================================
# VISUALIZE: PSD as a bowl shape in quadratic form
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PSD matrix → quadratic form is a bowl (≥ 0 everywhere)
M1 = np.array([[3.0, 1.0],
               [1.0, 2.0]])  # PSD

# Indefinite matrix → quadratic form has negative region
M2 = np.array([[2.0, 3.0],
               [3.0, 1.0]])  # Not PSD

v1_range = np.linspace(-2, 2, 100)
v2_range = np.linspace(-2, 2, 100)
V1, V2 = np.meshgrid(v1_range, v2_range)

for ax, M, title in zip(axes, [M1, M2],
                          ['PSD Matrix\nvᵀMv ≥ 0 everywhere (bowl)',
                           'Not PSD Matrix\nvᵀMv < 0 in some regions']):
    QF = np.zeros_like(V1)
    for i in range(len(v1_range)):
        for j in range(len(v2_range)):
            v = np.array([V1[j, i], V2[j, i]])
            QF[j, i] = v @ M @ v

    cmap = 'RdYlGn'
    levels = np.linspace(QF.min(), QF.max(), 30)
    cf = ax.contourf(V1, V2, QF, levels=levels, cmap=cmap)
    ax.contour(V1, V2, QF, levels=[0], colors='black', linewidths=2)
    plt.colorbar(cf, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('v₁'); ax.set_ylabel('v₂')

    eigs = np.linalg.eigvalsh(M)
    ax.text(0.05, 0.95, f'Eigenvalues: {eigs.round(2)}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(facecolor='white', alpha=0.7))

plt.suptitle('Quadratic Form vᵀMv\n(Black contour = zero level)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('psd_visualization.png', dpi=150, bbox_inches='tight')
print("\nSaved: psd_visualization.png")
```

### Connection: PSD ↔ Valid Kernel ↔ Inner Product

```
PSD Matrix M
    ↕
∃ factorization M = AᵀA
    ↕
K[i,j] = ⟨A[:,i], A[:,j]⟩   (inner product of columns)
    ↕
K(xᵢ,xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩   (kernel as inner product)
    ↕
Valid kernel function K
```

---

## Reproducing Kernel Hilbert Spaces (RKHS)

### What Is an RKHS?

A **Reproducing Kernel Hilbert Space (RKHS)** is a Hilbert space H of functions where evaluation is a bounded linear functional.

In plain terms:

> For every point x, the function K(·, x) = K_x lives in H, and it "reproduces" the value of any function f ∈ H at x via the inner product.

**The Reproducing Property:**

```
⟨f, K(·, x)⟩_H = f(x)     ∀f ∈ H, ∀x ∈ X

Especially:
⟨K(·, x), K(·, x')⟩_H = K(x, x')
```

This is why it is called "reproducing" — the kernel reproduces function evaluations!

### Building Intuition

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# RKHS INTUITION: Kernel as a "template"
# ============================================================

print("="*60)
print("RKHS INTUITION: K(·, x₀) as a function in H")
print("="*60)

# Fix a "center" point x₀
x0_values = [0.5, 2.0, 3.5, 5.0]
x_range   = np.linspace(-1, 6, 300)
gamma     = 1.0

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for x0 in x0_values:
    # K(·, x₀) is a Gaussian centered at x₀
    k_x0 = np.exp(-gamma * (x_range - x0)**2)
    plt.plot(x_range, k_x0, linewidth=2, label=f'K(·, {x0})')

plt.title('K(·, x₀): RBF kernel as a function of one argument\n'
          'Each K(·, x₀) is an element of the RKHS H')
plt.xlabel('x'); plt.ylabel('K(x, x₀)')
plt.legend(fontsize=9); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Any function in the RKHS is a linear combination of kernels
alpha = np.array([1.5, -1.0, 2.0, -0.5])
f_rkhs = np.zeros_like(x_range)
for a, x0 in zip(alpha, x0_values):
    k_x0    = np.exp(-gamma * (x_range - x0)**2)
    f_rkhs += a * k_x0

plt.plot(x_range, f_rkhs, 'k-', linewidth=2.5, label='f(x) ∈ H')
for a, x0 in zip(alpha, x0_values):
    k_x0 = np.exp(-gamma * (x_range - x0)**2)
    plt.plot(x_range, a * k_x0, '--', alpha=0.5, label=f'{a:.1f}·K(·,{x0})')

plt.title('f(x) = Σᵢ αᵢ K(·, xᵢ)\nFunctions in RKHS = weighted kernel sums')
plt.xlabel('x'); plt.legend(fontsize=8); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rkhs_functions.png', dpi=150, bbox_inches='tight')
print("Saved: rkhs_functions.png")
```

### The RKHS Norm: Why Regularization Works

The RKHS norm **measures function complexity**:

```
||f||²_H = Σᵢ Σⱼ αᵢαⱼ K(xᵢ, xⱼ)    (for f = Σᵢ αᵢ K(·, xᵢ))

Small ||f||_H  → smooth, simple function
Large ||f||_H  → complex, wiggly function
```

This is exactly what regularization penalizes!

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# RKHS NORM: Smooth vs Complex Functions
# ============================================================

np.random.seed(42)
n = 15
X_pts = np.sort(np.random.uniform(0, 6, n)).reshape(-1, 1)
y_pts  = np.sin(X_pts.ravel()) + np.random.randn(n) * 0.2

gamma = 1.0
K = rbf_kernel_matrix(X_pts, X_pts, gamma)

x_range = np.linspace(-0.5, 6.5, 300).reshape(-1, 1)
K_pred  = rbf_kernel_matrix(x_range, X_pts, gamma)

regularizations = [1e-6, 0.01, 0.1, 10.0]
fig, axes = plt.subplots(1, len(regularizations),
                         figsize=(16, 4), sharey=True)

for ax, lam in zip(axes, regularizations):
    alpha  = np.linalg.solve(K + lam * np.eye(n), y_pts)
    y_pred = K_pred @ alpha

    # RKHS norm: ||f||²_H = αᵀKα
    rkhs_norm = np.sqrt(alpha @ K @ alpha)

    ax.scatter(X_pts, y_pts, c='k', s=40, zorder=5, label='Data')
    ax.plot(x_range, y_pred, 'b-', linewidth=2, label='Prediction')
    ax.plot(x_range, np.sin(x_range), 'g--',
            linewidth=1.5, alpha=0.5, label='True sin')

    ax.set_title(f'λ = {lam}\n||f||_H = {rkhs_norm:.2f}')
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2.5, 2.5)

axes[0].set_ylabel('y')
axes[0].legend(fontsize=8)

plt.suptitle('RKHS Norm Controls Smoothness\n'
             'Large λ → Small norm → Smoother function',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('rkhs_norm_regularization.png', dpi=150, bbox_inches='tight')
print("Saved: rkhs_norm_regularization.png")

print("\nEffect of regularization on RKHS norm:")
print(f"{'λ':<10} {'RKHS norm ||f||_H':<20}")
print("-"*30)
for lam in regularizations:
    alpha = np.linalg.solve(K + lam * np.eye(n), y_pts)
    norm  = np.sqrt(alpha @ K @ alpha)
    print(f"{lam:<10} {norm:<20.4f}")
```

---

## Kernel Construction Rules

Mercer's theorem tells us WHEN a kernel is valid. But how do we **build new valid kernels**?

### The Closure Rules

If K₁ and K₂ are valid kernels, then so are:

```
1. SCALING:          c · K₁(x,x')                    where c > 0
2. SUM:              K₁(x,x') + K₂(x,x')
3. PRODUCT:          K₁(x,x') · K₂(x,x')
4. POLYNOMIAL:       (K₁(x,x') + c)ᵈ                 where c ≥ 0, d ∈ ℤ⁺
5. EXPONENTIATION:   exp(K₁(x,x'))
6. NORMALIZATION:    K₁(x,x') / √(K₁(x,x)·K₁(x',x'))
7. COMPOSITION:      K₁(φ(x), φ(x'))                 for any φ: X → X
```

```python
import numpy as np

# ============================================================
# KERNEL CONSTRUCTION: Build new kernels from old ones
# ============================================================

print("="*60)
print("KERNEL CONSTRUCTION RULES")
print("="*60)

np.random.seed(42)
X_test = np.random.randn(15, 2)

# Base kernels
def K1(x, xp):
    """Linear kernel"""
    return np.dot(x, xp)

def K2(x, xp, gamma=0.5):
    """RBF kernel"""
    return np.exp(-gamma * np.sum((x - xp)**2))


# Build kernel matrices
def gram(kernel_fn, X):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_fn(X[i], X[j])
    return K


# Rule 1: Scaling
def K_scaled(x, xp, c=3.0):
    return c * K1(x, xp)

# Rule 2: Sum
def K_sum(x, xp):
    return K1(x, xp) + K2(x, xp)

# Rule 3: Product
def K_product(x, xp):
    return K1(x, xp) * K2(x, xp)

# Rule 4: Polynomial
def K_poly(x, xp, d=3, c=1):
    return (K1(x, xp) + c) ** d

# Rule 5: Exponentiation
def K_exp(x, xp):
    return np.exp(K1(x, xp))

# Rule 6: Normalization
def K_normalized(x, xp):
    num   = K2(x, xp)
    denom = np.sqrt(K2(x, x) * K2(xp, xp))
    return num / (denom + 1e-12)

constructed_kernels = {
    'K₁ (linear, base)':   K1,
    'K₂ (RBF, base)':      K2,
    '3·K₁ (scaled)':       K_scaled,
    'K₁ + K₂ (sum)':       K_sum,
    'K₁ · K₂ (product)':   K_product,
    '(K₁+1)³ (poly)':      K_poly,
    'exp(K₁)':              K_exp,
    'K₂/||K₂|| (normed)':  K_normalized,
}

print(f"\n{'Kernel':<26} {'Min eigenvalue':<18} {'Valid?':<10}")
print("-"*55)

for name, kfn in constructed_kernels.items():
    G = gram(kfn, X_test)
    eigvals = np.linalg.eigvalsh(G)
    min_eig = eigvals.min()
    valid   = min_eig >= -1e-8
    status  = "✓ YES" if valid else "✗ NO"
    print(f"{name:<26} {min_eig:<18.6f} {status}")
```

### Designing Domain-Specific Kernels

```python
import numpy as np

# ============================================================
# CUSTOM KERNEL DESIGN: String (Spectrum) Kernel
# ============================================================

def spectrum_kernel(s1, s2, k=3):
    """
    String spectrum kernel.
    K(s1, s2) = number of shared k-mers (substrings of length k).
    
    Used in: bioinformatics, text classification, NLP.
    """
    def get_kmers(s, k):
        return [s[i:i+k] for i in range(len(s) - k + 1)]

    from collections import Counter
    c1 = Counter(get_kmers(s1, k))
    c2 = Counter(get_kmers(s2, k))

    # Dot product of k-mer count vectors
    return sum(c1[km] * c2[km] for km in c1 if km in c2)

# Verify it is PSD
sequences = ["ATCGATCG", "ATCGATAA", "GGGGCAAA",
             "TACGATCG", "ATCGGCTA", "CCCCATCG"]

n = len(sequences)
K_str = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        K_str[i, j] = spectrum_kernel(sequences[i], sequences[j], k=3)

eigvals = np.linalg.eigvalsh(K_str)

print("\n" + "="*60)
print("STRING SPECTRUM KERNEL (k=3)")
print("="*60)
print(f"\nKernel matrix:\n{K_str.astype(int)}")
print(f"\nEigenvalues: {np.round(eigvals, 4)}")
print(f"Valid kernel (PSD): {eigvals.min() >= -1e-8}")

print("\nSimilarity matrix (normalized):")
D = np.diag(1.0 / np.sqrt(np.diag(K_str)))
K_norm = D @ K_str @ D
print(np.round(K_norm, 2))
print("\n(1.0 = identical, 0.0 = no shared 3-mers)")
```

---

## From Scratch: Verifying Valid Kernels

### Complete Verification Pipeline

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# COMPLETE KERNEL VALIDATION SUITE
# ============================================================

class KernelValidator:
    """
    Comprehensive validator for kernel functions.
    Checks all conditions from Mercer's theorem.
    """

    def __init__(self, kernel_fn, name="Kernel"):
        self.kernel_fn = kernel_fn
        self.name      = name

    def check_symmetry(self, X, n_trials=100):
        """K(x,x') must equal K(x',x)"""
        n = len(X)
        errors = []
        for _ in range(n_trials):
            i, j = np.random.randint(0, n, 2)
            diff = abs(self.kernel_fn(X[i], X[j]) -
                       self.kernel_fn(X[j], X[i]))
            errors.append(diff)
        max_err = max(errors)
        passed  = max_err < 1e-10
        print(f"  Symmetry K(x,x')=K(x',x):  "
              f"{'✓ PASS' if passed else '✗ FAIL'}  "
              f"(max |K(x,x')-K(x',x)| = {max_err:.2e})")
        return passed

    def check_psd(self, X):
        """Kernel matrix must be PSD"""
        n = len(X)
        K = np.array([[self.kernel_fn(X[i], X[j])
                       for j in range(n)]
                      for i in range(n)])
        eigvals = np.linalg.eigvalsh(K)
        min_eig = eigvals.min()
        passed  = min_eig >= -1e-8
        print(f"  PSD (min eigenvalue ≥ 0):   "
              f"{'✓ PASS' if passed else '✗ FAIL'}  "
              f"(min eigenvalue = {min_eig:.6f})")
        return passed, K

    def check_diagonal_positive(self, X, n_trials=50):
        """K(x,x) ≥ 0  (self-similarity must be non-negative)"""
        n = len(X)
        diag_vals = [self.kernel_fn(X[i], X[i]) for i in range(n)]
        passed = all(v >= -1e-10 for v in diag_vals)
        print(f"  K(x,x) ≥ 0:                 "
              f"{'✓ PASS' if passed else '✗ FAIL'}  "
              f"(min K(x,x) = {min(diag_vals):.6f})")
        return passed

    def check_cauchy_schwarz(self, X, n_trials=100):
        """K(x,x')² ≤ K(x,x) · K(x',x')  (Cauchy-Schwarz)"""
        n = len(X)
        violations = 0
        for _ in range(n_trials):
            i, j = np.random.randint(0, n, 2)
            lhs = self.kernel_fn(X[i], X[j])**2
            rhs = self.kernel_fn(X[i], X[i]) * self.kernel_fn(X[j], X[j])
            if lhs > rhs + 1e-8:
                violations += 1
        passed = violations == 0
        print(f"  Cauchy-Schwarz K(x,x')²≤K(x,x)K(x',x'):  "
              f"{'✓ PASS' if passed else f'✗ FAIL ({violations}/{n_trials} violations)'}")
        return passed

    def validate(self, X):
        """Run all checks"""
        print(f"\n{'='*60}")
        print(f"KERNEL VALIDATION: {self.name}")
        print(f"{'='*60}")
        print(f"  Test set: {len(X)} points in ℝ^{X.shape[1]}\n")

        s1 = self.check_symmetry(X)
        s2, K = self.check_psd(X)
        s3 = self.check_diagonal_positive(X)
        s4 = self.check_cauchy_schwarz(X)

        overall = s1 and s2 and s3 and s4
        print(f"\n  OVERALL: {'✓ VALID KERNEL' if overall else '✗ INVALID KERNEL'}")
        return overall, K


# ============================================================
# TEST SEVERAL KERNELS
# ============================================================

np.random.seed(42)
X_val = np.random.randn(25, 3)

kernels_to_test = [
    ("Linear",          lambda x, xp: np.dot(x, xp)),
    ("Polynomial d=3",  lambda x, xp: (np.dot(x, xp) + 1)**3),
    ("RBF γ=0.5",       lambda x, xp: np.exp(-0.5 * np.sum((x-xp)**2))),
    ("Laplacian",       lambda x, xp: np.exp(-np.sum(np.abs(x-xp)))),
    ("INVALID: x[0]*xp[1]",
                        lambda x, xp: x[0]*xp[1]),     # Not symmetric
]

results = {}
for name, kfn in kernels_to_test:
    v = KernelValidator(kfn, name)
    is_valid, _ = v.validate(X_val)
    results[name] = is_valid

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, valid in results.items():
    icon = "✓" if valid else "✗"
    print(f"  {icon}  {name}")
```

---

## Summary and Next Steps

### The Mathematical Chain

```
MERCER'S THEOREM
        ↓
K(x,x') is PSD
        ↓
∃ feature map φ: K(x,x') = ⟨φ(x), φ(x')⟩_H
        ↓
Replace xᵀx' → K(x,x') in any linear algorithm
        ↓
Algorithm now operates in H implicitly
        ↓
Non-linear patterns in X = linear patterns in H
```

### Key Equations to Memorize

```
1. Kernel as inner product:
   K(x, x') = ⟨φ(x), φ(x')⟩_H

2. Mercer's condition:
   K valid ⟺ kernel matrix is PSD
   ⟺ ∀c: Σᵢ Σⱼ cᵢcⱼ K(xᵢ,xⱼ) ≥ 0

3. Reproducing property:
   ⟨f, K(·,x)⟩_H = f(x)

4. Representer theorem:
   f*(x) = Σᵢ αᵢ K(xᵢ, x)

5. RKHS norm (complexity measure):
   ||f||²_H = Σᵢ Σⱼ αᵢ αⱼ K(xᵢ,xⱼ) = αᵀKα

6. Kernel ridge solution:
   α = (K + λI)⁻¹y
```

### Prerequisites Checklist

Before moving to Chapter 03, confirm you can:
- [ ] Explain why dot products are central to ML algorithms
- [ ] Verify that K(x,x') = ⟨φ(x),φ(x')⟩ for the degree-2 poly kernel
- [ ] State Mercer's theorem in your own words
- [ ] Check whether a given kernel matrix is PSD
- [ ] Apply at least 3 kernel construction rules
- [ ] Explain what the RKHS norm measures
- [ ] Implement kernel ridge regression from scratch

### What's in the Next Chapters

**03_Common_Kernel_Functions.md**
→ Detailed study of every kernel: linear, polynomial, RBF, sigmoid, Laplacian, custom

**04_Dual_Formulations.md** ← already exists
→ How the primal SVM becomes the dual problem; KKT conditions

**07_SVM_Hard_Margin.md**
→ The complete SVM optimization derivation using the math from this chapter

---

## Quick Reference

```
╔══════════════════════════════════════════════════════════════╗
║       KERNEL TRICK: MATHEMATICAL FOUNDATION                  ║
║                   Quick Reference                            ║
╠══════════════════════════════════════════════════════════════╣
║  THE TRICK IN ONE LINE:                                      ║
║    Replace xᵢᵀxⱼ  →  K(xᵢ, xⱼ)  in any algorithm           ║
╠══════════════════════════════════════════════════════════════╣
║  MERCER'S THEOREM:                                           ║
║    K valid  ⟺  kernel matrix K is PSD                        ║
║    PSD      ⟺  all eigenvalues ≥ 0                           ║
║    PSD      ⟺  vᵀKv ≥ 0  for all v                          ║
╠══════════════════════════════════════════════════════════════╣
║  RKHS:                                                       ║
║    H = function space defined by K                           ║
║    Reproducing property: ⟨f, K(·,x)⟩_H = f(x)               ║
║    RKHS norm = function complexity                           ║
╠══════════════════════════════════════════════════════════════╣
║  REPRESENTER THEOREM:                                        ║
║    Optimal f*(x) = Σᵢ αᵢ K(xᵢ, x)                           ║
║    → n parameters, not infinite!                             ║
╠══════════════════════════════════════════════════════════════╣
║  CONSTRUCTION RULES (K₁,K₂ valid → these also valid):       ║
║    c·K₁          (c > 0)                                     ║
║    K₁ + K₂                                                   ║
║    K₁ · K₂                                                   ║
║    (K₁ + c)ᵈ     (c ≥ 0, d ∈ ℤ⁺)                           ║
║    exp(K₁)                                                   ║
╠══════════════════════════════════════════════════════════════╣
║  KERNEL RIDGE REGRESSION (from scratch):                     ║
║    K[i,j] = K(xᵢ, xⱼ)                                       ║
║    α = (K + λI)⁻¹ y                                          ║
║    ŷ = K_test · α                                            ║
╚══════════════════════════════════════════════════════════════╝
```

---

*End of 02_Kernel_Trick_Mathematical_Foundation.md*