# Common Kernel Functions

## Table of Contents
1. [Overview of Kernel Functions](#overview-of-kernel-functions)
2. [Linear Kernel](#linear-kernel)
3. [Polynomial Kernel](#polynomial-kernel)
4. [Radial Basis Function (RBF) Kernel](#radial-basis-function-rbf-kernel)
5. [Sigmoid Kernel](#sigmoid-kernel)
6. [Laplacian Kernel](#laplacian-kernel)
7. [Matérn Kernel](#matérn-kernel)
8. [Kernel Comparison on Real Datasets](#kernel-comparison-on-real-datasets)
9. [How to Choose the Right Kernel](#how-to-choose-the-right-kernel)
10. [Summary and Quick Reference](#summary-and-quick-reference)

---

## Overview of Kernel Functions

### What We Cover in This Chapter

Chapter 02 showed **why** kernels work mathematically. This chapter covers the **what** — the specific kernel functions you will actually use in practice, with:

- Complete mathematical definition
- Feature space it corresponds to
- Hyperparameter effects
- From-scratch implementation
- Visualizations
- When to use it

### The Kernel Zoo at a Glance

```
KERNEL FAMILY TREE
│
├── Stationary (depend only on ||x - x'||)
│   ├── RBF / Gaussian      K = exp(-γ||x-x'||²)
│   ├── Laplacian           K = exp(-γ||x-x'||)
│   └── Matérn              K = f(√(2ν)||x-x'||/l) · exp(...)
│
├── Non-stationary (depend on x and x' separately)
│   ├── Linear              K = xᵀx'
│   ├── Polynomial          K = (xᵀx' + c)ᵈ
│   └── Sigmoid             K = tanh(γxᵀx' + c)
│
└── Specialized (for structured data)
    ├── String kernels       (text, DNA)
    ├── Graph kernels        (molecules, networks)
    └── Histogram kernels    (image features)
```

### Setup: Utilities Used Throughout This Chapter

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# SHARED UTILITIES
# ============================================================

def plot_decision_boundary(ax, clf, X, y, title="", resolution=200):
    """Plot classifier decision boundary on given axes."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='k', linewidths=0.8, alpha=0.4)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='royalblue', s=22,
               edgecolors='k', linewidths=0.3, alpha=0.9)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='tomato', s=22,
               marker='s', edgecolors='k', linewidths=0.3, alpha=0.9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xticks([]); ax.set_yticks([])


def kernel_matrix(X1, X2, kernel_fn, **kwargs):
    """Compute the kernel matrix K[i,j] = K(X1[i], X2[j])."""
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_fn(X1[i], X2[j], **kwargs)
    return K


def check_psd(K_mat, name=""):
    """Check positive semi-definiteness."""
    eigvals = np.linalg.eigvalsh(K_mat)
    is_psd  = eigvals.min() >= -1e-8
    print(f"  {name}: min eigenvalue = {eigvals.min():.6f}  "
          f"→  {'✓ PSD (valid)' if is_psd else '✗ NOT PSD'}")
    return is_psd
```

---

## Linear Kernel

### Mathematical Definition

```
K(x, x') = xᵀx' = Σᵢ xᵢ xᵢ'

Feature map:  φ(x) = x   (identity — no transformation!)
Feature space: original input space ℝᵈ
```

### Properties

| Property | Value |
|----------|-------|
| Stationary? | No |
| Feature space dim | d (same as input) |
| Hyperparameters | None |
| Complexity | O(d) per evaluation |
| Decision boundary | Linear hyperplane |

### When the Linear Kernel Shines

Despite its simplicity, the linear kernel is often the **best choice** for:

- **High-dimensional data** where d >> n (text, genomics)
- **Data already linearly separable** in the input space
- **Large datasets** where RBF is too slow
- **Interpretability** is important (weights w = Σᵢ αᵢ yᵢ xᵢ are interpretable)

### Complete Implementation

```python
# ============================================================
# LINEAR KERNEL: COMPLETE IMPLEMENTATION
# ============================================================

def linear_kernel(x, xp):
    """
    K(x, x') = xᵀx'
    The simplest valid kernel.
    """
    return np.dot(x, xp)


class LinearKernelSVM:
    """
    SVM with linear kernel — from scratch.
    Uses the primal form: w = Σᵢ αᵢ yᵢ xᵢ
    """

    def __init__(self, C=1.0, lr=0.001, n_iter=1000):
        self.C      = C
        self.lr     = lr
        self.n_iter = n_iter
        self.w      = None
        self.b      = 0.0
        self.losses = []

    def hinge_loss(self, X, y):
        """Hinge loss: max(0, 1 - yᵢ(wᵀxᵢ + b))"""
        margins = y * (X @ self.w + self.b)
        loss    = 0.5 * np.dot(self.w, self.w) + \
                  self.C * np.mean(np.maximum(0, 1 - margins))
        return loss

    def fit(self, X, y):
        n, d    = X.shape
        self.w  = np.zeros(d)
        self.b  = 0.0

        for epoch in range(self.n_iter):
            for i in range(n):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                if margin < 1:
                    # Misclassified or within margin
                    self.w = (1 - self.lr) * self.w + \
                             self.lr * self.C * y[i] * X[i]
                    self.b += self.lr * self.C * y[i]
                else:
                    # Correctly classified, just regularize
                    self.w = (1 - self.lr) * self.w

            if epoch % 100 == 0:
                loss = self.hinge_loss(X, y)
                self.losses.append(loss)

        return self

    def predict(self, X):
        return np.sign(X @ self.w + self.b).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ============================================================
# DEMO: Linear kernel SVM vs sklearn
# ============================================================

print("=" * 60)
print("LINEAR KERNEL SVM")
print("=" * 60)

# Generate linearly separable data
X_lin, y_lin = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, class_sep=2.5, random_state=42
)
y_svm = np.where(y_lin == 0, -1, 1)  # {0,1} → {-1,+1}

sc = StandardScaler()
X_s = sc.fit_transform(X_lin)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_s, y_svm, test_size=0.25, random_state=42
)

# From scratch
lin_svm = LinearKernelSVM(C=1.0, lr=0.001, n_iter=500)
lin_svm.fit(X_tr, y_tr)
scratch_acc = lin_svm.score(X_te, y_te)

# Sklearn
sk_lin = SVC(kernel='linear', C=1.0)
sk_lin.fit(X_tr, y_tr)
sklearn_acc = sk_lin.score(X_te, y_te)

print(f"\nFrom-scratch accuracy:  {scratch_acc*100:.1f}%")
print(f"Sklearn accuracy:       {sklearn_acc*100:.1f}%")
print(f"Weight vector w:        {lin_svm.w.round(4)}")
print(f"Bias b:                 {lin_svm.b:.4f}")

# ============================================================
# LINEAR KERNEL ON TEXT DATA (HIGH-DIM)
# ============================================================

print("\n--- Linear Kernel on High-Dimensional Text Data ---")

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

categories = ['sci.tech', 'rec.sport.hockey',
              'talk.politics.guns', 'comp.graphics']

try:
    news_train = fetch_20newsgroups(
        subset='train', categories=categories, random_state=42
    )
    news_test  = fetch_20newsgroups(
        subset='test',  categories=categories, random_state=42
    )

    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, sublinear_tf=True)),
        ('svm',   SVC(kernel='linear', C=0.1))
    ])

    text_pipeline.fit(news_train.data, news_train.target)
    text_acc = text_pipeline.score(news_test.data, news_test.target)

    d_text = text_pipeline.named_steps['tfidf']\
                           .transform(news_train.data).shape[1]
    print(f"  Text features (d):   {d_text}")
    print(f"  Training samples:    {len(news_train.data)}")
    print(f"  Test accuracy:       {text_acc*100:.1f}%")
    print(f"  → Linear kernel excels in high-dim text!")

except Exception:
    print("  (Dataset not cached; showing synthetic high-dim example)")
    # Synthetic high-dim
    X_hd, y_hd = make_classification(
        n_samples=300, n_features=1000, n_informative=50,
        n_redundant=50, random_state=42
    )
    X_hd_tr, X_hd_te, y_hd_tr, y_hd_te = train_test_split(
        X_hd, y_hd, test_size=0.25, random_state=42
    )
    sk_lin_hd = SVC(kernel='linear', C=1.0)
    sk_lin_hd.fit(X_hd_tr, y_hd_tr)
    print(f"  High-dim (d=1000) accuracy:  "
          f"{sk_lin_hd.score(X_hd_te, y_hd_te)*100:.1f}%")

# ============================================================
# KERNEL MATRIX VISUALIZATION
# ============================================================

X_small = np.array([[1,0],[0,1],[1,1],[2,0],[0,2]], dtype=float)
K_lin   = kernel_matrix(X_small, X_small, linear_kernel)

print("\nLinear kernel matrix (5 points):")
print(np.round(K_lin, 2))
check_psd(K_lin, "Linear kernel")
```

### Effect of the Linear Kernel: Decision Boundary

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

datasets = [
    (make_classification(n_samples=200, n_features=2, n_redundant=0,
                         class_sep=2.0, random_state=42),
     "Linearly Separable"),
    (make_moons(n_samples=200, noise=0.2, random_state=42),
     "Moons (Non-linear)"),
    (make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42),
     "Circles (Radial)"),
]

for ax, ((X, y), name) in zip(axes, datasets):
    Xs = StandardScaler().fit_transform(X)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(Xs, y)
    acc = clf.score(Xs, y) * 100
    plot_decision_boundary(ax, clf, Xs, y,
                           f"Linear Kernel\n{name}\nAcc={acc:.0f}%")

plt.suptitle("Linear Kernel Decision Boundaries",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('linear_kernel_boundaries.png', dpi=150, bbox_inches='tight')
print("Saved: linear_kernel_boundaries.png")
```

---

## Polynomial Kernel

### Mathematical Definition

```
K(x, x') = (γ xᵀx' + c)ᵈ

Parameters:
  d  = degree     (integer ≥ 1, controls complexity)
  c  = coef0      (≥ 0, trades off lower vs higher degree terms)
  γ  = gamma      (scale of inner product)

Feature map for degree d:
  φ(x) contains all monomials of degree ≤ d
  Dimension: C(n+d, d)  grows combinatorially with d and n
```

### The Feature Space (Degree 2 Example)

For x = (x₁, x₂) and K(x,x') = (xᵀx' + 1)²:

```
φ(x) = (1,  √2 x₁,  √2 x₂,  x₁²,  √2 x₁x₂,  x₂²)

This captures:
  - Bias term         (constant 1)
  - Linear terms      (x₁, x₂)
  - Quadratic terms   (x₁², x₁x₂, x₂²)

Decision boundary becomes:
  a₀ + a₁x₁ + a₂x₂ + a₃x₁² + a₄x₁x₂ + a₅x₂² = 0
  → conic section (ellipse, parabola, or hyperbola)
```

### Complete Implementation

```python
# ============================================================
# POLYNOMIAL KERNEL: COMPLETE IMPLEMENTATION
# ============================================================

def polynomial_kernel(x, xp, degree=3, gamma=1.0, coef0=1.0):
    """
    K(x, x') = (γ xᵀx' + c)ᵈ

    Args:
        degree:  polynomial degree d
        gamma:   scale factor γ
        coef0:   constant offset c (≥ 0 for valid kernel)
    """
    return (gamma * np.dot(x, xp) + coef0) ** degree


def poly_feature_map_degree2(x, gamma=1.0, coef0=1.0):
    """
    Explicit degree-2 feature map (for 2D input).
    φ(x) such that K(x,x') = ⟨φ(x), φ(x')⟩
    """
    x1, x2 = x[0], x[1]
    g       = gamma
    c       = coef0

    # Expanding (γ(x₁x₁' + x₂x₂') + c)² gives these terms:
    return np.array([
        c**2,                          # constant
        np.sqrt(2*c*g) * x1,          # x₁ term
        np.sqrt(2*c*g) * x2,          # x₂ term
        g * x1**2,                    # x₁² term
        np.sqrt(2) * g * x1 * x2,    # x₁x₂ term
        g * x2**2                     # x₂² term
    ])


print("=" * 60)
print("POLYNOMIAL KERNEL")
print("=" * 60)

# Verify: K(x,x') = ⟨φ(x), φ(x')⟩
x  = np.array([1.0, 2.0])
xp = np.array([3.0, -1.0])

phi_x  = poly_feature_map_degree2(x)
phi_xp = poly_feature_map_degree2(xp)

K_explicit = np.dot(phi_x, phi_xp)
K_formula  = polynomial_kernel(x, xp, degree=2)

print(f"\nVerification (degree=2):")
print(f"  x  = {x},  x' = {xp}")
print(f"  ⟨φ(x), φ(x')⟩ = {K_explicit:.4f}")
print(f"  K(x, x')       = {K_formula:.4f}")
print(f"  Match: {np.isclose(K_explicit, K_formula)}")

# Feature space size
for d in [1, 2, 3, 4, 5]:
    n_features = 10  # input dimension
    from math import comb
    n_poly_features = comb(n_features + d, d)
    print(f"  d={d}, input_dim=10 → poly features: {n_poly_features:,}")


# ============================================================
# HYPERPARAMETER SENSITIVITY: DEGREE d
# ============================================================

np.random.seed(42)
X_moon, y_moon = make_moons(n_samples=200, noise=0.2, random_state=42)
X_moon_s = StandardScaler().fit_transform(X_moon)

degrees = [1, 2, 3, 4, 5, 6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.ravel()

train_accs = []
for ax, d in zip(axes_flat, degrees):
    clf = SVC(kernel='poly', degree=d, coef0=1, gamma='scale', C=1.0)
    clf.fit(X_moon_s, y_moon)
    acc = clf.score(X_moon_s, y_moon)
    train_accs.append(acc)
    plot_decision_boundary(ax, clf, X_moon_s, y_moon,
                           f"Poly d={d}\nAcc={acc*100:.0f}%  "
                           f"SV={clf.n_support_.sum()}")

plt.suptitle("Polynomial Kernel: Effect of Degree d\n"
             "(Higher d = more complex boundary, risk of overfit)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('poly_kernel_degree.png', dpi=150, bbox_inches='tight')
print("\nSaved: poly_kernel_degree.png")


# ============================================================
# HYPERPARAMETER SENSITIVITY: coef0 (c)
# ============================================================

print("\n--- Effect of coef0 (c) ---")

coef0_values = [0, 0.5, 1, 2, 5, 10]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat  = axes.ravel()

for ax, c in zip(axes_flat, coef0_values):
    clf = SVC(kernel='poly', degree=3, coef0=c, gamma='scale', C=1.0)
    clf.fit(X_moon_s, y_moon)
    acc = clf.score(X_moon_s, y_moon)
    plot_decision_boundary(ax, clf, X_moon_s, y_moon,
                           f"Poly d=3, c={c}\nAcc={acc*100:.0f}%")

plt.suptitle("Polynomial Kernel: Effect of coef0 (c)\n"
             "c=0: only degree-d terms  |  Large c: lower-degree terms dominate",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('poly_kernel_coef0.png', dpi=150, bbox_inches='tight')
print("Saved: poly_kernel_coef0.png")


# ============================================================
# GRID SEARCH: Best polynomial hyperparameters
# ============================================================

print("\n--- Grid Search: Polynomial Kernel ---")

from sklearn.pipeline import Pipeline

poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='poly'))
])

param_grid_poly = {
    'svm__degree': [2, 3, 4, 5],
    'svm__coef0':  [0, 1, 2, 5],
    'svm__C':      [0.1, 1, 10],
    'svm__gamma':  ['scale', 'auto'],
}

gs_poly = GridSearchCV(poly_pipeline, param_grid_poly,
                       cv=5, scoring='accuracy', n_jobs=-1)
gs_poly.fit(X_moon_s, y_moon)

print(f"Best params: {gs_poly.best_params_}")
print(f"Best CV acc: {gs_poly.best_score_*100:.1f}%")
```

---

## Radial Basis Function (RBF) Kernel

### Mathematical Definition

```
K(x, x') = exp(-γ ||x - x'||²)

Parameters:
  γ (gamma) > 0   controls width of the Gaussian bell

Equivalent forms:
  K(x, x') = exp(-||x - x'||² / (2σ²))   where γ = 1/(2σ²)

Feature space:  INFINITE dimensional (Taylor expansion of exp)
```

### Why RBF Is the Default Choice

The RBF kernel has several remarkable properties:

```
1. Universal approximator:
   Can approximate any continuous function
   given enough data and proper γ

2. Always valid (PSD):
   K(x,x) = exp(0) = 1  (normalized)
   K(x,x') = K(x',x)    (symmetric)
   Kernel matrix always PSD

3. Localness:
   K(x,x') → 0 as ||x-x'|| → ∞
   Each training point influences only nearby predictions

4. Bounded:
   0 < K(x,x') ≤ 1  always

5. Interpretable:
   K(x,x') = "Gaussian similarity" between x and x'
```

### Complete Implementation

```python
# ============================================================
# RBF KERNEL: COMPLETE IMPLEMENTATION
# ============================================================

def rbf_kernel(x, xp, gamma=1.0):
    """
    K(x, x') = exp(-γ ||x - x'||²)

    Equivalent to a Gaussian similarity function.
    γ large → narrow bell → local influence
    γ small → wide bell  → global influence
    """
    diff = x - xp
    return np.exp(-gamma * np.dot(diff, diff))


def rbf_kernel_matrix_fast(X1, X2, gamma=1.0):
    """
    Vectorized RBF kernel matrix computation.
    Uses: ||x - x'||² = ||x||² - 2xᵀx' + ||x'||²
    """
    sq1   = np.sum(X1**2, axis=1, keepdims=True)    # (n1, 1)
    sq2   = np.sum(X2**2, axis=1, keepdims=True).T  # (1, n2)
    cross = X1 @ X2.T                                # (n1, n2)
    dist2 = sq1 - 2*cross + sq2                      # (n1, n2)
    dist2 = np.maximum(dist2, 0)                     # numerical safety
    return np.exp(-gamma * dist2)


print("=" * 60)
print("RBF KERNEL")
print("=" * 60)

# Basic properties
x  = np.array([1.0, 0.0])
xp = np.array([1.0, 0.0])   # identical point

print(f"\nK(x, x) (self-similarity): {rbf_kernel(x, x):.6f}  (always = 1)")
print(f"K(x, x') at distance 1:     {rbf_kernel(x, np.array([2.0,0.0])):.6f}")
print(f"K(x, x') at distance 2:     {rbf_kernel(x, np.array([3.0,0.0])):.6f}")
print(f"K(x, x') at distance 5:     {rbf_kernel(x, np.array([6.0,0.0])):.6f}")
print("\nObservation: Similarity decays to 0 with distance.")


# ============================================================
# GAMMA SENSITIVITY ANALYSIS
# ============================================================

print("\n--- Gamma Sensitivity ---")

np.random.seed(42)
X_circ, y_circ = make_circles(
    n_samples=200, noise=0.1, factor=0.4, random_state=42
)
X_circ_s = StandardScaler().fit_transform(X_circ)

gammas    = [0.01, 0.1, 0.5, 1.0, 5.0, 20.0]
n_sv_list = []
acc_list  = []

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat  = axes.ravel()

for ax, gamma in zip(axes_flat, gammas):
    clf = SVC(kernel='rbf', gamma=gamma, C=1.0)
    clf.fit(X_circ_s, y_circ)
    acc  = clf.score(X_circ_s, y_circ) * 100
    n_sv = clf.n_support_.sum()
    n_sv_list.append(n_sv)
    acc_list.append(acc)
    plot_decision_boundary(
        ax, clf, X_circ_s, y_circ,
        f"RBF γ={gamma}\nAcc={acc:.0f}%  SV={n_sv}"
    )

plt.suptitle(
    "RBF Kernel: Effect of γ (gamma)\n"
    "Small γ → smooth/global  |  Large γ → spiky/local → overfit",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('rbf_gamma_effect.png', dpi=150, bbox_inches='tight')
print("Saved: rbf_gamma_effect.png")

# Summary table
print(f"\n{'γ':<8} {'Accuracy':<12} {'# Support Vectors'}")
print("-"*35)
for gamma, acc, n_sv in zip(gammas, acc_list, n_sv_list):
    print(f"{gamma:<8} {acc:<12.1f} {n_sv}")


# ============================================================
# C AND GAMMA JOINT SENSITIVITY
# ============================================================

print("\n--- Joint C and γ Grid Search ---")

rbf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf'))
])

param_grid_rbf = {
    'svm__C':     [0.1, 1, 10, 100],
    'svm__gamma': [0.001, 0.01, 0.1, 1, 10],
}

np.random.seed(42)
X_data, y_data = make_moons(n_samples=300, noise=0.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_data, y_data, test_size=0.25, random_state=42
)

gs_rbf = GridSearchCV(rbf_pipeline, param_grid_rbf,
                      cv=5, scoring='accuracy', n_jobs=-1)
gs_rbf.fit(X_tr, y_tr)

print(f"Best params:   {gs_rbf.best_params_}")
print(f"Best CV acc:   {gs_rbf.best_score_*100:.2f}%")
print(f"Test acc:      {gs_rbf.score(X_te, y_te)*100:.2f}%")

# Heatmap of CV scores
results_df = np.array(gs_rbf.cv_results_['mean_test_score'])
scores_mat = results_df.reshape(
    len(param_grid_rbf['svm__C']),
    len(param_grid_rbf['svm__gamma'])
)

plt.figure(figsize=(8, 5))
import seaborn as sns
sns.heatmap(
    scores_mat * 100,
    annot=True, fmt='.1f', cmap='YlOrRd',
    xticklabels=param_grid_rbf['svm__gamma'],
    yticklabels=param_grid_rbf['svm__C']
)
plt.xlabel('γ (gamma)', fontsize=12)
plt.ylabel('C', fontsize=12)
plt.title('RBF SVM Cross-Validation Accuracy (%)\n'
          '(Higher = better)', fontsize=12)
plt.tight_layout()
plt.savefig('rbf_cv_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: rbf_cv_heatmap.png")


# ============================================================
# RBF: GAMMA SELECTION RULES OF THUMB
# ============================================================

print("\n--- Gamma Selection Rules of Thumb ---")

def gamma_scale(X):
    """sklearn 'scale': 1 / (n_features * X.var())"""
    return 1.0 / (X.shape[1] * X.var())

def gamma_auto(X):
    """sklearn 'auto': 1 / n_features"""
    return 1.0 / X.shape[1]

def gamma_median_heuristic(X):
    """Median heuristic: γ = 1 / (2 * median²(pairwise distances))"""
    from sklearn.metrics import pairwise_distances
    D  = pairwise_distances(X)
    med = np.median(D[D > 0])
    return 1.0 / (2 * med**2)

X_ex = StandardScaler().fit_transform(X_data)

print(f"  gamma='scale' (1/(d·var)):    {gamma_scale(X_ex):.4f}")
print(f"  gamma='auto'  (1/d):          {gamma_auto(X_ex):.4f}")
print(f"  Median heuristic:             {gamma_median_heuristic(X_ex):.4f}")
print(f"\n  → Start with gamma='scale' and tune from there")
```

---

## Sigmoid Kernel

### Mathematical Definition

```
K(x, x') = tanh(γ xᵀx' + c)

Parameters:
  γ (gamma) > 0
  c (coef0): offset

Caution: The sigmoid kernel is NOT always PSD!
         It is only valid (PSD) for certain γ and c values.
         Use with care.
```

### Connection to Neural Networks

```
A single hidden layer neural network computes:
  f(x) = Σᵢ wᵢ tanh(vᵢᵀx + b)

The sigmoid kernel SVM approximates this:
  f(x) = Σᵢ αᵢ yᵢ tanh(γ xᵢᵀx + c)

→ The sigmoid kernel SVM behaves like a 1-hidden-layer neural network
  with tanh activations, but is trained via convex optimization!
```

### Complete Implementation

```python
# ============================================================
# SIGMOID KERNEL: COMPLETE IMPLEMENTATION
# ============================================================

def sigmoid_kernel(x, xp, gamma=0.1, coef0=0.0):
    """
    K(x, x') = tanh(γ xᵀx' + c)

    WARNING: Not always PSD. Valid only for certain parameter ranges.
    Check PSD before use in practice.
    """
    return np.tanh(gamma * np.dot(x, xp) + coef0)


print("=" * 60)
print("SIGMOID KERNEL")
print("=" * 60)

# Check validity for different parameter values
print("\n--- Checking PSD Validity ---")

np.random.seed(42)
X_chk = np.random.randn(20, 2)

gamma_coef_pairs = [
    (0.01, 0.0), (0.1, 0.0), (0.5, 0.0), (1.0, 0.0),
    (0.1, 1.0),  (0.1, -1.0),(0.5, 1.0), (0.01, -0.5),
]

print(f"{'γ':<8} {'c':<8} {'Min eigenvalue':<20} {'Valid PSD?'}")
print("-"*50)

valid_params = []
for gamma, coef0 in gamma_coef_pairs:
    K_sig = kernel_matrix(X_chk, X_chk, sigmoid_kernel,
                          gamma=gamma, coef0=coef0)
    eigvals = np.linalg.eigvalsh(K_sig)
    is_psd  = eigvals.min() >= -1e-8
    status  = "✓ YES" if is_psd else "✗ NO"
    print(f"{gamma:<8} {coef0:<8} {eigvals.min():<20.6f} {status}")
    if is_psd:
        valid_params.append((gamma, coef0))

print(f"\nValid parameter combinations found: {len(valid_params)}")
print("→ Always verify PSD before using sigmoid kernel!")


# ============================================================
# DECISION BOUNDARIES WITH SIGMOID KERNEL
# ============================================================

X_sin, y_sin = make_moons(n_samples=200, noise=0.2, random_state=42)
X_sin_s = StandardScaler().fit_transform(X_sin)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

sigmoid_params = [
    {'gamma': 0.01, 'coef0': 0.0},
    {'gamma': 0.1,  'coef0': 0.0},
    {'gamma': 0.5,  'coef0': 0.0},
    {'gamma': 0.1,  'coef0': 1.0},
]

for ax, params in zip(axes, sigmoid_params):
    clf = SVC(kernel='sigmoid', C=1.0, **params)
    clf.fit(X_sin_s, y_sin)
    acc = clf.score(X_sin_s, y_sin) * 100
    plot_decision_boundary(
        ax, clf, X_sin_s, y_sin,
        f"Sigmoid\nγ={params['gamma']}, c={params['coef0']}\nAcc={acc:.0f}%"
    )

plt.suptitle("Sigmoid Kernel Decision Boundaries\n"
             "(Results vary widely — verify PSD for each param set)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('sigmoid_kernel_boundaries.png', dpi=150, bbox_inches='tight')
print("\nSaved: sigmoid_kernel_boundaries.png")


# ============================================================
# SIGMOID vs RBF: HEAD-TO-HEAD
# ============================================================

print("\n--- Sigmoid vs RBF Comparison ---")

for name, dataset in [
    ("Moons",   make_moons(n_samples=200, noise=0.2, random_state=42)),
    ("Circles", make_circles(n_samples=200, noise=0.1,
                             factor=0.4, random_state=42)),
]:
    X_cmp, y_cmp = dataset
    Xs = StandardScaler().fit_transform(X_cmp)
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y_cmp, test_size=0.25, random_state=42
    )

    rbf_clf = SVC(kernel='rbf', gamma='scale', C=1.0).fit(Xtr, ytr)
    sig_clf = SVC(kernel='sigmoid', gamma=0.1, coef0=0, C=1.0).fit(Xtr, ytr)

    print(f"  {name}:")
    print(f"    RBF test acc:     {rbf_clf.score(Xte, yte)*100:.1f}%")
    print(f"    Sigmoid test acc: {sig_clf.score(Xte, yte)*100:.1f}%")
```

---

## Laplacian Kernel

### Mathematical Definition

```
K(x, x') = exp(-γ ||x - x'||₁)

Where ||·||₁ is the L1 (Manhattan) distance:
  ||x - x'||₁ = Σᵢ |xᵢ - xᵢ'|

Compare with RBF which uses L2 (Euclidean) distance:
  K_RBF(x, x') = exp(-γ ||x - x'||₂²)
```

### Laplacian vs RBF: Key Differences

```
RBF (Gaussian):
  - Uses squared L2 distance → penalizes large differences heavily
  - Smooth, infinitely differentiable feature map
  - Boundary: smooth curves

Laplacian:
  - Uses L1 distance → more robust to outliers
  - Less smooth feature map (only once differentiable)
  - Better for sparse or discontinuous data
  - Boundary: can have sharp corners
```

### Complete Implementation

```python
# ============================================================
# LAPLACIAN KERNEL: COMPLETE IMPLEMENTATION
# ============================================================

def laplacian_kernel(x, xp, gamma=1.0):
    """
    K(x, x') = exp(-γ ||x - x'||₁)

    Uses L1 (Manhattan) distance.
    More robust to outliers than RBF.
    """
    l1_dist = np.sum(np.abs(x - xp))
    return np.exp(-gamma * l1_dist)


print("=" * 60)
print("LAPLACIAN KERNEL")
print("=" * 60)

# Verify PSD
np.random.seed(42)
X_lap = np.random.randn(25, 3)
K_lap = kernel_matrix(X_lap, X_lap, laplacian_kernel, gamma=0.5)
check_psd(K_lap, "Laplacian kernel (γ=0.5)")


# ============================================================
# LAPLACIAN vs RBF: BOUNDARY COMPARISON
# ============================================================

np.random.seed(42)
X_lp, y_lp = make_moons(n_samples=300, noise=0.25, random_state=42)
X_lp_s     = StandardScaler().fit_transform(X_lp)

Xtr, Xte, ytr, yte = train_test_split(
    X_lp_s, y_lp, test_size=0.25, random_state=42
)

# sklearn uses 'rbf' for Gaussian; for Laplacian use chi2 or precomputed
# We demonstrate using additive_chi2 as a comparison
# and precomputed kernel for Laplacian

K_train = kernel_matrix(Xtr, Xtr, laplacian_kernel, gamma=0.5)
K_test  = kernel_matrix(Xte, Xtr, laplacian_kernel, gamma=0.5)

lap_clf = SVC(kernel='precomputed', C=1.0)
lap_clf.fit(K_train, ytr)
lap_acc = lap_clf.score(K_test, yte) * 100

rbf_clf2 = SVC(kernel='rbf', gamma='scale', C=1.0)
rbf_clf2.fit(Xtr, ytr)
rbf_acc = rbf_clf2.score(Xte, yte) * 100

print(f"\nLaplacian kernel test accuracy: {lap_acc:.1f}%")
print(f"RBF kernel test accuracy:       {rbf_acc:.1f}%")


# ============================================================
# VISUAL: L1 vs L2 distance geometry
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Heatmap: L1 distance from origin
xx, yy  = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
origin  = np.array([0.0, 0.0])
l1_dist = np.abs(xx) + np.abs(yy)
l2_dist = np.sqrt(xx**2 + yy**2)

lap_vals = np.exp(-0.5 * l1_dist)
rbf_vals = np.exp(-0.5 * l2_dist**2)

ax = axes[0]
cf = ax.contourf(xx, yy, lap_vals, levels=20, cmap='viridis')
ax.contour(xx, yy, lap_vals, levels=6, colors='white', linewidths=0.5)
plt.colorbar(cf, ax=ax)
ax.set_title('Laplacian K(x, 0)\nL1 distance → diamond contours')
ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

ax = axes[1]
cf = ax.contourf(xx, yy, rbf_vals, levels=20, cmap='viridis')
ax.contour(xx, yy, rbf_vals, levels=6, colors='white', linewidths=0.5)
plt.colorbar(cf, ax=ax)
ax.set_title('RBF K(x, 0)\nL2 distance → circular contours')
ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

ax = axes[2]
ax.plot(np.linspace(0, 4, 200),
        np.exp(-0.5 * np.linspace(0, 4, 200)), 'b-', linewidth=2,
        label='Laplacian: exp(-γ·r₁)')
ax.plot(np.linspace(0, 4, 200),
        np.exp(-0.5 * np.linspace(0, 4, 200)**2), 'r--', linewidth=2,
        label='RBF: exp(-γ·r₂²)')
ax.set_xlabel('Distance (r)'); ax.set_ylabel('K(x, x₀)')
ax.set_title('Similarity Decay:\nLaplacian vs RBF')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('laplacian_vs_rbf.png', dpi=150, bbox_inches='tight')
print("Saved: laplacian_vs_rbf.png")
```

---

## Matérn Kernel

### Mathematical Definition

```
Matérn kernel (general form):

K_ν(x, x') = (2^(1-ν) / Γ(ν)) · (√(2ν) r / l)^ν · K_ν(√(2ν) r / l)

Where:
  r = ||x - x'||    (Euclidean distance)
  l = length scale  (controls spatial reach)
  ν = smoothness    (controls differentiability)
  K_ν = modified Bessel function of second kind
  Γ   = gamma function

Common special cases (much simpler):

ν = 1/2:   K(x,x') = exp(-r/l)               ← Laplacian!
ν = 3/2:   K(x,x') = (1 + √3·r/l)·exp(-√3·r/l)
ν = 5/2:   K(x,x') = (1 + √5·r/l + 5r²/(3l²))·exp(-√5·r/l)
ν → ∞:     K(x,x') = exp(-r²/(2l²))          ← RBF/Gaussian!
```

### Why Matérn Matters

The Matérn kernel is dominant in **Gaussian Processes** because ν controls:

```
ν = 1/2 → once differentiable (rough, jagged)
ν = 3/2 → once differentiable (moderate smoothness)
ν = 5/2 → twice differentiable (smooth, common default)
ν → ∞   → infinitely differentiable (very smooth = RBF)

Match the kernel smoothness to your physical domain!
  Weather data:    ν = 5/2 (varies smoothly)
  Terrain height:  ν = 3/2 (moderate roughness)
  Stock prices:    ν = 1/2 (rough, unpredictable)
```

### Complete Implementation

```python
# ============================================================
# MATERN KERNEL: COMPLETE IMPLEMENTATION
# ============================================================

def matern_kernel_1_2(x, xp, l=1.0):
    """ν=1/2: K = exp(-r/l)  [same as Laplacian with γ=1/l]"""
    r = np.sqrt(np.sum((x - xp)**2))
    return np.exp(-r / l)

def matern_kernel_3_2(x, xp, l=1.0):
    """ν=3/2: K = (1 + √3·r/l) · exp(-√3·r/l)"""
    r     = np.sqrt(np.sum((x - xp)**2))
    sqrt3 = np.sqrt(3)
    return (1 + sqrt3 * r / l) * np.exp(-sqrt3 * r / l)

def matern_kernel_5_2(x, xp, l=1.0):
    """ν=5/2: K = (1 + √5·r/l + 5r²/(3l²)) · exp(-√5·r/l)"""
    r     = np.sqrt(np.sum((x - xp)**2))
    sqrt5 = np.sqrt(5)
    return (1 + sqrt5*r/l + 5*r**2/(3*l**2)) * np.exp(-sqrt5 * r / l)

def rbf_kernel_gp(x, xp, l=1.0):
    """ν→∞: Gaussian (RBF) K = exp(-r²/(2l²))"""
    r2 = np.sum((x - xp)**2)
    return np.exp(-r2 / (2 * l**2))


print("=" * 60)
print("MATERN KERNEL")
print("=" * 60)

# Verify all are valid
np.random.seed(42)
X_mat = np.random.randn(20, 2)

for name, kfn in [
    ("Matérn ν=1/2", matern_kernel_1_2),
    ("Matérn ν=3/2", matern_kernel_3_2),
    ("Matérn ν=5/2", matern_kernel_5_2),
    ("RBF (ν→∞)",    rbf_kernel_gp),
]:
    K = kernel_matrix(X_mat, X_mat, kfn, l=1.0)
    check_psd(K, name)


# ============================================================
# VISUALIZE: Smoothness difference
# ============================================================

np.random.seed(42)
n_pts = 10
X_1d  = np.sort(np.random.uniform(0, 5, n_pts)).reshape(-1, 1)
y_1d  = np.sin(X_1d.ravel()) + np.random.randn(n_pts) * 0.15

x_range = np.linspace(-0.5, 5.5, 300).reshape(-1, 1)
lam     = 0.01
l       = 1.0

matern_kernels = [
    ("ν=1/2 (rough)",   matern_kernel_1_2),
    ("ν=3/2 (medium)",  matern_kernel_3_2),
    ("ν=5/2 (smooth)",  matern_kernel_5_2),
    ("ν→∞  (RBF)",      rbf_kernel_gp),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)

for ax, (label, kfn) in zip(axes, matern_kernels):
    K_tr = kernel_matrix(X_1d, X_1d, kfn, l=l)
    K_pr = kernel_matrix(x_range, X_1d, kfn, l=l)
    alpha = np.linalg.solve(K_tr + lam * np.eye(n_pts), y_1d)
    y_pred = K_pr @ alpha

    ax.scatter(X_1d, y_1d, c='k', s=60, zorder=5, label='Data')
    ax.plot(x_range, y_pred, 'b-', linewidth=2, label='Prediction')
    ax.plot(x_range, np.sin(x_range), 'g--',
            linewidth=1.5, alpha=0.6, label='True sin')
    ax.set_title(f'Matérn {label}')
    ax.set_xlabel('x'); ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 2)

axes[0].set_ylabel('y')
axes[0].legend(fontsize=8)

plt.suptitle('Matérn Kernel: Effect of Smoothness ν\n'
             'ν=1/2 (Laplacian) → ν→∞ (RBF/Gaussian)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('matern_smoothness.png', dpi=150, bbox_inches='tight')
print("Saved: matern_smoothness.png")
```

---

## Kernel Comparison on Real Datasets

### Head-to-Head Benchmark

```python
# ============================================================
# COMPLETE KERNEL COMPARISON
# ============================================================

import time
from sklearn.metrics import accuracy_score

print("=" * 70)
print("COMPLETE KERNEL COMPARISON ON MULTIPLE DATASETS")
print("=" * 70)

# Datasets
datasets = {
    "Moons":       make_moons(n_samples=400, noise=0.2, random_state=42),
    "Circles":     make_circles(n_samples=400, noise=0.1,
                                factor=0.4, random_state=42),
    "Linear":      make_classification(n_samples=400, n_features=2,
                                       n_redundant=0, n_informative=2,
                                       class_sep=2.0, random_state=42),
}

# Kernels to compare
kernels_config = {
    "Linear":      {'kernel': 'linear', 'C': 1.0},
    "Poly d=2":    {'kernel': 'poly', 'degree': 2, 'coef0': 1, 'C': 1.0},
    "Poly d=3":    {'kernel': 'poly', 'degree': 3, 'coef0': 1, 'C': 1.0},
    "RBF":         {'kernel': 'rbf', 'gamma': 'scale', 'C': 1.0},
    "Sigmoid":     {'kernel': 'sigmoid', 'gamma': 0.1, 'coef0': 0, 'C': 1.0},
}

results = {}

for ds_name, (X, y) in datasets.items():
    Xs = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.25, random_state=42
    )
    results[ds_name] = {}

    for kname, kparams in kernels_config.items():
        clf = SVC(**kparams)
        t0  = time.perf_counter()
        clf.fit(Xtr, ytr)
        t_train = time.perf_counter() - t0

        acc    = clf.score(Xte, yte) * 100
        n_sv   = clf.n_support_.sum()
        results[ds_name][kname] = {
            'acc': acc, 'n_sv': n_sv, 'time': t_train
        }

# Print table
print(f"\n{'Kernel':<15}", end="")
for ds in datasets:
    print(f"{ds + ' Acc%':<15}", end="")
print()
print("-" * (15 + 15 * len(datasets)))

for kname in kernels_config:
    print(f"{kname:<15}", end="")
    for ds in datasets:
        r = results[ds][kname]
        print(f"{r['acc']:<15.1f}", end="")
    print()

# Visualization
fig, axes = plt.subplots(len(datasets), len(kernels_config),
                         figsize=(25, 15))

for row, (ds_name, (X, y)) in enumerate(datasets.items()):
    Xs = StandardScaler().fit_transform(X)
    Xtr, _, ytr, _ = train_test_split(Xs, y, test_size=0.25,
                                      random_state=42)

    for col, (kname, kparams) in enumerate(kernels_config.items()):
        ax  = axes[row, col]
        clf = SVC(**kparams).fit(Xtr, ytr)
        acc = clf.score(Xs, y) * 100
        plot_decision_boundary(
            ax, clf, Xs, y,
            f"{kname}\n{ds_name}\nAcc={acc:.0f}%"
        )

plt.suptitle("All Kernels × All Datasets",
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('kernel_comparison_complete.png', dpi=150, bbox_inches='tight')
print("\nSaved: kernel_comparison_complete.png")


# ============================================================
# CROSS-VALIDATION COMPARISON WITH ERROR BARS
# ============================================================

from sklearn.model_selection import cross_val_score

print("\n--- Cross-Validation Comparison ---")

np.random.seed(42)
X_cv, y_cv = make_moons(n_samples=300, noise=0.2, random_state=42)
X_cv_s     = StandardScaler().fit_transform(X_cv)

cv_results = {}
for kname, kparams in kernels_config.items():
    clf    = SVC(**kparams)
    scores = cross_val_score(clf, X_cv_s, y_cv, cv=10)
    cv_results[kname] = scores
    print(f"  {kname:<15}  "
          f"mean={scores.mean()*100:.1f}%  "
          f"std={scores.std()*100:.1f}%")

# Error bar plot
names  = list(cv_results.keys())
means  = [cv_results[k].mean() * 100 for k in names]
stds   = [cv_results[k].std()  * 100 for k in names]

plt.figure(figsize=(8, 5))
plt.barh(range(len(names)), means, xerr=stds,
         capsize=4, color='steelblue', alpha=0.7)
plt.yticks(range(len(names)), names)
plt.xlabel('CV Accuracy (%)')
plt.title('10-Fold CV Accuracy by Kernel\n(Error bars = std)')
plt.xlim(min(means) - 15, 105)
plt.grid(True, alpha=0.3, axis='x')
plt.axvline(x=90, color='red', linestyle='--', alpha=0.5, label='90%')
plt.legend()
plt.tight_layout()
plt.savefig('kernel_cv_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: kernel_cv_comparison.png")
```

---

## How to Choose the Right Kernel

### Decision Guide

```python
# ============================================================
# KERNEL SELECTION GUIDE
# ============================================================

def kernel_selection_guide(
        n_samples, n_features, data_type,
        need_interp, has_outliers, is_smooth
):
    """
    Step-by-step kernel selection guide.
    """
    print("\n" + "="*60)
    print("KERNEL SELECTION GUIDE")
    print("="*60)

    print(f"\nYour data:")
    print(f"  Samples:         {n_samples}")
    print(f"  Features:        {n_features}")
    print(f"  Data type:       {data_type}")
    print(f"  Need interp:     {need_interp}")
    print(f"  Has outliers:    {has_outliers}")
    print(f"  Smooth patterns: {is_smooth}")

    recommendations = []

    # Rule 1: High-dimensional or text
    if n_features > 500 or data_type in ['text', 'genomics']:
        recommendations.append(("Linear", "HIGH",
            "High-dim data often linearly separable; fast; interpretable"))

    # Rule 2: Small dataset, non-linear
    if n_samples < 5000 and data_type == 'numerical':
        recommendations.append(("RBF", "HIGH",
            "Universal approximator; most versatile; start here"))

    # Rule 3: Outliers
    if has_outliers:
        recommendations.append(("Laplacian", "MEDIUM",
            "L1 distance is more robust to outliers than L2"))

    # Rule 4: Known polynomial structure
    if data_type == 'image_features':
        recommendations.append(("Polynomial d=2", "MEDIUM",
            "Image features often have quadratic interactions"))

    # Rule 5: Time series / spatial
    if data_type in ['spatial', 'time_series']:
        recommendations.append(("Matérn ν=5/2", "HIGH",
            "Controls smoothness; matches physical processes"))

    # Rule 6: Large dataset
    if n_samples > 50000:
        recommendations.append(("Linear", "HIGH",
            "Non-linear kernels scale O(n²-n³); too slow"))
        recommendations.append(("RBF (approximate)", "MEDIUM",
            "Use Nyström or Random Fourier Features approximation"))

    print("\nRecommendations:")
    print("-"*60)
    for kernel, priority, reason in recommendations:
        print(f"  [{priority}] {kernel}")
        print(f"        {reason}")

    print("\nUniversal workflow:")
    print("  1. Always try Linear first (fast baseline)")
    print("  2. If linear fails → try RBF with GridSearchCV")
    print("  3. If domain-specific → use specialized kernel")
    print("  4. Validate with 5-fold CV")


# Examples
kernel_selection_guide(500,  10,    'numerical',     False, False, True)
kernel_selection_guide(200,  5000,  'text',          True,  False, True)
kernel_selection_guide(1000, 20,    'numerical',     False, True,  False)
kernel_selection_guide(50,   3,     'spatial',       False, False, True)
```

### Properties Summary Table

```
╔══════════════════════════════════════════════════════════════════════╗
║           KERNEL COMPARISON TABLE                                    ║
╠══════════╦══════════════════════╦══════════╦══════════╦═════════════╣
║ Kernel   ║ Formula              ║ Always   ║ Feature  ║ Best for    ║
║          ║                      ║ PSD?     ║ Space    ║             ║
╠══════════╬══════════════════════╬══════════╬══════════╬═════════════╣
║ Linear   ║ xᵀx'                 ║ ✓ Yes    ║ ℝᵈ       ║ Text, high-d║
║ Poly     ║ (γxᵀx' + c)ᵈ        ║ ✓ Yes*   ║ ℝᶜ⁽ⁿ'ᵈ⁾  ║ Images      ║
║ RBF      ║ exp(-γ||x-x'||²)    ║ ✓ Yes    ║ ℝ^∞      ║ General     ║
║ Sigmoid  ║ tanh(γxᵀx' + c)     ║ ✗ Maybe  ║ ℝ^∞?     ║ Rarely used ║
║ Laplacian║ exp(-γ||x-x'||₁)   ║ ✓ Yes    ║ ℝ^∞      ║ Outliers    ║
║ Matérn   ║ (see formula)        ║ ✓ Yes    ║ ℝ^∞      ║ Spatial/GP  ║
╚══════════╩══════════════════════╩══════════╩══════════╩═════════════╝
* Polynomial PSD requires c ≥ 0 and integer d ≥ 1
```

### The Three-Step Kernel Selection Process

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

def select_best_kernel(X_train, y_train, cv=5):
    """
    Automated three-step kernel selection.
    Step 1: Quick accuracy comparison
    Step 2: Hyperparameter tuning on best kernels
    Step 3: Final validation
    """

    print("="*60)
    print("STEP 1: Quick Kernel Comparison (default params)")
    print("="*60)

    candidate_kernels = {
        'Linear':    SVC(kernel='linear',  C=1.0),
        'Poly d=2':  SVC(kernel='poly', degree=2, coef0=1, C=1.0),
        'Poly d=3':  SVC(kernel='poly', degree=3, coef0=1, C=1.0),
        'RBF':       SVC(kernel='rbf', gamma='scale', C=1.0),
        'Sigmoid':   SVC(kernel='sigmoid', gamma=0.1, C=1.0),
    }

    Xs = StandardScaler().fit_transform(X_train)
    step1_results = {}

    for name, clf in candidate_kernels.items():
        scores = cross_val_score(clf, Xs, y_train, cv=cv)
        step1_results[name] = scores.mean()
        print(f"  {name:<15} CV acc = {scores.mean()*100:.1f}% "
              f"± {scores.std()*100:.1f}%")

    # Identify top 2
    top2 = sorted(step1_results, key=step1_results.get, reverse=True)[:2]
    print(f"\nTop 2 candidates: {top2}")

    print("\n" + "="*60)
    print("STEP 2: Hyperparameter Tuning on Top Candidates")
    print("="*60)

    param_grids = {
        'Linear':  {'svm__C': [0.01, 0.1, 1, 10, 100]},
        'Poly d=2':{'svm__C': [0.1, 1, 10],
                    'svm__coef0': [0, 1, 2],
                    'svm__gamma': ['scale', 'auto']},
        'Poly d=3':{'svm__C': [0.1, 1, 10],
                    'svm__coef0': [0, 1, 2],
                    'svm__gamma': ['scale', 'auto']},
        'RBF':     {'svm__C': [0.1, 1, 10, 100],
                    'svm__gamma': [0.001, 0.01, 0.1, 1, 'scale']},
        'Sigmoid': {'svm__C': [0.1, 1, 10],
                    'svm__gamma': [0.01, 0.1, 0.5]},
    }

    step2_results = {}
    for kname in top2:
        params_key = kname
        if params_key not in param_grids:
            continue

        base_params = candidate_kernels[kname].get_params()
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**base_params))
        ])

        # Fix param names for pipeline
        grid_params = {
            f'svm__{k}': v
            for k, v in param_grids[params_key].items()
            if k.startswith('svm__')
        } if any(k.startswith('svm__') for k in param_grids[params_key]) \
        else param_grids[params_key]

        gs = GridSearchCV(pipe, grid_params, cv=cv,
                         scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)

        step2_results[kname] = {
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        }
        print(f"  {kname}:")
        print(f"    Best CV: {gs.best_score_*100:.1f}%")
        print(f"    Params:  {gs.best_params_}")

    winner = max(step2_results, key=lambda k: step2_results[k]['best_score'])
    print(f"\n  → WINNER: {winner} "
          f"({step2_results[winner]['best_score']*100:.1f}%)")

    return winner, step2_results[winner]


# Run on moons dataset
np.random.seed(42)
X_sel, y_sel = make_moons(n_samples=300, noise=0.2, random_state=42)
best_kernel, best_info = select_best_kernel(X_sel, y_sel, cv=5)
```

---

## Summary and Quick Reference

### Kernel Properties at a Glance

```python
# Print comprehensive summary
print("\n" + "="*70)
print("KERNEL FUNCTIONS: COMPLETE SUMMARY")
print("="*70)

summary = [
    ("1. Linear", "xᵀx'",
     "None", "Linear", "Text, genomics, high-d"),
    ("2. Polynomial", "(γxᵀx' + c)ᵈ",
     "d, c, γ", "Polynomial/conic", "Image features, moderate non-linearity"),
    ("3. RBF", "exp(-γ||x-x'||²)",
     "γ", "Infinite (Gaussian)", "Default choice; universal approximator"),
    ("4. Sigmoid", "tanh(γxᵀx' + c)",
     "γ, c", "Neural-net-like", "Verify PSD first; rarely best"),
    ("5. Laplacian", "exp(-γ||x-x'||₁)",
     "γ", "Infinite (Laplace)", "Outlier-robust; discontinuous data"),
    ("6. Matérn", "See formulas",
     "ν, l", "Infinite (smooth)",  "Spatial data; Gaussian processes"),
]

print(f"\n{'Kernel':<18} {'Formula':<24} {'Params':<12} "
      f"{'Feature Space':<22} {'Best for'}")
print("-"*100)
for row in summary:
    print(f"{row[0]:<18} {row[1]:<24} {row[2]:<12} {row[3]:<22} {row[4]}")
```

### Prerequisites Checklist

Before moving to Chapter 04, confirm you can:
- [ ] Write the formula for all 6 kernels from memory
- [ ] Explain which kernel is always PSD and which might not be
- [ ] Implement any kernel function in NumPy from scratch
- [ ] Explain the effect of γ on RBF decision boundaries
- [ ] Explain the effect of d and c on the polynomial kernel
- [ ] Choose the right kernel for a given dataset using the selection guide
- [ ] Run a GridSearchCV over kernel hyperparameters in sklearn
- [ ] Explain why linear kernel is preferred for text classification

### What's Next

**04_Dual_Formulations.md** ← already exists
→ How kernels are plugged into the SVM dual problem mathematically

**07_SVM_Hard_Margin.md**
→ The full optimization derivation; where these kernel formulas appear

**15_Hyperparameter_Tuning_Kernels.md**
→ Advanced tuning strategies: nested CV, Bayesian optimization, learning curves

---

## Quick Reference Card

```
╔═══════════════════════════════════════════════════════════════════╗
║              COMMON KERNEL FUNCTIONS — QUICK REFERENCE            ║
╠═══════════════════════════════════════════════════════════════════╣
║  LINEAR                                                           ║
║    K(x,x') = xᵀx'                                                 ║
║    sklearn: SVC(kernel='linear', C=...)                           ║
║    Use when: high-dim data, text, already separable               ║
╠═══════════════════════════════════════════════════════════════════╣
║  POLYNOMIAL                                                       ║
║    K(x,x') = (γ xᵀx' + c)ᵈ                                        ║
║    sklearn: SVC(kernel='poly', degree=d, coef0=c, gamma=γ)        ║
║    Tune:  d ∈ {2,3,4},  c ∈ {0,1,2},  γ = 'scale'               ║
║    Use when: known polynomial structure, image features           ║
╠═══════════════════════════════════════════════════════════════════╣
║  RBF (default choice)                                             ║
║    K(x,x') = exp(-γ ||x-x'||²)                                    ║
║    sklearn: SVC(kernel='rbf', gamma=γ, C=C)                       ║
║    Tune:  C ∈ [0.1,100],  γ ∈ [0.001,10] or 'scale'             ║
║    Use when: general non-linear; start here always                ║
╠═══════════════════════════════════════════════════════════════════╣
║  SIGMOID                                                          ║
║    K(x,x') = tanh(γ xᵀx' + c)                                     ║
║    sklearn: SVC(kernel='sigmoid', gamma=γ, coef0=c)               ║
║    ⚠️  NOT always PSD — verify before use                          ║
╠═══════════════════════════════════════════════════════════════════╣
║  LAPLACIAN                                                        ║
║    K(x,x') = exp(-γ ||x-x'||₁)                                    ║
║    sklearn: No direct support; use precomputed kernel matrix      ║
║    Use when: outliers present; L1-robust setting                  ║
╠═══════════════════════════════════════════════════════════════════╣
║  MATERN                                                           ║
║    ν=3/2:  (1 + √3r/l) exp(-√3r/l)                               ║
║    ν=5/2:  (1 + √5r/l + 5r²/3l²) exp(-√5r/l)                    ║
║    sklearn: Available in GaussianProcessClassifier/Regressor      ║
║    Use when: spatial data, Gaussian Processes                     ║
╠═══════════════════════════════════════════════════════════════════╣
║  SELECTION WORKFLOW                                               ║
║    1. Try Linear  (fast baseline)                                 ║
║    2. Try RBF     (if linear fails)                               ║
║    3. GridSearchCV on C and gamma                                 ║
║    4. Use Laplacian if outliers present                           ║
║    5. Use Matérn if spatial/GP context                            ║
╠═══════════════════════════════════════════════════════════════════╣
║  SKLEARN TEMPLATE                                                 ║
║    from sklearn.pipeline import Pipeline                          ║
║    from sklearn.svm import SVC                                    ║
║    from sklearn.preprocessing import StandardScaler               ║
║    from sklearn.model_selection import GridSearchCV               ║
║                                                                   ║
║    pipe = Pipeline([('sc', StandardScaler()), ('svm', SVC())])    ║
║    grid = {                                                       ║
║        'svm__kernel': ['linear', 'rbf', 'poly'],                  ║
║        'svm__C':      [0.1, 1, 10, 100],                         ║
║        'svm__gamma':  ['scale', 0.01, 0.1, 1],                   ║
║    }                                                              ║
║    gs = GridSearchCV(pipe, grid, cv=5).fit(X_train, y_train)      ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

*End of 03_Common_Kernel_Functions.md*