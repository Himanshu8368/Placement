# Introduction to Kernel Methods

## Table of Contents
1. [What Are Kernel Methods?](#what-are-kernel-methods)
2. [Why Kernel Methods Matter](#why-kernel-methods-matter)
3. [The Kernel Trick Intuition](#the-kernel-trick-intuition)
4. [Linear vs Non-Linear Decision Boundaries](#linear-vs-non-linear-decision-boundaries)
5. [Overview of Kernel-Based Algorithms](#overview-of-kernel-based-algorithms)
6. [When to Use Kernel Methods](#when-to-use-kernel-methods)
7. [Summary and Next Steps](#summary-and-next-steps)

---

## What Are Kernel Methods?

### Definition

**Kernel methods** are a class of machine learning algorithms that work with **non-linear patterns** using **linear algorithms** — without ever explicitly computing the transformation into high-dimensional space.

```
Linear World:    Simple but limited (can't handle curves)
Non-linear World: Powerful but expensive
Kernel Methods:  Get the power of non-linear at the cost of linear!
```

### The Core Idea

Every kernel method relies on one fundamental insight:

> We never need the actual transformed features φ(x).
> We only need the **inner product** ⟨φ(x), φ(x')⟩.
> And we can compute this directly with a **kernel function** K(x, x').

```
Traditional ML pipeline:
  x → φ(x) → linear_algorithm(φ(x))
      ↑
  EXPENSIVE if φ(x) is high-dimensional!

Kernel ML pipeline:
  x, x' → K(x, x') = ⟨φ(x), φ(x')⟩ → linear_algorithm
                ↑
  CHEAP! Never compute φ(x) explicitly.
```

### Real-World Analogy

Imagine classifying emails as spam or not. The words in emails interact — "free" alone is OK, but "free" + "click" + "now" together = spam.

- **Linear model**: Treats each word independently → misses combinations
- **Kernel SVM**: Uses a kernel that captures word combinations implicitly → catches spam!

### Mathematical Definition

A **kernel function** K: X × X → ℝ satisfies:

```
K(x, x') = ⟨φ(x), φ(x')⟩_H

Where:
  x, x'  = input data points
  φ       = feature map to a high-dimensional Hilbert space H
  ⟨·, ·⟩_H = inner product in that space
```

**Valid kernels** must satisfy Mercer's condition (positive semi-definite).

### From Scratch: Computing a Kernel

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# KERNEL FUNCTIONS FROM SCRATCH
# ============================================================

def linear_kernel(x1, x2):
    """K(x, x') = x^T x'"""
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=2, coef=1):
    """K(x, x') = (x^T x' + c)^d"""
    return (np.dot(x1, x2) + coef) ** degree

def rbf_kernel(x1, x2, gamma=1.0):
    """K(x, x') = exp(-gamma * ||x - x'||^2)"""
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))

def sigmoid_kernel(x1, x2, gamma=0.1, coef=0):
    """K(x, x') = tanh(gamma * x^T x' + c)"""
    return np.tanh(gamma * np.dot(x1, x2) + coef)


# ============================================================
# DEMO: What does each kernel compute?
# ============================================================

x1 = np.array([1.0, 2.0])
x2 = np.array([3.0, 4.0])

print("="*60)
print("KERNEL VALUES FOR x1=[1,2], x2=[3,4]")
print("="*60)

print(f"\nLinear    K(x1,x2) = {linear_kernel(x1, x2):.4f}")
print(f"  → Raw dot product: 1×3 + 2×4 = 11")

print(f"\nPolynomial K(x1,x2) = {polynomial_kernel(x1, x2, degree=2):.4f}")
print(f"  → (11 + 1)^2 = 144")

print(f"\nRBF       K(x1,x2) = {rbf_kernel(x1, x2, gamma=0.1):.4f}")
print(f"  → exp(-0.1 * ||[1,2]-[3,4]||^2) = exp(-0.8)")

print(f"\nSigmoid   K(x1,x2) = {sigmoid_kernel(x1, x2):.4f}")
print(f"  → tanh(0.1 * 11 + 0) = tanh(1.1)")


# ============================================================
# KERNEL MATRIX (Gram Matrix)
# ============================================================

def compute_kernel_matrix(X, kernel_fn, **kwargs):
    """
    Compute kernel matrix K where K[i,j] = K(x_i, x_j)
    This is also called the Gram matrix.
    """
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_fn(X[i], X[j], **kwargs)
    return K


# Small dataset
X_demo = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 2.0],
])

print("\n" + "="*60)
print("KERNEL MATRICES (4×4 for 4 data points)")
print("="*60)

K_linear = compute_kernel_matrix(X_demo, linear_kernel)
K_rbf    = compute_kernel_matrix(X_demo, rbf_kernel, gamma=0.5)

print("\nLinear Kernel Matrix:")
print(np.round(K_linear, 3))
print("\nRBF Kernel Matrix (gamma=0.5):")
print(np.round(K_rbf, 3))

print("\nNote: K[i,j] = similarity between point i and point j")
print("  Large K[i,j] → points are SIMILAR in feature space")
print("  Small K[i,j] → points are DIFFERENT in feature space")
```

---

## Why Kernel Methods Matter

### The Problem: Linear Models Are Limited

Linear models can only find straight-line decision boundaries. Real-world data rarely cooperates.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons

# ============================================================
# SHOW WHERE LINEAR MODELS FAIL
# ============================================================

np.random.seed(42)

# Dataset 1: Concentric circles (radial pattern)
X_circles, y_circles = make_circles(n_samples=200, noise=0.1,
                                    factor=0.3, random_state=42)

# Dataset 2: Interleaved moons (curved pattern)
X_moons, y_moons = make_moons(n_samples=200, noise=0.15,
                               random_state=42)

# Dataset 3: XOR pattern
X_xor = np.array([[0.0, 0.0], [0.0, 1.0],
                  [1.0, 0.0], [1.0, 1.0]])
y_xor = np.array([0, 1, 1, 0])

datasets = [
    (X_circles, y_circles, "Concentric Circles"),
    (X_moons, y_moons, "Interleaved Moons"),
    (X_xor, y_xor, "XOR Pattern"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for col, (X, y, name) in enumerate(datasets):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # ---- Linear classifier ----
    ax = axes[0, col]
    linear = LogisticRegression(max_iter=1000)
    linear.fit(X_s, y)

    xx, yy = np.meshgrid(
        np.linspace(X_s[:,0].min()-0.5, X_s[:,0].max()+0.5, 200),
        np.linspace(X_s[:,1].min()-0.5, X_s[:,1].max()+0.5, 200),
    )
    Z = linear.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_s[y==0,0], X_s[y==0,1], c='blue', s=40,
               edgecolors='k', linewidths=0.5, alpha=0.8)
    ax.scatter(X_s[y==1,0], X_s[y==1,1], c='red', s=40,
               marker='s', edgecolors='k', linewidths=0.5, alpha=0.8)
    acc = linear.score(X_s, y)
    ax.set_title(f"Linear Model\n{name}\nAccuracy: {acc*100:.0f}%")
    ax.grid(True, alpha=0.3)

    # ---- Kernel classifier ----
    ax = axes[1, col]
    kernel = SVC(kernel='rbf', gamma='auto', C=1.0)
    kernel.fit(X_s, y)

    Z = kernel.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_s[y==0,0], X_s[y==0,1], c='blue', s=40,
               edgecolors='k', linewidths=0.5, alpha=0.8)
    ax.scatter(X_s[y==1,0], X_s[y==1,1], c='red', s=40,
               marker='s', edgecolors='k', linewidths=0.5, alpha=0.8)
    acc = kernel.score(X_s, y)
    ax.set_title(f"Kernel SVM (RBF)\n{name}\nAccuracy: {acc*100:.0f}%")
    ax.grid(True, alpha=0.3)

plt.suptitle("Linear vs Kernel Methods", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('linear_vs_kernel.png', dpi=150, bbox_inches='tight')
print("Saved: linear_vs_kernel.png")
```

**What you will observe**:

| Dataset | Linear Accuracy | Kernel Accuracy |
|---------|----------------|-----------------|
| Circles | ~50% (random!) | ~99% |
| Moons   | ~85%           | ~98% |
| XOR     | 50% (random!)  | 100% |

### Five Reasons Kernel Methods Matter

**1. Handle Non-Linearity Without Feature Engineering**
```
Manual approach: You must THINK about and CREATE polynomial features
Kernel approach: The kernel does it AUTOMATICALLY and IMPLICITLY
```

**2. Work in Infinite-Dimensional Spaces**
```
RBF kernel corresponds to a feature space with INFINITE dimensions.
We could never compute φ(x) explicitly, yet we work there effortlessly.
```

**3. Mathematically Principled**
```
Convex optimization → guaranteed global optimum (no local minima!)
Well-understood theory → reliable generalization bounds
Mercer's theorem → clear conditions for valid kernels
```

**4. Versatile: Works on Non-Vector Data**
```
Standard ML requires vectors. Kernel methods only need a similarity function.
→ Strings (DNA sequences, text)
→ Graphs (molecules, social networks)
→ Sets (image patches)
→ Trees (parse trees in NLP)
```

**5. Excellent on Small-to-Medium Datasets**
```
Deep learning needs millions of samples to shine.
SVMs regularly achieve state-of-art with just hundreds of samples.
```

---

## The Kernel Trick Intuition

### Step 1: Why We Need High-Dimensional Spaces

Some data is not separable in original space but becomes separable when lifted to higher dimensions.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# LIFTING DATA INTO HIGHER DIMENSIONS
# ============================================================

np.random.seed(42)
n = 150

# Concentric circles in 2D
r_inner = np.random.uniform(0, 1.0, n)
t_inner = np.random.uniform(0, 2*np.pi, n)
X_inner = np.column_stack([r_inner * np.cos(t_inner),
                           r_inner * np.sin(t_inner)])
y_inner = np.zeros(n)

r_outer = np.random.uniform(2.0, 3.0, n)
t_outer = np.random.uniform(0, 2*np.pi, n)
X_outer = np.column_stack([r_outer * np.cos(t_outer),
                           r_outer * np.sin(t_outer)])
y_outer = np.ones(n)

X = np.vstack([X_inner, X_outer])
y = np.hstack([y_inner, y_outer])


# Feature map: φ(x₁, x₂) = (x₁, x₂, x₁² + x₂²)
def feature_map_circle(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([x1, x2, x1**2 + x2**2])

X_lifted = feature_map_circle(X)

fig = plt.figure(figsize=(14, 5))

# 2D original (not separable)
ax1 = fig.add_subplot(131)
ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=20, alpha=0.6, label='Inner')
ax1.scatter(X[y==1, 0], X[y==1, 1], c='red',  s=20, alpha=0.6, label='Outer')
ax1.set_title('Original 2D Space\nNOT separable')
ax1.legend(fontsize=9)
ax1.axis('equal')
ax1.grid(True, alpha=0.3)

# 3D lifted (separable!)
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(X_lifted[y==0,0], X_lifted[y==0,1], X_lifted[y==0,2],
            c='blue', s=20, alpha=0.5, label='Inner')
ax2.scatter(X_lifted[y==1,0], X_lifted[y==1,1], X_lifted[y==1,2],
            c='red',  s=20, alpha=0.5, label='Outer')

# Draw separating plane at z = 4
xx_p, yy_p = np.meshgrid(np.linspace(-3,3,10), np.linspace(-3,3,10))
zz_p = np.full_like(xx_p, 4.0)
ax2.plot_surface(xx_p, yy_p, zz_p, alpha=0.25, color='green')
ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂'); ax2.set_zlabel('x₁²+x₂²')
ax2.set_title('Lifted 3D Space\nSeparable with a plane!')
ax2.legend(fontsize=9)

# Explanation panel
ax3 = fig.add_subplot(133)
ax3.axis('off')
txt = (
    "THE LIFT:\n\n"
    "φ(x₁, x₂) = (x₁, x₂, x₁²+x₂²)\n\n"
    "Inner circle:\n"
    "  r < 1 → x₁²+x₂² < 1\n\n"
    "Outer ring:\n"
    "  r > 2 → x₁²+x₂² > 4\n\n"
    "In 3D: just cut at z = 4!\n\n"
    "PROBLEM:\n"
    "Computing φ(x) for 1000\n"
    "features is VERY expensive\n\n"
    "SOLUTION:\n"
    "Use a kernel! Never\n"
    "compute φ(x) directly."
)
ax3.text(0.5, 0.5, txt, ha='center', va='center', fontsize=11,
         family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow'))

plt.tight_layout()
plt.savefig('lifting_trick.png', dpi=150, bbox_inches='tight')
print("Saved: lifting_trick.png")
```

### Step 2: The Trick — Skip the Lift!

```python
# ============================================================
# THE KERNEL TRICK: VERIFY MATHEMATICALLY
# ============================================================

print("="*60)
print("THE KERNEL TRICK: MATHEMATICAL VERIFICATION")
print("="*60)

x  = np.array([2.0, 3.0])
xp = np.array([4.0, 1.0])

# ---- Explicit way: compute φ(x), then inner product ----
def phi(v):
    """φ(x₁, x₂) = (x₁², √2·x₁x₂, x₂²)"""
    return np.array([v[0]**2,
                     np.sqrt(2) * v[0] * v[1],
                     v[1]**2])

phi_x  = phi(x)
phi_xp = phi(xp)
explicit_inner = np.dot(phi_x, phi_xp)

print(f"\nx  = {x}")
print(f"x' = {xp}")
print(f"\nφ(x)  = {phi_x}")
print(f"φ(x') = {phi_xp}")
print(f"\nExplicit: ⟨φ(x), φ(x')⟩ = {explicit_inner:.4f}")

# ---- Kernel way: compute (x^T x')^2 directly ----
kernel_value = np.dot(x, xp) ** 2
print(f"\nKernel: K(x,x') = (x^T x')^2 = {kernel_value:.4f}")

print(f"\nAre they equal? {np.isclose(explicit_inner, kernel_value)}")
print("\n✓ SAME RESULT with much less computation!")

print("\n--- Operation count ---")
print(f"Explicit path: {len(phi_x) * 2} multiplications to build φ(x),")
print(f"               {len(phi_x)} multiplications for inner product")
print(f"               Total ≈ {len(phi_x)*3} operations")
print(f"Kernel path:   2 multiplications + 1 squaring")
print(f"               Total ≈ 3 operations")
print(f"\nSpeedup: {len(phi_x)*3 // 3}× (grows exponentially with degree/dim!)")
```

### Step 3: The Magic of RBF — Infinite Dimensions

```python
# ============================================================
# RBF KERNEL: INFINITE DIMENSIONAL FEATURE SPACE
# ============================================================

print("\n" + "="*60)
print("RBF KERNEL = INFINITE DIMENSIONAL FEATURE SPACE")
print("="*60)

print("""
RBF (Gaussian) Kernel:

  K(x, x') = exp(-γ ||x - x'||²)

Using Taylor expansion of exp():

  exp(-γ ||x-x'||²)
  = exp(-γ||x||²) · exp(2γ xᵀx') · exp(-γ||x'||²)

  = Σₙ₌₀^∞  (2γ)ⁿ/n! · xⁿ · x'ⁿ  (component-wise)

This is equivalent to an INFINITE feature map:

  φ(x) = [φ₁(x), φ₂(x), φ₃(x), ...]  ← infinitely many!

We could NEVER compute this φ(x) explicitly.
Yet K(x, x') = exp(-γ||x-x'||²) computes it in O(d) time!

This is the true power of the kernel trick.
""")

# Visualize RBF kernel: how similarity decays with distance
import numpy as np
import matplotlib.pyplot as plt

distances = np.linspace(0, 5, 300)
gammas = [0.1, 0.5, 1.0, 2.0, 5.0]

plt.figure(figsize=(10, 5))
for g in gammas:
    k_vals = np.exp(-g * distances**2)
    plt.plot(distances, k_vals, linewidth=2, label=f'γ = {g}')

plt.xlabel('Distance  ||x - x\'||', fontsize=13)
plt.ylabel('K(x, x\')  (similarity)', fontsize=13)
plt.title('RBF Kernel: Similarity vs. Distance for Different γ', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rbf_similarity.png', dpi=150, bbox_inches='tight')
print("Saved: rbf_similarity.png")
print("\nKey insight: Larger γ → faster decay → more local boundary")
print("             Smaller γ → slower decay → smoother boundary")
```

### Step 4: The Kernel Matrix

```python
# ============================================================
# KERNEL MATRIX (GRAM MATRIX)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# 20 points: 10 from class 0, 10 from class 1
X_c0 = np.random.randn(10, 2) + np.array([0, 0])
X_c1 = np.random.randn(10, 2) + np.array([4, 4])
X_all = np.vstack([X_c0, X_c1])
y_all = np.array([0]*10 + [1]*10)

def kernel_matrix(X, kernel_fn, **kw):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_fn(X[i], X[j], **kw)
    return K

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Linear kernel matrix
K_lin = kernel_matrix(X_all, linear_kernel)
sns.heatmap(K_lin, ax=axes[0], cmap='viridis', square=True, cbar=True)
axes[0].set_title('Linear Kernel Matrix\nK(x,x\') = x·x\'')
axes[0].axhline(y=10, color='white', linewidth=3)
axes[0].axvline(x=10, color='white', linewidth=3)

# RBF kernel matrix (γ = 0.1)
K_rbf = kernel_matrix(X_all, rbf_kernel, gamma=0.1)
sns.heatmap(K_rbf, ax=axes[1], cmap='viridis', square=True, cbar=True)
axes[1].set_title('RBF Kernel Matrix (γ=0.1)\nK(x,x\') = exp(-γ||x-x\'||²)')
axes[1].axhline(y=10, color='white', linewidth=3)
axes[1].axvline(x=10, color='white', linewidth=3)

# RBF kernel matrix (γ = 1.0)
K_rbf2 = kernel_matrix(X_all, rbf_kernel, gamma=1.0)
sns.heatmap(K_rbf2, ax=axes[2], cmap='viridis', square=True, cbar=True)
axes[2].set_title('RBF Kernel Matrix (γ=1.0)\nMore local similarity')
axes[2].axhline(y=10, color='white', linewidth=3)
axes[2].axvline(x=10, color='white', linewidth=3)

for ax in axes:
    ax.set_xlabel('Point index')
    ax.set_ylabel('Point index')

plt.suptitle('Kernel (Gram) Matrices — White line separates classes',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('kernel_matrices.png', dpi=150, bbox_inches='tight')
print("Saved: kernel_matrices.png")
print("\nNote the block structure: high similarity within class, low between classes")
```

---

## Linear vs Non-Linear Decision Boundaries

### What Is a Decision Boundary?

A decision boundary is the surface where the model's prediction changes from one class to another.

```
Linear boundary (2D):  w₁x₁ + w₂x₂ + b = 0   → a straight line
Linear boundary (3D):  w₁x₁ + w₂x₂ + w₃x₃ + b = 0 → a flat plane
Linear boundary (nD):  wᵀx + b = 0 → a hyperplane
```

Kernel methods create **non-linear** boundaries in input space, while still solving a **linear** problem in feature space.

### Complete Comparison: 5 Kernels on 3 Datasets

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

np.random.seed(42)

# Three datasets
X_lin, y_lin = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, class_sep=2.0, random_state=42)
X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
X_circ, y_circ  = make_circles(n_samples=200, noise=0.1,
                                factor=0.4, random_state=42)

datasets = [
    ('Linearly Separable', X_lin,   y_lin),
    ('Moons',             X_moons, y_moons),
    ('Circles',           X_circ,  y_circ),
]

# Five classifiers
classifiers = [
    ('Logistic\n(Linear)',        LogisticRegression(max_iter=1000)),
    ('SVM\nLinear',               SVC(kernel='linear',  C=1)),
    ('SVM\nPoly d=2',             SVC(kernel='poly', degree=2, coef0=1, C=1)),
    ('SVM\nPoly d=3',             SVC(kernel='poly', degree=3, coef0=1, C=1)),
    ('SVM\nRBF',                  SVC(kernel='rbf', gamma='scale', C=1)),
]

fig, axes = plt.subplots(len(datasets), len(classifiers),
                         figsize=(20, 12))

for row, (dname, X, y) in enumerate(datasets):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    x0min, x0max = Xs[:,0].min()-0.5, Xs[:,0].max()+0.5
    x1min, x1max = Xs[:,1].min()-0.5, Xs[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x0min, x0max, 200),
                         np.linspace(x1min, x1max, 200))

    for col, (cname, clf) in enumerate(classifiers):
        ax = axes[row, col]
        clf.fit(Xs, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.contour(xx, yy, Z, colors='k', linewidths=0.8, alpha=0.5)
        ax.scatter(Xs[y==0,0], Xs[y==0,1], c='royalblue', s=18,
                   edgecolors='k', linewidths=0.3, alpha=0.8)
        ax.scatter(Xs[y==1,0], Xs[y==1,1], c='tomato', s=18,
                   marker='s', edgecolors='k', linewidths=0.3, alpha=0.8)

        acc = clf.score(Xs, y)
        ax.set_title(f'{cname}\nAcc={acc*100:.0f}%', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        if col == 0:
            ax.set_ylabel(dname, fontsize=10, fontweight='bold')

plt.suptitle('Decision Boundaries: Linear vs Kernel Methods\n'
             '(Row = dataset type, Column = classifier)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('all_decision_boundaries.png', dpi=150, bbox_inches='tight')
print("Saved: all_decision_boundaries.png")
```

### Understanding the Boundary Shapes

```
Linear kernel:
  Boundary: wᵀx + b = 0
  Shape:    Straight line in 2D, flat plane in 3D
  Good for: Already separable data, text classification

Polynomial kernel (degree 2):
  Boundary: w₀ + w₁x₁ + w₂x₂ + w₃x₁² + w₄x₁x₂ + w₅x₂² = 0
  Shape:    Conic sections (ellipses, parabolas, hyperbolas)
  Good for: Moderate curvature, image features

Polynomial kernel (degree 3):
  Shape:    Cubic curves, more flexibility
  Good for: Higher curvature patterns

RBF kernel:
  Boundary: Arbitrary smooth curves
  Shape:    Can approximate ANY continuous boundary
  Good for: Most non-linear problems, default choice
```

### The gamma Effect on RBF Boundaries

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

np.random.seed(42)
X, y = make_moons(n_samples=150, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)

gammas = [0.01, 0.1, 1.0, 10.0, 100.0]

fig, axes = plt.subplots(1, len(gammas), figsize=(20, 4))

xx, yy = np.meshgrid(np.linspace(-3, 3, 200),
                     np.linspace(-3, 3, 200))

for ax, gamma in zip(axes, gammas):
    svm = SVC(kernel='rbf', gamma=gamma, C=1.0)
    svm.fit(X, y)

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='k', linewidths=1)
    ax.scatter(X[y==0,0], X[y==0,1], c='royalblue', s=25,
               edgecolors='k', linewidths=0.4)
    ax.scatter(X[y==1,0], X[y==1,1], c='tomato', s=25,
               marker='s', edgecolors='k', linewidths=0.4)

    acc = svm.score(X, y) * 100
    n_sv = svm.n_support_.sum()
    ax.set_title(f'γ = {gamma}\nAcc={acc:.0f}%, SV={n_sv}', fontsize=10)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel('Feature 2')
plt.suptitle('Effect of γ (gamma) on RBF Decision Boundary\n'
             'Small γ = smooth, Large γ = complex/overfit',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('gamma_effect.png', dpi=150, bbox_inches='tight')
print("Saved: gamma_effect.png")

print("\nGamma analysis:")
print("γ = 0.01  → Very smooth, possibly underfit")
print("γ = 0.1   → Smooth, good generalization")
print("γ = 1.0   → Balanced (often best)")
print("γ = 10.0  → Complex, risk of overfit")
print("γ = 100.0 → Each point gets its own island = overfit")
```

---

## Overview of Kernel-Based Algorithms

### 1. Support Vector Machine (SVM) — Classification

**Goal**: Find the hyperplane with **maximum margin** between classes.

```
               |← margin →|
  ○  ○  ○      |           |      ●  ●  ●
         ○     |   w·x+b=0 |    ●
          ○ ←→ support    ←→ ●
               vector
```

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# ALWAYS use Pipeline so scaling is part of the model
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', probability=True))
])

# Tune C (regularization) and gamma (kernel width)
param_grid = {
    'svm__C':     [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
}

grid_cv = GridSearchCV(svm_pipeline, param_grid,
                       cv=5, scoring='accuracy', n_jobs=-1)
grid_cv.fit(X_train, y_train)

print(f"Best params: {grid_cv.best_params_}")
print(f"Best CV acc: {grid_cv.best_score_*100:.2f}%")
print(f"Test acc:    {grid_cv.score(X_test, y_test)*100:.2f}%")

# Inspect support vectors
best_svm = grid_cv.best_estimator_.named_steps['svm']
print(f"Support vectors: {best_svm.n_support_}")
```

**Key hyperparameters**:

| Parameter | Effect | Range |
|-----------|--------|-------|
| `C` (large) | Smaller margin, fit data closely | Risk: overfit |
| `C` (small) | Larger margin, allow misclassifications | Risk: underfit |
| `gamma` (large) | Each point acts as its own "island" | Risk: overfit |
| `gamma` (small) | Smooth, global boundary | Risk: underfit |

### 2. Support Vector Regression (SVR)

**Goal**: Fit a function within an **epsilon-tube** around the data.

```
         Predictions inside this tube
         get ZERO loss — epsilon-insensitive!

   y
   │        ┌─────────────────┐ ← ε-tube upper bound
   │   ●    │ ●    ●    ●     │
   │  ●     │  function       │
   │  ●     │                 │
   │        └─────────────────┘ ← ε-tube lower bound
   └─────────────────────────── x
```

```python
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate non-linear regression data
np.random.seed(42)
X_reg = np.sort(np.random.uniform(0, 6, 100)).reshape(-1, 1)
y_reg = np.sin(X_reg).ravel() + np.random.randn(100) * 0.2

# SVR pipeline
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(kernel='rbf', C=10, gamma=0.5, epsilon=0.1))
])

svr_pipeline.fit(X_reg, y_reg)

# Predict
X_pred = np.linspace(0, 6, 300).reshape(-1, 1)
y_pred = svr_pipeline.predict(X_pred)

print(f"SVR R² on training data: {svr_pipeline.score(X_reg, y_reg):.4f}")
```

**When to use SVR vs Kernel Ridge Regression**:

| SVR | Kernel Ridge |
|-----|-------------|
| Epsilon-insensitive (robust to small errors) | L2 loss (punishes all errors) |
| Sparse solution (support vectors only) | Dense solution (all points) |
| More hyperparameters (C, ε) | Fewer hyperparameters (α) |
| Better with outliers | Better for smooth problems |

### 3. Kernel Ridge Regression

**Goal**: Ridge regression in the kernel feature space.

```
Traditional Ridge: min_w  ||y - Xw||² + λ||w||²
                   Solution: w = (XᵀX + λI)⁻¹Xᵀy

Kernel Ridge:      min_α  ||y - Kα||² + λαᵀKα
                   Solution: α = (K + λI)⁻¹y
                   Predict:   ŷ = K_test · α
```

```python
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline

kr = Pipeline([
    ('scaler', StandardScaler()),
    ('kr',     KernelRidge(kernel='rbf', alpha=0.1, gamma=0.5))
])

kr.fit(X_train, y_train)
print(f"Kernel Ridge R²: {kr.score(X_test, y_test):.4f}")
```

### 4. Kernel PCA — Non-linear Dimensionality Reduction

**Goal**: PCA in the kernel feature space — finds non-linear structure.

```
Standard PCA:  Finds straight principal components
Kernel PCA:    Finds CURVED principal components that follow
               the true manifold structure of the data
```

```python
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate Swiss roll-like data
from sklearn.datasets import make_swiss_roll
X_roll, color = make_swiss_roll(n_samples=500, noise=0.1, random_state=42)

# Standard PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_roll)

# Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.02)
X_kpca = kpca.fit_transform(X_roll)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 3D original
ax = axes[0]
ax.scatter(X_roll[:,0], X_roll[:,2], c=color, cmap='viridis', s=10)
ax.set_title('Original Data (2 of 3 dims shown)')
ax.grid(True, alpha=0.3)

# Standard PCA
ax = axes[1]
ax.scatter(X_pca[:,0], X_pca[:,1], c=color, cmap='viridis', s=10)
ax.set_title('Standard PCA (2D)\nMissed the manifold!')
ax.grid(True, alpha=0.3)

# Kernel PCA
ax = axes[2]
ax.scatter(X_kpca[:,0], X_kpca[:,1], c=color, cmap='viridis', s=10)
ax.set_title('Kernel PCA (2D)\nUnrolled the manifold!')
ax.grid(True, alpha=0.3)

plt.suptitle('Standard PCA vs Kernel PCA on Swiss Roll', fontsize=13)
plt.tight_layout()
plt.savefig('kpca_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: kpca_comparison.png")
```

### 5. One-Class SVM — Anomaly Detection

**Goal**: Learn a boundary around "normal" data; flag anything outside as an anomaly.

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

# Normal data (training)
np.random.seed(42)
X_normal = np.random.randn(200, 2)

# Anomalies
X_anomaly = np.random.uniform(-6, 6, (20, 2))

# Train One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
oc_svm.fit(StandardScaler().fit_transform(X_normal))

# Predict: +1 = normal, -1 = anomaly
scaler = StandardScaler().fit(X_normal)
pred_normal  = oc_svm.predict(scaler.transform(X_normal))
pred_anomaly = oc_svm.predict(scaler.transform(X_anomaly))

print(f"Normal data classified as normal: "
      f"{(pred_normal == 1).sum()}/{len(X_normal)}")
print(f"Anomalies detected: "
      f"{(pred_anomaly == -1).sum()}/{len(X_anomaly)}")
```

### Summary Table: Kernel Algorithms at a Glance

```python
# Print a formatted summary
algorithms = [
    ("SVM",               "Classification",       "Max margin hyperplane",     "C, gamma, kernel"),
    ("SVR",               "Regression",           "Epsilon-insensitive tube",  "C, epsilon, gamma"),
    ("Kernel Ridge",      "Regression",           "L2-regularized kernels",    "alpha, gamma"),
    ("Kernel PCA",        "Dim Reduction",        "Non-linear components",     "n_components, gamma"),
    ("One-Class SVM",     "Anomaly Detection",    "Boundary around normal",    "nu, gamma"),
    ("Spectral Cluster.", "Clustering",           "Graph Laplacian + k-means", "n_clusters, gamma"),
]

print("="*90)
print(f"{'Algorithm':<22} {'Task':<20} {'Idea':<32} {'Key Params':<20}")
print("="*90)
for row in algorithms:
    print(f"{row[0]:<22} {row[1]:<20} {row[2]:<32} {row[3]:<20}")
print("="*90)
```

**Output**:
```
══════════════════════════════════════════════════════════════════════════════════════════
Algorithm              Task                 Idea                             Key Params
══════════════════════════════════════════════════════════════════════════════════════════
SVM                    Classification       Max margin hyperplane             C, gamma, kernel
SVR                    Regression           Epsilon-insensitive tube          C, epsilon, gamma
Kernel Ridge           Regression           L2-regularized kernels            alpha, gamma
Kernel PCA             Dim Reduction        Non-linear components             n_components, gamma
One-Class SVM          Anomaly Detection    Boundary around normal            nu, gamma
Spectral Cluster.      Clustering           Graph Laplacian + k-means         n_clusters, gamma
══════════════════════════════════════════════════════════════════════════════════════════
```

---

## When to Use Kernel Methods

### Decision Framework

```python
def should_use_kernel(n_samples, n_features, linear_acc, need_interp):
    """
    Simple decision helper for kernel methods.
    """
    print("\n" + "="*55)
    print("  KERNEL METHODS: USE OR NOT?")
    print("="*55)

    score = 0

    # Factor 1: Dataset size
    if n_samples < 1_000:
        print("✓ Small dataset       → Kernels excellent")
        score += 2
    elif n_samples < 10_000:
        print("~ Medium dataset      → Kernels feasible")
        score += 1
    else:
        print("✗ Large dataset       → Kernels slow (O(n²-n³))")
        score -= 2

    # Factor 2: Linear performance
    if linear_acc < 0.70:
        print("✓ Low linear acc      → Kernels likely help")
        score += 2
    elif linear_acc < 0.90:
        print("~ OK linear acc       → Kernels may help")
        score += 1
    else:
        print("✗ High linear acc     → Kernels won't add much")
        score -= 1

    # Factor 3: Interpretability required
    if need_interp:
        print("✗ Need interpretability → Kernels = black box")
        score -= 2
    else:
        print("✓ No interpretability needed → Kernels fine")

    # Factor 4: High-dimensional
    if n_features > 100:
        print("✓ High-dimensional    → Linear kernel often great!")
        score += 1

    print("-"*55)
    if score >= 3:
        print("  → RECOMMENDATION: USE KERNEL METHODS ✅")
    elif score >= 1:
        print("  → RECOMMENDATION: TRY KERNELS, COMPARE ⚠️")
    else:
        print("  → RECOMMENDATION: SKIP KERNELS ❌")
    print("="*55)

# Try it
should_use_kernel(n_samples=500,   n_features=10,  linear_acc=0.55, need_interp=False)
should_use_kernel(n_samples=500000,n_features=20,  linear_acc=0.60, need_interp=False)
should_use_kernel(n_samples=1000,  n_features=1000,linear_acc=0.95, need_interp=True)
```

### Common Scenarios

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| Small dataset, non-linear patterns | ✅ RBF-SVM | Great generalization, handles non-linearity |
| Large dataset (>100K), non-linear | ❌ Use neural network | Kernels too slow |
| Text classification | ✅ Linear SVM | Already high-dim, often linearly separable |
| Image recognition (small dataset) | ✅ RBF-SVM or Poly-SVM | Good with pixel/histogram features |
| Genomic data (strings) | ✅ Spectrum/string kernel | Handles variable-length sequences |
| Medical diagnosis (small n) | ✅ SVM | Works well with few samples |
| Recommendation systems (millions) | ❌ Matrix factorization | Too large for kernels |

---

## Summary and Next Steps

### Core Concepts Recap

```
1. KERNEL FUNCTION
   K(x, x') = ⟨φ(x), φ(x')⟩
   → Computes similarity in high-dimensional space
   → Without computing the transformation φ explicitly

2. THE KERNEL TRICK
   Replace every dot product xᵀx' with K(x, x')
   → The algorithm now works in feature space for free!

3. MAIN KERNELS
   Linear:     K = xᵀx'             → straight boundaries
   Polynomial: K = (xᵀx' + c)ᵈ     → curved boundaries
   RBF:        K = exp(-γ||x-x'||²) → any shape (most popular)
   Sigmoid:    K = tanh(γxᵀx' + c)  → neural network-like

4. KEY ALGORITHMS
   SVM      → Classification (max margin)
   SVR      → Regression (epsilon-tube)
   KPCA     → Dimensionality reduction
   One-Class → Anomaly detection

5. GOLDEN RULES
   • Always scale features before kernel methods
   • Start with RBF kernel, tune C and gamma
   • Use GridSearchCV with cross-validation
   • Compare against linear baseline first
   • Check if dataset size is feasible
```

### Prerequisites Checklist

Before moving on, make sure you can:
- [ ] Explain the kernel trick in plain English
- [ ] Write the formula for at least 3 kernels from memory
- [ ] Explain why scaling is critical
- [ ] Describe when you would vs. would not use kernels
- [ ] Interpret the effect of C and gamma on SVM boundaries
- [ ] Name 4 kernel-based algorithms and their use cases

### What's Next in Section 06

1. **Kernel_Trick_Mathematical_Foundation.md**
   - Mercer's theorem (when is a function a valid kernel?)
   - Reproducing Kernel Hilbert Spaces (RKHS)
   - Positive semi-definite matrices

2. **Common_Kernel_Functions.md**
   - Every kernel in depth with visualizations
   - Hyperparameter sensitivity analysis
   - Kernel composition rules

3. **SVM_Hard_Margin.md**
   - Optimization problem formulation
   - Geometric interpretation
   - Lagrangian dual

4. **SVM_Soft_Margin.md**
   - Slack variables, C parameter
   - Hinge loss connection
   - Practical tuning

5. **Dual_Formulations.md** ← already exists
6. **Radial_Basis_Functions.md** ← already exists
7. **Kernel_Functions.md** ← already exists
8. **Support_Vector_Machines.md** ← already exists

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════╗
║              INTRODUCTION TO KERNEL METHODS                      ║
║                     Quick Reference                              ║
╠══════════════════════════════════════════════════════════════════╣
║  WHAT IS A KERNEL?                                               ║
║    K(x, x') = ⟨φ(x), φ(x')⟩                                     ║
║    = inner product in high-dim space, computed cheaply           ║
╠══════════════════════════════════════════════════════════════════╣
║  FOUR COMMON KERNELS                                             ║
║    Linear:      K = xᵀx'                  → line/plane           ║
║    Polynomial:  K = (xᵀx' + c)ᵈ           → conic sections       ║
║    RBF:         K = exp(-γ||x - x'||²)    → any smooth shape     ║
║    Sigmoid:     K = tanh(γxᵀx' + c)       → NN-like              ║
╠══════════════════════════════════════════════════════════════════╣
║  KEY ALGORITHMS                                                  ║
║    SVM (sklearn.svm.SVC)         → classification                ║
║    SVR (sklearn.svm.SVR)         → regression                    ║
║    KernelRidge                   → regression (L2)               ║
║    KernelPCA                     → dim reduction                 ║
║    OneClassSVM                   → anomaly detection             ║
╠══════════════════════════════════════════════════════════════════╣
║  USE KERNELS WHEN                  AVOID WHEN                    ║
║    ✓ < 10K samples                ✗ > 100K samples               ║
║    ✓ Linear model fails           ✗ Linear model works well      ║
║    ✓ High-dim features            ✗ Need interpretability        ║
║    ✓ Non-vector data (strings)    ✗ Real-time inference needed    ║
╠══════════════════════════════════════════════════════════════════╣
║  MINIMAL SKLEARN TEMPLATE                                        ║
║    from sklearn.svm import SVC                                   ║
║    from sklearn.preprocessing import StandardScaler              ║
║    from sklearn.pipeline import Pipeline                         ║
║    from sklearn.model_selection import GridSearchCV              ║
║                                                                  ║
║    pipe = Pipeline([                                             ║
║        ('scaler', StandardScaler()),                             ║
║        ('svm',    SVC(kernel='rbf'))                             ║
║    ])                                                            ║
║    grid = GridSearchCV(pipe,                                     ║
║        {'svm__C':[0.1,1,10], 'svm__gamma':['scale',0.1,1]},     ║
║        cv=5, scoring='accuracy')                                 ║
║    grid.fit(X_train, y_train)                                    ║
╠══════════════════════════════════════════════════════════════════╣
║  HYPERPARAMETER CHEAT SHEET (SVM RBF)                            ║
║    C small  → large margin, more misclassify → underfit          ║
║    C large  → small margin, few misclassify  → overfit           ║
║    γ small  → smooth, global boundary        → underfit          ║
║    γ large  → spiky, local boundary          → overfit           ║
║    Start: C=1, gamma='scale'. Then grid search.                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

*End of Introduction_to_Kernel_Methods.md*