# Dual Formulations

## Table of Contents
1. [Why Dual Formulations?](#why-dual-formulations)
2. [Primal Problem: Linear SVM](#primal-problem-linear-svm)
3. [Lagrangian Mechanics](#lagrangian-mechanics)
4. [Deriving the Dual Problem](#deriving-the-dual-problem)
5. [KKT Conditions](#kkt-conditions)
6. [Applying the Kernel Trick to the Dual](#applying-the-kernel-trick-to-the-dual)
7. [Dual for Soft-Margin SVM](#dual-for-soft-margin-svm)
8. [Solving the Dual: Quadratic Programming](#solving-the-dual-quadratic-programming)
9. [From Scratch: Dual SVM Implementation](#from-scratch-dual-svm-implementation)
10. [Summary and Quick Reference](#summary-and-quick-reference)

---

## Why Dual Formulations?

### The Core Motivation

The SVM can be solved in two equivalent ways:

```
PRIMAL FORM                    DUAL FORM
─────────────────────────────────────────────────────
Variables:  w, b               Variables:  α₁,...,αₙ
Dimension:  d (feature dim)    Dimension:  n (samples)
Kernel:     Not directly        Kernel:     YES — plug in K(x,x')!
Support     Not explicit        Explicit — αᵢ > 0 = support vector
Vectors:
```

**The dual unlocks three things:**

```
1. KERNEL TRICK
   Primal depends on w (in feature space = possibly infinite dim)
   Dual depends only on K(xᵢ,xⱼ) = inner products
   → Swap xᵢᵀxⱼ for K(xᵢ,xⱼ) and the algorithm works in ∞-dim space!

2. SPARSITY
   Most αᵢ = 0  (non-support vectors)
   Prediction uses only support vectors → memory efficient

3. THEORETICAL INSIGHT
   Support vectors = the only points that define the boundary
   Everything else is irrelevant once training is done
```

### Primal vs Dual: When Each Is Better

```
Use PRIMAL when:
  - d << n  (many features, few samples)
  - Linear kernel only
  - sklearn: LinearSVC (primal solver)

Use DUAL when:
  - n << d  (few samples, many/infinite features)
  - Any kernel (RBF, poly, etc.)
  - sklearn: SVC (dual solver by default)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_classification
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# DEMONSTRATE: Primal and Dual give identical results
# ============================================================

X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, class_sep=2.0, random_state=42
)
y_pm = np.where(y == 0, -1, 1)   # labels in {-1, +1}

sc = StandardScaler()
X_s = sc.fit_transform(X)

# Primal solver
primal = LinearSVC(C=1.0, max_iter=10000, loss='hinge')
primal.fit(X_s, y_pm)

# Dual solver (SVC with linear kernel)
dual = SVC(kernel='linear', C=1.0)
dual.fit(X_s, y_pm)

print("=" * 60)
print("PRIMAL vs DUAL: Identical Results")
print("=" * 60)
print(f"\nPrimal (LinearSVC) accuracy:  {primal.score(X_s, y_pm)*100:.1f}%")
print(f"Dual   (SVC linear) accuracy: {dual.score(X_s, y_pm)*100:.1f}%")

# Compare weight vectors
w_primal = primal.coef_[0]
w_dual   = dual.coef_[0]      # SVC reconstructs w = Σ αᵢ yᵢ xᵢ

print(f"\nPrimal w:  {w_primal.round(4)}")
print(f"Dual   w:  {w_dual.round(4)}")
print(f"Cosine similarity: "
      f"{np.dot(w_primal, w_dual) / (np.linalg.norm(w_primal) * np.linalg.norm(w_dual)):.6f}")
print(f"\n→ Both find the same hyperplane (cos≈1 means same direction)")
```

---

## Primal Problem: Linear SVM

### Hard-Margin Primal (Linearly Separable Data)

Find the weight vector **w** and bias **b** that:

```
Minimize:    ½ ||w||²                      ← maximize margin = 1/||w||

Subject to:  yᵢ (wᵀxᵢ + b) ≥ 1    ∀i    ← correct classification
                                            with margin
```

**Geometric meaning:**

```
         Class -1                 Class +1
                                  
    ●  ●  ●          |          ○  ○  ○
          ●     ←margin→    ○
           ● ←──┤ wᵀx+b=0 ├──→ ○
                │           │
           wᵀx+b=-1   wᵀx+b=+1
                ↑           ↑
           support      support
           vectors      vectors

    Margin width = 2 / ||w||
    Maximizing margin ↔ Minimizing ||w||²
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ============================================================
# VISUALIZE THE PRIMAL OBJECTIVE
# ============================================================

def plot_svm_geometry(X, y, clf, ax, title=""):
    """Show SVM margin, decision boundary, support vectors."""
    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5

    xx = np.linspace(x_min, x_max, 200)

    # Decision boundary: wᵀx + b = 0
    w = clf.coef_[0]
    b = clf.intercept_[0]

    def get_y_at_x(x_vals, level=0):
        return -(w[0] * x_vals + b - level) / w[1]

    y_boundary   = get_y_at_x(xx, 0)
    y_margin_pos = get_y_at_x(xx, +1)
    y_margin_neg = get_y_at_x(xx, -1)

    # Plot regions
    ax.fill_between(xx, y_margin_pos, y_max, alpha=0.08, color='tomato')
    ax.fill_between(xx, y_min, y_margin_neg, alpha=0.08, color='royalblue')

    # Plot lines
    ax.plot(xx, y_boundary,   'k-',  linewidth=2.5, label='Decision boundary')
    ax.plot(xx, y_margin_pos, 'r--', linewidth=1.5, label='Margin +1')
    ax.plot(xx, y_margin_neg, 'b--', linewidth=1.5, label='Margin -1')

    # Data points
    ax.scatter(X[y==-1, 0], X[y==-1, 1], c='royalblue', s=50,
               edgecolors='k', linewidths=0.5, zorder=3)
    ax.scatter(X[y== 1, 0], X[y== 1, 1], c='tomato', s=50,
               marker='s', edgecolors='k', linewidths=0.5, zorder=3)

    # Support vectors
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
               edgecolors='gold', linewidths=2.5,
               zorder=4, label=f'Support Vectors ({len(sv)})')

    # Margin width annotation
    margin_width = 2.0 / np.linalg.norm(w)
    ax.set_title(f"{title}\n"
                 f"||w|| = {np.linalg.norm(w):.3f}, "
                 f"Margin = {margin_width:.3f}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# Generate separable data
np.random.seed(42)
X_sep, y_sep = make_classification(
    n_samples=60, n_features=2, n_redundant=0,
    n_informative=2, class_sep=2.5, random_state=42
)
y_pm = np.where(y_sep == 0, -1, 1)
sc   = StandardScaler()
X_s  = sc.fit_transform(X_sep)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, C_val in zip(axes, [0.1, 1.0, 100.0]):
    clf = SVC(kernel='linear', C=C_val)
    clf.fit(X_s, y_pm)
    plot_svm_geometry(X_s, y_pm, clf, ax,
                      title=f"Hard-Margin SVM\nC = {C_val}")

plt.suptitle("SVM Primal: Margin Geometry for Different C",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('svm_primal_geometry.png', dpi=150, bbox_inches='tight')
print("Saved: svm_primal_geometry.png")

# Print primal quantities
clf_demo = SVC(kernel='linear', C=1.0)
clf_demo.fit(X_s, y_pm)
w = clf_demo.coef_[0]
b = clf_demo.intercept_[0]

print("\n" + "="*60)
print("PRIMAL QUANTITIES")
print("="*60)
print(f"w (weight vector):     {w.round(4)}")
print(f"b (bias):              {b:.4f}")
print(f"||w|| (norm):          {np.linalg.norm(w):.4f}")
print(f"Margin = 2/||w||:      {2/np.linalg.norm(w):.4f}")
print(f"Support vectors:       {len(clf_demo.support_vectors_)}")

# Verify margin constraints: yᵢ(wᵀxᵢ + b) ≥ 1
margins = y_pm * (X_s @ w + b)
print(f"\nMargin values for each point:")
print(f"  Min margin: {margins.min():.4f}  (support vectors have margin ≈ 1)")
print(f"  Mean margin: {margins.mean():.4f}")
print(f"  Points with margin ≈ 1: "
      f"{np.sum(np.isclose(margins, 1.0, atol=0.01))}")
```

---

## Lagrangian Mechanics

### Why Lagrangians?

Constrained optimization problems can be converted to unconstrained ones using **Lagrange multipliers**:

```
Constrained problem:
  Minimize:    f(w)
  Subject to:  gᵢ(w) ≤ 0   ∀i

Lagrangian:
  L(w, α) = f(w) + Σᵢ αᵢ gᵢ(w)
  
  where αᵢ ≥ 0 are Lagrange multipliers

Saddle point condition:
  min_w max_α L(w, α) = max_α min_w L(w, α)   ← strong duality
```

### Building the SVM Lagrangian

**SVM constraints** in standard form:

```
gᵢ(w, b) = 1 - yᵢ(wᵀxᵢ + b) ≤ 0    ∀i = 1,...,n
```

**Lagrangian** (introduce multipliers αᵢ ≥ 0):

```
L(w, b, α) = ½||w||² - Σᵢ αᵢ [yᵢ(wᵀxᵢ + b) - 1]

           = ½||w||² - Σᵢ αᵢ yᵢ(wᵀxᵢ + b) + Σᵢ αᵢ
```

```python
import numpy as np

# ============================================================
# LAGRANGIAN: NUMERICAL ILLUSTRATION
# ============================================================

print("="*60)
print("LAGRANGIAN ILLUSTRATION")
print("="*60)

def lagrangian(w, b, alpha, X, y):
    """
    L(w, b, α) = ½||w||² - Σᵢ αᵢ[yᵢ(wᵀxᵢ + b) - 1]

    Computes the SVM Lagrangian for given w, b, α.
    """
    primal_obj  = 0.5 * np.dot(w, w)
    constraints = y * (X @ w + b) - 1          # should be ≥ 0
    penalty     = np.sum(alpha * constraints)   # subtract: we MAXIMIZE over α
    return primal_obj - penalty

# Small example: 4 points
X_tiny = np.array([
    [1.0,  1.0],   # Class +1
    [2.0,  2.0],   # Class +1
    [-1.0, -1.0],  # Class -1
    [-2.0, -1.0],  # Class -1
])
y_tiny = np.array([1.0, 1.0, -1.0, -1.0])

# Known solution for this toy data
w_opt   = np.array([0.5, 0.5])
b_opt   = 0.0
alpha_opt = np.array([0.5, 0.0, 0.5, 0.0])  # αᵢ > 0 at support vectors

print(f"\nToy example: {len(X_tiny)} points")
print(f"Optimal w:       {w_opt}")
print(f"Optimal b:       {b_opt}")
print(f"Optimal alphas:  {alpha_opt}")

# Compute Lagrangian at optimum
L_opt = lagrangian(w_opt, b_opt, alpha_opt, X_tiny, y_tiny)
print(f"\nLagrangian at optimum: {L_opt:.4f}")
print(f"Primal objective ½||w||²: {0.5*np.dot(w_opt, w_opt):.4f}")

# Check KKT: αᵢ[yᵢ(wᵀxᵢ + b) - 1] = 0
print("\nKKT check αᵢ·(margin - 1) = 0:")
for i, (xi, yi, ai) in enumerate(zip(X_tiny, y_tiny, alpha_opt)):
    margin  = yi * (np.dot(w_opt, xi) + b_opt)
    product = ai * (margin - 1)
    sv      = " ← support vector" if ai > 0 else ""
    print(f"  i={i}: αᵢ={ai}, margin={margin:.3f}, αᵢ(m-1)={product:.6f}{sv}")
```

---

## Deriving the Dual Problem

### Step 1: Stationarity Conditions

At the saddle point, derivatives of L w.r.t. primal variables = 0:

```
∂L/∂w = 0:
  w - Σᵢ αᵢ yᵢ xᵢ = 0
  → w = Σᵢ αᵢ yᵢ xᵢ        ← w is a linear combination of training points!

∂L/∂b = 0:
  -Σᵢ αᵢ yᵢ = 0
  → Σᵢ αᵢ yᵢ = 0            ← equality constraint on αᵢ
```

### Step 2: Substitute Back into Lagrangian

Substitute **w = Σᵢ αᵢ yᵢ xᵢ** into L(w, b, α):

```
L(w, b, α) = ½||w||² - Σᵢ αᵢ yᵢ wᵀxᵢ - b Σᵢ αᵢ yᵢ + Σᵢ αᵢ

Step A: ½||w||² = ½ (Σᵢ αᵢ yᵢ xᵢ)ᵀ (Σⱼ αⱼ yⱼ xⱼ)
                = ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢᵀxⱼ)

Step B: Σᵢ αᵢ yᵢ wᵀxᵢ = Σᵢ αᵢ yᵢ (Σⱼ αⱼ yⱼ xⱼ)ᵀ xᵢ
                       = Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢᵀxⱼ)

Step C: b Σᵢ αᵢ yᵢ = b · 0 = 0    (from stationarity)

Combining:
  L_dual(α) = Σᵢ αᵢ - ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢᵀxⱼ)
```

### Step 3: The Dual Problem

```
DUAL PROBLEM (Hard-Margin SVM):

Maximize:    W(α) = Σᵢ αᵢ  -  ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢᵀxⱼ)

Subject to:  Σᵢ αᵢ yᵢ = 0
             αᵢ ≥ 0    ∀i

After solving → recover primal variables:
  w = Σᵢ αᵢ yᵢ xᵢ
  b = yₛ - wᵀxₛ    for any support vector s (where αₛ > 0)
```

**This is a Quadratic Program (QP)** — convex, guaranteed global optimum!

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DUAL OBJECTIVE: VISUALIZE FOR SIMPLE CASE
# ============================================================

print("="*60)
print("DUAL OBJECTIVE VISUALIZATION")
print("="*60)

# 2-point example: x₁ = +1 class, x₂ = -1 class
# Simplest possible SVM
x1 = np.array([1.0, 0.0])    # class +1
x2 = np.array([-1.0, 0.0])   # class -1
y1, y2 = 1.0, -1.0

# Dual: max W(α₁, α₂) = α₁ + α₂ - ½[(α₁y₁)²||x₁||² + 2α₁α₂y₁y₂x₁ᵀx₂ + (α₂y₂)²||x₂||²]
# Subject to: α₁y₁ + α₂y₂ = 0  → α₁ = α₂  (since y₁ = -y₂)
#             α₁, α₂ ≥ 0

def dual_objective_2pt(alpha1, alpha2):
    """Dual objective for 2-point example."""
    term1 = alpha1 + alpha2
    term2 = 0.5 * (
        alpha1**2 * y1**2 * np.dot(x1, x1) +
        2 * alpha1 * alpha2 * y1 * y2 * np.dot(x1, x2) +
        alpha2**2 * y2**2 * np.dot(x2, x2)
    )
    return term1 - term2

# Scan over α₁ (with constraint α₁ = α₂)
alphas = np.linspace(0, 2, 200)
W_vals = [dual_objective_2pt(a, a) for a in alphas]  # α₁ = α₂

a_opt_idx = np.argmax(W_vals)
a_opt     = alphas[a_opt_idx]

print(f"\n2-point example: x₁=[1,0] (class+1), x₂=[-1,0] (class-1)")
print(f"Optimal α* = {a_opt:.4f}")
print(f"Maximum W(α*) = {max(W_vals):.4f}")

# Recover w from dual
w_from_dual = a_opt * y1 * x1 + a_opt * y2 * x2
print(f"\nRecovered w = Σᵢ αᵢ yᵢ xᵢ = {w_from_dual}")
print(f"Margin = 2/||w|| = {2/np.linalg.norm(w_from_dual):.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(alphas, W_vals, 'b-', linewidth=2)
ax.axvline(x=a_opt, color='red', linestyle='--',
           label=f'Optimal α* = {a_opt:.3f}')
ax.scatter([a_opt], [max(W_vals)], c='red', s=100, zorder=5)
ax.set_xlabel('α (= α₁ = α₂ by constraint)')
ax.set_ylabel('W(α) — Dual Objective')
ax.set_title('Dual Objective (2-point example)\nMaximize W(α)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
# Show the geometry
ax.scatter([x1[0]], [x1[1]], c='tomato', s=200, marker='s',
           edgecolors='k', zorder=5, label='Class +1')
ax.scatter([x2[0]], [x2[1]], c='royalblue', s=200,
           edgecolors='k', zorder=5, label='Class -1')

# Decision boundary
ax.axvline(x=0, color='k', linewidth=2, label='Boundary (x₁=0)')
ax.axvline(x= 1/np.linalg.norm(w_from_dual),
           color='r', linestyle='--', alpha=0.6, label='Margin +1')
ax.axvline(x=-1/np.linalg.norm(w_from_dual),
           color='b', linestyle='--', alpha=0.6, label='Margin -1')
ax.set_xlim(-2, 2); ax.set_ylim(-1, 1)
ax.set_title('Geometric Picture\nMargin maximized')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dual_objective.png', dpi=150, bbox_inches='tight')
print("Saved: dual_objective.png")


# ============================================================
# FROM PRIMAL TO DUAL: STEP BY STEP NUMERICALLY
# ============================================================

print("\n" + "="*60)
print("PRIMAL TO DUAL: FULL DERIVATION NUMERICALLY")
print("="*60)

from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X_d, y_d = make_classification(
    n_samples=50, n_features=2, n_redundant=0,
    n_informative=2, class_sep=2.5, random_state=42
)
y_d = np.where(y_d == 0, -1, 1)
X_d = StandardScaler().fit_transform(X_d)

clf = SVC(kernel='linear', C=1e6)   # Hard margin approx with large C
clf.fit(X_d, y_d)

# Dual coefficients αᵢ yᵢ (sklearn stores αᵢ yᵢ in dual_coef_)
alpha_y  = clf.dual_coef_.ravel()        # shape (n_sv,)
sv_idx   = clf.support_                  # indices of support vectors
alpha_sv = np.abs(alpha_y)               # αᵢ (since it stores αᵢ·yᵢ)
y_sv     = clf.support_vectors_          # support vector features

print(f"\nNumber of training points: {len(X_d)}")
print(f"Number of support vectors: {len(sv_idx)}")
print(f"Support vector indices:    {sv_idx}")
print(f"\n|αᵢ| for support vectors:")
for i, (idx, a) in enumerate(zip(sv_idx, alpha_sv)):
    print(f"  i={idx}: |α| = {a:.4f}")

# Reconstruct w from dual: w = Σᵢ αᵢ yᵢ xᵢ
w_dual = np.sum(alpha_y[:, np.newaxis] * clf.support_vectors_, axis=0)
w_primal = clf.coef_[0]

print(f"\nw from dual:   {w_dual.round(4)}")
print(f"w from primal: {w_primal.round(4)}")
print(f"Difference:    {np.abs(w_dual - w_primal).max():.2e}")

# Verify dual objective value
K_sv = clf.support_vectors_ @ clf.support_vectors_.T
W_alpha = alpha_sv.sum() - 0.5 * (alpha_y @ K_sv @ alpha_y)
print(f"\nDual objective W(α*) = {W_alpha:.4f}")
print(f"Primal objective ½||w||² = {0.5*np.dot(w_primal, w_primal):.4f}")
print(f"Strong duality: |W(α*) - ½||w||²| = "
      f"{abs(W_alpha - 0.5*np.dot(w_primal, w_primal)):.2e}")
```

---

## KKT Conditions

### The Four KKT Conditions for SVM

The Karush-Kuhn-Tucker (KKT) conditions are **necessary and sufficient** for optimality (since the SVM is convex):

```
1. STATIONARITY:
   ∂L/∂w = 0  →  w = Σᵢ αᵢ yᵢ xᵢ
   ∂L/∂b = 0  →  Σᵢ αᵢ yᵢ = 0

2. PRIMAL FEASIBILITY:
   yᵢ(wᵀxᵢ + b) - 1 ≥ 0   ∀i
   (All points satisfy margin constraint)

3. DUAL FEASIBILITY:
   αᵢ ≥ 0   ∀i

4. COMPLEMENTARY SLACKNESS:
   αᵢ [yᵢ(wᵀxᵢ + b) - 1] = 0   ∀i

   Either:  αᵢ = 0  (point not a support vector, doesn't influence w)
   Or:      yᵢ(wᵀxᵢ + b) = 1  (point exactly on the margin = support vector)
```

**Complementary slackness is the key!**

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# ============================================================
# KKT CONDITIONS: VERIFY NUMERICALLY
# ============================================================

print("="*60)
print("KKT CONDITIONS VERIFICATION")
print("="*60)

np.random.seed(42)
X_kkt, y_kkt = make_classification(
    n_samples=80, n_features=2, n_redundant=0,
    n_informative=2, class_sep=2.5, random_state=42
)
y_kkt = np.where(y_kkt == 0, -1, 1).astype(float)
X_kkt = StandardScaler().fit_transform(X_kkt)

clf = SVC(kernel='linear', C=1e6)   # Hard margin
clf.fit(X_kkt, y_kkt)

w = clf.coef_[0]
b = clf.intercept_[0]
sv_idx = clf.support_

# Get alphas (stored as αᵢ yᵢ in sklearn)
alpha_y = clf.dual_coef_.ravel()    # αᵢ·yᵢ for support vectors
alpha   = np.abs(alpha_y)           # αᵢ (always positive)

print(f"\nw = {w.round(4)}")
print(f"b = {b:.4f}")

# Condition 1: Stationarity — w = Σᵢ αᵢ yᵢ xᵢ
w_reconstructed = np.sum(
    alpha_y[:, np.newaxis] * clf.support_vectors_,
    axis=0
)
print(f"\n--- Condition 1: Stationarity ---")
print(f"w:               {w.round(6)}")
print(f"Σᵢ αᵢ yᵢ xᵢ:   {w_reconstructed.round(6)}")
print(f"Match:           {np.allclose(w, w_reconstructed, atol=1e-4)}")

# Condition 2: Primal Feasibility — yᵢ(wᵀxᵢ + b) ≥ 1
margins  = y_kkt * (X_kkt @ w + b)
feasible = np.all(margins >= 1 - 1e-4)
print(f"\n--- Condition 2: Primal Feasibility ---")
print(f"Min margin: {margins.min():.4f}  (should be ≥ 1)")
print(f"Feasible:   {feasible}")

# Condition 3: Dual Feasibility — αᵢ ≥ 0
print(f"\n--- Condition 3: Dual Feasibility ---")
print(f"All αᵢ ≥ 0:  {np.all(alpha >= -1e-8)}")
print(f"Min α:       {alpha.min():.6f}")

# Condition 4: Complementary Slackness
print(f"\n--- Condition 4: Complementary Slackness ---")
print(f"{'Point i':<10} {'αᵢ':<10} {'margin':<12} {'αᵢ(m-1)':<15} {'SV?'}")
print("-"*55)

for i, (idx, ai) in enumerate(zip(sv_idx, alpha)):
    margin  = margins[idx]
    cs      = ai * (margin - 1)
    is_sv   = "✓ SV" if ai > 1e-4 else ""
    print(f"{idx:<10} {ai:<10.4f} {margin:<12.4f} {cs:<15.6f} {is_sv}")

print(f"\nAll complementary slackness ≈ 0: "
      f"{np.allclose([a * (margins[idx] - 1) for a, idx in zip(alpha, sv_idx)], 0, atol=1e-3)}")

# Condition 1b: Σᵢ αᵢ yᵢ = 0
sum_alpha_y = np.sum(alpha_y)
print(f"\n--- Condition 1b: Σᵢ αᵢ yᵢ = 0 ---")
print(f"Σᵢ αᵢ yᵢ = {sum_alpha_y:.6f}  (should ≈ 0)")
```

### KKT Conditions: What They Tell Us

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# VISUALIZE KKT GEOMETRY
# ============================================================

np.random.seed(42)
X_vis, y_vis = make_classification(
    n_samples=30, n_features=2, n_redundant=0,
    n_informative=2, class_sep=2.0, random_state=42
)
y_vis = np.where(y_vis == 0, -1, 1)
X_vis = StandardScaler().fit_transform(X_vis)

clf_vis = SVC(kernel='linear', C=1e6)
clf_vis.fit(X_vis, y_vis)

w_vis   = clf_vis.coef_[0]
b_vis   = clf_vis.intercept_[0]
sv_idx_vis = clf_vis.support_
margins_vis = y_vis * (X_vis @ w_vis + b_vis)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Geometry
ax = axes[0]
xx = np.linspace(X_vis[:, 0].min()-0.5, X_vis[:, 0].max()+0.5, 200)

ax.plot(xx, -(w_vis[0]*xx + b_vis)/w_vis[1], 'k-', lw=2,
        label='Decision boundary')
ax.plot(xx, -(w_vis[0]*xx + b_vis - 1)/w_vis[1], 'r--', lw=1.5,
        label='Margin +1 (αᵢ>0 here)')
ax.plot(xx, -(w_vis[0]*xx + b_vis + 1)/w_vis[1], 'b--', lw=1.5,
        label='Margin -1 (αᵢ>0 here)')

# Color points by KKT role
for i, (xi, yi, mi) in enumerate(zip(X_vis, y_vis, margins_vis)):
    color  = 'royalblue' if yi == -1 else 'tomato'
    marker = 'o' if yi == -1 else 's'
    is_sv  = i in sv_idx_vis

    ax.scatter(xi[0], xi[1], c=color, marker=marker, s=60,
               edgecolors='k', linewidths=0.5, zorder=3)

    if is_sv:
        ax.scatter(xi[0], xi[1], s=200, facecolors='none',
                   edgecolors='gold', linewidths=2.5, zorder=4)
        ax.annotate(f'α>0\nm={mi:.2f}',
                    xy=(xi[0], xi[1]),
                    xytext=(xi[0]+0.15, xi[1]+0.15),
                    fontsize=6, color='darkgreen')

ax.set_title('KKT Geometry\nGold circles = support vectors (αᵢ > 0)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Right: Margin histogram
ax = axes[1]
sv_margins  = margins_vis[sv_idx_vis]
nsv_margins = np.delete(margins_vis, sv_idx_vis)

ax.hist(nsv_margins, bins=12, color='steelblue', alpha=0.7,
        label=f'Non-SV (αᵢ=0, n={len(nsv_margins)})', edgecolor='k')
ax.hist(sv_margins, bins=5, color='gold', alpha=0.8,
        label=f'Support vectors (αᵢ>0, n={len(sv_margins)})', edgecolor='k')

ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
           label='Margin = 1 (boundary)')
ax.set_xlabel('Functional Margin yᵢ(wᵀxᵢ + b)')
ax.set_ylabel('Count')
ax.set_title('Margin Distribution\nKKT: SV must have margin = 1')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle("KKT Conditions Visualized", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('kkt_conditions.png', dpi=150, bbox_inches='tight')
print("Saved: kkt_conditions.png")
```

---

## Applying the Kernel Trick to the Dual

### The Key Substitution

The dual problem contains data **only as dot products** xᵢᵀxⱼ:

```
Dual objective:
  W(α) = Σᵢ αᵢ  -  ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢᵀxⱼ)
                                            ↑
                              Replace with K(xᵢ, xⱼ)!

Kernelized dual:
  W(α) = Σᵢ αᵢ  -  ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)

Constraints (unchanged):
  Σᵢ αᵢ yᵢ = 0
  αᵢ ≥ 0   ∀i

Kernelized decision function:
  f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b
```

**That's it!** The primal variable **w** never appears again.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles

# ============================================================
# KERNEL TRICK IN DUAL: SIDE BY SIDE
# ============================================================

print("="*60)
print("KERNEL TRICK APPLIED TO DUAL")
print("="*60)

np.random.seed(42)
X_k, y_k = make_circles(n_samples=150, noise=0.1,
                         factor=0.4, random_state=42)
y_k = np.where(y_k == 0, -1, 1)
X_k = StandardScaler().fit_transform(X_k)

print("\n--- Linear Kernel (fails on circles) ---")
clf_lin = SVC(kernel='linear', C=1.0)
clf_lin.fit(X_k, y_k)
print(f"  Dual: uses K(x,x') = xᵀx'")
print(f"  Decision: f(x) = Σᵢ αᵢ yᵢ (xᵢᵀx) + b")
print(f"  Accuracy: {clf_lin.score(X_k, y_k)*100:.0f}%")
print(f"  Support vectors: {clf_lin.n_support_}")

print("\n--- RBF Kernel (succeeds on circles) ---")
clf_rbf = SVC(kernel='rbf', gamma=1.0, C=1.0)
clf_rbf.fit(X_k, y_k)
print(f"  Dual: uses K(x,x') = exp(-γ||x-x'||²)")
print(f"  Decision: f(x) = Σᵢ αᵢ yᵢ exp(-γ||xᵢ-x||²) + b")
print(f"  Accuracy: {clf_rbf.score(X_k, y_k)*100:.0f}%")
print(f"  Support vectors: {clf_rbf.n_support_}")

print("\n--- Poly Kernel (degree=2) ---")
clf_poly = SVC(kernel='poly', degree=2, coef0=1, C=1.0)
clf_poly.fit(X_k, y_k)
print(f"  Dual: uses K(x,x') = (xᵀx' + 1)²")
print(f"  Decision: f(x) = Σᵢ αᵢ yᵢ (xᵢᵀx + 1)² + b")
print(f"  Accuracy: {clf_poly.score(X_k, y_k)*100:.0f}%")
print(f"  Support vectors: {clf_poly.n_support_}")


# ============================================================
# PREDICTION: HOW THE DUAL DECISION FUNCTION WORKS
# ============================================================

print("\n\n--- Manual Dual Prediction (from scratch) ---")

clf_rbf2 = SVC(kernel='rbf', gamma=1.0, C=1.0)
clf_rbf2.fit(X_k, y_k)

# Get SVM components
alpha_y_rbf = clf_rbf2.dual_coef_.ravel()  # αᵢ yᵢ
sv_rbf      = clf_rbf2.support_vectors_
b_rbf       = clf_rbf2.intercept_[0]

def rbf(x, xp, gamma=1.0):
    return np.exp(-gamma * np.sum((x - xp)**2))

def manual_decision(x_test, sv, alpha_y, b, gamma=1.0):
    """
    f(x) = Σᵢ (αᵢ yᵢ) K(xᵢ, x) + b
    """
    kernel_vals = np.array([rbf(x_test, sv[i], gamma)
                             for i in range(len(sv))])
    return np.dot(alpha_y, kernel_vals) + b

# Predict a few points
X_new = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])

print(f"{'Point':<25} {'Manual f(x)':<15} {'Sklearn f(x)':<15} {'Class'}")
print("-"*60)

sk_decisions = clf_rbf2.decision_function(X_new)
for pt, manual, sk_d in zip(X_new, 
                              [manual_decision(pt, sv_rbf, alpha_y_rbf, b_rbf)
                               for pt in X_new],
                              sk_decisions):
    pred = "+1" if manual > 0 else "-1"
    print(f"{str(pt):<25} {manual:<15.4f} {sk_d:<15.4f} {pred}")


# ============================================================
# VISUALIZATION: Dual decision function landscape
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

kernels = [
    ('linear', dict(C=1.0), "Linear Kernel\nf(x)=Σαᵢyᵢ(xᵢᵀx)+b"),
    ('rbf',    dict(C=1.0, gamma=1.0), "RBF Kernel\nf(x)=Σαᵢyᵢexp(-γ||xᵢ-x||²)+b"),
    ('poly',   dict(C=1.0, degree=2, coef0=1), "Poly d=2 Kernel\nf(x)=Σαᵢyᵢ(xᵢᵀx+1)²+b"),
]

for ax, (kernel, params, title) in zip(axes, kernels):
    clf = SVC(kernel=kernel, **params)
    clf.fit(X_k, y_k)

    xx, yy = np.meshgrid(np.linspace(-2, 2, 200),
                         np.linspace(-2, 2, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    D = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlBu')
    ax.contour(xx, yy, D, levels=[-1, 0, 1],
               colors=['blue', 'black', 'red'],
               linewidths=[1, 2, 1],
               linestyles=['--', '-', '--'])

    ax.scatter(X_k[y_k==-1, 0], X_k[y_k==-1, 1],
               c='royalblue', s=25, edgecolors='k', linewidths=0.4)
    ax.scatter(X_k[y_k== 1, 0], X_k[y_k== 1, 1],
               c='tomato', s=25, marker='s', edgecolors='k', linewidths=0.4)

    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors='none',
               edgecolors='gold', linewidths=2, zorder=5)

    acc = clf.score(X_k, y_k)
    ax.set_title(f"{title}\nAcc={acc*100:.0f}%, SV={len(sv)}")
    ax.grid(True, alpha=0.3)

plt.suptitle("Dual Decision Functions with Different Kernels\n"
             "(dashed lines = margins, gold circles = support vectors)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('kernel_dual_decisions.png', dpi=150, bbox_inches='tight')
print("\nSaved: kernel_dual_decisions.png")
```

---

## Dual for Soft-Margin SVM

### Adding Slack Variables

For non-separable data, we add slack variables ξᵢ ≥ 0:

```
SOFT-MARGIN PRIMAL:

Minimize:    ½||w||² + C Σᵢ ξᵢ

Subject to:  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ    ∀i   ← relaxed margin
             ξᵢ ≥ 0                      ∀i
```

### Soft-Margin Dual Derivation

Introduce Lagrange multipliers αᵢ ≥ 0 and μᵢ ≥ 0:

```
L(w, b, ξ, α, μ) = ½||w||² + C Σᵢ ξᵢ
                   - Σᵢ αᵢ [yᵢ(wᵀxᵢ + b) - 1 + ξᵢ]
                   - Σᵢ μᵢ ξᵢ

Stationarity conditions:
  ∂L/∂w  = 0  →  w = Σᵢ αᵢ yᵢ xᵢ         (same as hard margin!)
  ∂L/∂b  = 0  →  Σᵢ αᵢ yᵢ = 0             (same!)
  ∂L/∂ξᵢ = 0  →  C - αᵢ - μᵢ = 0
                →  αᵢ = C - μᵢ
                →  αᵢ ≤ C  (since μᵢ ≥ 0)

SOFT-MARGIN DUAL:

Maximize:    W(α) = Σᵢ αᵢ - ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ,xⱼ)

Subject to:  Σᵢ αᵢ yᵢ = 0
             0 ≤ αᵢ ≤ C    ∀i          ← Box constraint (new!)
```

**Only difference from hard-margin**: αᵢ bounded above by C!

### KKT Conditions for Soft-Margin

```
Complementary slackness has two cases:

αᵢ(yᵢ(wᵀxᵢ + b) - 1 + ξᵢ) = 0   and   μᵢ ξᵢ = 0

This gives THREE types of points:

1. αᵢ = 0:
   μᵢ = C, ξᵢ = 0
   Point correctly classified, outside margin
   → Doesn't influence decision boundary at all

2. 0 < αᵢ < C:
   μᵢ > 0, ξᵢ = 0
   Point exactly on the margin boundary
   → True support vector (margin = 1 exactly)

3. αᵢ = C:
   μᵢ = 0, ξᵢ > 0
   Point inside margin or misclassified
   → "Bounded" support vector (margin violation)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# ============================================================
# SOFT MARGIN: THREE TYPES OF POINTS
# ============================================================

print("="*60)
print("SOFT-MARGIN SVM: THREE TYPES OF POINTS")
print("="*60)

np.random.seed(42)
X_soft, y_soft = make_classification(
    n_samples=100, n_features=2, n_redundant=0,
    n_informative=2, class_sep=1.0, random_state=42  # class_sep=1 → overlapping
)
y_soft = np.where(y_soft == 0, -1, 1).astype(float)
X_soft = StandardScaler().fit_transform(X_soft)

C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

print(f"\n{'C':<10} {'#SV (0<α<C)':<15} {'#Bounded (α=C)':<18} "
      f"{'#Non-SV (α=0)':<18} {'Accuracy'}")
print("-"*70)

for C in C_values:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_soft, y_soft)

    w = clf.coef_[0]
    b = clf.intercept_[0]

    alpha_y = clf.dual_coef_.ravel()
    alpha   = np.abs(alpha_y)

    n_free    = np.sum((alpha > 1e-5) & (alpha < C - 1e-5))
    n_bounded = np.sum(alpha >= C - 1e-5)
    n_nonsv   = len(X_soft) - len(clf.support_)
    acc       = clf.score(X_soft, y_soft) * 100

    print(f"{C:<10} {n_free:<15} {n_bounded:<18} {n_nonsv:<18} {acc:.1f}%")


# ============================================================
# VISUALIZE: Effect of C on soft-margin boundary
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.ravel()

for ax, C in zip(axes_flat, C_values):
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_soft, y_soft)

    w = clf.coef_[0]
    b = clf.intercept_[0]

    xx = np.linspace(X_soft[:, 0].min()-0.5,
                     X_soft[:, 0].max()+0.5, 200)

    ax.plot(xx, -(w[0]*xx + b    )/w[1], 'k-',  lw=2)
    ax.plot(xx, -(w[0]*xx + b - 1)/w[1], 'r--', lw=1.5)
    ax.plot(xx, -(w[0]*xx + b + 1)/w[1], 'b--', lw=1.5)

    margins = y_soft * (X_soft @ w + b)
    alpha   = np.abs(clf.dual_coef_.ravel())

    for i, (xi, yi, mi) in enumerate(zip(X_soft, y_soft, margins)):
        color  = 'royalblue' if yi == -1 else 'tomato'
        marker = 'o' if yi == -1 else 's'

        if i in clf.support_:
            ai = alpha[np.where(clf.support_ == i)[0][0]] \
                 if i in clf.support_ else 0

            if ai >= C - 1e-5:  # Bounded SV
                ax.scatter(xi[0], xi[1], c=color, s=80, marker=marker,
                           edgecolors='purple', linewidths=2, zorder=4)
            else:               # Free SV
                ax.scatter(xi[0], xi[1], c=color, s=80, marker=marker,
                           edgecolors='gold', linewidths=2, zorder=4)
        else:                   # Non-SV
            ax.scatter(xi[0], xi[1], c=color, s=25, marker=marker,
                       alpha=0.5, zorder=3)

    n_sv  = len(clf.support_)
    acc   = clf.score(X_soft, y_soft) * 100
    marg  = 2 / np.linalg.norm(w)

    ax.set_title(f"C = {C}\nSV={n_sv}, Margin={marg:.2f}, Acc={acc:.0f}%\n"
                 "Gold=free SV, Purple=bounded SV")
    ax.grid(True, alpha=0.3)

# Last panel: summary
ax = axes_flat[-1]
cs_list  = C_values
accs_c   = []
margins_c = []
n_svs_c  = []

for C in cs_list:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_soft, y_soft)
    w = clf.coef_[0]
    accs_c.append(clf.score(X_soft, y_soft) * 100)
    margins_c.append(2 / np.linalg.norm(w))
    n_svs_c.append(len(clf.support_))

ax2 = ax.twinx()
ax.plot(range(len(cs_list)), accs_c, 'bo-', lw=2, label='Accuracy (%)')
ax2.plot(range(len(cs_list)), margins_c, 'rs-', lw=2, label='Margin width')
ax.set_xticks(range(len(cs_list)))
ax.set_xticklabels([str(c) for c in cs_list])
ax.set_xlabel('C value')
ax.set_ylabel('Accuracy (%)', color='blue')
ax2.set_ylabel('Margin width', color='red')
ax.set_title('Accuracy vs Margin\nas C increases')
ax.grid(True, alpha=0.3)

plt.suptitle("Soft-Margin SVM: Effect of C\n"
             "Small C → Wide margin (underfitting)  "
             "Large C → Narrow margin (overfitting)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('soft_margin_dual.png', dpi=150, bbox_inches='tight')
print("\nSaved: soft_margin_dual.png")
```

---

## Solving the Dual: Quadratic Programming

### The QP Problem Structure

```
DUAL SVM as Standard QP:

Minimize:    ½ αᵀ Q α - 1ᵀ α

where Q[i,j] = yᵢ yⱼ K(xᵢ, xⱼ)   (positive semi-definite!)

Subject to:  yᵀ α = 0              (equality constraint)
             0 ≤ α ≤ C             (box constraints)

This is a CONVEX QP → guaranteed global optimum
```

```python
import numpy as np
from scipy.optimize import minimize

# ============================================================
# SOLVE DUAL WITH SCIPY (SMALL EXAMPLE)
# ============================================================

print("="*60)
print("SOLVING THE DUAL QP")
print("="*60)

np.random.seed(42)
X_qp, y_qp = make_classification(
    n_samples=30, n_features=2, n_redundant=0,
    n_informative=2, class_sep=2.5, random_state=42
)
y_qp = np.where(y_qp == 0, -1, 1).astype(float)
X_qp = StandardScaler().fit_transform(X_qp)

n = len(X_qp)
C = 1.0

# Build Q matrix: Q[i,j] = yᵢ yⱼ K(xᵢ,xⱼ)
def rbf_kernel_matrix(X, gamma=0.5):
    sq   = np.sum(X**2, axis=1, keepdims=True)
    dist = sq + sq.T - 2 * X @ X.T
    return np.exp(-gamma * dist)

K_mat = rbf_kernel_matrix(X_qp, gamma=0.5)
Q     = np.outer(y_qp, y_qp) * K_mat  # Q[i,j] = yᵢ yⱼ K(xᵢ,xⱼ)

# Dual objective: minimize ½αᵀQα - 1ᵀα
def dual_objective(alpha):
    return 0.5 * alpha @ Q @ alpha - alpha.sum()

def dual_gradient(alpha):
    return Q @ alpha - np.ones(n)

# Constraints and bounds
constraints = {'type': 'eq',
               'fun': lambda a: np.dot(a, y_qp),
               'jac': lambda a: y_qp}

bounds = [(0, C) for _ in range(n)]

# Solve
alpha0  = np.zeros(n)
result  = minimize(dual_objective, alpha0,
                   jac=dual_gradient,
                   method='SLSQP',
                   bounds=bounds,
                   constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-10})

alpha_sol = result.x
print(f"\nOptimization success: {result.success}")
print(f"Dual objective value: {-result.fun:.4f}")
print(f"Non-zero αᵢ (support vectors): {np.sum(alpha_sol > 1e-4)}")

# Recover w and b
sv_mask = alpha_sol > 1e-4
alpha_sv_sol = alpha_sol[sv_mask]
X_sv_sol     = X_qp[sv_mask]
y_sv_sol     = y_qp[sv_mask]

w_sol = np.sum(
    (alpha_sv_sol * y_sv_sol)[:, np.newaxis] * X_sv_sol,
    axis=0
)

# b from free support vectors: yₛ(wᵀxₛ + b) = 1  →  b = yₛ - wᵀxₛ
free_mask = (alpha_sol > 1e-4) & (alpha_sol < C - 1e-4)
if free_mask.sum() > 0:
    b_sol = np.mean(y_qp[free_mask] - X_qp[free_mask] @ w_sol)
else:
    b_sol = np.mean(y_qp[sv_mask] - X_qp[sv_mask] @ w_sol)

# Compare with sklearn
clf_sk = SVC(kernel='rbf', C=1.0, gamma=0.5)
clf_sk.fit(X_qp, y_qp)

pred_qp = np.sign(X_qp @ w_sol + b_sol)
print(f"\nSolved QP accuracy:  {np.mean(pred_qp == y_qp)*100:.1f}%")
print(f"Sklearn accuracy:    {clf_sk.score(X_qp, y_qp)*100:.1f}%")

# Verify KKT
margins_qp = y_qp * (X_qp @ w_sol + b_sol)
cs_vals    = alpha_sol * (margins_qp - 1)
print(f"\nKKT complementary slackness max violation: "
      f"{np.abs(cs_vals).max():.4e}")


# ============================================================
# SEQUENTIAL MINIMAL OPTIMIZATION (SMO) SKETCH
# ============================================================

print("\n\n--- SMO Algorithm Sketch ---")
print("""
The Sequential Minimal Optimization (SMO) algorithm
(Platt, 1998) solves the dual QP efficiently by:

1. Pick two indices i, j to optimize (heuristic)
2. Solve the 2-variable QP analytically (closed form!)
3. Update αᵢ, αⱼ, b
4. Repeat until convergence

Why two at a time?
  The constraint Σ αᵢ yᵢ = 0 means if we change one αᵢ,
  we must change at least one other to maintain feasibility.
  Two variables is the smallest feasible subproblem.

2-variable update:
  αⱼ_new = αⱼ_old + yⱼ(Eᵢ - Eⱼ) / η
  where Eᵢ = f(xᵢ) - yᵢ  (prediction error)
        η  = 2K(xᵢ,xⱼ) - K(xᵢ,xᵢ) - K(xⱼ,xⱼ)

This is what sklearn's SVC uses internally (via LIBSVM)!
""")
```

---

## From Scratch: Dual SVM Implementation

### Complete Dual SVM with SMO-Style Updates

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# ============================================================
# DUAL SVM FROM SCRATCH (Simplified SMO)
# ============================================================

class DualSVM:
    """
    Binary SVM using the dual formulation.
    Solves via simplified SMO (Sequential Minimal Optimization).
    
    Implements: max W(α) = Σαᵢ - ½ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ,xⱼ)
                s.t. Σαᵢyᵢ = 0, 0 ≤ αᵢ ≤ C
    """

    def __init__(self, C=1.0, kernel='rbf', gamma=1.0,
                 degree=3, coef0=1.0,
                 max_iter=200, tol=1e-4):
        self.C        = C
        self.kernel   = kernel
        self.gamma    = gamma
        self.degree   = degree
        self.coef0    = coef0
        self.max_iter = max_iter
        self.tol      = tol

        self.alpha    = None
        self.b        = 0.0
        self.X_train  = None
        self.y_train  = None
        self.K_cache  = None

    def _kernel(self, x1, x2):
        """Compute kernel between two vectors."""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            diff = x1 - x2
            return np.exp(-self.gamma * np.dot(diff, diff))
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _build_kernel_matrix(self, X):
        """Build full kernel matrix K[i,j] = K(xᵢ,xⱼ)."""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k = self._kernel(X[i], X[j])
                K[i, j] = k
                K[j, i] = k
        return K

    def _decision_function_all(self):
        """f(xᵢ) = Σⱼ αⱼ yⱼ K(xⱼ,xᵢ) + b  for all training points."""
        return (self.alpha * self.y_train) @ self.K_cache + self.b

    def _smo_step(self, i):
        """
        Try to optimize αᵢ by finding best j to pair with.
        Returns True if meaningful step was taken.
        """
        n    = len(self.y_train)
        Ei   = self._decision_function_all()[i] - self.y_train[i]
        ri   = Ei * self.y_train[i]

        # Check KKT violation
        if not ((ri < -self.tol and self.alpha[i] < self.C) or
                (ri >  self.tol and self.alpha[i] > 0)):
            return False

        # Pick j: index with largest |Ej - Ei| (heuristic)
        errors  = np.abs(self._decision_function_all() - self.y_train)
        j       = np.argmax(np.abs(errors - abs(Ei)))
        if j == i:
            j = (i + 1) % n

        # 2-variable SMO update
        return self._update_pair(i, j)

    def _update_pair(self, i, j):
        """Solve 2-variable QP for αᵢ and αⱼ."""
        if i == j:
            return False

        y  = self.y_train
        K  = self.K_cache
        f  = self._decision_function_all()

        # Compute eta: η = 2K(i,j) - K(i,i) - K(j,j)
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        if eta >= 0:
            return False

        # Compute errors
        Ei = f[i] - y[i]
        Ej = f[j] - y[j]

        # Compute bounds L and H
        if y[i] == y[j]:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

        if abs(H - L) < 1e-8:
            return False

        # Unconstrained new αⱼ
        alpha_j_new = self.alpha[j] - y[j] * (Ei - Ej) / eta

        # Clip to [L, H]
        alpha_j_new = np.clip(alpha_j_new, L, H)

        if abs(alpha_j_new - self.alpha[j]) < 1e-8:
            return False

        # Update αᵢ to maintain Σ αᵢ yᵢ = 0
        alpha_i_new = (self.alpha[i] +
                       y[i] * y[j] * (self.alpha[j] - alpha_j_new))

        # Update bias b
        b1 = (self.b - Ei
              - y[i] * (alpha_i_new - self.alpha[i]) * K[i, i]
              - y[j] * (alpha_j_new - self.alpha[j]) * K[i, j])

        b2 = (self.b - Ej
              - y[i] * (alpha_i_new - self.alpha[i]) * K[i, j]
              - y[j] * (alpha_j_new - self.alpha[j]) * K[j, j])

        if 0 < alpha_i_new < self.C:
            self.b = b1
        elif 0 < alpha_j_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        return True

    def fit(self, X, y):
        """Train using simplified SMO."""
        n = len(X)
        self.X_train  = X.copy()
        self.y_train  = y.astype(float)
        self.alpha    = np.zeros(n)
        self.b        = 0.0

        print(f"Building {n}×{n} kernel matrix...")
        self.K_cache  = self._build_kernel_matrix(X)

        for iteration in range(self.max_iter):
            changed = 0
            for i in range(n):
                if self._smo_step(i):
                    changed += 1

            if iteration % 50 == 0:
                obj = self._dual_objective()
                sv  = np.sum(self.alpha > 1e-4)
                print(f"  Iter {iteration:3d}: "
                      f"changed={changed:3d}, "
                      f"W(α)={obj:.4f}, "
                      f"SV={sv}")

            if changed == 0:
                print(f"  Converged at iteration {iteration}")
                break

        return self

    def _dual_objective(self):
        """W(α) = Σαᵢ - ½ΣΣ αᵢ αⱼ yᵢ yⱼ K(xᵢ,xⱼ)"""
        ay = self.alpha * self.y_train
        return self.alpha.sum() - 0.5 * ay @ self.K_cache @ ay

    def decision_function(self, X_test):
        """f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b"""
        ay = self.alpha * self.y_train
        scores = np.array([
            sum(ay[i] * self._kernel(self.X_train[i], x)
                for i in range(len(self.X_train)))
            + self.b
            for x in X_test
        ])
        return scores

    def predict(self, X_test):
        return np.sign(self.decision_function(X_test)).astype(int)

    def score(self, X_test, y_test):
        return np.mean(self.predict(X_test) == y_test)


# ============================================================
# TEST THE DUAL SVM
# ============================================================

print("="*60)
print("DUAL SVM FROM SCRATCH: TEST")
print("="*60)

datasets = {
    "Moons":   make_moons(n_samples=80, noise=0.15, random_state=42),
    "Circles": make_circles(n_samples=80, noise=0.08,
                            factor=0.4, random_state=42),
}

for ds_name, (X_ds, y_ds) in datasets.items():
    y_pm_ds = np.where(y_ds == 0, -1, 1)
    X_sc    = StandardScaler().fit_transform(X_ds)

    print(f"\n{'='*40}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*40}")

    # Our implementation
    our_svm = DualSVM(C=1.0, kernel='rbf', gamma=1.0,
                      max_iter=200, tol=1e-3)
    our_svm.fit(X_sc, y_pm_ds)
    our_acc = our_svm.score(X_sc, y_pm_ds)
    our_sv  = np.sum(our_svm.alpha > 1e-4)

    # Sklearn reference
    sk_svm  = SVC(kernel='rbf', gamma=1.0, C=1.0)
    sk_svm.fit(X_sc, y_pm_ds)
    sk_acc  = sk_svm.score(X_sc, y_pm_ds)
    sk_sv   = sk_svm.n_support_.sum()

    print(f"\nOur DualSVM:  acc={our_acc*100:.1f}%,  SV={our_sv}")
    print(f"Sklearn SVC:  acc={sk_acc*100:.1f}%,  SV={sk_sv}")
```

---

## Summary and Quick Reference

### The Complete Primal-to-Dual Roadmap

```
PRIMAL SVM (Hard Margin)
─────────────────────────
min_{w,b}  ½||w||²
s.t.  yᵢ(wᵀxᵢ + b) ≥ 1

        ↓  Form Lagrangian L(w,b,α)
        ↓  Set ∂L/∂w = 0 → w = Σαᵢyᵢxᵢ
        ↓  Set ∂L/∂b = 0 → Σαᵢyᵢ = 0
        ↓  Substitute back into L

DUAL SVM (Hard Margin)
─────────────────────────
max_{α}  Σαᵢ - ½ΣΣ αᵢαⱼ yᵢyⱼ (xᵢᵀxⱼ)
s.t.  Σαᵢyᵢ = 0,  αᵢ ≥ 0

        ↓  Replace (xᵢᵀxⱼ) → K(xᵢ,xⱼ)

KERNELIZED DUAL SVM
─────────────────────────
max_{α}  Σαᵢ - ½ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ,xⱼ)
s.t.  Σαᵢyᵢ = 0,  αᵢ ≥ 0

        ↓  Add slack variables ξᵢ for soft margin

KERNELIZED SOFT-MARGIN DUAL SVM
─────────────────────────────────
max_{α}  Σαᵢ - ½ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ,xⱼ)
s.t.  Σαᵢyᵢ = 0,  0 ≤ αᵢ ≤ C

Decision:  f(x) = Σᵢ αᵢ yᵢ K(xᵢ,x) + b
```

### Prerequisites Checklist

Before moving to Chapter 05, confirm you can:
- [ ] Explain why the dual enables the kernel trick
- [ ] Write the primal and dual objectives from memory
- [ ] Explain what each KKT condition means geometrically
- [ ] Identify the three types of points in soft-margin SVM (α=0, 0<α<C, α=C)
- [ ] Implement the dual decision function f(x) = Σαᵢyᵢ K(xᵢ,x) + b
- [ ] Explain how C changes the bound on α in the dual

### What's Next

**05_Kernel_Functions.md** ← already exists  
**06_Radial_Basis_Functions.md** ← already exists  
**07_SVM_Hard_Margin.md**  
→ Full optimization derivation, geometric margin, hard-margin theory

**08_SVM_Soft_Margin.md**  
→ Slack variables, hinge loss, the C parameter in depth

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════╗
║                DUAL FORMULATIONS — QUICK REFERENCE               ║
╠══════════════════════════════════════════════════════════════════╣
║  HARD-MARGIN DUAL                                                ║
║    max  W(α) = Σαᵢ - ½ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ,xⱼ)                   ║
║    s.t. Σαᵢyᵢ = 0,   αᵢ ≥ 0                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  SOFT-MARGIN DUAL (adds box constraint)                          ║
║    max  W(α) = Σαᵢ - ½ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ,xⱼ)                   ║
║    s.t. Σαᵢyᵢ = 0,   0 ≤ αᵢ ≤ C                                ║
╠══════════════════════════════════════════════════════════════════╣
║  RECOVER PRIMAL FROM DUAL                                        ║
║    w = Σᵢ αᵢ yᵢ xᵢ   (or φ(xᵢ) in feature space)               ║
║    b = yₛ - Σᵢ αᵢ yᵢ K(xᵢ,xₛ)  for any free SV s               ║
╠══════════════════════════════════════════════════════════════════╣
║  DECISION FUNCTION                                               ║
║    f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b                                ║
║    pred = sign(f(x))                                             ║
╠══════════════════════════════════════════════════════════════════╣
║  KKT: THREE TYPES OF TRAINING POINTS                             ║
║    αᵢ = 0:         non-SV, outside margin, ignored              ║
║    0 < αᵢ < C:     free SV, exactly on margin                   ║
║    αᵢ = C:         bounded SV, inside margin or misclassified    ║
╠══════════════════════════════════════════════════════════════════╣
║  KEY INSIGHT                                                     ║
║    Data appears ONLY as K(xᵢ,xⱼ) in the dual                    ║
║    → Replace dot product with any valid kernel                   ║
║    → Algorithm implicitly operates in feature space              ║
╠══════════════════════════════════════════════════════════════════╣
║  SKLEARN                                                         ║
║    clf = SVC(kernel='rbf', C=1.0, gamma='scale')                 ║
║    clf.dual_coef_       # αᵢ·yᵢ for support vectors             ║
║    clf.support_vectors_ # xᵢ for support vectors                ║
║    clf.support_         # indices of support vectors             ║
║    clf.intercept_       # bias b                                 ║
║    clf.decision_function(X) # f(x) = Σαᵢyᵢ K(xᵢ,x) + b         ║
╚══════════════════════════════════════════════════════════════════╝
```

---

*End of 04_Dual_Formulations.md*