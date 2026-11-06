# SVM: Hard Margin

## Table of Contents
1. [The Maximum Margin Principle](#the-maximum-margin-principle)
2. [Geometric Interpretation](#geometric-interpretation)
3. [Mathematical Formulation](#mathematical-formulation)
4. [The Optimization Problem](#the-optimization-problem)
5. [Solving the Hard-Margin SVM](#solving-the-hard-margin-svm)
6. [Support Vectors](#support-vectors)
7. [Limitations of Hard-Margin SVM](#limitations-of-hard-margin-svm)
8. [Summary](#summary)

---

## The Maximum Margin Principle

### The Core Idea

Given linearly separable data, there are **infinitely many** hyperplanes that can separate the classes. Which one should we choose?

**SVM's answer**: Choose the hyperplane with the **maximum margin** — the largest minimum distance to any training point.

```
Why maximum margin?

1. GENERALIZATION: Larger margin → better generalization
2. ROBUSTNESS: Farther from noise and perturbations  
3. UNIQUENESS: Only one maximum-margin hyperplane exists
4. GEOMETRIC: Natural notion of "best" separation
```

### Visual Intuition

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate linearly separable data
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, class_sep=2.5, random_state=42)
y = np.where(y == 0, -1, 1)
X = StandardScaler().fit_transform(X)

# Train SVM
svm = SVC(kernel='linear', C=1e6)  # Large C ≈ hard margin
svm.fit(X, y)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Multiple separating hyperplanes
ax = axes[0]
ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', s=50, label='Class -1')
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, marker='s', label='Class +1')

# Draw several possible separating lines
for slope, intercept in [(0.5, 0.2), (0.7, -0.1), (0.9, 0.3), (1.1, 0.0)]:
    x_line = np.linspace(-3, 3, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'k--', alpha=0.3, linewidth=1)

ax.text(0, 2, 'Many possible\nseparating hyperplanes', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title('Infinite Separating Hyperplanes')

# Panel 2: Maximum margin hyperplane
ax = axes[1]
w = svm.coef_[0]
b = svm.intercept_[0]

xx = np.linspace(-3, 3, 100)
yy_boundary = -(w[0] * xx + b) / w[1]
yy_margin_pos = -(w[0] * xx + b - 1) / w[1]
yy_margin_neg = -(w[0] * xx + b + 1) / w[1]

ax.plot(xx, yy_boundary, 'k-', linewidth=2, label='Decision boundary')
ax.plot(xx, yy_margin_pos, 'r--', linewidth=1.5, label='Margin +1')
ax.plot(xx, yy_margin_neg, 'b--', linewidth=1.5, label='Margin -1')

ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', s=50)
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, marker='s')

# Highlight support vectors
sv = svm.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
           edgecolors='gold', linewidths=3, label='Support Vectors')

margin_width = 2 / np.linalg.norm(w)
ax.text(0, 2, f'Maximum margin\nwidth = {margin_width:.3f}', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title('SVM: Maximum Margin Hyperplane')

plt.tight_layout()
plt.savefig('hard_margin_principle.png', dpi=150, bbox_inches='tight')
print("Saved: hard_margin_principle.png")
```

---

## Geometric Interpretation

### The Decision Boundary

A hyperplane in ℝᵈ is defined by:

```
wᵀx + b = 0

Where:
  w ∈ ℝᵈ  is the weight vector (normal to the hyperplane)
  b ∈ ℝ   is the bias term
  x ∈ ℝᵈ  is any point on the hyperplane
```

### Distance from Point to Hyperplane

The **signed distance** from a point x to the hyperplane wᵀx + b = 0 is:

```
distance = (wᵀx + b) / ||w||

Where ||w|| = √(w₁² + w₂² + ... + wₐ²)

Sign indicates which side of the hyperplane:
  > 0:  point is on the +1 side
  < 0:  point is on the -1 side
```

### Margin Definition

The **margin** is the perpendicular distance from the decision boundary to the nearest data point.

```
For correctly classified points:
  yᵢ(wᵀxᵢ + b) > 0    (correct side)

Functional margin:  yᵢ(wᵀxᵢ + b)
Geometric margin:   yᵢ(wᵀxᵢ + b) / ||w||

The margin of the classifier is:
  γ = min_{i=1,...,n} yᵢ(wᵀxᵢ + b) / ||w||
```

```python
# Verify margin calculation
print("="*60)
print("MARGIN CALCULATION")
print("="*60)

w = svm.coef_[0]
b = svm.intercept_[0]
w_norm = np.linalg.norm(w)

# Compute signed distances for all points
signed_distances = y * (X @ w + b) / w_norm

print(f"\nWeight vector w: {w}")
print(f"Bias b: {b:.4f}")
print(f"||w||: {w_norm:.4f}")

print(f"\nSigned distances (geometric margins):")
print(f"  Min: {signed_distances.min():.4f}")
print(f"  Max: {signed_distances.max():.4f}")
print(f"  Mean: {signed_distances.mean():.4f}")

# Support vectors have margin ≈ 1/||w||
sv_indices = svm.support_
sv_distances = signed_distances[sv_indices]
print(f"\nSupport vector distances:")
print(f"  {sv_distances}")
print(f"  All ≈ {1/w_norm:.4f} = 1/||w||  ✓")

print(f"\nMargin width = 2/||w|| = {2/w_norm:.4f}")
```

---

## Mathematical Formulation

### Primal Problem

The **hard-margin SVM** optimization problem:

```
Minimize:    ½ ||w||²

Subject to:  yᵢ(wᵀxᵢ + b) ≥ 1    for i = 1, ..., n
```

**Intuition**:
- **Objective**: Minimize ||w||² ⟺ Maximize margin = 2/||w||
- **Constraints**: All points correctly classified with margin ≥ 1

**Why ||w||² instead of ||w||?**
- ||w||² is differentiable everywhere
- Makes optimization convex quadratic (easier to solve)
- Factor ½ makes derivative cleaner

### Constraint Normalization

Why the constraint yᵢ(wᵀxᵢ + b) ≥ 1 and not yᵢ(wᵀxᵢ + b) ≥ γ?

```
The "1" is a normalization convention:

If (w, b) satisfies  yᵢ(wᵀxᵢ + b) ≥ γ
Then (w/γ, b/γ) satisfies  yᵢ((w/γ)ᵀxᵢ + b/γ) ≥ 1

We can always rescale to make the margin equal 1.
This eliminates one degree of freedom and simplifies the problem.
```

### Geometric Margin in Terms of w

```
From yᵢ(wᵀxᵢ + b) ≥ 1:

Geometric margin = 1 / ||w||

Therefore:
  Maximize margin  ⟺  Maximize 1/||w||
                    ⟺  Minimize ||w||
                    ⟺  Minimize ½||w||²

The last form is preferred because it's a quadratic function.
```

---

## The Optimization Problem

### Problem Classification

The hard-margin SVM is a **Convex Quadratic Program (QP)**:

```
Minimize:    f(w) = ½||w||² = ½wᵀw        (convex quadratic objective)

Subject to:  gᵢ(w,b) = 1 - yᵢ(wᵀxᵢ + b) ≤ 0    (linear constraints)
```

**Key properties**:
1. **Convex**: Objective is convex, constraints are convex
2. **Quadratic**: Objective is quadratic in w
3. **Linear constraints**: Each constraint is a linear inequality
4. **Unique solution**: Convex QP has unique global minimum

### Lagrangian Formulation

Introduce Lagrange multipliers αᵢ ≥ 0:

```
L(w, b, α) = ½||w||² - Σᵢ αᵢ[yᵢ(wᵀxᵢ + b) - 1]

           = ½||w||² - Σᵢ αᵢyᵢ(wᵀxᵢ + b) + Σᵢ αᵢ
```

### Dual Problem Derivation

**Step 1**: Set derivatives to zero:

```
∂L/∂w = 0:   w - Σᵢ αᵢyᵢxᵢ = 0
              ⟹  w = Σᵢ αᵢyᵢxᵢ

∂L/∂b = 0:   -Σᵢ αᵢyᵢ = 0
              ⟹  Σᵢ αᵢyᵢ = 0
```

**Step 2**: Substitute back into Lagrangian:

```
L_dual(α) = Σᵢ αᵢ - ½ Σᵢ Σⱼ αᵢαⱼyᵢyⱼ(xᵢᵀxⱼ)
```

**Step 3**: Dual optimization problem:

```
Maximize:    W(α) = Σᵢ αᵢ - ½ Σᵢ Σⱼ αᵢαⱼyᵢyⱼ(xᵢᵀxⱼ)

Subject to:  Σᵢ αᵢyᵢ = 0
             αᵢ ≥ 0    for i = 1, ..., n
```

### KKT Conditions

The **Karush-Kuhn-Tucker (KKT) conditions** are necessary and sufficient for optimality:

```
1. Stationarity:
   w = Σᵢ αᵢyᵢxᵢ
   Σᵢ αᵢyᵢ = 0

2. Primal feasibility:
   yᵢ(wᵀxᵢ + b) ≥ 1    ∀i

3. Dual feasibility:
   αᵢ ≥ 0    ∀i

4. Complementary slackness:
   αᵢ[yᵢ(wᵀxᵢ + b) - 1] = 0    ∀i
```

**Key insight from complementary slackness**:

```
Either:  αᵢ = 0     →  Point is not a support vector
Or:      yᵢ(wᵀxᵢ + b) = 1  →  Point is exactly on the margin
```

---

## Solving the Hard-Margin SVM

### Small Example: 2D Linearly Separable

```python
import numpy as np
from scipy.optimize import minimize

# Simple 2D example
X_simple = np.array([[1, 1], [2, 2], [2, 0],  # Class +1
                     [-1, -1], [-2, -2], [-2, 0]])  # Class -1
y_simple = np.array([1, 1, 1, -1, -1, -1])

print("="*60)
print("HARD-MARGIN SVM: MANUAL SOLUTION")
print("="*60)
print(f"\nData shape: {X_simple.shape}")
print(f"Classes: {np.unique(y_simple)}")

# Dual objective: max Σαᵢ - ½ΣΣ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
# Convert to minimization: min -Σαᵢ + ½ΣΣ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
def dual_objective(alpha, X, y):
    n = len(y)
    K = X @ X.T  # Kernel matrix (linear kernel)
    
    # Objective: -Σαᵢ + ½ΣΣ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
    term1 = -np.sum(alpha)
    term2 = 0.5 * np.sum((alpha * y).reshape(-1, 1) @ 
                         (alpha * y).reshape(1, -1) * K)
    return term1 + term2

# Constraints
from scipy.optimize import LinearConstraint, Bounds

# Equality: Σαᵢyᵢ = 0
eq_constraint = LinearConstraint(y_simple, 0, 0)

# Bounds: αᵢ ≥ 0
bounds = Bounds(0, np.inf)

# Initial guess
alpha0 = np.zeros(len(y_simple))

# Solve
result = minimize(dual_objective, alpha0, args=(X_simple, y_simple),
                  method='SLSQP', constraints=eq_constraint, bounds=bounds)

alpha_sol = result.x

print(f"\nOptimization success: {result.success}")
print(f"Alpha values: {alpha_sol.round(4)}")

# Recover w and b
w_sol = np.sum((alpha_sol * y_simple).reshape(-1, 1) * X_simple, axis=0)
print(f"\nWeight vector w: {w_sol}")

# Compute b from support vectors (where αᵢ > 0)
sv_mask = alpha_sol > 1e-4
b_sol = np.mean(y_simple[sv_mask] - X_simple[sv_mask] @ w_sol)
print(f"Bias b: {b_sol:.4f}")

# Verify constraints
margins = y_simple * (X_simple @ w_sol + b_sol)
print(f"\nMargins yᵢ(wᵀxᵢ + b):")
print(margins.round(4))
print(f"All ≥ 1: {np.all(margins >= 0.99)}")

print(f"\nSupport vectors (αᵢ > 0):")
print(f"  Indices: {np.where(sv_mask)[0]}")
print(f"  Alpha values: {alpha_sol[sv_mask].round(4)}")

# Compare with sklearn
from sklearn.svm import SVC
svm_check = SVC(kernel='linear', C=1e6)
svm_check.fit(X_simple, y_simple)

print(f"\nsklearn w: {svm_check.coef_[0]}")
print(f"sklearn b: {svm_check.intercept_[0]:.4f}")
print(f"Our w:     {w_sol}")
print(f"Our b:     {b_sol:.4f}")
```

---

## Support Vectors

### Definition

**Support vectors** are the training points that lie exactly on the margin boundaries (where yᵢ(wᵀxᵢ + b) = 1).

```
Properties of support vectors:
1. αᵢ > 0  (non-zero dual coefficient)
2. Lie on the margin boundary
3. Determine the decision boundary
4. Removing non-support vectors doesn't change the solution
```

### Why "Support" Vectors?

```
The decision function is:

f(x) = sign(wᵀx + b) = sign(Σᵢ αᵢyᵢxᵢᵀx + b)

Since αᵢ = 0 for non-support vectors:

f(x) = sign(Σ_{i∈SV} αᵢyᵢxᵢᵀx + b)

Only support vectors "support" (define) the hyperplane!
```

### Visual Analysis

```python
# Analyze support vectors in detail
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Annotated support vectors
ax = axes[0]
ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', s=50, alpha=0.6)
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, marker='s', alpha=0.6)

# Decision boundary and margins
xx = np.linspace(-3, 3, 100)
ax.plot(xx, -(w[0]*xx + b)/w[1], 'k-', linewidth=2)
ax.plot(xx, -(w[0]*xx + b - 1)/w[1], 'r--', linewidth=1.5)
ax.plot(xx, -(w[0]*xx + b + 1)/w[1], 'b--', linewidth=1.5)

# Highlight and annotate support vectors
for i, (sv_point, alpha_val) in enumerate(zip(svm.support_vectors_, 
                                               np.abs(svm.dual_coef_[0]))):
    ax.scatter(sv_point[0], sv_point[1], s=300, facecolors='none',
               edgecolors='gold', linewidths=3)
    ax.annotate(f'α={alpha_val:.3f}', xy=sv_point, 
                xytext=(sv_point[0]+0.3, sv_point[1]+0.3),
                fontsize=9, color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_title(f'Support Vectors (n={len(svm.support_vectors_)})')
ax.grid(True, alpha=0.3)

# Panel 2: Distance distribution
ax = axes[1]
all_distances = signed_distances
sv_distances = signed_distances[sv_indices]
nonsv_distances = np.delete(signed_distances, sv_indices)

ax.hist(nonsv_distances, bins=15, alpha=0.6, color='gray', 
        label=f'Non-SV (n={len(nonsv_distances)})')
ax.hist(sv_distances, bins=5, alpha=0.8, color='gold',
        label=f'Support Vectors (n={len(sv_distances)})')
ax.axvline(x=1/w_norm, color='red', linestyle='--', linewidth=2,
           label=f'Margin boundary (d={1/w_norm:.3f})')
ax.set_xlabel('Signed Distance to Hyperplane')
ax.set_ylabel('Count')
ax.set_title('Distance Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('support_vectors_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: support_vectors_analysis.png")

# Statistics
print("\n" + "="*60)
print("SUPPORT VECTOR STATISTICS")
print("="*60)
print(f"Total training points: {len(X)}")
print(f"Support vectors: {len(sv_indices)} ({len(sv_indices)/len(X)*100:.1f}%)")
print(f"Non-support vectors: {len(X) - len(sv_indices)}")
print(f"\nSupport vector distances: {sv_distances.round(4)}")
print(f"Expected distance (1/||w||): {1/w_norm:.4f}")
```

---

## Limitations of Hard-Margin SVM

### When Hard-Margin Fails

```
Hard-margin SVM requires:
1. Data must be PERFECTLY linearly separable
2. No noise or outliers can exist in the data
3. All constraints yᵢ(wᵀxᵢ + b) ≥ 1 must be satisfiable

If data is not separable → NO SOLUTION EXISTS
```

### Demonstration: Non-Separable Data

```python
# Create non-separable data
np.random.seed(42)
X_nonsep = np.vstack([
    np.random.randn(40, 2) + [2, 2],    # Class +1 cluster
    np.random.randn(40, 2) + [-2, -2],  # Class -1 cluster  
    np.random.randn(10, 2)               # Noise in the middle
])
y_nonsep = np.array([1]*40 + [-1]*40 + [1]*5 + [-1]*5)
y_nonsep = np.where(y_nonsep == 0, -1, 1)

print("="*60)
print("HARD-MARGIN SVM ON NON-SEPARABLE DATA")
print("="*60)

# Try hard-margin SVM (large C approximates hard margin)
try:
    svm_hard = SVC(kernel='linear', C=1e6)
    svm_hard.fit(X_nonsep, y_nonsep)
    print(f"Training accuracy: {svm_hard.score(X_nonsep, y_nonsep)*100:.1f}%")
    
    # Check if any points violate margin
    w_h = svm_hard.coef_[0]
    b_h = svm_hard.intercept_[0]
    margins_h = y_nonsep * (X_nonsep @ w_h + b_h)
    violations = np.sum(margins_h < 0.99)
    print(f"Margin violations: {violations}/{len(X_nonsep)}")
    
    if violations > 0:
        print("→ Hard-margin SVM failed! Some points inside margin.")
except:
    print("Hard-margin SVM failed to converge!")

print("\nSOLUTION: Use Soft-Margin SVM (Chapter 08)")
```

### Three Failure Modes

```
1. OVERLAPPING CLASSES
   • Classes mixed together in feature space
   • No linear separator exists
   • Solution: Soft margin or non-linear kernel

2. OUTLIERS
   • One or two mislabeled or noisy points
   • Would require very complex boundary to accommodate
   • Solution: Soft margin (allow some violations)

3. NON-LINEAR PATTERNS
   • Data separated by curved boundary
   • Linear separator doesn't exist
   • Solution: Kernel trick (Chapter 09)
```

---

## Summary

### Key Concepts

```
HARD-MARGIN SVM:

Goal:
  Find the hyperplane that maximally separates linearly separable data

Optimization:
  Minimize:   ½||w||²
  Subject to: yᵢ(wᵀxᵢ + b) ≥ 1    ∀i

Geometric Meaning:
  Margin = 2/||w||
  Maximize margin ⟺ Minimize ||w||

Dual Problem:
  Maximize:   Σαᵢ - ½ΣΣ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
  Subject to: Σαᵢyᵢ = 0,  αᵢ ≥ 0

Decision Function:
  f(x) = sign(wᵀx + b) = sign(Σᵢ αᵢyᵢxᵢᵀx + b)

Support Vectors:
  Points with αᵢ > 0
  Lie exactly on margin boundaries
  Determine the decision boundary
```

### Advantages vs Disadvantages

```
ADVANTAGES:
✓ Unique global optimum (convex optimization)
✓ Principled: maximum margin → good generalization
✓ Sparse solution (only support vectors matter)
✓ Theoretically well-founded

DISADVANTAGES:
✗ Requires perfect linear separability
✗ Sensitive to outliers
✗ No solution exists if data not separable
✗ Only works for linearly separable problems
```

### What's Next

```
Hard-Margin SVM limitations → Soft-Margin SVM (Chapter 08)

Soft-Margin adds:
  • Slack variables ξᵢ to allow violations
  • Parameter C to control margin vs violations trade-off
  • Works on non-separable data

Then: Kernel Trick (Chapter 09)
  • Handle non-linear patterns
  • Apply soft-margin SVM in transformed space
```

---

## Quick Reference

```
╔══════════════════════════════════════════════════════════════╗
║              HARD-MARGIN SVM — QUICK REFERENCE               ║
╠══════════════════════════════════════════════════════════════╣
║  PRIMAL PROBLEM                                              ║
║    min  ½||w||²                                              ║
║    s.t. yᵢ(wᵀxᵢ + b) ≥ 1    ∀i                              ║
╠══════════════════════════════════════════════════════════════╣
║  DUAL PROBLEM                                                ║
║    max  Σαᵢ - ½ΣΣ αᵢαⱼyᵢyⱼxᵢᵀxⱼ                            ║
║    s.t. Σαᵢyᵢ = 0,  αᵢ ≥ 0                                  ║
╠══════════════════════════════════════════════════════════════╣
║  KEY EQUATIONS                                               ║
║    Margin width:  2/||w||                                    ║
║    w = Σᵢ αᵢyᵢxᵢ                                            ║
║    f(x) = sign(wᵀx + b)                                      ║
╠══════════════════════════════════════════════════════════════╣
║  SUPPORT VECTORS                                             ║
║    Points with αᵢ > 0                                        ║
║    Satisfy yᵢ(wᵀxᵢ + b) = 1                                 ║
║    Lie on margin boundaries                                  ║
╠══════════════════════════════════════════════════════════════╣
║  KKT CONDITIONS                                              ║
║    Complementary slackness:                                  ║
║    αᵢ[yᵢ(wᵀxᵢ + b) - 1] = 0                                ║
╠══════════════════════════════════════════════════════════════╣
║  REQUIREMENTS                                                ║
║    ✓ Data must be linearly separable                         ║
║    ✓ No noise or outliers                                    ║
║    ✗ Fails on overlapping classes                            ║
╠══════════════════════════════════════════════════════════════╣
║  SOLUTION → SOFT-MARGIN SVM (Chapter 08)                     ║
║    Add slack variables to handle violations                  ║
╚══════════════════════════════════════════════════════════════╝
```

---

*End of 07_SVM_Hard_Margin.md*