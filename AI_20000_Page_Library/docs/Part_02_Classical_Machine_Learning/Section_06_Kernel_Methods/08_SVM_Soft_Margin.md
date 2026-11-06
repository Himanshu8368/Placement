# SVM: Soft Margin

## Table of Contents
1. [The Need for Soft Margins](#the-need-for-soft-margins)
2. [Slack Variables](#slack-variables)
3. [The Soft-Margin Optimization Problem](#the-soft-margin-optimization-problem)
4. [The C Parameter](#the-c-parameter)
5. [Dual Formulation](#dual-formulation)
6. [Hinge Loss Interpretation](#hinge-loss-interpretation)
7. [Support Vectors in Soft-Margin SVM](#support-vectors-in-soft-margin-svm)
8. [Choosing C](#choosing-c)
9. [Summary](#summary)

---

## The Need for Soft Margins

### Hard-Margin Limitations

Hard-margin SVM requires **perfect linear separability**:

```
Problems:
1. Real data is rarely perfectly separable
2. Outliers force complex boundaries
3. Noise causes overfitting
4. No solution exists if data overlaps

Example:
  Class +1: ○○○○○○○ ●     ← One outlier
  Class -1:        ○○○○○○○
  
  Hard margin: NO SOLUTION (infeasible)
  Soft margin: Allow the ● to violate the margin
```

### Solution: Allow Violations

**Soft-margin SVM** introduces **slack variables** ξᵢ ≥ 0 to allow some points to:
- Be within the margin (0 < ξᵢ < 1)
- Be on the wrong side of the decision boundary (ξᵢ > 1)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate data with overlap
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, class_sep=1.0, random_state=42)
y = np.where(y == 0, -1, 1)
X = StandardScaler().fit_transform(X)

# Add outliers
X = np.vstack([X, [2.5, -2.0], [-2.5, 2.0]])
y = np.hstack([y, [-1], [1]])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Hard margin attempt (will fail/overfit)
ax = axes[0]
try:
    svm_hard = SVC(kernel='linear', C=1e6)  # Very large C ≈ hard margin
    svm_hard.fit(X, y)
    
    w = svm_hard.coef_[0]
    b = svm_hard.intercept_[0]
    
    xx = np.linspace(-3, 4, 100)
    yy = -(w[0]*xx + b)/w[1]
    
    ax.plot(xx, yy, 'k-', linewidth=2, label='Decision boundary')
    ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', s=50, label='Class -1')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, marker='s', label='Class +1')
    
    # Highlight outliers
    ax.scatter([2.5, -2.5], [-2.0, 2.0], s=300, facecolors='none',
               edgecolors='orange', linewidths=3, label='Outliers')
    
    margin_width = 2 / np.linalg.norm(w)
    ax.set_title(f'Hard Margin (C=1e6)\nMargin={margin_width:.3f}\nOverfits to outliers')
except:
    ax.text(0.5, 0.5, 'Hard Margin\nFAILED\n(No solution)',
            ha='center', va='center', transform=ax.transAxes, fontsize=16)
    ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', s=50)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, marker='s')

ax.set_xlim(-3, 4); ax.set_ylim(-3, 3)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 2: Soft margin (works!)
ax = axes[1]
svm_soft = SVC(kernel='linear', C=0.1)  # Small C = soft margin
svm_soft.fit(X, y)

w = svm_soft.coef_[0]
b = svm_soft.intercept_[0]

xx = np.linspace(-3, 4, 100)
yy = -(w[0]*xx + b)/w[1]
yy_m1 = -(w[0]*xx + b - 1)/w[1]
yy_m2 = -(w[0]*xx + b + 1)/w[1]

ax.plot(xx, yy, 'k-', linewidth=2, label='Decision boundary')
ax.plot(xx, yy_m1, 'r--', linewidth=1.5, alpha=0.5)
ax.plot(xx, yy_m2, 'b--', linewidth=1.5, alpha=0.5)

ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', s=50, label='Class -1')
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, marker='s', label='Class +1')

# Highlight outliers
ax.scatter([2.5, -2.5], [-2.0, 2.0], s=300, facecolors='none',
           edgecolors='orange', linewidths=3, label='Outliers (allowed)')

margin_width = 2 / np.linalg.norm(w)
ax.set_title(f'Soft Margin (C=0.1)\nMargin={margin_width:.3f}\nIgnores outliers')

ax.set_xlim(-3, 4); ax.set_ylim(-3, 3)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle('Hard Margin vs Soft Margin SVM', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('soft_margin_motivation.png', dpi=150, bbox_inches='tight')
print("Saved: soft_margin_motivation.png")
```

---

## Slack Variables

### Definition

For each training point xᵢ, introduce a **slack variable** ξᵢ ≥ 0:

```
ξᵢ = 0:     Point is correctly classified outside margin
ξᵢ ∈ (0,1): Point is correctly classified but inside margin
ξᵢ = 1:     Point is exactly on the decision boundary
ξᵢ > 1:     Point is misclassified

Relaxed constraint:
  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ

If ξᵢ = 0:     yᵢ(wᵀxᵢ + b) ≥ 1       (hard margin)
If ξᵢ > 0:     yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ  (violation allowed)
```

### Geometric Meaning

```
Distance from xᵢ to decision boundary:
  dᵢ = yᵢ(wᵀxᵢ + b) / ||w||

Slack variable:
  ξᵢ = max(0, 1 - yᵢ(wᵀxᵢ + b))

Types of points:
  ξᵢ = 0:    Correctly classified, outside margin
  0 < ξᵢ < 1: Inside margin but correct side
  ξᵢ ≥ 1:    Wrong side (misclassified)
```

```python
# Visualize slack variables
print("="*60)
print("SLACK VARIABLES: GEOMETRIC INTERPRETATION")
print("="*60)

svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]

# Compute functional margins
functional_margins = y * (X @ w + b)

# Compute slack variables
slack = np.maximum(0, 1 - functional_margins)

# Classify points
zero_slack = slack < 1e-4
margin_violation = (slack > 1e-4) & (slack < 1)
misclassified = slack >= 1

print(f"\nTotal points: {len(X)}")
print(f"  ξ = 0 (no violation):      {zero_slack.sum()} points")
print(f"  0 < ξ < 1 (margin violn):  {margin_violation.sum()} points")
print(f"  ξ ≥ 1 (misclassified):     {misclassified.sum()} points")

print(f"\nTotal slack: Σξᵢ = {slack.sum():.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Points colored by slack
ax = axes[0]
scatter = ax.scatter(X[:, 0], X[:, 1], c=slack, cmap='Reds',
                     s=100, edgecolors='k', linewidths=0.5)
plt.colorbar(scatter, ax=ax, label='Slack ξᵢ')

xx = np.linspace(-3, 4, 100)
ax.plot(xx, -(w[0]*xx + b)/w[1], 'k-', linewidth=2)
ax.plot(xx, -(w[0]*xx + b - 1)/w[1], 'k--', linewidth=1, alpha=0.5)
ax.plot(xx, -(w[0]*xx + b + 1)/w[1], 'k--', linewidth=1, alpha=0.5)

ax.set_title('Points Colored by Slack Variable ξᵢ\nRed = larger violation')
ax.grid(True, alpha=0.3)

# Panel 2: Slack distribution
ax = axes[1]
ax.hist(slack[slack > 1e-4], bins=15, edgecolor='k', alpha=0.7, color='coral')
ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
           label='ξ=1 (misclassification boundary)')
ax.set_xlabel('Slack ξᵢ')
ax.set_ylabel('Count')
ax.set_title('Distribution of Slack Variables\n(only showing ξ > 0)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('slack_variables.png', dpi=150, bbox_inches='tight')
print("\nSaved: slack_variables.png")
```

---

## The Soft-Margin Optimization Problem

### Primal Formulation

```
Minimize:    ½||w||² + C Σᵢ ξᵢ

Subject to:  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ    ∀i
             ξᵢ ≥ 0                     ∀i

Where:
  ½||w||²:     Maximize margin (same as before)
  C Σᵢ ξᵢ:     Penalty for violations
  C > 0:       Regularization parameter
```

### Trade-off

```
Objective = ½||w||² + C Σᵢ ξᵢ
            ↑           ↑
         Margin    Violations
       (complexity) (error)

Large C:  Focus on minimizing violations → narrow margin
Small C:  Focus on maximizing margin → wide margin, allow violations
```

### Lagrangian

```
L(w, b, ξ, α, μ) = ½||w||² + C Σᵢ ξᵢ
                   - Σᵢ αᵢ[yᵢ(wᵀxᵢ + b) - 1 + ξᵢ]
                   - Σᵢ μᵢ ξᵢ

Where:
  αᵢ ≥ 0:  Lagrange multipliers for margin constraints
  μᵢ ≥ 0:  Lagrange multipliers for ξᵢ ≥ 0
```

---

## The C Parameter

### What C Controls

```
C → ∞:    Hard margin (no violations allowed)
           Small margin, overfits

C = 1:     Balanced trade-off
           Standard choice

C → 0:     Maximum margin (violations don't matter)
           Wide margin, underfits
```

### Effect on Margin and Violations

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

print("\n" + "="*60)
print("EFFECT OF C PARAMETER")
print("="*60)
print(f"\n{'C':<10} {'Train Acc':<12} {'Test Acc':<12} {'# SV':<8} {'Margin':<10} {'Σξᵢ'}")
print("-"*70)

results = []
for C in C_values:
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train, y_train)
    
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)
    n_sv = svm.n_support_.sum()
    
    w = svm.coef_[0]
    margin = 2 / np.linalg.norm(w)
    
    # Estimate total slack
    margins_train = y_train * (X_train @ w + svm.intercept_[0])
    slack_train = np.maximum(0, 1 - margins_train)
    total_slack = slack_train.sum()
    
    print(f"{C:<10.2f} {train_acc*100:<12.1f} {test_acc*100:<12.1f} "
          f"{n_sv:<8} {margin:<10.3f} {total_slack:.2f}")
    
    results.append({
        'C': C, 'train_acc': train_acc, 'test_acc': test_acc,
        'n_sv': n_sv, 'margin': margin, 'slack': total_slack
    })

# Visualize decision boundaries
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for ax, C in zip(axes[:-1], C_values):
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train, y_train)
    
    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    xx = np.linspace(-3, 4, 100)
    ax.plot(xx, -(w[0]*xx + b)/w[1], 'k-', linewidth=2)
    ax.plot(xx, -(w[0]*xx + b - 1)/w[1], 'r--', linewidth=1, alpha=0.5)
    ax.plot(xx, -(w[0]*xx + b + 1)/w[1], 'b--', linewidth=1, alpha=0.5)
    
    ax.scatter(X_train[y_train==-1, 0], X_train[y_train==-1, 1],
               c='blue', s=50, alpha=0.6)
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
               c='red', s=50, marker='s', alpha=0.6)
    
    sv = svm.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
               edgecolors='gold', linewidths=2.5)
    
    margin = 2 / np.linalg.norm(w)
    ax.set_title(f'C = {C}\nMargin = {margin:.3f}, SV = {len(sv)}')
    ax.set_xlim(-3, 4); ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)

# Summary plot
ax = axes[-1]
ax.plot([r['C'] for r in results], [r['test_acc']*100 for r in results],
        'o-', linewidth=2, markersize=8, label='Test Accuracy')
ax.plot([r['C'] for r in results], [r['margin']*10 for r in results],
        's-', linewidth=2, markersize=8, label='Margin × 10')
ax.set_xscale('log')
ax.set_xlabel('C (log scale)')
ax.set_ylabel('Accuracy (%) / Margin × 10')
ax.set_title('Trade-off: Accuracy vs Margin')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Effect of C Parameter on Soft-Margin SVM\n'
             'Small C → wide margin, Large C → narrow margin',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('C_parameter_effect.png', dpi=150, bbox_inches='tight')
print("\nSaved: C_parameter_effect.png")
```

---

## Dual Formulation

### Deriving the Dual

Taking derivatives of the Lagrangian:

```
∂L/∂w = 0:   w = Σᵢ αᵢyᵢxᵢ           (same as hard margin)

∂L/∂b = 0:   Σᵢ αᵢyᵢ = 0             (same as hard margin)

∂L/∂ξᵢ = 0:  C - αᵢ - μᵢ = 0
             ⟹ αᵢ = C - μᵢ
             ⟹ αᵢ ≤ C    (since μᵢ ≥ 0)
```

### Soft-Margin Dual Problem

```
Maximize:    W(α) = Σᵢ αᵢ - ½ Σᵢ Σⱼ αᵢαⱼyᵢyⱼ(xᵢᵀxⱼ)

Subject to:  Σᵢ αᵢyᵢ = 0
             0 ≤ αᵢ ≤ C    ∀i

ONLY DIFFERENCE from hard margin: αᵢ bounded above by C
```

### KKT Conditions

```
1. Stationarity:
   w = Σᵢ αᵢyᵢxᵢ
   Σᵢ αᵢyᵢ = 0
   C - αᵢ - μᵢ = 0

2. Primal feasibility:
   yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
   ξᵢ ≥ 0

3. Dual feasibility:
   αᵢ ≥ 0
   μᵢ ≥ 0

4. Complementary slackness:
   αᵢ[yᵢ(wᵀxᵢ + b) - 1 + ξᵢ] = 0
   μᵢ ξᵢ = 0
```

### Three Types of Points

From complementary slackness:

```
αᵢ = 0:
  → μᵢ = C
  → ξᵢ = 0 (from μᵢ ξᵢ = 0)
  → Point is correctly classified outside margin
  → Not a support vector

0 < αᵢ < C:
  → 0 < μᵢ < C
  → ξᵢ = 0
  → yᵢ(wᵀxᵢ + b) = 1
  → Point is exactly on margin boundary
  → Free support vector

αᵢ = C:
  → μᵢ = 0
  → ξᵢ ≥ 0 (can be > 0)
  → yᵢ(wᵀxᵢ + b) = 1 - ξᵢ ≤ 1
  → Point violates margin (inside margin or misclassified)
  → Bounded support vector
```

```python
# Analyze the three types
print("\n" + "="*60)
print("THREE TYPES OF POINTS IN SOFT-MARGIN SVM")
print("="*60)

svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]

# Get alphas
alpha_times_y = svm.dual_coef_.ravel()
alpha = np.abs(alpha_times_y)

# Classify all points
margins = y * (X @ w + b)

# Type 1: αᵢ = 0 (non-support vectors)
non_sv_mask = np.ones(len(X), dtype=bool)
non_sv_mask[svm.support_] = False

# Type 2: 0 < αᵢ < C (free support vectors)
free_sv_mask = (alpha > 1e-4) & (alpha < svm.C - 1e-4)

# Type 3: αᵢ = C (bounded support vectors)
bounded_sv_mask = alpha >= svm.C - 1e-4

print(f"\nC = {svm.C}")
print(f"\nType 1 (αᵢ=0): {non_sv_mask.sum()} points")
print(f"  Not support vectors")
print(f"  Correctly classified outside margin")

print(f"\nType 2 (0<αᵢ<C): {free_sv_mask.sum()} points")
print(f"  Free support vectors")
print(f"  Exactly on margin boundary")
print(f"  Margins: {margins[svm.support_][free_sv_mask].round(4)}")

print(f"\nType 3 (αᵢ=C): {bounded_sv_mask.sum()} points")
print(f"  Bounded support vectors")
print(f"  Inside margin or misclassified")
print(f"  Margins: {margins[svm.support_][bounded_sv_mask].round(4)}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 7))

# Plot decision boundary
xx = np.linspace(-3, 4, 100)
ax.plot(xx, -(w[0]*xx + b)/w[1], 'k-', linewidth=2, label='Decision boundary')
ax.plot(xx, -(w[0]*xx + b - 1)/w[1], 'k--', linewidth=1, alpha=0.5)
ax.plot(xx, -(w[0]*xx + b + 1)/w[1], 'k--', linewidth=1, alpha=0.5)

# Plot points by type
# Non-SV
ax.scatter(X[non_sv_mask & (y==-1), 0], X[non_sv_mask & (y==-1), 1],
           c='lightblue', s=50, alpha=0.5, label='Non-SV (α=0)')
ax.scatter(X[non_sv_mask & (y==1), 0], X[non_sv_mask & (y==1), 1],
           c='lightcoral', s=50, marker='s', alpha=0.5)

# Free SV
sv_indices = svm.support_
free_sv_indices = sv_indices[free_sv_mask]
ax.scatter(X[free_sv_indices & (y==-1), 0], X[free_sv_indices & (y==-1), 1],
           c='blue', s=100, edgecolors='gold', linewidths=2,
           label='Free SV (0<α<C)')
ax.scatter(X[free_sv_indices & (y==1), 0], X[free_sv_indices & (y==1), 1],
           c='red', s=100, marker='s', edgecolors='gold', linewidths=2)

# Bounded SV
bounded_sv_indices = sv_indices[bounded_sv_mask]
ax.scatter(X[bounded_sv_indices & (y==-1), 0], X[bounded_sv_indices & (y==-1), 1],
           c='blue', s=100, edgecolors='purple', linewidths=3,
           label='Bounded SV (α=C)')
ax.scatter(X[bounded_sv_indices & (y==1), 0], X[bounded_sv_indices & (y==1), 1],
           c='red', s=100, marker='s', edgecolors='purple', linewidths=3)

ax.set_xlim(-3, 4); ax.set_ylim(-3, 3)
ax.legend(fontsize=9)
ax.set_title('Three Types of Points in Soft-Margin SVM\n'
             'Gold = free SV (on margin), Purple = bounded SV (violation)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('three_point_types.png', dpi=150, bbox_inches='tight')
print("\nSaved: three_point_types.png")
```

---

## Hinge Loss Interpretation

### Alternative View: Loss + Regularization

The soft-margin SVM can be rewritten as:

```
Minimize:  λ||w||² + Σᵢ max(0, 1 - yᵢ(wᵀxᵢ + b))
           ↑          ↑
        Regularizer  Hinge Loss

Where: λ = 1/(2C)

Hinge Loss: L(yᵢ, f(xᵢ)) = max(0, 1 - yᵢf(xᵢ))

Properties:
  yᵢf(xᵢ) ≥ 1:  Loss = 0 (correct, outside margin)
  yᵢf(xᵢ) < 1:  Loss = 1 - yᵢf(xᵢ) (proportional to violation)
```

### Comparison with Other Losses

```python
# Visualize different loss functions
z = np.linspace(-2, 3, 300)  # z = yf(x)

# Different losses
hinge_loss = np.maximum(0, 1 - z)
zero_one_loss = (z < 0).astype(float)
logistic_loss = np.log(1 + np.exp(-z))
squared_hinge = np.maximum(0, 1 - z)**2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Loss functions
ax = axes[0]
ax.plot(z, hinge_loss, linewidth=2.5, label='Hinge (SVM)')
ax.plot(z, zero_one_loss, linewidth=2.5, label='0-1 Loss')
ax.plot(z, logistic_loss, linewidth=2.5, label='Logistic')
ax.plot(z, squared_hinge, linewidth=2.5, label='Squared Hinge')

ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Decision boundary')
ax.axvline(x=1, color='orange', linestyle='--', alpha=0.5, label='Margin')
ax.set_xlabel('yf(x) = y(wᵀx + b)', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Loss Functions Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 3)

# Panel 2: Hinge loss detail
ax = axes[1]
ax.plot(z, hinge_loss, 'b-', linewidth=3)
ax.fill_between(z, 0, hinge_loss, alpha=0.3, color='blue')

# Annotate regions
ax.text(-1, 1.5, 'Misclassified\nLoss > 1', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
ax.text(0.3, 0.5, 'Margin\nviolation\n0 < Loss < 1', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
ax.text(2, 0.1, 'Correct\nLoss = 0', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(x=1, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('yf(x)', fontsize=12)
ax.set_ylabel('Hinge Loss', fontsize=12)
ax.set_title('Hinge Loss Breakdown')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 3)

plt.suptitle('Hinge Loss: max(0, 1 - yf(x))', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('hinge_loss.png', dpi=150, bbox_inches='tight')
print("\nSaved: hinge_loss.png")

print("\n" + "="*60)
print("HINGE LOSS PROPERTIES")
print("="*60)
print("""
Hinge Loss: L(y, f(x)) = max(0, 1 - yf(x))

Advantages:
  ✓ Convex (easy to optimize)
  ✓ Sparsity (zero loss for correct points outside margin)
  ✓ Margin-aware (penalizes points near boundary)
  ✓ Upper bound on 0-1 loss

Comparison:
  vs 0-1 Loss:      Convex approximation (0-1 is non-convex)
  vs Logistic:      Hinge reaches zero, logistic never does
  vs Squared Hinge: Hinge is more robust to outliers
""")
```

---

## Support Vectors in Soft-Margin SVM

### Definition

In soft-margin SVM:

```
Support vectors are points with αᵢ > 0

Two types:
1. Free SV:    0 < αᵢ < C  (on margin boundary, ξᵢ = 0)
2. Bounded SV: αᵢ = C      (violate margin, ξᵢ > 0)

All support vectors contribute to the decision boundary:
  f(x) = Σᵢ αᵢyᵢxᵢᵀx + b
       = Σ_{i∈SV} αᵢyᵢxᵢᵀx + b
```

### Computing Bias b

```
Use free support vectors (0 < αᵢ < C):

For any free SV with index s:
  ξₛ = 0
  yₛ(wᵀxₛ + b) = 1
  
  ⟹ b = yₛ - wᵀxₛ
      = yₛ - Σᵢ αᵢyᵢxᵢᵀxₛ

Average over all free SVs for numerical stability:
  b = (1/|FSV|) Σₛ∈FSV [yₛ - Σᵢ αᵢyᵢxᵢᵀxₛ]
```

---

## Choosing C

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

print("\n" + "="*60)
print("CHOOSING C: GRID SEARCH")
print("="*60)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(
    SVC(kernel='linear'),
    param_grid,
    cv=5,
    scoring='accuracy',
    return_train_score=True
)

grid.fit(X, y)

print(f"\nBest C: {grid.best_params_['C']}")
print(f"Best CV score: {grid.best_score_*100:.2f}%")

# Show all results
results = grid.cv_results_
print(f"\n{'C':<10} {'Mean CV Acc':<15} {'Std CV Acc':<15}")
print("-"*40)
for i, C in enumerate(param_grid['C']):
    mean = results['mean_test_score'][i]
    std = results['std_test_score'][i]
    print(f"{C:<10} {mean*100:<15.2f} {std*100:<15.2f}")
```

### Validation Curve

```python
from sklearn.model_selection import validation_curve

C_range = np.logspace(-3, 3, 20)

train_scores, val_scores = validation_curve(
    SVC(kernel='linear'), X, y,
    param_name='C', param_range=C_range,
    cv=5, scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.semilogx(C_range, train_mean, 'o-', linewidth=2, label='Training')
plt.fill_between(C_range, train_mean - train_std, train_mean + train_std,
                 alpha=0.2)

plt.semilogx(C_range, val_mean, 's-', linewidth=2, label='Validation')
plt.fill_between(C_range, val_mean - val_std, val_mean + val_std,
                 alpha=0.2)

plt.xlabel('C (log scale)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Validation Curve: Choosing C\n'
          'Underfit ← Optimal → Overfit', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_curve_C.png', dpi=150, bbox_inches='tight')
print("\nSaved: validation_curve_C.png")

optimal_C = C_range[np.argmax(val_mean)]
print(f"\nOptimal C from validation curve: {optimal_C:.4f}")
```

---

## Summary

### Key Concepts

```
SOFT-MARGIN SVM:

Motivation:
  Real data is noisy and rarely perfectly separable

Solution:
  Introduce slack variables ξᵢ ≥ 0 to allow violations

Optimization:
  Minimize:   ½||w||² + C Σᵢ ξᵢ
  Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0

C Parameter:
  Controls trade-off between margin and violations
  Large C: Narrow margin, fewer violations
  Small C: Wide margin, more violations allowed

Dual Problem:
  Maximize:   Σαᵢ - ½ΣΣ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
  Subject to: Σαᵢyᵢ = 0,  0 ≤ αᵢ ≤ C

Hinge Loss:
  L = max(0, 1 - yf(x))
  Convex, sparse, margin-aware

Three Types of Points:
  αᵢ = 0:     Non-SV (correctly classified, outside margin)
  0 < αᵢ < C: Free SV (on margin boundary)
  αᵢ = C:     Bounded SV (inside margin or misclassified)
```

### Comparison: Hard vs Soft

```
╔════════════════════════╦════════════════════╦═══════════════════╗
║ Aspect                 ║ Hard Margin        ║ Soft Margin       ║
╠════════════════════════╬════════════════════╬═══════════════════╣
║ Data requirement       ║ Linearly separable ║ Any data          ║
║ Constraint             ║ yᵢ(wᵀxᵢ+b) ≥ 1     ║ yᵢ(wᵀxᵢ+b)≥1-ξᵢ   ║
║ Slack variables        ║ No                 ║ Yes (ξᵢ ≥ 0)      ║
║ Dual constraint        ║ αᵢ ≥ 0             ║ 0 ≤ αᵢ ≤ C        ║
║ Hyperparameter         ║ None               ║ C                 ║
║ Robustness to outliers ║ Poor               ║ Good              ║
║ Solution exists        ║ Only if separable  ║ Always            ║
║ Practical use          ║ Rarely             ║ Standard          ║
╚════════════════════════╩════════════════════╩═══════════════════╝
```

### What's Next

```
Soft-Margin SVM + Kernel Trick (Chapter 09)
  → Handle non-linear patterns
  → Replace xᵢᵀxⱼ with K(xᵢ, xⱼ) in dual
  → Most powerful SVM formulation
```

---

## Quick Reference

```
╔══════════════════════════════════════════════════════════════╗
║            SOFT-MARGIN SVM — QUICK REFERENCE                 ║
╠══════════════════════════════════════════════════════════════╣
║  PRIMAL PROBLEM                                              ║
║    min  ½||w||² + C Σξᵢ                                      ║
║    s.t. yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0                      ║
╠══════════════════════════════════════════════════════════════╣
║  DUAL PROBLEM                                                ║
║    max  Σαᵢ - ½ΣΣ αᵢαⱼyᵢyⱼxᵢᵀxⱼ                            ║
║    s.t. Σαᵢyᵢ = 0,  0 ≤ αᵢ ≤ C                              ║
╠══════════════════════════════════════════════════════════════╣
║  SLACK VARIABLES                                             ║
║    ξᵢ = 0:     Correct, outside margin                       ║
║    0 < ξᵢ < 1: Inside margin, correct side                   ║
║    ξᵢ ≥ 1:     Misclassified                                 ║
╠══════════════════════════════════════════════════════════════╣
║  C PARAMETER                                                 ║
║    Large C → narrow margin, few violations                   ║
║    Small C → wide margin, allow violations                   ║
║    Choose via cross-validation                               ║
╠══════════════════════════════════════════════════════════════╣
║  HINGE LOSS                                                  ║
║    L = max(0, 1 - yf(x))                                     ║
║    Equivalent to: λ||w||² + ΣL(yᵢ,f(xᵢ))                    ║
╠══════════════════════════════════════════════════════════════╣
║  SUPPORT VECTORS                                             ║
║    Free SV:    0 < αᵢ < C  (on margin)                       ║
║    Bounded SV: αᵢ = C      (violate margin)                  ║
╠══════════════════════════════════════════════════════════════╣
║  SKLEARN                                                     ║
║    SVC(kernel='linear', C=1.0)                               ║
║    Use GridSearchCV to tune C                                ║
╚══════════════════════════════════════════════════════════════╝
```

---

*End of 08_SVM_Soft_Margin.md*