# Radial Basis Functions (RBF)

## Table of Contents
1. [What Are Radial Basis Functions?](#what-are-radial-basis-functions)
2. [The RBF Kernel in Detail](#the-rbf-kernel-in-detail)
3. [The Gamma Parameter](#the-gamma-parameter)
4. [RBF vs Other Kernels](#rbf-vs-other-kernels)
5. [RBF Networks](#rbf-networks)
6. [Gaussian Processes with RBF](#gaussian-processes-with-rbf)
7. [Choosing Gamma: Practical Strategies](#choosing-gamma-practical-strategies)
8. [RBF for Interpolation](#rbf-for-interpolation)
9. [Computational Considerations](#computational-considerations)
10. [Summary and Quick Reference](#summary-and-quick-reference)

---

## What Are Radial Basis Functions?

### Definition

A **radial basis function** is any function φ: ℝ → ℝ whose value depends only on the distance from a center point:

```
φ(x) = φ(||x - c||)

Where:
  x = input point
  c = center
  ||·|| = distance (typically Euclidean)

"Radial" = depends only on radius (distance)
"Basis" = can be used as basis functions for function approximation
```

### Common RBF Types

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# RADIAL BASIS FUNCTIONS: COMMON TYPES
# ============================================================

def rbf_gaussian(r, epsilon=1.0):
    """Gaussian RBF: φ(r) = exp(-ε²r²)"""
    return np.exp(-epsilon**2 * r**2)

def rbf_multiquadric(r, epsilon=1.0):
    """Multiquadric: φ(r) = √(1 + (εr)²)"""
    return np.sqrt(1 + (epsilon * r)**2)

def rbf_inverse_multiquadric(r, epsilon=1.0):
    """Inverse Multiquadric: φ(r) = 1/√(1 + (εr)²)"""
    return 1.0 / np.sqrt(1 + (epsilon * r)**2)

def rbf_inverse_quadratic(r, epsilon=1.0):
    """Inverse Quadratic: φ(r) = 1/(1 + (εr)²)"""
    return 1.0 / (1 + (epsilon * r)**2)

def rbf_thin_plate_spline(r):
    """Thin Plate Spline: φ(r) = r² log(r)"""
    r_safe = np.maximum(r, 1e-10)  # Avoid log(0)
    return r_safe**2 * np.log(r_safe)

def rbf_linear(r):
    """Linear: φ(r) = r"""
    return r

def rbf_cubic(r):
    """Cubic: φ(r) = r³"""
    return r**3


print("="*70)
print("RADIAL BASIS FUNCTIONS: FUNCTION FAMILY")
print("="*70)

r_vals = np.linspace(0, 3, 300)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

rbf_functions = [
    ("Gaussian\nφ(r) = exp(-ε²r²)", rbf_gaussian, {'epsilon': 1.0}),
    ("Multiquadric\nφ(r) = √(1+(εr)²)", rbf_multiquadric, {'epsilon': 1.0}),
    ("Inverse Multiquadric\nφ(r) = 1/√(1+(εr)²)", rbf_inverse_multiquadric, {'epsilon': 1.0}),
    ("Inverse Quadratic\nφ(r) = 1/(1+(εr)²)", rbf_inverse_quadratic, {'epsilon': 1.0}),
    ("Thin Plate Spline\nφ(r) = r²log(r)", rbf_thin_plate_spline, {}),
    ("Linear\nφ(r) = r", rbf_linear, {}),
    ("Cubic\nφ(r) = r³", rbf_cubic, {}),
]

for ax, (name, rbf_fn, params) in zip(axes, rbf_functions):
    y_vals = rbf_fn(r_vals, **params)
    ax.plot(r_vals, y_vals, linewidth=2.5, color='steelblue')
    ax.set_xlabel('Distance r')
    ax.set_ylabel('φ(r)')
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

# Summary panel
ax = axes[-1]
ax.text(0.5, 0.5,
        "RBF PROPERTIES:\n\n"
        "Gaussian:\n"
        "• Most common\n"
        "• Smooth, local\n"
        "• Infinitely differentiable\n\n"
        "Multiquadric:\n"
        "• Global influence\n"
        "• Good interpolation\n\n"
        "Thin Plate:\n"
        "• Smoothest interpolant\n"
        "• Used in image warping",
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.axis('off')

plt.suptitle('Radial Basis Functions: Function Family\n'
             'All depend only on distance r = ||x - c||',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('rbf_function_family.png', dpi=150, bbox_inches='tight')
print("Saved: rbf_function_family.png")


# ============================================================
# 2D VISUALIZATION: Radial symmetry
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Create 2D grid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Center at origin
center = np.array([0.0, 0.0])
R = np.sqrt(X**2 + Y**2)  # Distance from center

rbfs_2d = [
    ("Gaussian\nε=1", rbf_gaussian, {'epsilon': 1.0}),
    ("Inverse Quadratic\nε=1", rbf_inverse_quadratic, {'epsilon': 1.0}),
    ("Multiquadric\nε=1", rbf_multiquadric, {'epsilon': 1.0}),
]

for ax, (name, rbf_fn, params) in zip(axes, rbfs_2d):
    Z = rbf_fn(R, **params)
    
    cf = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax.contour(X, Y, Z, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    plt.colorbar(cf, ax=ax)
    ax.plot([0], [0], 'r*', markersize=15, label='Center c')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(name)
    ax.legend()
    ax.axis('equal')

plt.suptitle('RBF in 2D: Radial Symmetry\n'
             'Circular contours → depends only on ||x - c||',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('rbf_2d_symmetry.png', dpi=150, bbox_inches='tight')
print("Saved: rbf_2d_symmetry.png")
```

---

## The RBF Kernel in Detail

### Mathematical Definition

The **RBF kernel** (also called Gaussian kernel) is:

```
K(x, x') = exp(-γ ||x - x'||²)

Where:
  γ (gamma) > 0    is the bandwidth parameter
  ||x - x'||²      is the squared Euclidean distance

Alternative parameterizations:
  K(x, x') = exp(-||x - x'||² / (2σ²))    where γ = 1/(2σ²)
  K(x, x') = exp(-r² / l²)                where γ = 1/l²
```

### Why It's Called "RBF Kernel"

```
RBF Kernel is actually a radial function:

K(x, x') = φ(||x - x'||)    where φ(r) = exp(-γr²)

The value depends ONLY on the distance between x and x'
→ Stationary kernel
→ Translation-invariant
→ Rotation-invariant
```

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# RBF KERNEL: DETAILED PROPERTIES
# ============================================================

def rbf_kernel(x, xp, gamma=1.0):
    """Standard RBF kernel."""
    diff = x - xp
    return np.exp(-gamma * np.dot(diff, diff))


print("="*70)
print("RBF KERNEL: MATHEMATICAL PROPERTIES")
print("="*70)

# Property 1: Value at identical points
print("\n1. SELF-SIMILARITY")
x = np.array([1.0, 2.0, 3.0])
print(f"   K(x, x) = {rbf_kernel(x, x, gamma=1.0):.6f}")
print(f"   Always equals 1.0 (normalized)")

# Property 2: Decay with distance
print("\n2. DECAY WITH DISTANCE")
x_center = np.array([0.0, 0.0])
distances = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

print(f"   {'Distance':<12} {'K(0, x)':<15} {'% of max'}")
print(f"   {'-'*40}")
for d in distances:
    x_test = np.array([d, 0.0])
    k_val  = rbf_kernel(x_center, x_test, gamma=1.0)
    pct    = k_val * 100
    print(f"   {d:<12.1f} {k_val:<15.6f} {pct:>6.1f}%")

print("\n   → Exponential decay: farther points have less influence")

# Property 3: Smoothness
print("\n3. SMOOTHNESS")
print("   RBF kernel is C^∞ (infinitely differentiable)")
print("   → Produces smooth decision boundaries")
print("   → No sharp corners or discontinuities")

# Property 4: Bounded
print("\n4. BOUNDED")
print("   0 < K(x, x') ≤ 1  for all x, x'")
print("   Maximum at x = x' (K = 1)")
print("   Approaches 0 as ||x - x'|| → ∞")

# Property 5: Positive definite
print("\n5. POSITIVE DEFINITE")
print("   RBF kernel is ALWAYS a valid kernel")
print("   → Kernel matrix is always PSD")
print("   → Guaranteed for any γ > 0")


# ============================================================
# VISUALIZE: RBF kernel properties
# ============================================================

fig = plt.figure(figsize=(16, 5))

# Panel 1: Decay curve for different gamma
ax1 = fig.add_subplot(131)
r_vals = np.linspace(0, 4, 200)
gammas = [0.1, 0.5, 1.0, 2.0, 5.0]

for gamma in gammas:
    k_vals = np.exp(-gamma * r_vals**2)
    ax1.plot(r_vals, k_vals, linewidth=2, label=f'γ={gamma}')

ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
            label='50% threshold')
ax1.set_xlabel('Distance ||x - x\'||')
ax1.set_ylabel('K(x, x\')')
ax1.set_title('RBF Kernel Decay\nLarger γ → faster decay')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: 2D cross-section
ax2 = fig.add_subplot(132, projection='3d')

x_range = np.linspace(-3, 3, 50)
X_grid, Y_grid = np.meshgrid(x_range, x_range)
Z_grid = np.exp(-1.0 * (X_grid**2 + Y_grid**2))

surf = ax2.plot_surface(X_grid, Y_grid, Z_grid,
                        cmap='viridis', alpha=0.9)
ax2.set_xlabel('x₁ - x\'₁')
ax2.set_ylabel('x₂ - x\'₂')
ax2.set_zlabel('K(x, x\')')
ax2.set_title('RBF Kernel Surface\nγ=1.0')

# Panel 3: Effective range
ax3 = fig.add_subplot(133)

# For different gamma, compute "effective range" (distance where K drops to 0.05)
gammas_range = np.logspace(-1, 1, 50)
effective_ranges = np.sqrt(-np.log(0.05) / gammas_range)

ax3.semilogx(gammas_range, effective_ranges, linewidth=2.5)
ax3.set_xlabel('γ (gamma)')
ax3.set_ylabel('Effective Range\n(distance where K=0.05)')
ax3.set_title('RBF Effective Range vs γ\nLarger γ → smaller range')
ax3.grid(True, alpha=0.3, which='both')
ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

plt.suptitle('RBF Kernel Properties', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rbf_kernel_properties.png', dpi=150, bbox_inches='tight')
print("\nSaved: rbf_kernel_properties.png")


# ============================================================
# KERNEL MATRIX ANALYSIS
# ============================================================

print("\n" + "="*70)
print("RBF KERNEL MATRIX PROPERTIES")
print("="*70)

np.random.seed(42)
X_rbf = np.random.randn(20, 3)

def rbf_kernel_matrix(X, gamma=1.0):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i,j] = rbf_kernel(X[i], X[j], gamma)
    return K

K_rbf = rbf_kernel_matrix(X_rbf, gamma=1.0)

# Analyze
eigvals = np.linalg.eigvalsh(K_rbf)
diag = np.diag(K_rbf)
off_diag = K_rbf[~np.eye(len(X_rbf), dtype=bool)]

print(f"\nKernel matrix shape: {K_rbf.shape}")
print(f"\nDiagonal (self-similarity):")
print(f"  All K(xᵢ,xᵢ) = 1.0: {np.allclose(diag, 1.0)}")
print(f"  Min: {diag.min():.6f}, Max: {diag.max():.6f}")

print(f"\nOff-diagonal (cross-similarity):")
print(f"  Range: [{off_diag.min():.6f}, {off_diag.max():.6f}]")
print(f"  Mean: {off_diag.mean():.6f}")

print(f"\nEigenvalues:")
print(f"  Min: {eigvals.min():.6f}  (> 0 → PSD ✓)")
print(f"  Max: {eigvals.max():.6f}")
print(f"  Condition number: {eigvals.max() / eigvals.min():.2e}")

print(f"\nFrobenius norm: {np.linalg.norm(K_rbf, 'fro'):.4f}")
print(f"Trace (sum of eigenvalues): {np.trace(K_rbf):.4f}")
```

---

## The Gamma Parameter

### What Gamma Controls

```
γ (gamma) is the MOST IMPORTANT hyperparameter for RBF kernels

Small γ:
  - Wide Gaussian bell
  - Each training point influences far-away test points
  - Smooth, global decision boundary
  - Risk: UNDERFIT (too simple)

Large γ:
  - Narrow Gaussian bell
  - Each training point influences only nearby test points
  - Complex, local decision boundary
  - Risk: OVERFIT (memorizes training data)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# ============================================================
# GAMMA: COMPLETE SENSITIVITY ANALYSIS
# ============================================================

print("="*70)
print("GAMMA PARAMETER: COMPLETE ANALYSIS")
print("="*70)

np.random.seed(42)
X_gamma, y_gamma = make_moons(n_samples=200, noise=0.2, random_state=42)
X_gamma = StandardScaler().fit_transform(X_gamma)

gammas_test = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

print(f"\n{'γ':<8} {'Train Acc':<12} {'# SV':<8} {'Margin Width*':<15} {'Status'}")
print("-"*60)

results_gamma = []

for gamma in gammas_test:
    clf = SVC(kernel='rbf', gamma=gamma, C=1.0)
    clf.fit(X_gamma, y_gamma)
    
    train_acc = clf.score(X_gamma, y_gamma) * 100
    n_sv = clf.n_support_.sum()
    
    # Estimate "effective margin width" from support vector spread
    sv_coords = clf.support_vectors_
    sv_spread = np.std(sv_coords)
    
    # Classify behavior
    if gamma < 0.1:
        status = "Underfit"
    elif gamma > 5.0:
        status = "Overfit"
    else:
        status = "Good"
    
    results_gamma.append({
        'gamma': gamma,
        'acc': train_acc,
        'n_sv': n_sv,
        'spread': sv_spread,
        'status': status
    })
    
    print(f"{gamma:<8.2f} {train_acc:<12.1f} {n_sv:<8} {sv_spread:<15.3f} {status}")


# ============================================================
# VISUALIZE: Effect of gamma on decision boundary
# ============================================================

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.ravel()

for ax, gamma in zip(axes, gammas_test):
    clf = SVC(kernel='rbf', gamma=gamma, C=1.0)
    clf.fit(X_gamma, y_gamma)
    
    # Create mesh
    xx, yy = np.meshgrid(
        np.linspace(X_gamma[:,0].min()-0.5, X_gamma[:,0].max()+0.5, 200),
        np.linspace(X_gamma[:,1].min()-0.5, X_gamma[:,1].max()+0.5, 200)
    )
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='k', linewidths=1, alpha=0.5)
    
    ax.scatter(X_gamma[y_gamma==0, 0], X_gamma[y_gamma==0, 1],
               c='royalblue', s=25, edgecolors='k', linewidths=0.3, alpha=0.8)
    ax.scatter(X_gamma[y_gamma==1, 0], X_gamma[y_gamma==1, 1],
               c='tomato', s=25, marker='s',
               edgecolors='k', linewidths=0.3, alpha=0.8)
    
    # Highlight support vectors
    sv = clf.support_vectors_
    ax.scatter(sv[:,0], sv[:,1], s=150, facecolors='none',
               edgecolors='gold', linewidths=2.5, zorder=5)
    
    acc = clf.score(X_gamma, y_gamma) * 100
    n_sv = clf.n_support_.sum()
    
    ax.set_title(f'γ = {gamma}\nAcc={acc:.0f}%, SV={n_sv}')
    ax.set_xlim(X_gamma[:,0].min()-0.5, X_gamma[:,0].max()+0.5)
    ax.set_ylim(X_gamma[:,1].min()-0.5, X_gamma[:,1].max()+0.5)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of γ on RBF-SVM Decision Boundary\n'
             'Top-left (small γ) = smooth, global → Bottom-right (large γ) = complex, local',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('gamma_sensitivity.png', dpi=150, bbox_inches='tight')
print("\nSaved: gamma_sensitivity.png")


# ============================================================
# GAMMA vs ACCURACY: LEARNING CURVE
# ============================================================

from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(
    X_gamma, y_gamma, test_size=0.3, random_state=42
)

gammas_fine = np.logspace(-2, 2, 50)
train_accs = []
test_accs = []
n_svs = []

for gamma in gammas_fine:
    clf = SVC(kernel='rbf', gamma=gamma, C=1.0)
    clf.fit(X_tr, y_tr)
    train_accs.append(clf.score(X_tr, y_tr))
    test_accs.append(clf.score(X_te, y_te))
    n_svs.append(clf.n_support_.sum())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy vs gamma
ax = axes[0]
ax.semilogx(gammas_fine, train_accs, 'b-', linewidth=2.5, label='Train')
ax.semilogx(gammas_fine, test_accs, 'r--', linewidth=2.5, label='Test')
ax.set_xlabel('γ (gamma)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy vs γ\n(Underfit ← optimal → Overfit)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
ax.axvline(x=gammas_fine[np.argmax(test_accs)],
           color='green', linestyle=':', linewidth=2,
           label=f'Best γ = {gammas_fine[np.argmax(test_accs)]:.3f}')

# Number of support vectors vs gamma
ax = axes[1]
ax.semilogx(gammas_fine, n_svs, 'purple', linewidth=2.5)
ax.set_xlabel('γ (gamma)', fontsize=12)
ax.set_ylabel('# Support Vectors', fontsize=12)
ax.set_title('Support Vectors vs γ\nMore SV = simpler model (less overfitting)')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('gamma_learning_curve.png', dpi=150, bbox_inches='tight')
print("Saved: gamma_learning_curve.png")

print(f"\nOptimal γ: {gammas_fine[np.argmax(test_accs)]:.4f}")
print(f"Best test accuracy: {max(test_accs)*100:.2f}%")
```

### Gamma Interpretation

```python
# ============================================================
# GAMMA: THREE INTERPRETATIONS
# ============================================================

print("\n" + "="*70)
print("GAMMA: THREE EQUIVALENT INTERPRETATIONS")
print("="*70)

gamma_val = 1.0

# Interpretation 1: Bandwidth
sigma = 1 / np.sqrt(2 * gamma_val)
print(f"\n1. BANDWIDTH PARAMETER")
print(f"   γ = {gamma_val}")
print(f"   K(x,x') = exp(-γ||x-x'||²)")
print(f"   Equivalent σ = 1/√(2γ) = {sigma:.4f}")
print(f"   → Gaussian with standard deviation σ")

# Interpretation 2: Length scale
length_scale = 1 / np.sqrt(gamma_val)
print(f"\n2. LENGTH SCALE")
print(f"   l = 1/√γ = {length_scale:.4f}")
print(f"   K(x,x') = exp(-||x-x'||²/l²)")
print(f"   → Points farther than l have low correlation")

# Interpretation 3: Inverse reach
print(f"\n3. INVERSE REACH")
print(f"   γ controls how far influence extends:")

distances_test = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
print(f"\n   {'Distance':<12} {'K(x,x\')':<15} {'Influence'}")
print(f"   {'-'*45}")
for d in distances_test:
    k_val = np.exp(-gamma_val * d**2)
    influence = "Strong" if k_val > 0.5 else "Moderate" if k_val > 0.1 else "Weak"
    print(f"   {d:<12.1f} {k_val:<15.4f} {influence}")

print(f"\n   Effective radius (K > 0.05): {np.sqrt(-np.log(0.05)/gamma_val):.3f}")
```

---

## RBF vs Other Kernels

### Head-to-Head Comparison

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import time

# ============================================================
# RBF VS OTHER KERNELS: COMPREHENSIVE COMPARISON
# ============================================================

print("="*70)
print("RBF VS OTHER KERNELS")
print("="*70)

datasets_comp = {
    'Linear Sep': make_classification(n_samples=200, n_features=2,
                                      n_redundant=0, n_informative=2,
                                      class_sep=2.5, random_state=42),
    'Moons': make_moons(n_samples=200, noise=0.2, random_state=42),
    'Circles': make_circles(n_samples=200, noise=0.1,
                            factor=0.4, random_state=42),
}

kernels_comp = {
    'Linear':    SVC(kernel='linear', C=1.0),
    'Poly d=2':  SVC(kernel='poly', degree=2, coef0=1, C=1.0),
    'Poly d=3':  SVC(kernel='poly', degree=3, coef0=1, C=1.0),
    'RBF γ=auto': SVC(kernel='rbf', gamma='auto', C=1.0),
    'RBF γ=scale': SVC(kernel='rbf', gamma='scale', C=1.0),
    'Sigmoid':   SVC(kernel='sigmoid', gamma=0.1, coef0=0, C=1.0),
}

print(f"\n{'Dataset':<15} {'Kernel':<15} {'CV Acc':<12} {'Train Time':<12} {'# SV'}")
print("-"*65)

results_comp = {}

for ds_name, (X_ds, y_ds) in datasets_comp.items():
    X_s = StandardScaler().fit_transform(X_ds)
    results_comp[ds_name] = {}
    
    for k_name, clf in kernels_comp.items():
        # Cross-validation score
        scores = cross_val_score(clf, X_s, y_ds, cv=5)
        mean_score = scores.mean()
        
        # Training time
        t0 = time.perf_counter()
        clf.fit(X_s, y_ds)
        train_time = time.perf_counter() - t0
        
        n_sv = clf.n_support_.sum() if hasattr(clf, 'n_support_') else 0
        
        results_comp[ds_name][k_name] = {
            'cv_acc': mean_score,
            'time': train_time,
            'n_sv': n_sv
        }
        
        print(f"{ds_name:<15} {k_name:<15} {mean_score*100:<12.1f} "
              f"{train_time*1000:<12.2f} {n_sv}")


# ============================================================
# SUMMARY: When RBF wins
# ============================================================

print("\n" + "="*70)
print("WHEN RBF KERNEL WINS")
print("="*70)

summary = """
RBF is the BEST choice when:
  ✓ Data has non-linear patterns (moons, circles, spirals)
  ✓ No prior knowledge about data structure
  ✓ Need universal approximation capability
  ✓ Want smooth, continuous boundaries
  ✓ Can afford to tune gamma via cross-validation

RBF may NOT be the best when:
  ✗ Data is linearly separable (linear kernel faster + simpler)
  ✗ Very high-dimensional text data (linear often better)
  ✗ Need polynomial interactions specifically (use poly kernel)
  ✗ Very large datasets (>100K samples, RBF becomes slow)

RULE OF THUMB:
  Start with RBF (gamma='scale'), tune if needed
  Try linear kernel as baseline
  If RBF doesn't improve much → use linear (simpler)
"""

print(summary)


# ============================================================
# VISUALIZATION: RBF vs Others on Each Dataset
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for row, (ds_name, (X_ds, y_ds)) in enumerate(datasets_comp.items()):
    X_s = StandardScaler().fit_transform(X_ds)
    
    kernels_viz = [
        ('Linear', SVC(kernel='linear', C=1.0)),
        ('Poly d=2', SVC(kernel='poly', degree=2, coef0=1, C=1.0)),
        ('RBF γ=0.5', SVC(kernel='rbf', gamma=0.5, C=1.0)),
        ('RBF γ=1.0', SVC(kernel='rbf', gamma=1.0, C=1.0)),
    ]
    
    for col, (k_name, clf) in enumerate(kernels_viz):
        ax = axes[row, col]
        clf.fit(X_s, y_ds)
        
        xx, yy = np.meshgrid(
            np.linspace(X_s[:,0].min()-0.5, X_s[:,0].max()+0.5, 100),
            np.linspace(X_s[:,1].min()-0.5, X_s[:,1].max()+0.5, 100)
        )
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlBu')
        ax.scatter(X_s[y_ds==0,0], X_s[y_ds==0,1], c='royalblue',
                   s=20, edgecolors='k', linewidths=0.3)
        ax.scatter(X_s[y_ds==1,0], X_s[y_ds==1,1], c='tomato',
                   s=20, marker='s', edgecolors='k', linewidths=0.3)
        
        acc = clf.score(X_s, y_ds) * 100
        ax.set_title(f'{k_name}\nAcc={acc:.0f}%')
        ax.grid(True, alpha=0.3)
        
        if col == 0:
            ax.set_ylabel(ds_name, fontsize=12, fontweight='bold')

plt.suptitle('RBF vs Other Kernels: Decision Boundaries\n'
             'RBF excels on non-linear patterns',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('rbf_vs_others.png', dpi=150, bbox_inches='tight')
print("\nSaved: rbf_vs_others.png")
```

---

## RBF Networks

### What Is an RBF Network?

An **RBF Network** is a neural network architecture that uses radial basis functions as activation functions:

```
RBF Network Structure:

Input x → Hidden Layer (RBF neurons) → Output Layer (linear)
          φ₁(||x - c₁||)
          φ₂(||x - c₂||)              Σ wᵢφᵢ + b
          ...                    →
          φₖ(||x - cₖ||)

Where:
  cᵢ = center of RBF neuron i
  φᵢ = RBF activation (typically Gaussian)
  wᵢ = output weight

Output: f(x) = Σᵢ wᵢ exp(-γ||x - cᵢ||²) + b
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ============================================================
# RBF NETWORK: FROM SCRATCH
# ============================================================

class RBFNetwork:
    """
    Radial Basis Function Network for regression/classification.
    
    Architecture:
      Input → RBF hidden layer → Linear output
    """
    
    def __init__(self, n_centers=10, gamma=1.0):
        self.n_centers = n_centers
        self.gamma = gamma
        self.centers = None
        self.weights = None
        self.bias = 0.0
    
    def _rbf_activation(self, X, center):
        """Compute RBF activation: exp(-γ||x - c||²)"""
        distances_sq = np.sum((X - center)**2, axis=1)
        return np.exp(-self.gamma * distances_sq)
    
    def _compute_activations(self, X):
        """Compute activations for all RBF neurons."""
        n = len(X)
        activations = np.zeros((n, self.n_centers))
        for i, center in enumerate(self.centers):
            activations[:, i] = self._rbf_activation(X, center)
        return activations
    
    def fit(self, X, y, center_method='kmeans'):
        """
        Train RBF network.
        
        Step 1: Choose centers (k-means or random)
        Step 2: Compute RBF activations
        Step 3: Solve for output weights (linear least squares)
        """
        if center_method == 'kmeans':
            # Use k-means to find centers
            kmeans = KMeans(n_clusters=self.n_centers, random_state=42)
            kmeans.fit(X)
            self.centers = kmeans.cluster_centers_
        elif center_method == 'random':
            # Random subset of training points
            indices = np.random.choice(len(X), self.n_centers, replace=False)
            self.centers = X[indices]
        else:
            raise ValueError(f"Unknown center_method: {center_method}")
        
        # Compute activations
        H = self._compute_activations(X)
        
        # Solve for weights: y = Hw + b
        # Add bias column
        H_bias = np.column_stack([H, np.ones(len(X))])
        
        # Least squares
        w_full = np.linalg.lstsq(H_bias, y, rcond=None)[0]
        self.weights = w_full[:-1]
        self.bias = w_full[-1]
        
        return self
    
    def predict(self, X):
        """Predict using trained RBF network."""
        H = self._compute_activations(X)
        return H @ self.weights + self.bias
    
    def score(self, X, y):
        """R² score for regression."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return 1 - ss_res / (ss_tot + 1e-10)


print("="*70)
print("RBF NETWORK: DEMONSTRATION")
print("="*70)

# Generate non-linear regression data
np.random.seed(42)
n_samples = 200
X_rbfn = np.sort(np.random.uniform(0, 10, n_samples)).reshape(-1, 1)
y_rbfn = np.sin(X_rbfn.ravel()) + 0.3*np.sin(3*X_rbfn.ravel()) + \
         np.random.randn(n_samples) * 0.2

X_test_rbfn = np.linspace(0, 10, 300).reshape(-1, 1)
y_true_rbfn = np.sin(X_test_rbfn.ravel()) + 0.3*np.sin(3*X_test_rbfn.ravel())

# Train RBF networks with different number of centers
n_centers_list = [5, 10, 20, 40]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for ax, n_centers in zip(axes, n_centers_list):
    rbfn = RBFNetwork(n_centers=n_centers, gamma=1.0)
    rbfn.fit(X_rbfn, y_rbfn, center_method='kmeans')
    
    y_pred = rbfn.predict(X_test_rbfn)
    r2 = rbfn.score(X_rbfn, y_rbfn)
    
    ax.scatter(X_rbfn, y_rbfn, c='gray', s=10, alpha=0.5, label='Data')
    ax.plot(X_test_rbfn, y_true_rbfn, 'g-', linewidth=2, label='True')
    ax.plot(X_test_rbfn, y_pred, 'r--', linewidth=2, label='RBF Network')
    
    # Plot centers
    ax.scatter(rbfn.centers, np.zeros(len(rbfn.centers)),
               c='blue', s=100, marker='^', zorder=5,
               label=f'Centers ({n_centers})')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'RBF Network: {n_centers} centers\nR² = {r2:.3f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('RBF Network: Effect of Number of Centers\n'
             'More centers = more capacity',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('rbf_network.png', dpi=150, bbox_inches='tight')
print("Saved: rbf_network.png")

print(f"\nRBF Network trained with {n_centers_list[-1]} centers:")
print(f"  Training R²: {rbfn.score(X_rbfn, y_rbfn):.4f}")


# ============================================================
# RBF NETWORK vs KERNEL RIDGE REGRESSION
# ============================================================

from sklearn.kernel_ridge import KernelRidge

print("\n" + "="*70)
print("RBF NETWORK vs KERNEL RIDGE REGRESSION")
print("="*70)

# Train both
rbfn_comp = RBFNetwork(n_centers=20, gamma=1.0)
rbfn_comp.fit(X_rbfn, y_rbfn)

kr = KernelRidge(kernel='rbf', alpha=0.01, gamma=1.0)
kr.fit(X_rbfn, y_rbfn)

# Compare
y_rbfn_pred = rbfn_comp.predict(X_test_rbfn)
y_kr_pred = kr.predict(X_test_rbfn)

mse_rbfn = np.mean((y_rbfn_pred - y_true_rbfn)**2)
mse_kr = np.mean((y_kr_pred - y_true_rbfn)**2)

print(f"\nRBF Network (20 centers):")
print(f"  MSE: {mse_rbfn:.4f}")
print(f"  Parameters: {n_centers * 1 + n_centers} (centers + weights)")

print(f"\nKernel Ridge Regression:")
print(f"  MSE: {mse_kr:.4f}")
print(f"  Parameters: {len(X_rbfn)} (one per training point)")

print(f"\nDifference:")
print(f"  RBF Network is {len(X_rbfn) / n_centers:.1f}x more compact")
print(f"  But Kernel Ridge is typically more accurate")
```

---

## Gaussian Processes with RBF

### GP Regression with RBF Kernel

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ============================================================
# GAUSSIAN PROCESS with RBF KERNEL
# ============================================================

print("="*70)
print("GAUSSIAN PROCESS REGRESSION with RBF")
print("="*70)

np.random.seed(42)
n_gp = 30
X_gp = np.sort(np.random.uniform(0, 10, n_gp)).reshape(-1, 1)
y_gp = np.sin(X_gp.ravel()) + np.random.randn(n_gp) * 0.2

X_test_gp = np.linspace(0, 10, 300).reshape(-1, 1)
y_true_gp = np.sin(X_test_gp.ravel())

# Define GP with RBF kernel
kernel = C(1.0) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)

print(f"\nKernel: {kernel}")

# Fit
gp.fit(X_gp, y_gp)

print(f"\nOptimized kernel: {gp.kernel_}")
print(f"Log marginal likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.4f}")

# Predict with uncertainty
y_pred_gp, sigma = gp.predict(X_test_gp, return_std=True)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Predictions with uncertainty
ax = axes[0]
ax.scatter(X_gp, y_gp, c='red', s=50, zorder=10, label='Training data')
ax.plot(X_test_gp, y_true_gp, 'g--', linewidth=1.5, alpha=0.7, label='True function')
ax.plot(X_test_gp, y_pred_gp, 'b-', linewidth=2, label='GP mean')
ax.fill_between(X_test_gp.ravel(),
                y_pred_gp - 1.96*sigma,
                y_pred_gp + 1.96*sigma,
                alpha=0.2, color='blue', label='95% confidence')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gaussian Process with RBF Kernel\n(Prediction + Uncertainty)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Uncertainty alone
ax = axes[1]
ax.plot(X_test_gp, sigma, 'purple', linewidth=2.5)
for x_train in X_gp:
    ax.axvline(x=x_train, color='red', alpha=0.3, linewidth=0.8)
ax.set_xlabel('x')
ax.set_ylabel('Uncertainty (σ)')
ax.set_title('Predictive Uncertainty\n(High where no training data)')
ax.grid(True, alpha=0.3)

plt.suptitle('Gaussian Process with RBF Kernel',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('gp_rbf.png', dpi=150, bbox_inches='tight')
print("\nSaved: gp_rbf.png")

print("\nKey observations:")
print("  - GP provides uncertainty estimates (not just point predictions)")
print("  - Uncertainty is LOW near training data, HIGH far from it")
print("  - RBF kernel controls smoothness via length_scale parameter")
```

---

## Choosing Gamma: Practical Strategies

### Strategy 1: sklearn Defaults

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# ============================================================
# SKLEARN GAMMA DEFAULTS
# ============================================================

print("="*70)
print("GAMMA SELECTION: SKLEARN DEFAULTS")
print("="*70)

np.random.seed(42)
X_default = np.random.randn(100, 5)
y_default = (X_default.sum(axis=1) > 0).astype(int)

sc = StandardScaler()
X_s = sc.fit_transform(X_default)

# Two sklearn defaults
clf_scale = SVC(kernel='rbf', gamma='scale', C=1.0)
clf_auto = SVC(kernel='rbf', gamma='auto', C=1.0)

clf_scale.fit(X_s, y_default)
clf_auto.fit(X_s, y_default)

print("\nsklearn gamma options:")
print(f"\n1. gamma='scale' (default, recommended)")
print(f"   Formula: 1 / (n_features × variance(X))")
print(f"   Value: {clf_scale._gamma:.6f}")
print(f"   Accuracy: {clf_scale.score(X_s, y_default)*100:.1f}%")

print(f"\n2. gamma='auto' (older default)")
print(f"   Formula: 1 / n_features")
print(f"   Value: {clf_auto._gamma:.6f}")
print(f"   Accuracy: {clf_auto.score(X_s, y_default)*100:.1f}%")

print("\nRECOMMENDATION: Use gamma='scale' as starting point")
```

### Strategy 2: Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

# ============================================================
# GRID SEARCH for GAMMA
# ============================================================

print("\n" + "="*70)
print("GAMMA SELECTION: GRID SEARCH")
print("="*70)

X_grid, y_grid = make_classification(
    n_samples=300, n_features=10, n_informative=8,
    n_redundant=2, random_state=42
)
X_grid = StandardScaler().fit_transform(X_grid)

# Define grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 'scale', 'auto']
}

# Grid search
clf_grid = GridSearchCV(
    SVC(kernel='rbf'),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

clf_grid.fit(X_grid, y_grid)

print(f"\nBest parameters: {clf_grid.best_params_}")
print(f"Best CV score: {clf_grid.best_score_*100:.2f}%")

# Show top 5
results_df = clf_grid.cv_results_
indices = np.argsort(results_df['mean_test_score'])[::-1][:5]

print(f"\nTop 5 parameter combinations:")
print(f"{'Rank':<6} {'C':<10} {'gamma':<12} {'Mean CV Acc'}")
print("-"*42)
for rank, idx in enumerate(indices, 1):
    c = results_df['param_C'][idx]
    g = results_df['param_gamma'][idx]
    acc = results_df['mean_test_score'][idx]
    print(f"{rank:<6} {c:<10} {str(g):<12} {acc*100:.2f}%")
```

### Strategy 3: Median Heuristic

```python
from sklearn.metrics import pairwise_distances

# ============================================================
# MEDIAN HEURISTIC for GAMMA
# ============================================================

print("\n" + "="*70)
print("GAMMA SELECTION: MEDIAN HEURISTIC")
print("="*70)

def median_heuristic_gamma(X):
    """
    Median heuristic: set γ = 1 / (2 × median²(pairwise distances))
    
    Rationale: median distance is a robust estimate of typical scale
    """
    # Compute pairwise distances
    dists = pairwise_distances(X)
    
    # Get upper triangle (exclude diagonal and duplicates)
    n = len(X)
    triu_indices = np.triu_indices(n, k=1)
    dists_upper = dists[triu_indices]
    
    # Median distance
    median_dist = np.median(dists_upper)
    
    # Gamma
    gamma = 1.0 / (2 * median_dist**2)
    
    return gamma, median_dist


X_median = np.random.randn(100, 3)

gamma_median, med_dist = median_heuristic_gamma(X_median)
gamma_scale = 1.0 / (X_median.shape[1] * X_median.var())
gamma_auto = 1.0 / X_median.shape[1]

print(f"\nThree gamma strategies:")
print(f"  Median heuristic: γ = {gamma_median:.6f}  (median dist = {med_dist:.4f})")
print(f"  sklearn 'scale':  γ = {gamma_scale:.6f}")
print(f"  sklearn 'auto':   γ = {gamma_auto:.6f}")

print(f"\nMedian heuristic is useful when:")
print(f"  - Data has been standardized")
print(f"  - You want a data-driven default")
print(f"  - Grid search is too expensive")
```

### Strategy 4: Learning Curves

```python
# ============================================================
# GAMMA SELECTION: LEARNING CURVES
# ============================================================

from sklearn.model_selection import learning_curve

print("\n" + "="*70)
print("GAMMA SELECTION: LEARNING CURVES")
print("="*70)

X_lc, y_lc = make_classification(
    n_samples=500, n_features=10, n_informative=8,
    random_state=42
)
X_lc = StandardScaler().fit_transform(X_lc)

gammas_lc = [0.01, 0.1, 1.0, 10.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ax, gamma in zip(axes, gammas_lc):
    train_sizes, train_scores, test_scores = learning_curve(
        SVC(kernel='rbf', gamma=gamma, C=1.0),
        X_lc, y_lc,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', linewidth=2, label='Training')
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.2)
    
    ax.plot(train_sizes, test_mean, 's-', linewidth=2, label='Validation')
    ax.fill_between(train_sizes, test_mean - test_std,
                    test_mean + test_std, alpha=0.2)
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'γ = {gamma}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])

plt.suptitle('Learning Curves for Different γ\n'
             'Look for: high validation acc, small train-val gap',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('gamma_learning_curves.png', dpi=150, bbox_inches='tight')
print("Saved: gamma_learning_curves.png")

print("\nHow to interpret learning curves:")
print("  - Large gap (train >> val): OVERFITTING → decrease γ")
print("  - Both low: UNDERFITTING → increase γ")
print("  - Both high, small gap: GOOD FIT ✓")
```

---

## RBF for Interpolation

### Exact Interpolation with RBFs

```python
# ============================================================
# RBF INTERPOLATION
# ============================================================

print("="*70)
print("RBF INTERPOLATION")
print("="*70)

from scipy.interpolate import Rbf

# Generate scattered data
np.random.seed(42)
n_interp = 20
X_interp = np.random.uniform(-5, 5, n_interp)
Y_interp = np.random.uniform(-5, 5, n_interp)
Z_interp = np.sin(np.sqrt(X_interp**2 + Y_interp**2))

# Create RBF interpolator
rbf_interp = Rbf(X_interp, Y_interp, Z_interp, function='gaussian', epsilon=2)

# Evaluate on grid
xi = np.linspace(-5, 5, 100)
yi = np.linspace(-5, 5, 100)
XI, YI = np.meshgrid(xi, yi)
ZI = rbf_interp(XI, YI)

# True function
Z_true = np.sin(np.sqrt(XI**2 + YI**2))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scattered data
ax = axes[0]
scatter = ax.scatter(X_interp, Y_interp, c=Z_interp,
                     s=100, cmap='viridis', edgecolors='k', linewidths=1.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Scattered Data Points\n(20 samples)')
plt.colorbar(scatter, ax=ax)
ax.grid(True, alpha=0.3)

# RBF interpolation
ax = axes[1]
cf = ax.contourf(XI, YI, ZI, levels=20, cmap='viridis')
ax.scatter(X_interp, Y_interp, c='red', s=30, marker='x', linewidths=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('RBF Interpolation\n(Smooth reconstruction)')
plt.colorbar(cf, ax=ax)

# True function
ax = axes[2]
cf = ax.contourf(XI, YI, Z_true, levels=20, cmap='viridis')
ax.scatter(X_interp, Y_interp, c='red', s=30, marker='x', linewidths=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('True Function\nsin(√(x²+y²))')
plt.colorbar(cf, ax=ax)

plt.suptitle('RBF Interpolation for Scattered Data',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('rbf_interpolation.png', dpi=150, bbox_inches='tight')
print("Saved: rbf_interpolation.png")

# Error
error = np.abs(ZI - Z_true)
print(f"\nInterpolation error:")
print(f"  Mean absolute error: {error.mean():.4f}")
print(f"  Max absolute error: {error.max():.4f}")
```

---

## Computational Considerations

### Time Complexity

```
Training:
  - Kernel matrix computation: O(n² × d)
  - QP solving: O(n² × n_sv) to O(n³)
  - Total: O(n²d) to O(n³)

Prediction (single point):
  - Compute K(x, xᵢ) for all support vectors: O(n_sv × d)

Memory:
  - Kernel matrix: O(n²)  (can be HUGE!)
  - Support vectors: O(n_sv × d)
```

```python
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ============================================================
# COMPUTATIONAL COMPLEXITY: EMPIRICAL ANALYSIS
# ============================================================

print("="*70)
print("RBF KERNEL: COMPUTATIONAL COMPLEXITY")
print("="*70)

sizes = [100, 200, 500, 1000, 2000]
dimensions = [5, 10, 20]

print(f"\n{'n':<8} {'d':<8} {'Train Time (s)':<18} {'Predict Time (ms)':<20} {'# SV'}")
print("-"*70)

for n in sizes:
    for d in dimensions:
        if n * d > 50000:  # Skip very large cases
            continue
        
        X = np.random.randn(n, d)
        y = (X.sum(axis=1) > 0).astype(int)
        X = StandardScaler().fit_transform(X)
        
        X_train, X_test = X[:int(0.8*n)], X[int(0.8*n):]
        y_train, y_test = y[:int(0.8*n)], y[int(0.8*n):]
        
        # Train
        t0 = time.perf_counter()
        clf = SVC(kernel='rbf', gamma='scale', C=1.0)
        clf.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        # Predict
        t0 = time.perf_counter()
        _ = clf.predict(X_test)
        predict_time = (time.perf_counter() - t0) / len(X_test) * 1000
        
        n_sv = clf.n_support_.sum()
        
        print(f"{n:<8} {d:<8} {train_time:<18.4f} {predict_time:<20.2f} {n_sv}")


# ============================================================
# SCALING: Visualize complexity
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Theoretical complexity
n_vals = np.logspace(2, 4, 50)

ax = axes[0]
ax.loglog(n_vals, n_vals**2 * 5, label='O(n²) — Kernel matrix', linewidth=2.5)
ax.loglog(n_vals, n_vals**3 / 1000, label='O(n³) — QP solving', linewidth=2.5)
ax.set_xlabel('Dataset Size (n)')
ax.set_ylabel('Operations')
ax.set_title('Training Time Complexity')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Memory complexity
ax = axes[1]
ax.loglog(n_vals, n_vals**2 * 8 / 1e9, linewidth=2.5, color='purple')
ax.set_xlabel('Dataset Size (n)')
ax.set_ylabel('Memory (GB)')
ax.set_title('Memory Complexity\n(Kernel matrix: n² doubles)')
ax.grid(True, alpha=0.3, which='both')
ax.axhline(y=8, color='red', linestyle='--', label='8 GB RAM')
ax.axhline(y=32, color='orange', linestyle='--', label='32 GB RAM')
ax.legend()

plt.suptitle('RBF-SVM Computational Complexity',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('rbf_complexity.png', dpi=150, bbox_inches='tight')
print("\nSaved: rbf_complexity.png")

print("\nPractical limits:")
print("  n < 10,000:   Fast, practical")
print("  n = 10-100K:  Slow but feasible with patience")
print("  n > 100K:     Impractical for standard RBF-SVM")
print("                → Use linear SVM or approximate kernels")
```

### Kernel Approximation Methods

```python
from sklearn.kernel_approximation import RBFSampler, Nystroem

# ============================================================
# KERNEL APPROXIMATION for LARGE DATASETS
# ============================================================

print("\n" + "="*70)
print("KERNEL APPROXIMATION METHODS")
print("="*70)

# Large dataset
n_large = 10000
X_large = np.random.randn(n_large, 20)
y_large = (X_large.sum(axis=1) > 0).astype(int)
X_large = StandardScaler().fit_transform(X_large)

X_tr_l, X_te_l = X_large[:8000], X_large[8000:]
y_tr_l, y_te_l = y_large[:8000], y_large[8000:]

print(f"\nDataset size: {n_large} samples, 20 features")

# Method 1: Random Fourier Features (RFF)
print("\n1. RANDOM FOURIER FEATURES (Rahimi & Recht, 2007)")
rbf_sampler = RBFSampler(gamma=1.0, n_components=100, random_state=42)
X_tr_rff = rbf_sampler.fit_transform(X_tr_l)
X_te_rff = rbf_sampler.transform(X_te_l)

from sklearn.linear_model import LogisticRegression
clf_rff = LogisticRegression()

t0 = time.perf_counter()
clf_rff.fit(X_tr_rff, y_tr_l)
time_rff = time.perf_counter() - t0

acc_rff = clf_rff.score(X_te_rff, y_te_l)

print(f"  Explicit features: {X_tr_rff.shape[1]}")
print(f"  Training time: {time_rff:.4f}s")
print(f"  Test accuracy: {acc_rff*100:.2f}%")

# Method 2: Nyström approximation
print("\n2. NYSTRÖM APPROXIMATION")
nystroem = Nystroem(kernel='rbf', gamma=1.0, n_components=100, random_state=42)
X_tr_nys = nystroem.fit_transform(X_tr_l)
X_te_nys = nystroem.transform(X_te_l)

clf_nys = LogisticRegression()

t0 = time.perf_counter()
clf_nys.fit(X_tr_nys, y_tr_l)
time_nys = time.perf_counter() - t0

acc_nys = clf_nys.score(X_te_nys, y_te_l)

print(f"  Explicit features: {X_tr_nys.shape[1]}")
print(f"  Training time: {time_nys:.4f}s")
print(f"  Test accuracy: {acc_nys*100:.2f}%")

# Method 3: Exact kernel (too slow for large data, but shown for comparison on subset)
print("\n3. EXACT RBF KERNEL (on 1000-sample subset)")
X_tr_sub = X_tr_l[:1000]
y_tr_sub = y_tr_l[:1000]

clf_exact = SVC(kernel='rbf', gamma=1.0, C=1.0)

t0 = time.perf_counter()
clf_exact.fit(X_tr_sub, y_tr_sub)
time_exact = time.perf_counter() - t0

acc_exact = clf_exact.score(X_te_l, y_te_l)

print(f"  Training size: {len(X_tr_sub)}")
print(f"  Training time: {time_exact:.4f}s")
print(f"  Test accuracy: {acc_exact*100:.2f}%")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Method':<25} {'Train Time':<15} {'Accuracy':<12} {'Speedup'}")
print("-"*65)
print(f"{'Random Fourier Features':<25} {time_rff:<15.4f} {acc_rff*100:<12.2f} {time_exact/time_rff:.1f}x")
print(f"{'Nyström':<25} {time_nys:<15.4f} {acc_nys*100:<12.2f} {time_exact/time_nys:.1f}x")
print(f"{'Exact (subset)':<25} {time_exact:<15.4f} {acc_exact*100:<12.2f} 1.0x")

print("\nFor large datasets (n > 10K):")
print("  → Use kernel approximation methods")
print("  → Or switch to linear models")
```

---

## Summary and Quick Reference

### RBF Kernel: Complete Summary

```python
print("="*70)
print("RBF KERNEL: COMPLETE SUMMARY")
print("="*70)

summary_text = """
DEFINITION:
  K(x, x') = exp(-γ ||x - x'||²)

WHY RBF IS POPULAR:
  ✓ Universal approximator (can fit any continuous function)
  ✓ Always valid (PSD for any γ > 0)
  ✓ Smooth, infinitely differentiable
  ✓ Stationary (translation-invariant)
  ✓ Single hyperparameter (γ) to tune
  ✓ Works well as default choice

GAMMA CONTROLS:
  Small γ → Wide influence, smooth boundaries, underfit risk
  Large γ → Narrow influence, complex boundaries, overfit risk

GAMMA SELECTION STRATEGIES:
  1. gamma='scale' (sklearn default): 1/(d × var(X))
  2. Grid search with CV
  3. Median heuristic: 1/(2 × median²(distances))
  4. Learning curves

WHEN TO USE RBF:
  ✓ Non-linear patterns
  ✓ No prior knowledge of data structure
  ✓ Small-to-medium datasets (n < 10K)
  ✓ Need smooth boundaries

WHEN NOT TO USE RBF:
  ✗ Linearly separable data (use linear kernel)
  ✗ Very large datasets (n > 100K) → too slow
  ✗ High-dimensional sparse data (text) → linear often better

COMPUTATIONAL COST:
  Training: O(n²d) to O(n³)
  Memory: O(n²) for kernel matrix
  Prediction: O(n_sv × d) per point

ALTERNATIVES FOR LARGE DATA:
  - Random Fourier Features
  - Nyström approximation
  - Linear models
"""

print(summary_text)
```

### Prerequisites Checklist

Before moving to Chapter 07, confirm you can:
- [ ] Write the RBF kernel formula from memory
- [ ] Explain what gamma controls intuitively
- [ ] Implement RBF kernel from scratch in NumPy
- [ ] Choose gamma using at least 2 strategies
- [ ] Explain why RBF is a universal kernel
- [ ] Describe when RBF is better/worse than linear kernel
- [ ] Build a simple RBF network
- [ ] Estimate computational complexity of RBF-SVM

### What's Next

**07_SVM_Hard_Margin.md**  
→ Complete SVM optimization theory with hard-margin formulation

**08_SVM_Soft_Margin.md**  
→ Slack variables, hinge loss, and the C parameter

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════╗
║                RBF KERNEL — QUICK REFERENCE                      ║
╠══════════════════════════════════════════════════════════════════╣
║  FORMULA                                                         ║
║    K(x, x') = exp(-γ ||x - x'||²)                               ║
║    γ > 0  (gamma parameter)                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  PROPERTIES                                                      ║
║    - Universal approximator                                      ║
║    - Always PSD (valid kernel)                                   ║
║    - Stationary (translation-invariant)                          ║
║    - Infinitely differentiable (C^∞)                            ║
║    - Bounded: 0 < K ≤ 1                                          ║
╠══════════════════════════════════════════════════════════════════╣
║  GAMMA INTERPRETATION                                            ║
║    Small γ  →  wide bell  →  smooth  →  underfit risk           ║
║    Large γ  →  narrow bell  →  complex  →  overfit risk         ║
║                                                                  ║
║    Effective radius: r_eff ≈ √(-log(0.05)/γ)                   ║
╠══════════════════════════════════════════════════════════════════╣
║  GAMMA SELECTION                                                 ║
║    1. gamma='scale':  1/(n_features × var(X))  [sklearn default]║
║    2. gamma='auto':   1/n_features                               ║
║    3. Median heuristic: 1/(2 × median²(distances))              ║
║    4. Grid search: try [0.001, 0.01, 0.1, 1, 10]               ║
╠══════════════════════════════════════════════════════════════════╣
║  SKLEARN USAGE                                                   ║
║    from sklearn.svm import SVC                                   ║
║    clf = SVC(kernel='rbf', gamma='scale', C=1.0)                 ║
║    clf.fit(X_train, y_train)                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  COMPUTATIONAL LIMITS                                            ║
║    n < 10K:    ✓ Fast                                            ║
║    n = 10-100K: ~ Slow but feasible                              ║
║    n > 100K:   ✗ Use approximation or linear kernel             ║
╠══════════════════════════════════════════════════════════════════╣
║  WHEN TO USE                                                     ║
║    ✓ Non-linear patterns                                         ║
║    ✓ Unknown data structure                                      ║
║    ✓ Need smooth boundaries                                      ║
║    ✓ Small-medium datasets                                       ║
║                                                                  ║
║  WHEN NOT TO USE                                                 ║
║    ✗ Linear data (use linear kernel)                             ║
║    ✗ Very large datasets (too slow)                              ║
║    ✗ High-dim sparse (text → linear better)                      ║
╚══════════════════════════════════════════════════════════════════╝
```

---

*End of 06_Radial_Basis_Functions.md*