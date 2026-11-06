# Kernel Functions: Theory and Properties

## Table of Contents
1. [What Makes a Function a Valid Kernel?](#what-makes-a-function-a-valid-kernel)
2. [Properties of Kernel Functions](#properties-of-kernel-functions)
3. [Kernel Composition and Combination](#kernel-composition-and-combination)
4. [Feature Map Interpretations](#feature-map-interpretations)
5. [Kernel Alignment and Similarity](#kernel-alignment-and-similarity)
6. [Infinite-Dimensional Feature Spaces](#infinite-dimensional-feature-spaces)
7. [Normalized Kernels](#normalized-kernels)
8. [Stationary vs Non-Stationary Kernels](#stationary-vs-non-stationary-kernels)
9. [Universal Kernels](#universal-kernels)
10. [Summary and Quick Reference](#summary-and-quick-reference)

---

## What Makes a Function a Valid Kernel?

### The Fundamental Question

Given any symmetric function K: X × X → ℝ, when can we use it as a kernel in SVM or other kernel methods?

**Answer**: K is a valid kernel if and only if it is **positive semi-definite** (PSD).

### Three Equivalent Conditions

A function K(x, x') is a valid kernel if ANY of these hold:

```
CONDITION 1 (Mercer's Theorem):
  For any finite set {x₁, ..., xₙ} ⊂ X,
  the kernel matrix K with K[i,j] = K(xᵢ, xⱼ)
  is positive semi-definite

CONDITION 2 (Feature Map):
  There exists a Hilbert space H and a map φ: X → H
  such that K(x, x') = ⟨φ(x), φ(x')⟩_H

CONDITION 3 (Positive Semi-Definiteness):
  For any function f with ∫f²dx < ∞:
  ∬ K(x, x') f(x) f(x') dx dx' ≥ 0
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# ============================================================
# TEST KERNEL VALIDITY
# ============================================================

def is_valid_kernel(K_func, X, name="Kernel"):
    """
    Test if K_func is a valid kernel by checking PSD property.
    
    Returns:
        valid (bool): Whether kernel is PSD
        min_eig (float): Minimum eigenvalue
        K_matrix (ndarray): The kernel matrix
    """
    n = len(X)
    
    # Build kernel matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = K_func(X[i], X[j])
    
    # Check symmetry
    if not np.allclose(K, K.T):
        print(f"✗ {name}: NOT SYMMETRIC")
        return False, None, K
    
    # Check PSD via eigenvalues
    eigvals = np.linalg.eigvalsh(K)
    min_eig = eigvals.min()
    is_psd  = min_eig >= -1e-8
    
    status = "✓ VALID" if is_psd else "✗ INVALID"
    print(f"{status} — {name}: min eigenvalue = {min_eig:.6e}")
    
    return is_psd, min_eig, K


print("=" * 70)
print("KERNEL VALIDITY TESTING")
print("=" * 70)

np.random.seed(42)
X_test = np.random.randn(20, 3)

# Valid kernels
def k_linear(x, xp):
    return np.dot(x, xp)

def k_poly(x, xp, d=2):
    return (np.dot(x, xp) + 1)**d

def k_rbf(x, xp, gamma=0.5):
    return np.exp(-gamma * np.sum((x - xp)**2))

def k_laplacian(x, xp, gamma=0.5):
    return np.exp(-gamma * np.sum(np.abs(x - xp)))

# Invalid kernel examples
def k_not_symmetric(x, xp):
    return np.dot(x, xp) + np.linalg.norm(x)  # Breaks symmetry

def k_negative_definite(x, xp):
    return -np.exp(-np.sum((x - xp)**2))  # Negative of valid kernel

print("\n--- VALID KERNELS ---")
is_valid_kernel(k_linear, X_test, "Linear")
is_valid_kernel(k_poly, X_test, "Polynomial")
is_valid_kernel(k_rbf, X_test, "RBF")
is_valid_kernel(k_laplacian, X_test, "Laplacian")

print("\n--- INVALID KERNELS ---")
is_valid_kernel(k_not_symmetric, X_test, "Not Symmetric")
is_valid_kernel(k_negative_definite, X_test, "Negative Definite")


# ============================================================
# VISUALIZE KERNEL MATRICES
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

kernels_viz = [
    ("Linear", k_linear),
    ("Polynomial", k_poly),
    ("RBF", k_rbf),
    ("Laplacian", k_laplacian),
    ("Not Symmetric", k_not_symmetric),
    ("Negative Definite", k_negative_definite),
]

for ax, (name, k_fn) in zip(axes, kernels_viz):
    valid, min_eig, K = is_valid_kernel(k_fn, X_test[:12], name)
    
    if K is not None:
        im = ax.imshow(K, cmap='RdYlBu_r', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        title_color = 'green' if valid else 'red'
        status = "✓ Valid" if valid else "✗ Invalid"
        ax.set_title(f"{name}\n{status}\nλ_min={min_eig:.2e}" if min_eig else name,
                     color=title_color, fontweight='bold')
    
    ax.set_xlabel('Sample j')
    ax.set_ylabel('Sample i')

plt.suptitle("Kernel Matrix Visualization\n"
             "Valid kernels are PSD (all eigenvalues ≥ 0)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('kernel_validity_matrices.png', dpi=150, bbox_inches='tight')
print("\nSaved: kernel_validity_matrices.png")
```

### The Gram Matrix Criterion

For practical validation, we use the **Gram matrix** (kernel matrix):

```
Given n points {x₁, ..., xₙ},
build K where K[i,j] = K(xᵢ, xⱼ)

K is valid kernel ⟺ K is positive semi-definite
                  ⟺ all eigenvalues of K are ≥ 0
                  ⟺ for all vectors v: vᵀKv ≥ 0
```

```python
import numpy as np

# ============================================================
# GRAM MATRIX: DETAILED ANALYSIS
# ============================================================

def analyze_gram_matrix(K_func, X, name=""):
    """Complete analysis of a kernel's Gram matrix."""
    n = len(X)
    K = np.array([[K_func(X[i], X[j]) for j in range(n)]
                  for i in range(n)])
    
    print(f"\n{'='*60}")
    print(f"GRAM MATRIX ANALYSIS: {name}")
    print(f"{'='*60}")
    
    # 1. Symmetry
    is_sym = np.allclose(K, K.T)
    print(f"1. Symmetric: {is_sym}")
    if not is_sym:
        max_asym = np.abs(K - K.T).max()
        print(f"   Max asymmetry: {max_asym:.2e}")
    
    # 2. Eigenspectrum
    eigvals = np.linalg.eigvalsh(K)
    print(f"\n2. Eigenspectrum:")
    print(f"   Min eigenvalue: {eigvals.min():.6f}")
    print(f"   Max eigenvalue: {eigvals.max():.6f}")
    print(f"   Condition number: {eigvals.max()/eigvals.min():.2e}" 
          if eigvals.min() > 1e-10 else "   Condition number: ∞ (singular)")
    
    # 3. Positive semi-definite?
    is_psd = eigvals.min() >= -1e-8
    print(f"\n3. Positive Semi-Definite: {is_psd}")
    
    # 4. Rank
    rank = np.sum(eigvals > 1e-10)
    print(f"\n4. Rank: {rank}/{n}")
    if rank < n:
        print(f"   → Kernel matrix is rank-deficient")
        print(f"   → {n - rank} eigenvalues ≈ 0")
    
    # 5. Diagonal values (self-similarity)
    diag = np.diag(K)
    print(f"\n5. Diagonal (self-similarity):")
    print(f"   K(x,x) range: [{diag.min():.4f}, {diag.max():.4f}]")
    print(f"   K(x,x) mean: {diag.mean():.4f}")
    
    # 6. Off-diagonal (cross-similarity)
    mask = ~np.eye(n, dtype=bool)
    off_diag = K[mask]
    print(f"\n6. Off-diagonal (cross-similarity):")
    print(f"   K(x,x') range: [{off_diag.min():.4f}, {off_diag.max():.4f}]")
    print(f"   K(x,x') mean: {off_diag.mean():.4f}")
    
    # 7. Frobenius norm
    frob = np.linalg.norm(K, 'fro')
    print(f"\n7. Frobenius norm: {frob:.4f}")
    
    # 8. Trace
    trace = np.trace(K)
    print(f"8. Trace: {trace:.4f}")
    
    return K, eigvals, is_psd


# Test on different kernels
np.random.seed(42)
X_sample = np.random.randn(15, 2)

# Linear kernel
K_lin, eig_lin, valid_lin = analyze_gram_matrix(
    k_linear, X_sample, "Linear Kernel"
)

# RBF kernel
K_rbf, eig_rbf, valid_rbf = analyze_gram_matrix(
    k_rbf, X_sample, "RBF Kernel (γ=0.5)"
)

# Polynomial kernel
K_poly, eig_poly, valid_poly = analyze_gram_matrix(
    k_poly, X_sample, "Polynomial Kernel (d=2)"
)


# ============================================================
# EIGENSPECTRUM COMPARISON
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

kernels_eig = [
    ("Linear", eig_lin),
    ("RBF γ=0.5", eig_rbf),
    ("Polynomial d=2", eig_poly),
]

for ax, (name, eigvals) in zip(axes, kernels_eig):
    ax.plot(range(len(eigvals)), sorted(eigvals, reverse=True),
            'o-', linewidth=2, markersize=6)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5,
               label='Zero threshold')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'{name}\nEigenspectrum')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text: number of positive eigenvalues
    n_pos = np.sum(eigvals > 1e-10)
    ax.text(0.95, 0.05, f'{n_pos}/{len(eigvals)} positive',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("Eigenspectrum Comparison\n"
             "All eigenvalues ≥ 0 → Valid kernel",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('kernel_eigenspectrum.png', dpi=150, bbox_inches='tight')
print("\nSaved: kernel_eigenspectrum.png")
```

---

## Properties of Kernel Functions

### Essential Properties All Kernels Must Have

```python
import numpy as np

# ============================================================
# KERNEL PROPERTIES: SYSTEMATIC CHECK
# ============================================================

def check_kernel_properties(K_func, X, name="Kernel"):
    """
    Check all essential properties a valid kernel must satisfy.
    """
    print(f"\n{'='*60}")
    print(f"KERNEL PROPERTIES: {name}")
    print(f"{'='*60}")
    
    n = len(X)
    K = np.array([[K_func(X[i], X[j]) for j in range(n)]
                  for i in range(n)])
    
    properties = {}
    
    # Property 1: Symmetry K(x,x') = K(x',x)
    is_symmetric = np.allclose(K, K.T)
    properties['symmetric'] = is_symmetric
    print(f"\n1. Symmetry: {is_symmetric}")
    if not is_symmetric:
        print(f"   Max |K(x,x') - K(x',x)|: {np.abs(K - K.T).max():.2e}")
    
    # Property 2: PSD (all eigenvalues ≥ 0)
    eigvals = np.linalg.eigvalsh(K)
    is_psd  = eigvals.min() >= -1e-8
    properties['psd'] = is_psd
    print(f"\n2. Positive Semi-Definite: {is_psd}")
    print(f"   Min eigenvalue: {eigvals.min():.6e}")
    
    # Property 3: Bounded (if normalized)
    max_val = K.max()
    min_val = K.min()
    print(f"\n3. Value Range:")
    print(f"   Min K(x,x'): {min_val:.4f}")
    print(f"   Max K(x,x'): {max_val:.4f}")
    
    # Property 4: Diagonal dominance (optional, common in many kernels)
    diag = np.diag(K)
    off_diag_max = np.max(K - np.diag(diag))
    diag_dominant = np.all(diag >= off_diag_max - 1e-8)
    properties['diag_dominant'] = diag_dominant
    print(f"\n4. Diagonal Dominance: {diag_dominant}")
    print(f"   (K(x,x) ≥ K(x,x') for all x,x')")
    if not diag_dominant:
        print(f"   Max diagonal: {diag.max():.4f}")
        print(f"   Max off-diagonal: {off_diag_max:.4f}")
    
    # Property 5: Positive diagonal
    all_pos_diag = np.all(diag > -1e-8)
    properties['pos_diagonal'] = all_pos_diag
    print(f"\n5. Positive Diagonal: {all_pos_diag}")
    print(f"   All K(x,x) > 0: {np.all(diag > 0)}")
    
    # Property 6: Cauchy-Schwarz inequality
    # |K(x,x')|² ≤ K(x,x) · K(x',x')
    violations = 0
    for i in range(n):
        for j in range(i+1, n):
            lhs = K[i,j]**2
            rhs = K[i,i] * K[j,j]
            if lhs > rhs + 1e-8:
                violations += 1
    
    cs_satisfied = violations == 0
    properties['cauchy_schwarz'] = cs_satisfied
    print(f"\n6. Cauchy-Schwarz: {cs_satisfied}")
    if not cs_satisfied:
        print(f"   Violations: {violations}/{n*(n-1)//2}")
    
    # Overall validity
    is_valid = properties['symmetric'] and properties['psd']
    print(f"\n{'='*60}")
    print(f"VALID KERNEL: {is_valid}")
    print(f"{'='*60}")
    
    return properties, K


# Test suite
np.random.seed(42)
X_props = np.random.randn(12, 2)

check_kernel_properties(k_linear, X_props, "Linear")
check_kernel_properties(k_rbf, X_props, "RBF")
check_kernel_properties(k_poly, X_props, "Polynomial")
```

### Cauchy-Schwarz Inequality for Kernels

A valid kernel MUST satisfy:

```
|K(x, x')|² ≤ K(x, x) · K(x', x')

Proof: Since K(x,x') = ⟨φ(x), φ(x')⟩,
       by Cauchy-Schwarz in Hilbert space:
       |⟨φ(x), φ(x')⟩|² ≤ ||φ(x)||² · ||φ(x')||²
                       = ⟨φ(x),φ(x)⟩ · ⟨φ(x'),φ(x')⟩
                       = K(x,x) · K(x',x')
```

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CAUCHY-SCHWARZ VISUALIZATION
# ============================================================

def visualize_cauchy_schwarz(K_func, X, name=""):
    """
    Visualize Cauchy-Schwarz inequality for a kernel.
    Plot |K(x,x')|² vs K(x,x)·K(x',x') for all pairs.
    """
    n = len(X)
    
    lhs_vals = []  # |K(x,x')|²
    rhs_vals = []  # K(x,x) · K(x',x')
    
    for i in range(n):
        for j in range(i+1, n):
            k_ij = K_func(X[i], X[j])
            k_ii = K_func(X[i], X[i])
            k_jj = K_func(X[j], X[j])
            
            lhs_vals.append(k_ij**2)
            rhs_vals.append(k_ii * k_jj)
    
    lhs_vals = np.array(lhs_vals)
    rhs_vals = np.array(rhs_vals)
    
    plt.figure(figsize=(10, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(rhs_vals, lhs_vals, alpha=0.6, s=30)
    
    # Diagonal line: lhs = rhs
    max_val = max(lhs_vals.max(), rhs_vals.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2,
             label='lhs = rhs (equality)')
    
    plt.xlabel('K(x,x) · K(x\',x\')  (RHS)')
    plt.ylabel('|K(x,x\')|²  (LHS)')
    plt.title(f'Cauchy-Schwarz for {name}\n'
              f'All points must be BELOW red line')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Check violations
    violations = np.sum(lhs_vals > rhs_vals + 1e-8)
    plt.text(0.05, 0.95, f'Violations: {violations}/{len(lhs_vals)}',
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Histogram of ratio
    plt.subplot(1, 2, 2)
    ratios = lhs_vals / (rhs_vals + 1e-10)
    plt.hist(ratios, bins=30, edgecolor='k', alpha=0.7)
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                label='Ratio = 1 (boundary)')
    plt.xlabel('Ratio = |K(x,x\')|² / [K(x,x)·K(x\',x\')]')
    plt.ylabel('Count')
    plt.title('Distribution of Ratios\n(Must be ≤ 1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return lhs_vals, rhs_vals


np.random.seed(42)
X_cs = np.random.randn(30, 2)

print("\n" + "="*60)
print("CAUCHY-SCHWARZ INEQUALITY VERIFICATION")
print("="*60)

for name, k_fn in [("Linear", k_linear),
                    ("RBF", k_rbf),
                    ("Polynomial", k_poly)]:
    lhs, rhs = visualize_cauchy_schwarz(k_fn, X_cs, name)
    plt.savefig(f'cauchy_schwarz_{name.lower()}.png',
                dpi=150, bbox_inches='tight')
    print(f"\n{name} kernel:")
    print(f"  Max ratio: {(lhs/(rhs+1e-10)).max():.6f} (should be ≤ 1)")
    print(f"  Violations: {np.sum(lhs > rhs + 1e-8)}/{len(lhs)}")
    plt.close()
```

---

## Kernel Composition and Combination

### Rules for Building New Kernels

If K₁ and K₂ are valid kernels, then so are:

```
1. Scaling:          c · K₁(x,x')                 c > 0
2. Sum:              K₁(x,x') + K₂(x,x')
3. Product:          K₁(x,x') · K₂(x,x')
4. Polynomial:       [K₁(x,x') + c]ᵈ              c ≥ 0, d ∈ ℤ⁺
5. Exponential:      exp(K₁(x,x'))
6. Composition:      K₁(φ(x), φ(x'))              any map φ
7. Tensor product:   K₁(x,x') ⊗ K₂(y,y')          for x,x' ∈ X, y,y' ∈ Y
```

```python
import numpy as np

# ============================================================
# KERNEL COMPOSITION: BUILD NEW KERNELS
# ============================================================

def verify_composition_rule(K1, K2, composer, X, rule_name):
    """
    Verify that a composition of valid kernels is also valid.
    """
    print(f"\n--- Rule: {rule_name} ---")
    
    # Compose the kernel
    def K_composed(x, xp):
        return composer(K1(x, xp), K2(x, xp))
    
    # Check validity
    valid, min_eig, K_mat = is_valid_kernel(K_composed, X, rule_name)
    
    return valid


np.random.seed(42)
X_comp = np.random.randn(15, 2)

print("="*60)
print("KERNEL COMPOSITION RULES")
print("="*60)

# Rule 1: Scaling
print("\n1. SCALING: c · K(x,x')")
for c in [0.5, 1.0, 2.0, 10.0]:
    def composer(k1, k2):
        return c * k1
    verify_composition_rule(k_rbf, None, composer, X_comp,
                           f"c={c} · RBF")

# Rule 2: Sum
print("\n2. SUM: K₁(x,x') + K₂(x,x')")
def composer_sum(k1, k2):
    return k1 + k2
verify_composition_rule(k_linear, k_rbf, composer_sum, X_comp,
                       "Linear + RBF")
verify_composition_rule(k_rbf, k_poly, composer_sum, X_comp,
                       "RBF + Polynomial")

# Rule 3: Product
print("\n3. PRODUCT: K₁(x,x') · K₂(x,x')")
def composer_product(k1, k2):
    return k1 * k2
verify_composition_rule(k_linear, k_rbf, composer_product, X_comp,
                       "Linear × RBF")

# Rule 4: Polynomial
print("\n4. POLYNOMIAL: [K(x,x') + c]ᵈ")
for d in [2, 3, 4]:
    def composer_poly(k1, k2):
        return (k1 + 1.0)**d
    verify_composition_rule(k_linear, None, composer_poly, X_comp,
                           f"[Linear + 1]^{d}")

# Rule 5: Exponential
print("\n5. EXPONENTIAL: exp(K(x,x'))")
def composer_exp(k1, k2):
    return np.exp(k1)
verify_composition_rule(k_linear, None, composer_exp, X_comp,
                       "exp(Linear)")


# ============================================================
# PRACTICAL EXAMPLE: CUSTOM COMPOSED KERNEL
# ============================================================

print("\n\n" + "="*60)
print("PRACTICAL: BUILD A CUSTOM KERNEL")
print("="*60)

def custom_kernel(x, xp, alpha=0.5, gamma_rbf=1.0, degree=2):
    """
    Custom kernel combining linear, RBF, and polynomial:
    K(x,x') = α·(xᵀx' + 1)² + (1-α)·exp(-γ||x-x'||²)
    
    This allows:
    - Local smoothness (RBF term)
    - Global polynomial structure (poly term)
    - Adjustable mix (α parameter)
    """
    k_lin  = np.dot(x, xp)
    k_poly = (k_lin + 1)**degree
    k_rbf  = np.exp(-gamma_rbf * np.sum((x - xp)**2))
    
    return alpha * k_poly + (1 - alpha) * k_rbf

# Test validity
print("\nTesting custom kernel with different α:")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    def k_custom_alpha(x, xp):
        return custom_kernel(x, xp, alpha=alpha)
    
    is_valid_kernel(k_custom_alpha, X_comp,
                    f"Custom(α={alpha})")


# ============================================================
# VISUALIZE: Effect of mixing parameter α
# ============================================================

from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X_moon, y_moon = make_moons(n_samples=200, noise=0.2, random_state=42)
X_moon_s = StandardScaler().fit_transform(X_moon)

def kernel_matrix_custom(X1, X2, alpha):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i,j] = custom_kernel(X1[i], X2[j], alpha=alpha)
    return K

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, alpha in zip(axes, [0.0, 0.25, 0.5, 0.75, 1.0]):
    # Train with precomputed kernel
    K_train = kernel_matrix_custom(X_moon_s, X_moon_s, alpha)
    
    clf = SVC(kernel='precomputed', C=1.0)
    clf.fit(K_train, y_moon)
    
    # Predict
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100),
                         np.linspace(-3, 3, 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    K_grid = kernel_matrix_custom(X_grid, X_moon_s, alpha)
    Z = clf.predict(K_grid).reshape(xx.shape)
    
    acc = clf.score(K_train, y_moon)
    
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlBu')
    ax.scatter(X_moon_s[y_moon==0, 0], X_moon_s[y_moon==0, 1],
               c='royalblue', s=20, edgecolors='k', linewidths=0.3)
    ax.scatter(X_moon_s[y_moon==1, 0], X_moon_s[y_moon==1, 1],
               c='tomato', s=20, marker='s',
               edgecolors='k', linewidths=0.3)
    ax.set_title(f'α={alpha}\nAcc={acc*100:.0f}%')
    ax.grid(True, alpha=0.3)

plt.suptitle('Custom Kernel: K = α·Poly + (1-α)·RBF\n'
             'α=0 (pure RBF) → α=1 (pure polynomial)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('custom_kernel_mixture.png', dpi=150, bbox_inches='tight')
print("\nSaved: custom_kernel_mixture.png")
```

---

## Feature Map Interpretations

### Explicit vs Implicit Feature Maps

```
EXPLICIT: φ(x) is computed directly
  Example: Polynomial degree 2
  φ(x₁, x₂) = [x₁², √2·x₁x₂, x₂²]
  Cost: O(d_φ) where d_φ = dimension of feature space

IMPLICIT: φ(x) never computed, only K(x,x') = ⟨φ(x), φ(x')⟩
  Example: RBF kernel
  φ(x) has INFINITE dimensions!
  Cost: O(d) where d = input dimension
```

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# FEATURE MAP VISUALIZATION
# ============================================================

def explicit_poly_map_2d(x, degree=2):
    """
    Explicit polynomial feature map for 2D input.
    Returns φ(x) such that K(x,x') = (xᵀx' + 1)ᵈ = ⟨φ(x), φ(x')⟩
    """
    x1, x2 = x[0], x[1]
    
    if degree == 2:
        # φ(x) = [1, √2·x₁, √2·x₂, x₁², √2·x₁x₂, x₂²]
        return np.array([
            1.0,
            np.sqrt(2) * x1,
            np.sqrt(2) * x2,
            x1**2,
            np.sqrt(2) * x1 * x2,
            x2**2
        ])
    elif degree == 3:
        # Even more terms for degree 3
        return np.array([
            1.0,
            np.sqrt(3) * x1,
            np.sqrt(3) * x2,
            np.sqrt(3) * x1**2,
            np.sqrt(6) * x1 * x2,
            np.sqrt(3) * x2**2,
            x1**3,
            np.sqrt(3) * x1**2 * x2,
            np.sqrt(3) * x1 * x2**2,
            x2**3
        ])
    else:
        raise ValueError("Only degree 2 and 3 implemented")


print("="*60)
print("EXPLICIT FEATURE MAPS")
print("="*60)

# Verify explicit map matches kernel
x  = np.array([1.0, 2.0])
xp = np.array([3.0, -1.0])

for d in [2, 3]:
    phi_x  = explicit_poly_map_2d(x, degree=d)
    phi_xp = explicit_poly_map_2d(xp, degree=d)
    
    # Via explicit feature map
    k_explicit = np.dot(phi_x, phi_xp)
    
    # Via kernel function
    k_formula  = (np.dot(x, xp) + 1)**d
    
    print(f"\nDegree {d}:")
    print(f"  Feature dim: {len(phi_x)}")
    print(f"  ⟨φ(x), φ(x')⟩: {k_explicit:.6f}")
    print(f"  K(x, x'):      {k_formula:.6f}")
    print(f"  Match: {np.isclose(k_explicit, k_formula)}")


# ============================================================
# VISUALIZE: Data in feature space
# ============================================================

# 2D data
np.random.seed(42)
X_orig = np.array([
    [-1, -1], [- 1, 1], [1, -1], [1, 1],
    [0, 0], [0.5, 0.5], [-0.5, -0.5]
])
colors = ['r', 'r', 'b', 'b', 'g', 'g', 'g']

# Map to 3D via polynomial degree 2
# Use first 3 components for visualization
phi_3d = np.array([explicit_poly_map_2d(x, degree=2)[:3]
                   for x in X_orig])

fig = plt.figure(figsize=(14, 6))

# Original 2D
ax1 = fig.add_subplot(121)
for x, c in zip(X_orig, colors):
    ax1.scatter(x[0], x[1], c=c, s=100, edgecolors='k', linewidths=1.5)
ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.set_title('Original Input Space (2D)')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Transformed 3D (first 3 dims only)
ax2 = fig.add_subplot(122, projection='3d')
for phi, c in zip(phi_3d, colors):
    ax2.scatter(phi[0], phi[1], phi[2],
                c=c, s=100, edgecolors='k', linewidths=1.5)
ax2.set_xlabel('φ₁ (constant)')
ax2.set_ylabel('φ₂ (√2·x₁)')
ax2.set_zlabel('φ₃ (√2·x₂)')
ax2.set_title('Feature Space (first 3 of 6 dims)')

plt.suptitle('Polynomial Feature Map Visualization\n'
             'φ(x) = [1, √2·x₁, √2·x₂, x₁², √2·x₁x₂, x₂²]',
             fontsize=12)
plt.tight_layout()
plt.savefig('explicit_feature_map.png', dpi=150, bbox_inches='tight')
print("\nSaved: explicit_feature_map.png")
```

### Dimensionality Growth

```python
from math import comb

# ============================================================
# FEATURE SPACE DIMENSION EXPLOSION
# ============================================================

print("\n" + "="*60)
print("FEATURE SPACE DIMENSIONALITY")
print("="*60)

input_dims = [2, 5, 10, 20, 50, 100]
degrees    = [2, 3, 4, 5, 6]

print(f"\n{'Input d':<12}", end="")
for deg in degrees:
    print(f"{'Degree ' + str(deg):<15}", end="")
print()
print("-" * 80)

for d in input_dims:
    print(f"{d:<12}", end="")
    for deg in degrees:
        # Number of monomials of degree ≤ deg in d variables
        # = C(d + deg, deg)
        n_features = comb(d + deg, deg)
        if n_features < 1000:
            print(f"{n_features:<15}", end="")
        elif n_features < 1_000_000:
            print(f"{n_features/1000:.1f}K{'':<10}", end="")
        else:
            print(f"{n_features/1_000_000:.1f}M{'':<10}", end="")
    print()

print("\nObservation: Exponential growth makes explicit maps impractical!")
print("→ This is why the kernel trick is essential")
```

---

## Kernel Alignment and Similarity

### Kernel Target Alignment

**Question**: How well does a kernel K match the task (target labels)?

**Kernel Alignment** measures the similarity between kernel matrix K and the ideal kernel Y·Yᵀ:

```
Alignment(K, y) = ⟨K, y·yᵀ⟩_F / (||K||_F · ||y·yᵀ||_F)

where ⟨·,·⟩_F is Frobenius inner product: ⟨A,B⟩_F = Σᵢⱼ Aᵢⱼ Bᵢⱼ
      ||A||_F is Frobenius norm: ||A||_F = √(Σᵢⱼ Aᵢⱼ²)

Range: [-1, +1]
  +1 = perfect alignment (kernel perfectly matches labels)
   0 = no correlation
  -1 = perfectly anti-aligned
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# ============================================================
# KERNEL ALIGNMENT
# ============================================================

def kernel_target_alignment(K, y):
    """
    Compute alignment between kernel matrix K and target y.
    
    Returns:
        alignment (float): Value in [-1, +1]
    """
    n = len(y)
    
    # Center y to {-1, +1}
    y_pm = np.where(y == 0, -1, 1).reshape(-1, 1)
    
    # Ideal kernel: Y·Yᵀ
    Y = y_pm @ y_pm.T
    
    # Frobenius inner product
    numerator   = np.sum(K * Y)
    denominator = np.sqrt(np.sum(K**2)) * np.sqrt(np.sum(Y**2))
    
    alignment = numerator / (denominator + 1e-10)
    
    return alignment


def compute_all_alignments(X, y, kernels_dict):
    """Compute alignment for multiple kernels."""
    results = {}
    
    for name, k_func in kernels_dict.items():
        n = len(X)
        K = np.array([[k_func(X[i], X[j]) for j in range(n)]
                      for i in range(n)])
        align = kernel_target_alignment(K, y)
        results[name] = align
    
    return results


print("="*60)
print("KERNEL-TARGET ALIGNMENT")
print("="*60)

# Test on different datasets
datasets = [
    ("Linear Sep", *make_classification(n_samples=100, n_features=2,
                                        n_redundant=0, n_informative=2,
                                        class_sep=2.5, random_state=42)),
    ("Moons", *make_moons(n_samples=100, noise=0.2, random_state=42)),
    ("Circles", *make_circles(n_samples=100, noise=0.1,
                              factor=0.4, random_state=42)),
]

kernels_test = {
    'Linear':    lambda x, xp: np.dot(x, xp),
    'Poly d=2':  lambda x, xp: (np.dot(x, xp) + 1)**2,
    'Poly d=3':  lambda x, xp: (np.dot(x, xp) + 1)**3,
    'RBF γ=0.5': lambda x, xp: np.exp(-0.5 * np.sum((x-xp)**2)),
    'RBF γ=1.0': lambda x, xp: np.exp(-1.0 * np.sum((x-xp)**2)),
    'RBF γ=2.0': lambda x, xp: np.exp(-2.0 * np.sum((x-xp)**2)),
}

all_results = {}

for ds_name, X_ds, y_ds in datasets:
    X_s = StandardScaler().fit_transform(X_ds)
    alignments = compute_all_alignments(X_s, y_ds, kernels_test)
    all_results[ds_name] = alignments
    
    print(f"\n--- {ds_name} ---")
    for k_name in sorted(alignments, key=alignments.get, reverse=True):
        align = alignments[k_name]
        print(f"  {k_name:<15}: {align:>7.4f}")


# ============================================================
# VISUALIZE ALIGNMENT
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, ds_name in zip(axes, all_results.keys()):
    alignments = all_results[ds_name]
    
    names  = list(alignments.keys())
    values = [alignments[n] for n in names]
    colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red'
              for v in values]
    
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.7,
                   edgecolor='k')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Alignment')
    ax.set_title(f'{ds_name}\n(Higher = better match to labels)')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-0.2, 1.0)

plt.suptitle('Kernel-Target Alignment Comparison\n'
             'Green (>0.7) = Excellent, Orange (>0.4) = Good, Red = Poor',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('kernel_alignment.png', dpi=150, bbox_inches='tight')
print("\nSaved: kernel_alignment.png")


print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("""
Kernel-target alignment helps you choose the right kernel BEFORE training:

1. Compute alignment for candidate kernels
2. Choose kernel with highest alignment
3. This kernel is most "compatible" with the classification task

Note: High alignment ≠ guaranteed good performance
      But it's a useful heuristic for kernel selection
""")
```

---

## Infinite-Dimensional Feature Spaces

### RBF Kernel: The Infinite-Dimensional Case

```
RBF Kernel: K(x, x') = exp(-γ||x - x'||²)

Taylor expansion of exp():
  exp(z) = 1 + z + z²/2! + z³/3! + ...

This gives an INFINITE series of polynomial terms!
→ Feature map φ(x) has infinitely many dimensions

Yet K(x, x') can be computed in O(d) time where d = input dimension!
```

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# RBF: INFINITE FEATURE SPACE
# ============================================================

print("="*60)
print("RBF KERNEL: INFINITE-DIMENSIONAL FEATURE SPACE")
print("="*60)

print("""
RBF kernel: K(x, x') = exp(-γ||x - x'||²)

Expanding:
  exp(-γ||x - x'||²)
  = exp(-γ||x||²) · exp(2γ xᵀx') · exp(-γ||x'||²)

The middle term exp(2γ xᵀx') has Taylor expansion:
  exp(2γ xᵀx') = Σₙ₌₀^∞ (2γ)ⁿ/(n!) · (xᵀx')ⁿ

This is a weighted sum of ALL polynomial degrees!
→ Infinite-dimensional feature space

Each degree n contributes with weight (2γ)ⁿ/n!
""")

# Show weight decay
gammas = [0.1, 0.5, 1.0, 2.0]
degrees = np.arange(0, 20)

plt.figure(figsize=(10, 6))

for gamma in gammas:
    weights = [(2*gamma)**n / np.math.factorial(n)
               for n in degrees]
    plt.plot(degrees, weights, 'o-', linewidth=2,
             label=f'γ = {gamma}')

plt.xlabel('Polynomial Degree n')
plt.ylabel('Weight = (2γ)ⁿ / n!')
plt.title('RBF Kernel: Contribution by Polynomial Degree\n'
          'All degrees contribute → infinite-dimensional')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rbf_infinite_weights.png', dpi=150, bbox_inches='tight')
print("Saved: rbf_infinite_weights.png")

print("\nKey insight:")
print("  Smaller γ → more weight on higher degrees → more complex")
print("  Larger γ → weight concentrated on lower degrees → simpler")


# ============================================================
# FINITE APPROXIMATION
# ============================================================

print("\n\n" + "="*60)
print("FINITE-RANK APPROXIMATION OF RBF")
print("="*60)

def rbf_truncated(x, xp, gamma=1.0, max_degree=5):
    """
    Approximate RBF using first max_degree terms of Taylor series.
    """
    dot = np.dot(x, xp)
    
    # exp(-γ||x||²) * exp(-γ||x'||²) * Σ (2γ)ⁿ/n! (xᵀx')ⁿ
    norm_x  = np.exp(-gamma * np.dot(x, x))
    norm_xp = np.exp(-gamma * np.dot(xp, xp))
    
    poly_sum = sum((2*gamma)**n / np.math.factorial(n) * dot**n
                   for n in range(max_degree + 1))
    
    return norm_x * norm_xp * poly_sum


# Compare exact vs truncated
np.random.seed(42)
X_approx = np.random.randn(10, 2)

print(f"\n{'Degree limit':<15} {'Max |error|':<15} {'Mean |error|'}")
print("-" * 45)

for max_deg in [1, 2, 3, 5, 10, 15, 20]:
    errors = []
    for i in range(len(X_approx)):
        for j in range(len(X_approx)):
            k_exact = k_rbf(X_approx[i], X_approx[j])
            k_trunc = rbf_truncated(X_approx[i], X_approx[j],
                                   gamma=0.5, max_degree=max_deg)
            errors.append(abs(k_exact - k_trunc))
    
    print(f"{max_deg:<15} {max(errors):<15.2e} {np.mean(errors):.2e}")

print("\nConclusion: Even with 10-20 terms, approximation is excellent!")
print("But RBF kernel computes exact value in O(d) time")
```

---

## Normalized Kernels

### Why Normalize?

Normalizing a kernel ensures K(x,x) = 1 for all x:

```
K_normalized(x, x') = K(x, x') / √(K(x,x) · K(x',x'))

Benefits:
  - Scale-invariant: K̃(cx, cx') = K̃(x, x') for any c > 0
  - Bounded: -1 ≤ K̃(x, x') ≤ 1
  - Interpretable as cosine similarity in feature space
```

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# KERNEL NORMALIZATION
# ============================================================

def normalize_kernel(K_func):
    """
    Return normalized version of a kernel function.
    K̃(x,x') = K(x,x') / √(K(x,x)·K(x',x'))
    """
    def K_normalized(x, xp):
        k_xx  = K_func(x, x)
        k_xpxp = K_func(xp, xp)
        k_xxp  = K_func(x, xp)
        return k_xxp / (np.sqrt(k_xx * k_xpxp) + 1e-10)
    
    return K_normalized


print("="*60)
print("KERNEL NORMALIZATION")
print("="*60)

# Original kernels
k_poly_unnorm = lambda x, xp: (np.dot(x, xp) + 1)**2
k_rbf_unnorm  = lambda x, xp: np.exp(-0.5 * np.sum((x - xp)**2))

# Normalized versions
k_poly_norm = normalize_kernel(k_poly_unnorm)
k_rbf_norm  = normalize_kernel(k_rbf_unnorm)

# Test points
np.random.seed(42)
x_test  = np.array([1.0, 2.0])
xp_test = np.array([3.0, -1.0])

print("\nPolynomial Kernel (degree 2):")
print(f"  Original K(x,x'):    {k_poly_unnorm(x_test, xp_test):.4f}")
print(f"  Normalized K̃(x,x'): {k_poly_norm(x_test, xp_test):.4f}")
print(f"  K̃(x,x):             {k_poly_norm(x_test, x_test):.4f}  (= 1 always)")
print(f"  K̃(x',x'):           {k_poly_norm(xp_test, xp_test):.4f}  (= 1 always)")

print("\nRBF Kernel:")
print(f"  Original K(x,x'):    {k_rbf_unnorm(x_test, xp_test):.4f}")
print(f"  Normalized K̃(x,x'): {k_rbf_norm(x_test, xp_test):.4f}")
print(f"  K̃(x,x):             {k_rbf_norm(x_test, x_test):.4f}  (= 1 always)")


# ============================================================
# VISUALIZATION: Before vs After Normalization
# ============================================================

X_norm = np.random.randn(20, 2) * np.array([1, 5])  # Different scales

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for col, (name, k_orig, k_norm) in enumerate([
    ("Polynomial", k_poly_unnorm, k_poly_norm),
    ("RBF", k_rbf_unnorm, k_rbf_norm),
]):
    # Original kernel matrix
    K_orig = np.array([[k_orig(X_norm[i], X_norm[j])
                        for j in range(len(X_norm))]
                       for i in range(len(X_norm))])
    
    # Normalized kernel matrix
    K_norm_mat = np.array([[k_norm(X_norm[i], X_norm[j])
                            for j in range(len(X_norm))]
                           for i in range(len(X_norm))])
    
    # Plot original
    ax = axes[0, col]
    im = ax.imshow(K_orig, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f'{name}\nOriginal')
    
    diag_orig = np.diag(K_orig)
    ax.text(0.5, -0.15, f'Diagonal range: [{diag_orig.min():.2f}, {diag_orig.max():.2f}]',
            transform=ax.transAxes, ha='center', fontsize=9)
    
    # Plot normalized
    ax = axes[1, col]
    im = ax.imshow(K_norm_mat, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f'{name}\nNormalized')
    
    diag_norm = np.diag(K_norm_mat)
    ax.text(0.5, -0.15, f'Diagonal: all = 1.00',
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Comparison panel
ax = axes[0, 2]
ax.text(0.5, 0.5,
        "NORMALIZATION EFFECT:\n\n"
        "Before:\n"
        "• K(x,x) varies\n"
        "• Scale-dependent\n"
        "• Unbounded range\n\n"
        "After:\n"
        "• K̃(x,x) = 1 always\n"
        "• Scale-invariant\n"
        "• Range: [-1, +1]\n"
        "• = cosine similarity",
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.axis('off')

ax = axes[1, 2]
ax.axis('off')

plt.suptitle('Kernel Normalization',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('kernel_normalization.png', dpi=150, bbox_inches='tight')
print("\nSaved: kernel_normalization.png")
```

---

## Stationary vs Non-Stationary Kernels

### Definitions

```
STATIONARY KERNEL:
  K(x, x') = f(x - x')  or  f(||x - x'||)
  
  Depends only on the difference or distance between points
  → Translation-invariant: K(x+c, x'+c) = K(x, x')
  
  Examples: RBF, Laplacian, Matérn

NON-STATIONARY KERNEL:
  K(x, x') = f(x, x')  (general function of both arguments)
  
  Can vary across input space
  
  Examples: Linear, Polynomial, Neural network kernels
```

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STATIONARY VS NON-STATIONARY
# ============================================================

print("="*60)
print("STATIONARY VS NON-STATIONARY KERNELS")
print("="*60)

# Test translation invariance
x1  = np.array([0.0, 0.0])
x1p = np.array([1.0, 1.0])

shift = np.array([10.0, 5.0])
x2  = x1 + shift
x2p = x1p + shift

print("\nOriginal points:")
print(f"  x = {x1}, x' = {x1p}")
print(f"\nShifted points (+{shift}):")
print(f"  x = {x2}, x' = {x2p}")

kernels_test_stat = [
    ("RBF (stationary)", k_rbf, True),
    ("Laplacian (stationary)", k_laplacian, True),
    ("Linear (non-stationary)", k_linear, False),
    ("Polynomial (non-stationary)", k_poly, False),
]

print(f"\n{'Kernel':<30} {'K(x,x\')':<12} {'K(x+c,x\'+c)':<15} {'Invariant?'}")
print("-"*70)

for name, k_fn, expected_stat in kernels_test_stat:
    k_orig = k_fn(x1, x1p)
    k_shift = k_fn(x2, x2p)
    is_invariant = np.isclose(k_orig, k_shift)
    
    status = "✓" if is_invariant == expected_stat else "✗"
    print(f"{name:<30} {k_orig:<12.4f} {k_shift:<15.4f} {status}")


# ============================================================
# VISUALIZE: Kernel values across space
# ============================================================

# Fix one point at origin
x_fixed = np.array([0.0, 0.0])

# Scan over a grid
x_range = np.linspace(-3, 3, 50)
X_grid, Y_grid = np.meshgrid(x_range, x_range)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

kernels_viz_stat = [
    ("RBF (Stationary)", k_rbf),
    ("Laplacian (Stationary)", k_laplacian),
    ("Linear (Non-stationary)", k_linear),
    ("Polynomial d=2 (Non-stationary)", k_poly),
]

for ax, (name, k_fn) in zip(axes.ravel(), kernels_viz_stat):
    K_vals = np.zeros_like(X_grid)
    for i in range(len(x_range)):
        for j in range(len(x_range)):
            x_test = np.array([X_grid[j, i], Y_grid[j, i]])
            K_vals[j, i] = k_fn(x_fixed, x_test)
    
    cf = ax.contourf(X_grid, Y_grid, K_vals, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax)
    ax.contour(X_grid, Y_grid, K_vals, levels=10,
               colors='white', linewidths=0.5, alpha=0.5)
    ax.plot([0], [0], 'r*', markersize=20, label='Fixed point x=(0,0)')
    ax.set_xlabel('x\'₁')
    ax.set_ylabel('x\'₂')
    ax.set_title(f'{name}\nK(x=(0,0), x\')')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

plt.suptitle('Stationary vs Non-Stationary Kernels\n'
             'Stationary = circular/radial contours (translation-invariant)\n'
             'Non-stationary = arbitrary contours',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('stationary_vs_nonstationary.png', dpi=150, bbox_inches='tight')
print("\nSaved: stationary_vs_nonstationary.png")
```

---

## Universal Kernels

### Definition

A kernel K is **universal** if the RKHS it induces is dense in C(X), the space of continuous functions on a compact domain X.

**Practical meaning**: A universal kernel can approximate any continuous function arbitrarily well (given enough data).

**Universal kernels** include:
- RBF (Gaussian)
- Laplacian
- Matérn (for ν > 0)

**Not universal**:
- Linear
- Polynomial (finite degree)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

# ============================================================
# UNIVERSAL KERNELS: APPROXIMATION POWER
# ============================================================

print("="*60)
print("UNIVERSAL KERNELS")
print("="*60)

print("""
Theorem (Micchelli, 1986; Steinwart, 2001):
  The RBF kernel is universal on ℝᵈ for any γ > 0.
  
  This means: Given enough data and proper regularization,
              RBF-based methods can approximate any continuous
              function arbitrarily well.

This is why RBF is the default choice for kernel methods!
""")

# Test approximation of a complex function
np.random.seed(42)
n_train = 50

def complex_function(x):
    """A complex target function to approximate."""
    return np.sin(4*x) + 0.3*np.sin(12*x) + 0.5*np.cos(6*x)

X_train = np.sort(np.random.uniform(0, 5, n_train)).reshape(-1, 1)
y_train = complex_function(X_train.ravel()) + np.random.randn(n_train) * 0.1

X_test = np.linspace(0, 5, 300).reshape(-1, 1)
y_true = complex_function(X_test.ravel())

# Try different kernels
kernels_univ = [
    ('Linear (not universal)', 'linear', {}),
    ('Polynomial d=3 (not universal)', 'poly', {'degree': 3, 'coef0': 1}),
    ('RBF γ=1 (universal)', 'rbf', {'gamma': 1.0}),
    ('RBF γ=5 (universal)', 'rbf', {'gamma': 5.0}),
    ('Laplacian (universal)', 'laplacian', {}),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for ax, (name, kernel, params) in zip(axes, kernels_univ):
    if kernel == 'laplacian':
        # sklearn doesn't have laplacian, use additive_chi2 as proxy
        # or implement custom
        kr = KernelRidge(kernel='rbf', alpha=0.01, gamma=1.0)
    else:
        kr = KernelRidge(kernel=kernel, alpha=0.01, **params)
    
    kr.fit(X_train, y_train)
    y_pred = kr.predict(X_test)
    
    mse = np.mean((y_pred - y_true)**2)
    
    ax.scatter(X_train, y_train, c='gray', s=20, alpha=0.7,
               label='Training data')
    ax.plot(X_test, y_true, 'g-', linewidth=2, label='True function')
    ax.plot(X_test, y_pred, 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{name}\nMSE = {mse:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Summary panel
ax = axes[-1]
ax.text(0.5, 0.5,
        "UNIVERSAL KERNELS:\n\n"
        "Can approximate\nANY continuous function\n\n"
        "→ RBF succeeds where\n"
        "   linear & polynomial fail\n\n"
        "→ This is theoretical guarantee\n"
        "   of approximation power\n\n"
        "→ Practical: use RBF as default!",
        ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax.axis('off')

plt.suptitle('Universal Kernels: Function Approximation Power\n'
             'Universal kernels can approximate any continuous function',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('universal_kernels.png', dpi=150, bbox_inches='tight')
print("\nSaved: universal_kernels.png")
```

---

## Summary and Quick Reference

### Key Concepts Recap

```python
print("="*70)
print("KERNEL FUNCTIONS: KEY CONCEPTS")
print("="*70)

concepts = """
1. VALID KERNEL ⟺ POSITIVE SEMI-DEFINITE
   - Kernel matrix K has all eigenvalues ≥ 0
   - Equivalently: there exists feature map φ such that K(x,x') = ⟨φ(x),φ(x')⟩

2. ESSENTIAL PROPERTIES
   - Symmetry: K(x,x') = K(x',x)
   - PSD: vᵀKv ≥ 0 for all v
   - Cauchy-Schwarz: |K(x,x')|² ≤ K(x,x)·K(x',x')

3. COMPOSITION RULES
   - Sum: K₁ + K₂
   - Product: K₁ · K₂
   - Scaling: c·K (c > 0)
   - Polynomial: (K + c)ᵈ
   - Exponential: exp(K)
   → All preserve validity!

4. FEATURE MAPS
   - Explicit (polynomial): compute φ(x) directly
   - Implicit (RBF): φ(x) has infinite dimensions, never computed

5. NORMALIZATION
   - K̃(x,x') = K(x,x') / √(K(x,x)·K(x',x'))
   - Ensures K̃(x,x) = 1
   - Bounded: -1 ≤ K̃ ≤ 1

6. STATIONARY KERNELS
   - K(x,x') = f(x - x')
   - Translation-invariant
   - Examples: RBF, Laplacian, Matérn

7. UNIVERSAL KERNELS
   - Can approximate any continuous function
   - RBF, Laplacian are universal
   - Linear, polynomial are NOT universal

8. KERNEL ALIGNMENT
   - Measures how well K matches target labels
   - Range: [-1, +1]
   - Useful for kernel selection
"""

print(concepts)
```

### Prerequisites Checklist

Before moving to Chapter 06, confirm you can:
- [ ] Explain the three equivalent conditions for kernel validity
- [ ] Check if a kernel is PSD using eigenvalues
- [ ] Apply at least 3 composition rules to build new kernels
- [ ] Explain the difference between explicit and implicit feature maps
- [ ] Compute kernel-target alignment
- [ ] Explain why RBF has infinite-dimensional feature space
- [ ] Distinguish between stationary and non-stationary kernels
- [ ] Explain what makes a kernel "universal"

### What's Next

**06_Radial_Basis_Functions.md** ← already exists  
→ Deep dive into RBF kernels, gamma selection, and properties

**07_SVM_Hard_Margin.md**  
→ Complete SVM optimization with hard-margin constraints

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════╗
║            KERNEL FUNCTIONS — QUICK REFERENCE                    ║
╠══════════════════════════════════════════════════════════════════╣
║  VALID KERNEL CONDITIONS (any one sufficient)                    ║
║    1. Kernel matrix K is PSD (Mercer's theorem)                  ║
║    2. K(x,x') = ⟨φ(x), φ(x')⟩ for some φ                        ║
║    3. All eigenvalues of K ≥ 0                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  COMPOSITION RULES (K₁, K₂ valid → result valid)                ║
║    c·K₁             (c > 0)                                      ║
║    K₁ + K₂                                                       ║
║    K₁ · K₂                                                       ║
║    (K₁ + c)ᵈ        (c ≥ 0, d ∈ ℤ⁺)                           ║
║    exp(K₁)                                                       ║
╠══════════════════════════════════════════════════════════════════╣
║  KERNEL PROPERTIES                                               ║
║    Symmetry:        K(x,x') = K(x',x)                            ║
║    PSD:             vᵀKv ≥ 0  ∀v                                 ║
║    Cauchy-Schwarz:  |K(x,x')|² ≤ K(x,x)·K(x',x')               ║
╠══════════════════════════════════════════════════════════════════╣
║  NORMALIZATION                                                   ║
║    K̃(x,x') = K(x,x') / √(K(x,x)·K(x',x'))                      ║
║    → K̃(x,x) = 1 always                                          ║
║    → Range: [-1, +1]                                             ║
╠══════════════════════════════════════════════════════════════════╣
║  KERNEL TYPES                                                    ║
║    Stationary:     K(x,x') = f(||x-x'||)                        ║
║                    Examples: RBF, Laplacian                      ║
║    Non-stationary: K(x,x') = f(x,x')                            ║
║                    Examples: Linear, Polynomial                  ║
║    Universal:      Can approximate any continuous function       ║
║                    Examples: RBF, Laplacian (NOT linear/poly)   ║
╠══════════════════════════════════════════════════════════════════╣
║  KERNEL ALIGNMENT                                                ║
║    A(K,y) = ⟨K, yyᵀ⟩_F / (||K||_F · ||yyᵀ||_F)                ║
║    Range: [-1, +1]                                               ║
║    Use: Select kernel BEFORE training                            ║
╠══════════════════════════════════════════════════════════════════╣
║  PYTHON CHECK VALIDITY                                           ║
║    eigvals = np.linalg.eigvalsh(K)                               ║
║    is_valid = eigvals.min() >= -1e-8                             ║
╚══════════════════════════════════════════════════════════════════╝
```

---

*End of 05_Kernel_Functions.md*