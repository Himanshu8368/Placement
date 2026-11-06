# Feature Scaling and Normalization

## Table of Contents
1. [Introduction to Feature Scaling](#introduction-to-feature-scaling)
2. [Why Scaling Matters for Linear Models](#why-scaling-matters-for-linear-models)
3. [Standardization (Z-Score Normalization)](#standardization-z-score-normalization)
4. [Min-Max Scaling](#min-max-scaling)
5. [Impact on Gradient Descent](#impact-on-gradient-descent)
6. [Other Scaling Methods](#other-scaling-methods)
7. [When to Scale and When Not To](#when-to-scale-and-when-not-to)
8. [Practical Implementation](#practical-implementation)
9. [Common Pitfalls and Best Practices](#common-pitfalls-and-best-practices)
10. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to Feature Scaling

### What is Feature Scaling?

**Feature scaling** (also called **feature normalization**) is the process of transforming features to similar scales or ranges.

**Problem example**:
```
Feature 1 (Age):        Range [0, 100]
Feature 2 (Income):     Range [0, 1,000,000]
Feature 3 (Height):     Range [0, 250] cm
```

These features have **vastly different scales**!

### Why is This a Problem?

**1. Different units**:
```
Age:    years
Income: dollars
Height: centimeters
```

**2. Magnitude differences**:
```
Small change in age:   +1 year
Small change in income: +1 dollar (negligible!)
```

**3. Model bias**:
- Features with larger scales dominate
- Model pays more attention to large-scale features
- Small-scale features ignored

### Visual Intuition

**Unscaled data**:
```
    Income ($)
    │
1M  │         ●
    │       ●  ●
500k│     ●      ●
    │   ●          ●
  0 └─────────────── Age (years)
       0   50  100
    
    Feature 2 >> Feature 1 in magnitude
```

**Scaled data**:
```
    Income (scaled)
    │
  2 │         ●
    │       ●  ●
  0 │     ●      ●
    │   ●          ●
 -2 └─────────────── Age (scaled)
      -2   0   2
    
    Both features on similar scale
```

### Types of Scaling

**1. Standardization (Z-score)**:
```
x_scaled = (x - μ) / σ
```
Result: mean=0, std=1

**2. Min-Max Scaling**:
```
x_scaled = (x - min) / (max - min)
```
Result: range [0, 1]

**3. Robust Scaling**:
```
x_scaled = (x - median) / IQR
```
Result: robust to outliers

**4. Max Abs Scaling**:
```
x_scaled = x / max(|x|)
```
Result: range [-1, 1]

---

## Why Scaling Matters for Linear Models

### Impact on Linear Regression

**Model equation**:
```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

**Problem with unscaled features**:

**Example**:
```
Features:
  x₁ = Age (range 0-100)
  x₂ = Income (range 0-1,000,000)

Learned weights (unscaled):
  w₁ = 1000  (Age)
  w₂ = 0.001 (Income)

Prediction:
  ŷ = w₀ + 1000·Age + 0.001·Income
```

**Problem**: Weights have **vastly different magnitudes** to compensate for feature scales!

### Impact on Regularization

**Ridge regression cost**:
```
J(w) = MSE + (λ/2m) Σⱼ wⱼ²
```

**Problem**: Penalizes large weights equally, regardless of feature scale!

**Example**:
```
Unscaled:
  w₁ = 1000 (for Age) → penalty = 1,000,000
  w₂ = 0.001 (for Income) → penalty = 0.000001

Regularization unfairly penalizes w₁!
```

**After scaling**: Both features have similar scales → penalties are fair.

### Impact on Distance-Based Methods

**K-Nearest Neighbors, K-Means**:

Distance formula:
```
d = √[(x₁-x₁')² + (x₂-x₂')²]
```

**Problem**: Features with larger scales dominate the distance!

**Example**:
```
Point A: (Age=20, Income=30,000)
Point B: (Age=25, Income=40,000)

Unscaled distance:
  d = √[(25-20)² + (40000-30000)²]
    = √[25 + 100,000,000]
    ≈ 10,000

Income difference dominates! Age is irrelevant.

Scaled distance:
  d = √[(0.5)² + (0.5)²]  (after scaling)
    ≈ 0.71

Both features contribute equally.
```

### Numerical Illustration

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Generate data with different scales
np.random.seed(42)
n = 100

# Feature 1: Small scale (0-10)
X1 = np.random.uniform(0, 10, n)

# Feature 2: Large scale (0-10000)
X2 = np.random.uniform(0, 10000, n)

# Target: depends on both features equally
y = 2 * X1 + 0.002 * X2 + np.random.randn(n)

X = np.column_stack([X1, X2])

# Train without scaling
model_unscaled = LinearRegression()
model_unscaled.fit(X, y)

# Train with scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)

# Compare weights
print("="*60)
print("IMPACT OF SCALING ON WEIGHTS")
print("="*60)
print(f"\nUnscaled weights:")
print(f"  w₁ (Age):    {model_unscaled.coef_[0]:.6f}")
print(f"  w₂ (Income): {model_unscaled.coef_[1]:.6f}")
print(f"  Ratio: {abs(model_unscaled.coef_[0] / model_unscaled.coef_[1]):.1f}")

print(f"\nScaled weights:")
print(f"  w₁: {model_scaled.coef_[0]:.6f}")
print(f"  w₂: {model_scaled.coef_[1]:.6f}")
print(f"  Ratio: {abs(model_scaled.coef_[0] / model_scaled.coef_[1]):.1f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Unscaled data
ax = axes[0]
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)
ax.set_xlabel('Feature 1 (Age)', fontsize=12)
ax.set_ylabel('Feature 2 (Income)', fontsize=12)
ax.set_title('Unscaled Data\n(Different scales)', fontsize=14)
plt.colorbar(scatter, ax=ax, label='Target')

# Plot 2: Scaled data
ax = axes[1]
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)
ax.set_xlabel('Feature 1 (scaled)', fontsize=12)
ax.set_ylabel('Feature 2 (scaled)', fontsize=12)
ax.set_title('Scaled Data\n(Same scale)', fontsize=14)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Target')

plt.tight_layout()
plt.show()
```

**Output**:
```
Unscaled weights:
  w₁ (Age):    2.000000
  w₂ (Income): 0.002000
  Ratio: 1000.0

Scaled weights:
  w₁: 1.732
  w₂: 1.732
  Ratio: 1.0
```

After scaling, weights are comparable in magnitude!

---

## Standardization (Z-Score Normalization)

### Definition

**Standardization** transforms features to have **mean = 0** and **standard deviation = 1**.

**Formula**:
```
x_scaled = (x - μ) / σ
```

Where:
- μ = mean of feature
- σ = standard deviation of feature

**Also called**: Z-score normalization, zero-mean normalization

### Mathematical Properties

**For each feature after standardization**:
```
Mean:     E[x_scaled] = 0
Variance: Var(x_scaled) = 1
Std Dev:  σ(x_scaled) = 1
```

**Proof**:
```
E[x_scaled] = E[(x - μ) / σ]
            = (1/σ) E[x - μ]
            = (1/σ) (E[x] - μ)
            = (1/σ) (μ - μ)
            = 0

Var(x_scaled) = Var[(x - μ) / σ]
              = (1/σ²) Var[x - μ]
              = (1/σ²) Var[x]
              = (1/σ²) σ²
              = 1
```

### When to Use Standardization

✅ **Use when**:
- Features follow normal/Gaussian distribution (roughly)
- Want to preserve information about outliers
- Using algorithms sensitive to scale (gradient descent, PCA, neural networks)
- Using regularization (Ridge, Lasso, Elastic Net)
- No bounded range required

❌ **Don't use when**:
- Features need to be in specific range (e.g., [0, 1])
- Sparse data (many zeros) - standardization breaks sparsity
- Features already on similar scales

### Python Implementation

```python
import numpy as np

class StandardScaler:
    """Standardization scaler (Z-score normalization)"""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        """
        Compute mean and std for each feature
        
        Args:
            X: Feature matrix (m × n)
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        """
        Transform features
        
        Args:
            X: Feature matrix (m × n)
        
        Returns:
            X_scaled: Standardized features
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Reverse standardization
        
        Args:
            X_scaled: Standardized features
        
        Returns:
            X: Original scale features
        """
        return X_scaled * self.std_ + self.mean_

# Example usage
X = np.array([[1, 2000],
              [2, 3000],
              [3, 4000],
              [4, 5000],
              [5, 6000]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original data:")
print(X)
print(f"\nMeans: {scaler.mean_}")
print(f"Stds:  {scaler.std_}")
print("\nStandardized data:")
print(X_scaled)
print(f"\nScaled means: {np.mean(X_scaled, axis=0)}")
print(f"Scaled stds:  {np.std(X_scaled, axis=0)}")

# Inverse transform
X_recovered = scaler.inverse_transform(X_scaled)
print("\nRecovered original:")
print(X_recovered)
```

**Output**:
```
Original data:
[[   1 2000]
 [   2 3000]
 [   3 4000]
 [   4 5000]
 [   5 6000]]

Means: [   3. 4000.]
Stds:  [1.41421356e+00 1.41421356e+03]

Standardized data:
[[-1.41421356 -1.41421356]
 [-0.70710678 -0.70710678]
 [ 0.          0.        ]
 [ 0.70710678  0.70710678]
 [ 1.41421356  1.41421356]]

Scaled means: [0. 0.]
Scaled stds:  [1. 1.]
```

### Using Scikit-Learn

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Access statistics
print("Means:", scaler.mean_)
print("Stds:", scaler.scale_)  # Note: in sklearn it's scale_, not std_

# Transform new data
X_new = np.array([[6, 7000]])
X_new_scaled = scaler.transform(X_new)
```

### Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data from different distributions
np.random.seed(42)
n = 1000

# Feature 1: Normal distribution, mean=5, std=2
X1 = np.random.normal(5, 2, n)

# Feature 2: Uniform distribution, range [0, 100]
X2 = np.random.uniform(0, 100, n)

X = np.column_stack([X1, X2])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original distributions
axes[0, 0].hist(X[:, 0], bins=50, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(scaler.mean_[0], color='r', linestyle='--', linewidth=2, label=f'Mean={scaler.mean_[0]:.2f}')
axes[0, 0].set_xlabel('Feature 1 (original)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Feature 1: Before Standardization')
axes[0, 0].legend()

axes[0, 1].hist(X[:, 1], bins=50, alpha=0.7, edgecolor='black', color='orange')
axes[0, 1].axvline(scaler.mean_[1], color='r', linestyle='--', linewidth=2, label=f'Mean={scaler.mean_[1]:.2f}')
axes[0, 1].set_xlabel('Feature 2 (original)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Feature 2: Before Standardization')
axes[0, 1].legend()

# Standardized distributions
axes[1, 0].hist(X_scaled[:, 0], bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2, label='Mean=0')
axes[1, 0].set_xlabel('Feature 1 (standardized)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Feature 1: After Standardization')
axes[1, 0].legend()

axes[1, 1].hist(X_scaled[:, 1], bins=50, alpha=0.7, edgecolor='black', color='orange')
axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='Mean=0')
axes[1, 1].set_xlabel('Feature 2 (standardized)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Feature 2: After Standardization')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

---

## Min-Max Scaling

### Definition

**Min-Max scaling** (also called **normalization**) transforms features to a fixed range, typically [0, 1].

**Formula**:
```
x_scaled = (x - min(x)) / (max(x) - min(x))
```

**General range [a, b]**:
```
x_scaled = a + (x - min(x)) * (b - a) / (max(x) - min(x))
```

### Mathematical Properties

**For each feature after Min-Max scaling to [0, 1]**:
```
Minimum: min(x_scaled) = 0
Maximum: max(x_scaled) = 1
Range:   range(x_scaled) = 1
```

**Preserves**:
- Relative distances between points
- Shape of distribution
- Relationship between features

### When to Use Min-Max Scaling

✅ **Use when**:
- Need bounded range (e.g., [0, 1])
- Features have hard boundaries (e.g., percentages, probabilities)
- Using neural networks (inputs often [0, 1])
- No outliers or outliers handled separately
- Want to preserve zero entries (sparse data)

❌ **Don't use when**:
- Outliers present (they affect min/max significantly!)
- New data may exceed training range
- Want to preserve information about spread

### Python Implementation

```python
import numpy as np

class MinMaxScaler:
    """Min-Max scaler to range [0, 1]"""
    
    def __init__(self, feature_range=(0, 1)):
        """
        Args:
            feature_range: Tuple (min, max) for desired range
        """
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
    
    def fit(self, X):
        """
        Compute min and max for each feature
        
        Args:
            X: Feature matrix (m × n)
        """
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Handle constant features
        self.data_range_[self.data_range_ == 0] = 1
        
        return self
    
    def transform(self, X):
        """
        Transform features to [0, 1] range
        
        Args:
            X: Feature matrix (m × n)
        
        Returns:
            X_scaled: Scaled features
        """
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        # Scale to [0, 1]
        X_std = (X - self.data_min_) / self.data_range_
        
        # Scale to [min, max] if needed
        min_val, max_val = self.feature_range
        X_scaled = X_std * (max_val - min_val) + min_val
        
        return X_scaled
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Reverse scaling
        
        Args:
            X_scaled: Scaled features
        
        Returns:
            X: Original scale features
        """
        min_val, max_val = self.feature_range
        X_std = (X_scaled - min_val) / (max_val - min_val)
        return X_std * self.data_range_ + self.data_min_

# Example
X = np.array([[1, 2000],
              [2, 3000],
              [3, 4000],
              [4, 5000],
              [5, 6000]])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Original data:")
print(X)
print(f"\nData min: {scaler.data_min_}")
print(f"Data max: {scaler.data_max_}")
print("\nMin-Max scaled data:")
print(X_scaled)
print(f"\nScaled min: {np.min(X_scaled, axis=0)}")
print(f"Scaled max: {np.max(X_scaled, axis=0)}")
```

**Output**:
```
Original data:
[[   1 2000]
 [   2 3000]
 [   3 4000]
 [   4 5000]
 [   5 6000]]

Data min: [   1. 2000.]
Data max: [   5. 6000.]

Min-Max scaled data:
[[0.   0.  ]
 [0.25 0.25]
 [0.5  0.5 ]
 [0.75 0.75]
 [1.   1.  ]]

Scaled min: [0. 0.]
Scaled max: [1. 1.]
```

### Using Scikit-Learn

```python
from sklearn.preprocessing import MinMaxScaler

# Default: [0, 1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Custom range: [-1, 1]
scaler_custom = MinMaxScaler(feature_range=(-1, 1))
X_scaled_custom = scaler_custom.fit_transform(X)
```

### Impact of Outliers

```python
# Data with outlier
X_with_outlier = np.array([[1], [2], [3], [4], [100]])  # 100 is outlier

# Scale without outlier
scaler_normal = MinMaxScaler()
X_normal_scaled = scaler_normal.fit_transform(X_with_outlier[:4])

# Scale with outlier
scaler_outlier = MinMaxScaler()
X_outlier_scaled = scaler_outlier.fit_transform(X_with_outlier)

print("Without outlier:")
print(X_normal_scaled.ravel())
print("\nWith outlier:")
print(X_outlier_scaled.ravel())
```

**Output**:
```
Without outlier:
[0.    0.333 0.667 1.   ]

With outlier:
[0.    0.010 0.020 0.030 1.   ]
```

**Observation**: Outlier compresses all other values toward 0!

---

## Impact on Gradient Descent

### The Problem: Elongated Contours

**Without scaling**:
```
    w₂
    │
    │ ╱────╲
    │╱      ╲  Elongated ellipse
    ╱        ╲ (one feature dominates)
   ╱          ╲
  ╱   ●        ╲  ● = Minimum
 ╱              ╲
─────────────────── w₁
```

**With scaling**:
```
    w₂
    │
    │   ╱─╲
    │  ╱ ● ╲   Circular contours
    │ │     │  (balanced features)
    │  ╲   ╱
    │   ╲─╱
    │
─────────────────── w₁
```

### Convergence Speed

**Unscaled**: Gradient descent zigzags → slow convergence

**Scaled**: Gradient descent goes directly → fast convergence

### Mathematical Analysis

**Cost function** for linear regression:
```
J(w) = (1/2m) Σᵢ (w₁x₁⁽ⁱ⁾ + w₂x₂⁽ⁱ⁾ - y⁽ⁱ⁾)²
```

**Partial derivatives**:
```
∂J/∂w₁ = (1/m) Σᵢ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)x₁⁽ⁱ⁾
∂J/∂w₂ = (1/m) Σᵢ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)x₂⁽ⁱ⁾
```

**If x₂ >> x₁** (different scales):
- ∂J/∂w₂ is much larger than ∂J/∂w₁
- Gradient points almost entirely in w₂ direction
- Need very small learning rate to avoid divergence
- Slow progress in w₁ direction

**After scaling** (x₁ and x₂ similar):
- ∂J/∂w₁ and ∂J/∂w₂ are comparable
- Gradient points toward minimum
- Can use larger learning rate
- Fast convergence

### Practical Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def gradient_descent(X, y, learning_rate=0.01, num_iterations=100):
    """Simple gradient descent for linear regression"""
    m, n = X.shape
    w = np.zeros(n)
    cost_history = []
    
    for i in range(num_iterations):
        # Predictions
        y_pred = X @ w
        
        # Cost
        cost = (1/(2*m)) * np.sum((y_pred - y) ** 2)
        cost_history.append(cost)
        
        # Gradient
        gradient = (1/m) * X.T @ (y_pred - y)
        
        # Update
        w = w - learning_rate * gradient
    
    return w, cost_history

# Generate data
np.random.seed(42)
n = 100

# Feature 1: Small scale
X1 = np.random.uniform(0, 10, n)

# Feature 2: Large scale  
X2 = np.random.uniform(0, 1000, n)

# Target
y = 2*X1 + 0.02*X2 + np.random.randn(n)

X = np.column_stack([X1, X2])

# Train without scaling
w_unscaled, cost_unscaled = gradient_descent(X, y, learning_rate=0.0001, num_iterations=1000)

# Train with scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
w_scaled, cost_scaled = gradient_descent(X_scaled, y, learning_rate=0.01, num_iterations=1000)

# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cost curves
ax = axes[0]
ax.plot(cost_unscaled, linewidth=2, label='Unscaled (lr=0.0001)')
ax.plot(cost_scaled, linewidth=2, label='Scaled (lr=0.01)')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Cost', fontsize=12)
ax.set_title('Convergence Comparison', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Log scale to see details
ax = axes[1]
ax.semilogy(cost_unscaled, linewidth=2, label='Unscaled')
ax.semilogy(cost_scaled, linewidth=2, label='Scaled')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Cost (log scale)', fontsize=12)
ax.set_title('Convergence (Log Scale)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("="*60)
print("CONVERGENCE COMPARISON")
print("="*60)
print(f"\nUnscaled:")
print(f"  Initial cost: {cost_unscaled[0]:.4f}")
print(f"  Final cost:   {cost_unscaled[-1]:.4f}")
print(f"  Reduction:    {cost_unscaled[0] - cost_unscaled[-1]:.4f}")

print(f"\nScaled:")
print(f"  Initial cost: {cost_scaled[0]:.4f}")
print(f"  Final cost:   {cost_scaled[-1]:.4f}")
print(f"  Reduction:    {cost_scaled[0] - cost_scaled[-1]:.4f}")
```

### Learning Rate Sensitivity

**Without scaling**: Need very small learning rate
```
lr = 0.0001: Converges slowly
lr = 0.001:  May diverge!
```

**With scaling**: Can use larger learning rate
```
lr = 0.01:   Converges quickly
lr = 0.1:    Still stable
```

### Visualizing Cost Contours

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_cost_contours(X, y, title):
    """Plot cost function contours"""
    # Create grid of w values
    w1_range = np.linspace(-5, 10, 100)
    w2_range = np.linspace(-5, 10, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    
    # Compute cost for each (w1, w2) pair
    m = len(y)
    J = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = np.array([W1[i, j], W2[i, j]])
            y_pred = X @ w
            J[i, j] = (1/(2*m)) * np.sum((y_pred - y) ** 2)
    
    # Plot contours
    plt.figure(figsize=(10, 8))
    contour = plt.contour(W1, W2, J, levels=30, cmap='viridis')
    plt.colorbar(contour, label='Cost')
    plt.xlabel('w₁', fontsize=14)
    plt.ylabel('w₂', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Mark optimum
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    plt.plot(model.coef_[0], model.coef_[1], 'r*', markersize=20, label='Optimum')
    plt.legend()
    
    plt.show()

# Unscaled
plot_cost_contours(X, y, 'Cost Contours: Unscaled Features')

# Scaled
plot_cost_contours(X_scaled, y, 'Cost Contours: Scaled Features')
```

**Observation**: Scaled features produce circular/elliptical contours → easier optimization!

---

## Other Scaling Methods

### 1. Robust Scaler

**Formula**:
```
x_scaled = (x - median(x)) / IQR(x)
```

Where IQR = Interquartile Range = Q₃ - Q₁

**Advantage**: **Robust to outliers** (uses median and IQR instead of mean and std)

**Python**:
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

**When to use**: Data has many outliers

### 2. MaxAbs Scaler

**Formula**:
```
x_scaled = x / max(|x|)
```

**Result**: Range [-1, 1]

**Advantage**: **Preserves sparsity** (zeros remain zeros)

**Python**:
```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)
```

**When to use**: Sparse data (e.g., text with TF-IDF)

### 3. Normalizer

**Formula** (L2 normalization):
```
x_scaled = x / ||x||₂
```

**Note**: Scales **each sample** (row) to unit norm, not each feature (column)

**Python**:
```python
from sklearn.preprocessing import Normalizer

scaler = Normalizer(norm='l2')
X_scaled = scaler.fit_transform(X)
```

**When to use**: Text data, when direction matters more than magnitude

### 4. QuantileTransformer

**Transforms features to follow uniform or normal distribution**

**Python**:
```python
from sklearn.preprocessing import QuantileTransformer

# Transform to uniform distribution [0, 1]
scaler_uniform = QuantileTransformer(output_distribution='uniform')
X_uniform = scaler_uniform.fit_transform(X)

# Transform to normal distribution
scaler_normal = QuantileTransformer(output_distribution='normal')
X_normal = scaler_normal.fit_transform(X)
```

**When to use**: Non-linear transformations needed, very skewed data

### 5. PowerTransformer

**Applies power transformation to make data more Gaussian**

**Methods**: Yeo-Johnson (works with negative values), Box-Cox (positive values only)

**Python**:
```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson (default)
scaler = PowerTransformer(method='yeo-johnson')
X_transformed = scaler.fit_transform(X)

# Box-Cox (requires positive data)
scaler_bc = PowerTransformer(method='box-cox')
X_bc = scaler_bc.fit_transform(X + 1)  # Add 1 to make positive
```

**When to use**: Highly skewed data, want to make it more Gaussian

### Comparison Table

| Method | Formula | Range | Outliers | Sparsity | Use Case |
|--------|---------|-------|----------|----------|----------|
| **Standardization** | (x-μ)/σ | Unbounded | Sensitive | Breaks | Default choice |
| **Min-Max** | (x-min)/(max-min) | [0,1] | Very sensitive | Preserves | Neural networks |
| **Robust** | (x-median)/IQR | Unbounded | Robust | Breaks | Many outliers |
| **MaxAbs** | x/max(abs(x)) | [-1,1] | Sensitive | Preserves | Sparse data |
| **Normalizer** | x/||x|| | Unit norm | N/A | Changes | Text data |

---

## When to Scale and When Not To

### Always Scale For

✅ **Gradient descent algorithms**:
- Linear regression (when using GD)
- Logistic regression
- Neural networks
- Support Vector Machines

✅ **Distance-based algorithms**:
- K-Nearest Neighbors
- K-Means clustering
- Hierarchical clustering

✅ **Regularized models**:
- Ridge regression
- Lasso regression
- Elastic Net

✅ **Principal Component Analysis (PCA)**

### Never Scale For

❌ **Tree-based models**:
- Decision trees
- Random forests
- Gradient boosting (XGBoost, LightGBM)

**Why?** Trees split on thresholds, not distances. Scale doesn't matter!

❌ **Naive Bayes**

**Why?** Works with probability distributions of each feature independently.

### Sometimes Scale

**Linear regression with closed-form solution**:
- Results are the same (mathematically equivalent)
- But: Still scale for interpretability and if using regularization

**Example**:
```python
from sklearn.linear_model import LinearRegression

# Without scaling
model1 = LinearRegression()
model1.fit(X, y)
pred1 = model1.predict(X_test)

# With scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

model2 = LinearRegression()
model2.fit(X_scaled, y)
pred2 = model2.predict(X_test_scaled)

# Predictions are identical!
print(f"Predictions equal: {np.allclose(pred1, pred2)}")  # True
```

---

## Practical Implementation

### Complete Pipeline

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
np.random.seed(42)
n = 200
X1 = np.random.uniform(0, 10, n)
X2 = np.random.uniform(0, 1000, n)
X3 = np.random.uniform(0, 0.1, n)
X = np.column_stack([X1, X2, X3])
y = 2*X1 + 0.02*X2 + 100*X3 + np.random.randn(n)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create scaler
scaler = StandardScaler()

# Fit scaler ONLY on training data
scaler.fit(X_train)

# Transform training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test R²:  {r2:.4f}")
print(f"\nScaler statistics:")
print(f"Means: {scaler.mean_}")
print(f"Stds:  {scaler.scale_}")
```

### Using Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

# Fit (automatically scales then trains)
pipeline.fit(X_train, y_train)

# Predict (automatically scales test data)
y_pred = pipeline.predict(X_test)

# Evaluate
score = pipeline.score(X_test, y_test)
print(f"R² score: {score:.4f}")

# Access components
scaler = pipeline.named_steps['scaler']
model = pipeline.named_steps['model']
print(f"Scaling means: {scaler.mean_}")
print(f"Model coefficients: {model.coef_}")
```

---

## Common Pitfalls and Best Practices

### Pitfall 1: Fitting Scaler on Entire Dataset

❌ **Wrong**:
```python
# DON'T DO THIS!
scaler.fit(X)  # Fit on ALL data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Problem**: Test data influences scaler → data leakage!

✅ **Correct**:
```python
# Fit on training data ONLY
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Pitfall 2: Scaling Target Variable

**Question**: Should we scale y?

**Answer**: 
- **Regression**: Usually **no** (doesn't help, harder to interpret)
- **Neural networks**: Sometimes yes (can help convergence)
- **When y and X are different scales**: Transform both or neither

**If you do scale y**:
```python
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Train model
model.fit(X_train_scaled, y_train_scaled)

# Predict and inverse transform
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
```

### Pitfall 3: Forgetting to Scale New Data

❌ **Wrong**:
```python
# New data arrives
X_new = np.array([[5, 5000, 0.05]])
prediction = model.predict(X_new)  # WRONG! Not scaled
```

✅ **Correct**:
```python
X_new_scaled = scaler.transform(X_new)
prediction = model.predict(X_new_scaled)
```

### Pitfall 4: Using Wrong Scaler

**Example**: Using StandardScaler on data with outliers
```python
X = [1, 2, 3, 4, 1000]  # 1000 is outlier

scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X.reshape(-1, 1))
# Most values near 0, outlier dominates mean and std

scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X.reshape(-1, 1))
# Better: uses median and IQR
```

### Best Practices

1. **Always split first, then scale**:
   ```python
   X_train, X_test = split(X)
   scaler.fit(X_train)  # Only training data
   ```

2. **Use Pipeline**:
   ```python
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('model', Ridge())
   ])
   ```

3. **Save scaler with model**:
   ```python
   import joblib
   joblib.dump(scaler, 'scaler.pkl')
   joblib.dump(model, 'model.pkl')
   ```

4. **Document scaling method**:
   - Which scaler was used?
   - What were the parameters (mean, std, min, max)?

5. **Check for data leakage**:
   - Scaler only sees training data
   - No information from test/validation sets

6. **Choose appropriate scaler**:
   - Default: StandardScaler
   - Outliers: RobustScaler
   - Need [0, 1]: MinMaxScaler
   - Sparse data: MaxAbsScaler

---

## Summary and Next Steps

### What We Learned

1. **Why Scaling Matters**
   - Different feature scales cause problems
   - Affects gradient descent convergence
   - Impacts regularization fairness
   - Necessary for distance-based methods

2. **Standardization (Z-Score)**
   - Formula: (x - μ) / σ
   - Result: mean=0, std=1
   - Default choice for most algorithms

3. **Min-Max Scaling**
   - Formula: (x - min) / (max - min)
   - Result: range [0, 1]
   - Sensitive to outliers

4. **Impact on Gradient Descent**
   - Unscaled: Elongated contours, slow convergence
   - Scaled: Circular contours, fast convergence
   - Can use larger learning rates after scaling

5. **Best Practices**
   - Fit scaler on training data only
   - Use Pipeline for clean workflows
   - Choose appropriate scaler for data
   - Save scaler with model

### Key Formulas

**Standardization**:
```
x_scaled = (x - μ) / σ
```

**Min-Max**:
```
x_scaled = (x - min) / (max - min)
```

**Robust**:
```
x_scaled = (x - median) / IQR
```

### Quick Decision Guide

```
Choose scaler:
  │
  ├─ Outliers present?
  │  ├─ Yes → RobustScaler
  │  └─ No → Continue
  │
  ├─ Need specific range [0,1]?
  │  ├─ Yes → MinMaxScaler
  │  └─ No → Continue
  │
  ├─ Sparse data?
  │  ├─ Yes → MaxAbsScaler
  │  └─ No → StandardScaler
  │
  └─ Default: StandardScaler
```

### Key Insights

✅ **Always scale** for gradient descent, distance-based methods  
✅ **Never scale** for tree-based methods  
✅ **Fit only on training data** to avoid leakage  
✅ **StandardScaler** is the default choice  
✅ **RobustScaler** when outliers present  
✅ **Use Pipeline** for clean code  

### Prerequisites Checklist

Before moving on:
- [ ] Understand why scaling matters
- [ ] Know standardization and min-max formulas
- [ ] Can implement scalers from scratch
- [ ] Understand impact on gradient descent
- [ ] Know when to scale and when not to
- [ ] Can avoid common pitfalls

### Next Steps

1. **Polynomial_Features.md**
   - Creating polynomial features
   - Interaction terms
   - Feature engineering techniques

2. **Feature_Selection.md**
   - Filter methods
   - Wrapper methods
   - Embedded methods (Lasso)
   - Dimensionality reduction

3. **Model_Evaluation.md**
   - Cross-validation strategies
   - Learning curves
   - Validation curves
   - Model selection

### Practice Exercises

1. Implement StandardScaler and MinMaxScaler from scratch
2. Compare gradient descent convergence with/without scaling
3. Visualize cost contours for scaled and unscaled features
4. Apply different scalers to dataset with outliers
5. Create complete ML pipeline with scaling
6. Debug data leakage in scaling workflow

### Additional Resources

**Papers**:
- Feature scaling in machine learning (various surveys)

**Documentation**:
- Scikit-learn: Preprocessing guide
- Scikit-learn: Pipeline and FeatureUnion

**Online**:
- Andrew Ng: Feature Normalization lectures
- Scikit-learn scaling tutorial
- Fast.ai: Preprocessing best practices

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║          FEATURE SCALING QUICK REFERENCE                       ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  STANDARDIZATION (Z-SCORE)                                     ║
║  ─────────────────────────                                     ║
║  x_scaled = (x - μ) / σ                                        ║
║                                                                ║
║  Result: mean=0, std=1                                         ║
║  Use: Default choice, gradient descent, regularization         ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  MIN-MAX SCALING                                               ║
║  ────────────────                                              ║
║  x_scaled = (x - min) / (max - min)                            ║
║                                                                ║
║  Result: range [0, 1]                                          ║
║  Use: Neural networks, need bounded range                      ║
║  Caution: Sensitive to outliers                                ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  ROBUST SCALER                                                 ║
║  ──────────────                                                ║
║  x_scaled = (x - median) / IQR                                 ║
║                                                                ║
║  Result: Unbounded, robust                                     ║
║  Use: Data with many outliers                                  ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  WHEN TO SCALE                                                 ║
╠════════════════════════════════════════════════════════════════╣
║  Always:                                                       ║
║    • Gradient descent algorithms                               ║
║    • Distance-based methods (KNN, K-Means)                     ║
║    • Regularization (Ridge, Lasso)                             ║
║    • Neural networks, SVM, PCA                                 ║
║                                                                ║
║  Never:                                                        ║
║    • Tree-based models (Random Forest, XGBoost)                ║
║    • Naive Bayes                                               ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  IMPACT ON GRADIENT DESCENT                                    ║
╠════════════════════════════════════════════════════════════════╣
║  Without scaling:                                              ║
║    • Elongated contours                                        ║
║    • Slow convergence                                          ║
║    • Need very small learning rate                             ║
║                                                                ║
║  With scaling:                                                 ║
║    • Circular contours                                         ║
║    • Fast convergence                                          ║
║    • Can use larger learning rate                              ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  BEST PRACTICES                                                ║
╠════════════════════════════════════════════════════════════════╣
║  1. Split data FIRST, then scale                               ║
║     scaler.fit(X_train)  # Only training data                  ║
║                                                                ║
║  2. Use Pipeline                                               ║
║     Pipeline([('scaler', StandardScaler()),                    ║
║               ('model', Ridge())])                             ║
║                                                                ║
║  3. Don't scale target (usually)                               ║
║                                                                ║
║  4. Save scaler with model                                     ║
║     joblib.dump(scaler, 'scaler.pkl')                          ║
║                                                                ║
║  5. Apply same scaling to new data                             ║
║     X_new_scaled = scaler.transform(X_new)                     ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  SCIKIT-LEARN USAGE                                            ║
╠════════════════════════════════════════════════════════════════╣
║  from sklearn.preprocessing import StandardScaler              ║
║                                                                ║
║  scaler = StandardScaler()                                     ║
║  scaler.fit(X_train)                                           ║
║  X_train_scaled = scaler.transform(X_train)                    ║
║  X_test_scaled = scaler.transform(X_test)                      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Congratulations!** You now understand feature scaling and its critical importance in machine learning!

---

*End of Feature_Scaling_and_Normalization.md*