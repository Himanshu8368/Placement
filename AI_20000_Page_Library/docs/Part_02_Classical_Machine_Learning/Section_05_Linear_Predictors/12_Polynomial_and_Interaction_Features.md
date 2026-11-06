# Polynomial and Interaction Features

## Table of Contents
1. [Introduction to Feature Engineering](#introduction-to-feature-engineering)
2. [Polynomial Regression](#polynomial-regression)
3. [Feature Interactions](#feature-interactions)
4. [Creating Non-Linear Decision Boundaries](#creating-non-linear-decision-boundaries)
5. [Degree Selection](#degree-selection)
6. [Regularization with Polynomial Features](#regularization-with-polynomial-features)
7. [Practical Implementation](#practical-implementation)
8. [Advanced Topics](#advanced-topics)
9. [Common Pitfalls and Best Practices](#common-pitfalls-and-best-practices)
10. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to Feature Engineering

### What is Feature Engineering?

**Feature engineering** is the process of creating new features from existing ones to improve model performance.

**Basic idea**: Transform the input space to make patterns more apparent.

### Why Engineer Features?

**Problem**: Linear models can only learn linear relationships.

```
Linear model:  ŷ = w₀ + w₁x₁ + w₂x₂
```

**Limitation**: Can't capture:
- Non-linear relationships (curves, parabolas)
- Interactions between features
- Complex patterns

**Solution**: Create new features that make patterns linear in the transformed space!

### Types of Feature Engineering

**1. Polynomial features**: x, x², x³, ...
**2. Interaction features**: x₁·x₂, x₁·x₂·x₃
**3. Domain-specific features**: log(x), √x, sin(x)
**4. Aggregations**: mean, max, count
**5. Categorical encoding**: one-hot, ordinal

This chapter focuses on **polynomial and interaction features**.

### Visual Intuition

**Original space** (linear model fails):
```
    y
    │     ●
    │   ●   ●
    │ ●       ●
    │●         ●
    └─────────── x
    
    Non-linear relationship
    Linear fit is poor
```

**Transformed space** (linear model works):
```
    y
    │     ●
    │   ●   ●
    │ ●   ●   ●
    │●  ●      ●
    └─────────── x²
    
    With x² feature, relationship is linear!
```

---

## Polynomial Regression

### Definition

**Polynomial regression** fits a polynomial function to data by treating polynomial terms as new features.

**Model**:
```
ŷ = w₀ + w₁x + w₂x² + w₃x³ + ... + wₐxᵈ
```

Where d is the **polynomial degree**.

**Key insight**: This is still **linear** in the weights w, just with polynomial features!

### Degrees Explained

**Degree 1 (Linear)**:
```
ŷ = w₀ + w₁x
```
Straight line

**Degree 2 (Quadratic)**:
```
ŷ = w₀ + w₁x + w₂x²
```
Parabola (one curve)

**Degree 3 (Cubic)**:
```
ŷ = w₀ + w₁x + w₂x² + w₃x³
```
S-curve (two curves)

**Degree 4 (Quartic)**:
```
ŷ = w₀ + w₁x + w₂x² + w₃x³ + w₄x⁴
```
W-curve (three curves)

**General rule**: Degree d polynomial can have up to d-1 turns.

### Mathematical Formulation

**Original features**:
```
x ∈ ℝⁿ  (n features)
```

**Polynomial transformation** (degree d):
```
φ(x) = [1, x₁, x₂, ..., xₙ, x₁², x₁x₂, ..., xₙᵈ]
```

**Model in transformed space**:
```
ŷ = wᵀφ(x)
```

This is **linear** in w, so we can use linear regression!

### Example: One Feature

**Data**: (x, y) pairs

**Degree 1**:
```
Features: [1, x]
Model: ŷ = w₀ + w₁x
```

**Degree 2**:
```
Features: [1, x, x²]
Model: ŷ = w₀ + w₁x + w₂x²
```

**Degree 3**:
```
Features: [1, x, x², x³]
Model: ŷ = w₀ + w₁x + w₂x² + w₃x³
```

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate non-linear data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 + 3*X.ravel() - 0.5*X.ravel()**2 + 0.02*X.ravel()**3 + np.random.randn(100)*2

# Fit polynomials of different degrees
degrees = [1, 2, 3, 5, 10]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, degree in enumerate(degrees):
    ax = axes[idx]
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_pred = model.predict(X_plot_poly)
    
    # Compute R²
    y_pred_train = model.predict(X_poly)
    ss_res = np.sum((y - y_pred_train) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Plot
    ax.scatter(X, y, alpha=0.5, s=20, label='Data')
    ax.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'Degree {degree}')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Polynomial Degree {degree}\nR² = {r2:.4f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-10, 30])

# Last plot: All together
ax = axes[-1]
ax.scatter(X, y, alpha=0.5, s=20, label='Data', color='black')

colors = ['blue', 'green', 'red', 'purple', 'orange']
for degree, color in zip(degrees, colors):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_pred = model.predict(X_plot_poly)
    
    ax.plot(X_plot, y_pred, linewidth=2, label=f'Degree {degree}', color=color)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('All Degrees Compared', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([-10, 30])

plt.tight_layout()
plt.show()
```

### Number of Features

For **one original feature** and polynomial degree d:
```
Number of features = d + 1
```
(including bias)

For **n original features** and degree d:
```
Number of features = C(n+d, d) = (n+d)! / (n! × d!)
```

**Example** (2 features, degree 3):
```
Original: [x₁, x₂]

Polynomial features:
[1,                        ← bias
 x₁, x₂,                   ← degree 1
 x₁², x₁x₂, x₂²,           ← degree 2
 x₁³, x₁²x₂, x₁x₂², x₂³]   ← degree 3

Total: 10 features
Formula: C(2+3, 3) = C(5, 3) = 10 ✓
```

**Feature explosion**: Number grows rapidly!

```
Features  Degree  Total Features
────────  ──────  ──────────────
    2        2           6
    2        3          10
    2        5          21
    5        2          21
    5        3          56
   10        2          66
   10        3         286
```

---

## Feature Interactions

### What are Interaction Features?

**Interaction features** capture the **combined effect** of two or more features.

**Example**: House price prediction
```
Price depends on:
  Size: bigger house → higher price
  Location: better location → higher price
  
But also:
  Size × Location: big house in good location → MUCH higher price!
  
The effect is multiplicative, not just additive.
```

### Mathematical Definition

**For two features** x₁ and x₂:
```
Interaction term: x₁ · x₂
```

**Model without interaction**:
```
ŷ = w₀ + w₁x₁ + w₂x₂
```

**Model with interaction**:
```
ŷ = w₀ + w₁x₁ + w₂x₂ + w₃(x₁·x₂)
```

### Why Interactions Matter

**Example**: Temperature and Humidity

**Without interaction**:
```
Comfort = -2·Temperature + -3·Humidity

Temperature ↑ → Comfort ↓
Humidity ↑ → Comfort ↓
```

**With interaction**:
```
Comfort = -2·Temperature - 3·Humidity - 0.5·(Temperature × Humidity)

High temp + high humidity → VERY uncomfortable!
The interaction term captures this synergy.
```

### Types of Interactions

**Two-way interactions**: x₁·x₂
```
Price = Size + Location + Size×Location
```

**Three-way interactions**: x₁·x₂·x₃
```
Price = Size + Location + Age + Size×Location×Age
```

**Higher-order**: Generally not useful (hard to interpret, overfit)

### Polynomial Features Include Interactions

**Degree 2 polynomial** with 2 features:
```
φ(x) = [1, x₁, x₂, x₁², x₁x₂, x₂²]
                      ↑
                 Interaction term!
```

**Degree 3 polynomial** with 2 features:
```
φ(x) = [1, 
        x₁, x₂,                    ← degree 1
        x₁², x₁x₂, x₂²,            ← degree 2 (includes interaction)
        x₁³, x₁²x₂, x₁x₂², x₂³]    ← degree 3 (includes interactions)
```

### Creating Only Interactions

Sometimes you want **only interaction terms**, not all polynomials:

```python
from sklearn.preprocessing import PolynomialFeatures

# Only interactions (no powers)
poly_interactions = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

X = np.array([[1, 2],
              [3, 4]])

X_interactions = poly_interactions.fit_transform(X)
print(X_interactions)
# [[1, 2, 2]     ← [x₁, x₂, x₁×x₂]
#  [3, 4, 12]]   ← [x₁, x₂, x₁×x₂]
```

### Example: Real-World Application

**Predicting ad click-through rate**:

**Features**:
- x₁ = User age
- x₂ = Time of day (hour)
- x₃ = Is mobile device (0/1)

**Without interactions**:
```
CTR = w₀ + w₁·age + w₂·hour + w₃·is_mobile
```

**With interactions**:
```
CTR = w₀ + w₁·age + w₂·hour + w₃·is_mobile
      + w₄·(age × hour)           ← Young people click more at night
      + w₅·(age × is_mobile)      ← Young people use mobile more
      + w₆·(hour × is_mobile)     ← Mobile usage peaks at certain hours
```

### Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate data with interaction
np.random.seed(42)
n = 100
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)

# Model WITH interaction
y_with = 5 + 2*x1 + 3*x2 + 0.5*x1*x2 + np.random.randn(n)*2

# Model WITHOUT interaction (for comparison)
y_without = 5 + 2*x1 + 3*x2 + np.random.randn(n)*2

# Fit models
from sklearn.linear_model import LinearRegression

# Without interaction
X_without = np.column_stack([x1, x2])
model_without = LinearRegression()
model_without.fit(X_without, y_with)

# With interaction
X_with = np.column_stack([x1, x2, x1*x2])
model_with = LinearRegression()
model_with.fit(X_with, y_with)

# Plot
fig = plt.figure(figsize=(16, 6))

# Surface for prediction
x1_grid = np.linspace(0, 10, 50)
x2_grid = np.linspace(0, 10, 50)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)

# Plot 1: Without interaction
ax1 = fig.add_subplot(121, projection='3d')
X_grid_without = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
Z_without = model_without.predict(X_grid_without).reshape(X1_grid.shape)

ax1.scatter(x1, x2, y_with, alpha=0.5, s=20)
ax1.plot_surface(X1_grid, X2_grid, Z_without, alpha=0.3, cmap='viridis')
ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.set_zlabel('y')
ax1.set_title('Without Interaction Term\n(Planar surface)', fontsize=14)

# Plot 2: With interaction
ax2 = fig.add_subplot(122, projection='3d')
X_grid_with = np.column_stack([X1_grid.ravel(), X2_grid.ravel(), 
                                (X1_grid * X2_grid).ravel()])
Z_with = model_with.predict(X_grid_with).reshape(X1_grid.shape)

ax2.scatter(x1, x2, y_with, alpha=0.5, s=20)
ax2.plot_surface(X1_grid, X2_grid, Z_with, alpha=0.3, cmap='viridis')
ax2.set_xlabel('x₁')
ax2.set_ylabel('x₂')
ax2.set_zlabel('y')
ax2.set_title('With Interaction Term\n(Curved surface)', fontsize=14)

plt.tight_layout()
plt.show()

print("Model without interaction:")
print(f"Coefficients: {model_without.coef_}")
print(f"R²: {model_without.score(X_without, y_with):.4f}")

print("\nModel with interaction:")
print(f"Coefficients: {model_with.coef_}")
print(f"R²: {model_with.score(X_with, y_with):.4f}")
```

---

## Creating Non-Linear Decision Boundaries

### Linear Decision Boundaries

**Binary classification** with linear model:
```
Decision boundary: w₀ + w₁x₁ + w₂x₂ = 0
```

This is a **straight line** (or hyperplane).

**Limitation**: Can't separate data that's not linearly separable!

```
    x₂
    │
    │  ●  ○  ●
    │ ○  ●  ○
    │  ●  ○  ●
    │
    └──────── x₁
    
    XOR problem - no straight line can separate!
```

### Solution: Polynomial Features

Add polynomial features → Decision boundary becomes **non-linear** in original space!

**Example**: Degree 2
```
Model: w₀ + w₁x₁ + w₂x₂ + w₃x₁² + w₄x₁x₂ + w₅x₂² = 0
```

In **original space** (x₁, x₂): This is a **conic section** (circle, ellipse, parabola, hyperbola)

In **transformed space** (x₁, x₂, x₁², x₁x₂, x₂²): This is a **hyperplane** (linear!)

### XOR Problem Solved

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# XOR data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR

# Try linear classifier (will fail)
model_linear = LogisticRegression()
model_linear.fit(X, y)
accuracy_linear = model_linear.score(X, y)

# Add polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train with polynomial features
model_poly = LogisticRegression()
model_poly.fit(X_poly, y)
accuracy_poly = model_poly.score(X_poly, y)

print("XOR Problem:")
print(f"Linear accuracy: {accuracy_linear * 100:.0f}%")
print(f"Polynomial (degree 2) accuracy: {accuracy_poly * 100:.0f}%")

# Visualize decision boundaries
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Create mesh
xx1, xx2 = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                        np.linspace(-0.5, 1.5, 200))

# Plot 1: Linear
ax = axes[0]
Z_linear = model_linear.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z_linear = Z_linear.reshape(xx1.shape)
ax.contourf(xx1, xx2, Z_linear, alpha=0.3, cmap='RdYlBu')
ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', s=200, 
          edgecolors='k', linewidth=2, label='Class 0')
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='s', s=200, 
          edgecolors='k', linewidth=2, label='Class 1')
ax.set_xlabel('x₁', fontsize=14)
ax.set_ylabel('x₂', fontsize=14)
ax.set_title(f'Linear Classifier\nAccuracy: {accuracy_linear*100:.0f}%', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Plot 2: Polynomial
ax = axes[1]
X_mesh_poly = poly.transform(np.c_[xx1.ravel(), xx2.ravel()])
Z_poly = model_poly.predict(X_mesh_poly)
Z_poly = Z_poly.reshape(xx1.shape)
ax.contourf(xx1, xx2, Z_poly, alpha=0.3, cmap='RdYlBu')
ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', s=200, 
          edgecolors='k', linewidth=2, label='Class 0')
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='s', s=200, 
          edgecolors='k', linewidth=2, label='Class 1')
ax.set_xlabel('x₁', fontsize=14)
ax.set_ylabel('x₂', fontsize=14)
ax.set_title(f'Polynomial Classifier (degree 2)\nAccuracy: {accuracy_poly*100:.0f}%', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Circular Decision Boundary

```python
# Generate circular data
np.random.seed(42)
n_inner = 100
n_outer = 100

# Inner circle (class 0)
r_inner = np.random.uniform(0, 2, n_inner)
theta_inner = np.random.uniform(0, 2*np.pi, n_inner)
X_inner = np.column_stack([r_inner * np.cos(theta_inner),
                           r_inner * np.sin(theta_inner)])
y_inner = np.zeros(n_inner)

# Outer ring (class 1)
r_outer = np.random.uniform(3, 5, n_outer)
theta_outer = np.random.uniform(0, 2*np.pi, n_outer)
X_outer = np.column_stack([r_outer * np.cos(theta_outer),
                           r_outer * np.sin(theta_outer)])
y_outer = np.ones(n_outer)

# Combine
X = np.vstack([X_inner, X_outer])
y = np.hstack([y_inner, y_outer])

# Polynomial features (degree 2 for circular boundary)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train
model = LogisticRegression()
model.fit(X_poly, y)

# Visualize
fig, ax = plt.subplots(figsize=(10, 10))

# Mesh
xx1, xx2 = np.meshgrid(np.linspace(-6, 6, 300),
                        np.linspace(-6, 6, 300))
X_mesh_poly = poly.transform(np.c_[xx1.ravel(), xx2.ravel()])
Z = model.predict(X_mesh_poly).reshape(xx1.shape)

# Plot
ax.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdYlBu')
ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', s=50, 
          edgecolors='k', alpha=0.7, label='Class 0')
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='s', s=50, 
          edgecolors='k', alpha=0.7, label='Class 1')
ax.set_xlabel('x₁', fontsize=14)
ax.set_ylabel('x₂', fontsize=14)
ax.set_title('Circular Decision Boundary\n(Polynomial Features, Degree 2)', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.show()

print(f"Accuracy: {model.score(X_poly, y) * 100:.2f}%")
```

### Different Degrees, Different Boundaries

```python
from sklearn.datasets import make_moons

# Generate moons data
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

degrees = [1, 2, 3, 5]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, degree in enumerate(degrees):
    ax = axes[idx // 2, idx % 2]
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_poly, y)
    
    # Mesh
    xx1, xx2 = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                            np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
    X_mesh_poly = poly.transform(np.c_[xx1.ravel(), xx2.ravel()])
    Z = model.predict(X_mesh_poly).reshape(xx1.shape)
    
    # Plot
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', s=50, 
              edgecolors='k', alpha=0.7)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='s', s=50, 
              edgecolors='k', alpha=0.7)
    
    accuracy = model.score(X_poly, y)
    ax.set_title(f'Degree {degree}\nAccuracy: {accuracy*100:.1f}%', fontsize=14)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

plt.tight_layout()
plt.show()
```

---

## Degree Selection

### The Bias-Variance Tradeoff

**Polynomial degree** controls model complexity:

```
Degree 1: Underfitting (high bias)
    → Too simple, misses patterns

Degree 2-4: Good fit (balanced)
    → Captures patterns without overfitting

Degree 10+: Overfitting (high variance)
    → Fits noise, poor generalization
```

### Visual Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate data
np.random.seed(42)
X_train = np.linspace(0, 1, 20).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train).ravel() + np.random.normal(0, 0.1, 20)

X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test_true = np.sin(2 * np.pi * X_test).ravel()

# Test different degrees
degrees = [1, 2, 4, 9, 15]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, degree in enumerate(degrees):
    ax = axes[idx // 3, idx % 3]
    
    # Fit
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    # Errors
    train_mse = np.mean((y_train - y_pred_train) ** 2)
    test_mse = np.mean((y_test_true - y_pred_test) ** 2)
    
    # Plot
    ax.scatter(X_train, y_train, color='blue', s=50, alpha=0.7, label='Training data')
    ax.plot(X_test, y_test_true, 'g--', linewidth=2, label='True function')
    ax.plot(X_test, y_pred_test, 'r-', linewidth=2, label='Prediction')
    ax.set_title(f'Degree {degree}\nTrain MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}', 
                fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-2, 2])

# Last plot: Learning curve
ax = axes[1, 2]
degrees_range = range(1, 16)
train_errors = []
test_errors = []

for degree in degrees_range:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    train_errors.append(np.mean((y_train - y_pred_train) ** 2))
    test_errors.append(np.mean((y_test_true - y_pred_test) ** 2))

ax.plot(degrees_range, train_errors, 'b-o', linewidth=2, label='Training error')
ax.plot(degrees_range, test_errors, 'r-s', linewidth=2, label='Test error')
ax.set_xlabel('Polynomial Degree', fontsize=12)
ax.set_ylabel('Mean Squared Error', fontsize=12)
ax.set_title('Model Complexity vs Error', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.axvline(x=4, color='green', linestyle='--', alpha=0.5, label='Optimal')

plt.tight_layout()
plt.show()
```

### Cross-Validation for Degree Selection

**Best method**: Use cross-validation to choose optimal degree.

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Test degrees 1-10
degrees = range(1, 11)
cv_scores = []

for degree in degrees:
    # Create pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('model', LinearRegression())
    ])
    
    # 5-fold cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, 
                            cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())  # Negate to get positive MSE

# Find best degree
best_degree = degrees[np.argmin(cv_scores)]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(degrees, cv_scores, 'o-', linewidth=2, markersize=8)
plt.axvline(x=best_degree, color='r', linestyle='--', linewidth=2, 
           label=f'Best degree = {best_degree}')
plt.xlabel('Polynomial Degree', fontsize=14)
plt.ylabel('Cross-Validation MSE', fontsize=14)
plt.title('Degree Selection via Cross-Validation', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print(f"Best polynomial degree: {best_degree}")
print(f"CV MSE: {cv_scores[best_degree-1]:.4f}")
```

### Information Criteria

**Alternative methods** for model selection:

**Akaike Information Criterion (AIC)**:
```
AIC = 2k - 2ln(L)
```
Where k = number of parameters, L = likelihood

**Bayesian Information Criterion (BIC)**:
```
BIC = k·ln(m) - 2ln(L)
```
Where m = number of samples

**Lower is better** for both AIC and BIC.

**BIC** penalizes complexity more than AIC.

### Practical Guidelines

**Rule of thumb**:
- Start with degree 2 or 3
- Rarely go beyond degree 5
- Use cross-validation to select
- Watch for overfitting signs

**Signs of overfitting**:
- Training error << Test error
- Model too wiggly (many oscillations)
- Poor generalization to new data
- Coefficients very large in magnitude

**Remedies**:
- Lower polynomial degree
- Add regularization (Ridge/Lasso)
- Get more training data
- Feature selection

---

## Regularization with Polynomial Features

### Why Regularization Matters More

**Problem**: Polynomial features create **many features** → prone to overfitting!

**Example**: 10 features, degree 3 → 286 features!

**Solution**: **Always use regularization** with polynomial features.

### Ridge with Polynomials

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline with scaling and regularization
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=10)),
    ('scaler', StandardScaler()),  # IMPORTANT: Scale after polynomial!
    ('ridge', Ridge(alpha=1.0))
])

# Fit
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
```

**Note**: Always **scale** after creating polynomial features!

### Comparing Regularization Strengths

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# Generate data
np.random.seed(42)
X_train = np.linspace(0, 1, 20).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train).ravel() + np.random.normal(0, 0.1, 20)
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test_true = np.sin(2 * np.pi * X_test).ravel()

# High degree polynomial with different regularization
degree = 15
alphas = [0, 0.001, 0.01, 0.1, 1.0, 10]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, alpha in enumerate(alphas):
    ax = axes[idx // 3, idx % 3]
    
    # Pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    
    # Fit
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Errors
    train_mse = np.mean((y_train - y_pred_train) ** 2)
    test_mse = np.mean((y_test_true - y_pred_test) ** 2)
    
    # Plot
    ax.scatter(X_train, y_train, color='blue', s=50, alpha=0.7, label='Training')
    ax.plot(X_test, y_test_true, 'g--', linewidth=2, label='True')
    ax.plot(X_test, y_pred_test, 'r-', linewidth=2, label='Prediction')
    ax.set_title(f'α = {alpha}\nTrain MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}', 
                fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-2, 2])

plt.tight_layout()
plt.show()
```

### Lasso for Feature Selection

**Lasso** can select which polynomial terms are important:

```python
from sklearn.linear_model import Lasso

# High degree polynomial
poly = PolynomialFeatures(degree=10)
X_train_poly = poly.fit_transform(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)

# Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)

# Check which features were selected (non-zero coefficients)
non_zero = np.sum(np.abs(lasso.coef_) > 1e-5)
print(f"Total features: {len(lasso.coef_)}")
print(f"Non-zero coefficients: {non_zero}")
print(f"Selected features: {np.where(np.abs(lasso.coef_) > 1e-5)[0]}")

# Feature names
feature_names = poly.get_feature_names_out()
selected_features = feature_names[np.abs(lasso.coef_) > 1e-5]
print(f"\nSelected polynomial terms:")
for feat in selected_features:
    print(f"  {feat}")
```

---

## Practical Implementation

### Complete Pipeline

```python
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
np.random.seed(42)
X = np.random.uniform(-3, 3, 200).reshape(-1, 1)
y = 0.5*X.ravel()**3 - 2*X.ravel()**2 + 3*X.ravel() + np.random.randn(200)*2

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Hyperparameter grid
param_grid = {
    'poly__degree': [1, 2, 3, 4, 5],
    'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, 
                          scoring='neg_mean_squared_error', 
                          verbose=1, n_jobs=-1)

# Fit
grid_search.fit(X_train, y_train)

# Best parameters
print("="*60)
print("BEST PARAMETERS")
print("="*60)
print(f"Polynomial degree: {grid_search.best_params_['poly__degree']}")
print(f"Ridge alpha: {grid_search.best_params_['ridge__alpha']}")
print(f"Best CV score: {-grid_search.best_score_:.4f}")

# Evaluate on test set
y_pred = grid_search.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("TEST SET PERFORMANCE")
print("="*60)
print(f"MSE: {test_mse:.4f}")
print(f"R²:  {test_r2:.4f}")

# Visualize best model
best_model = grid_search.best_estimator_
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
y_plot = best_model.predict(X_plot)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.5, s=30, label='Training data')
plt.scatter(X_test, y_test, alpha=0.5, s=30, label='Test data')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Best model')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title(f'Best Model: Degree {grid_search.best_params_["poly__degree"]}, ' +
         f'α = {grid_search.best_params_["ridge__alpha"]}', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Advanced Topics

### Splines

**Alternative to polynomials**: Piecewise polynomials

**Advantages**:
- More flexible
- Less prone to overfitting
- Better for long ranges

**Python**:
```python
from scipy.interpolate import UnivariateSpline

# Fit spline
spline = UnivariateSpline(X_train.ravel(), y_train, s=1.0)  # s controls smoothness
y_pred = spline(X_test.ravel())
```

### Kernel Methods

**Kernel trick**: Implicitly map to high-dimensional polynomial space

**Example**: Polynomial kernel in SVM
```python
from sklearn.svm import SVC

# Polynomial kernel
svm = SVC(kernel='poly', degree=3)
svm.fit(X_train, y_train)
```

**Advantage**: Efficient for very high degrees without explicitly creating features

---

## Common Pitfalls and Best Practices

### Pitfall 1: Not Scaling After Polynomials

❌ **Wrong**:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=3)
X_poly = poly.transform(X_scaled)  # Features have different scales again!
```

✅ **Correct**:
```python
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)  # Scale AFTER polynomial
```

### Pitfall 2: Too High Degree

❌ **Wrong**:
```python
poly = PolynomialFeatures(degree=20)  # Too high!
```

**Problem**: 
- Feature explosion
- Overfitting
- Numerical instability

✅ **Correct**:
```python
# Use cross-validation to select degree
# Rarely go beyond degree 5
# Always use regularization
```

### Pitfall 3: Forgetting Regularization

❌ **Wrong**:
```python
poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X)
model = LinearRegression()  # No regularization!
model.fit(X_poly, y)  # Will overfit!
```

✅ **Correct**:
```python
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=10)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))  # Regularization!
])
```

### Best Practices

1. **Always use Pipeline**:
   ```python
   Pipeline([
       ('poly', PolynomialFeatures()),
       ('scaler', StandardScaler()),
       ('model', Ridge())
   ])
   ```

2. **Scale after creating polynomials**

3. **Use cross-validation** for degree selection

4. **Always regularize** (Ridge or Lasso)

5. **Start simple** (degree 2-3), increase if needed

6. **Monitor train vs test error**

7. **Feature selection** with Lasso if too many features

---

## Summary and Next Steps

### What We Learned

1. **Polynomial Regression**
   - Transform x → [x, x², x³, ...]
   - Still linear in weights
   - Number of features grows rapidly

2. **Feature Interactions**
   - Capture combined effects: x₁·x₂
   - Included in polynomial features
   - Essential for many real-world problems

3. **Non-Linear Decision Boundaries**
   - Polynomial features enable non-linear boundaries
   - Linear in transformed space
   - Solves XOR and circular problems

4. **Degree Selection**
   - Use cross-validation
   - Start with degree 2-3
   - Rarely exceed degree 5
   - Watch for overfitting

5. **Regularization**
   - Always use with polynomial features
   - Ridge for stability
   - Lasso for feature selection
   - Scale after creating polynomials

### Key Formulas

**Polynomial features**:
```
φ(x) = [1, x, x², x³, ..., xᵈ]
```

**Number of features** (n original, degree d):
```
Total = C(n+d, d) = (n+d)! / (n! × d!)
```

**Interaction term**:
```
x₁ · x₂
```

### Key Insights

✅ **Polynomials** make linear models flexible  
✅ **Interactions** capture combined effects  
✅ **Always regularize** polynomial features  
✅ **Always scale** after creating polynomials  
✅ **Use CV** for degree selection  
✅ **Feature explosion** happens quickly  

### Prerequisites Checklist

Before moving on:
- [ ] Understand polynomial feature transformation
- [ ] Know how to create interaction features
- [ ] Can solve XOR with polynomial features
- [ ] Understand degree selection process
- [ ] Know why regularization is essential
- [ ] Can implement complete pipeline

### Next Steps

1. **Model_Evaluation.md**
   - Cross-validation strategies
   - Learning curves
   - Validation curves
   - Model selection techniques

2. **Bias_Variance_Tradeoff.md**
   - Understanding bias and variance
   - Diagnosis techniques
   - Remedies for each

3. **Feature_Selection.md**
   - Filter methods
   - Wrapper methods
   - Embedded methods
   - Dimensionality reduction

### Practice Exercises

1. Solve XOR with polynomial features
2. Create circular decision boundary
3. Implement polynomial regression from scratch
4. Use CV to select optimal degree
5. Compare Ridge vs Lasso with polynomials
6. Visualize decision boundaries for different degrees
7. Handle feature explosion with Lasso

### Additional Resources

**Papers**:
- Original polynomial regression papers
- Splines and smoothing methods

**Books**:
- "Elements of Statistical Learning" (Hastie) - Chapter 5
- "Pattern Recognition and Machine Learning" (Bishop)

**Online**:
- Scikit-learn: Polynomial features guide
- Andrew Ng: Feature engineering lectures
- Fast.ai: Practical ML lessons

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║       POLYNOMIAL & INTERACTION FEATURES QUICK REFERENCE        ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  POLYNOMIAL FEATURES                                           ║
║  ───────────────────                                           ║
║  Degree 1: [1, x]                                              ║
║  Degree 2: [1, x, x²]                                          ║
║  Degree 3: [1, x, x², x³]                                      ║
║                                                                ║
║  Two features, degree 2:                                       ║
║    [1, x₁, x₂, x₁², x₁x₂, x₂²]                                 ║
║                        ↑                                       ║
║                   Interaction!                                 ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  NUMBER OF FEATURES                                            ║
╠════════════════════════════════════════════════════════════════╣
║  Formula: C(n+d, d) = (n+d)! / (n! × d!)                       ║
║                                                                ║
║  Examples:                                                     ║
║    n=2, d=2  →  6 features                                     ║
║    n=2, d=3  →  10 features                                    ║
║    n=5, d=3  →  56 features                                    ║
║    n=10, d=3 →  286 features (explosion!)                      ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  DEGREE SELECTION                                              ║
╠════════════════════════════════════════════════════════════════╣
║  1. Use cross-validation                                       ║
║  2. Start with degree 2-3                                      ║
║  3. Rarely exceed degree 5                                     ║
║  4. Monitor train vs test error                                ║
║                                                                ║
║  Degree 1: Underfit (too simple)                               ║
║  Degree 2-4: Usually good                                      ║
║  Degree 10+: Overfit (too complex)                             ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  SCIKIT-LEARN USAGE                                            ║
╠════════════════════════════════════════════════════════════════╣
║  from sklearn.preprocessing import PolynomialFeatures          ║
║  from sklearn.pipeline import Pipeline                         ║
║  from sklearn.linear_model import Ridge                        ║
║                                                                ║
║  pipeline = Pipeline([                                         ║
║      ('poly', PolynomialFeatures(degree=3)),                   ║
║      ('scaler', StandardScaler()),                             ║
║      ('ridge', Ridge(alpha=1.0))                               ║
║  ])                                                            ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  BEST PRACTICES                                                ║
╠════════════════════════════════════════════════════════════════╣
║  1. Always use Pipeline                                        ║
║  2. Scale AFTER creating polynomials                           ║
║  3. Always use regularization (Ridge/Lasso)                    ║
║  4. Use CV for degree selection                                ║
║  5. Watch for feature explosion                                ║
║  6. Monitor overfitting signs                                  ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  CREATING ONLY INTERACTIONS                                    ║
╠════════════════════════════════════════════════════════════════╣
║  poly = PolynomialFeatures(degree=2,                           ║
║                            interaction_only=True,              ║
║                            include_bias=False)                 ║
║                                                                ║
║  Result: [x₁, x₂, x₁x₂]  (no x₁², x₂²)                         ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  NON-LINEAR BOUNDARIES                                         ║
╠════════════════════════════════════════════════════════════════╣
║  In original space: Non-linear (circle, ellipse, curves)       ║
║  In transformed space: Linear (hyperplane)                     ║
║                                                                ║
║  Example: XOR solved with degree 2                             ║
║           Circular boundary with degree 2                      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Congratulations!** You now understand how to create flexible models with polynomial and interaction features!

---

*End of Polynomial_and_Interaction_Features.md*