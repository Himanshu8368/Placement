# Regularization: L1 and L2

## Table of Contents
1. [Introduction to Regularization](#introduction-to-regularization)
2. [The Overfitting Problem](#the-overfitting-problem)
3. [Ridge Regression (L2 Regularization)](#ridge-regression-l2-regularization)
4. [Lasso Regression (L1 Regularization)](#lasso-regression-l1-regularization)
5. [Elastic Net](#elastic-net)
6. [Regularization Parameter Tuning](#regularization-parameter-tuning)
7. [Comparison and When to Use Each](#comparison-and-when-to-use-each)
8. [Practical Implementation](#practical-implementation)
9. [Advanced Topics](#advanced-topics)
10. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to Regularization

### What is Regularization?

**Regularization** is a technique to prevent **overfitting** by adding a penalty term to the loss function that discourages complex models.

**Basic idea**: 
```
Total Cost = Data Fit + Model Complexity Penalty
```

**Why?** Simple models generalize better to new data!

### The Regularization Framework

**Unregularized cost**:
```
J(w) = Loss(w)
```

**Regularized cost**:
```
J(w) = Loss(w) + λ·Penalty(w)
```

Where:
- **Loss(w)**: How well model fits training data (e.g., MSE, cross-entropy)
- **Penalty(w)**: How complex the model is (function of weights)
- **λ** (lambda): Regularization strength (hyperparameter)

### Types of Regularization

**L2 Regularization (Ridge)**:
```
Penalty = ||w||₂² = Σⱼ wⱼ²
```

**L1 Regularization (Lasso)**:
```
Penalty = ||w||₁ = Σⱼ |wⱼ|
```

**Elastic Net**:
```
Penalty = α·||w||₁ + (1-α)·||w||₂²
```

### Intuition

**Without regularization**:
- Model can have large weights
- Overfits to training data
- Poor generalization

**With regularization**:
- Large weights are penalized
- Model stays simpler
- Better generalization

---

## The Overfitting Problem

### What is Overfitting?

**Overfitting** occurs when a model learns the training data **too well**, including its noise and peculiarities, failing to generalize to new data.

```
Underfit          Just Right         Overfit
(High Bias)     (Balanced)        (High Variance)

   ●              ●                  ●
   ●  ╱          ●╱                ● ╱╲
   ● ╱          ●╱                ●╱  ╲●
   ●╱          ●                 ╱     ●
  ╱●          ╱●               ●╱      ●
 ╱           ╱                ╱  ●
╱           ╱                ╱    ●

Too simple    Good fit      Too complex
```

### Signs of Overfitting

**1. Performance gap**:
```
Training error: 2%
Test error: 20%        ← Big gap = overfitting
```

**2. High model complexity**:
```
Many features (high n)
Few examples (small m)
→ n >> m → likely overfit
```

**3. Large weights**:
```
w = [0.1, 500, -300, 0.05, 1000]  ← Very large values
```

### Visual Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

# Generate data
np.random.seed(42)
X = np.linspace(0, 1, 20).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, 20)

# Test data
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test_true = np.sin(2 * np.pi * X_test).ravel()

# Different polynomial degrees
degrees = [1, 4, 15]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, degree in enumerate(degrees):
    ax = axes[idx]
    
    # Transform features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_test_poly = poly.transform(X_test)
    
    # Fit model (no regularization)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict
    y_pred = model.predict(X_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Compute errors
    train_error = np.mean((y - y_pred) ** 2)
    test_error = np.mean((y_test_true - y_test_pred) ** 2)
    
    # Plot
    ax.scatter(X, y, color='blue', s=50, alpha=0.7, label='Training data')
    ax.plot(X_test, y_test_true, 'g--', linewidth=2, label='True function')
    ax.plot(X_test, y_test_pred, 'r-', linewidth=2, label='Prediction')
    
    ax.set_title(f'Degree {degree}\nTrain MSE: {train_error:.3f}, Test MSE: {test_error:.3f}', 
                fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add label
    if degree == 1:
        ax.text(0.5, -1.2, 'UNDERFIT', ha='center', fontsize=14, color='red', weight='bold')
    elif degree == 4:
        ax.text(0.5, -1.2, 'GOOD FIT', ha='center', fontsize=14, color='green', weight='bold')
    else:
        ax.text(0.5, -1.2, 'OVERFIT', ha='center', fontsize=14, color='red', weight='bold')

plt.tight_layout()
plt.show()
```

### Causes of Overfitting

**1. Too many features**:
```
m = 100 examples
n = 200 features
→ Model memorizes training data
```

**2. High model complexity**:
```
Polynomial degree too high
Deep neural network with many layers
```

**3. Insufficient training data**:
```
Complex model + small dataset = overfitting
```

**4. Training too long**:
```
Gradient descent iterates too many times
Model starts fitting noise
```

### Bias-Variance Tradeoff

**Total Error = Bias² + Variance + Irreducible Error**

```
    Error
    │
    │        Total Error
    │       ╱
    │      ╱╲
    │     ╱  ╲     Variance
    │    ╱    ╲   ╱
    │   ╱      ╲ ╱
    │  ╱    Bias²
    │ ╱          ╲
    │╱____________╲_______________
    │
    └────────────────────────── Model Complexity
    
    Underfit  Optimal  Overfit
```

**Bias**: Error from overly simple assumptions (underfitting)
**Variance**: Error from sensitivity to training data fluctuations (overfitting)

**Regularization reduces variance** by constraining model complexity!

---

## Ridge Regression (L2 Regularization)

### Definition

**Ridge regression** adds an **L2 penalty** (sum of squared weights) to the cost function.

**For linear regression**:
```
J(w) = (1/2m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)² + (λ/2m) Σⱼ₌₁ⁿ wⱼ²
     = MSE + (λ/2m)||w||₂²
```

**For logistic regression**:
```
J(w) = -(1/m) Σᵢ [y⁽ⁱ⁾ log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-ŷ⁽ⁱ⁾)] + (λ/2m) Σⱼ wⱼ²
```

**Key points**:
- Only penalize feature weights (w₁, w₂, ..., wₙ)
- **Don't penalize bias** (w₀)
- Factor 1/2 is for convenience (simplifies derivative)

### Why L2?

**Squared penalty**:
- **Differentiable everywhere**: Smooth optimization
- **Shrinks all weights**: Proportional to their size
- **Never exactly zero**: Keeps all features
- **Closed-form solution exists**: Can solve analytically

### Effect on Weights

Ridge **shrinks** weights toward zero but never makes them exactly zero:

```
Without regularization: w = [10, -8, 0.1, 15, -12]
With Ridge (λ=0.1):     w = [5, -4, 0.05, 7.5, -6]
With Ridge (λ=1.0):     w = [2, -1.6, 0.02, 3, -2.4]
With Ridge (λ=10):      w = [0.5, -0.4, 0.005, 0.75, -0.6]
```

All weights shrink, but **none become exactly 0**.

### Mathematical Derivation

**Cost function**:
```
J(w) = (1/2m)||Xw - y||² + (λ/2m)||w||²
```

**Matrix form**:
```
J(w) = (1/2m)[(Xw - y)ᵀ(Xw - y) + λwᵀw]
```

**Gradient**:
```
∇J(w) = (1/m)Xᵀ(Xw - y) + (λ/m)w
```

**Setting gradient to zero**:
```
Xᵀ(Xw - y) + λw = 0
XᵀXw - Xᵀy + λw = 0
(XᵀX + λI)w = Xᵀy
```

**Closed-form solution**:
```
w = (XᵀX + λI)⁻¹Xᵀy
```

### Key Insight

Adding λI to XᵀX:
- **Ensures invertibility**: Even if XᵀX is singular
- **Improves conditioning**: More numerically stable
- **Regularizes**: Prevents overfitting

### Gradient Descent Update

**For Ridge regression**:
```
w ← w - α[(1/m)Xᵀ(Xw - y) + (λ/m)w]
  = w - α(1/m)Xᵀ(Xw - y) - α(λ/m)w
  = w(1 - αλ/m) - α(1/m)Xᵀ(Xw - y)
```

**Weight decay**: The term (1 - αλ/m) multiplies w, causing decay!

### Python Implementation

```python
import numpy as np

class RidgeRegression:
    """Ridge Regression with L2 regularization"""
    
    def __init__(self, alpha=1.0, learning_rate=0.01, num_iterations=1000):
        """
        Args:
            alpha: Regularization strength (λ)
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of iterations
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.cost_history = []
    
    def _compute_cost(self, X, y, y_pred):
        """Compute cost with L2 regularization"""
        m = len(y)
        # MSE loss
        mse = (1/(2*m)) * np.sum((y_pred - y) ** 2)
        # L2 penalty (don't regularize bias)
        l2_penalty = (self.alpha/(2*m)) * np.sum(self.weights[1:] ** 2)
        return mse + l2_penalty
    
    def fit(self, X, y):
        """Train Ridge regression using gradient descent"""
        # Add bias column
        X_b = np.c_[np.ones(X.shape[0]), X]
        y = y.reshape(-1, 1)
        
        m, n = X_b.shape
        self.weights = np.zeros((n, 1))
        
        for i in range(self.num_iterations):
            # Predictions
            y_pred = X_b @ self.weights
            
            # Compute cost
            cost = self._compute_cost(X_b, y, y_pred)
            self.cost_history.append(cost)
            
            # Gradient with L2 regularization
            gradient = (1/m) * X_b.T @ (y_pred - y)
            
            # Add regularization gradient (don't regularize bias)
            reg_gradient = (self.alpha/m) * self.weights
            reg_gradient[0] = 0  # Don't regularize bias
            gradient += reg_gradient
            
            # Update weights
            self.weights -= self.learning_rate * gradient
            
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
        
        return self
    
    def fit_closed_form(self, X, y):
        """Train using closed-form solution"""
        X_b = np.c_[np.ones(X.shape[0]), X]
        y = y.reshape(-1, 1)
        
        n = X_b.shape[1]
        # Identity matrix (don't regularize bias)
        I = np.eye(n)
        I[0, 0] = 0
        
        # Closed-form solution
        self.weights = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b @ self.weights
    
    def score(self, X, y):
        """Compute R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y.reshape(-1, 1) - y_pred) ** 2)
        ss_tot = np.sum((y.reshape(-1, 1) - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Example usage
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_regression(n_samples=100, n_features=20, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with gradient descent
ridge_gd = RidgeRegression(alpha=1.0, learning_rate=0.01, num_iterations=1000)
ridge_gd.fit(X_train, y_train)

# Train with closed form
ridge_cf = RidgeRegression(alpha=1.0)
ridge_cf.fit_closed_form(X_train, y_train)

print(f"\nGradient Descent - Test R²: {ridge_gd.score(X_test, y_test):.4f}")
print(f"Closed Form - Test R²: {ridge_cf.score(X_test, y_test):.4f}")
```

### Using Scikit-Learn

```python
from sklearn.linear_model import Ridge

# alpha parameter is λ
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Predict
y_pred = ridge.predict(X_test)

# Score
score = ridge.score(X_test, y_test)
print(f"R² score: {score:.4f}")

# Coefficients
print(f"Weights: {ridge.coef_}")
print(f"Bias: {ridge.intercept_}")
```

### Visualization

```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Generate data with polynomial features
X = np.linspace(0, 1, 20).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, 20)

X_test = np.linspace(0, 1, 100).reshape(-1, 1)

# High-degree polynomial (prone to overfitting)
poly = PolynomialFeatures(degree=15)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)

# Different regularization strengths
alphas = [0, 0.001, 0.1, 10]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, alpha in enumerate(alphas):
    ax = axes[idx // 2, idx % 2]
    
    # Train
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_poly, y)
    
    # Predict
    y_pred = ridge.predict(X_test_poly)
    
    # Plot
    ax.scatter(X, y, color='blue', s=50, alpha=0.7, label='Data')
    ax.plot(X_test, y_pred, 'r-', linewidth=2, label=f'Ridge (α={alpha})')
    ax.set_title(f'λ = {alpha}\nWeights: max={np.max(np.abs(ridge.coef_)):.2f}', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-2, 2])

plt.tight_layout()
plt.show()
```

---

## Lasso Regression (L1 Regularization)

### Definition

**Lasso** (Least Absolute Shrinkage and Selection Operator) adds an **L1 penalty** (sum of absolute values of weights).

**Cost function**:
```
J(w) = (1/2m) Σᵢ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)² + (λ/m) Σⱼ |wⱼ|
     = MSE + (λ/m)||w||₁
```

### Why L1?

**Absolute value penalty**:
- **Creates sparsity**: Sets some weights to **exactly zero**
- **Feature selection**: Automatically selects important features
- **Interpretable**: Simpler models with fewer features
- **No closed-form**: Must use iterative methods

### Effect on Weights

Lasso can **eliminate** features by setting weights to zero:

```
Without regularization: w = [10, -8, 0.1, 15, -12]
With Lasso (λ=0.1):     w = [5, -4, 0, 7.5, -6]        ← One zero
With Lasso (λ=1.0):     w = [2, 0, 0, 3, 0]            ← Three zeros
With Lasso (λ=10):      w = [0, 0, 0, 0.5, 0]          ← Four zeros
```

**Key difference from Ridge**: Some weights become **exactly 0** → automatic feature selection!

### Gradient (Subgradient)

**Problem**: |w| is not differentiable at w=0.

**Solution**: Use **subgradient**:
```
∂|w|/∂w = { +1  if w > 0
          { -1  if w < 0
          { [-1, 1]  if w = 0  (any value in this range)
```

In practice, use **sign** function:
```
∂J/∂wⱼ = (1/m) Σᵢ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)x⁽ⁱ⁾ⱼ + (λ/m) sign(wⱼ)
```

### Why Lasso Creates Sparsity

**Geometric intuition**:

```
Ridge (L2):                    Lasso (L1):
Constraint: Σwⱼ² ≤ c          Constraint: Σ|wⱼ| ≤ c

    w₂                            w₂
    │                             │
    │    ╱○╲                       │    ╱│╲
    │  ╱  ●  ╲                     │  ╱ │ ╲
    │ ╱   │   ╲                    │ ╱  ●  ╲
    │────●────●──── w₁              │────●────── w₁
    │ ╲   │   ╱                    │ ╲  │  ╱
    │  ╲  │  ╱                     │  ╲ │ ╱
    │   ╲○╱                        │   ╲│╱
    
Circle (smooth)               Diamond (corners)
Likely to touch at non-axis   Likely to touch at axis
→ Non-zero w                  → Zero w (sparse!)
```

The **corners** of the L1 constraint (diamond) tend to align with axes, producing zeros!

### Python Implementation

```python
import numpy as np

class LassoRegression:
    """Lasso Regression with L1 regularization"""
    
    def __init__(self, alpha=1.0, learning_rate=0.01, num_iterations=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.cost_history = []
    
    def _soft_threshold(self, w, threshold):
        """
        Soft thresholding operator for L1 regularization
        
        This is used in coordinate descent for Lasso
        """
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)
    
    def _compute_cost(self, X, y, y_pred):
        """Compute cost with L1 regularization"""
        m = len(y)
        # MSE loss
        mse = (1/(2*m)) * np.sum((y_pred - y) ** 2)
        # L1 penalty (don't regularize bias)
        l1_penalty = (self.alpha/m) * np.sum(np.abs(self.weights[1:]))
        return mse + l1_penalty
    
    def fit(self, X, y):
        """Train Lasso using proximal gradient descent"""
        X_b = np.c_[np.ones(X.shape[0]), X]
        y = y.reshape(-1, 1)
        
        m, n = X_b.shape
        self.weights = np.zeros((n, 1))
        
        for i in range(self.num_iterations):
            # Predictions
            y_pred = X_b @ self.weights
            
            # Compute cost
            cost = self._compute_cost(X_b, y, y_pred)
            self.cost_history.append(cost)
            
            # Gradient of MSE part
            gradient = (1/m) * X_b.T @ (y_pred - y)
            
            # Gradient step
            self.weights -= self.learning_rate * gradient
            
            # Proximal step (soft thresholding) for L1
            threshold = self.learning_rate * self.alpha / m
            self.weights[1:] = self._soft_threshold(self.weights[1:], threshold)
            # Don't threshold bias
            
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}, Non-zero weights: {np.sum(np.abs(self.weights[1:]) > 1e-5)}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b @ self.weights
    
    def score(self, X, y):
        """Compute R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y.reshape(-1, 1) - y_pred) ** 2)
        ss_tot = np.sum((y.reshape(-1, 1) - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Example
X, y = make_regression(n_samples=100, n_features=20, n_informative=10, 
                      noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso = LassoRegression(alpha=1.0, learning_rate=0.01, num_iterations=1000)
lasso.fit(X_train, y_train)

print(f"\nTest R²: {lasso.score(X_test, y_test):.4f}")
print(f"Non-zero weights: {np.sum(np.abs(lasso.weights[1:]) > 1e-5)} out of 20")
```

### Using Scikit-Learn

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1.0, max_iter=10000)
lasso.fit(X_train, y_train)

# Predict
y_pred = lasso.predict(X_test)

# Score
score = lasso.score(X_test, y_test)
print(f"R² score: {score:.4f}")

# Check sparsity
non_zero = np.sum(np.abs(lasso.coef_) > 1e-5)
print(f"Non-zero coefficients: {non_zero} out of {len(lasso.coef_)}")
print(f"Selected features: {np.where(np.abs(lasso.coef_) > 1e-5)[0]}")
```

### Visualization: Feature Selection

```python
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Generate data with many features, only some relevant
np.random.seed(42)
n_samples, n_features = 100, 50
n_informative = 10

X = np.random.randn(n_samples, n_features)
# Only first 10 features are informative
true_coef = np.zeros(n_features)
true_coef[:n_informative] = np.random.randn(n_informative) * 5
y = X @ true_coef + np.random.randn(n_samples) * 2

# Try different alphas
alphas = [0.001, 0.01, 0.1, 1.0, 10]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, alpha in enumerate(alphas):
    ax = axes[idx // 3, idx % 3]
    
    # Train Lasso
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X, y)
    
    # Plot coefficients
    ax.stem(range(n_features), lasso.coef_, basefmt=' ')
    ax.stem(range(n_informative), true_coef[:n_informative], 
           linefmt='r-', markerfmt='ro', basefmt=' ', label='True')
    ax.set_title(f'α = {alpha}\nNon-zero: {np.sum(np.abs(lasso.coef_) > 1e-5)}', 
                fontsize=12)
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Coefficient value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Last plot: True coefficients
ax = axes[1, 2]
ax.stem(range(n_features), true_coef, basefmt=' ')
ax.set_title('True Coefficients', fontsize=12)
ax.set_xlabel('Feature index')
ax.set_ylabel('Coefficient value')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()
```

---

## Elastic Net

### Definition

**Elastic Net** combines **L1 and L2** penalties, getting benefits of both!

**Cost function**:
```
J(w) = MSE + λ₁||w||₁ + λ₂||w||₂²
     = MSE + λ[α||w||₁ + (1-α)/2·||w||₂²]
```

Where:
- **λ**: Overall regularization strength
- **α ∈ [0, 1]**: Mixing parameter
  - α = 0: Pure Ridge (L2)
  - α = 1: Pure Lasso (L1)
  - 0 < α < 1: Mix of both

### Why Elastic Net?

**Combines advantages**:
- **From L1**: Feature selection (sparsity)
- **From L2**: Stability, handles correlated features

**When to use**:
- **Correlated features**: Lasso picks one arbitrarily, Elastic Net includes both
- **High-dimensional data**: n >> m
- **Want feature selection**: But also stability

### Example: Correlated Features

```
Features x₁ and x₂ are highly correlated

Lasso:       Picks one (x₁ or x₂), sets other to 0
Ridge:       Spreads weight between both
Elastic Net: Picks both but with shrinkage
```

### Python Implementation

```python
class ElasticNet:
    """Elastic Net regression"""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01, num_iterations=1000):
        """
        Args:
            alpha: Overall regularization strength (λ)
            l1_ratio: L1 mixing parameter (α), between 0 and 1
            learning_rate: Step size
            num_iterations: Number of iterations
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.cost_history = []
    
    def _soft_threshold(self, w, threshold):
        """Soft thresholding for L1 component"""
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)
    
    def _compute_cost(self, X, y, y_pred):
        """Compute cost with Elastic Net penalty"""
        m = len(y)
        # MSE loss
        mse = (1/(2*m)) * np.sum((y_pred - y) ** 2)
        # L1 penalty
        l1_penalty = (self.alpha * self.l1_ratio / m) * np.sum(np.abs(self.weights[1:]))
        # L2 penalty
        l2_penalty = (self.alpha * (1 - self.l1_ratio) / (2*m)) * np.sum(self.weights[1:] ** 2)
        return mse + l1_penalty + l2_penalty
    
    def fit(self, X, y):
        """Train Elastic Net"""
        X_b = np.c_[np.ones(X.shape[0]), X]
        y = y.reshape(-1, 1)
        
        m, n = X_b.shape
        self.weights = np.zeros((n, 1))
        
        for i in range(self.num_iterations):
            # Predictions
            y_pred = X_b @ self.weights
            
            # Compute cost
            cost = self._compute_cost(X_b, y, y_pred)
            self.cost_history.append(cost)
            
            # Gradient of MSE
            gradient = (1/m) * X_b.T @ (y_pred - y)
            
            # Add L2 gradient
            l2_gradient = (self.alpha * (1 - self.l1_ratio) / m) * self.weights
            l2_gradient[0] = 0  # Don't regularize bias
            gradient += l2_gradient
            
            # Update with gradient
            self.weights -= self.learning_rate * gradient
            
            # Soft thresholding for L1
            threshold = self.learning_rate * self.alpha * self.l1_ratio / m
            self.weights[1:] = self._soft_threshold(self.weights[1:], threshold)
            
            if i % 100 == 0:
                non_zero = np.sum(np.abs(self.weights[1:]) > 1e-5)
                print(f"Iteration {i}: Cost = {cost:.4f}, Non-zero: {non_zero}")
        
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b @ self.weights
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y.reshape(-1, 1) - y_pred) ** 2)
        ss_tot = np.sum((y.reshape(-1, 1) - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Example
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, learning_rate=0.01, num_iterations=1000)
elastic.fit(X_train, y_train)

print(f"\nTest R²: {elastic.score(X_test, y_test):.4f}")
```

### Using Scikit-Learn

```python
from sklearn.linear_model import ElasticNet

# alpha = λ, l1_ratio = α (mixing parameter)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
elastic.fit(X_train, y_train)

score = elastic.score(X_test, y_test)
non_zero = np.sum(np.abs(elastic.coef_) > 1e-5)

print(f"R² score: {score:.4f}")
print(f"Non-zero coefficients: {non_zero} out of {X_train.shape[1]}")
```

### Comparison Visualization

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt

# Generate data with correlated features
np.random.seed(42)
n_samples = 100
# Create two highly correlated features
x1 = np.random.randn(n_samples)
x2 = x1 + np.random.randn(n_samples) * 0.1  # Highly correlated with x1
X = np.column_stack([x1, x2])
y = 3*x1 + 2*x2 + np.random.randn(n_samples) * 0.5

# Train different models
models = {
    'Ridge': Ridge(alpha=0.1),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx]
    
    # Train
    model.fit(X, y)
    
    # Plot coefficients
    ax.bar(['x₁', 'x₂'], model.coef_)
    ax.set_title(f'{name}\nCoefficients: [{model.coef_[0]:.2f}, {model.coef_[1]:.2f}]',
                fontsize=14)
    ax.set_ylabel('Coefficient value')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

print("Analysis of correlated features:")
print(f"Correlation between x₁ and x₂: {np.corrcoef(x1, x2)[0, 1]:.3f}")
print(f"\nRidge:       w₁={models['Ridge'].coef_[0]:.3f}, w₂={models['Ridge'].coef_[1]:.3f}")
print(f"Lasso:       w₁={models['Lasso'].coef_[0]:.3f}, w₂={models['Lasso'].coef_[1]:.3f}")
print(f"Elastic Net: w₁={models['Elastic Net'].coef_[0]:.3f}, w₂={models['Elastic Net'].coef_[1]:.3f}")
```

---

## Regularization Parameter Tuning

### The λ Parameter

**λ (lambda)** controls the strength of regularization:

```
λ = 0:    No regularization (prone to overfitting)
λ small:  Light regularization
λ medium: Balanced
λ large:  Heavy regularization (prone to underfitting)
λ → ∞:   All weights → 0 (predicts mean)
```

### Finding Optimal λ

**Methods**:
1. **Cross-validation**: Most reliable
2. **Validation set**: Simple but less data for training
3. **Information criteria**: AIC, BIC (for statistical models)
4. **Learning curves**: Visual inspection

### Cross-Validation

**K-Fold Cross-Validation**:
```
1. Split data into K folds
2. For each λ:
   For each fold k:
      Train on K-1 folds
      Validate on fold k
   Average validation error
3. Choose λ with lowest average error
```

**Python implementation**:
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import numpy as np

# Range of lambda values to try
alphas = np.logspace(-3, 3, 100)  # [0.001, 0.003, ..., 1000]

# Store mean CV scores
mean_scores = []
std_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    # 5-fold cross-validation
    scores = cross_val_score(ridge, X_train, y_train, cv=5, 
                            scoring='neg_mean_squared_error')
    mean_scores.append(-scores.mean())  # Negate to get positive MSE
    std_scores.append(scores.std())

mean_scores = np.array(mean_scores)
std_scores = np.array(std_scores)

# Find best alpha
best_idx = np.argmin(mean_scores)
best_alpha = alphas[best_idx]

print(f"Best λ: {best_alpha:.4f}")
print(f"Best CV MSE: {mean_scores[best_idx]:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, mean_scores, linewidth=2, label='CV Error')
plt.fill_between(alphas, 
                 mean_scores - std_scores,
                 mean_scores + std_scores,
                 alpha=0.3)
plt.axvline(x=best_alpha, color='r', linestyle='--', 
           label=f'Best λ = {best_alpha:.4f}')
plt.xlabel('λ (Regularization strength)', fontsize=14)
plt.ylabel('Cross-Validation MSE', fontsize=14)
plt.title('Ridge Regression: λ Selection via CV', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Using RidgeCV and LassoCV

Scikit-learn has built-in CV versions:

```python
from sklearn.linear_model import RidgeCV, LassoCV

# Ridge with automatic CV
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Best λ: {ridge_cv.alpha_:.4f}")
print(f"Test R²: {ridge_cv.score(X_test, y_test):.4f}")

# Lasso with automatic CV
lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train, y_train)

print(f"\nLasso best λ: {lasso_cv.alpha_:.4f}")
print(f"Lasso test R²: {lasso_cv.score(X_test, y_test):.4f}")
```

### Regularization Path

Visualize how coefficients change with λ:

```python
from sklearn.linear_model import lasso_path, ridge_regression
import matplotlib.pyplot as plt

# Lasso path
alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train)

# Ridge path
alphas_ridge = np.logspace(-3, 3, 100)
coefs_ridge = []
for alpha in alphas_ridge:
    coef = ridge_regression(X_train, y_train, alpha=alpha)
    coefs_ridge.append(coef)
coefs_ridge = np.array(coefs_ridge).T

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Lasso path
ax = axes[0]
for i in range(coefs_lasso.shape[0]):
    ax.semilogx(alphas_lasso, coefs_lasso[i, :], linewidth=2)
ax.set_xlabel('λ', fontsize=14)
ax.set_ylabel('Coefficients', fontsize=14)
ax.set_title('Lasso Regularization Path', fontsize=16)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Ridge path
ax = axes[1]
for i in range(coefs_ridge.shape[0]):
    ax.semilogx(alphas_ridge, coefs_ridge[i, :], linewidth=2)
ax.set_xlabel('λ', fontsize=14)
ax.set_ylabel('Coefficients', fontsize=14)
ax.set_title('Ridge Regularization Path', fontsize=16)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()
```

**Observations**:
- **Lasso**: Coefficients go to exactly 0 (sparse)
- **Ridge**: Coefficients shrink but stay non-zero

---

## Comparison and When to Use Each

### Summary Table

| Method | Penalty | Sparsity | Closed-Form | Feature Selection | Correlated Features |
|--------|---------|----------|-------------|-------------------|---------------------|
| **None** | - | No | Yes | No | Unstable |
| **Ridge (L2)** | ||w||₂² | No | Yes | No | Groups together |
| **Lasso (L1)** | ||w||₁ | No | No | Yes | Picks one |
| **Elastic Net** | α||w||₁ + (1-α)||w||₂² | Moderate | No | Yes | Groups + selects |

### Decision Guide

```
Start
  │
  ├─ Do you have correlated features?
  │  ├─ Yes → Elastic Net or Ridge
  │  └─ No → Continue
  │
  ├─ Do you want feature selection?
  │  ├─ Yes → Lasso or Elastic Net
  │  └─ No → Ridge
  │
  ├─ Is n >> m? (many features, few examples)
  │  ├─ Yes → Elastic Net
  │  └─ No → Ridge or Lasso
  │
  └─ Default: Try all, use CV to select
```

### Use Cases

**Ridge Regression**:
- ✅ All features potentially useful
- ✅ Correlated features (keeps them all)
- ✅ Want smooth solutions
- ✅ Multicollinearity issues
- ❌ Need sparse model

**Lasso Regression**:
- ✅ Feature selection needed
- ✅ Many irrelevant features
- ✅ Want interpretable sparse model
- ✅ Feature independence
- ❌ Correlated features present

**Elastic Net**:
- ✅ Best of both worlds
- ✅ Correlated features + selection
- ✅ High-dimensional data (n >> m)
- ✅ Not sure which to use
- ✅ **Generally recommended**

---

## Practical Implementation

### Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
X, y = make_regression(n_samples=200, n_features=50, n_informative=20,
                      noise=10, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for regularization!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
}

# Train and evaluate
results = {}
for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Count non-zero coefficients
    if hasattr(model, 'coef_'):
        non_zero = np.sum(np.abs(model.coef_) > 1e-5)
    else:
        non_zero = X_train.shape[1]
    
    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'non_zero': non_zero
    }

# Print results
print("="*80)
print("MODEL COMPARISON")
print("="*80)
print(f"{'Model':<20} {'Train MSE':<12} {'Test MSE':<12} {'Test R²':<12} {'Non-zero':<10}")
print("-"*80)
for name, res in results.items():
    print(f"{name:<20} {res['train_mse']:<12.2f} {res['test_mse']:<12.2f} "
          f"{res['test_r2']:<12.4f} {res['non_zero']:<10}")

# Visualize coefficients
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx // 2, idx % 2]
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        ax.stem(range(len(coefs)), coefs, basefmt=' ')
        ax.set_title(f'{name}\nNon-zero: {np.sum(np.abs(coefs) > 1e-5)}', fontsize=14)
        ax.set_xlabel('Feature index')
        ax.set_ylabel('Coefficient value')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()
```

---

## Advanced Topics

### Early Stopping

Stop training before overfitting occurs:

```python
def train_with_early_stopping(X_train, y_train, X_val, y_val, 
                              alpha=1.0, patience=10):
    """Train with early stopping based on validation error"""
    from sklearn.linear_model import Ridge
    
    best_val_error = float('inf')
    best_model = None
    wait = 0
    
    for i in range(1000):
        model = Ridge(alpha=alpha, max_iter=i+1)
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        val_error = np.mean((val_pred - y_val) ** 2)
        
        if val_error < best_val_error:
            best_val_error = val_error
            best_model = model
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at iteration {i}")
                break
    
    return best_model
```

### Bayesian Interpretation

Regularization can be viewed as imposing **prior distributions** on weights:

**Ridge (L2)**: Gaussian prior
```
p(w) ∝ exp(-||w||₂²)  (Gaussian distribution)
```

**Lasso (L1)**: Laplace prior
```
p(w) ∝ exp(-||w||₁)  (Laplace distribution)
```

**Maximum A Posteriori (MAP)** estimation with priors = regularized MLE!

### Group Lasso

Select groups of features together:

```python
from sklearn.linear_model import MultiTaskLasso

# For problems with multiple related outputs
# Selects features that are relevant for all tasks
```

---

## Summary and Next Steps

### What We Learned

1. **Overfitting Problem**
   - Training error low, test error high
   - Caused by model complexity, insufficient data
   - Bias-variance tradeoff

2. **Ridge Regression (L2)**
   - Penalty: ||w||₂²
   - Shrinks all weights
   - Closed-form solution exists
   - Good for correlated features

3. **Lasso Regression (L1)**
   - Penalty: ||w||₁
   - Creates sparsity (zeros)
   - Automatic feature selection
   - No closed-form solution

4. **Elastic Net**
   - Combines L1 and L2
   - Best of both worlds
   - Handles correlated features + selection
   - Generally recommended

5. **Parameter Tuning**
   - Use cross-validation
   - Try multiple λ values
   - Visualize regularization path
   - RidgeCV/LassoCV for automation

### Key Formulas

**Ridge**:
```
J(w) = MSE + (λ/2m)||w||₂²
w = (XᵀX + λI)⁻¹Xᵀy
```

**Lasso**:
```
J(w) = MSE + (λ/m)||w||₁
∇J includes sign(w) term
```

**Elastic Net**:
```
J(w) = MSE + λ[α||w||₁ + (1-α)/2·||w||₂²]
```

### Key Insights

✅ **Regularization prevents overfitting**  
✅ **L2 shrinks, L1 selects**  
✅ **Always scale features first**  
✅ **Use CV for tuning λ**  
✅ **Elastic Net is often best default**  
✅ **Visualize regularization paths**  

### Prerequisites Checklist

Before moving on:
- [ ] Understand overfitting problem
- [ ] Know difference between L1 and L2
- [ ] Can implement Ridge and Lasso
- [ ] Understand when to use each
- [ ] Know how to tune λ with CV
- [ ] Can visualize regularization paths

### Next Steps

1. **Feature_Engineering.md**
   - Polynomial features
   - Interaction terms
   - Feature selection methods
   - Dimensionality reduction

2. **Model_Evaluation.md**
   - Cross-validation strategies
   - Learning curves
   - Bias-variance analysis
   - Model selection

3. **Advanced_Topics.md**
   - Kernel ridge regression
   - Group lasso
   - Sparse coding
   - Bayesian regularization

### Practice Exercises

1. Implement Ridge and Lasso from scratch
2. Compare all methods on real dataset
3. Visualize regularization paths
4. Tune λ using cross-validation
5. Apply to high-dimensional data (n > m)
6. Analyze feature selection by Lasso
7. Compare Elastic Net with different α values

### Additional Resources

**Papers**:
- Hoerl & Kennard (1970): Ridge Regression
- Tibshirani (1996): Lasso Regression
- Zou & Hastie (2005): Elastic Net

**Books**:
- "Elements of Statistical Learning" (Hastie) - Chapter 3
- "Statistical Learning with Sparsity" (Hastie)
- "Pattern Recognition and Machine Learning" (Bishop)

**Online**:
- Stanford CS229: Regularization notes
- Scikit-learn: Linear Models documentation
- Andrew Ng: Regularization lectures

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║            REGULARIZATION QUICK REFERENCE                      ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  RIDGE (L2)                                                    ║
║  ──────────                                                    ║
║  J(w) = MSE + (λ/2m)||w||₂²                                    ║
║                                                                ║
║  • Shrinks all weights                                         ║
║  • No sparsity (all weights non-zero)                          ║
║  • Closed-form: w = (XᵀX + λI)⁻¹Xᵀy                            ║
║  • Good for correlated features                                ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  LASSO (L1)                                                    ║
║  ───────────                                                   ║
║  J(w) = MSE + (λ/m)||w||₁                                      ║
║                                                                ║
║  • Creates sparsity (some weights = 0)                         ║
║  • Automatic feature selection                                 ║
║  • No closed-form solution                                     ║
║  • Picks one from correlated features                          ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  ELASTIC NET                                                   ║
║  ────────────                                                  ║
║  J(w) = MSE + λ[α||w||₁ + (1-α)/2·||w||₂²]                     ║
║                                                                ║
║  • Combines L1 and L2                                          ║
║  • α = 0: Pure Ridge                                           ║
║  • α = 1: Pure Lasso                                           ║
║  • α ∈ (0,1): Mix                                              ║
║  • Best of both worlds                                         ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  WHEN TO USE                                                   ║
╠════════════════════════════════════════════════════════════════╣
║  Ridge:                                                        ║
║    • All features useful                                       ║
║    • Correlated features                                       ║
║    • Want smooth solutions                                     ║
║                                                                ║
║  Lasso:                                                        ║
║    • Feature selection needed                                  ║
║    • Many irrelevant features                                  ║
║    • Sparse model desired                                      ║
║                                                                ║
║  Elastic Net:                                                  ║
║    • High-dimensional (n >> m)                                 ║
║    • Correlated + selection                                    ║
║    • Default choice                                            ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  TUNING λ                                                      ║
╠════════════════════════════════════════════════════════════════╣
║  1. Try: λ ∈ [0.001, 0.01, 0.1, 1, 10, 100]                   ║
║  2. Use cross-validation                                       ║
║  3. RidgeCV / LassoCV for automation                           ║
║  4. Visualize regularization path                              ║
║                                                                ║
║  λ = 0:    No regularization                                   ║
║  λ small:  Light regularization                                ║
║  λ large:  Heavy regularization                                ║
║  λ → ∞:    All weights → 0                                     ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  BEST PRACTICES                                                ║
╠════════════════════════════════════════════════════════════════╣
║  1. Always scale features first                                ║
║  2. Don't regularize bias term                                 ║
║  3. Use CV to select λ                                         ║
║  4. Start with Elastic Net                                     ║
║  5. Visualize regularization paths                             ║
║  6. Check for overfitting (train vs test)                      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Congratulations!** You've completed the core linear predictors content!

---

*End of Regularization_L1_L2.md*
