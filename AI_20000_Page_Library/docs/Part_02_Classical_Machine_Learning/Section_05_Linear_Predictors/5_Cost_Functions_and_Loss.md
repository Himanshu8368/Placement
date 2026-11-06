# Cost Functions and Loss

## Table of Contents
1. [Introduction to Loss Functions](#introduction-to-loss-functions)
2. [Mean Squared Error (MSE)](#mean-squared-error-mse)
3. [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
4. [Root Mean Squared Error (RMSE)](#root-mean-squared-error-rmse)
5. [Huber Loss](#huber-loss)
6. [Comparison of Loss Functions](#comparison-of-loss-functions)
7. [Why We Use Different Losses](#why-we-use-different-losses)
8. [Probabilistic Interpretation](#probabilistic-interpretation)
9. [Loss Functions for Classification](#loss-functions-for-classification)
10. [Choosing the Right Loss](#choosing-the-right-loss)
11. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to Loss Functions

### What is a Loss Function?

A **loss function** (or **cost function**) measures how well our model's predictions match the actual values.

**Terminology**:
- **Loss function** L(ŷ, y): Error for a **single example**
- **Cost function** J(w): Average loss over **all examples**

```
Cost = Average of all losses
J(w) = (1/m) Σᵢ₌₁ᵐ L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
```

### Why Do We Need Loss Functions?

**Purpose**:
1. **Quantify error**: How wrong are our predictions?
2. **Guide optimization**: What direction to move parameters?
3. **Compare models**: Which model is better?

**The learning process**:
```
1. Choose loss function L
2. Compute predictions ŷ = f(x; w)
3. Compute cost J(w) = average loss
4. Minimize J(w) by adjusting w
```

### Desirable Properties

A good loss function should be:
1. **Non-negative**: L(ŷ, y) ≥ 0
2. **Zero at perfect prediction**: L(y, y) = 0
3. **Differentiable**: Can compute gradients
4. **Convex** (ideally): One global minimum
5. **Robust**: Not too sensitive to outliers (sometimes)

### Visual Intuition

```
    Loss
    │
    │      /
    │     /
    │    /
    │   /
    │  /
    │ /
    │/___________  Correct prediction
    │
    │  (Error = 0)
    │
    └──────────────────── Prediction Error
    
    Good loss: Small when close, large when far
```

---

## Mean Squared Error (MSE)

### Definition

**Mean Squared Error** is the average of squared differences between predictions and actual values.

**For single example**:
```
L_MSE(ŷ, y) = (ŷ - y)²
```

**For all examples** (cost function):
```
J_MSE(w) = (1/m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
```

**Sometimes with 1/2 factor** (simplifies derivative):
```
J_MSE(w) = (1/2m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
```

### Matrix Form

```
J_MSE(w) = (1/2m) ||Xw - y||²
         = (1/2m) (Xw - y)ᵀ(Xw - y)
```

### Why "Squared"?

**Squaring has several effects**:

1. **Makes errors positive**: (-3)² = 9, same as (3)²
2. **Penalizes large errors more**: 
   - Error 1 → Loss 1
   - Error 2 → Loss 4
   - Error 3 → Loss 9
3. **Smooth and differentiable**: No corners or kinks
4. **Convex**: Guarantees single global minimum

### Visual Representation

```
    Loss
    │
  9 │                   ●  (error=3, loss=9)
    │
  4 │         ●            (error=2, loss=4)
    │
  1 │   ●                  (error=1, loss=1)
    │
  0 │●                     (error=0, loss=0)
    │
    └─────────────────────────── Prediction Error
    0     1     2     3
    
    Parabolic shape - penalizes outliers heavily
```

### Gradient Derivation

For linear regression: ŷ = wᵀx

**Step 1**: Loss for one example
```
L(ŷ, y) = (ŷ - y)²
```

**Step 2**: Derivative with respect to prediction
```
∂L/∂ŷ = 2(ŷ - y)
```

**Step 3**: Chain rule for weights
```
∂L/∂wⱼ = ∂L/∂ŷ · ∂ŷ/∂wⱼ
       = 2(ŷ - y) · xⱼ
```

**Step 4**: Average over all examples
```
∂J/∂wⱼ = (1/m) Σᵢ₌₁ᵐ 2(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) · x⁽ⁱ⁾ⱼ
```

With 1/2 factor:
```
∂J/∂wⱼ = (1/m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) · x⁽ⁱ⁾ⱼ
```

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error loss
    
    Args:
        y_true: Actual values (m,)
        y_pred: Predicted values (m,)
    
    Returns:
        MSE loss (scalar)
    """
    return np.mean((y_pred - y_true) ** 2)

def mse_gradient(X, y_true, y_pred):
    """
    Gradient of MSE with respect to weights
    
    Args:
        X: Feature matrix (m × n)
        y_true: Actual values (m,)
        y_pred: Predicted values (m,)
    
    Returns:
        Gradient vector (n,)
    """
    m = len(y_true)
    return (1/m) * X.T @ (y_pred - y_true)

# Example
y_true = np.array([2, 4, 5, 4, 5])
y_pred = np.array([2.5, 3.8, 5.2, 3.9, 5.1])

mse = mse_loss(y_true, y_pred)
print(f"MSE: {mse:.4f}")

# Visualize loss landscape for single parameter
errors = np.linspace(-3, 3, 100)
losses = errors ** 2

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(errors, losses, linewidth=2)
plt.xlabel('Prediction Error (ŷ - y)')
plt.ylabel('Loss')
plt.title('MSE Loss Function')
plt.grid(True)
plt.axvline(x=0, color='r', linestyle='--', label='Perfect prediction')
plt.legend()

# Show individual errors
plt.subplot(1, 2, 2)
actual_errors = y_pred - y_true
plt.bar(range(len(actual_errors)), actual_errors ** 2)
plt.xlabel('Example')
plt.ylabel('Squared Error')
plt.title('Individual Squared Errors')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Properties of MSE

**Advantages**:
✅ **Smooth and differentiable**: Easy to optimize
✅ **Convex**: One global minimum
✅ **Closed-form solution**: Normal equation exists
✅ **Penalizes large errors**: Good for most cases
✅ **Statistical interpretation**: Maximum likelihood under Gaussian noise

**Disadvantages**:
❌ **Sensitive to outliers**: Large errors dominate
❌ **Different units**: Squared units (hard to interpret)
❌ **Not robust**: Few outliers can ruin the model

### Example: Outlier Sensitivity

```python
# Data without outliers
y_true = np.array([2, 4, 5, 4, 5])
y_pred = np.array([2.1, 4.1, 4.9, 4.2, 4.8])
mse_normal = mse_loss(y_true, y_pred)

# Data with one outlier
y_true_outlier = np.array([2, 4, 5, 4, 100])  # Outlier!
y_pred_outlier = np.array([2.1, 4.1, 4.9, 4.2, 4.8])
mse_outlier = mse_loss(y_true_outlier, y_pred_outlier)

print(f"MSE without outlier: {mse_normal:.4f}")
print(f"MSE with outlier:    {mse_outlier:.4f}")
print(f"Increase factor:     {mse_outlier/mse_normal:.1f}x")
```

**Output**:
```
MSE without outlier: 0.0220
MSE with outlier:    1820.0220
Increase factor:     82731.8x
```

One outlier can **dominate** the entire loss!

---

## Mean Absolute Error (MAE)

### Definition

**Mean Absolute Error** is the average of absolute differences between predictions and actual values.

**For single example**:
```
L_MAE(ŷ, y) = |ŷ - y|
```

**For all examples**:
```
J_MAE(w) = (1/m) Σᵢ₌₁ᵐ |ŷ⁽ⁱ⁾ - y⁽ⁱ⁾|
```

### Visual Representation

```
    Loss
    │
  3 │             ╱●  (error=3, loss=3)
    │           ╱
  2 │         ╱●      (error=2, loss=2)
    │       ╱
  1 │     ╱●          (error=1, loss=1)
    │   ╱
  0 │ ●               (error=0, loss=0)
    │╱
    └─────────────────────────── Prediction Error
    0     1     2     3
    
    V-shape - linear penalty
```

### Comparison with MSE

```
Error  | MSE Loss | MAE Loss | MSE/MAE Ratio
-------|----------|----------|---------------
  0.1  |   0.01   |   0.1    |    0.1
  0.5  |   0.25   |   0.5    |    0.5
  1.0  |   1.00   |   1.0    |    1.0
  2.0  |   4.00   |   2.0    |    2.0
  5.0  |  25.00   |   5.0    |    5.0
 10.0  | 100.00   |  10.0    |   10.0
```

**Key observation**: MSE grows **quadratically**, MAE grows **linearly**.

### Gradient

**Problem**: Absolute value is **not differentiable at 0**!

```
∂|x|/∂x = { +1  if x > 0
          { -1  if x < 0
          { undefined if x = 0
```

**In practice**: We use the **sign function**:

```
∂J_MAE/∂wⱼ = (1/m) Σᵢ₌₁ᵐ sign(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) · x⁽ⁱ⁾ⱼ
```

Where:
```
sign(x) = { +1  if x > 0
          {  0  if x = 0
          { -1  if x < 0
```

### Python Implementation

```python
def mae_loss(y_true, y_pred):
    """Mean Absolute Error loss"""
    return np.mean(np.abs(y_pred - y_true))

def mae_gradient(X, y_true, y_pred):
    """Gradient of MAE with respect to weights"""
    m = len(y_true)
    return (1/m) * X.T @ np.sign(y_pred - y_true)

# Example
y_true = np.array([2, 4, 5, 4, 5])
y_pred = np.array([2.5, 3.8, 5.2, 3.9, 5.1])

mae = mae_loss(y_true, y_pred)
mse = mse_loss(y_true, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

# Visualize both
errors = np.linspace(-3, 3, 100)
mse_losses = errors ** 2
mae_losses = np.abs(errors)

plt.figure(figsize=(10, 5))
plt.plot(errors, mse_losses, label='MSE', linewidth=2)
plt.plot(errors, mae_losses, label='MAE', linewidth=2)
plt.xlabel('Prediction Error (ŷ - y)')
plt.ylabel('Loss')
plt.title('MSE vs MAE')
plt.legend()
plt.grid(True)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
plt.show()
```

### Properties of MAE

**Advantages**:
✅ **Robust to outliers**: Linear penalty
✅ **Interpretable**: Same units as target variable
✅ **Median estimate**: Minimizer is the median
✅ **Balanced**: All errors weighted equally

**Disadvantages**:
❌ **Not differentiable at zero**: Optimization challenges
❌ **No closed-form solution**: Must use iterative methods
❌ **Slower convergence**: Constant gradient magnitude
❌ **Not smooth**: Can't use some optimization methods

### Example: Outlier Robustness

```python
# Same data as before
y_true = np.array([2, 4, 5, 4, 5])
y_pred = np.array([2.1, 4.1, 4.9, 4.2, 4.8])
mae_normal = mae_loss(y_true, y_pred)

# With outlier
y_true_outlier = np.array([2, 4, 5, 4, 100])
y_pred_outlier = np.array([2.1, 4.1, 4.9, 4.2, 4.8])
mae_outlier = mae_loss(y_true_outlier, y_pred_outlier)

print(f"MAE without outlier: {mae_normal:.4f}")
print(f"MAE with outlier:    {mae_outlier:.4f}")
print(f"Increase factor:     {mae_outlier/mae_normal:.1f}x")
```

**Output**:
```
MAE without outlier: 0.1400
MAE with outlier:    19.1400
Increase factor:     136.7x
```

MAE increases much less than MSE (137x vs 82732x)!

### Training with MAE

```python
def train_with_mae(X, y, alpha=0.01, num_iterations=1000):
    """Train linear regression with MAE loss"""
    m, n = X.shape
    w = np.zeros((n, 1))
    losses = []
    
    for i in range(num_iterations):
        # Predictions
        y_pred = X @ w
        
        # MAE gradient
        gradient = (1/m) * X.T @ np.sign(y_pred - y)
        
        # Update
        w = w - alpha * gradient.reshape(-1, 1)
        
        # Record loss
        loss = np.mean(np.abs(y_pred - y))
        losses.append(loss)
    
    return w, losses

# Example usage
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), X]

w_mae, losses_mae = train_with_mae(X_b, y, alpha=0.01, num_iterations=1000)

print(f"Trained weights (MAE):")
print(f"w0: {w_mae[0][0]:.4f}")
print(f"w1: {w_mae[1][0]:.4f}")
```

---

## Root Mean Squared Error (RMSE)

### Definition

**Root Mean Squared Error** is the square root of MSE.

```
RMSE = √MSE = √[(1/m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²]
```

### Why Take the Square Root?

**Problem with MSE**: Units are squared (e.g., dollars²)
**Solution**: Take square root to get original units (dollars)

**Example**:
```
Predicting house prices in $1000s
MSE  = 4.0 (units: $1000²) ← Hard to interpret!
RMSE = 2.0 (units: $1000)  ← "Average error is $2000"
```

### Properties

**Same as MSE for optimization**:
```
min MSE  ⟺  min RMSE
```

Because square root is monotonic increasing.

**But different values**:
```
If MSE = 4.0  →  RMSE = 2.0
If MSE = 9.0  →  RMSE = 3.0
```

### When to Use RMSE

**Use RMSE for**:
- **Reporting**: More interpretable than MSE
- **Comparing models**: Same scale as target variable
- **Evaluation metrics**: Standard in many competitions

**Use MSE for**:
- **Training**: Simpler gradient computation
- **Theory**: Cleaner mathematical properties

### Python Implementation

```python
def rmse_loss(y_true, y_pred):
    """Root Mean Squared Error"""
    mse = np.mean((y_pred - y_true) ** 2)
    return np.sqrt(mse)

# Example
y_true = np.array([2, 4, 5, 4, 5])
y_pred = np.array([2.5, 3.8, 5.2, 3.9, 5.1])

mse = mse_loss(y_true, y_pred)
rmse = rmse_loss(y_true, y_pred)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
```

---

## Huber Loss

### Definition

**Huber loss** combines MSE and MAE - it's quadratic for small errors, linear for large errors.

```
L_δ(ŷ, y) = { ½(ŷ - y)²              if |ŷ - y| ≤ δ
            { δ|ŷ - y| - ½δ²         if |ŷ - y| > δ
```

Where δ is a hyperparameter (typically δ = 1.0).

### Visual Representation

```
    Loss
    │
    │          ╱  (Linear for large errors)
    │        ╱
    │      ╱
    │    ╱
    │   ╱╲  (Quadratic for small errors)
    │  ╱  ╲
    │ ╱    ╲
    │╱      ╲
    └─────────────────── Prediction Error
        -δ   0   +δ
```

### Why Huber Loss?

**Best of both worlds**:
- **Small errors**: Smooth (like MSE) → good for optimization
- **Large errors**: Linear (like MAE) → robust to outliers

### Comparison Table

```
Error  | MSE  | MAE  | Huber(δ=1)
-------|------|------|------------
  0.5  | 0.25 | 0.5  |   0.125
  1.0  | 1.00 | 1.0  |   0.5
  2.0  | 4.00 | 2.0  |   1.5
  5.0  | 25.0 | 5.0  |   4.5
 10.0  |100.0 | 10.0 |   9.5
```

For errors ≤ δ: Huber ≈ ½×MSE
For errors > δ: Huber ≈ MAE

### Python Implementation

```python
def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber loss
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        delta: Threshold for switching from quadratic to linear
    
    Returns:
        Huber loss (scalar)
    """
    error = y_pred - y_true
    abs_error = np.abs(error)
    
    # Quadratic for small errors
    quadratic = 0.5 * error ** 2
    
    # Linear for large errors
    linear = delta * abs_error - 0.5 * delta ** 2
    
    # Choose based on error magnitude
    loss = np.where(abs_error <= delta, quadratic, linear)
    
    return np.mean(loss)

def huber_gradient(X, y_true, y_pred, delta=1.0):
    """Gradient of Huber loss"""
    m = len(y_true)
    error = y_pred - y_true
    abs_error = np.abs(error)
    
    # Gradient is error for small, sign*delta for large
    grad_per_example = np.where(
        abs_error <= delta,
        error,
        delta * np.sign(error)
    )
    
    return (1/m) * X.T @ grad_per_example

# Visualize all three losses
errors = np.linspace(-5, 5, 1000)
mse_losses = errors ** 2
mae_losses = np.abs(errors)

delta = 1.0
huber_losses = np.where(
    np.abs(errors) <= delta,
    0.5 * errors ** 2,
    delta * np.abs(errors) - 0.5 * delta ** 2
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(errors, mse_losses, label='MSE', linewidth=2)
plt.plot(errors, mae_losses, label='MAE', linewidth=2)
plt.plot(errors, huber_losses, label=f'Huber (δ={delta})', linewidth=2)
plt.xlabel('Prediction Error')
plt.ylabel('Loss')
plt.title('Loss Functions Comparison')
plt.legend()
plt.grid(True)
plt.ylim([0, 10])

plt.subplot(1, 2, 2)
plt.plot(errors, mse_losses, label='MSE', linewidth=2)
plt.plot(errors, mae_losses, label='MAE', linewidth=2)
plt.plot(errors, huber_losses, label=f'Huber (δ={delta})', linewidth=2)
plt.xlabel('Prediction Error')
plt.ylabel('Loss')
plt.title('Loss Functions (Zoomed)')
plt.legend()
plt.grid(True)
plt.xlim([-2, 2])
plt.ylim([0, 4])

plt.tight_layout()
plt.show()
```

### Training with Huber Loss

```python
def train_with_huber(X, y, alpha=0.01, delta=1.0, num_iterations=1000):
    """Train linear regression with Huber loss"""
    m, n = X.shape
    w = np.zeros((n, 1))
    losses = []
    
    for i in range(num_iterations):
        y_pred = X @ w
        
        # Huber gradient
        gradient = huber_gradient(X, y, y_pred, delta).reshape(-1, 1)
        
        # Update
        w = w - alpha * gradient
        
        # Record loss
        loss = huber_loss(y, y_pred, delta)
        losses.append(loss)
    
    return w, losses
```

### Choosing δ

**Small δ** (e.g., 0.1):
- More like MAE
- Very robust to outliers
- Slower convergence

**Large δ** (e.g., 10.0):
- More like MSE
- Less robust to outliers
- Faster convergence

**Typical value**: δ = 1.0 is a good default.

---

## Comparison of Loss Functions

### Side-by-Side Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data with outlier
np.random.seed(42)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_true = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 40])  # Outlier at end!
X_b = np.c_[np.ones(X.shape[0]), X]

# Train with different losses
from sklearn.linear_model import LinearRegression, HuberRegressor

# MSE (standard linear regression)
model_mse = LinearRegression()
model_mse.fit(X, y_true)
y_pred_mse = model_mse.predict(X)

# Huber
model_huber = HuberRegressor(epsilon=1.5)
model_huber.fit(X, y_true)
y_pred_huber = model_huber.predict(X)

# MAE (manual implementation with SGD)
from sklearn.linear_model import SGDRegressor
model_mae = SGDRegressor(loss='epsilon_insensitive', epsilon=0, max_iter=10000, tol=1e-3)
model_mae.fit(X, y_true)
y_pred_mae = model_mae.predict(X)

# Plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X, y_true, color='black', s=100, label='Data (with outlier)')
plt.plot(X, y_pred_mse, 'r-', linewidth=2, label='MSE fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with MSE')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(X, y_true, color='black', s=100, label='Data (with outlier)')
plt.plot(X, y_pred_huber, 'g-', linewidth=2, label='Huber fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Huber Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(X, y_true, color='black', s=100, label='Data (with outlier)')
plt.plot(X, y_pred_mae, 'b-', linewidth=2, label='MAE fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate errors
mse_mse = np.mean((y_pred_mse - y_true) ** 2)
mae_mse = np.mean(np.abs(y_pred_mse - y_true))

mse_huber = np.mean((y_pred_huber - y_true) ** 2)
mae_huber = np.mean(np.abs(y_pred_huber - y_true))

mse_mae = np.mean((y_pred_mae - y_true) ** 2)
mae_mae = np.mean(np.abs(y_pred_mae - y_true))

print("\nPerformance Metrics:")
print("="*50)
print(f"{'Method':<15} {'MSE':<12} {'MAE':<12}")
print("-"*50)
print(f"{'MSE Loss':<15} {mse_mse:<12.2f} {mae_mse:<12.2f}")
print(f"{'Huber Loss':<15} {mse_huber:<12.2f} {mae_huber:<12.2f}")
print(f"{'MAE Loss':<15} {mse_mae:<12.2f} {mae_mae:<12.2f}")
```

### Summary Table

| Property | MSE | MAE | Huber |
|----------|-----|-----|-------|
| **Smoothness** | Smooth everywhere | Not smooth at 0 | Smooth everywhere |
| **Convexity** | Convex | Convex | Convex |
| **Robustness** | Not robust | Robust | Moderately robust |
| **Gradient** | Proportional to error | Constant magnitude | Both |
| **Optimization** | Easy | Challenging | Easy |
| **Closed-form** | Yes (normal equation) | No | No |
| **Outlier sensitivity** | High | Low | Medium |
| **Convergence** | Fast | Slow | Medium |
| **Estimator** | Mean | Median | Between mean/median |

### Loss Landscape Visualization

```python
# Visualize loss landscape for simple case
def compute_losses_2d(w0_range, w1_range, X, y):
    """Compute loss for grid of weight values"""
    mse_losses = np.zeros((len(w0_range), len(w1_range)))
    mae_losses = np.zeros((len(w0_range), len(w1_range)))
    huber_losses = np.zeros((len(w0_range), len(w1_range)))
    
    for i, w0 in enumerate(w0_range):
        for j, w1 in enumerate(w1_range):
            w = np.array([[w0], [w1]])
            y_pred = X @ w
            
            mse_losses[i, j] = np.mean((y_pred - y) ** 2)
            mae_losses[i, j] = np.mean(np.abs(y_pred - y))
            huber_losses[i, j] = huber_loss(y, y_pred, delta=1.0)
    
    return mse_losses, mae_losses, huber_losses

# Simple dataset
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([[2], [4], [5], [4], [5]])

w0_range = np.linspace(0, 5, 50)
w1_range = np.linspace(0, 2, 50)

mse_grid, mae_grid, huber_grid = compute_losses_2d(w0_range, w1_range, X, y)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im1 = axes[0].contour(w1_range, w0_range, mse_grid, levels=20)
axes[0].set_xlabel('w1')
axes[0].set_ylabel('w0')
axes[0].set_title('MSE Loss Landscape')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].contour(w1_range, w0_range, mae_grid, levels=20)
axes[1].set_xlabel('w1')
axes[1].set_ylabel('w0')
axes[1].set_title('MAE Loss Landscape')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].contour(w1_range, w0_range, huber_grid, levels=20)
axes[2].set_xlabel('w1')
axes[2].set_ylabel('w0')
axes[2].set_title('Huber Loss Landscape')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()
```

---

## Why We Use Different Losses

### 1. Data Characteristics

**When data has outliers**:
- Use **MAE** or **Huber**
- MSE will fit outliers, ruining the model for normal points

**When data is clean** (no outliers):
- Use **MSE**
- Faster convergence, smoother optimization

**Example scenario**: Predicting housing prices
- If dataset has a few luxury mansions (outliers) → MAE
- If dataset is homogeneous → MSE

### 2. Domain Requirements

**Financial applications**:
- Large errors may be catastrophic → Use MSE (penalizes large errors)
- Example: Fraud detection (false negatives are very costly)

**Forecasting applications**:
- Want balanced errors → Use MAE
- Example: Sales forecasting (over/under predictions equally bad)

**Medical applications**:
- Underestimation dangerous → Asymmetric loss
- Example: Drug dosage (underdose worse than overdose)

### 3. Statistical Interpretation

**MSE minimizer** = **Mean**:
```
argmin Σ(ŷ - yᵢ)²  →  ŷ = mean(y)
   ŷ
```

**MAE minimizer** = **Median**:
```
argmin Σ|ŷ - yᵢ|  →  ŷ = median(y)
   ŷ
```

**Why this matters**:
- Mean sensitive to outliers
- Median robust to outliers

### 4. Optimization Considerations

**MSE**:
✅ Smooth gradient
✅ Closed-form solution
✅ Fast convergence
❌ Can be dominated by outliers

**MAE**:
❌ Non-smooth at zero
❌ No closed-form solution
❌ Slower convergence
✅ Robust

**Huber**:
✅ Smooth gradient
✅ Reasonably robust
✅ Good convergence
❌ No closed-form
❌ Extra hyperparameter δ

### 5. Interpretability

**MSE**: 
- Units are squared → Hard to interpret
- "Average squared error is 25 dollars²" ← What does this mean?

**MAE**: 
- Same units as target → Easy to interpret
- "Average error is 5 dollars" ← Clear!

**RMSE**: 
- Same units as target → Easy to interpret
- Square root of MSE

### 6. Custom Losses

Sometimes standard losses aren't enough!

**Asymmetric losses** (over/under predictions different costs):
```python
def asymmetric_loss(y_true, y_pred, over_cost=1.0, under_cost=2.0):
    """Penalize under-predictions more heavily"""
    error = y_pred - y_true
    loss = np.where(
        error >= 0,  # Over-prediction
        over_cost * error ** 2,
        under_cost * error ** 2  # Under-prediction
    )
    return np.mean(loss)
```

**Quantile loss** (predict specific quantile, not mean):
```python
def quantile_loss(y_true, y_pred, quantile=0.9):
    """Loss for quantile regression"""
    error = y_true - y_pred
    loss = np.where(
        error >= 0,
        quantile * error,
        (quantile - 1) * error
    )
    return np.mean(loss)
```

### Decision Framework

```
START
  │
  ├─ Do you have outliers?
  │  ├─ Yes → Use MAE or Huber
  │  └─ No → Continue
  │
  ├─ Need smooth gradients?
  │  ├─ Yes → Use MSE or Huber
  │  └─ No → MAE is fine
  │
  ├─ Large errors catastrophic?
  │  ├─ Yes → Use MSE (penalizes large errors)
  │  └─ No → Use MAE (balanced)
  │
  ├─ Need interpretability?
  │  ├─ Yes → Use MAE or RMSE
  │  └─ No → MSE is fine
  │
  └─ Default: Start with MSE, switch if needed
```

---

## Probabilistic Interpretation

### Maximum Likelihood Estimation

Loss functions can be derived from **probability distributions**!

**General principle**: 
```
Minimize loss = Maximize likelihood
```

### MSE and Gaussian Distribution

**Assumption**: Errors are **normally distributed**

```
y = wᵀx + ε
ε ~ N(0, σ²)
```

This means:
```
p(y | x, w) = (1/√(2πσ²)) exp(-(y - wᵀx)² / (2σ²))
```

**Log-likelihood** for all examples:
```
log L(w) = Σᵢ log p(y⁽ⁱ⁾ | x⁽ⁱ⁾, w)
         = Σᵢ [constant - (y⁽ⁱ⁾ - wᵀx⁽ⁱ⁾)² / (2σ²)]
```

**Maximizing log-likelihood** = **Minimizing squared error**!

```
max log L(w)  ⟺  min Σ(y - wᵀx)²
```

**Conclusion**: If errors are Gaussian → MSE is optimal!

### MAE and Laplace Distribution

**Assumption**: Errors follow **Laplace distribution** (double exponential)

```
p(ε) = (1/2b) exp(-|ε|/b)
```

**Log-likelihood**:
```
log L(w) = Σᵢ [constant - |y⁽ⁱ⁾ - wᵀx⁽ⁱ⁾|/b]
```

**Maximizing log-likelihood** = **Minimizing absolute error**!

```
max log L(w)  ⟺  min Σ|y - wᵀx|
```

**Conclusion**: If errors follow Laplace → MAE is optimal!

### Visual Comparison

```
    Probability Density
    │
    │   Gaussian (MSE)
    │      ╱╲
    │    ╱  ╲
    │  ╱      ╲___
    │╱            ╲___
    │
    │   Laplace (MAE)
    │       ╱╲
    │      ╱  ╲
    │     ╱    ╲
    │    ╱      ╲
    │   ╱        ╲
    │  ╱          ╲
    │ ╱            ╲
    │╱              ╲
    └──────────────────── Error
         0
    
    Laplace has heavier tails → more outliers expected
```

### Choosing Based on Data Distribution

**If you know the error distribution**:
- Gaussian errors → MSE
- Laplace errors → MAE
- Heavy-tailed errors → Huber or robust methods

**In practice**: Often don't know distribution
- Try both MSE and MAE
- Use cross-validation to choose
- Look at residual distribution

---

## Loss Functions for Classification

### Preview: Binary Classification Losses

While this section focuses on regression, here's a quick preview:

**0-1 Loss** (counting mistakes):
```
L(ŷ, y) = { 0  if ŷ = y
          { 1  if ŷ ≠ y
```
Problem: Not differentiable!

**Logistic Loss** (cross-entropy):
```
L(ŷ, y) = -[y log(ŷ) + (1-y) log(1-ŷ)]
```
Where ŷ = σ(wᵀx) is probability.

**Hinge Loss** (SVM):
```
L(ŷ, y) = max(0, 1 - y·ŷ)
```

More details in **Logistic_Regression.md**!

---

## Choosing the Right Loss

### Practical Guidelines

**1. Start with MSE**:
- Standard choice
- Works well in most cases
- Easy to optimize

**2. Switch to MAE if**:
- You have outliers
- Want robust predictions
- Don't mind slower convergence

**3. Use Huber if**:
- Have some outliers but not many
- Want smooth optimization
- Need balance between MSE and MAE

**4. Use custom loss if**:
- Domain-specific requirements
- Asymmetric costs
- Special constraints

### Example Decision Tree

```
Do you have outliers?
├─ No → MSE
└─ Yes
   ├─ Many outliers (>10%) → MAE
   ├─ Few outliers (<10%) → Huber
   └─ Not sure → Try both, use validation set
```

### Cross-Validation Approach

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor

# Try different losses
models = {
    'MSE': LinearRegression(),
    'Huber': HuberRegressor(),
    'MAE': QuantileRegressor(quantile=0.5, solver='highs')  # Median = MAE
}

for name, model in models.items():
    scores = cross_val_score(
        model, X, y, 
        cv=5, 
        scoring='neg_mean_squared_error'
    )
    print(f"{name}: MSE = {-scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Performance Comparison Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor

def compare_losses(X, y, X_test=None, y_test=None):
    """Compare different loss functions"""
    
    # Train models
    models = {
        'MSE': LinearRegression(),
        'Huber': HuberRegressor(epsilon=1.5),
    }
    
    results = {}
    
    for name, model in models.items():
        # Fit
        model.fit(X, y)
        
        # Predict
        y_pred_train = model.predict(X)
        
        # Metrics on training set
        train_mse = np.mean((y_pred_train - y) ** 2)
        train_mae = np.mean(np.abs(y_pred_train - y))
        train_rmse = np.sqrt(train_mse)
        
        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
        }
        
        # If test set provided
        if X_test is not None and y_test is not None:
            y_pred_test = model.predict(X_test)
            test_mse = np.mean((y_pred_test - y_test) ** 2)
            test_mae = np.mean(np.abs(y_pred_test - y_test))
            test_rmse = np.sqrt(test_mse)
            
            results[name]['test_mse'] = test_mse
            results[name]['test_mae'] = test_mae
            results[name]['test_rmse'] = test_rmse
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<15} {'Train MSE':<12} {'Train MAE':<12} {'Train RMSE':<12}")
    print("-"*60)
    
    for name, res in results.items():
        print(f"{name:<15} {res['train_mse']:<12.4f} {res['train_mae']:<12.4f} {res['train_rmse']:<12.4f}")
    
    if 'test_mse' in results['MSE']:
        print("\n" + "="*60)
        print("TEST SET PERFORMANCE")
        print("="*60)
        print(f"{'Model':<15} {'Test MSE':<12} {'Test MAE':<12} {'Test RMSE':<12}")
        print("-"*60)
        
        for name, res in results.items():
            print(f"{name:<15} {res['test_mse']:<12.4f} {res['test_mae']:<12.4f} {res['test_rmse']:<12.4f}")
    
    return results

# Example usage
np.random.seed(42)
X_train = 2 * np.random.rand(100, 1)
y_train = 4 + 3 * X_train.ravel() + np.random.randn(100)

# Add a few outliers
outlier_indices = [5, 15, 25]
y_train[outlier_indices] += 20

X_test = 2 * np.random.rand(20, 1)
y_test = 4 + 3 * X_test.ravel() + np.random.randn(20)

results = compare_losses(X_train, y_train, X_test, y_test)
```

---

## Summary and Next Steps

### What We Learned

1. **Mean Squared Error (MSE)**
   ```
   L = (ŷ - y)²
   ```
   - Standard choice, smooth, closed-form solution
   - Sensitive to outliers

2. **Mean Absolute Error (MAE)**
   ```
   L = |ŷ - y|
   ```
   - Robust to outliers, interpretable
   - Not smooth, slower convergence

3. **Root Mean Squared Error (RMSE)**
   ```
   L = √MSE
   ```
   - For reporting, same units as target

4. **Huber Loss**
   ```
   L = { ½(ŷ-y)²     if |ŷ-y| ≤ δ
       { δ|ŷ-y|-½δ²  if |ŷ-y| > δ
   ```
   - Compromise: smooth + robust

5. **Probabilistic Interpretation**
   - MSE ↔ Gaussian errors
   - MAE ↔ Laplace errors

### Key Takeaways

✅ **Different losses for different needs**  
✅ **MSE = standard, MAE = robust, Huber = balanced**  
✅ **Loss choice affects optimization and results**  
✅ **Outliers → use MAE or Huber**  
✅ **Losses derived from probability distributions**  

### Comparison Summary

| Loss | Best For | Avoid When | Gradient |
|------|----------|------------|----------|
| **MSE** | Clean data, fast convergence | Outliers present | Smooth |
| **MAE** | Outliers, robust predictions | Need speed | Non-smooth |
| **Huber** | Some outliers, balance | Extra hyperparameter OK | Smooth |
| **RMSE** | Reporting metrics | Training (use MSE) | Same as MSE |

### Prerequisites Checklist

Before moving on:
- [ ] Understand MSE formula and gradient
- [ ] Understand MAE formula and gradient
- [ ] Know when to use which loss
- [ ] Understand outlier sensitivity
- [ ] Know probabilistic interpretation
- [ ] Can implement all losses in code

### Next Steps

1. **Linear_Classification.md**
   - Decision boundaries
   - Linear separability
   - Perceptron algorithm

2. **Logistic_Regression.md**
   - Classification with linear models
   - **Cross-entropy loss** (classification equivalent of MSE)
   - Sigmoid function
   - Gradient descent for classification

3. **Regularization_L1_L2.md**
   - Adding penalties to loss
   - Ridge: MSE + L2 penalty
   - Lasso: MSE + L1 penalty
   - Preventing overfitting

### Practice Exercises

1. Implement all loss functions from scratch
2. Generate data with outliers, compare MSE vs MAE
3. Visualize loss landscapes for different losses
4. Derive gradient for Huber loss
5. Implement asymmetric loss function

### Additional Resources

**Papers**:
- Huber (1964): "Robust Estimation of a Location Parameter"
- Original papers on robust regression

**Books**:
- "Robust Statistics" by Huber & Ronchetti
- "Elements of Statistical Learning" - Chapter 2

**Online**:
- Fast.ai: Practical Deep Learning - Loss functions
- CS231n: Loss functions notes
- Distill.pub: Visualizing loss functions

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║              LOSS FUNCTIONS QUICK REFERENCE                    ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  MEAN SQUARED ERROR (MSE)                                      ║
║  ────────────────────────────                                  ║
║  L = (ŷ - y)²                                                  ║
║  ∇L = 2(ŷ - y)                                                 ║
║                                                                ║
║  ✓ Smooth, convex, closed-form                                 ║
║  ✗ Sensitive to outliers                                       ║
║  Use: Clean data, fast convergence                             ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  MEAN ABSOLUTE ERROR (MAE)                                     ║
║  ──────────────────────────                                    ║
║  L = |ŷ - y|                                                   ║
║  ∇L = sign(ŷ - y)                                              ║
║                                                                ║
║  ✓ Robust to outliers, interpretable                           ║
║  ✗ Not smooth at 0, no closed-form                             ║
║  Use: Data with outliers                                       ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  ROOT MEAN SQUARED ERROR (RMSE)                                ║
║  ───────────────────────────────                               ║
║  L = √MSE                                                      ║
║                                                                ║
║  ✓ Same units as target (interpretable)                        ║
║  Use: Reporting metrics                                        ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  HUBER LOSS                                                    ║
║  ───────────                                                   ║
║  L = { ½(ŷ-y)²       if |ŷ-y| ≤ δ                              ║
║      { δ|ŷ-y|-½δ²    if |ŷ-y| > δ                             ║
║                                                                ║
║  ✓ Smooth, moderately robust                                   ║
║  Use: Few outliers, balanced approach                          ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  PROBABILISTIC INTERPRETATION                                  ║
╠════════════════════════════════════════════════════════════════╣
║  MSE ↔ Gaussian errors:   ε ~ N(0, σ²)                         ║
║  MAE ↔ Laplace errors:    ε ~ Laplace(0, b)                    ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  DECISION GUIDE                                                ║
╠════════════════════════════════════════════════════════════════╣
║  Clean data          → MSE                                     ║
║  Many outliers (>10%) → MAE                                    ║
║  Few outliers (<10%)  → Huber                                  ║
║  Need speed          → MSE                                     ║
║  Need robustness     → MAE                                     ║
║  Balanced            → Huber                                   ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  KEY PROPERTIES                                                ║
╠════════════════════════════════════════════════════════════════╣
║  MSE Estimator: Mean                                           ║
║  MAE Estimator: Median                                         ║
║                                                                ║
║  MSE: Penalizes large errors heavily (quadratic)               ║
║  MAE: Treats all errors equally (linear)                       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Ready to move on?** Next: Learn about **linear classification** and decision boundaries!

---

*End of Cost_Functions_and_Loss.md*