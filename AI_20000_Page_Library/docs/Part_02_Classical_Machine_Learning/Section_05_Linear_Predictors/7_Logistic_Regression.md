# Logistic Regression

## Table of Contents
1. [Introduction to Logistic Regression](#introduction-to-logistic-regression)
2. [The Sigmoid Function](#the-sigmoid-function)
3. [Probability Interpretation](#probability-interpretation)
4. [Binary Classification](#binary-classification)
5. [Decision Boundary](#decision-boundary)
6. [Cost Function - Cross-Entropy](#cost-function-cross-entropy)
7. [Gradient Descent for Logistic Regression](#gradient-descent-for-logistic-regression)
8. [Model Training and Evaluation](#model-training-and-evaluation)
9. [Regularization in Logistic Regression](#regularization-in-logistic-regression)
10. [Practical Implementation](#practical-implementation)
11. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to Logistic Regression

### What is Logistic Regression?

Despite its name, **logistic regression** is a **classification** algorithm, not regression!

**Purpose**: Predict the **probability** that an example belongs to a particular class.

```
Input: Feature vector x
Output: Probability P(y=1|x) ∈ [0, 1]
```

**Key difference from perceptron**:
- Perceptron: Hard predictions (0 or 1)
- Logistic Regression: Soft predictions (probability between 0 and 1)

### Why "Logistic"?

The name comes from the **logistic function** (sigmoid function) used to transform linear predictions into probabilities.

### Comparison with Linear Methods

| Method | Output | Use Case |
|--------|--------|----------|
| **Linear Regression** | Continuous value | Predict house prices |
| **Perceptron** | Class label {-1, +1} | Hard classification |
| **Logistic Regression** | Probability [0, 1] | Probabilistic classification |

### Real-World Applications

**Medical Diagnosis**:
```
P(Disease = Yes | Age, Blood Pressure, ...) = 0.73
→ 73% chance of disease
```

**Email Spam Detection**:
```
P(Spam = Yes | Word counts, Links, ...) = 0.92
→ 92% chance of spam
```

**Credit Approval**:
```
P(Default = Yes | Income, Credit Score, ...) = 0.15
→ 15% chance of default
```

**Advantage**: Provides **confidence** in predictions, not just labels.

---

## The Sigmoid Function

### Definition

The **sigmoid function** (also called **logistic function**) maps any real number to the range (0, 1):

```
σ(z) = 1 / (1 + e^(-z))
```

Where:
- z = input (can be any real number)
- σ(z) = output (always between 0 and 1)
- e = Euler's number ≈ 2.71828

### Properties

**Key properties**:
1. **Range**: σ(z) ∈ (0, 1) for all z
2. **Smooth**: Continuously differentiable
3. **S-shaped curve**: Smooth transition from 0 to 1
4. **Symmetric**: σ(-z) = 1 - σ(z)
5. **Middle point**: σ(0) = 0.5

### Mathematical Properties

**Limits**:
```
lim     σ(z) = 1    (approaches 1 as z → ∞)
z → ∞

lim     σ(z) = 0    (approaches 0 as z → -∞)
z → -∞

σ(0) = 0.5          (exactly 0.5 at zero)
```

**Symmetry**:
```
σ(-z) = 1 / (1 + e^z)
      = e^(-z) / (e^(-z)(1 + e^z))
      = 1 / (1 + e^(-(-z)))
      = 1 - σ(z)
```

### Visualization

```
    σ(z)
    │
  1 │         ___________
    │       ╱
0.5 │─────●────────────  (σ(0) = 0.5)
    │   ╱
  0 │_╱_________________
    │
    └──────────────────── z
   -10  -5  0  5  10

   S-shaped curve
```

### Python Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))

# Create range of z values
z = np.linspace(-10, 10, 1000)
sigma = sigmoid(z)

# Plot
plt.figure(figsize=(12, 5))

# Subplot 1: Sigmoid function
plt.subplot(1, 2, 1)
plt.plot(z, sigma, linewidth=3, color='blue')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='σ(0) = 0.5')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.xlabel('z', fontsize=14)
plt.ylabel('σ(z)', fontsize=14)
plt.title('Sigmoid Function', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.ylim([-0.1, 1.1])

# Subplot 2: Key values
plt.subplot(1, 2, 2)
z_points = np.array([-5, -2, 0, 2, 5])
sigma_points = sigmoid(z_points)

plt.plot(z, sigma, linewidth=2, color='blue', alpha=0.3)
plt.scatter(z_points, sigma_points, s=200, c='red', zorder=5)

for zp, sp in zip(z_points, sigma_points):
    plt.annotate(f'σ({zp}) = {sp:.3f}', 
                xy=(zp, sp), 
                xytext=(zp + 0.5, sp + 0.1),
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.xlabel('z', fontsize=14)
plt.ylabel('σ(z)', fontsize=14)
plt.title('Key Values of Sigmoid', fontsize=16)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print values
print("Key values:")
for z_val in [-10, -5, -2, -1, 0, 1, 2, 5, 10]:
    print(f"σ({z_val:3}) = {sigmoid(z_val):.6f}")
```

**Output**:
```
Key values:
σ(-10) = 0.000045
σ( -5) = 0.006693
σ( -2) = 0.119203
σ( -1) = 0.268941
σ(  0) = 0.500000
σ(  1) = 0.731059
σ(  2) = 0.880797
σ(  5) = 0.993307
σ( 10) = 0.999955
```

### Derivative of Sigmoid

**Important for gradient descent**!

```
dσ/dz = σ(z) · (1 - σ(z))
```

**Derivation**:
```
σ(z) = 1 / (1 + e^(-z))

Let u = 1 + e^(-z)
Then σ(z) = u^(-1)

dσ/dz = d/dz[u^(-1)]
      = -u^(-2) · du/dz
      = -1/(1 + e^(-z))² · e^(-z) · (-1)
      = e^(-z) / (1 + e^(-z))²
      
Now, multiply numerator and denominator by e^z:
      = 1 / ((1 + e^(-z))(e^z + 1))
      = 1 / (1 + e^(-z)) · e^z / (e^z + 1)
      = σ(z) · (1 - σ(z))
```

**Why this is beautiful**:
```
dσ/dz = σ(z) · (1 - σ(z))
```

We can compute the derivative using only the function value itself!

**Python**:
```python
def sigmoid_derivative(z):
    """Derivative of sigmoid"""
    s = sigmoid(z)
    return s * (1 - s)

# Plot derivative
z = np.linspace(-10, 10, 1000)
sigma = sigmoid(z)
d_sigma = sigmoid_derivative(z)

plt.figure(figsize=(10, 6))
plt.plot(z, sigma, linewidth=2, label='σ(z)')
plt.plot(z, d_sigma, linewidth=2, label="σ'(z)")
plt.xlabel('z', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title('Sigmoid and its Derivative', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

**Key observation**: Derivative is maximum at z = 0:
```
σ'(0) = σ(0) · (1 - σ(0)) = 0.5 · 0.5 = 0.25
```

---

## Probability Interpretation

### The Logistic Regression Model

**Step 1**: Compute linear combination (like linear regression):
```
z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = wᵀx
```

**Step 2**: Apply sigmoid to get probability:
```
P(y=1|x) = σ(wᵀx) = 1 / (1 + e^(-wᵀx))
```

**Complete model**:
```
ŷ = P(y=1|x; w) = σ(wᵀx)
```

### Why Sigmoid?

**Problem**: wᵀx can be any value (-∞, +∞)
**Need**: Probability must be in [0, 1]
**Solution**: Apply sigmoid!

```
wᵀx ∈ (-∞, +∞)  →  σ(wᵀx) ∈ (0, 1)
```

**Visual transformation**:
```
    wᵀx                      P(y=1|x)
    
  +∞ │                     1 │         ________
     │                       │       ╱
   5 │                   0.99│     ╱
     │                       │   ╱
   0 │         →          0.5│  ●
     │                       │ ╱
  -5 │                   0.01│╱
     │                       │
  -∞ │                     0 │
     │                       │
     
  Linear score            Probability
```

### Interpretation as Odds

The **odds** of an event is:
```
Odds = P(event) / P(not event) = P(y=1) / P(y=0)
```

For logistic regression:
```
Odds = P(y=1|x) / P(y=0|x)
     = σ(wᵀx) / (1 - σ(wᵀx))
     = e^(wᵀx)
```

Taking logarithm:
```
log(Odds) = wᵀx
```

This is called the **log-odds** or **logit**!

**Key insight**: Logistic regression models **log-odds** as a linear function.

### Example: Medical Diagnosis

**Features**: Age, Blood Pressure, Cholesterol
**Model**: P(Disease | Age, BP, Cholesterol)

Suppose:
```
w = [−5, 0.05, 0.02, 0.01]
```

**Patient 1**: Age=50, BP=120, Cholesterol=200
```
z = -5 + 0.05(50) + 0.02(120) + 0.01(200)
  = -5 + 2.5 + 2.4 + 2
  = 1.9

P(Disease|x) = σ(1.9) = 1/(1 + e^(-1.9)) = 0.870
→ 87% probability of disease
```

**Patient 2**: Age=30, BP=110, Cholesterol=150
```
z = -5 + 0.05(30) + 0.02(110) + 0.01(150)
  = -5 + 1.5 + 2.2 + 1.5
  = 0.2

P(Disease|x) = σ(0.2) = 0.550
→ 55% probability of disease
```

---

## Binary Classification

### From Probability to Prediction

**Predicted probability**: P(y=1|x)

**Classification rule** (threshold τ, typically 0.5):
```
ŷ = { 1  if P(y=1|x) ≥ τ
    { 0  if P(y=1|x) < τ
```

**Standard threshold**: τ = 0.5
```
ŷ = { 1  if σ(wᵀx) ≥ 0.5
    { 0  if σ(wᵀx) < 0.5
```

Since σ(0) = 0.5, this is equivalent to:
```
ŷ = { 1  if wᵀx ≥ 0
    { 0  if wᵀx < 0
```

**Same as linear classification**, but with probability estimates!

### Choosing Different Thresholds

Sometimes we want different thresholds:

**Conservative** (τ = 0.8): Only predict positive if very confident
```
Use case: Medical diagnosis (avoid false alarms)
```

**Aggressive** (τ = 0.2): Predict positive with lower confidence
```
Use case: Fraud detection (don't miss any fraud)
```

### Classification Metrics

For binary classification with threshold τ:

**True Positives (TP)**: Correctly predicted 1
**True Negatives (TN)**: Correctly predicted 0
**False Positives (FP)**: Wrongly predicted 1 (Type I error)
**False Negatives (FN)**: Wrongly predicted 0 (Type II error)

**Confusion Matrix**:
```
                Predicted
                0       1
Actual  0   │  TN  │  FP  │
        1   │  FN  │  TP  │
```

**Metrics**:
```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)  (of predicted positive, how many correct?)
Recall    = TP / (TP + FN)  (of actual positive, how many found?)
F1 Score  = 2 · Precision · Recall / (Precision + Recall)
```

### Python Example

```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Predicted probabilities
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
y_proba = np.array([0.1, 0.3, 0.8, 0.6, 0.9, 0.2, 0.7, 0.4, 0.55, 0.15])

# Different thresholds
thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    
    print(f"\nThreshold: {threshold}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
```

---

## Decision Boundary

### Mathematical Definition

The **decision boundary** is where the classifier switches from one class to another.

For logistic regression with threshold 0.5:
```
Decision boundary: P(y=1|x) = 0.5
                  σ(wᵀx) = 0.5
                  wᵀx = 0
```

**Same as linear classification**: A hyperplane!

### In 2D: A Line

For two features x₁, x₂:
```
w₀ + w₁x₁ + w₂x₂ = 0

Solving for x₂:
x₂ = -(w₀ + w₁x₁) / w₂
```

**Example**:
```
w = [-3, 1, 2]

Boundary: -3 + x₁ + 2x₂ = 0
          x₂ = (3 - x₁) / 2
```

### Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)

# Class 0
X0 = np.random.randn(50, 2) * 0.8 + [2, 2]
y0 = np.zeros(50)

# Class 1
X1 = np.random.randn(50, 2) * 0.8 + [5, 5]
y1 = np.ones(50)

# Combine
X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# Train logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
X_with_bias = X  # sklearn adds bias automatically
model.fit(X, y)

# Get weights
w0 = model.intercept_[0]
w1, w2 = model.coef_[0]

print(f"Weights: w₀={w0:.3f}, w₁={w1:.3f}, w₂={w2:.3f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Data and decision boundary
ax = axes[0]
ax.scatter(X0[:, 0], X0[:, 1], c='blue', marker='o', s=100, alpha=0.7, label='Class 0')
ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='s', s=100, alpha=0.7, label='Class 1')

# Decision boundary
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_boundary = np.linspace(x1_min, x1_max, 100)
x2_boundary = -(w0 + w1 * x1_boundary) / w2
ax.plot(x1_boundary, x2_boundary, 'g-', linewidth=3, label='Decision Boundary')

ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_title('Logistic Regression Decision Boundary', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Probability contours
ax = axes[1]
x1_grid = np.linspace(x1_min, x1_max, 200)
x2_grid = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)

# Compute probabilities
Z = model.predict_proba(np.c_[X1_grid.ravel(), X2_grid.ravel()])[:, 1]
Z = Z.reshape(X1_grid.shape)

# Plot probability surface
contour = ax.contourf(X1_grid, X2_grid, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
ax.scatter(X0[:, 0], X0[:, 1], c='blue', marker='o', s=100, edgecolors='k', alpha=0.9)
ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='s', s=100, edgecolors='k', alpha=0.9)

# Decision boundary (P=0.5)
ax.contour(X1_grid, X2_grid, Z, levels=[0.5], colors='black', linewidths=3)

ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_title('Probability P(y=1|x)', fontsize=14)
plt.colorbar(contour, ax=ax, label='P(y=1|x)')

# Plot 3: Classification regions
ax = axes[2]
predictions = model.predict(np.c_[X1_grid.ravel(), X2_grid.ravel()])
predictions = predictions.reshape(X1_grid.shape)

ax.contourf(X1_grid, X2_grid, predictions, levels=[-0.5, 0.5, 1.5], 
            colors=['lightblue', 'lightcoral'], alpha=0.5)
ax.scatter(X0[:, 0], X0[:, 1], c='blue', marker='o', s=100, edgecolors='k', alpha=0.9, label='Class 0')
ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='s', s=100, edgecolors='k', alpha=0.9, label='Class 1')
ax.plot(x1_boundary, x2_boundary, 'g-', linewidth=3, label='Boundary')

ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_title('Classification Regions', fontsize=14)
ax.legend()

plt.tight_layout()
plt.show()
```

### Distance from Boundary

The **signed distance** from a point to the decision boundary is proportional to the **confidence**:

```
distance ∝ |wᵀx| / ||w||
```

**Large |wᵀx|**: Far from boundary → high confidence
**Small |wᵀx|**: Near boundary → low confidence

**Probability interpretation**:
```
|wᵀx| large  →  P(y=1|x) ≈ 0 or ≈ 1  (confident)
|wᵀx| small  →  P(y=1|x) ≈ 0.5       (uncertain)
```

---

## Cost Function - Cross-Entropy

### Why Not MSE?

For logistic regression, MSE is **non-convex** (multiple local minima)!

```
J(w) = (1/m) Σ(σ(wᵀx⁽ⁱ⁾) - y⁽ⁱ⁾)²  ← Non-convex!
```

**Problem**: Gradient descent may get stuck in local minimum.

### Cross-Entropy Loss

**For single example**:
```
L(ŷ, y) = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

Where:
- ŷ = σ(wᵀx) = predicted probability
- y ∈ {0, 1} = true label

**For all examples** (cost function):
```
J(w) = -(1/m) Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-ŷ⁽ⁱ⁾)]
```

### Intuition

**Case 1**: y = 1 (actual class is 1)
```
L = -log(ŷ)

If ŷ → 1: L → 0      (Good! Predicting correct class)
If ŷ → 0: L → ∞      (Bad! Predicting wrong class)
```

**Case 2**: y = 0 (actual class is 0)
```
L = -log(1-ŷ)

If ŷ → 0: L → 0      (Good! Predicting correct class)
If ŷ → 1: L → ∞      (Bad! Predicting wrong class)
```

### Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Predicted probabilities
y_pred = np.linspace(0.001, 0.999, 1000)

# Loss when y=1
loss_y1 = -np.log(y_pred)

# Loss when y=0
loss_y0 = -np.log(1 - y_pred)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(y_pred, loss_y1, linewidth=3, color='blue')
plt.xlabel('Predicted Probability ŷ', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Cross-Entropy Loss when y=1', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(y_pred, loss_y0, linewidth=3, color='red')
plt.xlabel('Predicted Probability ŷ', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Cross-Entropy Loss when y=0', fontsize=16)
plt.grid(True, alpha=0.3)
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

### Properties

**Convex**: Has single global minimum
**Smooth**: Continuously differentiable
**Penalizes confidence**: Wrong confident predictions have high loss
**Probabilistic**: Derived from maximum likelihood

---

## Gradient Descent for Logistic Regression

### Gradient Derivation

**Cost function**:
```
J(w) = -(1/m) Σᵢ [y⁽ⁱ⁾ log(σ(wᵀx⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1-σ(wᵀx⁽ⁱ⁾))]
```

**Gradient with respect to wⱼ**:

Step 1: Derivative of loss for one example:
```
∂L/∂wⱼ = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂wⱼ
```

Where:
- z = wᵀx
- ŷ = σ(z)
- L = -[y log(ŷ) + (1-y) log(1-ŷ)]

Step 2: Compute each part:
```
∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)

∂ŷ/∂z = σ(z)(1-σ(z))

∂z/∂wⱼ = xⱼ
```

Step 3: Chain rule:
```
∂L/∂wⱼ = [-y/σ(z) + (1-y)/(1-σ(z))] · σ(z)(1-σ(z)) · xⱼ
       = [-y(1-σ(z)) + (1-y)σ(z)] · xⱼ
       = [-y + yσ(z) + σ(z) - yσ(z)] · xⱼ
       = (σ(z) - y) · xⱼ
       = (ŷ - y) · xⱼ
```

**Beautiful result**: Same form as linear regression!
```
∂J/∂wⱼ = (1/m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) · x⁽ⁱ⁾ⱼ
```

**Vector form**:
```
∇J(w) = (1/m) Xᵀ(ŷ - y)
```

Where ŷ = σ(Xw).

### Update Rule

```
w ← w - α · ∇J(w)
  = w - α · (1/m) Xᵀ(σ(Xw) - y)
```

### Algorithm

```
LOGISTIC REGRESSION GRADIENT DESCENT
─────────────────────────────────────

Input: X (m × n), y (m × 1), learning rate α, iterations T
Initialize: w = 0

For t = 1 to T:
    # Compute predictions (probabilities)
    z = Xw
    ŷ = σ(z) = 1 / (1 + e^(-z))
    
    # Compute gradient
    ∇J = (1/m) Xᵀ(ŷ - y)
    
    # Update weights
    w ← w - α·∇J
    
    # Optional: compute and store cost
    J = -(1/m) Σ[y log(ŷ) + (1-y) log(1-ŷ)]

Output: w
```

### Python Implementation

```python
import numpy as np

def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y_true, y_pred):
    """Cross-entropy loss"""
    m = len(y_true)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def logistic_regression_gd(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Train logistic regression using gradient descent
    
    Args:
        X: Feature matrix (m × n) with bias column
        y: Labels (m × 1) in {0, 1}
        learning_rate: Step size
        num_iterations: Number of iterations
    
    Returns:
        w: Learned weights
        costs: Cost at each iteration
    """
    m, n = X.shape
    w = np.zeros((n, 1))
    costs = []
    
    for i in range(num_iterations):
        # Forward pass
        z = X @ w
        y_pred = sigmoid(z)
        
        # Compute cost
        cost = cross_entropy_loss(y, y_pred)
        costs.append(cost)
        
        # Compute gradient
        gradient = (1/m) * X.T @ (y_pred - y)
        
        # Update weights
        w = w - learning_rate * gradient
        
        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return w, costs

# Example usage
np.random.seed(42)

# Generate data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)

# Add bias column
X_b = np.c_[np.ones(X.shape[0]), X]
y = y.reshape(-1, 1)

# Train
w, costs = logistic_regression_gd(X_b, y, learning_rate=0.1, num_iterations=1000)

print(f"\nFinal weights: {w.ravel()}")

# Plot convergence
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(costs, linewidth=2)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Cost (Cross-Entropy)', fontsize=14)
plt.title('Logistic Regression Training', fontsize=16)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Model Training and Evaluation

### Complete Training Pipeline

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Generate data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, 
                           flip_y=0.1, random_state=42)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# 6. Evaluate
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# 7. Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Decision boundary
ax = axes[0, 0]
x1_min, x1_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
x2_min, x2_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                        np.linspace(x2_min, x2_max, 200))
Z = model.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])[:, 1]
Z = Z.reshape(xx1.shape)

ax.contourf(xx1, xx2, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
ax.scatter(X_train_scaled[y_train==0, 0], X_train_scaled[y_train==0, 1], 
          c='blue', marker='o', s=50, edgecolors='k', label='Class 0')
ax.scatter(X_train_scaled[y_train==1, 0], X_train_scaled[y_train==1, 1], 
          c='red', marker='s', s=50, edgecolors='k', label='Class 1')
ax.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=3)
ax.set_title('Training Data with Decision Boundary', fontsize=14)
ax.legend()

# Plot 2: Test set predictions
ax = axes[0, 1]
ax.scatter(X_test_scaled[y_test==0, 0], X_test_scaled[y_test==0, 1],
          c='blue', marker='o', s=100, edgecolors='k', label='True Class 0')
ax.scatter(X_test_scaled[y_test==1, 0], X_test_scaled[y_test==1, 1],
          c='red', marker='s', s=100, edgecolors='k', label='True Class 1')

# Mark misclassifications
misclassified = y_test != y_test_pred
ax.scatter(X_test_scaled[misclassified, 0], X_test_scaled[misclassified, 1],
          s=300, facecolors='none', edgecolors='yellow', linewidths=3, label='Misclassified')

ax.set_title('Test Set Predictions', fontsize=14)
ax.legend()

# Plot 3: Probability histogram
ax = axes[1, 0]
ax.hist(y_test_proba[y_test==0], bins=20, alpha=0.7, label='Class 0', color='blue')
ax.hist(y_test_proba[y_test==1], bins=20, alpha=0.7, label='Class 1', color='red')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Predicted Probability P(y=1|x)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Predicted Probabilities', fontsize=14)
ax.legend()

# Plot 4: ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

ax = axes[1, 1]
ax.plot(fpr, tpr, linewidth=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Making Predictions

```python
# Predict probabilities
def predict_proba(X, w):
    """Predict probabilities"""
    z = X @ w
    return sigmoid(z)

# Predict class labels
def predict(X, w, threshold=0.5):
    """Predict class labels"""
    probas = predict_proba(X, w)
    return (probas >= threshold).astype(int)

# Example
X_new = np.array([[1, 3.5, 4.2]])  # New data point with bias
proba = predict_proba(X_new, w)
pred = predict(X_new, w)

print(f"Predicted probability: {proba[0][0]:.3f}")
print(f"Predicted class: {pred[0][0]}")
```

---

## Regularization in Logistic Regression

### L2 Regularization (Ridge)

**Cost function with L2 penalty**:
```
J(w) = -(1/m) Σ[y log(ŷ) + (1-y) log(1-ŷ)] + (λ/2m) Σwⱼ²
```

**Gradient**:
```
∇J = (1/m) Xᵀ(ŷ - y) + (λ/m) w
```

**Update rule**:
```
w ← w - α[(1/m) Xᵀ(ŷ - y) + (λ/m) w]
  = w(1 - αλ/m) - α(1/m) Xᵀ(ŷ - y)
```

### L1 Regularization (Lasso)

**Cost function with L1 penalty**:
```
J(w) = -(1/m) Σ[y log(ŷ) + (1-y) log(1-ŷ)] + (λ/m) Σ|wⱼ|
```

**Effect**: Produces sparse solutions (many weights = 0)

### Using Sklearn

```python
from sklearn.linear_model import LogisticRegression

# L2 regularization (default)
model_l2 = LogisticRegression(penalty='l2', C=1.0)  # C = 1/λ

# L1 regularization
model_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')

# No regularization
model_none = LogisticRegression(penalty='none')
```

---

## Practical Implementation

### Complete Class Implementation

```python
import numpy as np

class LogisticRegressionFromScratch:
    """Logistic Regression implemented from scratch"""
    
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_=0.01):
        """
        Args:
            learning_rate: Step size for gradient descent
            num_iterations: Number of training iterations
            regularization: 'l1', 'l2', or None
            lambda_: Regularization strength
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow
    
    def _compute_cost(self, X, y, y_pred):
        """Compute cross-entropy cost"""
        m = len(y)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy
        cost = -(1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # Add regularization
        if self.regularization == 'l2':
            cost += (self.lambda_ / (2*m)) * np.sum(self.weights[1:]**2)
        elif self.regularization == 'l1':
            cost += (self.lambda_ / m) * np.sum(np.abs(self.weights[1:]))
        
        return cost
    
    def fit(self, X, y):
        """Train the model"""
        # Add bias column
        X_b = np.c_[np.ones(X.shape[0]), X]
        y = y.reshape(-1, 1)
        
        m, n = X_b.shape
        self.weights = np.zeros((n, 1))
        
        for i in range(self.num_iterations):
            # Forward pass
            z = X_b @ self.weights
            y_pred = self._sigmoid(z)
            
            # Compute cost
            cost = self._compute_cost(X_b, y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradient
            gradient = (1/m) * X_b.T @ (y_pred - y)
            
            # Add regularization gradient
            if self.regularization == 'l2':
                reg_gradient = (self.lambda_ / m) * self.weights
                reg_gradient[0] = 0  # Don't regularize bias
                gradient += reg_gradient
            elif self.regularization == 'l1':
                reg_gradient = (self.lambda_ / m) * np.sign(self.weights)
                reg_gradient[0] = 0
                gradient += reg_gradient
            
            # Update weights
            self.weights -= self.learning_rate * gradient
            
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X_b = np.c_[np.ones(X.shape[0]), X]
        z = X_b @ self.weights
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score(self, X, y):
        """Compute accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions.ravel() == y)

# Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegressionFromScratch(learning_rate=0.1, num_iterations=1000, 
                                      regularization='l2', lambda_=0.01)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\nTraining accuracy: {train_acc * 100:.2f}%")
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Plot cost
plt.figure(figsize=(10, 5))
plt.plot(model.cost_history, linewidth=2)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.title('Training Cost vs Iteration', fontsize=16)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Summary and Next Steps

### What We Learned

1. **Sigmoid Function**
   ```
   σ(z) = 1 / (1 + e^(-z))
   ```
   - Maps any real number to (0, 1)
   - Smooth, differentiable, S-shaped

2. **Logistic Regression Model**
   ```
   P(y=1|x) = σ(wᵀx)
   ```
   - Outputs probabilities, not hard labels
   - Decision boundary: wᵀx = 0

3. **Cross-Entropy Loss**
   ```
   J(w) = -(1/m) Σ[y log(ŷ) + (1-y) log(1-ŷ)]
   ```
   - Convex loss function
   - Derived from maximum likelihood

4. **Gradient Descent**
   ```
   ∇J = (1/m) Xᵀ(ŷ - y)
   ```
   - Same form as linear regression!
   - But ŷ = σ(wᵀx) instead of wᵀx

5. **Key Differences from Perceptron**
   - Probabilistic predictions
   - Smooth loss function
   - Works even if not linearly separable
   - Provides confidence estimates

### Key Formulas

```
Sigmoid:        σ(z) = 1/(1 + e^(-z))
Derivative:     σ'(z) = σ(z)(1 - σ(z))
Model:          P(y=1|x) = σ(wᵀx)
Loss:           L = -[y log(ŷ) + (1-y) log(1-ŷ)]
Gradient:       ∇J = (1/m) Xᵀ(σ(Xw) - y)
Update:         w ← w - α·∇J
```

### Key Takeaways

✅ **Logistic regression** for probabilistic classification  
✅ **Sigmoid** transforms linear scores to probabilities  
✅ **Cross-entropy** is the right loss function  
✅ **No closed-form** solution → use gradient descent  
✅ **Same boundary** as linear classifier but with probabilities  
✅ **Regularization** prevents overfitting  

### Prerequisites Checklist

Before moving on:
- [ ] Understand sigmoid function
- [ ] Know probability interpretation
- [ ] Can derive gradient
- [ ] Implemented from scratch
- [ ] Understand cross-entropy loss
- [ ] Know how to regularize

### Next Steps

1. **Logistic_Loss_and_Logits.md**
   - Deep dive into cross-entropy
   - Log-odds and logits
   - Maximum likelihood derivation
   - Information theory connection

2. **Multiclass_Linear_Classification.md**
   - Softmax function
   - Multinomial logistic regression
   - One-vs-Rest vs One-vs-One

3. **Regularization_L1_L2.md**
   - Ridge and Lasso for logistic regression
   - Feature selection
   - Preventing overfitting

### Practice Exercises

1. Implement logistic regression from scratch (done!)
2. Derive the gradient step-by-step
3. Visualize sigmoid and its derivative
4. Compare different thresholds for classification
5. Implement L1 and L2 regularization
6. Plot ROC curves and compute AUC

### Additional Resources

**Papers**:
- Cox (1958): "The Regression Analysis of Binary Sequences"
- Original logistic regression paper

**Books**:
- "Pattern Recognition and Machine Learning" (Bishop) - Chapter 4
- "Elements of Statistical Learning" (Hastie) - Chapter 4

**Online**:
- Andrew Ng: ML Course (Coursera)
- Stanford CS229: Logistic Regression notes
- 3Blue1Brown: Neural Networks (starts with logistic)

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║           LOGISTIC REGRESSION QUICK REFERENCE                  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  SIGMOID FUNCTION                                              ║
║  ────────────────                                              ║
║  σ(z) = 1 / (1 + e^(-z))                                       ║
║  σ'(z) = σ(z) · (1 - σ(z))                                     ║
║                                                                ║
║  Properties:                                                   ║
║    • Range: (0, 1)                                             ║
║    • σ(0) = 0.5                                                ║
║    • σ(-z) = 1 - σ(z)                                          ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  MODEL                                                         ║
╠════════════════════════════════════════════════════════════════╣
║  P(y=1|x; w) = σ(wᵀx)                                          ║
║                                                                ║
║  Prediction:                                                   ║
║    ŷ = 1  if P(y=1|x) ≥ 0.5  (or wᵀx ≥ 0)                      ║
║    ŷ = 0  otherwise                                            ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  DECISION BOUNDARY                                             ║
╠════════════════════════════════════════════════════════════════╣
║  wᵀx = 0                                                       ║
║                                                                ║
║  • Same as linear classification                               ║
║  • Hyperplane in feature space                                 ║
║  • But provides probabilities!                                 ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  CROSS-ENTROPY LOSS                                            ║
╠════════════════════════════════════════════════════════════════╣
║  Single example:                                               ║
║    L = -[y log(ŷ) + (1-y) log(1-ŷ)]                            ║
║                                                                ║
║  Cost function:                                                ║
║    J(w) = -(1/m) Σ[yⁱ log(ŷⁱ) + (1-yⁱ) log(1-ŷⁱ)]             ║
║                                                                ║
║  • Convex (one global minimum)                                 ║
║  • Derived from maximum likelihood                             ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  GRADIENT DESCENT                                              ║
╠════════════════════════════════════════════════════════════════╣
║  Gradient:                                                     ║
║    ∇J(w) = (1/m) Xᵀ(ŷ - y)                                     ║
║                                                                ║
║  where ŷ = σ(Xw)                                               ║
║                                                                ║
║  Update rule:                                                  ║
║    w ← w - α·∇J(w)                                             ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  REGULARIZATION                                                ║
╠════════════════════════════════════════════════════════════════╣
║  L2 (Ridge):                                                   ║
║    J(w) = CrossEntropy + (λ/2m)Σwⱼ²                            ║
║                                                                ║
║  L1 (Lasso):                                                   ║
║    J(w) = CrossEntropy + (λ/m)Σ|wⱼ|                            ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  ADVANTAGES                                                    ║
╠════════════════════════════════════════════════════════════════╣
║  ✓ Probabilistic predictions                                   ║
║  ✓ Convex loss (guaranteed global minimum)                     ║
║  ✓ Works with non-separable data                               ║
║  ✓ Confidence estimates                                        ║
║  ✓ Well-calibrated probabilities                               ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  COMPARED TO PERCEPTRON                                        ║
╠════════════════════════════════════════════════════════════════╣
║  Perceptron:      Hard predictions (0 or 1)                    ║
║  Logistic Reg:    Soft predictions (probability)               ║
║                                                                ║
║  Perceptron:      Only works if linearly separable             ║
║  Logistic Reg:    Works always (minimizes loss)                ║
║                                                                ║
║  Perceptron:      No confidence                                ║
║  Logistic Reg:    Provides confidence                          ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Ready to move on?** Next: Deep dive into **cross-entropy loss and logits**!

---

*End of Logistic_Regression.md*