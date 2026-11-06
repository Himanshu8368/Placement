# Linear Classification

## Table of Contents
1. [Introduction to Classification](#introduction-to-classification)
2. [From Regression to Classification](#from-regression-to-classification)
3. [Decision Boundaries](#decision-boundaries)
4. [Linear Separability](#linear-separability)
5. [Perceptron Algorithm](#perceptron-algorithm)
6. [Multi-Class Classification](#multi-class-classification)
7. [Limitations of Linear Classification](#limitations-of-linear-classification)
8. [Practical Implementation](#practical-implementation)
9. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to Classification

### What is Classification?

**Classification** is a supervised learning task where we predict **discrete categories** (classes) instead of continuous values.

**Examples**:
- Email: Spam or Not Spam (2 classes)
- Image: Cat, Dog, or Bird (3 classes)
- Diagnosis: Healthy or Disease (2 classes)
- Handwriting: Digits 0-9 (10 classes)

### Regression vs Classification

```
REGRESSION                    CLASSIFICATION
──────────                    ──────────────
Predict: Continuous value     Predict: Discrete category
Output: ŷ ∈ ℝ                 Output: ŷ ∈ {0, 1, 2, ...}
Example: House price $275k    Example: Email is spam
Loss: MSE, MAE                Loss: Cross-entropy, 0-1 loss
```

### Types of Classification

**Binary Classification** (2 classes):
```
y ∈ {0, 1}  or  y ∈ {-1, +1}
Examples: spam/not spam, healthy/sick
```

**Multi-class Classification** (>2 classes):
```
y ∈ {0, 1, 2, ..., K-1}
Examples: digit recognition (0-9), species classification
```

**Multi-label Classification** (multiple labels per example):
```
y ∈ {0,1}^K  (K binary outputs)
Example: Photo tags (beach, sunset, people)
```

This chapter focuses on **binary classification**.

---

## From Regression to Classification

### The Problem with Direct Regression

**Naive approach**: Use linear regression for classification?

```
For y ∈ {0, 1}:
ŷ = w₀ + w₁x₁ + w₂x₂ + ...
```

**Problems**:
1. **Predictions outside [0,1]**: ŷ can be -5 or 100 (not probabilities!)
2. **No threshold**: What value separates classes?
3. **Sensitive to outliers**: Far points distort the line
4. **Wrong assumptions**: Assumes linear relationship between x and y

### Visual Example

```
    y (class)
    │
  1 │  ●  ●  ●  ●          Class 1
    │                ╱
    │              ╱
0.5 │            ╱
    │          ╱
    │        ╱
  0 │  ●  ●╱ ●  ●           Class 0
    │
    └─────────────────── x
    
    Line crosses at 0.5, but predictions
    can be < 0 or > 1 !
```

### The Classification Setting

Instead of predicting y directly, we predict a **decision function**:

```
f(x) = wᵀx  (linear combination)
```

Then classify based on **sign**:
```
ŷ = { 1  if f(x) ≥ 0
    { 0  if f(x) < 0
```

Or with threshold τ:
```
ŷ = { 1  if f(x) ≥ τ
    { 0  if f(x) < τ
```

### Linear Decision Function

```
f(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = wᵀx
```

**Interpretation**:
- `f(x) > 0`: Predict class 1
- `f(x) < 0`: Predict class 0
- `f(x) = 0`: Decision boundary

**Visual in 2D**:
```
    x₂
    │
    │ ●  ●  ●  (f(x) > 0, Class 1)
    │   ●  ●
    │────────── (f(x) = 0, boundary)
    │ ○  ○
    │ ○  ○  ○  (f(x) < 0, Class 0)
    │
    └──────────── x₁
```

### Example: Email Spam Classification

**Features**:
- x₁ = count of word "free"
- x₂ = count of word "winner"
- x₃ = number of exclamation marks

**Decision function**:
```
f(x) = w₀ + w₁·x₁ + w₂·x₂ + w₃·x₃
     = -5 + 2·x₁ + 3·x₂ + 0.5·x₃
```

**Classification**:
```
If f(x) ≥ 0: Spam
If f(x) < 0: Not Spam
```

**Example email**: 3 "free", 2 "winner", 10 "!"
```
f(x) = -5 + 2(3) + 3(2) + 0.5(10)
     = -5 + 6 + 6 + 5
     = 12 > 0
     → Spam!
```

---

## Decision Boundaries

### Definition

A **decision boundary** is the surface that separates different classes in feature space.

For linear classification:
```
Decision boundary: f(x) = wᵀx = 0
```

This is a **hyperplane** in n-dimensional space:
- **1D** (one feature): A point
- **2D** (two features): A line
- **3D** (three features): A plane
- **nD** (n features): A hyperplane

### In 2D: The Separating Line

For two features x₁ and x₂:
```
f(x) = w₀ + w₁x₁ + w₂x₂ = 0
```

Solving for x₂:
```
x₂ = -(w₀ + w₁x₁)/w₂
```

This is a **straight line**!

**Example**:
```
w₀ = -3, w₁ = 1, w₂ = 1

Decision boundary: -3 + x₁ + x₂ = 0
                  x₂ = 3 - x₁
```

**Visualization**:
```
    x₂
    │
  6 │
    │   ●  ●  ●
  4 │  ●  ●  ●     (Class 1: x₂ > 3 - x₁)
    │ ●  ●
  2 │──────────    (Boundary: x₂ = 3 - x₁)
    │○  ○
  0 │○  ○  ○  ○    (Class 0: x₂ < 3 - x₁)
    │
    └──────────── x₁
    0  2  4  6
```

### Python Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)

# Class 0: cluster around (1, 1)
X0 = np.random.randn(50, 2) * 0.5 + [1, 1]

# Class 1: cluster around (4, 4)
X1 = np.random.randn(50, 2) * 0.5 + [4, 4]

# Combine
X = np.vstack([X0, X1])
y = np.hstack([np.zeros(50), np.ones(50)])

# Decision boundary parameters
w0, w1, w2 = -3, 1, 1

# Plot
plt.figure(figsize=(10, 8))

# Plot points
plt.scatter(X0[:, 0], X0[:, 1], c='blue', marker='o', s=100, alpha=0.7, label='Class 0')
plt.scatter(X1[:, 0], X1[:, 1], c='red', marker='s', s=100, alpha=0.7, label='Class 1')

# Plot decision boundary
x1_line = np.linspace(-1, 6, 100)
x2_line = -(w0 + w1 * x1_line) / w2
plt.plot(x1_line, x2_line, 'g-', linewidth=3, label='Decision Boundary')

# Shade regions
x1_grid = np.linspace(-1, 6, 50)
x2_grid = np.linspace(-1, 6, 50)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
Z = w0 + w1 * X1_grid + w2 * X2_grid
plt.contourf(X1_grid, X2_grid, Z, levels=[-100, 0, 100], colors=['lightblue', 'lightcoral'], alpha=0.3)

plt.xlabel('x₁', fontsize=14)
plt.ylabel('x₂', fontsize=14)
plt.title('Linear Decision Boundary', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

### In 3D: The Separating Plane

For three features x₁, x₂, x₃:
```
f(x) = w₀ + w₁x₁ + w₂x₂ + w₃x₃ = 0
```

This defines a **plane** in 3D space.

**Visualization (conceptual)**:
```
         x₃
         │
         │    ●  ●
         │   ●  ●
         │  ╱────╲  (Plane separates classes)
         │╱   ○  ○╲
         ○  ○   ○  
        ╱│         
       ╱ │         
      ╱  └────────── x₁
     ╱  ╱
    x₂ ╱
```

### Properties of Linear Decision Boundaries

1. **Straight**: No curves (that's non-linear!)
2. **Infinite extent**: Extends forever in all directions
3. **Divides space**: Creates two half-spaces
4. **Parametrized by w**: Different w → different boundary

### Distance from Boundary

The **signed distance** from a point x to the boundary is:
```
distance = f(x) / ||w||
         = wᵀx / ||w||
```

Where `||w|| = √(w₁² + w₂² + ... + wₙ²)` (norm of weight vector).

**Interpretation**:
- **Positive distance**: Class 1 side
- **Negative distance**: Class 0 side
- **Zero distance**: On the boundary

**Confidence**:
- Points far from boundary → high confidence
- Points near boundary → low confidence

---

## Linear Separability

### Definition

Data is **linearly separable** if there exists a **linear decision boundary** that perfectly separates the classes.

```
Linearly separable: Can draw a line/plane that separates all points
```

### Examples

**Linearly Separable**:
```
    x₂
    │
    │  ●  ●
    │ ●  ●
    │────────  (Line separates perfectly)
    │○  ○
    │ ○  ○
    │
    └──────── x₁
    
    ✓ Can separate with a line!
```

**Not Linearly Separable**:
```
    x₂
    │
    │ ●  ○  ●
    │ ○  ●  ○
    │ ●  ○  ●
    │
    └──────── x₁
    
    ✗ No line can separate!
```

### The XOR Problem

**Classic example** of non-linear separability:

```
x₁  x₂  |  y
────────┼────
0   0   |  0
0   1   |  1
1   0   |  1
1   1   |  0
```

**Visualization**:
```
    x₂
    │
  1 │  ○     ●
    │
  0 │  ●     ○
    │
    └──────────── x₁
       0     1
    
    No straight line can separate!
```

**Why?**: Classes are arranged in a checkerboard pattern.

### Formal Definition

Data `{(x⁽ⁱ⁾, y⁽ⁱ⁾)}` is linearly separable if there exists **w** such that:

```
y⁽ⁱ⁾ = 1  ⟹  wᵀx⁽ⁱ⁾ > 0  (Class 1 on positive side)
y⁽ⁱ⁾ = 0  ⟹  wᵀx⁽ⁱ⁾ < 0  (Class 0 on negative side)
```

Or using y ∈ {-1, +1} notation:
```
y⁽ⁱ⁾ · wᵀx⁽ⁱ⁾ > 0  for all i
```

This means: **every point is on the correct side** of the boundary.

### Testing Linear Separability

**Programmatically**:
```python
def is_linearly_separable(X, y, w):
    """
    Check if data is separated by hyperplane w
    
    Args:
        X: Feature matrix (m × n)
        y: Labels {0, 1} or {-1, +1} (m,)
        w: Weight vector (n,)
    
    Returns:
        True if all points correctly classified
    """
    # Convert y to {-1, +1} if needed
    if set(y) == {0, 1}:
        y = 2 * y - 1  # {0,1} → {-1,+1}
    
    # Compute predictions
    scores = X @ w
    
    # Check if all have correct sign
    return np.all(y * scores > 0)
```

**Example usage**:
```python
# Linearly separable data
X = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
y = np.array([0, 0, 1, 1])
w = np.array([1, 1, -2.5])  # [w0, w1, w2]

X_b = np.c_[np.ones(len(X)), X]  # Add bias
print(is_linearly_separable(X_b, y, w))  # True

# XOR - not separable
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
print(is_linearly_separable(np.c_[np.ones(len(X_xor)), X_xor], y_xor, w))  # False
```

### When Data is Not Linearly Separable

**Options**:
1. **Feature engineering**: Add polynomial features (x₁², x₁x₂, etc.)
2. **Kernel methods**: SVM with non-linear kernels
3. **Neural networks**: Multiple layers create non-linear boundaries
4. **Accept errors**: Use soft-margin classifiers

**Example**: Transform XOR to be separable
```python
# Original XOR features: [x1, x2]
# Add polynomial features: [x1, x2, x1*x2]

X_xor_poly = np.c_[X_xor, X_xor[:, 0] * X_xor[:, 1]]
print(X_xor_poly)
# [[0, 0, 0],
#  [0, 1, 0],
#  [1, 0, 0],
#  [1, 1, 1]]

# Now it's separable!
# Decision: y = 1 if x1*x2 = 1, else y = 0
```

### Margin

For linearly separable data, there may be **many** separating hyperplanes.

**Margin**: Minimum distance from boundary to any point.

```
    x₂
    │
    │  ●  ●
    │ ●│ ●  │
    │──┼────┼──  (Large margin - good!)
    │ ○│ ○  │
    │○  ○
    │
    └──────── x₁
       ↑    ↑
     margin
```

**Support Vector Machines (SVM)** find the **maximum margin** separator.

---

## Perceptron Algorithm

### Introduction

The **Perceptron** is the simplest linear classification algorithm, invented by Frank Rosenblatt in 1958.

**Key idea**: Iteratively update weights to correctly classify misclassified points.

### The Perceptron Model

**Decision function**:
```
f(x) = wᵀx
```

**Prediction**:
```
ŷ = sign(f(x)) = { +1  if f(x) ≥ 0
                 { -1  if f(x) < 0
```

**Note**: Uses labels y ∈ {-1, +1} (not {0, 1}).

### The Perceptron Update Rule

**For each misclassified point** (x, y):
```
if y · wᵀx ≤ 0:  (wrong prediction)
    w ← w + η · y · x
```

Where:
- η = learning rate (typically 1 for perceptron)
- y = true label {-1, +1}
- x = feature vector

**Intuition**:
- If y = +1 but wᵀx < 0: Add yx to w (move boundary toward point)
- If y = -1 but wᵀx > 0: Subtract yx from w (move boundary away)

### Visual Intuition

```
Before update:
    x₂
    │
    │ ●  ●
    │────────  (Boundary)
    │ ○  ●  ← Misclassified! (should be ○)
    │○
    │
    └──────── x₁

After update (w ← w - x):
    x₂
    │
    │ ●  ●
    │   ╱
    │  ╱────  (Boundary rotated)
    │ ╱  ○  ← Now correctly classified!
    │○
    │
    └──────── x₁
```

### The Perceptron Algorithm

```
PERCEPTRON ALGORITHM
──────────────────────

Input: Training data {(x⁽ⁱ⁾, y⁽ⁱ⁾)}, max_iterations
Initialize: w = 0 (or small random values)

Repeat for max_iterations or until converged:
    converged = True
    
    For each training example (x, y):
        # Make prediction
        ŷ = sign(wᵀx)
        
        # Update if wrong
        if y ≠ ŷ:  (or: if y · wᵀx ≤ 0)
            w ← w + η · y · x
            converged = False
    
    if converged:
        break  # All points classified correctly!

Output: Learned weights w
```

### Pseudocode

```python
def perceptron(X, y, learning_rate=1.0, max_iterations=1000):
    """
    Perceptron learning algorithm
    
    Args:
        X: Feature matrix (m × n)
        y: Labels {-1, +1} (m,)
        learning_rate: Step size (typically 1.0)
        max_iterations: Maximum epochs
    
    Returns:
        w: Learned weights
        errors: Number of misclassifications per epoch
    """
    m, n = X.shape
    w = np.zeros(n)
    errors = []
    
    for epoch in range(max_iterations):
        error_count = 0
        
        for i in range(m):
            # Prediction
            if y[i] * (X[i] @ w) <= 0:  # Misclassified
                # Update
                w = w + learning_rate * y[i] * X[i]
                error_count += 1
        
        errors.append(error_count)
        
        # Check convergence
        if error_count == 0:
            print(f"Converged at epoch {epoch}")
            break
    
    return w, errors
```

### Complete Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """Perceptron classifier"""
    
    def __init__(self, learning_rate=1.0, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.errors = []
    
    def fit(self, X, y):
        """
        Train perceptron
        
        Args:
            X: Feature matrix (m × n)
            y: Labels {-1, +1} (m,)
        """
        # Add bias term
        X_b = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize weights
        self.weights = np.zeros(X_b.shape[1])
        self.errors = []
        
        # Training loop
        for epoch in range(self.max_iterations):
            error_count = 0
            
            for i in range(len(X_b)):
                # Make prediction
                prediction = np.sign(X_b[i] @ self.weights)
                
                # Update if wrong
                if y[i] * (X_b[i] @ self.weights) <= 0:
                    self.weights += self.learning_rate * y[i] * X_b[i]
                    error_count += 1
            
            self.errors.append(error_count)
            
            # Check convergence
            if error_count == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_b = np.c_[np.ones(X.shape[0]), X]
        return np.sign(X_b @ self.weights)
    
    def score(self, X, y):
        """Compute accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Generate linearly separable data
np.random.seed(42)

# Class -1
X_neg = np.random.randn(50, 2) * 0.5 + [1, 1]
y_neg = -np.ones(50)

# Class +1
X_pos = np.random.randn(50, 2) * 0.5 + [3, 3]
y_pos = np.ones(50)

# Combine
X = np.vstack([X_neg, X_pos])
y = np.hstack([y_neg, y_pos])

# Train perceptron
perceptron = Perceptron(learning_rate=1.0, max_iterations=100)
perceptron.fit(X, y)

# Evaluate
accuracy = perceptron.score(X, y)
print(f"Training accuracy: {accuracy * 100:.2f}%")

# Visualize
plt.figure(figsize=(15, 5))

# Plot 1: Data and decision boundary
plt.subplot(1, 3, 1)
plt.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', marker='o', s=100, alpha=0.7, label='Class -1')
plt.scatter(X_pos[:, 0], X_pos[:, 1], c='red', marker='s', s=100, alpha=0.7, label='Class +1')

# Decision boundary
w0, w1, w2 = perceptron.weights
x1_line = np.linspace(-1, 5, 100)
x2_line = -(w0 + w1 * x1_line) / w2
plt.plot(x1_line, x2_line, 'g-', linewidth=3, label='Decision Boundary')

plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.title('Perceptron Decision Boundary', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Error convergence
plt.subplot(1, 3, 2)
plt.plot(perceptron.errors, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Number of Errors', fontsize=12)
plt.title('Perceptron Convergence', fontsize=14)
plt.grid(True)

# Plot 3: Decision regions
plt.subplot(1, 3, 3)
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                        np.linspace(x2_min, x2_max, 200))
Z = perceptron.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', marker='o', s=100, alpha=0.7, label='Class -1')
plt.scatter(X_pos[:, 0], X_pos[:, 1], c='red', marker='s', s=100, alpha=0.7, label='Class +1')
plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.title('Decision Regions', fontsize=14)
plt.legend()

plt.tight_layout()
plt.show()
```

### Convergence Theorem

**Perceptron Convergence Theorem**: If data is **linearly separable**, the perceptron algorithm will converge to a solution in **finite steps**.

**Proof sketch**:
1. There exists a perfect separator w*
2. Each update moves w closer to w* (in terms of angle)
3. After finite updates, w becomes aligned with w*
4. Number of updates bounded by: `R² / γ²`

Where:
- R = maximum distance of any point from origin
- γ = margin (minimum distance to boundary)

**Key insight**: If linearly separable, perceptron **will** find a solution (but not necessarily the best one).

### What if Not Linearly Separable?

**Problem**: Perceptron will **never converge** if data is not linearly separable.

**Behavior**:
- Keeps making mistakes
- Weights oscillate
- Never stops (unless max_iterations reached)

**Solutions**:
1. **Limit iterations**: Stop after max_iterations
2. **Pocket algorithm**: Keep track of best weights seen
3. **Averaged perceptron**: Average all weights during training
4. **Use different algorithm**: Logistic regression, SVM

### Pocket Algorithm

Modified perceptron that keeps the **best weights** seen so far:

```python
def pocket_perceptron(X, y, max_iterations=1000):
    """Perceptron with pocket algorithm"""
    X_b = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X_b.shape[1])
    w_best = w.copy()
    best_accuracy = 0
    
    for epoch in range(max_iterations):
        for i in range(len(X_b)):
            if y[i] * (X_b[i] @ w) <= 0:
                w = w + y[i] * X_b[i]
        
        # Compute accuracy
        predictions = np.sign(X_b @ w)
        accuracy = np.mean(predictions == y)
        
        # Update best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            w_best = w.copy()
    
    return w_best
```

---

## Multi-Class Classification

### Strategies

For K classes (K > 2), we need strategies to reduce to binary classification.

#### 1. One-vs-Rest (OvR) / One-vs-All (OvA)

**Idea**: Train K binary classifiers, one for each class vs all others.

```
For class k:
    Positive examples: y = k
    Negative examples: y ≠ k
```

**Prediction**: Choose class with highest score.
```
ŷ = argmax_k f_k(x)
```

**Example** (3 classes):
```
Classifier 1: Class 0 vs {Class 1, Class 2}
Classifier 2: Class 1 vs {Class 0, Class 2}
Classifier 3: Class 2 vs {Class 0, Class 1}

For new point x:
    Score_0 = f_0(x)
    Score_1 = f_1(x)
    Score_2 = f_2(x)
    
    Predict: class with max score
```

#### 2. One-vs-One (OvO)

**Idea**: Train K(K-1)/2 binary classifiers, one for each pair of classes.

```
For each pair (i, j):
    Train classifier: class i vs class j
```

**Prediction**: Each classifier votes, choose class with most votes.

**Example** (3 classes):
```
Classifier 1: Class 0 vs Class 1
Classifier 2: Class 0 vs Class 2
Classifier 3: Class 1 vs Class 2

For new point x:
    Votes = [0, 0, 0]
    
    If C1 predicts 0: Votes[0] += 1
    If C1 predicts 1: Votes[1] += 1
    
    If C2 predicts 0: Votes[0] += 1
    If C2 predicts 2: Votes[2] += 1
    
    If C3 predicts 1: Votes[1] += 1
    If C3 predicts 2: Votes[2] += 1
    
    Predict: class with most votes
```

### Implementation

```python
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset (3 classes)
iris = load_iris()
X, y = iris.data[:, :2], iris.target  # Use only 2 features for visualization

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# One-vs-Rest
ovr = OneVsRestClassifier(Perceptron(max_iter=1000, random_state=42))
ovr.fit(X_train, y_train)
print(f"OvR Accuracy: {ovr.score(X_test, y_test) * 100:.2f}%")

# One-vs-One
ovo = OneVsOneClassifier(Perceptron(max_iter=1000, random_state=42))
ovo.fit(X_train, y_train)
print(f"OvO Accuracy: {ovo.score(X_test, y_test) * 100:.2f}%")
```

---

## Limitations of Linear Classification

### 1. Linear Separability Requirement

**Limitation**: Only works well if data is (approximately) linearly separable.

**Example failures**:
- XOR problem
- Concentric circles
- Interleaved spirals

```
Concentric circles:
    │
    │  ● ● ●
    │ ●○○○●
    │ ●○ ○●
    │ ●○○○●
    │  ● ● ●
    └──────── 
    
    Inner circle: Class 0
    Outer ring: Class 1
    Cannot separate with line!
```

### 2. No Probability Estimates

**Limitation**: Perceptron gives hard predictions (0 or 1), no confidence.

**Better**: Logistic regression gives probabilities P(y=1|x)

```
Perceptron:  ŷ = +1 or -1  (hard prediction)
Logistic:    ŷ = 0.85      (85% confident it's class 1)
```

### 3. Sensitive to Outliers

**Limitation**: One outlier can drastically change the boundary.

```
Before outlier:
    │  ●●●
    │ ●●●
    │──────  (Good separation)
    │○○
    │ ○○

After outlier:
    │  ●●● ○  ← Outlier!
    │ ●●●╱
    │───╱───  (Boundary shifted)
    │○○╱
    │ ○
```

### 4. Many Solutions

**Limitation**: If linearly separable, many valid solutions exist. Which to choose?

```
    │  ●●●
    │ ●●●
    │───1───  (Solution 1)
    │───2───  (Solution 2)
    │───3───  (Solution 3)
    │○○
    │ ○○
    
    All three are valid!
```

**Better**: SVM finds **maximum margin** separator.

### 5. No Margin Optimization

**Limitation**: Perceptron stops at first perfect solution, which may have small margin.

**Better**: SVM maximizes margin → better generalization.

---

## Practical Implementation

### Using Scikit-Learn

```python
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic data
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# Convert to {-1, +1}
y = 2 * y - 1

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train perceptron
clf = Perceptron(max_iter=1000, eta0=1.0, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get weights
print(f"\nWeights: {clf.coef_}")
print(f"Bias: {clf.intercept_}")
```

### Complete Example with Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate well-separated data
X, y = make_blobs(n_samples=100, centers=2, n_features=2, 
                   center_box=(-5, 5), cluster_std=1.0, random_state=42)

# Convert to {-1, +1}
y = 2 * y - 1

# Train from scratch
class PerceptronFromScratch:
    def __init__(self, lr=1.0, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.history = []
    
    def fit(self, X, y):
        # Add bias
        X_b = np.c_[np.ones(len(X)), X]
        self.w = np.zeros(X_b.shape[1])
        
        for epoch in range(self.epochs):
            errors = 0
            for i in range(len(X_b)):
                if y[i] * (X_b[i] @ self.w) <= 0:
                    self.w += self.lr * y[i] * X_b[i]
                    errors += 1
            
            self.history.append(errors)
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
        
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones(len(X)), X]
        return np.sign(X_b @ self.w)
    
    def decision_function(self, X):
        X_b = np.c_[np.ones(len(X)), X]
        return X_b @ self.w

# Train
model = PerceptronFromScratch(lr=1.0, epochs=100)
model.fit(X, y)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Data and boundary
ax = axes[0]
ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='o', s=100, alpha=0.7, label='Class -1')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='s', s=100, alpha=0.7, label='Class +1')

# Decision boundary
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_boundary = np.linspace(x1_min, x1_max, 100)
w0, w1, w2 = model.w
x2_boundary = -(w0 + w1 * x1_boundary) / w2
ax.plot(x1_boundary, x2_boundary, 'g-', linewidth=3, label='Decision Boundary')

ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_title('Perceptron Classification', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Convergence
ax = axes[1]
ax.plot(model.history, linewidth=2, marker='o')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Number of Misclassifications', fontsize=12)
ax.set_title('Training Convergence', fontsize=14)
ax.grid(True)

# Plot 3: Decision function values
ax = axes[2]
x1_grid = np.linspace(x1_min, x1_max, 100)
x2_grid = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
Z = model.decision_function(np.c_[X1_grid.ravel(), X2_grid.ravel()])
Z = Z.reshape(X1_grid.shape)

contour = ax.contourf(X1_grid, X2_grid, Z, levels=20, cmap='RdYlBu', alpha=0.7)
ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='o', s=100, edgecolors='k', alpha=0.9)
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='s', s=100, edgecolors='k', alpha=0.9)
ax.contour(X1_grid, X2_grid, Z, levels=[0], colors='black', linewidths=3)

ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_title('Decision Function Values', fontsize=14)
plt.colorbar(contour, ax=ax)

plt.tight_layout()
plt.show()

# Print results
accuracy = np.mean(model.predict(X) == y)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
print(f"Learned weights: w₀={w0:.3f}, w₁={w1:.3f}, w₂={w2:.3f}")
```

---

## Summary and Next Steps

### What We Learned

1. **From Regression to Classification**
   - Use decision functions instead of direct prediction
   - f(x) = wᵀx, classify based on sign

2. **Decision Boundaries**
   - Hyperplanes that separate classes
   - Lines in 2D, planes in 3D, hyperplanes in nD
   - Defined by w: f(x) = wᵀx = 0

3. **Linear Separability**
   - Data that can be separated by linear boundary
   - XOR is classic non-separable example
   - Not all data is linearly separable

4. **Perceptron Algorithm**
   - Simplest linear classifier
   - Update rule: w ← w + η·y·x (for misclassified)
   - Converges if data is linearly separable
   - No probability estimates

5. **Multi-class Classification**
   - One-vs-Rest: K binary classifiers
   - One-vs-One: K(K-1)/2 binary classifiers

6. **Limitations**
   - Requires linear separability
   - No probabilities
   - No margin optimization
   - Sensitive to outliers

### Key Formulas

**Decision function**:
```
f(x) = wᵀx = w₀ + w₁x₁ + ... + wₙxₙ
```

**Prediction**:
```
ŷ = sign(f(x))
```

**Perceptron update**:
```
if y · wᵀx ≤ 0:
    w ← w + η·y·x
```

### Key Takeaways

✅ **Linear classification** uses hyperplanes to separate classes  
✅ **Perceptron** is simple but effective for linearly separable data  
✅ **Linear separability** is key requirement  
✅ **XOR** is classic example of non-separable data  
✅ **Many limitations** → need better methods  

### Prerequisites Checklist

Before moving on:
- [ ] Understand decision boundaries
- [ ] Know what linear separability means
- [ ] Can implement perceptron from scratch
- [ ] Understand convergence theorem
- [ ] Know perceptron limitations
- [ ] Understand multi-class strategies

### Next Steps

1. **Logistic_Regression.md**
   - **Probability estimates**: P(y=1|x)
   - **Sigmoid function**: Smooth predictions
   - **Cross-entropy loss**: Proper classification loss
   - **Better than perceptron**: Works even if not separable
   - **Gradient descent**: No closed-form, must iterate

2. **Logistic_Loss_and_Logits.md**
   - Deep dive into cross-entropy
   - Connection to information theory
   - Logits and log-odds

3. **Regularization_L1_L2.md**
   - Prevent overfitting
   - Ridge and Lasso for classification
   - Feature selection

### Practice Exercises

1. Implement perceptron from scratch (done!)
2. Test on XOR data (should not converge)
3. Add polynomial features to make XOR separable
4. Visualize decision boundaries for different weights
5. Implement pocket algorithm
6. Compare OvR and OvO for multi-class

### Additional Resources

**Papers**:
- Rosenblatt (1958): "The Perceptron: A Probabilistic Model"
- Minsky & Papert (1969): "Perceptrons" (showed limitations)
- Novikoff (1962): Perceptron convergence theorem

**Books**:
- "Pattern Recognition and Machine Learning" (Bishop) - Chapter 4
- "Neural Networks and Deep Learning" (Nielsen) - Chapter 1

**Online**:
- CS231n: Linear Classification notes
- 3Blue1Brown: Neural networks (starts with perceptron)
- Colah's blog: Neural networks

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║            LINEAR CLASSIFICATION QUICK REFERENCE               ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  DECISION FUNCTION                                             ║
║  ─────────────────                                             ║
║  f(x) = wᵀx = w₀ + w₁x₁ + ... + wₙxₙ                          ║
║                                                                ║
║  PREDICTION                                                    ║
║  ──────────                                                    ║
║  ŷ = sign(f(x)) = { +1  if f(x) ≥ 0                            ║
║                   { -1  if f(x) < 0                            ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  DECISION BOUNDARY                                             ║
╠════════════════════════════════════════════════════════════════╣
║  Set of points where f(x) = 0                                  ║
║                                                                ║
║  1D: Point                                                     ║
║  2D: Line      (w₀ + w₁x₁ + w₂x₂ = 0)                         ║
║  3D: Plane     (w₀ + w₁x₁ + w₂x₂ + w₃x₃ = 0)                  ║
║  nD: Hyperplane                                                ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  LINEAR SEPARABILITY                                           ║
╠════════════════════════════════════════════════════════════════╣
║  Data is linearly separable if ∃w such that:                   ║
║    y⁽ⁱ⁾ · wᵀx⁽ⁱ⁾ > 0  for all i                                 ║
║                                                                ║
║  XOR problem: NOT linearly separable                           ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  PERCEPTRON ALGORITHM                                          ║
╠════════════════════════════════════════════════════════════════╣
║  Initialize: w = 0                                             ║
║                                                                ║
║  Repeat:                                                       ║
║    For each (x, y):                                            ║
║      if y · wᵀx ≤ 0:  # Misclassified                          ║
║        w ← w + η·y·x                                           ║
║                                                                ║
║  Until: All points classified correctly                        ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  CONVERGENCE THEOREM                                           ║
╠════════════════════════════════════════════════════════════════╣
║  If data is linearly separable:                                ║
║    Perceptron WILL converge in finite steps                    ║
║                                                                ║
║  If NOT linearly separable:                                    ║
║    Perceptron will never converge                              ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  MULTI-CLASS                                                   ║
╠════════════════════════════════════════════════════════════════╣
║  One-vs-Rest: K binary classifiers                             ║
║    Predict: argmax_k f_k(x)                                    ║
║                                                                ║
║  One-vs-One: K(K-1)/2 binary classifiers                       ║
║    Predict: Majority vote                                      ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  LIMITATIONS                                                   ║
╠════════════════════════════════════════════════════════════════╣
║  ✗ Requires linear separability                                ║
║  ✗ No probability estimates                                    ║
║  ✗ No margin optimization                                      ║
║  ✗ Sensitive to outliers                                       ║
║  ✗ Many valid solutions (if separable)                         ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  BETTER ALTERNATIVES                                           ║
╠════════════════════════════════════════════════════════════════╣
║  Logistic Regression → Probabilities                           ║
║  SVM → Maximum margin                                          ║
║  Neural Networks → Non-linear boundaries                       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Ready to move on?** Next up: **Logistic Regression** - classification with probability estimates!

---

*End of Linear_Classification.md*