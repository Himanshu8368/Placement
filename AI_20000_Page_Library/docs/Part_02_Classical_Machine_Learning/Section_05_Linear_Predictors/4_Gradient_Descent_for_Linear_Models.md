# Gradient Descent for Linear Models

## Table of Contents
1. [Introduction to Gradient Descent](#introduction-to-gradient-descent)
2. [The Gradient Descent Algorithm](#the-gradient-descent-algorithm)
3. [Batch Gradient Descent (BGD)](#batch-gradient-descent-bgd)
4. [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
5. [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
6. [Learning Rate Selection](#learning-rate-selection)
7. [Convergence Analysis](#convergence-analysis)
8. [Practical Considerations](#practical-considerations)
9. [Comparison of Methods](#comparison-of-methods)
10. [Advanced Optimization](#advanced-optimization)
11. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to Gradient Descent

### What is Gradient Descent?

**Gradient Descent (GD)** is an iterative optimization algorithm that finds the minimum of a function by repeatedly taking steps in the direction of steepest descent (negative gradient).

**Think of it like**:
- You're on a mountain blindfolded
- You want to reach the valley (minimum)
- At each step, you feel the slope and walk downhill
- Repeat until you reach the bottom

### Why Not Use Closed-Form?

Recall the **closed-form solution** for linear regression:
```
w = (XᵀX)⁻¹Xᵀy
```

**Problems with closed-form**:
1. **Computational cost**: O(n³) for matrix inversion
2. **Memory**: Requires storing XᵀX which is O(n²)
3. **Not scalable**: Infeasible for n > 10,000
4. **No closed-form**: Doesn't exist for many models (logistic regression, neural networks)

**Gradient descent advantages**:
1. **Scales to large datasets**: O(n) per iteration
2. **Scales to many features**: Memory O(n)
3. **General purpose**: Works for any differentiable cost function
4. **Online learning**: Can update with streaming data

### The Basic Idea

**Goal**: Minimize cost function J(w)

**Method**: 
1. Start with random weights w
2. Compute gradient ∇J(w) (direction of steepest ascent)
3. Update: w ← w - α·∇J(w) (step in opposite direction)
4. Repeat until convergence

**Visual intuition**:
```
    J(w)
    │
    │     ●  (start)
    │      ╲
    │       ●
    │        ╲
    │         ●  (moving downhill)
    │          ╲
    │           ●
    │            ╲___●  (converge to minimum)
    │                ╲___
    └─────────────────────── w
```

### When to Use Gradient Descent?

**Use GD when**:
- Large datasets (m > 100,000)
- Many features (n > 10,000)
- No closed-form solution exists
- Need online/incremental learning
- Want to trade accuracy for speed

**Use closed-form when**:
- Small datasets (m < 10,000)
- Few features (n < 1,000)
- Need exact solution
- Closed-form exists and XᵀX is invertible

---

## The Gradient Descent Algorithm

### Generic Template

```
Input: Training data (X, y), learning rate α, max iterations T
Initialize: w randomly (or zeros)

For t = 1 to T:
    1. Compute cost J(w)
    2. Compute gradient ∇J(w)
    3. Update weights: w ← w - α·∇J(w)
    4. Check convergence (optional)
    
Output: Learned weights w
```

### Cost Function for Linear Regression

```
J(w) = (1/2m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
     = (1/2m) Σᵢ₌₁ᵐ (wᵀx⁽ⁱ⁾ - y⁽ⁱ⁾)²
```

In matrix form:
```
J(w) = (1/2m) ||Xw - y||²
     = (1/2m) (Xw - y)ᵀ(Xw - y)
```

**Note**: Factor of 1/2 simplifies derivative (doesn't affect optimization).

### Gradient Derivation

**Scalar form** (for one weight wⱼ):
```
∂J/∂wⱼ = (1/m) Σᵢ₌₁ᵐ (wᵀx⁽ⁱ⁾ - y⁽ⁱ⁾)·x⁽ⁱ⁾ⱼ
```

**Derivation**:
```
∂J/∂wⱼ = ∂/∂wⱼ [(1/2m) Σᵢ₌₁ᵐ (wᵀx⁽ⁱ⁾ - y⁽ⁱ⁾)²]

       = (1/2m) Σᵢ₌₁ᵐ 2(wᵀx⁽ⁱ⁾ - y⁽ⁱ⁾) · ∂/∂wⱼ(wᵀx⁽ⁱ⁾)
       
       = (1/m) Σᵢ₌₁ᵐ (wᵀx⁽ⁱ⁾ - y⁽ⁱ⁾) · x⁽ⁱ⁾ⱼ
```

**Vector form** (gradient of all weights):
```
∇J(w) = (1/m) Xᵀ(Xw - y)
```

**Derivation**:
```
J(w) = (1/2m)(Xw - y)ᵀ(Xw - y)

∇J(w) = (1/2m) · 2Xᵀ(Xw - y)
       = (1/m) Xᵀ(Xw - y)
```

### Update Rule

**For each weight**:
```
wⱼ ← wⱼ - α · (1/m) Σᵢ₌₁ᵐ (wᵀx⁽ⁱ⁾ - y⁽ⁱ⁾)·x⁽ⁱ⁾ⱼ
```

**Vector form**:
```
w ← w - α · (1/m) Xᵀ(Xw - y)
```

Where α is the **learning rate** (step size).

---

## Batch Gradient Descent (BGD)

### Definition

**Batch Gradient Descent** uses the **entire dataset** to compute the gradient at each iteration.

### Algorithm

```
Initialize: w = 0 (or random)
Repeat until convergence:
    # Compute predictions for ALL examples
    ŷ = Xw
    
    # Compute gradient using ALL examples
    ∇J(w) = (1/m) Xᵀ(ŷ - y)
    
    # Update weights
    w ← w - α·∇J(w)
```

### Pseudocode

```python
def batch_gradient_descent(X, y, alpha, num_iterations):
    """
    Batch Gradient Descent for Linear Regression
    
    Args:
        X: Feature matrix (m × n+1) with bias column
        y: Target vector (m × 1)
        alpha: Learning rate
        num_iterations: Number of iterations
    
    Returns:
        w: Learned weights
        J_history: Cost at each iteration
    """
    m, n = X.shape
    w = np.zeros(n)  # Initialize weights
    J_history = []
    
    for iteration in range(num_iterations):
        # Compute predictions
        y_pred = X @ w
        
        # Compute gradient
        gradient = (1/m) * X.T @ (y_pred - y)
        
        # Update weights
        w = w - alpha * gradient
        
        # Compute and store cost
        J = (1/(2*m)) * np.sum((y_pred - y)**2)
        J_history.append(J)
    
    return w, J_history
```

### Complete Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# Add bias term
X_b = np.c_[np.ones((m, 1)), X]

# Batch Gradient Descent
def batch_gradient_descent(X, y, alpha=0.1, num_iterations=1000):
    m, n = X.shape
    w = np.zeros((n, 1))
    J_history = []
    
    for i in range(num_iterations):
        # Predictions
        y_pred = X @ w
        
        # Gradient
        gradient = (1/m) * X.T @ (y_pred - y)
        
        # Update
        w = w - alpha * gradient
        
        # Cost
        J = (1/(2*m)) * np.sum((y_pred - y)**2)
        J_history.append(J)
        
        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {J:.4f}")
    
    return w, J_history

# Train
w_bgd, J_history = batch_gradient_descent(X_b, y, alpha=0.1, num_iterations=1000)

print("\nFinal weights:")
print(f"w0 (intercept): {w_bgd[0][0]:.4f}")
print(f"w1 (slope): {w_bgd[1][0]:.4f}")

# Compare with closed-form
w_closed = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print("\nClosed-form weights:")
print(f"w0: {w_closed[0][0]:.4f}")
print(f"w1: {w_closed[1][0]:.4f}")

# Plot convergence
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost J(w)')
plt.title('Batch GD Convergence')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, X_b @ w_bgd, 'r-', linewidth=2, label='BGD')
plt.plot(X, X_b @ w_closed, 'g--', linewidth=2, label='Closed-form')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Fitted Line')
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Output**:
```
Iteration 0: Cost = 4.2151
Iteration 100: Cost = 0.5129
Iteration 200: Cost = 0.5129
...
Iteration 900: Cost = 0.5129

Final weights:
w0 (intercept): 4.2152
w1 (slope): 2.7702

Closed-form weights:
w0: 4.2152
w1: 2.7702
```

### Characteristics of BGD

**Advantages**:
✅ Stable convergence (smooth cost curve)
✅ Guaranteed to converge to global minimum (for convex functions)
✅ Exact gradient computation
✅ Reproducible results

**Disadvantages**:
❌ Slow for large datasets (computes gradient over all m examples)
❌ Requires all data in memory
❌ Redundant computation if data has duplicate patterns
❌ Cannot do online learning

**Computational Cost per Iteration**:
- Computing Xw: O(mn)
- Computing gradient: O(mn)
- Total: **O(mn)**

For m = 1,000,000 and n = 100: ~100 million operations per iteration!

---

## Stochastic Gradient Descent (SGD)

### Definition

**Stochastic Gradient Descent** updates weights using **one randomly selected example** at a time.

### Key Idea

Instead of:
```
∇J(w) = (1/m) Σᵢ₌₁ᵐ ∇Jᵢ(w)  (average over all examples)
```

Use:
```
∇Jᵢ(w) for a single random example i
```

This is a **stochastic (noisy) estimate** of the true gradient.

### Algorithm

```
Initialize: w = 0 (or random)
Repeat for each epoch:
    Shuffle training data
    For each example (x⁽ⁱ⁾, y⁽ⁱ⁾):
        # Compute prediction for ONE example
        ŷ⁽ⁱ⁾ = wᵀx⁽ⁱ⁾
        
        # Compute gradient using ONE example
        ∇Jᵢ(w) = (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)·x⁽ⁱ⁾
        
        # Update weights
        w ← w - α·∇Jᵢ(w)
```

**Key difference**: Update happens **m times per epoch** (once per example), not once per epoch!

### Pseudocode

```python
def stochastic_gradient_descent(X, y, alpha, num_epochs):
    """
    Stochastic Gradient Descent for Linear Regression
    
    Args:
        X: Feature matrix (m × n+1)
        y: Target vector (m × 1)
        alpha: Learning rate
        num_epochs: Number of passes through data
    
    Returns:
        w: Learned weights
        J_history: Cost history
    """
    m, n = X.shape
    w = np.zeros(n)
    J_history = []
    
    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        
        for i in indices:
            # Select one example
            xi = X[i:i+1]
            yi = y[i:i+1]
            
            # Compute prediction
            y_pred = xi @ w
            
            # Compute gradient (for this one example)
            gradient = xi.T @ (y_pred - yi)
            
            # Update weights
            w = w - alpha * gradient
        
        # Compute cost on full dataset (for monitoring)
        y_pred_all = X @ w
        J = (1/(2*m)) * np.sum((y_pred_all - y)**2)
        J_history.append(J)
    
    return w, J_history
```

### Complete Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Same data as before
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), X]

def stochastic_gradient_descent(X, y, alpha=0.01, num_epochs=50):
    m, n = X.shape
    w = np.zeros((n, 1))
    J_history = []
    
    for epoch in range(num_epochs):
        # Shuffle
        indices = np.random.permutation(m)
        
        for i in indices:
            xi = X[i:i+1]
            yi = y[i:i+1]
            
            # Gradient for one example
            y_pred = xi @ w
            gradient = xi.T @ (y_pred - yi)
            
            # Update
            w = w - alpha * gradient
        
        # Cost (for monitoring)
        y_pred_all = X @ w
        J = (1/(2*m)) * np.sum((y_pred_all - y)**2)
        J_history.append(J)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Cost = {J:.4f}")
    
    return w, J_history

# Train with SGD
w_sgd, J_history_sgd = stochastic_gradient_descent(X_b, y, alpha=0.01, num_epochs=50)

print("\nFinal weights (SGD):")
print(f"w0: {w_sgd[0][0]:.4f}")
print(f"w1: {w_sgd[1][0]:.4f}")

# Compare convergence with BGD
w_bgd, J_history_bgd = batch_gradient_descent(X_b, y, alpha=0.1, num_iterations=50)

plt.figure(figsize=(10, 5))
plt.plot(J_history_bgd, label='Batch GD', linewidth=2)
plt.plot(J_history_sgd, label='Stochastic GD', linewidth=2)
plt.xlabel('Epoch/Iteration')
plt.ylabel('Cost J(w)')
plt.title('BGD vs SGD Convergence')
plt.legend()
plt.grid(True)
plt.show()
```

**Output**:
```
Epoch 0: Cost = 0.5884
Epoch 10: Cost = 0.5157
Epoch 20: Cost = 0.5134
Epoch 30: Cost = 0.5130
Epoch 40: Cost = 0.5129

Final weights (SGD):
w0: 4.2089
w1: 2.7759
```

### Characteristics of SGD

**Advantages**:
✅ Fast updates (one example at a time)
✅ Can handle very large datasets (streaming data)
✅ Memory efficient (only one example needed)
✅ Can escape shallow local minima (due to noise)
✅ Online learning possible

**Disadvantages**:
❌ Noisy convergence (oscillates around minimum)
❌ Never truly converges (always fluctuating)
❌ Sensitive to learning rate
❌ May need learning rate schedule (decay over time)

**Computational Cost per Epoch**:
- m updates, each O(n)
- Total: **O(mn)** (same as BGD per epoch)
- But: More frequent updates → faster convergence in wall-clock time

### Convergence Pattern

```
Cost
  │
  │  BGD: Smooth decrease
  │  ╲
  │   ╲_______________  (converges smoothly)
  │
  │  SGD: Noisy decrease
  │  ╲╱╲╱╲
  │   ╲╱╲╱╲___╱╲___    (oscillates around minimum)
  │
  └──────────────────── Iterations
```

### Learning Rate Decay

To reduce oscillations, **decrease learning rate over time**:

```python
# Simple decay
alpha_t = alpha_0 / (1 + decay_rate * epoch)

# Step decay
alpha_t = alpha_0 * 0.5^(epoch // drop_every)

# Exponential decay
alpha_t = alpha_0 * exp(-decay_rate * epoch)
```

**Example**:
```python
def sgd_with_decay(X, y, alpha_0=0.1, num_epochs=100, decay_rate=0.1):
    m, n = X.shape
    w = np.zeros((n, 1))
    
    for epoch in range(num_epochs):
        # Decay learning rate
        alpha = alpha_0 / (1 + decay_rate * epoch)
        
        indices = np.random.permutation(m)
        for i in indices:
            xi = X[i:i+1]
            yi = y[i:i+1]
            gradient = xi.T @ (xi @ w - yi)
            w = w - alpha * gradient
    
    return w
```

---

## Mini-Batch Gradient Descent

### Definition

**Mini-Batch Gradient Descent** is a compromise between BGD and SGD:
- Uses a **small random subset** (batch) of examples to compute gradient
- Batch size typically: 32, 64, 128, 256

### Key Idea

```
Batch GD:       Use all m examples → slow but stable
SGD:            Use 1 example → fast but noisy
Mini-batch GD:  Use b examples → balanced!
```

### Algorithm

```
Initialize: w = 0
Choose: batch_size (e.g., 32)

Repeat for each epoch:
    Shuffle training data
    For each mini-batch B of size b:
        # Compute predictions for batch B
        ŷ_B = X_B @ w
        
        # Compute gradient using batch B
        ∇J(w) = (1/b) X_Bᵀ(ŷ_B - y_B)
        
        # Update weights
        w ← w - α·∇J(w)
```

### Pseudocode

```python
def mini_batch_gradient_descent(X, y, alpha, num_epochs, batch_size=32):
    """
    Mini-Batch Gradient Descent for Linear Regression
    
    Args:
        X: Feature matrix (m × n+1)
        y: Target vector (m × 1)
        alpha: Learning rate
        num_epochs: Number of epochs
        batch_size: Size of each mini-batch
    
    Returns:
        w: Learned weights
        J_history: Cost history
    """
    m, n = X.shape
    w = np.zeros(n)
    J_history = []
    
    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute gradient on mini-batch
            y_pred = X_batch @ w
            gradient = (1/len(X_batch)) * X_batch.T @ (y_pred - y_batch)
            
            # Update weights
            w = w - alpha * gradient
        
        # Compute cost on full dataset
        y_pred_all = X @ w
        J = (1/(2*m)) * np.sum((y_pred_all - y)**2)
        J_history.append(J)
    
    return w, J_history
```

### Complete Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Same data
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), X]

def mini_batch_gd(X, y, alpha=0.1, num_epochs=50, batch_size=20):
    m, n = X.shape
    w = np.zeros((n, 1))
    J_history = []
    
    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            y_pred = X_batch @ w
            gradient = (1/len(X_batch)) * X_batch.T @ (y_pred - y_batch)
            w = w - alpha * gradient
        
        # Cost
        y_pred_all = X @ w
        J = (1/(2*m)) * np.sum((y_pred_all - y)**2)
        J_history.append(J)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Cost = {J:.4f}")
    
    return w, J_history

# Compare different batch sizes
batch_sizes = [1, 10, 32, 100]
results = {}

for bs in batch_sizes:
    w, J_hist = mini_batch_gd(X_b, y, alpha=0.1, num_epochs=50, batch_size=bs)
    results[bs] = J_hist
    print(f"\nBatch size {bs}: Final cost = {J_hist[-1]:.4f}")

# Plot comparison
plt.figure(figsize=(10, 6))
for bs, J_hist in results.items():
    label = f'Batch size {bs}' if bs > 1 else 'SGD (batch=1)'
    label = 'BGD (batch=100)' if bs == 100 else label
    plt.plot(J_hist, label=label, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Cost J(w)')
plt.title('Mini-Batch GD: Effect of Batch Size')
plt.legend()
plt.grid(True)
plt.show()
```

**Output**:
```
Batch size 1: Final cost = 0.5130
Batch size 10: Final cost = 0.5129
Batch size 32: Final cost = 0.5129
Batch size 100: Final cost = 0.5129
```

### Characteristics of Mini-Batch GD

**Advantages**:
✅ Faster than BGD (more frequent updates)
✅ More stable than SGD (averaging over batch)
✅ Efficient on GPUs (parallel computation)
✅ Good balance between speed and stability
✅ Most commonly used in practice!

**Disadvantages**:
❌ Still has some noise (less than SGD)
❌ Need to choose batch size (hyperparameter)

**Choosing Batch Size**:
- **Small (8-32)**: Faster updates, more noise, better generalization
- **Medium (32-128)**: Good balance (most common)
- **Large (256-512)**: More stable, slower per epoch
- **Powers of 2**: Optimal for GPU/CPU memory

**Typical values**: 32, 64, 128, 256

### Vectorization Benefits

Mini-batch GD can leverage **vectorized operations** (SIMD, GPU):

```python
# Inefficient: Loop over examples
for i in range(batch_size):
    gradient += (y_pred[i] - y[i]) * X[i]

# Efficient: Vectorized
gradient = X_batch.T @ (y_pred - y_batch)  # Single matrix operation!
```

Modern hardware (GPUs) processes batches **much faster** than individual examples.

---

## Learning Rate Selection

### What is the Learning Rate?

The **learning rate** (α or η) controls the **step size** in gradient descent:

```
w ← w - α·∇J(w)
```

- **Too small**: Slow convergence
- **Too large**: Divergence or oscillation
- **Just right**: Fast, stable convergence

### Effect of Learning Rate

```
Cost J(w)
  │
  │  α too small: Slow convergence
  │  ╲
  │   ╲___________________
  │
  │  α just right: Fast convergence
  │  ╲___
  │     ╲___
  │
  │  α too large: Oscillation/Divergence
  │  ╲╱╲╱╲╱╲
  │   ╲  ╲  ╲  (getting worse!)
  │
  └──────────────────── Iterations
```

### Visualizing Different Learning Rates

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple 1D example: J(w) = w²
def cost(w):
    return w**2

def gradient(w):
    return 2*w

# Test different learning rates
learning_rates = [0.1, 0.5, 0.9, 1.1]
num_iterations = 20
w_init = 2.0

plt.figure(figsize=(12, 8))

for i, alpha in enumerate(learning_rates, 1):
    w_history = [w_init]
    w = w_init
    
    for _ in range(num_iterations):
        w = w - alpha * gradient(w)
        w_history.append(w)
    
    plt.subplot(2, 2, i)
    iterations = range(len(w_history))
    plt.plot(iterations, w_history, 'bo-')
    plt.axhline(y=0, color='r', linestyle='--', label='Optimum')
    plt.xlabel('Iteration')
    plt.ylabel('w')
    plt.title(f'Learning Rate α = {alpha}')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
```

### Methods for Choosing Learning Rate

#### 1. Grid Search

Try multiple values, pick the best:

```python
alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
best_alpha = None
best_cost = float('inf')

for alpha in alphas:
    w, J_hist = gradient_descent(X, y, alpha, num_iterations=100)
    final_cost = J_hist[-1]
    
    if final_cost < best_cost:
        best_cost = final_cost
        best_alpha = alpha

print(f"Best learning rate: {best_alpha}")
```

#### 2. Learning Rate Range Test

Systematically increase learning rate and plot:

```python
def learning_rate_range_test(X, y, alpha_min=1e-5, alpha_max=1.0, num_steps=100):
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num_steps)
    costs = []
    
    for alpha in alphas:
        w, J_hist = gradient_descent(X, y, alpha, num_iterations=10)
        costs.append(J_hist[-1])
    
    plt.figure(figsize=(10, 5))
    plt.plot(alphas, costs)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Cost')
    plt.title('Learning Rate Range Test')
    plt.grid(True)
    plt.show()
    
    # Find best
    best_idx = np.argmin(costs)
    return alphas[best_idx]
```

#### 3. Adaptive Learning Rates

**Learning rate schedule**: Decrease over time

**Time-based decay**:
```python
alpha_t = alpha_0 / (1 + decay_rate * t)
```

**Step decay**:
```python
alpha_t = alpha_0 * drop_factor^floor(t / epochs_drop)
```

**Exponential decay**:
```python
alpha_t = alpha_0 * exp(-decay_rate * t)
```

**Example implementation**:
```python
def gd_with_schedule(X, y, alpha_0=0.1, num_epochs=100):
    m, n = X.shape
    w = np.zeros((n, 1))
    
    for epoch in range(num_epochs):
        # Time-based decay
        alpha = alpha_0 / (1 + 0.01 * epoch)
        
        # Update
        y_pred = X @ w
        gradient = (1/m) * X.T @ (y_pred - y)
        w = w - alpha * gradient
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: α = {alpha:.4f}")
    
    return w
```

### Rule of Thumb

For well-scaled data (mean=0, std=1):

**Starting values to try**:
- Start with α = 0.01
- If too slow, try α = 0.1
- If diverging, try α = 0.001

**Typical range**: 0.001 to 1.0

### Feature Scaling Impact

**Problem**: Features on different scales affect learning rate effectiveness.

```
Feature 1: [0, 1]      (small scale)
Feature 2: [0, 1000]   (large scale)
```

Gradient for feature 2 will be much larger → need very small α → slow convergence!

**Solution**: Standardize features

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now can use larger learning rate!
```

**After scaling**: Features have mean=0, std=1 → gradients similar magnitude.

---

## Convergence Analysis

### What is Convergence?

**Convergence** means the algorithm has reached (or is very close to) the minimum.

**Indicators**:
1. Cost stops decreasing significantly
2. Gradient magnitude becomes very small
3. Weights stop changing

### Convergence Criteria

#### 1. Cost Change Threshold

Stop when cost changes by less than ε:

```python
if abs(J_new - J_old) < epsilon:
    print("Converged!")
    break
```

**Example**:
```python
def gd_with_convergence(X, y, alpha=0.1, epsilon=1e-6, max_iter=10000):
    m, n = X.shape
    w = np.zeros((n, 1))
    J_old = float('inf')
    
    for iteration in range(max_iter):
        y_pred = X @ w
        gradient = (1/m) * X.T @ (y_pred - y)
        w = w - alpha * gradient
        
        # Check convergence
        J_new = (1/(2*m)) * np.sum((y_pred - y)**2)
        
        if abs(J_new - J_old) < epsilon:
            print(f"Converged at iteration {iteration}")
            break
        
        J_old = J_new
    
    return w, iteration
```

#### 2. Gradient Magnitude Threshold

Stop when gradient norm is small:

```python
if np.linalg.norm(gradient) < epsilon:
    print("Gradient sufficiently small")
    break
```

#### 3. Weight Change Threshold

Stop when weights stop changing:

```python
delta_w = alpha * gradient
if np.linalg.norm(delta_w) < epsilon:
    print("Weights converged")
    break
```

### Convergence Rate

**Definition**: How fast the algorithm approaches the minimum.

For **convex** functions (like linear regression):

**Linear convergence**: 
```
|w_t - w*| ≤ c^t |w_0 - w*|
```
Where c < 1 (exponentially decreasing error).

**Factors affecting convergence speed**:
1. **Learning rate**: Larger → faster (but risk divergence)
2. **Condition number** of XᵀX: Smaller → faster
3. **Feature scaling**: Scaled → faster
4. **Initialization**: Closer to optimum → faster

### Monitoring Convergence

**Always plot the cost curve**:

```python
plt.plot(J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost J(w)')
plt.title('Convergence Curve')
plt.yscale('log')  # Log scale to see small changes
plt.grid(True)
plt.show()
```

**Good convergence**:
```
Cost
  │
  │ ╲
  │  ╲___
  │     ╲_______________  (smooth decrease, plateaus)
  │
  └──────────────────── Iterations
```

**Bad convergence**:
```
Cost
  │
  │ ╲╱╲╱╲╱╲
  │  ╲  ╲  ╲  (oscillating → learning rate too large)
  │   ╲  ╲  ╲
  │    ╲  ╲  ╲
  └──────────────────── Iterations
```

### Early Stopping

For regularization, stop before full convergence:

```python
def gd_with_validation(X_train, y_train, X_val, y_val, alpha=0.1, patience=10):
    """Stop if validation error doesn't improve for 'patience' epochs"""
    best_val_error = float('inf')
    epochs_without_improvement = 0
    w = np.zeros((X_train.shape[1], 1))
    
    for epoch in range(1000):
        # Train
        y_pred = X_train @ w
        gradient = (1/len(X_train)) * X_train.T @ (y_pred - y_train)
        w = w - alpha * gradient
        
        # Validate
        val_pred = X_val @ w
        val_error = np.mean((val_pred - y_val)**2)
        
        if val_error < best_val_error:
            best_val_error = val_error
            best_w = w.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            return best_w
    
    return w
```

### Theoretical Analysis

For **strongly convex** functions with **Lipschitz continuous** gradient:

**Batch GD convergence rate**:
```
J(w_t) - J(w*) ≤ (1 - μ/L)^t [J(w_0) - J(w*)]
```

Where:
- μ = strong convexity constant
- L = Lipschitz constant
- Condition number κ = L/μ

**Better conditioning** (κ small) → faster convergence!

---

## Practical Considerations

### 1. Feature Scaling

**Always scale features** before gradient descent:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why?** Ensures gradients have similar magnitudes → can use larger learning rate.

### 2. Initialization

**Common strategies**:

**Zero initialization** (simple):
```python
w = np.zeros((n, 1))
```

**Small random** (breaks symmetry for neural networks):
```python
w = np.random.randn(n, 1) * 0.01
```

**He initialization** (for deeper networks):
```python
w = np.random.randn(n, 1) * np.sqrt(2/n)
```

For **linear regression**, zeros work fine!

### 3. Debugging Gradient Descent

**Check 1: Cost should decrease**
```python
assert J_history[-1] < J_history[0], "Cost not decreasing!"
```

**Check 2: Numerical gradient check**
```python
def numerical_gradient(X, y, w, epsilon=1e-5):
    """Compute gradient numerically (for debugging)"""
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += epsilon
        w_minus[i] -= epsilon
        
        J_plus = (1/(2*len(X))) * np.sum((X @ w_plus - y)**2)
        J_minus = (1/(2*len(X))) * np.sum((X @ w_minus - y)**2)
        
        grad[i] = (J_plus - J_minus) / (2 * epsilon)
    
    return grad

# Compare
analytical_grad = (1/m) * X.T @ (X @ w - y)
numerical_grad = numerical_gradient(X, y, w)

difference = np.linalg.norm(analytical_grad - numerical_grad)
print(f"Gradient difference: {difference:.2e}")  # Should be < 1e-7
```

### 4. Vectorization

**Slow** (loop):
```python
for i in range(m):
    gradient += (y_pred[i] - y[i]) * X[i]
```

**Fast** (vectorized):
```python
gradient = X.T @ (y_pred - y)
```

Vectorization can be **100-1000x faster**!

### 5. Batch Size Selection

**Guidelines**:
- Start with 32 or 64
- Larger batches → more stable, slower per epoch
- Smaller batches → faster per epoch, more noise, better generalization
- Powers of 2 work best with GPUs (32, 64, 128, 256)

### 6. When to Use Which Method?

| Dataset Size | Method | Why |
|--------------|--------|-----|
| m < 10,000 | Batch GD or Closed-form | Fast enough, stable |
| 10k < m < 1M | Mini-batch GD | Good balance |
| m > 1M | Mini-batch GD or SGD | Only feasible option |
| Streaming | SGD | Online learning |

---

## Comparison of Methods

### Summary Table

| Method | Batch Size | Updates/Epoch | Speed | Stability | Memory |
|--------|------------|---------------|-------|-----------|--------|
| **Batch GD** | m (all) | 1 | Slow | High | High |
| **SGD** | 1 | m | Fast | Low | Low |
| **Mini-batch** | b (e.g., 32) | m/b | Medium | Medium | Medium |

### Convergence Comparison

```
Cost
  │
  │ Batch GD
  │ ╲_______________  (smooth, slow)
  │
  │ Mini-batch GD
  │ ╲╲╲___________    (some noise, good balance)
  │
  │ SGD
  │ ╲╱╲╱╲╱╲╱╲____     (noisy, fast to good region)
  │
  └──────────────────── Iterations
```

### Complete Comparison Code

```python
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Generate larger dataset
np.random.seed(42)
m = 1000
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), X]

# BGD
start = time()
w_bgd, J_bgd = batch_gradient_descent(X_b, y, alpha=0.1, num_iterations=50)
time_bgd = time() - start

# SGD
start = time()
w_sgd, J_sgd = stochastic_gradient_descent(X_b, y, alpha=0.01, num_epochs=50)
time_sgd = time() - start

# Mini-batch
start = time()
w_mini, J_mini = mini_batch_gd(X_b, y, alpha=0.1, num_epochs=50, batch_size=32)
time_mini = time() - start

# Closed-form (for reference)
start = time()
w_closed = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
time_closed = time() - start

# Print results
print("Method           Time(s)  Final Cost  w0      w1")
print("-" * 60)
print(f"Closed-form:     {time_closed:.6f}  N/A         {w_closed[0][0]:.4f}  {w_closed[1][0]:.4f}")
print(f"Batch GD:        {time_bgd:.6f}  {J_bgd[-1]:.6f}  {w_bgd[0][0]:.4f}  {w_bgd[1][0]:.4f}")
print(f"Mini-batch GD:   {time_mini:.6f}  {J_mini[-1]:.6f}  {w_mini[0][0]:.4f}  {w_mini[1][0]:.4f}")
print(f"SGD:             {time_sgd:.6f}  {J_sgd[-1]:.6f}  {w_sgd[0][0]:.4f}  {w_sgd[1][0]:.4f}")

# Plot convergence
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(J_bgd, label='Batch GD', linewidth=2)
plt.plot(J_mini, label='Mini-batch GD', linewidth=2)
plt.plot(J_sgd, label='SGD', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Cost J(w)')
plt.title('Convergence Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(J_bgd, label='Batch GD', linewidth=2)
plt.plot(J_mini, label='Mini-batch GD', linewidth=2)
plt.plot(J_sgd, label='SGD', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Cost J(w)')
plt.title('Convergence (Log Scale)')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Advanced Optimization

### Beyond Basic Gradient Descent

While basic GD works, modern optimizers often perform better:

#### 1. Momentum

Accumulates gradients over time to smooth updates:

```python
v = 0.9 * v + alpha * gradient  # Momentum term
w = w - v
```

**Benefits**: Accelerates in relevant directions, dampens oscillations.

#### 2. AdaGrad

Adapts learning rate per parameter:

```python
G = G + gradient**2  # Accumulate squared gradients
w = w - alpha / sqrt(G + epsilon) * gradient
```

**Benefits**: Larger updates for infrequent features, smaller for frequent ones.

#### 3. RMSprop

Exponentially weighted moving average of squared gradients:

```python
G = beta * G + (1 - beta) * gradient**2
w = w - alpha / sqrt(G + epsilon) * gradient
```

**Benefits**: Fixes AdaGrad's aggressive learning rate decay.

#### 4. Adam (Most Popular!)

Combines momentum and RMSprop:

```python
m = beta1 * m + (1 - beta1) * gradient      # First moment
v = beta2 * v + (1 - beta2) * gradient**2   # Second moment
m_hat = m / (1 - beta1**t)  # Bias correction
v_hat = v / (1 - beta2**t)
w = w - alpha * m_hat / (sqrt(v_hat) + epsilon)
```

**Default hyperparameters**:
- alpha = 0.001
- beta1 = 0.9
- beta2 = 0.999
- epsilon = 1e-8

### Using Scikit-Learn's SGD

```python
from sklearn.linear_model import SGDRegressor

# Create model
model = SGDRegressor(
    max_iter=1000,
    learning_rate='optimal',  # Automatically determined
    eta0=0.01,  # Initial learning rate
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

---

## Summary and Next Steps

### What We Learned

1. **Batch Gradient Descent**
   - Uses all examples per iteration
   - Stable but slow
   - O(mn) per iteration

2. **Stochastic Gradient Descent**
   - Uses one example per iteration
   - Fast but noisy
   - Good for large datasets

3. **Mini-Batch Gradient Descent**
   - Uses small batches (e.g., 32 examples)
   - Best of both worlds
   - **Most commonly used**!

4. **Learning Rate**
   - Critical hyperparameter
   - Typical values: 0.001 to 0.1
   - Use schedules for better convergence

5. **Convergence**
   - Monitor cost curve
   - Use stopping criteria
   - Feature scaling helps!

### Key Formulas

**Gradient**:
```
∇J(w) = (1/m) Xᵀ(Xw - y)
```

**Update rule**:
```
w ← w - α·∇J(w)
```

**For mini-batch**:
```
∇J(w) = (1/b) X_batchᵀ(X_batch·w - y_batch)
```

### Key Takeaways

✅ **Gradient descent** is iterative optimization  
✅ **Mini-batch** is usually best in practice  
✅ **Feature scaling** is crucial  
✅ **Learning rate** needs tuning  
✅ **Monitor convergence** with cost curves  
✅ **Vectorization** makes it fast  

### When to Use What?

| Scenario | Method |
|----------|--------|
| Small dataset (< 10k) | Closed-form or Batch GD |
| Large dataset (> 100k) | Mini-batch GD |
| Streaming data | SGD |
| Non-convex (neural nets) | Mini-batch with Adam |
| Need exact solution | Closed-form |

### Prerequisites Checklist

Before moving on:
- [ ] Understand gradient computation
- [ ] Know difference between BGD, SGD, mini-batch
- [ ] Can implement basic gradient descent
- [ ] Understand learning rate impact
- [ ] Know how to check convergence
- [ ] Understand feature scaling importance

### Next Steps

1. **Cost_Functions_and_Loss.md**
   - Why MSE?
   - Alternative loss functions
   - Probabilistic interpretation

2. **Logistic_Regression.md**
   - Classification with linear models
   - Cross-entropy loss
   - **Must use gradient descent** (no closed-form!)

3. **Regularization_L1_L2.md**
   - Ridge and Lasso
   - Gradient descent with regularization
   - Feature selection

### Practice Exercises

1. Implement all three variants from scratch
2. Visualize convergence for different learning rates
3. Compare wall-clock time for different methods
4. Implement learning rate decay
5. Try on different dataset sizes

### Additional Resources

**Papers**:
- "An overview of gradient descent optimization algorithms" (Sebastian Ruder)
- Original SGD papers from 1950s

**Books**:
- "Deep Learning" by Goodfellow et al. - Chapter 8
- "Optimization for Machine Learning" by Sra et al.

**Online**:
- CS231n: Optimization notes
- Distill.pub: "Why Momentum Really Works"
- Sebastian Ruder's blog on gradient descent

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║           GRADIENT DESCENT QUICK REFERENCE                     ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  GRADIENT:  ∇J(w) = (1/m) Xᵀ(Xw - y)                          ║
║                                                                ║
║  UPDATE:    w ← w - α·∇J(w)                                    ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  BATCH GRADIENT DESCENT                                        ║
╠════════════════════════════════════════════════════════════════╣
║  Use ALL m examples per iteration                              ║
║  ∇J(w) = (1/m) Xᵀ(Xw - y)                                      ║
║                                                                ║
║  Pros: Stable, smooth convergence                              ║
║  Cons: Slow for large m                                        ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  STOCHASTIC GRADIENT DESCENT                                   ║
╠════════════════════════════════════════════════════════════════╣
║  Use ONE example per iteration                                 ║
║  ∇Jᵢ(w) = (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)·x⁽ⁱ⁾                                    ║
║                                                                ║
║  Pros: Fast, memory efficient, online learning                 ║
║  Cons: Noisy, doesn't fully converge                           ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  MINI-BATCH GRADIENT DESCENT                                   ║
╠════════════════════════════════════════════════════════════════╣
║  Use BATCH of b examples per iteration                         ║
║  ∇J(w) = (1/b) X_bᵀ(X_b·w - y_b)                               ║
║                                                                ║
║  Typical batch sizes: 32, 64, 128, 256                         ║
║  Pros: Fast + stable, GPU-friendly                             ║
║  Cons: Still has some noise                                    ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  LEARNING RATE                                                 ║
╠════════════════════════════════════════════════════════════════╣
║  Typical range: 0.001 to 1.0                                   ║
║  Start with: 0.01 or 0.1                                       ║
║                                                                ║
║  Too small → slow convergence                                  ║
║  Too large → divergence/oscillation                            ║
║  Just right → fast, stable convergence                         ║
║                                                                ║
║  Decay schedules:                                              ║
║    α_t = α₀ / (1 + decay * t)        [time-based]             ║
║    α_t = α₀ * 0.5^(t / drop)         [step]                   ║
║    α_t = α₀ * exp(-decay * t)        [exponential]            ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  CONVERGENCE                                                   ║
╠════════════════════════════════════════════════════════════════╣
║  Stop when:                                                    ║
║    |J_new - J_old| < ε                                         ║
║    ||∇J(w)|| < ε                                               ║
║    ||Δw|| < ε                                                  ║
║                                                                ║
║  Always monitor cost curve!                                    ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  BEST PRACTICES                                                ║
╠════════════════════════════════════════════════════════════════╣
║  1. Scale features (StandardScaler)                            ║
║  2. Use mini-batch (b=32 or 64)                                ║
║  3. Monitor convergence curve                                  ║
║  4. Try multiple learning rates                                ║
║  5. Vectorize operations                                       ║
║  6. Use learning rate decay                                    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Ready to move on?** Next up: explore different **cost functions** and their connection to probability!

---

*End of Gradient_Descent_for_Linear_Models.md*