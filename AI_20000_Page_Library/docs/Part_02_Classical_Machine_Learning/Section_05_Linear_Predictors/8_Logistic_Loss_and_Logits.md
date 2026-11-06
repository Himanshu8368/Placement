# Logistic Loss and Logits

## Table of Contents
1. [Introduction](#introduction)
2. [Cross-Entropy Loss](#cross-entropy-loss)
3. [Log-Odds and Logits](#log-odds-and-logits)
4. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
5. [Gradient Derivation](#gradient-derivation)
6. [Information Theory Connection](#information-theory-connection)
7. [Properties and Intuition](#properties-and-intuition)
8. [Multi-Class Extension](#multi-class-extension)
9. [Practical Considerations](#practical-considerations)
10. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction

### What Are Logits?

**Logits** are the raw, unnormalized predictions before applying the sigmoid function.

```
z = wᵀx          ← Logit (log-odds)
ŷ = σ(z)         ← Probability
```

**In neural networks**: The term "logits" refers to the final layer outputs before softmax/sigmoid.

### Why Study This?

Understanding logits and cross-entropy loss deeply helps with:
- **Designing neural networks**: Final layer design
- **Numerical stability**: Avoiding overflow/underflow
- **Theoretical understanding**: Connection to probability theory
- **Debugging**: Understanding what the model is really learning

### Overview

This chapter dives deep into:
1. **Mathematical foundation** of cross-entropy
2. **Probabilistic interpretation** via maximum likelihood
3. **Information theory** perspective
4. **Gradient computation** in detail

---

## Cross-Entropy Loss

### Definition

For binary classification, the **cross-entropy loss** (also called **log loss** or **binary cross-entropy**) is:

**For single example**:
```
L(ŷ, y) = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

Where:
- y ∈ {0, 1} = true label
- ŷ ∈ (0, 1) = predicted probability

**For all examples** (cost function):
```
J(w) = -(1/m) Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-ŷ⁽ⁱ⁾)]
```

### Case-by-Case Analysis

**When y = 1** (true class is 1):
```
L = -log(ŷ)
```

**Behavior**:
```
ŷ → 1:  L → 0      (Good prediction, low loss)
ŷ → 0:  L → ∞      (Bad prediction, high loss)
ŷ = 1:  L = 0      (Perfect prediction)
```

**When y = 0** (true class is 0):
```
L = -log(1-ŷ)
```

**Behavior**:
```
ŷ → 0:  L → 0      (Good prediction, low loss)
ŷ → 1:  L → ∞      (Bad prediction, high loss)
ŷ = 0:  L = 0      (Perfect prediction)
```

### Visual Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Predicted probabilities
y_pred = np.linspace(0.001, 0.999, 1000)

# Loss when y=1
loss_y1 = -np.log(y_pred)

# Loss when y=0
loss_y0 = -np.log(1 - y_pred)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Loss for y=1
ax = axes[0]
ax.plot(y_pred, loss_y1, linewidth=3, color='blue')
ax.set_xlabel('Predicted Probability ŷ', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_title('Cross-Entropy Loss when y=1', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 5])
ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='ŷ=0.5')
ax.legend()

# Add annotations
ax.annotate('Perfect prediction\nL=0', xy=(0.99, 0), xytext=(0.8, 1),
            fontsize=12, arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.annotate('Wrong prediction\nL→∞', xy=(0.01, 4.5), xytext=(0.2, 3.5),
            fontsize=12, arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Plot 2: Loss for y=0
ax = axes[1]
ax.plot(y_pred, loss_y0, linewidth=3, color='red')
ax.set_xlabel('Predicted Probability ŷ', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_title('Cross-Entropy Loss when y=0', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 5])
ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='ŷ=0.5')
ax.legend()

# Add annotations
ax.annotate('Perfect prediction\nL=0', xy=(0.01, 0), xytext=(0.2, 1),
            fontsize=12, arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.annotate('Wrong prediction\nL→∞', xy=(0.99, 4.5), xytext=(0.8, 3.5),
            fontsize=12, arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Plot 3: Combined view
ax = axes[2]
ax.plot(y_pred, loss_y1, linewidth=3, color='blue', label='y=1: -log(ŷ)')
ax.plot(y_pred, loss_y0, linewidth=3, color='red', label='y=0: -log(1-ŷ)')
ax.set_xlabel('Predicted Probability ŷ', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_title('Cross-Entropy Loss (Both Cases)', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 5])
ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='ŷ=0.5')
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()
```

### Numerical Examples

**Example 1**: Good prediction
```
True: y = 1
Predicted: ŷ = 0.9

L = -log(0.9) = 0.105
```

**Example 2**: Bad prediction
```
True: y = 1
Predicted: ŷ = 0.1

L = -log(0.1) = 2.303
```

**Example 3**: Uncertain prediction
```
True: y = 1
Predicted: ŷ = 0.5

L = -log(0.5) = 0.693
```

**Example 4**: Perfect prediction
```
True: y = 1
Predicted: ŷ = 1.0

L = -log(1.0) = 0
```

### Comparison with Other Losses

```python
import numpy as np
import matplotlib.pyplot as plt

# Error range
error = np.linspace(-1, 1, 1000)

# Different losses (assuming y=1, so error = ŷ - 1)
y_pred = 1 + error  # ŷ ranges from 0 to 2

# Clip to valid range for cross-entropy
y_pred_valid = np.clip(y_pred, 0.001, 0.999)

# Losses
squared_error = error ** 2
absolute_error = np.abs(error)
cross_entropy = -np.log(y_pred_valid)

plt.figure(figsize=(12, 6))

plt.plot(y_pred_valid, squared_error, linewidth=3, label='Squared Error')
plt.plot(y_pred_valid, absolute_error, linewidth=3, label='Absolute Error')
plt.plot(y_pred_valid, cross_entropy, linewidth=3, label='Cross-Entropy')

plt.xlabel('Predicted Probability ŷ (when y=1)', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss Functions Comparison (y=1)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim([0, 5])
plt.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
plt.show()
```

### Why Cross-Entropy?

**Reasons**:
1. **Convex**: Guarantees global minimum
2. **Smooth**: Differentiable everywhere
3. **Probabilistic**: Natural for probability predictions
4. **Penalizes confidence**: Wrong confident predictions get high loss
5. **Information theoretic**: Measures surprise/information
6. **Maximum likelihood**: Derived from probability theory

---

## Log-Odds and Logits

### What Are Odds?

**Odds** express the ratio of probability of success to failure:

```
Odds = P(event) / P(not event)
     = P(y=1) / P(y=0)
     = p / (1-p)
```

**Examples**:
```
P(y=1) = 0.5  →  Odds = 0.5/0.5 = 1    (1:1, even odds)
P(y=1) = 0.75 →  Odds = 0.75/0.25 = 3  (3:1, favorable)
P(y=1) = 0.25 →  Odds = 0.25/0.75 = 1/3 (1:3, unfavorable)
P(y=1) = 0.9  →  Odds = 0.9/0.1 = 9    (9:1, very favorable)
```

### Log-Odds (Logits)

The **log-odds** (or **logit**) is the natural logarithm of the odds:

```
Logit = log(Odds)
      = log(P(y=1) / P(y=0))
      = log(p / (1-p))
```

### Connection to Logistic Regression

In logistic regression:
```
P(y=1|x) = σ(wᵀx) = 1 / (1 + e^(-wᵀx))
```

Computing the odds:
```
Odds = P(y=1|x) / P(y=0|x)
     = σ(wᵀx) / (1 - σ(wᵀx))
```

**Key derivation**:
```
Let p = σ(z) = 1 / (1 + e^(-z))

Then: 1 - p = 1 - 1/(1 + e^(-z))
            = (1 + e^(-z) - 1) / (1 + e^(-z))
            = e^(-z) / (1 + e^(-z))

So: p / (1-p) = [1/(1 + e^(-z))] / [e^(-z)/(1 + e^(-z))]
              = 1 / e^(-z)
              = e^z
```

**Therefore**:
```
Odds = e^(wᵀx)
```

Taking logarithm:
```
log(Odds) = wᵀx
```

**Beautiful result**: The logit (log-odds) is exactly the linear combination wᵀx!

### Inverse Relationship

Given log-odds (logit) z, we can recover probability:

```
z = log(p/(1-p))

e^z = p/(1-p)

e^z(1-p) = p

e^z - e^z·p = p

e^z = p(1 + e^z)

p = e^z / (1 + e^z)
  = 1 / (1 + e^(-z))
  = σ(z)
```

**So**: Sigmoid converts logits to probabilities!

### Visual Relationship

```python
import numpy as np
import matplotlib.pyplot as plt

# Probability range
p = np.linspace(0.01, 0.99, 1000)

# Odds
odds = p / (1 - p)

# Log-odds (logits)
logits = np.log(odds)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Probability vs Odds
ax = axes[0]
ax.plot(p, odds, linewidth=3, color='blue')
ax.set_xlabel('Probability p', fontsize=14)
ax.set_ylabel('Odds', fontsize=14)
ax.set_title('Probability → Odds', fontsize=16)
ax.grid(True, alpha=0.3)
ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Even odds')
ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
ax.legend()

# Plot 2: Probability vs Log-Odds
ax = axes[1]
ax.plot(p, logits, linewidth=3, color='green')
ax.set_xlabel('Probability p', fontsize=14)
ax.set_ylabel('Log-Odds (Logit)', fontsize=14)
ax.set_title('Probability → Log-Odds', fontsize=16)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Even odds (log=0)')
ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
ax.legend()

# Plot 3: Log-Odds vs Probability (sigmoid)
ax = axes[2]
z = np.linspace(-5, 5, 1000)
p_from_z = 1 / (1 + np.exp(-z))
ax.plot(z, p_from_z, linewidth=3, color='purple')
ax.set_xlabel('Log-Odds (Logit) z', fontsize=14)
ax.set_ylabel('Probability σ(z)', fontsize=14)
ax.set_title('Log-Odds → Probability (Sigmoid)', fontsize=16)
ax.grid(True, alpha=0.3)
ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

### Interpretation Table

| Probability | Odds | Log-Odds (Logit) | Interpretation |
|-------------|------|------------------|----------------|
| 0.01 | 0.01 | -4.595 | Very unlikely |
| 0.1 | 0.111 | -2.197 | Unlikely |
| 0.25 | 0.333 | -1.099 | Less likely |
| 0.5 | 1.0 | 0.0 | Even odds |
| 0.75 | 3.0 | 1.099 | More likely |
| 0.9 | 9.0 | 2.197 | Likely |
| 0.99 | 99.0 | 4.595 | Very likely |

### Why Use Logits?

**Advantages of working with logits**:

1. **Unbounded**: z ∈ (-∞, +∞) vs p ∈ (0, 1)
2. **Linear modeling**: Can use linear regression techniques
3. **Symmetric**: Equal changes in z have equal effects on log-odds
4. **Additive**: Effects combine additively
5. **Numerical stability**: Avoid probability underflow/overflow

**Example**: Adding features
```
Feature 1 contributes: +1.5 to logit
Feature 2 contributes: -0.8 to logit
Total logit: 0.7
Probability: σ(0.7) = 0.668
```

---

## Maximum Likelihood Estimation

### Probabilistic Model

We model the probability of label y given features x:

```
P(y|x; w) = { σ(wᵀx)       if y = 1
            { 1 - σ(wᵀx)   if y = 0
```

**Compact form**:
```
P(y|x; w) = σ(wᵀx)^y · [1 - σ(wᵀx)]^(1-y)
```

**Verification**:
- If y=1: P = σ(wᵀx)¹ · [1-σ(wᵀx)]⁰ = σ(wᵀx) ✓
- If y=0: P = σ(wᵀx)⁰ · [1-σ(wᵀx)]¹ = 1-σ(wᵀx) ✓

### Likelihood Function

**For all examples** (assuming independence):

```
L(w) = P(y⁽¹⁾, y⁽²⁾, ..., y⁽ᵐ⁾ | X; w)
     = ∏ᵢ₌₁ᵐ P(y⁽ⁱ⁾ | x⁽ⁱ⁾; w)
     = ∏ᵢ₌₁ᵐ [σ(wᵀx⁽ⁱ⁾)]^y⁽ⁱ⁾ · [1 - σ(wᵀx⁽ⁱ⁾)]^(1-y⁽ⁱ⁾)
```

**Goal**: Find w that **maximizes** L(w)

### Log-Likelihood

Taking logarithm (easier to work with):

```
log L(w) = Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log(σ(wᵀx⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1 - σ(wᵀx⁽ⁱ⁾))]
```

**Maximizing log-likelihood** = **Minimizing negative log-likelihood**:

```
-log L(w) = -Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log(σ(wᵀx⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1 - σ(wᵀx⁽ⁱ⁾))]
```

Dividing by m:

```
J(w) = -(1/m) Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-ŷ⁽ⁱ⁾)]
```

**This is exactly the cross-entropy loss!**

### Key Insight

**Cross-entropy loss = Negative log-likelihood**

```
Minimizing cross-entropy ⟺ Maximizing likelihood
```

**Interpretation**: Finding the weights that make the observed data most probable!

### Step-by-Step Derivation

**Step 1**: Write down the probability model
```
P(y=1|x) = σ(wᵀx)
P(y=0|x) = 1 - σ(wᵀx)
```

**Step 2**: Express as single formula
```
P(y|x) = σ(wᵀx)^y · [1 - σ(wᵀx)]^(1-y)
```

**Step 3**: Likelihood for all examples
```
L(w) = ∏ᵢ₌₁ᵐ σ(wᵀx⁽ⁱ⁾)^y⁽ⁱ⁾ · [1 - σ(wᵀx⁽ⁱ⁾)]^(1-y⁽ⁱ⁾)
```

**Step 4**: Take logarithm
```
log L(w) = Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log σ(wᵀx⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1 - σ(wᵀx⁽ⁱ⁾))]
```

**Step 5**: Negate to get minimization objective
```
-log L(w) = -Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log ŷ⁽ⁱ⁾ + (1-y⁽ⁱ⁾) log(1-ŷ⁽ⁱ⁾)]
```

**Step 6**: Average over examples
```
J(w) = -(1/m) log L(w)
```

**Result**: Cross-entropy loss!

### Why Maximum Likelihood?

**Principle**: Choose parameters that make observed data most probable.

**Advantages**:
1. **Principled**: Based on probability theory
2. **Consistent**: Converges to true parameters as m → ∞
3. **Efficient**: Achieves Cramér-Rao lower bound
4. **General**: Works for many models

**Example interpretation**:
```
Observed data: 10 spam emails, 5 not spam
Model predicts: P(spam) = 0.7

Likelihood = 0.7¹⁰ × 0.3⁵ = 0.00017

If we adjust w to get P(spam) = 0.667:
Likelihood = 0.667¹⁰ × 0.333⁵ = 0.00028  ← Higher! Better!
```

---

## Gradient Derivation

### Goal

Derive the gradient of cross-entropy loss:

```
J(w) = -(1/m) Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-ŷ⁽ⁱ⁾)]
```

Where ŷ⁽ⁱ⁾ = σ(wᵀx⁽ⁱ⁾).

### For Single Example

**Loss**:
```
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

Where ŷ = σ(z) and z = wᵀx.

**Chain rule**:
```
∂L/∂wⱼ = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂wⱼ
```

### Step 1: Derivative with respect to ŷ

```
L = -[y log(ŷ) + (1-y) log(1-ŷ)]

∂L/∂ŷ = -[y/ŷ + (1-y)·(-1)/(1-ŷ)]
      = -[y/ŷ - (1-y)/(1-ŷ)]
      = -[y(1-ŷ) - (1-y)ŷ] / [ŷ(1-ŷ)]
      = -[y - yŷ - ŷ + yŷ] / [ŷ(1-ŷ)]
      = -(y - ŷ) / [ŷ(1-ŷ)]
```

### Step 2: Derivative of sigmoid

Recall: σ(z) = 1/(1 + e^(-z))

```
dσ/dz = σ(z) · (1 - σ(z))
```

So:
```
∂ŷ/∂z = ŷ(1-ŷ)
```

### Step 3: Derivative with respect to z

```
∂L/∂z = ∂L/∂ŷ · ∂ŷ/∂z
      = [-(y-ŷ) / (ŷ(1-ŷ))] · [ŷ(1-ŷ)]
      = -(y-ŷ)
      = ŷ - y
```

**Beautiful simplification!** The ŷ(1-ŷ) terms cancel.

### Step 4: Derivative with respect to wⱼ

```
z = w₀ + w₁x₁ + ... + wⱼxⱼ + ... + wₙxₙ

∂z/∂wⱼ = xⱼ
```

Therefore:
```
∂L/∂wⱼ = ∂L/∂z · ∂z/∂wⱼ
       = (ŷ - y) · xⱼ
```

### For All Examples

Average over all examples:

```
∂J/∂wⱼ = (1/m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) · x⁽ⁱ⁾ⱼ
```

**Vector form**:
```
∇J(w) = (1/m) Xᵀ(ŷ - y)
```

Where ŷ = σ(Xw).

### Complete Derivation Summary

```
Given:
  ŷ = σ(wᵀx)
  L = -[y log(ŷ) + (1-y) log(1-ŷ)]

Step 1: ∂L/∂ŷ = -(y-ŷ) / [ŷ(1-ŷ)]

Step 2: ∂ŷ/∂z = ŷ(1-ŷ)

Step 3: ∂L/∂z = ∂L/∂ŷ · ∂ŷ/∂z = ŷ - y

Step 4: ∂z/∂wⱼ = xⱼ

Step 5: ∂L/∂wⱼ = (ŷ - y) · xⱼ

Result: ∇J = (1/m) Xᵀ(ŷ - y)
```

### Comparison with Linear Regression

**Linear Regression**:
```
ŷ = wᵀx  (no sigmoid)
L = (ŷ - y)²
∇J = (1/m) Xᵀ(ŷ - y)  ← Same form!
```

**Logistic Regression**:
```
ŷ = σ(wᵀx)  (with sigmoid)
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
∇J = (1/m) Xᵀ(ŷ - y)  ← Same form!
```

**Key difference**: The ŷ includes sigmoid in logistic regression.

### Hessian (Second Derivative)

For completeness, the Hessian matrix:

```
H = ∂²J/∂w∂wᵀ = (1/m) Xᵀ diag(ŷ(1-ŷ)) X
```

Where diag(ŷ(1-ŷ)) is a diagonal matrix with ŷ⁽ⁱ⁾(1-ŷ⁽ⁱ⁾) on diagonal.

**Key property**: H is positive semi-definite → J is convex!

### Python Implementation

```python
import numpy as np

def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def cross_entropy_loss(y_true, y_pred):
    """Cross-entropy loss"""
    m = len(y_true)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def gradient_cross_entropy(X, y_true, y_pred):
    """
    Gradient of cross-entropy with respect to weights
    
    Args:
        X: Feature matrix (m × n) with bias
        y_true: True labels (m × 1)
        y_pred: Predicted probabilities (m × 1)
    
    Returns:
        Gradient vector (n × 1)
    """
    m = len(y_true)
    gradient = (1/m) * X.T @ (y_pred - y_true)
    return gradient

# Verify gradient numerically
def numerical_gradient(X, y, w, epsilon=1e-5):
    """Compute gradient numerically (for verification)"""
    grad = np.zeros_like(w)
    
    for i in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += epsilon
        w_minus[i] -= epsilon
        
        # Compute loss for w + epsilon
        y_pred_plus = sigmoid(X @ w_plus)
        loss_plus = cross_entropy_loss(y, y_pred_plus)
        
        # Compute loss for w - epsilon
        y_pred_minus = sigmoid(X @ w_minus)
        loss_minus = cross_entropy_loss(y, y_pred_minus)
        
        # Numerical gradient
        grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    return grad

# Test
np.random.seed(42)
X = np.random.randn(10, 3)
X_b = np.c_[np.ones(10), X]  # Add bias
y = np.random.randint(0, 2, (10, 1))
w = np.random.randn(4, 1)

# Analytical gradient
y_pred = sigmoid(X_b @ w)
grad_analytical = gradient_cross_entropy(X_b, y, y_pred)

# Numerical gradient
grad_numerical = numerical_gradient(X_b, y, w)

print("Analytical gradient:")
print(grad_analytical.ravel())
print("\nNumerical gradient:")
print(grad_numerical.ravel())
print("\nDifference:")
print(np.abs(grad_analytical.ravel() - grad_numerical.ravel()))
print(f"\nMax difference: {np.max(np.abs(grad_analytical - grad_numerical)):.2e}")
```

**Output** (should be very small difference):
```
Max difference: 1.23e-08
```

---

## Information Theory Connection

### Entropy

**Entropy** measures uncertainty or information content:

```
H(P) = -Σᵢ P(xᵢ) log P(xᵢ)
```

For binary variable:
```
H(p) = -[p log(p) + (1-p) log(1-p)]
```

**Interpretation**: Average number of bits needed to encode outcomes.

### Cross-Entropy

**Cross-entropy** measures the average number of bits needed to encode data from distribution P using distribution Q:

```
H(P, Q) = -Σᵢ P(xᵢ) log Q(xᵢ)
```

For binary:
```
H(p, q) = -[p log(q) + (1-p) log(1-q)]
```

### KL Divergence

**Kullback-Leibler divergence** measures how different Q is from P:

```
KL(P || Q) = H(P, Q) - H(P)
           = Σᵢ P(xᵢ) log[P(xᵢ) / Q(xᵢ)]
```

**Property**: KL(P || Q) ≥ 0, with equality iff P = Q.

### Connection to Loss Function

In classification:
- **P** = true distribution (labels)
- **Q** = predicted distribution (model output)

For single example with true label y:
```
P = (y, 1-y)      ← True distribution
Q = (ŷ, 1-ŷ)      ← Predicted distribution

Cross-entropy = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

**This is exactly our loss function!**

### Minimizing Cross-Entropy

Minimizing cross-entropy = Minimizing KL divergence (since H(P) is constant):

```
min H(P, Q) = min [H(P) + KL(P || Q)]
            = min KL(P || Q)
```

**Interpretation**: Make predicted distribution as close as possible to true distribution!

### Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# True probability p
p = 0.7

# Predicted probabilities q
q = np.linspace(0.01, 0.99, 1000)

# Entropy of true distribution (constant)
H_p = -p * np.log(p) - (1-p) * np.log(1-p)

# Cross-entropy H(P, Q)
H_pq = -p * np.log(q) - (1-p) * np.log(1-q)

# KL divergence
KL = H_pq - H_p

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.axhline(y=H_p, color='blue', linestyle='--', linewidth=2, label=f'H(P) = {H_p:.3f} (constant)')
plt.plot(q, H_pq, linewidth=3, color='red', label='H(P,Q) = Cross-Entropy')
plt.xlabel('Predicted Probability q', fontsize=14)
plt.ylabel('Entropy', fontsize=14)
plt.title(f'Entropy and Cross-Entropy (True p={p})', fontsize=16)
plt.axvline(x=p, color='green', linestyle='--', alpha=0.5, label='q=p (optimal)')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(q, KL, linewidth=3, color='purple')
plt.xlabel('Predicted Probability q', fontsize=14)
plt.ylabel('KL Divergence', fontsize=14)
plt.title(f'KL(P || Q) (True p={p})', fontsize=16)
plt.axvline(x=p, color='green', linestyle='--', alpha=0.5, label='q=p (minimum)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Properties and Intuition

### Key Properties

**1. Convexity**
```
J(w) is convex in w
→ Gradient descent finds global minimum
```

**2. Smoothness**
```
J(w) is continuously differentiable
→ Gradient descent works well
```

**3. Penalizes Confidence**
```
Wrong confident predictions → high loss
Wrong uncertain predictions → moderate loss
```

**4. Asymmetry**
```
ŷ=0.9, y=1: L = 0.105  (small loss)
ŷ=0.1, y=1: L = 2.303  (large loss)
```

### Intuitive Understanding

**Scenario 1**: Model is confident and correct
```
y = 1, ŷ = 0.99
L = -log(0.99) = 0.01  ← Very small loss, good!
```

**Scenario 2**: Model is confident and wrong
```
y = 1, ŷ = 0.01
L = -log(0.01) = 4.61  ← Very large loss, bad!
```

**Scenario 3**: Model is uncertain
```
y = 1, ŷ = 0.5
L = -log(0.5) = 0.69   ← Moderate loss
```

**Lesson**: The model learns to be confident when right, uncertain when unsure.

### Comparison Table

| True y | Predicted ŷ | Loss | Interpretation |
|--------|-------------|------|----------------|
| 1 | 1.0 | 0.00 | Perfect! |
| 1 | 0.9 | 0.11 | Good, confident |
| 1 | 0.7 | 0.36 | Okay, less confident |
| 1 | 0.5 | 0.69 | Uncertain |
| 1 | 0.3 | 1.20 | Bad, wrong direction |
| 1 | 0.1 | 2.30 | Very bad, confident wrong |
| 1 | 0.01 | 4.61 | Terrible! |

### Why It Works

**1. Gradient at wrong predictions is large**:
```
If y=1 and ŷ=0.1:
∂L/∂z = ŷ - y = 0.1 - 1 = -0.9  ← Large gradient
→ Strong push toward correct direction
```

**2. Gradient at correct predictions is small**:
```
If y=1 and ŷ=0.99:
∂L/∂z = ŷ - y = 0.99 - 1 = -0.01  ← Small gradient
→ Small adjustment needed
```

**3. Gradient is zero at perfect prediction**:
```
If y=1 and ŷ=1:
∂L/∂z = ŷ - y = 1 - 1 = 0  ← No gradient
→ No update needed
```

---

## Multi-Class Extension

### Softmax Function

For K classes, we use **softmax** instead of sigmoid:

```
ŷₖ = exp(zₖ) / Σⱼ exp(zⱼ)
```

Where zₖ = wₖᵀx are the logits for each class.

**Properties**:
- Σₖ ŷₖ = 1 (probabilities sum to 1)
- ŷₖ ∈ (0, 1) for all k
- Reduces to sigmoid for K=2

### Categorical Cross-Entropy

For multi-class:

```
L = -Σₖ yₖ log(ŷₖ)
```

Where yₖ is 1 for true class, 0 otherwise (one-hot encoding).

**Example** (3 classes):
```
True: y = [0, 1, 0]  (class 2)
Predicted: ŷ = [0.2, 0.7, 0.1]

L = -[0·log(0.2) + 1·log(0.7) + 0·log(0.1)]
  = -log(0.7)
  = 0.357
```

### Gradient for Multi-Class

**Beautiful result**: Same form!

```
∂L/∂zₖ = ŷₖ - yₖ
```

Where yₖ is one-hot encoded true label.

---

## Practical Considerations

### Numerical Stability

**Problem**: log(0) = -∞, exp(large) = overflow

**Solutions**:

**1. Clip probabilities**:
```python
epsilon = 1e-15
y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
loss = -np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
```

**2. LogSumExp trick** (for softmax):
```python
def stable_softmax(z):
    """Numerically stable softmax"""
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

**3. Compute loss from logits directly**:
```python
def binary_cross_entropy_from_logits(logits, y):
    """
    Compute BCE directly from logits (more stable)
    
    Uses: log(sigmoid(z)) = -log(1 + exp(-z))
    """
    # For numerical stability
    max_val = np.clip(logits, 0, None)
    loss = logits - logits * y + max_val + np.log(np.exp(-max_val) + np.exp((-logits - max_val)))
    return np.mean(loss)
```

### Regularization

Add penalty to prevent overfitting:

**L2 regularization**:
```
J(w) = CrossEntropy + (λ/2m) Σwⱼ²
```

**Gradient**:
```
∇J = (1/m) Xᵀ(ŷ - y) + (λ/m) w
```

### Implementation Tips

**1. Always use bias term**:
```python
X_with_bias = np.c_[np.ones(m), X]
```

**2. Feature scaling helps**:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**3. Check gradient numerically** during development:
```python
# Compare analytical vs numerical gradient
diff = np.abs(grad_analytical - grad_numerical)
assert np.max(diff) < 1e-5, "Gradient check failed!"
```

**4. Monitor training loss**:
```python
if loss > prev_loss:
    print("Warning: Loss increased!")
    # Maybe learning rate too high
```

---

## Summary and Next Steps

### What We Learned

1. **Cross-Entropy Loss**
   ```
   L = -[y log(ŷ) + (1-y) log(1-ŷ)]
   ```
   - Convex, smooth, proper for probabilities
   - Penalizes confident wrong predictions

2. **Logits (Log-Odds)**
   ```
   z = wᵀx = log(p/(1-p))
   ```
   - Unbounded linear score
   - Sigmoid converts to probability
   - Logit is the natural parameter space

3. **Maximum Likelihood**
   ```
   Cross-entropy = -log(Likelihood)
   ```
   - Principled probabilistic derivation
   - Minimizing loss = maximizing likelihood

4. **Gradient**
   ```
   ∇J = (1/m) Xᵀ(ŷ - y)
   ```
   - Beautiful simple form
   - Same as linear regression (but ŷ includes sigmoid)

5. **Information Theory**
   - Cross-entropy measures encoding efficiency
   - KL divergence measures distribution difference
   - Minimizing loss = matching distributions

### Key Formulas

```
Sigmoid:           σ(z) = 1/(1 + e^(-z))
Logit:            z = log(p/(1-p))
Cross-Entropy:    L = -[y log(ŷ) + (1-y) log(1-ŷ)]
Gradient:         ∇J = (1/m) Xᵀ(σ(Xw) - y)
Likelihood:       L(w) = ∏ᵢ σ(wᵀx⁽ⁱ⁾)^y⁽ⁱ⁾ [1-σ(wᵀx⁽ⁱ⁾)]^(1-y⁽ⁱ⁾)
```

### Key Insights

✅ **Logits** are the natural parameter space (unbounded, linear)  
✅ **Sigmoid** converts logits to probabilities  
✅ **Cross-entropy** derived from maximum likelihood  
✅ **Gradient** has beautiful simple form  
✅ **Information theory** provides deep understanding  
✅ **Numerical stability** matters in practice  

### Prerequisites Checklist

Before moving on:
- [ ] Understand logits vs probabilities
- [ ] Can derive cross-entropy gradient
- [ ] Know maximum likelihood connection
- [ ] Understand information theory perspective
- [ ] Aware of numerical stability issues
- [ ] Know multi-class extension (softmax)

### Next Steps

1. **Multiclass_Classification.md**
   - Softmax in detail
   - One-vs-Rest vs One-vs-One
   - Multi-label classification

2. **Regularization_L1_L2.md**
   - Ridge (L2) regularization
   - Lasso (L1) regularization
   - Elastic Net
   - Feature selection

3. **Model_Evaluation_Metrics.md**
   - Precision, Recall, F1
   - ROC curves, AUC
   - Calibration curves
   - Confusion matrices

### Practice Exercises

1. Derive the gradient from first principles
2. Verify gradient numerically
3. Implement from logits (numerical stability)
4. Plot loss landscapes
5. Implement softmax and categorical cross-entropy
6. Compare BCE vs MSE for classification

### Additional Resources

**Papers**:
- Shannon (1948): "A Mathematical Theory of Communication"
- Cox (1958): Logistic regression original paper

**Books**:
- "Pattern Recognition and Machine Learning" (Bishop) - Chapter 4
- "Elements of Statistical Learning" (Hastie) - Chapter 4
- "Information Theory, Inference, and Learning" (MacKay)

**Online**:
- Stanford CS229: Notes on logistic regression
- Colah's blog: Visual Information Theory
- 3Blue1Brown: Neural networks (loss functions)

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║          LOGISTIC LOSS AND LOGITS QUICK REFERENCE              ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  CROSS-ENTROPY LOSS                                            ║
║  ──────────────────                                            ║
║  L = -[y log(ŷ) + (1-y) log(1-ŷ)]                              ║
║                                                                ║
║  When y=1: L = -log(ŷ)        → 0 if ŷ→1, ∞ if ŷ→0            ║
║  When y=0: L = -log(1-ŷ)      → 0 if ŷ→0, ∞ if ŷ→1            ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  LOGITS (LOG-ODDS)                                             ║
╠════════════════════════════════════════════════════════════════╣
║  z = log(p/(1-p)) = log(odds)                                  ║
║                                                                ║
║  In logistic regression:                                       ║
║    z = wᵀx  (logit)                                            ║
║    p = σ(z) (probability)                                      ║
║                                                                ║
║  Key insight: Linear in logit space!                           ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  MAXIMUM LIKELIHOOD                                            ║
╠════════════════════════════════════════════════════════════════╣
║  Likelihood:                                                   ║
║    L(w) = ∏ᵢ σ(wᵀx⁽ⁱ⁾)^y⁽ⁱ⁾ [1-σ(wᵀx⁽ⁱ⁾)]^(1-y⁽ⁱ⁾)             ║
║                                                                ║
║  Log-likelihood:                                               ║
║    log L(w) = Σᵢ [y⁽ⁱ⁾ log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-ŷ⁽ⁱ⁾)]       ║
║                                                                ║
║  Cross-entropy = -log L(w)/m                                   ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  GRADIENT                                                      ║
╠════════════════════════════════════════════════════════════════╣
║  ∂L/∂ŷ = -(y-ŷ) / [ŷ(1-ŷ)]                                     ║
║  ∂ŷ/∂z = ŷ(1-ŷ)                                                ║
║  ∂L/∂z = ŷ - y           (beautiful cancellation!)             ║
║  ∂z/∂wⱼ = xⱼ                                                    ║
║                                                                ║
║  Result: ∇J = (1/m) Xᵀ(ŷ - y)                                  ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  INFORMATION THEORY                                            ║
╠════════════════════════════════════════════════════════════════╣
║  Entropy:        H(P) = -Σᵢ P(xᵢ) log P(xᵢ)                    ║
║  Cross-Entropy:  H(P,Q) = -Σᵢ P(xᵢ) log Q(xᵢ)                  ║
║  KL Divergence:  KL(P||Q) = H(P,Q) - H(P)                      ║
║                                                                ║
║  Minimizing H(P,Q) = Minimizing KL(P||Q)                       ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  MULTI-CLASS EXTENSION                                         ║
╠════════════════════════════════════════════════════════════════╣
║  Softmax:  ŷₖ = exp(zₖ) / Σⱼ exp(zⱼ)                           ║
║                                                                ║
║  Loss:     L = -Σₖ yₖ log(ŷₖ)                                  ║
║                                                                ║
║  Gradient: ∂L/∂zₖ = ŷₖ - yₖ     (same form!)                   ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  NUMERICAL STABILITY                                           ║
╠════════════════════════════════════════════════════════════════╣
║  • Clip probabilities: ŷ ∈ [ε, 1-ε]                            ║
║  • Compute from logits when possible                           ║
║  • Use LogSumExp trick for softmax                             ║
║  • Scale features before training                              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Ready to move on?** Next: Learn about **regularization** to prevent overfitting!

---

*End of Logistic_Loss_and_Logits.md*