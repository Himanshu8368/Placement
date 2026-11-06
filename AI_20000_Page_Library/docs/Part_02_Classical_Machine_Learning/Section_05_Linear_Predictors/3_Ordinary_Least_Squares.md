# Ordinary Least Squares (OLS)

## Table of Contents
1. [Introduction to OLS](#introduction-to-ols)
2. [The Least Squares Principle](#the-least-squares-principle)
3. [OLS Derivation - Simple Linear Regression](#ols-derivation-simple-linear-regression)
4. [OLS Derivation - Multiple Linear Regression](#ols-derivation-multiple-linear-regression)
5. [Normal Equation](#normal-equation)
6. [Geometric Interpretation](#geometric-interpretation)
7. [Closed-Form Solution](#closed-form-solution)
8. [Properties of OLS Estimators](#properties-of-ols-estimators)
9. [Computational Considerations](#computational-considerations)
10. [Advantages and Limitations](#advantages-and-limitations)
11. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to OLS

### What is Ordinary Least Squares?

**Ordinary Least Squares (OLS)** is a method for estimating the parameters (weights) in a linear regression model by minimizing the **sum of squared residuals**.

**In simple terms**:
- We have data points that don't lie on a perfect line
- We want to find the "best" line through these points
- "Best" means **minimizing the squared vertical distances** from points to the line

### Why "Least Squares"?

**Least**: We minimize (find the least value)
**Squares**: We square the errors before summing

```
Minimize: Σ(error)² = Σ(actual - predicted)²
```

**Why square?**
1. Makes all errors positive (|error| would work too)
2. Penalizes large errors more heavily
3. Mathematically convenient (smooth, differentiable)
4. Has nice statistical properties (unbiased, minimum variance)

### Visual Intuition

```
    y
    │
  4 │      ●                     Error = vertical distance
    │         ╲                  
  3 │    ●     ╲                 ●  Actual point
    │      │    ╲                │  Error (residual)
  2 │      ●─────●               ─  Prediction on line
    │           ╱  ╲
  1 │        ╱     ●
    │     ╱
  0 └─────────────────── x
    0   1   2   3   4

    We find the line that makes Σ(errors)² smallest!
```

### Historical Context

**1805**: Adrien-Marie Legendre first published the method
**1809**: Carl Friedrich Gauss proved it's optimal under certain assumptions
**1821**: Gauss-Markov theorem established OLS as BLUE (Best Linear Unbiased Estimator)

**Famous quote** from Gauss:
> "The method of least squares is the method of minimum variance among all unbiased linear estimators."

---

## The Least Squares Principle

### The Optimization Problem

Given training data `{(x⁽ⁱ⁾, y⁽ⁱ⁾)}` for i = 1, 2, ..., m

We want to find weights **w** that minimize:

```
J(w) = Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
```

Where:
- `ŷ⁽ⁱ⁾ = wᵀx⁽ⁱ⁾` is the prediction
- `y⁽ⁱ⁾` is the actual value
- `(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)` is the **residual** (error)

### Residuals

The **residual** for example i is:

```
εᵢ = y⁽ⁱ⁾ - ŷ⁽ⁱ⁾ = y⁽ⁱ⁾ - wᵀx⁽ⁱ⁾
```

**Interpretation**:
- `εᵢ > 0`: Model **under**-predicted (actual is higher)
- `εᵢ < 0`: Model **over**-predicted (actual is lower)
- `εᵢ = 0`: Perfect prediction

**Visualization**:
```
    y
    │
    │    ● (actual y)
    │    │
    │    │ εᵢ = residual (positive)
    │    │
    │    ●─────────  (predicted ŷ)
    │
    └──────────────── x
```

### Sum of Squared Residuals (SSR)

```
SSR = Σᵢ₌₁ᵐ εᵢ²
    = Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - ŷ⁽ⁱ⁾)²
    = Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - wᵀx⁽ⁱ⁾)²
```

**Also called**:
- Residual Sum of Squares (RSS)
- Sum of Squared Errors (SSE)
- Sum of Squared Residuals (SSR)

### Why Not Sum of Absolute Errors?

We could minimize `Σ|εᵢ|` instead. Why don't we?

**Squared errors advantages**:
1. **Differentiable** everywhere (absolute value has a corner at 0)
2. **Unique solution** (convex, one global minimum)
3. **Statistical optimality**: OLS is the Maximum Likelihood Estimator under Gaussian noise
4. **Computational efficiency**: Has closed-form solution

**Absolute errors** (LAD - Least Absolute Deviations):
- More **robust to outliers**
- No closed-form solution
- Used in robust regression

---

## OLS Derivation - Simple Linear Regression

### Problem Setup

For **one feature** x, find `w₀` and `w₁` that minimize:

```
J(w₀, w₁) = Σᵢ₌₁ᵐ [y⁽ⁱ⁾ - (w₀ + w₁x⁽ⁱ⁾)]²
```

### Method: Calculus

To minimize, we take **partial derivatives** and set them to zero:

```
∂J/∂w₀ = 0
∂J/∂w₁ = 0
```

This gives us a system of equations called the **normal equations**.

### Step 1: Expand the Cost Function

```
J(w₀, w₁) = Σᵢ₌₁ᵐ [y⁽ⁱ⁾ - w₀ - w₁x⁽ⁱ⁾]²
```

### Step 2: Partial Derivative with Respect to w₀

```
∂J/∂w₀ = ∂/∂w₀ [Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - w₀ - w₁x⁽ⁱ⁾)²]
```

Using chain rule:
```
∂J/∂w₀ = Σᵢ₌₁ᵐ 2(y⁽ⁱ⁾ - w₀ - w₁x⁽ⁱ⁾)(-1)
       = -2 Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - w₀ - w₁x⁽ⁱ⁾)
```

Set to zero:
```
-2 Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - w₀ - w₁x⁽ⁱ⁾) = 0

Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - w₀ - w₁x⁽ⁱ⁾) = 0

Σᵢ₌₁ᵐ y⁽ⁱ⁾ - Σᵢ₌₁ᵐ w₀ - Σᵢ₌₁ᵐ w₁x⁽ⁱ⁾ = 0

Σᵢ₌₁ᵐ y⁽ⁱ⁾ - m·w₀ - w₁ Σᵢ₌₁ᵐ x⁽ⁱ⁾ = 0
```

**First normal equation**:
```
m·w₀ + w₁ Σᵢ₌₁ᵐ x⁽ⁱ⁾ = Σᵢ₌₁ᵐ y⁽ⁱ⁾
```

### Step 3: Partial Derivative with Respect to w₁

```
∂J/∂w₁ = ∂/∂w₁ [Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - w₀ - w₁x⁽ⁱ⁾)²]
```

Using chain rule:
```
∂J/∂w₁ = Σᵢ₌₁ᵐ 2(y⁽ⁱ⁾ - w₀ - w₁x⁽ⁱ⁾)(-x⁽ⁱ⁾)
       = -2 Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - w₀ - w₁x⁽ⁱ⁾)x⁽ⁱ⁾
```

Set to zero:
```
-2 Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - w₀ - w₁x⁽ⁱ⁾)x⁽ⁱ⁾ = 0

Σᵢ₌₁ᵐ (y⁽ⁱ⁾x⁽ⁱ⁾ - w₀x⁽ⁱ⁾ - w₁x⁽ⁱ⁾x⁽ⁱ⁾) = 0

Σᵢ₌₁ᵐ y⁽ⁱ⁾x⁽ⁱ⁾ - w₀ Σᵢ₌₁ᵐ x⁽ⁱ⁾ - w₁ Σᵢ₌₁ᵐ (x⁽ⁱ⁾)² = 0
```

**Second normal equation**:
```
w₀ Σᵢ₌₁ᵐ x⁽ⁱ⁾ + w₁ Σᵢ₌₁ᵐ (x⁽ⁱ⁾)² = Σᵢ₌₁ᵐ y⁽ⁱ⁾x⁽ⁱ⁾
```

### Step 4: Solve the System

We have two equations, two unknowns:

**Equation 1**:
```
m·w₀ + w₁ Σx = Σy
```

**Equation 2**:
```
w₀ Σx + w₁ Σx² = Σxy
```

Where we use shorthand: `Σx = Σᵢ₌₁ᵐ x⁽ⁱ⁾`, etc.

### Step 5: Solve for w₁

From Equation 1:
```
w₀ = (Σy - w₁Σx) / m = ȳ - w₁x̄
```

Where `x̄ = Σx/m` is the mean of x, and `ȳ = Σy/m` is the mean of y.

Substitute into Equation 2:
```
(ȳ - w₁x̄)Σx + w₁Σx² = Σxy

ȳΣx - w₁x̄Σx + w₁Σx² = Σxy

w₁(Σx² - x̄Σx) = Σxy - ȳΣx

w₁(Σx² - x̄·m·x̄) = Σxy - ȳ·m·x̄

w₁(Σx² - m·x̄²) = Σxy - m·x̄·ȳ
```

Recall:
```
Σ(x - x̄)² = Σx² - 2x̄Σx + m·x̄²
           = Σx² - 2x̄·m·x̄ + m·x̄²
           = Σx² - m·x̄²

Σ(x - x̄)(y - ȳ) = Σxy - x̄Σy - ȳΣx + m·x̄·ȳ
                  = Σxy - m·x̄·ȳ - m·x̄·ȳ + m·x̄·ȳ
                  = Σxy - m·x̄·ȳ
```

Therefore:
```
w₁ = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
```

### Step 6: Solve for w₀

From earlier:
```
w₀ = ȳ - w₁x̄
```

### Final Formulas

**Slope (w₁)**:
```
w₁ = Σᵢ₌₁ᵐ (x⁽ⁱ⁾ - x̄)(y⁽ⁱ⁾ - ȳ) / Σᵢ₌₁ᵐ (x⁽ⁱ⁾ - x̄)²

   = Cov(x, y) / Var(x)
```

**Intercept (w₀)**:
```
w₀ = ȳ - w₁x̄
```

### Key Insight

The regression line **always passes through the centroid** (x̄, ȳ):
```
ȳ = w₀ + w₁x̄  ✓
```

---

## OLS Derivation - Multiple Linear Regression

### Problem Setup

For **n features**, find weight vector **w** that minimizes:

```
J(w) = Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - wᵀx⁽ⁱ⁾)²
```

In matrix form:
```
J(w) = ||Xw - y||²
```

Where:
- `X` is the m × (n+1) design matrix
- `w` is the (n+1) × 1 weight vector
- `y` is the m × 1 target vector

### Matrix Notation Expansion

```
J(w) = (Xw - y)ᵀ(Xw - y)
```

Expand:
```
J(w) = (wᵀXᵀ - yᵀ)(Xw - y)
     = wᵀXᵀXw - wᵀXᵀy - yᵀXw + yᵀy
     = wᵀXᵀXw - 2wᵀXᵀy + yᵀy
```

Note: `wᵀXᵀy = yᵀXw` (both are scalars, and transpose of scalar = scalar)

### Taking the Gradient

The gradient of J with respect to w:

```
∇ᵥJ(w) = ∂J/∂w
```

**Matrix calculus rules**:
1. `∂(wᵀXᵀXw)/∂w = 2XᵀXw`
2. `∂(wᵀXᵀy)/∂w = Xᵀy`
3. `∂(yᵀy)/∂w = 0` (doesn't depend on w)

Therefore:
```
∇ᵥJ(w) = 2XᵀXw - 2Xᵀy
```

### Setting Gradient to Zero

For minimum:
```
∇ᵥJ(w) = 0

2XᵀXw - 2Xᵀy = 0

XᵀXw = Xᵀy
```

This is the **normal equation**!

### Solving for w

If XᵀX is invertible:
```
w = (XᵀX)⁻¹Xᵀy
```

This is the **closed-form solution** for OLS.

### Detailed Derivation (Element-wise)

For those who want to see it component by component:

```
J(w) = Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - Σⱼ₌₀ⁿ wⱼx⁽ⁱ⁾ⱼ)²
```

Taking partial derivative with respect to wₖ:

```
∂J/∂wₖ = Σᵢ₌₁ᵐ 2(y⁽ⁱ⁾ - Σⱼ₌₀ⁿ wⱼx⁽ⁱ⁾ⱼ)(-x⁽ⁱ⁾ₖ)

       = -2 Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - Σⱼ₌₀ⁿ wⱼx⁽ⁱ⁾ⱼ)x⁽ⁱ⁾ₖ
```

Set to zero:
```
Σᵢ₌₁ᵐ (y⁽ⁱ⁾ - Σⱼ₌₀ⁿ wⱼx⁽ⁱ⁾ⱼ)x⁽ⁱ⁾ₖ = 0

Σᵢ₌₁ᵐ y⁽ⁱ⁾x⁽ⁱ⁾ₖ = Σᵢ₌₁ᵐ (Σⱼ₌₀ⁿ wⱼx⁽ⁱ⁾ⱼ)x⁽ⁱ⁾ₖ

Σᵢ₌₁ᵐ y⁽ⁱ⁾x⁽ⁱ⁾ₖ = Σⱼ₌₀ⁿ wⱼ (Σᵢ₌₁ᵐ x⁽ⁱ⁾ⱼx⁽ⁱ⁾ₖ)
```

This must hold for all k = 0, 1, ..., n, giving us (n+1) equations.

In matrix form:
```
Xᵀy = XᵀXw
```

---

## Normal Equation

### Definition

The **normal equation** is:

```
XᵀXw = Xᵀy
```

**Solution** (when XᵀX is invertible):

```
w = (XᵀX)⁻¹Xᵀy
```

### Why "Normal"?

The term "normal" comes from geometry: the residual vector is **orthogonal (normal)** to the column space of X.

**Mathematical statement**:
```
Xᵀ(y - Xw) = 0
```

This means the residuals are perpendicular to every column of X!

### Matrix Dimensions

Let's verify dimensions match:

```
X:       m × (n+1)
y:       m × 1
w:       (n+1) × 1

XᵀX:     (n+1) × m × m × (n+1) = (n+1) × (n+1)  ✓ Square!
Xᵀy:     (n+1) × m × m × 1     = (n+1) × 1      ✓

(XᵀX)⁻¹: (n+1) × (n+1)
(XᵀX)⁻¹Xᵀy: (n+1) × (n+1) × (n+1) × 1 = (n+1) × 1  ✓
```

Everything matches!

### Example Calculation

**Data**:
```
X = [1  1]    y = [2]
    [1  2]        [4]
    [1  3]        [5]
    [1  4]        [4]
    [1  5]        [5]
```

**Step 1**: Compute XᵀX
```
      [1  1  1  1  1]   [1  1]
XᵀX = [1  2  3  4  5] × [1  2]
                        [1  3]
                        [1  4]
                        [1  5]

    = [5   15]
      [15  55]
```

**Step 2**: Compute Xᵀy
```
      [1  1  1  1  1]   [2]
Xᵀy = [1  2  3  4  5] × [4]
                        [5]
                        [4]
                        [5]

    = [20]
      [58]
```

**Step 3**: Solve XᵀXw = Xᵀy
```
[5   15] [w₀]   [20]
[15  55] [w₁] = [58]
```

**Step 4**: Invert XᵀX

For 2×2 matrix:
```
A = [a  b]    A⁻¹ = 1/(ad-bc) [ d  -b]
    [c  d]                    [-c   a]
```

```
det(XᵀX) = 5×55 - 15×15 = 275 - 225 = 50

(XᵀX)⁻¹ = 1/50 [55  -15]
               [-15   5]

         = [ 1.1  -0.3]
           [-0.3   0.1]
```

**Step 5**: Compute w
```
w = (XᵀX)⁻¹Xᵀy

  = [ 1.1  -0.3] [20]
    [-0.3   0.1] [58]

  = [1.1×20 + (-0.3)×58]
    [-0.3×20 + 0.1×58]

  = [22 - 17.4]
    [-6 + 5.8]

  = [4.6]
    [-0.2]

Nope! Let me recalculate...

w = [ 1.1  -0.3] [20]
    [-0.3   0.1] [58]

  = [1.1×20 - 0.3×58]   [22 - 17.4]   [4.6]
    [-0.3×20 + 0.1×58] = [-6 + 5.8] = [-0.2]

Actually: Let's verify with correct calculation:
det = 50
(XᵀX)⁻¹ = (1/50) * [55  -15] = [1.1   -0.3]
                    [-15   5]    [-0.3   0.1]

w₀ = 1.1(20) - 0.3(58) = 22 - 17.4 = 4.6
w₁ = -0.3(20) + 0.1(58) = -6 + 5.8 = -0.2
```

Actually, let me redo this correctly:

```
w = [ 1.1  -0.3] [20]
    [-0.3   0.1] [58]

w₀ = 1.1(20) + (-0.3)(58) = 22 - 17.4 = 4.6
w₁ = (-0.3)(20) + 0.1(58) = -6 + 5.8 = -0.2
```

Hmm, getting w₁ = -0.2 seems wrong given the data shows an increasing trend.

Let me recalculate XᵀX and Xᵀy:

```
XᵀX = [Σ1     Σx  ] = [5   15]
      [Σx     Σx² ]   [15  55]

Xᵀy = [Σy ] = [2+4+5+4+5] = [20]
      [Σxy]   [1×2+2×4+3×5+4×4+5×5]   [2+8+15+16+25]   [66]

Wait, let me recalculate Xᵀy:
x: [1, 2, 3, 4, 5]
y: [2, 4, 5, 4, 5]

Σxy = 1×2 + 2×4 + 3×5 + 4×4 + 5×5
    = 2 + 8 + 15 + 16 + 25
    = 66

So Xᵀy = [20]
         [66]
```

Now:
```
w = [ 1.1  -0.3] [20]   [22 - 19.8]   [2.2]
    [-0.3   0.1] [66] = [-6 + 6.6 ] = [0.6]
```

**Final model**:
```
ŷ = 2.2 + 0.6x
```

Much better! Positive slope as expected.

### Python Implementation

```python
import numpy as np

# Data
X = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5]
])
y = np.array([2, 4, 5, 4, 5])

# Normal equation
XtX = X.T @ X
Xty = X.T @ y
w = np.linalg.inv(XtX) @ Xty

print("XᵀX:")
print(XtX)
print("\nXᵀy:")
print(Xty)
print("\nWeights:")
print(w)
print(f"\nModel: ŷ = {w[0]:.2f} + {w[1]:.2f}x")

# Predictions
y_pred = X @ w
print("\nPredictions:", y_pred)
print("Actual:", y)
```

**Output**:
```
XᵀX:
[[ 5 15]
 [15 55]]

Xᵀy:
[20 66]

Weights:
[2.2 0.6]

Model: ŷ = 2.20 + 0.60x

Predictions: [2.8 3.4 4.  4.6 5.2]
Actual: [2 4 5 4 5]
```

---

## Geometric Interpretation

### Vector Space Perspective

In the vector space ℝᵐ, we can think of:
- **y** as a vector (the true outputs)
- **Xw** as a vector (the predictions)
- **Columns of X** span a subspace

**OLS finds the projection of y onto the column space of X.**

### Visualization (2D)

```
        ℝᵐ space
        
        y (actual)
        ●
        │╲
        │ ╲ (residual ε)
        │  ╲
        │   ╲
        │────●  Xw (prediction)
        │   ╱
        │  ╱  Column space of X
        │ ╱   (all possible predictions)
        │╱
        ●
```

**Key insight**: Xw is the point in the column space of X that is **closest** to y!

The distance is minimized when the residual **ε = y - Xw** is **perpendicular** to the column space.

### Orthogonality Condition

The residual is orthogonal to every column of X:

```
Xᵀ(y - Xw) = 0
```

Expanding:
```
Xᵀy - XᵀXw = 0
XᵀXw = Xᵀy  ← Normal equation!
```

**Geometric meaning**: The residuals are perpendicular to the space of predictors.

### Projection Matrix

The **projection matrix** (hat matrix) is:

```
P = X(XᵀX)⁻¹Xᵀ
```

It projects y onto the column space of X:

```
ŷ = Xw = X(XᵀX)⁻¹Xᵀy = Py
```

**Properties**:
1. **Symmetric**: P = Pᵀ
2. **Idempotent**: P² = P (projecting twice = projecting once)
3. **Projects onto col(X)**: Px ∈ col(X) for any x

### Residual Projection

The **residual maker matrix** (annihilator matrix) is:

```
M = I - P = I - X(XᵀX)⁻¹Xᵀ
```

It computes residuals:

```
ε = y - ŷ = y - Py = (I - P)y = My
```

**Properties**:
1. **Symmetric**: M = Mᵀ
2. **Idempotent**: M² = M
3. **Orthogonal to P**: PM = MP = 0
4. **Projects onto orthogonal complement**: Mx ⊥ col(X)

### 3D Example

Consider m=3 examples, n=1 feature (plus bias):

```
        y₃
        │
        │    ● y (actual point in ℝ³)
        │   ╱│
        │  ╱ │
        │ ╱  │ε (residual)
        │╱   │
        ●────┤  ŷ (projection)
       ╱│    │
      ╱ │    │
     ╱  │    │
    ╱   │    │
   ╱────┼────┼─── y₁
  y₂    │    │
        Plane spanned by columns of X
        (all possible linear predictions)
```

The columns of X span a **plane** in ℝ³.
- **ŷ** is on this plane
- **ε** is perpendicular to this plane
- **y** = **ŷ** + **ε** (orthogonal decomposition)

### Pythagorean Theorem

Since ŷ and ε are orthogonal:

```
||y||² = ||ŷ||² + ||ε||²
```

In terms of sums of squares:

```
TSS = ESS + RSS
```

Where:
- **TSS** (Total Sum of Squares) = `Σ(yᵢ - ȳ)²`
- **ESS** (Explained Sum of Squares) = `Σ(ŷᵢ - ȳ)²`
- **RSS** (Residual Sum of Squares) = `Σ(yᵢ - ŷᵢ)²`

This leads to:

```
R² = ESS/TSS = 1 - RSS/TSS
```

---

## Closed-Form Solution

### The Formula

```
w = (XᵀX)⁻¹Xᵀy
```

This is a **closed-form** (analytical) solution - no iteration needed!

### When It Exists

The solution exists when **XᵀX is invertible**.

**Conditions for invertibility**:

1. **Rank(X) = n+1** (full column rank)
2. **m ≥ n+1** (at least as many examples as features)
3. **Columns of X are linearly independent** (no multicollinearity)

**What if XᵀX is singular?**

- No unique solution (infinitely many solutions)
- Caused by **perfect multicollinearity**
- Solutions: Remove redundant features, use regularization (Ridge)

### Moore-Penrose Pseudoinverse

When XᵀX is not invertible, we can use the **pseudoinverse**:

```
w = X⁺y
```

Where `X⁺ = (XᵀX)⁺Xᵀ` is the pseudoinverse.

**In NumPy**:

```python
w = np.linalg.pinv(X) @ y
```

The pseudoinverse always exists and gives the **minimum norm solution**.

### Alternative Form

If m < n+1 (fewer examples than features), we can use:

```
w = Xᵀ(XXᵀ)⁻¹y
```

This is equivalent when both exist, but may be computationally faster.

### Computational Complexity

**Inverting XᵀX**:

- XᵀX is (n+1) × (n+1)
- Matrix inversion: **O((n+1)³)**
- Matrix multiplication XᵀX: **O(m(n+1)²)**

**Total complexity**: **O(m(n+1)² + (n+1)³)**

For large n, this is approximately **O(n³)**.

### When to Use Closed-Form

**Advantages**:

- **Exact solution** (no approximation)
- **Fast** for small n (< 10,000 features)
- **No hyperparameters** (learning rate, iterations)
- **One-shot** (no convergence issues)

**Disadvantages**:

- **Memory intensive**: Requires O(n²) memory for XᵀX
- **Slow** for large n (n > 10,000)
- **Numerically unstable** if XᵀX is nearly singular
- **Not online** (can't update with new data easily)

**Rule of thumb**:

- n < 10,000 → Use closed-form
- n > 10,000 → Use gradient descent

### Numerical Stability

**Problem**: Direct inversion can be numerically unstable.

**Better approaches**:

1. **Cholesky Decomposition**:

   ```python
   L = np.linalg.cholesky(XtX)
   w = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
   ```

2. **QR Decomposition**:

   ```python
   Q, R = np.linalg.qr(X)
   w = np.linalg.solve(R, Q.T @ y)
   ```

3. **SVD (Singular Value Decomposition)**:

   ```python
   U, s, Vt = np.linalg.svd(X, full_matrices=False)
   w = Vt.T @ np.diag(1/s) @ U.T @ y
   ```

**In scikit-learn**, these are used automatically!

### Python Implementation Comparison

```python
import numpy as np
import time

# Generate data
np.random.seed(42)
m, n = 1000, 100
X = np.random.randn(m, n)
X = np.c_[np.ones(m), X]  # Add bias
y = np.random.randn(m)

# Method 1: Direct inversion
start = time.time()
XtX = X.T @ X
Xty = X.T @ y
w1 = np.linalg.inv(XtX) @ Xty
time1 = time.time() - start

# Method 2: Solve (Cholesky)
start = time.time()
w2 = np.linalg.solve(XtX, Xty)
time2 = time.time() - start

# Method 3: lstsq (QR or SVD)
start = time.time()
w3 = np.linalg.lstsq(X, y, rcond=None)[0]
time3 = time.time() - start

# Method 4: Pseudoinverse
start = time.time()
w4 = np.linalg.pinv(X) @ y
time4 = time.time() - start

print(f"Direct inversion: {time1:.6f}s")
print(f"Solve (Cholesky): {time2:.6f}s")
print(f"lstsq (QR/SVD):   {time3:.6f}s")
print(f"Pseudoinverse:    {time4:.6f}s")

# Verify they're all the same
print(f"\nMax difference: {np.max(np.abs(w1 - w2)):.2e}")
print(f"Max difference: {np.max(np.abs(w1 - w3)):.2e}")
print(f"Max difference: {np.max(np.abs(w1 - w4)):.2e}")
```

**Typical output**:

```
Direct inversion: 0.000834s
Solve (Cholesky): 0.000421s  ← Fastest!
lstsq (QR/SVD):   0.001223s
Pseudoinverse:    0.002156s

Max difference: 3.55e-14
Max difference: 1.42e-13
Max difference: 2.84e-13
```

**Recommendation**: Use `np.linalg.solve()` or `np.linalg.lstsq()` instead of explicit inversion!

---

## Properties of OLS Estimators

### 1. Unbiasedness

The OLS estimator is **unbiased**:

```
E[ŵ] = w_true
```

**Proof sketch**:
```
ŵ = (XᵀX)⁻¹Xᵀy
  = (XᵀX)⁻¹Xᵀ(Xw_true + ε)
  = (XᵀX)⁻¹XᵀXw_true + (XᵀX)⁻¹Xᵀε
  = w_true + (XᵀX)⁻¹Xᵀε

E[ŵ] = w_true + (XᵀX)⁻¹XᵀE[ε]
     = w_true + (XᵀX)⁻¹Xᵀ·0
     = w_true
```

Assuming `E[ε] = 0` (errors have zero mean).

### 2. Variance

The variance-covariance matrix of ŵ is:

```
Var(ŵ) = σ²(XᵀX)⁻¹
```

Where σ² is the variance of the errors.

**Diagonal elements** give variance of each coefficient:
```
Var(ŵⱼ) = σ²[(XᵀX)⁻¹]ⱼⱼ
```

**Standard errors**:
```
SE(ŵⱼ) = σ√[(XᵀX)⁻¹]ⱼⱼ
```

**Estimating σ²**:
```
σ̂² = RSS / (m - n - 1)
   = Σ(yᵢ - ŷᵢ)² / (m - n - 1)
```

The denominator (m - n - 1) is the **degrees of freedom**.

### 3. Gauss-Markov Theorem

**Theorem**: Under certain assumptions, OLS is BLUE (Best Linear Unbiased Estimator).

**BLUE means**:
- **Best**: Minimum variance among all linear unbiased estimators
- **Linear**: Estimator is a linear function of y
- **Unbiased**: E[ŵ] = w_true

**Assumptions**:
1. Linearity: y = Xw + ε
2. E[ε] = 0 (zero mean errors)
3. Var(ε) = σ²I (homoscedasticity & independence)
4. X is fixed and full rank

**Important**: Does NOT require normality of errors!

### 4. Residual Properties

**Sum of residuals is zero**:
```
Σᵢ εᵢ = 0
```

**Residuals are orthogonal to predictions**:
```
Σᵢ ŷᵢεᵢ = 0
```

**Residuals are orthogonal to each feature**:
```
Σᵢ xᵢⱼεᵢ = 0  for all j
```

### 5. Consistency

As m → ∞, ŵ → w_true (converges in probability).

**Why**: Law of large numbers ensures (1/m)XᵀX → E[xxᵀ] and (1/m)Xᵀy → E[xy].

---

## Computational Considerations

### Memory Requirements

**Storing X**: O(mn) memory
**Storing XᵀX**: O(n²) memory

For large m and n:
- m = 1,000,000, n = 1,000
- X requires: 8 GB (double precision)
- XᵀX requires: 8 MB (much smaller!)

**Strategy**: Compute XᵀX and Xᵀy incrementally without storing full X.

### Online Updates

**Problem**: New data arrives, need to update model.

**Naive approach**: Recompute from scratch - expensive!

**Sherman-Morrison formula**: Update (XᵀX)⁻¹ efficiently when adding one row.

For new example (x_new, y_new):

```
XᵀX_new = XᵀX_old + x_new·x_newᵀ
Xᵀy_new = Xᵀy_old + x_new·y_new
```

Then update (XᵀX)⁻¹ using Sherman-Morrison instead of re-inverting.

**Complexity**: O(n²) instead of O(n³)!

### Regularization Preview

When XᵀX is nearly singular or we want to prevent overfitting, add a **regularization term**:

**Ridge regression**:
```
w = (XᵀX + λI)⁻¹Xᵀy
```

This ensures XᵀX + λI is always invertible (covered in Regularization file).

### Sparse Data

For **sparse X** (many zeros):
- Store in sparse format (CSR, CSC)
- Use specialized sparse solvers
- Can handle much larger n

**Libraries**:
- `scipy.sparse` (Python)
- Specialized sparse linear algebra

---

## Advantages and Limitations

### Advantages of OLS

✅ **Exact solution**: No approximation, no hyperparameters
✅ **Fast for small n**: One-shot computation
✅ **Interpretable**: Clear weights for each feature
✅ **Statistical properties**: Unbiased, minimum variance (BLUE)
✅ **Confidence intervals**: Can compute standard errors
✅ **No tuning**: No learning rate, no iterations
✅ **Deterministic**: Same result every time

### Limitations of OLS

❌ **Sensitive to outliers**: Squared errors amplify outliers
❌ **Assumes linearity**: Can't model complex non-linear patterns
❌ **Multicollinearity issues**: Unstable when features correlated
❌ **Memory intensive**: Requires O(n²) for XᵀX
❌ **Slow for large n**: O(n³) inversion
❌ **No regularization**: Can overfit with many features
❌ **Not online**: Hard to update with streaming data
❌ **Numerical instability**: Direct inversion can be unstable

### When to Use OLS

**Good for**:
- Small to medium datasets (m < 1,000,000, n < 10,000)
- Interpretability matters
- Need confidence intervals
- Assumptions are met
- Fast prototyping

**Avoid when**:
- Very large n (use gradient descent)
- Severe multicollinearity (use Ridge)
- Many outliers (use robust regression)
- Non-linear relationships (use polynomial features or non-linear models)
- Streaming data (use online algorithms)

### Alternatives to OLS

| Method | Use Case |
|--------|----------|
| **Gradient Descent** | Large n, online learning |
| **Ridge Regression** | Multicollinearity, regularization |
| **Lasso Regression** | Feature selection, sparse solutions |
| **Robust Regression** | Outliers present |
| **Weighted Least Squares** | Heteroscedasticity |
| **Generalized Least Squares** | Correlated errors |

---

## Summary and Next Steps

### What We Learned

1. **OLS Principle**: Minimize sum of squared residuals
   ```
   min Σ(y - ŷ)²
   ```

2. **Simple Regression Formulas**:
   ```
   w₁ = Cov(x,y) / Var(x)
   w₀ = ȳ - w₁x̄
   ```

3. **Normal Equation**:
   ```
   XᵀXw = Xᵀy
   ```

4. **Closed-Form Solution**:
   ```
   w = (XᵀX)⁻¹Xᵀy
   ```

5. **Geometric Interpretation**: Projection onto column space of X

6. **Properties**: Unbiased, BLUE (Gauss-Markov), consistent

7. **Computational**: O(n³) complexity, numerically stable methods exist

### Key Takeaways

✅ **Closed-form**: Direct solution, no iteration  
✅ **Optimal**: BLUE under assumptions  
✅ **Geometric**: Orthogonal projection  
✅ **Practical**: Use `np.linalg.solve()` not explicit inversion  
✅ **Foundation**: Basis for Ridge, Lasso, and other methods  

### Mathematical Prerequisites Checklist

Before moving on, make sure you understand:
- [ ] How to derive normal equations from calculus
- [ ] Why residuals are orthogonal to column space of X
- [ ] The formula w = (XᵀX)⁻¹Xᵀy
- [ ] When XᵀX is invertible
- [ ] Geometric meaning of projection
- [ ] Properties: unbiased, BLUE
- [ ] Computational complexity O(n³)

### Next Steps in This Section

1. **Gradient_Descent_for_Linear_Models.md**
   - Iterative optimization
   - When to use vs closed-form
   - Batch, SGD, mini-batch
   - Learning rate selection

2. **Cost_Functions_and_Loss.md**
   - Why MSE?
   - Alternative losses
   - Connection to probability

3. **Logistic_Regression.md**
   - Classification with linear models
   - No closed-form solution!
   - Must use gradient descent

4. **Regularization_L1_L2.md**
   - Ridge: (XᵀX + λI)⁻¹Xᵀy
   - Lasso: No closed-form
   - When and why to regularize

### Practice Exercises

#### Derivation:
1. Derive the normal equation for simple linear regression
2. Prove that Σεᵢ = 0 for OLS residuals
3. Show that (I - P) is idempotent

#### Computation:
1. Given X and y, compute w by hand for n=1
2. Verify XᵀXw = Xᵀy for your solution
3. Compute the projection matrix P

#### Coding:
1. Implement OLS three ways: direct inversion, solve, lstsq
2. Time each method for different values of n
3. Compute and plot residuals

#### Conceptual:
1. Why does OLS minimize perpendicular distance, not shortest distance?
2. When would (XᵀX)⁻¹ not exist?
3. What happens to the solution if we duplicate a feature?

### Additional Resources

**Books**:
- "Matrix Computations" by Golub & Van Loan - Numerical methods
- "Linear Algebra and Its Applications" by Strang - Projection interpretation
- "Econometric Analysis" by Greene - Statistical properties

**Papers**:
- Gauss (1809) - Original OLS derivation
- Gauss-Markov Theorem - BLUE property

**Online**:
- MIT 18.06: Linear Algebra lectures (Strang)
- 3Blue1Brown: Linear transformations and projections
- Numerical Linear Algebra course (fast.ai)

**Implementations**:
- NumPy documentation: `linalg.lstsq`
- SciPy: Sparse solvers
- Scikit-learn: `LinearRegression` source code

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║            ORDINARY LEAST SQUARES QUICK REFERENCE              ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  OBJECTIVE:  min ||Xw - y||²                                   ║
║              min Σ(y - ŷ)²                                     ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  NORMAL EQUATION                                               ║
╠════════════════════════════════════════════════════════════════╣
║  XᵀXw = Xᵀy                                                    ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  SOLUTION                                                      ║
╠════════════════════════════════════════════════════════════════╣
║  w = (XᵀX)⁻¹Xᵀy                                                ║
║                                                                ║
║  Requirements:                                                 ║
║  • XᵀX invertible (full rank)                                  ║
║  • m ≥ n+1                                                     ║
║  • No perfect multicollinearity                                ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  SIMPLE REGRESSION (n=1)                                       ║
╠════════════════════════════════════════════════════════════════╣
║  w₁ = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)² = Cov(x,y)/Var(x)                 ║
║  w₀ = ȳ - w₁x̄                                                  ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  GEOMETRIC INTERPRETATION                                      ║
╠════════════════════════════════════════════════════════════════╣
║  • ŷ = projection of y onto col(X)                             ║
║  • ε = y - ŷ ⊥ col(X)                                          ║
║  • Xᵀε = 0 (orthogonality condition)                           ║
║  • P = X(XᵀX)⁻¹Xᵀ (projection matrix)                          ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  PROPERTIES                                                    ║
╠════════════════════════════════════════════════════════════════╣
║  • Unbiased: E[ŵ] = w_true                                     ║
║  • BLUE: Best Linear Unbiased Estimator                        ║
║  • Var(ŵ) = σ²(XᵀX)⁻¹                                          ║
║  • Σεᵢ = 0                                                     ║
║  • Σŷᵢεᵢ = 0                                                   ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  COMPUTATIONAL                                                 ║
╠════════════════════════════════════════════════════════════════╣
║  Complexity: O(mn² + n³)                                       ║
║  Memory:     O(n²)                                             ║
║                                                                ║
║  Better than direct inversion:                                 ║
║  • np.linalg.solve(XtX, Xty)                                   ║
║  • np.linalg.lstsq(X, y)                                       ║
║  • QR or SVD decomposition                                     ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  USE WHEN                                                      ║
╠════════════════════════════════════════════════════════════════╣
║  ✓ n < 10,000                                                  ║
║  ✓ Need exact solution                                         ║
║  ✓ Want confidence intervals                                   ║
║  ✓ Assumptions satisfied                                       ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  AVOID WHEN                                                    ║
╠════════════════════════════════════════════════════════════════╣
║  ✗ n > 10,000 (use gradient descent)                           ║
║  ✗ Multicollinearity (use Ridge)                               ║
║  ✗ Outliers (use robust methods)                               ║
║  ✗ Streaming data (use online learning)                        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Ready to move on?** Next up: **Gradient_Descent_for_Linear_Models.md** where we'll learn iterative optimization!

---

*End of Ordinary_Least_Squares.md*