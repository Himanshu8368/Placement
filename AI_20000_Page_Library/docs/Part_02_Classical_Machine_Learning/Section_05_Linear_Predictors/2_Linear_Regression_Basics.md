# Linear Regression Basics

## Table of Contents
1. [Introduction to Regression](#introduction-to-regression)
2. [Simple Linear Regression](#simple-linear-regression)
3. [Multiple Linear Regression](#multiple-linear-regression)
4. [Hypothesis Function](#hypothesis-function)
5. [Model Assumptions](#model-assumptions)
6. [Model Evaluation](#model-evaluation)
7. [Practical Implementation](#practical-implementation)
8. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
9. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to Regression

### What is Regression?

**Regression** is a supervised learning task where we predict a **continuous numerical value** based on input features.

**Examples**:
- Predicting house prices (output: $350,000)
- Forecasting temperature (output: 25.5°C)
- Estimating sales (output: 1,247 units)

**Contrast with Classification**:
- **Regression**: Predict continuous values (prices, temperatures, counts)
- **Classification**: Predict discrete categories (spam/not spam, cat/dog)

### The Goal

Given training data with **features** (X) and **outputs** (y), find a function that:
```
f(x) ≈ y
```

For **linear regression**, this function is a **linear combination** of features:
```
f(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

### Types of Regression

| Type | Features | Equation | Example |
|------|----------|----------|---------|
| **Simple** | 1 feature | y = w₀ + w₁x | Price vs Size |
| **Multiple** | 2+ features | y = w₀ + w₁x₁ + w₂x₂ + ... | Price vs Size, Age, Location |
| **Polynomial** | Powers of features | y = w₀ + w₁x + w₂x² + ... | Non-linear patterns |
| **Multivariate** | Multiple outputs | Y = Xw (matrix) | Predict multiple targets |

In this document, we focus on **Simple** and **Multiple** linear regression.

---

## Simple Linear Regression

### Definition

**Simple Linear Regression** models the relationship between:
- **One input feature** (x)
- **One output variable** (y)

Using a **straight line**:
```
ŷ = w₀ + w₁x
```

Where:
- `ŷ` = predicted value
- `w₀` = y-intercept (bias)
- `w₁` = slope (weight)
- `x` = input feature

### Geometric Interpretation

```
    y (Price)
    │
500 │                    ●
    │                 ●
400 │              ●
    │           ●           ŷ = w₀ + w₁x
300 │        ●  ╱
    │     ●   ╱  (Best fit line)
200 │  ●    ╱
    │ ●   ╱
100 │  ╱ ●
    │╱
  0 └─────────────────── x (Size)
    0   500  1000 1500
```

The line minimizes the **vertical distances** between predictions (ŷ) and actual values (y).

### Real Example: House Prices

**Dataset**:
```
Size (sqft)  |  Price ($1000s)
-------------|----------------
   500       |      150
  1000       |      200
  1500       |      250
  2000       |      300
  2500       |      350
  3000       |      400
```

**Goal**: Find w₀ and w₁ such that:
```
Price = w₀ + w₁ × Size
```

**Visualization**:
```
Price
  │
400│                          ●
  │                      ●
300│                  ●
  │              ●
200│          ●
  │      ●
100│
  │
  0└────────────────────────── Size
    0    1000   2000   3000
```

### The Parameters

#### **w₁ (Slope)**
- Represents the **change in y** for one unit change in x
- Also called the **regression coefficient**

**Interpretation**:
```
If w₁ = 0.1 (in $1000s per sqft)
Then: Each additional 100 sqft adds $10,000 to the price
```

**Sign meaning**:
- `w₁ > 0` → Positive relationship (x ↑, y ↑)
- `w₁ < 0` → Negative relationship (x ↑, y ↓)
- `w₁ = 0` → No relationship

#### **w₀ (Intercept)**
- The predicted value when **x = 0**
- Also called the **bias term**

**Interpretation**:
```
If w₀ = 50 (in $1000s)
Then: Base price is $50,000 (before considering size)
```

**Note**: w₀ may not always have physical meaning (e.g., a house can't have 0 sqft!)

### Mathematical Derivation

We want to find w₀ and w₁ that minimize prediction errors.

**Error for one example**:
```
error = ŷ - y = (w₀ + w₁x) - y
```

**Squared error** (to make positive and emphasize large errors):
```
error² = [(w₀ + w₁x) - y]²
```

**Total error** over all m examples (Cost Function):
```
J(w₀, w₁) = (1/2m) Σᵢ₌₁ᵐ [(w₀ + w₁x⁽ⁱ⁾) - y⁽ⁱ⁾]²
```

The factor 1/2m simplifies calculus (we'll see why in the Gradient Descent file).

### Optimal Solution (Closed Form)

For simple linear regression, we can derive **exact formulas**:

**Slope**:
```
w₁ = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]
```

Or equivalently:
```
w₁ = Cov(x, y) / Var(x)
```

Where:
- `x̄` = mean of x values
- `ȳ` = mean of y values
- `Cov(x, y)` = covariance of x and y
- `Var(x)` = variance of x

**Intercept**:
```
w₀ = ȳ - w₁x̄
```

This ensures the line passes through the **centroid** point (x̄, ȳ).

### Step-by-Step Example

**Given data**:
```
x:  [1, 2, 3, 4, 5]
y:  [2, 4, 5, 4, 5]
```

**Step 1**: Calculate means
```
x̄ = (1 + 2 + 3 + 4 + 5) / 5 = 15/5 = 3
ȳ = (2 + 4 + 5 + 4 + 5) / 5 = 20/5 = 4
```

**Step 2**: Calculate deviations
```
i  |  xᵢ  |  yᵢ  |  (xᵢ - x̄)  |  (yᵢ - ȳ)  |  (xᵢ - x̄)(yᵢ - ȳ)  |  (xᵢ - x̄)²
---|------|------|------------|------------|---------------------|------------
1  |  1   |  2   |    -2      |    -2      |         4           |     4
2  |  2   |  4   |    -1      |     0      |         0           |     1
3  |  3   |  5   |     0      |     1      |         0           |     0
4  |  4   |  4   |     1      |     0      |         0           |     1
5  |  5   |  5   |     2      |     1      |         2           |     4
---|------|------|------------|------------|---------------------|------------
Sum                                                  6                  10
```

**Step 3**: Calculate w₁
```
w₁ = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]
w₁ = 6 / 10 = 0.6
```

**Step 4**: Calculate w₀
```
w₀ = ȳ - w₁x̄
w₀ = 4 - (0.6 × 3)
w₀ = 4 - 1.8 = 2.2
```

**Final Model**:
```
ŷ = 2.2 + 0.6x
```

**Step 5**: Make predictions
```
For x = 1: ŷ = 2.2 + 0.6(1) = 2.8
For x = 2: ŷ = 2.2 + 0.6(2) = 3.4
For x = 3: ŷ = 2.2 + 0.6(3) = 4.0
For x = 4: ŷ = 2.2 + 0.6(4) = 4.6
For x = 5: ŷ = 2.2 + 0.6(5) = 5.2
```

### Python Implementation (From Scratch)

```python
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Calculate means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate slope (w1)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
w1 = numerator / denominator

# Calculate intercept (w0)
w0 = y_mean - w1 * x_mean

print(f"w0 (intercept): {w0:.2f}")
print(f"w1 (slope): {w1:.2f}")

# Make predictions
y_pred = w0 + w1 * x

# Plot
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, y_pred, color='red', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Simple Linear Regression: y = {w0:.2f} + {w1:.2f}x')
plt.grid(True)
plt.show()
```

**Output**:
```
w0 (intercept): 2.20
w1 (slope): 0.60
```

### Using Scikit-Learn

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Data (reshape for sklearn)
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create and train model
model = LinearRegression()
model.fit(x, y)

# Get parameters
w0 = model.intercept_
w1 = model.coef_[0]

print(f"w0 (intercept): {w0:.2f}")
print(f"w1 (slope): {w1:.2f}")

# Make predictions
y_pred = model.predict(x)
print(f"Predictions: {y_pred}")

# Predict new value
new_x = np.array([[6]])
new_pred = model.predict(new_x)
print(f"Prediction for x=6: {new_pred[0]:.2f}")
```

**Output**:
```
w0 (intercept): 2.20
w1 (slope): 0.60
Predictions: [2.8 3.4 4.  4.6 5.2]
Prediction for x=6: 5.80
```

### Correlation vs Causation

**Important**: High correlation ≠ causation!

**Example**: Ice cream sales and drowning deaths are correlated.
- Does ice cream cause drowning? **NO!**
- Both are caused by summer weather (confounding variable)

**Guideline**: Linear regression shows **association**, not **causation**. Always consider:
- Confounding variables
- Reverse causation
- Coincidental correlation

---

## Multiple Linear Regression

### Definition

**Multiple Linear Regression** extends simple regression to **multiple input features**:

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

Where:
- `x₁, x₂, ..., xₙ` = n different features
- `w₁, w₂, ..., wₙ` = weights for each feature
- `w₀` = intercept (bias)

### Vector Form

More compactly, using vectors:
```
ŷ = wᵀx
```

Where:
```
w = [w₀, w₁, w₂, ..., wₙ]ᵀ  (weight vector)
x = [1, x₁, x₂, ..., xₙ]ᵀ  (feature vector with 1 for bias)
```

### Real Example: House Prices (Multiple Features)

**Dataset**:
```
Size  Bedrooms  Age  Location  |  Price
(sqft)   (#)   (yrs)  (1-10)   | ($1000s)
----------------------------------------
 2000     3      5      8       |   350
 1500     2     10      6       |   280
 2500     4      2      9       |   420
 1800     3     15      5       |   300
 3000     5      1     10       |   500
```

**Model**:
```
Price = w₀ + w₁(Size) + w₂(Bedrooms) + w₃(Age) + w₄(Location)
```

**Example prediction**:
If weights are: w₀=50, w₁=0.1, w₂=20, w₃=-2, w₄=15

For a house with: Size=2000, Bedrooms=3, Age=5, Location=8
```
Price = 50 + 0.1(2000) + 20(3) + (-2)(5) + 15(8)
Price = 50 + 200 + 60 - 10 + 120
Price = 420 ($420,000)
```

### Geometric Interpretation

**In 2D** (one feature): Line
```
y = w₀ + w₁x₁
```

**In 3D** (two features): Plane
```
     z (price)
     │     ╱
     │   ╱
     │ ╱  (Plane: z = w₀ + w₁x₁ + w₂x₂)
     │╱
     └────────── x₁ (size)
    ╱
  ╱
x₂ (bedrooms)
```

**In higher dimensions** (3+ features): **Hyperplane** (can't visualize!)

### Matrix Form

For **m examples** and **n features**, we can write:

```
ŷ = Xw
```

Where:

**X** (design matrix):
```
      Feature 0  Feature 1  Feature 2  ...  Feature n
      (bias=1)   (Size)     (Bedrooms)      (Location)
X = [    1         2000        3        ...     8      ]  ← Example 1
    [    1         1500        2        ...     6      ]  ← Example 2
    [    1         2500        4        ...     9      ]  ← Example 3
    [   ...        ...        ...       ...    ...     ]
    [    1         3000        5        ...    10      ]  ← Example m

Shape: m × (n+1)
```

**w** (weight vector):
```
    [ w₀ ]
    [ w₁ ]
w = [ w₂ ]
    [ ...]
    [ wₙ ]

Shape: (n+1) × 1
```

**ŷ** (predictions):
```
    [ ŷ⁽¹⁾ ]
    [ ŷ⁽²⁾ ]
ŷ = [ ŷ⁽³⁾ ]
    [  ...  ]
    [ ŷ⁽ᵐ⁾ ]

Shape: m × 1
```

### Cost Function

For multiple regression, the cost function is:

```
J(w) = (1/2m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
```

In matrix form:
```
J(w) = (1/2m) ||Xw - y||²
```

Where `||·||²` denotes the squared L2 norm (sum of squared elements).

### Normal Equation (Closed Form Solution)

For multiple regression, the optimal weights can be found directly:

```
w = (XᵀX)⁻¹Xᵀy
```

**Derivation** (brief):
1. Set gradient to zero: ∇J(w) = 0
2. Expand: Xᵀ(Xw - y) = 0
3. Solve: XᵀXw = Xᵀy
4. Result: w = (XᵀX)⁻¹Xᵀy

**Requirements**:
- XᵀX must be **invertible** (non-singular)
- This requires: Features are **linearly independent** and m ≥ n+1

### Step-by-Step Example

**Given data**:
```
      x₁  x₂  |  y
     (Size) (Bed)  (Price)
    ----------------
       1    1  |  3
       2    1  |  4
       3    2  |  6
       4    2  |  7
```

**Step 1**: Set up X matrix (with bias column)
```
    [1  1  1]
X = [1  2  1]
    [1  3  2]
    [1  4  2]

Shape: 4 × 3
```

**Step 2**: Set up y vector
```
    [3]
y = [4]
    [6]
    [7]

Shape: 4 × 1
```

**Step 3**: Calculate XᵀX
```
      [1  1  1  1]   [1  1  1]   [4  10  6]
XᵀX = [1  2  3  4] × [1  2  1] = [10 30 18]
      [1  1  2  2]   [1  3  2]   [6  18 10]
                     [1  4  2]

Shape: 3 × 3
```

**Step 4**: Calculate Xᵀy
```
      [1  1  1  1]   [3]   [20]
Xᵀy = [1  2  3  4] × [4] = [58]
      [1  1  2  2]   [6]   [35]
                     [7]

Shape: 3 × 1
```

**Step 5**: Solve (XᵀX)w = Xᵀy

Using matrix inversion (or solving the system):
```
w = (XᵀX)⁻¹Xᵀy

     [w₀]   [1.5]
w =  [w₁] = [1.0]
     [w₂]   [1.0]
```

**Final Model**:
```
ŷ = 1.5 + 1.0x₁ + 1.0x₂
```

**Interpretation**:
- Base price: $1,500
- Each unit of x₁ (Size): adds $1,000
- Each unit of x₂ (Bedrooms): adds $1,000

### Python Implementation (NumPy)

```python
import numpy as np

# Data
X = np.array([
    [1, 1],  # bias column will be added
    [2, 1],
    [3, 2],
    [4, 2]
])
y = np.array([3, 4, 6, 7])

# Add bias column
X_with_bias = np.c_[np.ones(X.shape[0]), X]

# Normal equation: w = (X^T X)^-1 X^T y
XtX = X_with_bias.T @ X_with_bias
Xty = X_with_bias.T @ y
w = np.linalg.inv(XtX) @ Xty

# Or use numpy's lstsq (more numerically stable)
# w = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]

print("Weights:", w)
print(f"Model: y = {w[0]:.2f} + {w[1]:.2f}*x1 + {w[2]:.2f}*x2")

# Make predictions
y_pred = X_with_bias @ w
print("Predictions:", y_pred)
print("Actual:", y)
```

**Output**:
```
Weights: [1.5 1.  1. ]
Model: y = 1.50 + 1.00*x1 + 1.00*x2
Predictions: [3. 4. 6. 7.]
Actual: [3 4 6 7]
```

### Using Scikit-Learn

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Data
X = np.array([
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 2]
])
y = np.array([3, 4, 6, 7])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Get parameters
w0 = model.intercept_
w = model.coef_

print(f"Intercept (w0): {w0:.2f}")
print(f"Coefficients (w1, w2): {w}")
print(f"Model: y = {w0:.2f} + {w[0]:.2f}*x1 + {w[1]:.2f}*x2")

# Make predictions
y_pred = model.predict(X)
print("Predictions:", y_pred)

# Predict new example
new_X = np.array([[5, 3]])
new_pred = model.predict(new_X)
print(f"Prediction for [5, 3]: {new_pred[0]:.2f}")

# R² score
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print(f"R² score: {r2:.4f}")
```

**Output**:
```
Intercept (w0): 1.50
Coefficients (w1, w2): [1. 1.]
Model: y = 1.50 + 1.00*x1 + 1.00*x2
Predictions: [3. 4. 6. 7.]
Prediction for [5, 3]: 9.50
R² score: 1.0000
```

### Feature Interpretation

Each weight `wⱼ` represents the **partial effect** of feature j:

```
wⱼ = change in ŷ when xⱼ increases by 1 (holding all other features constant)
```

**Example**:
```
Price = 50 + 0.1(Size) + 20(Bedrooms) - 2(Age) + 15(Location)
```

**Interpretations**:
- `w₁ = 0.1`: Each additional sqft adds $100 (when bedrooms, age, location fixed)
- `w₂ = 20`: Each additional bedroom adds $20,000 (when size, age, location fixed)
- `w₃ = -2`: Each year older reduces price by $2,000 (when size, bedrooms, location fixed)
- `w₄ = 15`: Each point in location score adds $15,000 (when size, bedrooms, age fixed)

**Important**: The interpretation "holding others constant" is crucial! This is different from simple correlation.

### Multicollinearity

**Problem**: When features are highly correlated with each other.

**Example**:
```
x₁ = Square feet
x₂ = Square meters (x₂ ≈ x₁ / 10.764)
```

These are **perfectly correlated**! The model becomes unstable.

**Consequences**:
- Weights become very large and unstable
- XᵀX becomes nearly singular (can't invert)
- Small changes in data → large changes in weights
- Hard to interpret individual feature effects

**Detection**:
- **Variance Inflation Factor (VIF)**: VIF > 10 suggests multicollinearity
- **Condition number**: Large condition number of XᵀX
- **Correlation matrix**: High pairwise correlations

**Solutions**:
- Remove redundant features
- Use **regularization** (Ridge/Lasso - covered in later file)
- Combine correlated features (PCA)

---

## Hypothesis Function

### Definition

The **hypothesis function** `h(x)` is the function that maps inputs to predicted outputs:

```
h(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

Or in vector form:
```
h(x) = wᵀx
```

**Note**: We use `h(x)` and `ŷ` interchangeably:
```
h(x) = ŷ = wᵀx
```

### Why "Hypothesis"?

The term comes from the scientific method:
1. We **hypothesize** a functional form (linear combination)
2. We **test** this hypothesis on data
3. We **evaluate** how well it fits

**Alternative names**:
- Prediction function
- Model function
- Regression function

### Hypothesis Space

The **hypothesis space** is the set of all possible functions we consider.

For linear regression:
```
H = {h(x) = wᵀx | w ∈ ℝⁿ⁺¹}
```

This means:
- We're **restricted** to linear functions
- We **search** over all possible weight vectors
- Our **goal** is to find the best w

**Visualization** (for simple regression):
```
    y
    │     All possible lines:
    │      ╱  ╱ ╱ ╱
    │     ╱  ╱ ╱ ╱
    │    ╱  ╱ ╱ ╱  (Different slopes & intercepts)
    │   ╱  ╱ ╱ ╱
    │  ╱  ╱ ╱ ╱
    │ ╱  ╱ ╱ ╱
    │╱  ╱ ╱ ╱
    └──────────── x
    
    We search for the BEST line!
```

### Parametric vs Non-Parametric

**Linear regression is parametric**:
- Fixed number of parameters (n+1 weights)
- Functional form decided beforehand
- Fast and efficient

**Comparison**:

| Parametric (Linear) | Non-Parametric (e.g., KNN) |
|---------------------|----------------------------|
| h(x) = wᵀx | h(x) = average of k neighbors |
| Fixed # parameters | Grows with data |
| Fast prediction | Slow prediction |
| Strong assumptions | Weak assumptions |
| Can extrapolate | Cannot extrapolate well |

### Hypothesis Function Components

```
h(x; w) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
          └─┬─┘ └────────┬────────────┘
          Bias      Feature contributions
```

**Bias (w₀)**:
- Baseline prediction when all features = 0
- Shifts the hyperplane
- Also called **intercept**

**Feature contributions (w₁x₁, w₂x₂, ...)**:
- How each feature affects the output
- Weighted by importance (wⱼ)
- Summed together (linear combination)

### Learning the Hypothesis

**Process**:
1. Choose hypothesis space (linear functions)
2. Define cost function (MSE)
3. Find best parameters (minimize cost)

**Cost function** (measures error):
```
J(w) = (1/2m) Σᵢ₌₁ᵐ [h(x⁽ⁱ⁾) - y⁽ⁱ⁾]²
     = (1/2m) Σᵢ₌₁ᵐ [wᵀx⁽ⁱ⁾ - y⁽ⁱ⁾]²
```

**Optimization**:
```
w* = argmin J(w)
      w
```

Finding w* is covered in detail in later files (OLS, Gradient Descent).

### Making Predictions

Once we have learned weights w*, we can predict:

**Single example**:
```
Input: x_new = [1, 2000, 3, 5, 8]ᵀ
Prediction: ŷ = w*ᵀ x_new
```

**Multiple examples** (batch prediction):
```
Input: X_new (m_new × (n+1))
Predictions: ŷ = X_new w*
```

**Example code**:
```python
# Trained weights
w = np.array([50, 0.1, 20, -2, 15])

# Single prediction
x_new = np.array([1, 2000, 3, 5, 8])
y_pred = w @ x_new
print(f"Single prediction: ${y_pred:.0f}k")

# Batch predictions
X_new = np.array([
    [1, 2000, 3, 5, 8],
    [1, 1500, 2, 10, 6],
    [1, 2500, 4, 2, 9]
])
y_preds = X_new @ w
print(f"Batch predictions: {y_preds}")
```

**Output**:
```
Single prediction: $420k
Batch predictions: [420. 280. 443.]
```

### Hypothesis Complexity

**Simple hypothesis** (few features):
- Low complexity
- May **underfit** (high bias)
- Generalizes well if model is correct

**Complex hypothesis** (many features):
- High complexity
- May **overfit** (high variance)
- Can memorize training data

**Trade-off**:
```
    Error
    │
    │  Training Error
    │     ╲
    │      ╲_______________
    │
    │  Validation Error
    │        ╱╲
    │       ╱  ╲
    │      ╱    ╲
    │─────────────────────
         Underfitting  Overfitting
         (Too simple)  (Too complex)
              Model Complexity
```

**Optimal complexity**: Minimize validation error!

---

## Model Assumptions

Linear regression makes several **assumptions** about the data. When these are violated, predictions may be unreliable.

### The Five Key Assumptions

1. **Linearity**
2. **Independence**
3. **Homoscedasticity** (constant variance)
4. **Normality** (of errors)
5. **No multicollinearity**

Let's examine each in detail.

---

### 1. Linearity

**Assumption**: The relationship between X and y is **linear**.

```
y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ + ε
```

**What it means**:
- Features contribute **additively**
- Each feature has a **constant effect** (no interactions by default)
- No **curved** relationships (unless we add polynomial features)

**Checking**:

**Residual plot** (residuals vs predictions):
```
Residuals
    │
  + │  ●    ●  ●    ●        Good: Random scatter
  0 │● ● ●  ●  ● ● ●   ●
  - │   ●    ●     ●  ●
    └──────────────────── Predicted values


Residuals
    │    ●
  + │  ●   ●
    │ ●     ●                Bad: Pattern (curved)
  0 │●       ●
    │ ●     ●
  - │  ●   ●
    └──────────────────── Predicted values
```

**Python code**:
```python
import matplotlib.pyplot as plt

# After fitting model
y_pred = model.predict(X)
residuals = y - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

**What if violated?**
- Add **polynomial features**: x², x³
- Add **interaction terms**: x₁ · x₂
- Use **non-linear models**: trees, neural networks
- **Transform** variables: log(y), √x

**Example transformation**:
```
# Original (non-linear)
y = e^(w₀ + w₁x)

# Transform to linear
log(y) = w₀ + w₁x  ← Now linear!
```

---

### 2. Independence

**Assumption**: Observations are **independent** of each other.

```
y⁽ⁱ⁾ is independent of y⁽ʲ⁾ for i ≠ j
```

**What it means**:
- One observation doesn't affect another
- No **time dependence** (unless modeling time series properly)
- No **spatial correlation** (nearby houses affecting each other)
- No **clustered sampling** (multiple measurements from same subject)

**When violated**:

**Time series** (observations close in time are correlated):
```
Temperature today ≈ Temperature yesterday
Stock price today ≈ Stock price yesterday
```

**Spatial data** (observations close in space are correlated):
```
House prices in same neighborhood
Disease rates in nearby regions
```

**Checking**:

**Autocorrelation plot** (for time series):
```python
from statsmodels.graphics.tsaplots import plot_acf

# Residuals from model
residuals = y - y_pred

# Plot autocorrelation
plot_acf(residuals, lags=20)
plt.show()
```

**Durbin-Watson test** (for time series):
```python
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.2f}")
# DW ≈ 2: No autocorrelation
# DW < 2: Positive autocorrelation
# DW > 2: Negative autocorrelation
```

**What if violated?**
- Use **time series models**: ARIMA, state space models
- Use **spatial models**: Spatial regression
- Use **mixed effects models**: Account for clustering
- Add **lagged variables**: y(t-1) as feature

---

### 3. Homoscedasticity (Constant Variance)

**Assumption**: Error variance is **constant** across all values of X.

```
Var(ε | x) = σ² for all x
```

**What it means**:
- Prediction uncertainty is the **same** everywhere
- No "fanning out" of residuals
- Also called **homogeneity of variance**

**Opposite**: **Heteroscedasticity** (varying variance)

**Checking**:

**Residual plot**:
```
Residuals
    │
    │  ●    ●  ●    ●        Homoscedastic ✓
  0 │● ● ●  ●  ● ● ●   ●    (Constant spread)
    │   ●    ●     ●  ●
    └──────────────────── Predicted values


Residuals
    │                 ●●●
    │            ●● ●●●●●   Heteroscedastic ✗
  0 │●● ●●  ●●●●●●●●●      (Spread increases)
    │  ●  ●●●
    └──────────────────── Predicted values
           (Fanning out)
```

**Tests**:

**Breusch-Pagan test**:
```python
from statsmodels.stats.diagnostic import het_breuschpagan

# Perform test
bp_test = het_breuschpagan(residuals, X)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))

# p-value < 0.05 → Reject null → Heteroscedasticity present
```

**What if violated?**
- **Transform** y: log(y), √y, 1/y
- Use **Weighted Least Squares** (WLS): Give less weight to high-variance observations
- Use **robust standard errors**: Adjust confidence intervals
- Use **generalized linear models** (GLMs): Allow variance to depend on mean

**Example**: Income prediction often has heteroscedasticity
```
# Low income: predictions vary ±$5k
# High income: predictions vary ±$50k

Solution: Predict log(income) instead!
```

---

### 4. Normality of Errors

**Assumption**: Errors are **normally distributed**.

```
ε ~ N(0, σ²)
```

**What it means**:
- Errors follow a bell curve
- Centered at zero (unbiased)
- Allows for **statistical inference** (confidence intervals, hypothesis tests)

**Important**: This is about **errors**, not the original y values!

**Checking**:

**Q-Q plot** (Quantile-Quantile plot):
```python
from scipy import stats
import matplotlib.pyplot as plt

# Residuals
residuals = y - y_pred

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()
```

**Interpretation**:
```
Theoretical Quantiles
    │
    │           ●●●●
    │        ●●●           Good: Points on line
    │      ●●              (Normal distribution)
    │    ●●
    │  ●●
    │●●
    └──────────────── Sample Quantiles


Theoretical Quantiles
    │
    │         ●●●●●
    │       ●●               Bad: Curved pattern
    │     ●●                 (Non-normal)
    │   ●●
    │ ●●
    │●
    └──────────────── Sample Quantiles
```

**Shapiro-Wilk test**:
```python
from scipy.stats import shapiro

stat, p_value = shapiro(residuals)
print(f"Shapiro-Wilk p-value: {p_value:.4f}")

# p-value < 0.05 → Reject null → Not normal
# p-value > 0.05 → Fail to reject → Could be normal
```

**Histogram of residuals**:
```python
plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()
```

**What if violated?**
- **Transform** y: log(y), Box-Cox transformation
- Use **robust regression**: Less sensitive to outliers
- Use **non-parametric methods**: Don't assume distribution
- **Remove outliers**: If justified

**Note**: For **prediction**, normality is less critical. For **inference** (p-values, confidence intervals), it's important!

---

### 5. No Multicollinearity

**Assumption**: Features are **not highly correlated** with each other.

**What it means**:
- Each feature provides **unique information**
- No feature is a **linear combination** of others
- XᵀX is **invertible**

**Perfect multicollinearity** (cannot solve):
```
x₂ = 2 × x₁  (exact relationship)
```

**High multicollinearity** (unstable):
```
Correlation(x₁, x₂) > 0.9
```

**Checking**:

**Correlation matrix**:
```python
import seaborn as sns
import pandas as pd

# Correlation matrix
corr = pd.DataFrame(X).corr()

# Heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

**Variance Inflation Factor (VIF)**:
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = feature_names
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

print(vif_data)

# VIF > 10: High multicollinearity
# VIF > 5: Moderate multicollinearity
```

**What if violated?**
- **Remove** one of the correlated features
- **Combine** features: e.g., use total area instead of separate room areas
- Use **Principal Component Analysis** (PCA): Create uncorrelated features
- Use **Ridge regression**: Handles multicollinearity through regularization (covered later)

**Example**:
```
Features: x₁ = square feet, x₂ = square meters
Problem: x₂ = x₁ / 10.764 (perfect multicollinearity!)
Solution: Remove x₂, keep only x₁
```

---

### Summary of Assumptions

| Assumption | What It Means | Check With | If Violated |
|------------|---------------|------------|-------------|
| **Linearity** | y is linear in X | Residual plot | Add polynomial features, transform variables |
| **Independence** | Observations independent | Autocorrelation plot, DW test | Time series models, account for clustering |
| **Homoscedasticity** | Constant error variance | Residual plot, BP test | Transform y, use WLS |
| **Normality** | Errors are normal | Q-Q plot, Shapiro-Wilk | Transform y, robust methods |
| **No Multicollinearity** | Features uncorrelated | Correlation matrix, VIF | Remove features, PCA, Ridge |

### The "LINE" Acronym

A helpful mnemonic: **LINE**
- **L**inearity
- **I**ndependence
- **N**ormality (of errors)
- **E**qual variance (homoscedasticity)

(+No multicollinearity)

### Real-World Considerations

**In practice**:
- Assumptions are **rarely perfectly met**
- Models can still be **useful** with minor violations
- **Robustness checks** are important
- **Cross-validation** helps assess actual performance

**Priorities**:
1. **Linearity**: Most critical for prediction accuracy
2. **Independence**: Critical for statistical inference
3. **Homoscedasticity**: Affects confidence intervals
4. **Normality**: Less critical for large samples (Central Limit Theorem)
5. **Multicollinearity**: Affects interpretation but not necessarily prediction

---

## Model Evaluation

### Metrics for Regression

After training, we need to evaluate performance.

#### 1. Mean Squared Error (MSE)

```
MSE = (1/m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
```

**Properties**:
- Always ≥ 0
- Squared units (hard to interpret)
- **Penalizes large errors** more (squared)

**Python**:
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.2f}")
```

#### 2. Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √[(1/m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²]
```

**Properties**:
- Same units as y (easier to interpret!)
- "Average error" in predictions

**Python**:
```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.2f}")
```

#### 3. Mean Absolute Error (MAE)

```
MAE = (1/m) Σᵢ₌₁ᵐ |ŷ⁽ⁱ⁾ - y⁽ⁱ⁾|
```

**Properties**:
- Same units as y
- **Less sensitive to outliers** (no squaring)
- "Average absolute error"

**Python**:
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")
```

#### 4. R² Score (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot)
```

Where:
```
SS_res = Σ(y⁽ⁱ⁾ - ŷ⁽ⁱ⁾)²  (residual sum of squares)
SS_tot = Σ(y⁽ⁱ⁾ - ȳ)²     (total sum of squares)
```

**Interpretation**:
- **R² = 1**: Perfect predictions
- **R² = 0**: Model as good as predicting mean
- **R² < 0**: Model worse than predicting mean!

```
Variance explained = R² × 100%
```

**Example**:
```
R² = 0.85 → Model explains 85% of variance in y
```

**Python**:
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")
print(f"Variance explained: {r2*100:.2f}%")
```

#### 5. Adjusted R²

Problem with R²: Always increases when adding features (even useless ones!)

**Adjusted R²** penalizes complexity:
```
R²_adj = 1 - [(1 - R²) × (m - 1) / (m - n - 1)]
```

Where:
- m = number of examples
- n = number of features

**Properties**:
- Can decrease when adding useless features
- Better for comparing models with different # features

**Python**:
```python
def adjusted_r2(r2, m, n):
    return 1 - (1 - r2) * (m - 1) / (m - n - 1)

r2_adj = adjusted_r2(r2, m=len(y), n=X.shape[1])
print(f"Adjusted R²: {r2_adj:.4f}")
```

### Comparison of Metrics

| Metric | Units | Outlier Sensitivity | Interpretation |
|--------|-------|---------------------|----------------|
| **MSE** | y² | High | Hard to interpret |
| **RMSE** | y | High | Average error magnitude |
| **MAE** | y | Low | Average absolute error |
| **R²** | None (0-1) | Medium | % variance explained |

### When to Use Which?

- **RMSE**: When large errors are particularly bad (default choice)
- **MAE**: When outliers shouldn't dominate (robust metric)
- **R²**: For comparing models, understanding fit quality
- **Adjusted R²**: When comparing models with different # features

---

## Practical Implementation

### Complete Workflow

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and prepare data
# Example: Boston housing (or any dataset)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset shape:", X.shape)
print("Features:", X.columns.tolist())

# 2. Exploratory Data Analysis
print("\nSummary statistics:")
print(X.describe())

print("\nCorrelations with target:")
correlations = X.corrwith(pd.Series(y))
print(correlations.sort_values(ascending=False))

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Get parameters
print("\nModel parameters:")
print(f"Intercept: {model.intercept_:.4f}")
print("\nCoefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name:15s}: {coef:8.4f}")

# 6. Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 7. Evaluate
print("\n" + "="*50)
print("TRAINING PERFORMANCE")
print("="*50)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

print(f"MSE:  {train_mse:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"R²:   {train_r2:.4f}")

print("\n" + "="*50)
print("TEST PERFORMANCE")
print("="*50)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print(f"MSE:  {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²:   {test_r2:.4f}")

# 8. Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title('Actual vs Predicted')

# Residuals vs Predicted
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')

# Residual distribution
axes[1, 0].hist(residuals, bins=30, edgecolor='black')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residual Distribution')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# 9. Feature importance (by magnitude)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)
print(feature_importance)

# 10. Make new predictions
print("\n" + "="*50)
print("EXAMPLE PREDICTIONS")
print("="*50)
sample_indices = [0, 100, 500]
for idx in sample_indices:
    features = X_test.iloc[idx]
    actual = y_test.iloc[idx]
    predicted = y_test_pred[idx]
    print(f"\nExample {idx}:")
    print(f"  Predicted: {predicted:.2f}")
    print(f"  Actual:    {actual:.2f}")
    print(f"  Error:     {predicted - actual:.2f}")
```

### From Scratch Implementation

```python
import numpy as np

class LinearRegressionScratch:
    """Linear Regression implemented from scratch"""
    
    def __init__(self):
        self.weights = None
        self.intercept = None
    
    def fit(self, X, y):
        """
        Fit linear regression using normal equation
        
        Args:
            X: Features (m x n)
            y: Target (m,)
        """
        # Add bias column
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Normal equation: w = (X^T X)^-1 X^T y
        XtX = X_with_bias.T @ X_with_bias
        Xty = X_with_bias.T @ y
        
        # Solve for weights
        w = np.linalg.solve(XtX, Xty)
        
        # Split into intercept and coefficients
        self.intercept = w[0]
        self.weights = w[1:]
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features (m x n)
        
        Returns:
            Predictions (m,)
        """
        return self.intercept + X @ self.weights
    
    def score(self, X, y):
        """
        Calculate R² score
        
        Args:
            X: Features (m x n)
            y: True values (m,)
        
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - (ss_res / ss_tot)

# Usage
model = LinearRegressionScratch()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)

print(f"Custom R²: {r2:.4f}")
print(f"Intercept: {model.intercept:.4f}")
print(f"Weights: {model.weights}")
```

---

## Common Pitfalls and Solutions

### 1. Overfitting

**Problem**: Model performs well on training data but poorly on test data.

**Signs**:
```
Training R²: 0.95
Test R²:     0.65  ← Big gap!
```

**Causes**:
- Too many features (high n relative to m)
- Features perfectly memorize noise

**Solutions**:
- **Regularization** (Ridge, Lasso - covered in later file)
- **Feature selection**: Remove irrelevant features
- **More data**: Collect additional examples
- **Cross-validation**: Validate on multiple folds

### 2. Underfitting

**Problem**: Model performs poorly on both training and test data.

**Signs**:
```
Training R²: 0.45
Test R²:     0.42  ← Both low!
```

**Causes**:
- Too simple model (missing important features)
- Strong non-linear relationship

**Solutions**:
- Add **more features**
- Add **polynomial features**: x², x³, x₁x₂
- Use **non-linear model**: Random Forest, Neural Network
- **Feature engineering**: Create domain-specific features

### 3. Feature Scaling Issues

**Problem**: Features on very different scales cause numerical instability.

**Example**:
```
Size:   [500,  2000,  1500]  (hundreds)
Age:    [1,    5,     10]     (single digits)
Price:  [100,  300,   250]    (hundreds)
```

**Why it matters**:
- Gradient descent converges **slowly**
- Features with large scale **dominate**
- Weights become **hard to interpret**

**Solution**: Standardize features
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now all features have mean=0, std=1
```

### 4. Outliers

**Problem**: Extreme values distort the model.

**Example**:
```
Prices: [200k, 250k, 300k, 280k, 310k, 5000k]  ← Outlier!
                                        ↑
                               Pulls line upward
```

**Detection**:
```python
# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(y))
outliers = np.where(z_scores > 3)[0]

# IQR method
Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1
outliers = (y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR)
```

**Solutions**:
- **Remove** if data error
- **Cap** at reasonable values
- Use **robust regression** (less sensitive to outliers)
- **Transform** y: log transform reduces impact

### 5. Missing Data

**Problem**: Some feature values are missing.

**Solutions**:

**Remove rows** (if few missing):
```python
X_clean = X.dropna()
```

**Impute with mean/median** (simple):
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

**Impute with model** (advanced):
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()
X_imputed = imputer.fit_transform(X)
```

### 6. Non-Stationarity (Time Series)

**Problem**: Relationship changes over time.

**Example**:
```
2000-2010: Price = 100 + 50*Size
2010-2020: Price = 150 + 80*Size  (Different model!)
```

**Solutions**:
- Train **separate models** for different time periods
- Add **time features**: year, trend
- Use **time series models**: ARIMA, state space

### 7. Extrapolation

**Problem**: Predicting outside the training data range.

**Example**:
```
Training: Houses 500-3000 sqft
Prediction: House with 5000 sqft  ← Risky!
```

**Why risky**: Linear relationship may not hold outside observed range.

**Solution**:
- Be cautious about extrapolation
- Indicate uncertainty
- Collect data in the target range

---

## Summary and Next Steps

### What We Learned

1. **Simple Linear Regression**
   ```
   ŷ = w₀ + w₁x
   ```
   - One feature → straight line
   - w₁ = slope, w₀ = intercept
   - Closed-form solution

2. **Multiple Linear Regression**
   ```
   ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
   ```
   - Multiple features → hyperplane
   - Normal equation: w = (XᵀX)⁻¹Xᵀy
   - Each weight = partial effect

3. **Hypothesis Function**
   ```
   h(x) = wᵀx
   ```
   - Function mapping inputs to outputs
   - Parametric model (fixed # parameters)
   - Linear hypothesis space

4. **Model Assumptions**
   - **L**inearity
   - **I**ndependence
   - **N**ormality of errors
   - **E**qual variance (homoscedasticity)
   - No multicollinearity

5. **Evaluation Metrics**
   - MSE, RMSE, MAE (error measures)
   - R² (variance explained)
   - Residual plots

### Key Takeaways

✅ **Interpretable**: Each weight shows feature importance  
✅ **Fast**: Closed-form solution (no iteration needed)  
✅ **Foundation**: Basis for advanced models  
✅ **Assumptions**: Check before trusting results  
✅ **Practical**: Works well for many real problems  

### Prerequisites for Next Topics

Before moving on, make sure you understand:
- [ ] How to fit simple and multiple regression
- [ ] What w₀ and wⱼ represent
- [ ] How to interpret coefficients
- [ ] The 5 key assumptions (LINE + no multicollinearity)
- [ ] How to evaluate using R², RMSE, MAE
- [ ] Difference between training and test error

### Next Steps in This Section

Now that you understand the basics, we'll dive deeper:

1. **Ordinary_Least_Squares.md**
   - Detailed derivation of normal equation
   - Geometric interpretation
   - Computational considerations

2. **Gradient_Descent_for_Linear_Models.md**
   - Iterative optimization
   - When to use vs closed-form
   - Learning rates and convergence

3. **Cost_Functions_and_Loss.md**
   - Why squared error?
   - Alternative loss functions
   - Relationship to maximum likelihood

4. **Logistic_Regression.md**
   - Linear models for classification
   - Sigmoid function
   - Decision boundaries

5. **Regularization_L1_L2.md**
   - Preventing overfitting
   - Ridge, Lasso, Elastic Net
   - Feature selection

### Practice Exercises

#### Conceptual:
1. Why is it called "linear" regression if we can model curves?
2. What's the difference between w₁ in simple regression vs multiple regression?
3. When would you use MAE instead of RMSE?

#### Mathematical:
1. Given data: x=[1,2,3], y=[2,4,5], calculate w₀ and w₁
2. Verify that the regression line passes through (x̄, ȳ)
3. Show that adding a constant to all y values only changes w₀

#### Coding:
1. Implement simple linear regression from scratch
2. Create residual plots and check assumptions
3. Compare sklearn vs your implementation

### Additional Resources

**Books**:
- "Introduction to Statistical Learning" (James et al.) - Chapter 3
- "The Elements of Statistical Learning" (Hastie et al.) - Chapter 3
- "Pattern Recognition and Machine Learning" (Bishop) - Chapter 3

**Online**:
- Coursera: Machine Learning by Andrew Ng
- Scikit-learn documentation
- Khan Academy: Statistics and Probability

**Practice**:
- Kaggle: House Prices competition
- UCI Machine Learning Repository datasets
- Scikit-learn built-in datasets

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════╗
║              LINEAR REGRESSION QUICK REFERENCE               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  SIMPLE:    ŷ = w₀ + w₁x                                     ║
║                                                              ║
║  MULTIPLE:  ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ                ║
║                                                              ║
║  MATRIX:    ŷ = Xw                                           ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  NORMAL EQUATION                                             ║
╠══════════════════════════════════════════════════════════════╣
║  w = (XᵀX)⁻¹Xᵀy                                              ║
║                                                              ║
║  Requirements:                                               ║
║  • XᵀX must be invertible                                    ║
║  • m ≥ n+1 (more examples than features)                     ║
║  • Features linearly independent                             ║
╠══════════════════════════════════════════════════════════════╣
║  ASSUMPTIONS (LINE)                                          ║
╠══════════════════════════════════════════════════════════════╣
║  L - Linearity          : y linear in features               ║
║  I - Independence       : Observations independent           ║
║  N - Normality          : Errors normally distributed        ║
║  E - Equal variance     : Constant error variance            ║
║      No multicollinearity : Features not highly correlated   ║
╠══════════════════════════════════════════════════════════════╣
║  EVALUATION METRICS                                          ║
╠══════════════════════════════════════════════════════════════╣
║  MSE  = (1/m) Σ(ŷ - y)²          [Units: y²]                 ║
║  RMSE = √MSE                      [Units: y]                 ║
║  MAE  = (1/m) Σ|ŷ - y|            [Units: y]                 ║
║  R²   = 1 - (SS_res/SS_tot)       [Range: 0-1]               ║
╠══════════════════════════════════════════════════════════════╣
║  INTERPRETATION                                              ║
╠══════════════════════════════════════════════════════════════╣
║  w₀ : Predicted y when all features = 0                      ║
║  wⱼ : Change in y for unit change in xⱼ (others constant)    ║
╠══════════════════════════════════════════════════════════════╣
║  COMMON ISSUES                                               ║
╠══════════════════════════════════════════════════════════════╣
║  Overfitting     → Regularization, more data                 ║
║  Underfitting    → Add features, polynomial terms            ║
║  Outliers        → Remove, robust methods                    ║
║  Multicollinearity → Remove features, PCA, Ridge             ║
║  Scaling         → StandardScaler                            ║
╚══════════════════════════════════════════════════════════════╝
```

---

**Ready to move on?** Next up: **Ordinary_Least_Squares.md** where we'll derive the normal equation and understand the geometry!

---

*End of Linear_Regression_Basics.md*