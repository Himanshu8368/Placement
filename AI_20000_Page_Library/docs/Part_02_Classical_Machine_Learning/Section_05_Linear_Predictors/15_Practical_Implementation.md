# Practical Implementation

## Table of Contents
1. [Introduction](#introduction)
2. [Complete Workflow](#complete-workflow)
3. [NumPy Implementation from Scratch](#numpy-implementation-from-scratch)
4. [Scikit-Learn Usage](#scikit-learn-usage)
5. [Best Practices](#best-practices)
6. [Common Pitfalls](#common-pitfalls)
7. [Real-World Examples](#real-world-examples)
8. [Production Considerations](#production-considerations)
9. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
10. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction

### What This Chapter Covers

This chapter provides **practical, production-ready** code for implementing linear models.

**You will learn**:
- Complete end-to-end workflows
- How to implement models from scratch (for understanding)
- How to use scikit-learn properly (for production)
- Best practices for real projects
- Common mistakes and how to avoid them

### Why Both From-Scratch and Scikit-Learn?

**From scratch** (NumPy):
- ✅ Deep understanding of algorithms
- ✅ Debugging and customization
- ✅ Job interviews and research
- ❌ Slower, more error-prone

**Scikit-learn**:
- ✅ Production-ready, tested
- ✅ Fast, optimized
- ✅ Consistent API
- ✅ Industry standard
- ❌ Less control, "black box" feeling

**Best approach**: Understand from-scratch, use scikit-learn in production.

---

## Complete Workflow

### The Standard ML Workflow

```
1. Data Loading & Exploration
   ↓
2. Data Preprocessing
   ↓
3. Train-Test Split
   ↓
4. Feature Engineering
   ↓
5. Feature Scaling
   ↓
6. Model Training
   ↓
7. Hyperparameter Tuning
   ↓
8. Model Evaluation
   ↓
9. Error Analysis
   ↓
10. Deployment
```

### Complete Example: House Price Prediction

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================
print("="*70)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("="*70)

# Load data (example: California housing)
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"\nDataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"\nFirst few rows:")
print(X.head())
print(f"\nTarget statistics:")
print(f"Mean: {y.mean():.2f}, Std: {y.std():.2f}, Min: {y.min():.2f}, Max: {y.max():.2f}")

# Check for missing values
print(f"\nMissing values:")
print(X.isnull().sum())

# Basic statistics
print(f"\nFeature statistics:")
print(X.describe())

# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================
print("\n" + "="*70)
print("STEP 2: DATA PREPROCESSING")
print("="*70)

# Check for outliers (using IQR method)
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[column] < Q1 - 1.5*IQR) | (df[column] > Q3 + 1.5*IQR)).sum()
    return outliers

print("\nOutliers per feature:")
for col in X.columns:
    outliers = detect_outliers_iqr(X, col)
    print(f"{col}: {outliers} outliers ({outliers/len(X)*100:.1f}%)")

# Handle outliers (cap at 5th and 95th percentiles)
for col in X.columns:
    lower = X[col].quantile(0.05)
    upper = X[col].quantile(0.95)
    X[col] = X[col].clip(lower, upper)

print("\nOutliers handled: Capped at 5th and 95th percentiles")

# ============================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================
print("\n" + "="*70)
print("STEP 3: TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Split ratio: {X_train.shape[0]/len(X)*100:.0f}% train, {X_test.shape[0]/len(X)*100:.0f}% test")

# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================
print("\n" + "="*70)
print("STEP 4: FEATURE ENGINEERING")
print("="*70)

# Create polynomial features (degree 2 for interactions)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(f"\nOriginal features: {X_train.shape[1]}")
print(f"After polynomial features (degree 2): {X_train_poly.shape[1]}")
print(f"New features include: {len(poly.get_feature_names_out()) - X_train.shape[1]} interaction terms")

# ============================================================
# STEP 5: FEATURE SCALING
# ============================================================
print("\n" + "="*70)
print("STEP 5: FEATURE SCALING")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

print(f"\nFeatures scaled using StandardScaler")
print(f"Mean after scaling: {X_train_scaled.mean(axis=0)[:5]} (first 5 features)")
print(f"Std after scaling: {X_train_scaled.std(axis=0)[:5]} (first 5 features)")

# ============================================================
# STEP 6: MODEL TRAINING
# ============================================================
print("\n" + "="*70)
print("STEP 6: MODEL TRAINING")
print("="*70)

# Train baseline model (no regularization)
from sklearn.linear_model import LinearRegression
baseline = LinearRegression()
baseline.fit(X_train_scaled, y_train)

# Train Ridge model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

print("\nModels trained:")
print("1. Baseline (Linear Regression)")
print("2. Ridge Regression (α=1.0)")

# ============================================================
# STEP 7: HYPERPARAMETER TUNING
# ============================================================
print("\n" + "="*70)
print("STEP 7: HYPERPARAMETER TUNING")
print("="*70)

# Grid search for optimal alpha
param_grid = {'alpha': np.logspace(-3, 3, 50)}
grid_search = GridSearchCV(
    Ridge(), 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

best_alpha = grid_search.best_params_['alpha']
best_model = grid_search.best_estimator_

print(f"\nBest alpha found: {best_alpha:.4f}")
print(f"Best CV score (MSE): {-grid_search.best_score_:.4f}")

# ============================================================
# STEP 8: MODEL EVALUATION
# ============================================================
print("\n" + "="*70)
print("STEP 8: MODEL EVALUATION")
print("="*70)

models = {
    'Baseline': baseline,
    'Ridge (α=1.0)': ridge,
    'Ridge (optimized)': best_model
}

results = {}
for name, model in models.items():
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

print(f"\n{'Model':<20} {'Train MSE':<12} {'Test MSE':<12} {'Test R²':<10}")
print("-"*70)
for name, metrics in results.items():
    print(f"{name:<20} {metrics['train_mse']:<12.4f} {metrics['test_mse']:<12.4f} "
          f"{metrics['test_r2']:<10.4f}")

# ============================================================
# STEP 9: ERROR ANALYSIS
# ============================================================
print("\n" + "="*70)
print("STEP 9: ERROR ANALYSIS")
print("="*70)

y_pred = best_model.predict(X_test_scaled)
residuals = y_test - y_pred

print(f"\nResidual statistics:")
print(f"Mean: {residuals.mean():.4f}")
print(f"Std: {residuals.std():.4f}")
print(f"Min: {residuals.min():.4f}")
print(f"Max: {residuals.max():.4f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Predicted vs Actual
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
       'r--', linewidth=2)
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
ax.set_title('Predicted vs Actual')
ax.grid(True, alpha=0.3)

# Plot 2: Residuals
ax = axes[0, 1]
ax.scatter(y_pred, residuals, alpha=0.5)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Price')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
ax.grid(True, alpha=0.3)

# Plot 3: Residual distribution
ax = axes[1, 0]
ax.hist(residuals, bins=50, edgecolor='black')
ax.set_xlabel('Residuals')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution')
ax.grid(True, alpha=0.3)

# Plot 4: Feature importance (absolute coefficients)
ax = axes[1, 1]
feature_importance = np.abs(best_model.coef_)
top_n = 10
top_indices = np.argsort(feature_importance)[-top_n:]
ax.barh(range(top_n), feature_importance[top_indices])
ax.set_xlabel('Absolute Coefficient')
ax.set_ylabel('Feature Index')
ax.set_title(f'Top {top_n} Most Important Features')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('house_price_analysis.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'house_price_analysis.png'")

# ============================================================
# STEP 10: MODEL PERSISTENCE
# ============================================================
print("\n" + "="*70)
print("STEP 10: MODEL PERSISTENCE")
print("="*70)

import joblib

# Save model and preprocessing objects
joblib.dump(best_model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly_features.pkl')

print("\nModel and preprocessing saved:")
print("- house_price_model.pkl")
print("- scaler.pkl")
print("- poly_features.pkl")

# Example: Load and use
loaded_model = joblib.load('house_price_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_poly = joblib.load('poly_features.pkl')

# Predict on new data
X_new = X_test[:5]  # Take first 5 test samples as example
X_new_poly = loaded_poly.transform(X_new)
X_new_scaled = loaded_scaler.transform(X_new_poly)
predictions = loaded_model.predict(X_new_scaled)

print(f"\nExample predictions on new data:")
for i, (actual, pred) in enumerate(zip(y_test[:5], predictions)):
    print(f"Sample {i+1}: Actual={actual:.2f}, Predicted={pred:.2f}, "
          f"Error={abs(actual-pred):.2f}")

print("\n" + "="*70)
print("WORKFLOW COMPLETE!")
print("="*70)
```

---

## NumPy Implementation from Scratch

### Linear Regression (OLS)

```python
import numpy as np

class LinearRegressionNumPy:
    """
    Linear Regression implemented from scratch using NumPy
    
    Implements both gradient descent and closed-form solution
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, method='closed_form'):
        """
        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of iterations for gradient descent
            method: 'gradient_descent' or 'closed_form'
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the model
        
        Args:
            X: Feature matrix (m × n)
            y: Target vector (m,)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        m, n = X.shape
        
        if self.method == 'closed_form':
            self._fit_closed_form(X, y)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y)
        else:
            raise ValueError("method must be 'closed_form' or 'gradient_descent'")
        
        return self
    
    def _fit_closed_form(self, X, y):
        """Fit using normal equation"""
        # Add bias column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation: w = (X^T X)^-1 X^T y
        w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        self.bias = w[0, 0]
        self.weights = w[1:].flatten()
    
    def _fit_gradient_descent(self, X, y):
        """Fit using gradient descent"""
        m, n = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # Predictions
            y_pred = self.predict(X).reshape(-1, 1)
            
            # Compute cost
            cost = (1/(2*m)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/m) * (X.T @ (y_pred - y)).flatten()
            db = (1/m) * np.sum(y_pred - y)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if (i+1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Cost: {cost:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Compute R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100)
    
    # Closed-form solution
    print("="*60)
    print("LINEAR REGRESSION (Closed Form)")
    print("="*60)
    
    model_cf = LinearRegressionNumPy(method='closed_form')
    model_cf.fit(X, y)
    
    print(f"\nWeights: {model_cf.weights}")
    print(f"Bias: {model_cf.bias:.4f}")
    print(f"R² score: {model_cf.score(X, y):.4f}")
    
    # Gradient descent
    print("\n" + "="*60)
    print("LINEAR REGRESSION (Gradient Descent)")
    print("="*60)
    
    model_gd = LinearRegressionNumPy(
        method='gradient_descent',
        learning_rate=0.1,
        n_iterations=1000
    )
    model_gd.fit(X, y)
    
    print(f"\nWeights: {model_gd.weights}")
    print(f"Bias: {model_gd.bias:.4f}")
    print(f"R² score: {model_gd.score(X, y):.4f}")
    
    # Plot convergence
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(model_gd.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Convergence')
    plt.grid(True, alpha=0.3)
    plt.show()
```

### Ridge Regression

```python
class RidgeRegressionNumPy:
    """Ridge Regression with L2 regularization"""
    
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000, 
                 method='closed_form'):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """Train the model"""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        if self.method == 'closed_form':
            self._fit_closed_form(X, y)
        else:
            self._fit_gradient_descent(X, y)
        
        return self
    
    def _fit_closed_form(self, X, y):
        """Closed-form solution with L2 regularization"""
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n = X_b.shape[1]
        
        # Ridge: w = (X^T X + αI)^-1 X^T y
        I = np.eye(n)
        I[0, 0] = 0  # Don't regularize bias
        
        w = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        
        self.bias = w[0, 0]
        self.weights = w[1:].flatten()
    
    def _fit_gradient_descent(self, X, y):
        """Gradient descent with L2 regularization"""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for i in range(self.n_iterations):
            y_pred = self.predict(X).reshape(-1, 1)
            
            # Cost with L2 penalty
            mse = (1/(2*m)) * np.sum((y_pred - y) ** 2)
            l2_penalty = (self.alpha/(2*m)) * np.sum(self.weights ** 2)
            cost = mse + l2_penalty
            self.cost_history.append(cost)
            
            # Gradients with L2 penalty
            dw = (1/m) * (X.T @ (y_pred - y)).flatten() + (self.alpha/m) * self.weights
            db = (1/m) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Compute R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
```

### Logistic Regression

```python
class LogisticRegressionNumPy:
    """Logistic Regression from scratch"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha  # L2 regularization
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    @staticmethod
    def _sigmoid(z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """Train the model"""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        m, n = X.shape
        
        # Initialize
        self.weights = np.zeros(n)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z).reshape(-1, 1)
            
            # Compute cost (binary cross-entropy + L2)
            epsilon = 1e-15
            y_pred_clip = np.clip(y_pred, epsilon, 1 - epsilon)
            ce_loss = -(1/m) * np.sum(
                y * np.log(y_pred_clip) + (1-y) * np.log(1 - y_pred_clip)
            )
            l2_penalty = (self.alpha/(2*m)) * np.sum(self.weights ** 2)
            cost = ce_loss + l2_penalty
            self.cost_history.append(cost)
            
            # Gradients
            dw = (1/m) * (X.T @ (y_pred - y)).flatten()
            db = (1/m) * np.sum(y_pred - y)
            
            # Add L2 gradient
            if self.alpha > 0:
                dw += (self.alpha/m) * self.weights
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (i+1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Cost: {cost:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X = np.array(X)
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score(self, X, y):
        """Compute accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate binary classification data
    X, y = make_classification(n_samples=500, n_features=10, 
                               n_informative=8, random_state=42)
    
    # Train logistic regression
    print("="*60)
    print("LOGISTIC REGRESSION")
    print("="*60)
    
    model = LogisticRegressionNumPy(
        learning_rate=0.1,
        n_iterations=1000,
        alpha=0.01
    )
    model.fit(X, y)
    
    print(f"\nAccuracy: {model.score(X, y) * 100:.2f}%")
    
    # Plot cost
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(model.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (Cross-Entropy + L2)')
    plt.title('Logistic Regression Training')
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## Scikit-Learn Usage

### Basic Usage Patterns

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

# ============================================================
# PATTERN 1: Simple Linear Regression
# ============================================================
print("PATTERN 1: Simple Linear Regression")
print("-"*60)

# Create model
model = LinearRegression()

# Fit
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")

# ============================================================
# PATTERN 2: Ridge Regression with Scaling
# ============================================================
print("\n" + "="*60)
print("PATTERN 2: Ridge Regression with Scaling")
print("-"*60)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Predict
y_pred_ridge = ridge.predict(X_test_scaled)

print(f"Ridge R²: {ridge.score(X_test_scaled, y_test):.4f}")

# ============================================================
# PATTERN 3: Using Pipeline (RECOMMENDED)
# ============================================================
print("\n" + "="*60)
print("PATTERN 3: Using Pipeline (RECOMMENDED)")
print("-"*60)

from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

# Fit (automatically scales then trains)
pipeline.fit(X_train, y_train)

# Predict (automatically scales test data)
y_pred_pipeline = pipeline.predict(X_test)

print(f"Pipeline R²: {pipeline.score(X_test, y_test):.4f}")

# ============================================================
# PATTERN 4: Cross-Validation
# ============================================================
print("\n" + "="*60)
print("PATTERN 4: Cross-Validation")
print("-"*60)

from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5,
    scoring='r2'
)

print(f"CV R² scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ============================================================
# PATTERN 5: Hyperparameter Tuning with GridSearchCV
# ============================================================
print("\n" + "="*60)
print("PATTERN 5: Hyperparameter Tuning")
print("-"*60)

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# Grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# ============================================================
# PATTERN 6: Polynomial Features with Pipeline
# ============================================================
print("\n" + "="*60)
print("PATTERN 6: Polynomial Features")
print("-"*60)

from sklearn.preprocessing import PolynomialFeatures

poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

poly_pipeline.fit(X_train, y_train)

print(f"Polynomial pipeline R²: {poly_pipeline.score(X_test, y_test):.4f}")

# ============================================================
# PATTERN 7: Logistic Regression for Classification
# ============================================================
print("\n" + "="*60)
print("PATTERN 7: Logistic Regression")
print("-"*60)

from sklearn.datasets import make_classification

# Generate classification data
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, 
                                   random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Create classifier pipeline
clf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=1000))
])

clf_pipeline.fit(X_train_clf, y_train_clf)

# Predict
y_pred_clf = clf_pipeline.predict(X_test_clf)
y_proba_clf = clf_pipeline.predict_proba(X_test_clf)

print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_clf) * 100:.2f}%")
print(f"Probability shape: {y_proba_clf.shape}")

# ============================================================
# PATTERN 8: Model Persistence
# ============================================================
print("\n" + "="*60)
print("PATTERN 8: Model Persistence")
print("-"*60)

import joblib

# Save model
joblib.dump(grid_search.best_estimator_, 'best_model.pkl')

# Load model
loaded_model = joblib.load('best_model.pkl')

# Use loaded model
predictions = loaded_model.predict(X_test[:5])
print(f"Predictions from loaded model: {predictions}")

print("\nModel saved as 'best_model.pkl'")
```

### Advanced Scikit-Learn Patterns

```python
from sklearn.linear_model import ElasticNet, RidgeCV, LassoCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# ============================================================
# ELASTIC NET
# ============================================================
print("="*60)
print("ELASTIC NET (L1 + L2 Regularization)")
print("="*60)

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)

print(f"ElasticNet R²: {elastic.score(X_test_scaled, y_test):.4f}")
print(f"Non-zero coefficients: {np.sum(np.abs(elastic.coef_) > 1e-5)}")

# ============================================================
# AUTOMATIC CV FOR REGULARIZATION (RidgeCV, LassoCV)
# ============================================================
print("\n" + "="*60)
print("AUTOMATIC CROSS-VALIDATION")
print("="*60)

# RidgeCV automatically finds best alpha
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"RidgeCV best alpha: {ridge_cv.alpha_:.4f}")
print(f"RidgeCV R²: {ridge_cv.score(X_test_scaled, y_test):.4f}")

# LassoCV
lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

print(f"\nLassoCV best alpha: {lasso_cv.alpha_:.4f}")
print(f"LassoCV R²: {lasso_cv.score(X_test_scaled, y_test):.4f}")
print(f"Selected features: {np.sum(np.abs(lasso_cv.coef_) > 1e-5)}")

# ============================================================
# LEARNING CURVES
# ============================================================
print("\n" + "="*60)
print("LEARNING CURVES")
print("="*60)

train_sizes, train_scores, val_scores = learning_curve(
    pipeline,
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='r2',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training score')
plt.plot(train_sizes, val_mean, 's-', label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Learning curves plotted successfully")
```

---

## Best Practices

### 1. Always Split Data First

```python
# ✅ CORRECT: Split before any preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit!

# ❌ WRONG: Fitting on entire dataset causes data leakage
# scaler.fit(X)  # DON'T DO THIS!
# X_scaled = scaler.transform(X)
# X_train, X_test = train_test_split(X_scaled)  # Data leakage!
```

### 2. Use Pipelines

```python
# ✅ CORRECT: Use Pipeline for clean, reproducible code
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

# Everything happens automatically in correct order
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# ❌ AVOID: Manual step-by-step (error-prone)
# poly = PolynomialFeatures(degree=2)
# X_train_poly = poly.fit_transform(X_train)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_poly)
# # Easy to forget steps or do them in wrong order!
```

### 3. Always Scale Features for Regularized Models

```python
# ✅ CORRECT: Scale when using Ridge, Lasso, ElasticNet
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scale first!
    ('ridge', Ridge(alpha=1.0))
])

# ❌ WRONG: Not scaling with regularization
# ridge = Ridge(alpha=1.0)
# ridge.fit(X_train, y_train)  # Unfair penalty on different scales!
```

### 4. Use Cross-Validation for Hyperparameter Tuning

```python
# ✅ CORRECT: Use GridSearchCV or RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'ridge__alpha': np.logspace(-3, 3, 50)}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# ❌ WRONG: Tuning on test set
# for alpha in alphas:
#     model = Ridge(alpha=alpha)
#     model.fit(X_train, y_train)
#     score = model.score(X_test, y_test)  # DON'T TUNE ON TEST SET!
```

### 5. Set Random Seeds for Reproducibility

```python
# ✅ CORRECT: Set seeds everywhere
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # Set random_state!
)

model = Ridge(alpha=1.0, random_state=42)  # If model supports it
```

### 6. Monitor Training vs Validation Error

```python
# ✅ CORRECT: Check for overfitting
from sklearn.model_selection import cross_val_score

# Training score
train_score = model.score(X_train, y_train)

# Validation score (via CV)
val_scores = cross_val_score(model, X_train, y_train, cv=5)
val_score = val_scores.mean()

print(f"Training R²: {train_score:.4f}")
print(f"Validation R²: {val_score:.4f}")
print(f"Gap: {train_score - val_score:.4f}")

if train_score - val_score > 0.1:
    print("⚠ WARNING: Possible overfitting!")
```

### 7. Handle Categorical Variables Properly

```python
# ✅ CORRECT: Encode categorical variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identify categorical columns
categorical_features = ['category', 'location']
numerical_features = ['age', 'income', 'score']

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('ridge', Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)
```

### 8. Save Everything Needed for Deployment

```python
# ✅ CORRECT: Save model AND preprocessing
import joblib

# Save complete pipeline (includes preprocessing)
joblib.dump(pipeline, 'complete_model.pkl')

# Save feature names
joblib.dump(feature_names, 'feature_names.pkl')

# Save any additional metadata
metadata = {
    'model_version': '1.0',
    'training_date': '2024-01-01',
    'features': feature_names,
    'target': 'price',
    'performance': {
        'train_r2': train_r2,
        'test_r2': test_r2
    }
}
joblib.dump(metadata, 'metadata.pkl')
```

### 9. Validate Input Data

```python
# ✅ CORRECT: Validate inputs before prediction
def predict_safely(model, X_new):
    """Safe prediction with validation"""
    # Check for missing values
    if np.any(pd.isna(X_new)):
        raise ValueError("Input contains missing values")
    
    # Check feature count
    expected_features = model.n_features_in_
    if X_new.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {X_new.shape[1]}")
    
    # Check for infinite values
    if np.any(np.isinf(X_new)):
        raise ValueError("Input contains infinite values")
    
    # Make prediction
    return model.predict(X_new)
```

### 10. Document Your Models

```python
# ✅ CORRECT: Document model details
model_documentation = """
Model: Ridge Regression for House Price Prediction
Version: 1.0
Created: 2024-01-01

Features:
- Square footage
- Number of bedrooms
- Location (one-hot encoded)
- Age of house

Preprocessing:
- Polynomial features (degree=2)
- StandardScaler normalization

Hyperparameters:
- Ridge alpha: 1.0
- Selected via 5-fold CV

Performance:
- Training R²: 0.85
- Test R²: 0.82
- RMSE: $15,000

Usage:
    from joblib import load
    model = load('house_price_model.pkl')
    prediction = model.predict(X_new)
"""

# Save documentation
with open('model_documentation.txt', 'w') as f:
    f.write(model_documentation)
```

---

## Common Pitfalls

### Pitfall 1: Data Leakage

```python
# ❌ WRONG: Fitting scaler on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit(X)  # Uses information from test set!
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT: Fit only on training data
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Pitfall 2: Not Scaling Features

```python
# ❌ WRONG: Using Ridge without scaling
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)  # Features on different scales!

# ✅ CORRECT: Always scale with regularization
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])
pipeline.fit(X_train, y_train)
```

### Pitfall 3: Tuning on Test Set

```python
# ❌ WRONG: Evaluating on test set during hyperparameter tuning
best_alpha = None
best_score = -np.inf

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # DON'T DO THIS!
    if score > best_score:
        best_score = score
        best_alpha = alpha

# ✅ CORRECT: Use cross-validation
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(Ridge(), {'alpha': alphas}, cv=5)
grid.fit(X_train, y_train)
best_alpha = grid.best_params_['alpha']
```

### Pitfall 4: Forgetting Polynomial Scaling

```python
# ❌ WRONG: Creating polynomials before scaling
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)  # Scales wrong!

# ✅ CORRECT: Scale AFTER creating polynomials
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)  # Correct order
```

### Pitfall 5: Overfitting to Validation Set

```python
# ❌ WRONG: Repeatedly evaluating on same validation set
X_train_full, X_test = train_test_split(X, test_size=0.2)
X_train, X_val = train_test_split(X_train_full, test_size=0.2)

# Tune based on validation performance
for _ in range(100):  # Many iterations
    # Adjust model based on val performance
    # Eventually overfits to validation set!
    pass

# ✅ CORRECT: Use nested cross-validation
from sklearn.model_selection import cross_val_score

# Outer loop: Model evaluation
# Inner loop: Hyperparameter tuning (via GridSearchCV)
```

### Pitfall 6: Not Handling Missing Data

```python
# ❌ WRONG: Ignoring missing values
model.fit(X_train, y_train)  # May crash if NaN present!

# ✅ CORRECT: Handle missing values explicitly
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Or use in pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])
```

### Pitfall 7: Using Wrong Metrics

```python
# ❌ WRONG: Using accuracy for imbalanced classification
y_train = [0]*95 + [1]*5  # 95% class 0, 5% class 1

model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)  # May be misleading!

# ✅ CORRECT: Use appropriate metrics
from sklearn.metrics import f1_score, roc_auc_score

f1 = f1_score(y_test, model.predict(X_test))
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"F1: {f1:.4f}, AUC: {auc:.4f}")
```

### Pitfall 8: Not Saving Preprocessing

```python
# ❌ WRONG: Only saving model
joblib.dump(model, 'model.pkl')
# Lost: scaler, polynomial features, etc.

# ✅ CORRECT: Save complete pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'complete_pipeline.pkl')

# Now loading and using is simple:
loaded = joblib.load('complete_pipeline.pkl')
predictions = loaded.predict(X_new)  # All preprocessing automatic!
```

### Pitfall 9: Ignoring Multicollinearity

```python
# ❌ WRONG: Using OLS with highly correlated features
# May lead to unstable coefficients

# ✅ CORRECT: Use Ridge regression or remove correlated features
from sklearn.linear_model import Ridge

# Option 1: Use Ridge (handles multicollinearity)
ridge = Ridge(alpha=1.0)

# Option 2: Remove correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_reduced = X.drop(to_drop, axis=1)
```

### Pitfall 10: Not Checking Assumptions

```python
# ✅ CORRECT: Check linear regression assumptions
import matplotlib.pyplot as plt

# 1. Linearity (residuals vs fitted)
y_pred = model.predict(X_train)
residuals = y_train - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

# 2. Normality of residuals
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# 3. Homoscedasticity (constant variance)
plt.scatter(y_pred, np.abs(residuals))
plt.xlabel('Fitted values')
plt.ylabel('Absolute residuals')
plt.title('Scale-Location Plot')
plt.show()
```

---

## Real-World Examples

### Example 1: Customer Churn Prediction

```python
"""
Real-world example: Predicting customer churn
Binary classification using logistic regression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# Generate synthetic customer data
np.random.seed(42)
n_customers = 1000

data = {
    'age': np.random.randint(18, 70, n_customers),
    'tenure_months': np.random.randint(1, 72, n_customers),
    'monthly_charges': np.random.uniform(20, 120, n_customers),
    'total_charges': np.random.uniform(100, 8000, n_customers),
    'num_products': np.random.randint(1, 5, n_customers),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
    'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'], n_customers)
}

# Create target (churn) with some logic
churn_prob = (
    0.1 * (data['contract_type'] == 'Month-to-month') +
    0.05 * (data['tenure_months'] < 12) +
    0.02 * (data['monthly_charges'] > 80) +
    np.random.uniform(0, 0.2, n_customers)
)

data['churn'] = (churn_prob > 0.5).astype(int)

df = pd.DataFrame(data)

print("="*70)
print("CUSTOMER CHURN PREDICTION")
print("="*70)
print(f"\nDataset shape: {df.shape}")
print(f"Churn rate: {df['churn'].mean() * 100:.1f}%")

# Separate features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define preprocessing
categorical_features = ['contract_type', 'payment_method']
numerical_features = ['age', 'tenure_months', 'monthly_charges', 
                     'total_charges', 'num_products']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'classifier__C': [0.01, 0.1, 1.0, 10.0],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear']
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, 
    scoring='roc_auc', n_jobs=-1
)

print("\nTraining model with hyperparameter tuning...")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n" + "="*70)
print("TEST SET PERFORMANCE")
print("="*70)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(best_model, 'churn_prediction_model.pkl')
print("\nModel saved as 'churn_prediction_model.pkl'")
```

### Example 2: Sales Forecasting

```python
"""
Real-world example: Monthly sales forecasting
Time series regression with seasonal features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic monthly sales data
np.random.seed(42)
n_months = 48
dates = pd.date_range('2020-01-01', periods=n_months, freq='MS')

# Trend + seasonality + noise
trend = np.linspace(100, 200, n_months)
seasonality = 20 * np.sin(2 * np.pi * np.arange(n_months) / 12)
noise = np.random.normal(0, 10, n_months)
sales = trend + seasonality + noise

df = pd.DataFrame({
    'date': dates,
    'sales': sales
})

print("="*70)
print("SALES FORECASTING")
print("="*70)
print(f"\nData from {df['date'].min()} to {df['date'].max()}")
print(f"Total months: {len(df)}")

# Feature engineering
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['year'] = df['date'].dt.year
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Lag features
for lag in [1, 2, 3, 6, 12]:
    df[f'sales_lag_{lag}'] = df['sales'].shift(lag)

# Rolling statistics
df['sales_rolling_mean_3'] = df['sales'].rolling(window=3, min_periods=1).mean()
df['sales_rolling_std_3'] = df['sales'].rolling(window=3, min_periods=1).std()

# Drop rows with NaN (from lags)
df = df.dropna()

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['date', 'sales']]
X = df[feature_cols].values
y = df['sales'].values

print(f"\nFeatures: {len(feature_cols)}")
print(f"Samples after feature engineering: {len(df)}")

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

scores = []
predictions = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val_scaled)
    mae = mean_absolute_error(y_val, y_pred)
    scores.append(mae)
    
    print(f"Fold {fold} - MAE: {mae:.2f}")

print(f"\nAverage MAE: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")

# Train final model on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
final_model = Ridge(alpha=1.0)
final_model.fit(X_scaled, y)

# Save
import joblib
joblib.dump(final_model, 'sales_forecast_model.pkl')
joblib.dump(scaler, 'sales_forecast_scaler.pkl')

print("\nModels saved:")
print("- sales_forecast_model.pkl")
print("- sales_forecast_scaler.pkl")
```

---

## Production Considerations

### Model Versioning

```python
"""
Production best practice: Version your models
"""

import joblib
from datetime import datetime
import json

def save_model_with_metadata(model, metadata, base_path='models'):
    """
    Save model with versioning and metadata
    
    Args:
        model: Trained sklearn model/pipeline
        metadata: Dict with model information
        base_path: Directory to save models
    """
    import os
    
    # Create directory if doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate version
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f'{base_path}/model_v{version}.pkl'
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata['version'] = version
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_path'] = model_path
    
    metadata_path = f'{base_path}/metadata_v{version}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    
    return model_path, metadata_path

# Example usage
metadata = {
    'model_type': 'Ridge Regression',
    'task': 'House Price Prediction',
    'features': ['sqft', 'bedrooms', 'age'],
    'performance': {
        'train_r2': 0.85,
        'test_r2': 0.82,
        'test_mae': 15000
    },
    'hyperparameters': {
        'alpha': 1.0
    },
    'preprocessing': ['PolynomialFeatures(degree=2)', 'StandardScaler']
}

save_model_with_metadata(best_model, metadata)
```

### API Deployment Example

```python
"""
Example: Simple REST API for model serving
Using Flask (for production, use FastAPI or similar)
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model on startup
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly_features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict house prices
    
    Expected JSON:
    {
        "features": {
            "sqft": 1500,
            "bedrooms": 3,
            "age": 10,
            ...
        }
    }
    """
    try:
        # Parse input
        data = request.get_json()
        features = data['features']
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Validate
        required_features = ['sqft', 'bedrooms', 'age']  # Add all required
        if not all(f in X.columns for f in required_features):
            return jsonify({'error': 'Missing required features'}), 400
        
        # Preprocess
        X_poly = poly.transform(X)
        X_scaled = scaler.transform(X_poly)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        
        # Return result
        return jsonify({
            'prediction': float(prediction),
            'units': 'USD',
            'confidence': '82% (test R²)'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### Monitoring in Production

```python
"""
Production monitoring: Track model performance
"""

import logging
from datetime import datetime
import json

class ModelMonitor:
    """Monitor model predictions and performance"""
    
    def __init__(self, log_file='model_monitor.log'):
        self.log_file = log_file
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
    
    def log_prediction(self, input_data, prediction, actual=None):
        """Log a single prediction"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'prediction': float(prediction),
        }
        
        if actual is not None:
            log_entry['actual'] = float(actual)
            log_entry['error'] = float(abs(prediction - actual))
        
        logging.info(json.dumps(log_entry))
    
    def check_drift(self, recent_errors, threshold=0.1):
        """Check if model performance is degrading"""
        mean_error = np.mean(recent_errors)
        
        if mean_error > threshold:
            logging.warning(f"Performance degradation detected! Mean error: {mean_error:.4f}")
            return True
        return False

# Usage
monitor = ModelMonitor()

# In prediction function
prediction = model.predict(X)
monitor.log_prediction(
    input_data=X.to_dict(),
    prediction=prediction,
    actual=y_actual  # If available
)
```

---

## Debugging and Troubleshooting

### Common Issues and Solutions

```python
"""
Debugging guide for common linear model issues
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def diagnose_model_issues(X_train, y_train, X_test, y_test):
    """
    Comprehensive model diagnosis
    """
    print("="*70)
    print("MODEL DIAGNOSIS")
    print("="*70)
    
    # Check 1: Data quality
    print("\n1. DATA QUALITY CHECKS")
    print("-"*70)
    
    # Missing values
    missing_train = pd.DataFrame(X_train).isnull().sum().sum()
    missing_test = pd.DataFrame(X_test).isnull().sum().sum()
    print(f"Missing values - Train: {missing_train}, Test: {missing_test}")
    
    if missing_train > 0 or missing_test > 0:
        print("  ⚠ WARNING: Missing values detected!")
        print("  → Solution: Use SimpleImputer")
    
    # Infinite values
    inf_train = np.isinf(X_train).sum()
    inf_test = np.isinf(X_test).sum()
    print(f"Infinite values - Train: {inf_train}, Test: {inf_test}")
    
    if inf_train > 0 or inf_test > 0:
        print("  ⚠ WARNING: Infinite values detected!")
        print("  → Solution: Replace with np.nan and impute")
    
    # Check 2: Feature scales
    print("\n2. FEATURE SCALE CHECKS")
    print("-"*70)
    
    feature_ranges = pd.DataFrame(X_train).apply(lambda x: x.max() - x.min())
    print(f"Feature range variation: {feature_ranges.min():.2e} to {feature_ranges.max():.2e}")
    
    if feature_ranges.max() / feature_ranges.min() > 100:
        print("  ⚠ WARNING: Large scale differences detected!")
        print("  → Solution: Use StandardScaler")
    
    # Check 3: Target distribution
    print("\n3. TARGET DISTRIBUTION CHECKS")
    print("-"*70)
    
    print(f"Target mean: {y_train.mean():.2f}")
    print(f"Target std: {y_train.std():.2f}")
    print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"Target skewness: {pd.Series(y_train).skew():.2f}")
    
    if abs(pd.Series(y_train).skew()) > 1:
        print("  ⚠ WARNING: Highly skewed target!")
        print("  → Solution: Consider log transform")
    
    # Check 4: Model performance
    print("\n4. MODEL PERFORMANCE CHECKS")
    print("-"*70)
    
    # Train baseline model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    gap = train_score - test_score
    
    print(f"Training R²: {train_score:.4f}")
    print(f"Test R²: {test_score:.4f}")
    print(f"Gap: {gap:.4f}")
    
    if train_score < 0.5:
        print("  ⚠ WARNING: Low training score - High bias!")
        print("  → Solution: Add features, increase complexity")
    
    if gap > 0.1:
        print("  ⚠ WARNING: Large gap - High variance!")
        print("  → Solution: More data, regularization, fewer features")
    
    # Check 5: Residuals
    print("\n5. RESIDUAL CHECKS")
    print("-"*70)
    
    residuals = y_test - model.predict(X_test_scaled)
    
    print(f"Residual mean: {residuals.mean():.4f}")
    print(f"Residual std: {residuals.std():.4f}")
    
    if abs(residuals.mean()) > 0.01 * y_test.std():
        print("  ⚠ WARNING: Non-zero mean residuals - Bias!")
        print("  → Check: Feature engineering, model assumptions")
    
    # Check for patterns in residuals
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(model.predict(X_test_scaled), residuals)
    
    if abs(corr) > 0.1:
        print(f"  ⚠ WARNING: Correlation between predictions and residuals: {corr:.3f}")
        print("  → Check: Non-linear patterns, heteroscedasticity")
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)

# Example usage
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

diagnose_model_issues(X_train, y_train, X_test, y_test)
```

---

## Summary and Next Steps

### What We Covered

1. **Complete Workflow**
   - 10-step ML pipeline from data loading to deployment
   - Real production example

2. **NumPy from Scratch**
   - Linear regression (OLS and gradient descent)
   - Ridge regression
   - Logistic regression
   - Full implementations with explanations

3. **Scikit-Learn Usage**
   - 8 common usage patterns
   - Advanced techniques
   - Pipeline best practices

4. **Best Practices**
   - 10 essential best practices
   - Data splitting, scaling, pipelines
   - Cross-validation, documentation

5. **Common Pitfalls**
   - 10 common mistakes
   - Data leakage, scaling issues
   - How to avoid each

6. **Real-World Examples**
   - Customer churn prediction
   - Sales forecasting
   - Complete working code

7. **Production Considerations**
   - Model versioning
   - API deployment
   - Monitoring

8. **Debugging**
   - Comprehensive diagnosis
   - Issue detection
   - Solutions

### Key Takeaways

✅ **Always use pipelines** for clean, reproducible code  
✅ **Split data first** to avoid leakage  
✅ **Scale features** for regularized models  
✅ **Use cross-validation** for tuning  
✅ **Monitor** model performance in production  
✅ **Document everything** for maintainability  
✅ **Understand from scratch**, use scikit-learn in production  

### Next Steps

**For Learning**:
1. Implement more models from scratch
2. Practice with real datasets (Kaggle)
3. Read scikit-learn documentation
4. Study production ML systems

**For Projects**:
1. Start with template pipelines
2. Always follow best practices
3. Version and document models
4. Monitor in production

**Advanced Topics**:
1. Feature selection algorithms
2. Ensemble methods
3. AutoML pipelines
4. MLOps practices

---

## Quick Reference

```python
# ============================================================
# QUICK TEMPLATE: Regression Pipeline
# ============================================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Create pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# 3. Hyperparameter tuning
param_grid = {
    'poly__degree': [1, 2, 3],
    'ridge__alpha': [0.1, 1.0, 10.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# 4. Evaluate
best_model = grid.best_estimator_
test_score = best_model.score(X_test, y_test)

# 5. Save
import joblib
joblib.dump(best_model, 'model.pkl')

# ============================================================
# QUICK TEMPLATE: Classification Pipeline
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Pipeline
clf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=1000))
])

# Tune
param_grid = {
    'logistic__C': [0.1, 1.0, 10.0],
    'logistic__penalty': ['l1', 'l2'],
    'logistic__solver': ['liblinear']
}

grid = GridSearchCV(clf_pipeline, param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)

# Evaluate
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

**Congratulations!** You now have complete, production-ready implementations and best practices for linear models!

---

*End of Practical_Implementation.md*