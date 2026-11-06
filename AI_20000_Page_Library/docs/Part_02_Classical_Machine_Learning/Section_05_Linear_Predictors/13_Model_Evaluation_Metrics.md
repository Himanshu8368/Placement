# Model Evaluation Metrics

## Table of Contents
1. [Introduction to Model Evaluation](#introduction-to-model-evaluation)
2. [Regression Metrics](#regression-metrics)
3. [Classification Metrics](#classification-metrics)
4. [Confusion Matrix](#confusion-matrix)
5. [ROC Curve and AUC](#roc-curve-and-auc)
6. [Choosing the Right Metric](#choosing-the-right-metric)
7. [Multi-Class and Multi-Label Metrics](#multi-class-and-multi-label-metrics)
8. [Practical Implementation](#practical-implementation)
9. [Common Pitfalls](#common-pitfalls)
10. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to Model Evaluation

### Why Evaluate Models?

**Purpose**: Quantify how well our model performs.

**Questions to answer**:
- How accurate are the predictions?
- What types of errors does the model make?
- Is the model better than baseline?
- Which model should we deploy?

### Types of Evaluation

**1. Point estimates**: Single number summarizing performance
   - Examples: Accuracy, MSE, R²

**2. Detailed analysis**: Breakdown of errors
   - Examples: Confusion matrix, error distribution

**3. Visual analysis**: Plots and curves
   - Examples: ROC curve, learning curves

### Train vs Test Performance

**Critical distinction**:
```
Training error:  Performance on training data
Test error:      Performance on unseen data
```

**Always report test error** for honest evaluation!

**Goal**: Minimize test error, not training error.

---

## Regression Metrics

### Mean Squared Error (MSE)

**Definition**: Average of squared differences between predictions and actual values.

**Formula**:
```
MSE = (1/m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
```

**Properties**:
- Units: **Squared** units of target variable
- Range: [0, ∞)
- Lower is better
- Penalizes large errors heavily

**Example**:
```python
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

mse = np.mean((y_pred - y_true) ** 2)
print(f"MSE: {mse:.4f}")
# MSE: 0.3750
```

**Interpretation**: 
```
MSE = 4.0 for house prices (in $1000s)
→ Average squared error is 4.0 ($1000)²
→ Hard to interpret! Use RMSE instead.
```

**When to use**:
- Default metric for regression
- When you want to heavily penalize large errors
- In optimization (loss function)

---

### Root Mean Squared Error (RMSE)

**Definition**: Square root of MSE.

**Formula**:
```
RMSE = √MSE = √[(1/m) Σᵢ₌₁ᵐ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²]
```

**Properties**:
- Units: **Same** as target variable
- Range: [0, ∞)
- Lower is better
- More interpretable than MSE

**Example**:
```python
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")
# RMSE: 0.6124
```

**Interpretation**:
```
RMSE = 2.0 for house prices (in $1000s)
→ Average error is about $2,000
→ Easy to interpret!
```

**When to use**:
- Reporting to stakeholders (interpretable)
- Same error penalty as MSE (squared)
- When you need same units as target

---

### Mean Absolute Error (MAE)

**Definition**: Average of absolute differences.

**Formula**:
```
MAE = (1/m) Σᵢ₌₁ᵐ |ŷ⁽ⁱ⁾ - y⁽ⁱ⁾|
```

**Properties**:
- Units: Same as target variable
- Range: [0, ∞)
- Lower is better
- Robust to outliers (linear penalty)

**Example**:
```python
mae = np.mean(np.abs(y_pred - y_true))
print(f"MAE: {mae:.4f}")
# MAE: 0.5000
```

**Interpretation**:
```
MAE = 1.5 for house prices (in $1000s)
→ Average error is $1,500
→ All errors weighted equally
```

**MSE vs MAE**:
```
Errors: [0.5, 1.0, 1.5, 2.0, 10.0]  ← One outlier!

MSE  = 21.65  (heavily affected by outlier)
RMSE = 4.65
MAE  = 3.00   (less affected by outlier)
```

**When to use**:
- When outliers should not dominate
- When all errors should be weighted equally
- More robust metric

---

### R² Score (Coefficient of Determination)

**Definition**: Proportion of variance in the target explained by the model.

**Formula**:
```
R² = 1 - (SS_res / SS_tot)
   = 1 - [Σ(yⁱ - ŷⁱ)² / Σ(yⁱ - ȳ)²]
```

Where:
- SS_res = Residual sum of squares = Σ(y - ŷ)²
- SS_tot = Total sum of squares = Σ(y - ȳ)²
- ȳ = mean of y

**Properties**:
- Range: (-∞, 1]
- **R² = 1**: Perfect predictions
- **R² = 0**: Model as good as predicting mean
- **R² < 0**: Model worse than predicting mean
- Unitless (comparable across datasets)

**Interpretation**:
```
R² = 0.85
→ Model explains 85% of variance in target
→ Good fit!

R² = 0.20
→ Model explains only 20% of variance
→ Poor fit, lots of unexplained variance

R² = -0.5
→ Model is worse than just predicting mean!
→ Something is very wrong
```

**Example**:
```python
from sklearn.metrics import r2_score

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")
# R²: 0.9486

# Manual calculation
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_manual = 1 - (ss_res / ss_tot)
print(f"R² (manual): {r2_manual:.4f}")
```

**Visual intuition**:
```python
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 50)
y_true = 2 * X + 1 + np.random.randn(50) * 2

# Perfect model vs poor model
y_pred_good = 2 * X + 1
y_pred_bad = np.full_like(y_true, np.mean(y_true))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Good model
ax = axes[0]
ax.scatter(X, y_true, alpha=0.5, label='True values')
ax.plot(X, y_pred_good, 'r-', linewidth=2, label='Predictions')
ax.axhline(y=np.mean(y_true), color='g', linestyle='--', label='Mean baseline')
r2_good = r2_score(y_true, y_pred_good)
ax.set_title(f'Good Model: R² = {r2_good:.3f}', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Bad model (predicting mean)
ax = axes[1]
ax.scatter(X, y_true, alpha=0.5, label='True values')
ax.axhline(y=np.mean(y_true), color='r', linewidth=2, label='Predictions (mean)')
ax.axhline(y=np.mean(y_true), color='g', linestyle='--', label='Mean baseline')
r2_bad = r2_score(y_true, y_pred_bad)
ax.set_title(f'Bad Model: R² = {r2_bad:.3f}', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**When to use**:
- Comparing models on same dataset
- Understanding explained variance
- Default metric for reporting
- NOT for comparing across different datasets

---

### Adjusted R²

**Problem**: R² always increases when adding features, even if they're useless!

**Solution**: Adjusted R² penalizes for number of features.

**Formula**:
```
Adjusted R² = 1 - [(1 - R²) × (m - 1) / (m - n - 1)]
```

Where:
- m = number of samples
- n = number of features

**Example**:
```python
def adjusted_r2(r2, m, n):
    """Compute adjusted R²"""
    return 1 - (1 - r2) * (m - 1) / (m - n - 1)

m = 100  # samples
n = 10   # features
r2 = 0.85

adj_r2 = adjusted_r2(r2, m, n)
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
# Adjusted R² is lower, accounting for features
```

---

### Comparison of Regression Metrics

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Generate predictions with different error patterns
np.random.seed(42)
y_true = np.linspace(0, 10, 100)

# Scenario 1: Small consistent errors
y_pred_1 = y_true + np.random.randn(100) * 0.5

# Scenario 2: Large occasional errors (outliers)
y_pred_2 = y_true.copy()
y_pred_2[[10, 30, 50, 70, 90]] += np.random.randn(5) * 5

# Scenario 3: Systematic bias
y_pred_3 = y_true + 2

scenarios = [
    ('Small consistent errors', y_pred_1),
    ('Occasional large errors', y_pred_2),
    ('Systematic bias', y_pred_3)
]

print("="*70)
print("REGRESSION METRICS COMPARISON")
print("="*70)
print(f"{'Scenario':<25} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-"*70)

for name, y_pred in scenarios:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{name:<25} {mse:<10.4f} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f}")
```

**Output**:
```
Scenario                  MSE        RMSE       MAE        R²        
──────────────────────────────────────────────────────────────────
Small consistent errors   0.2618     0.5117     0.4009     0.9969    
Occasional large errors   0.5842     0.7643     0.4844     0.9931    
Systematic bias           4.0000     2.0000     2.0000     0.5273    
```

**Insights**:
- MSE/RMSE heavily affected by outliers
- MAE more robust to outliers
- Systematic bias visible in all metrics
- R² shows proportion of variance explained

---

## Classification Metrics

### Accuracy

**Definition**: Proportion of correct predictions.

**Formula**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = Correct predictions / Total predictions
```

Where:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

**Example**:
```python
from sklearn.metrics import accuracy_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.1f}%")
# Accuracy: 80.0%
```

**When to use**:
- Balanced classes
- All errors equally important
- Simple, intuitive metric

**When NOT to use**:
- **Imbalanced classes**! (Accuracy paradox)

**Accuracy paradox**:
```
Dataset: 95% negative, 5% positive

Model 1: Always predict negative
  Accuracy = 95%!  ← Looks great but useless!
  
Model 2: Smart classifier
  Accuracy = 90%   ← Lower but actually useful
```

---

### Precision

**Definition**: Of all positive predictions, how many were actually positive?

**Formula**:
```
Precision = TP / (TP + FP)
          = True Positives / All Predicted Positives
```

**Alternative name**: Positive Predictive Value (PPV)

**Interpretation**:
```
Precision = 0.8
→ 80% of positive predictions were correct
→ 20% were false alarms
```

**Example**:
```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision * 100:.1f}%")
```

**When high precision matters**:
- Spam detection (don't mark good emails as spam)
- Medical diagnosis (don't alarm patients unnecessarily)
- Legal judgments (don't convict innocent people)

**Cost**: May miss actual positives (low recall)

---

### Recall

**Definition**: Of all actual positives, how many did we find?

**Formula**:
```
Recall = TP / (TP + FN)
       = True Positives / All Actual Positives
```

**Alternative names**: 
- Sensitivity
- True Positive Rate (TPR)
- Hit Rate

**Interpretation**:
```
Recall = 0.9
→ Found 90% of actual positives
→ Missed 10%
```

**Example**:
```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall * 100:.1f}%")
```

**When high recall matters**:
- Disease screening (don't miss sick patients)
- Fraud detection (don't miss fraudulent transactions)
- Search engines (don't miss relevant results)

**Cost**: May have many false alarms (low precision)

---

### Precision-Recall Tradeoff

**Fundamental tradeoff**: Can't maximize both simultaneously!

```
High threshold:  Few predictions → High Precision, Low Recall
Low threshold:   Many predictions → Low Precision, High Recall
```

**Visual example**:
```
Actual positives: ●●●●● (5 total)
Actual negatives: ○○○○○○○○○○ (10 total)

Strict classifier (high threshold):
  Predicts positive: ●● (only very confident)
  Precision = 2/2 = 100%  ← Perfect!
  Recall = 2/5 = 40%      ← But missed 3!

Lenient classifier (low threshold):
  Predicts positive: ●●●●● ○○○ (predict more)
  Precision = 5/8 = 62.5%  ← Some false alarms
  Recall = 5/5 = 100%      ← Found all!
```

**Precision-Recall Curve**:
```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get predicted probabilities
y_scores = np.random.rand(100)
y_true = (y_scores > 0.5).astype(int)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, linewidth=2)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.grid(True, alpha=0.3)
plt.show()
```

---

### F1 Score

**Definition**: Harmonic mean of precision and recall.

**Formula**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2TP / (2TP + FP + FN)
```

**Why harmonic mean?**
- Harmonic mean favors balanced values
- If either precision or recall is low, F1 is low

**Example**:
```
Precision = 0.9, Recall = 0.9  →  F1 = 0.90
Precision = 0.9, Recall = 0.1  →  F1 = 0.18  (heavily penalized!)
Precision = 0.5, Recall = 0.5  →  F1 = 0.50
```

**Python**:
```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")
```

**When to use**:
- Need balance between precision and recall
- Imbalanced classes
- Single metric to optimize

---

### F-Beta Score

**Generalization** of F1 that allows weighting precision vs recall.

**Formula**:
```
F_β = (1 + β²) × (Precision × Recall) / (β²×Precision + Recall)
```

**β parameter**:
- **β = 1**: F1 score (equal weight)
- **β < 1**: Favor precision (e.g., β = 0.5)
- **β > 1**: Favor recall (e.g., β = 2)

**Example**:
```python
from sklearn.metrics import fbeta_score

# Favor recall (β=2)
f2 = fbeta_score(y_true, y_pred, beta=2)
print(f"F2 Score: {f2:.4f}")

# Favor precision (β=0.5)
f_half = fbeta_score(y_true, y_pred, beta=0.5)
print(f"F0.5 Score: {f_half:.4f}")
```

---

## Confusion Matrix

### Definition

A **confusion matrix** shows counts of predictions vs actual labels.

**Binary classification**:
```
                Predicted
                Neg    Pos
Actual  Neg  │  TN  │  FP  │
        Pos  │  FN  │  TP  │
```

Where:
- **TN** (True Negative): Correctly predicted negative
- **FP** (False Positive): Incorrectly predicted positive (Type I error)
- **FN** (False Negative): Incorrectly predicted negative (Type II error)
- **TP** (True Positive): Correctly predicted positive

### Example

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Example predictions
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 1])

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)
print(f"\nTN={cm[0,0]}, FP={cm[0,1]}")
print(f"FN={cm[1,0]}, TP={cm[1,1]}")

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.show()
```

### Deriving Metrics from Confusion Matrix

From confusion matrix, we can compute **all** classification metrics:

```python
def compute_metrics(cm):
    """Compute all metrics from confusion matrix"""
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    # Basic counts
    total = TN + FP + FN + TP
    
    # Accuracy
    accuracy = (TP + TN) / total
    
    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall (Sensitivity, TPR)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False Positive Rate
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    # False Negative Rate
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1': f1,
        'FPR': fpr,
        'FNR': fnr
    }

metrics = compute_metrics(cm)
for name, value in metrics.items():
    print(f"{name:<15}: {value:.4f}")
```

### Multi-Class Confusion Matrix

For K classes, confusion matrix is K × K:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load iris (3 classes)
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=iris.target_names,
           yticklabels=iris.target_names)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Multi-Class Confusion Matrix (Iris Dataset)', fontsize=16)
plt.show()
```

---

## ROC Curve and AUC

### ROC Curve

**ROC** = Receiver Operating Characteristic

**Definition**: Plot of True Positive Rate vs False Positive Rate at different classification thresholds.

**Axes**:
```
Y-axis: TPR = TP / (TP + FN) = Recall
X-axis: FPR = FP / (FP + TN) = 1 - Specificity
```

**How it works**:
1. Classifier outputs probabilities (not hard predictions)
2. Vary threshold from 0 to 1
3. For each threshold, compute TPR and FPR
4. Plot all (FPR, TPR) pairs

**Example**:
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
y_true = np.random.randint(0, 2, 100)
y_scores = np.random.rand(100)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random classifier')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

### Interpreting ROC Curves

**Perfect classifier**: 
```
Goes straight up (FPR=0, TPR=1), then right
AUC = 1.0
```

**Random classifier**:
```
Diagonal line from (0,0) to (1,1)
AUC = 0.5
```

**Good classifier**:
```
Curves toward upper-left corner
AUC > 0.7
```

**Visual comparison**:
```python
fig, ax = plt.subplots(figsize=(10, 8))

# Perfect classifier
ax.plot([0, 0, 1], [0, 1, 1], linewidth=3, label='Perfect (AUC=1.0)')

# Good classifier
fpr_good = np.linspace(0, 1, 100)
tpr_good = np.sqrt(fpr_good)  # Curves up faster
ax.plot(fpr_good, tpr_good, linewidth=3, label='Good (AUC≈0.67)')

# Random classifier
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.5)')

# Bad classifier (worse than random)
tpr_bad = fpr_good ** 2
ax.plot(fpr_good, tpr_bad, linewidth=3, label='Bad (AUC≈0.33)')

ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title('ROC Curves Comparison', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.show()
```

### AUC (Area Under Curve)

**Definition**: Area under the ROC curve.

**Interpretation**:
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.9-1.0**: Excellent
- **AUC = 0.8-0.9**: Very good
- **AUC = 0.7-0.8**: Good
- **AUC = 0.6-0.7**: Poor
- **AUC = 0.5**: Random (worthless)
- **AUC < 0.5**: Worse than random (invert predictions!)

**Probabilistic interpretation**:
```
AUC = Probability that classifier ranks a random positive
      example higher than a random negative example
```

**Example**:
```python
from sklearn.metrics import roc_auc_score

# Using predicted probabilities
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
y_scores = np.array([0.1, 0.3, 0.6, 0.9, 0.2, 0.8, 0.4, 0.7])

auc_score = roc_auc_score(y_true, y_scores)
print(f"AUC: {auc_score:.4f}")
```

### When to Use ROC/AUC

✅ **Use when**:
- Balanced or moderately imbalanced classes
- Want threshold-independent metric
- Care about ranking quality
- Comparing multiple models

❌ **Don't use when**:
- Severely imbalanced classes (use Precision-Recall curve instead)
- Specific threshold needed for deployment

### Real-World Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5,
                          weights=[0.9, 0.1],  # 90% class 0, 10% class 1
                          random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Find optimal threshold (Youden's J statistic)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"AUC: {roc_auc:.4f}")
print(f"Optimal threshold: {optimal_threshold:.4f}")

# Plot ROC curve with optimal point
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], s=200, c='red', marker='o',
           label=f'Optimal threshold = {optimal_threshold:.3f}')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve - Imbalanced Dataset', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Evaluate at optimal threshold
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
print("\nClassification Report (Optimal Threshold):")
print(classification_report(y_test, y_pred_optimal))
```

---

## Choosing the Right Metric

### Decision Framework

```
For REGRESSION:
  │
  ├─ Care about large errors more?
  │  └─ Yes → MSE or RMSE
  │  └─ No → MAE
  │
  ├─ Want interpretability?
  │  └─ Yes → RMSE or MAE (same units as target)
  │  └─ No → MSE is fine
  │
  ├─ Want to compare models?
  │  └─ Same dataset → R²
  │  └─ Different datasets → Cannot use R²
  │
  └─ Default: Use RMSE and R²

For CLASSIFICATION:
  │
  ├─ Balanced classes?
  │  ├─ Yes → Accuracy is fine
  │  └─ No → Don't use accuracy!
  │
  ├─ What matters more?
  │  ├─ Avoiding false positives → Precision
  │  ├─ Finding all positives → Recall
  │  └─ Balance → F1 Score
  │
  ├─ Need threshold-independent metric?
  │  └─ Yes → AUC-ROC
  │
  └─ Default: Use F1 and AUC
```

### Domain-Specific Considerations

**Medical Diagnosis**:
- High Recall (don't miss sick patients)
- Some false positives acceptable

**Spam Detection**:
- High Precision (don't mark good emails as spam)
- Some spam getting through is okay

**Fraud Detection**:
- High Recall (catch fraud)
- Can tolerate some false alarms

**Legal/Criminal Justice**:
- Very high Precision (don't convict innocent)
- "Better to let 10 guilty go free than convict 1 innocent"

---

## Multi-Class and Multi-Label Metrics

### Multi-Class Classification

**Macro-averaging**: Average metric for each class separately
```
Macro-F1 = (F1_class1 + F1_class2 + ... + F1_classK) / K
```

**Micro-averaging**: Pool all predictions together
```
Micro-F1 = F1(all_TPs, all_FPs, all_FNs)
```

**Weighted-averaging**: Weight by class frequency
```
Weighted-F1 = Σ(weight_i × F1_i)
```

**Example**:
```python
from sklearn.metrics import classification_report

# Multi-class predictions
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0])

print(classification_report(y_true, y_pred))
```

**When to use**:
- **Macro**: All classes equally important
- **Micro**: Overall performance across all samples
- **Weighted**: Account for class imbalance

---

## Practical Implementation

### Complete Evaluation Pipeline

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Generate data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ==== METRICS ====
print("="*70)
print("CLASSIFICATION METRICS")
print("="*70)

# Basic metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba):.4f}")

# Detailed report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print("="*70)
print("CROSS-VALIDATION")
print("="*70)
print(f"CV F1 Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==== VISUALIZATIONS ====
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Confusion Matrix
ax = axes[0, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

# Plot 2: ROC Curve
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)
ax.plot(fpr, tpr, linewidth=3, label=f'AUC = {auc_score:.3f}')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Prediction distribution
ax = axes[1, 0]
ax.hist(y_proba[y_test==0], bins=30, alpha=0.7, label='Class 0', color='blue')
ax.hist(y_proba[y_test==1], bins=30, alpha=0.7, label='Class 1', color='red')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Count')
ax.set_title('Prediction Distribution')
ax.legend()

# Plot 4: Precision-Recall Curve
ax = axes[1, 1]
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
ax.plot(recall, precision, linewidth=3)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Common Pitfalls

### 1. Using Accuracy with Imbalanced Data

❌ **Wrong**:
```python
# 95% negative, 5% positive
accuracy = 0.95  # Model always predicts negative!
# Looks good but model is useless
```

✅ **Correct**:
```python
# Use F1, Precision, Recall, or AUC instead
```

### 2. Not Using Probabilities

❌ **Wrong**:
```python
y_pred = model.predict(X_test)  # Hard predictions only
# Can't compute ROC curve or adjust threshold
```

✅ **Correct**:
```python
y_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities
# Can compute ROC, adjust threshold, calibrate
```

### 3. Evaluating on Training Data

❌ **Wrong**:
```python
model.fit(X_train, y_train)
score = model.score(X_train, y_train)  # Training score!
```

✅ **Correct**:
```python
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # Test score
```

### 4. Comparing R² Across Different Datasets

❌ **Wrong**:
```python
# Dataset A: R² = 0.85
# Dataset B: R² = 0.70
# Conclusion: Model works better on A
# WRONG! Can't compare R² across datasets
```

### 5. Optimizing for Wrong Metric

❌ **Wrong**:
```python
# Medical diagnosis: optimize for accuracy
# Should optimize for recall (don't miss patients!)
```

---

## Summary and Next Steps

### What We Learned

1. **Regression Metrics**
   - MSE, RMSE, MAE, R²
   - When to use each
   - Interpretation

2. **Classification Metrics**
   - Accuracy, Precision, Recall, F1
   - Precision-Recall tradeoff
   - When to use each

3. **Confusion Matrix**
   - Structure and interpretation
   - Computing all metrics from CM
   - Multi-class extension

4. **ROC Curve and AUC**
   - Threshold-independent evaluation
   - AUC interpretation
   - When to use

5. **Choosing Metrics**
   - Decision framework
   - Domain considerations
   - Common pitfalls

### Key Formulas

**Regression**:
```
MSE  = (1/m) Σ(ŷ - y)²
RMSE = √MSE
MAE  = (1/m) Σ|ŷ - y|
R²   = 1 - (SS_res / SS_tot)
```

**Classification**:
```
Accuracy  = (TP + TN) / Total
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

### Key Insights

✅ **No single metric** tells the whole story  
✅ **Domain knowledge** guides metric choice  
✅ **Imbalanced data** requires special metrics  
✅ **Always evaluate on test set**  
✅ **Use multiple metrics** for complete picture  

### Prerequisites Checklist

Before moving on:
- [ ] Understand all regression metrics
- [ ] Understand all classification metrics
- [ ] Can interpret confusion matrix
- [ ] Understand ROC curve and AUC
- [ ] Know when to use which metric
- [ ] Can avoid common pitfalls

### Next Steps

1. **Cross_Validation.md**
   - K-fold CV
   - Stratified CV
   - Time series CV
   - Learning curves

2. **Bias_Variance_Tradeoff.md**
   - Understanding underfitting/overfitting
   - Diagnosis techniques
   - Model selection

3. **Hyperparameter_Tuning.md**
   - Grid search
   - Random search
   - Bayesian optimization

### Practice Exercises

1. Implement all metrics from scratch
2. Compare metrics on imbalanced dataset
3. Plot and interpret ROC curves
4. Create comprehensive evaluation report
5. Choose appropriate metrics for different domains
6. Debug common metric pitfalls

### Additional Resources

**Papers**:
- Powers (2011): "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation"

**Books**:
- "Evaluating Learning Algorithms" (Japkowicz & Shah)
- "Pattern Recognition and Machine Learning" (Bishop)

**Online**:
- Scikit-learn: Metrics documentation
- Google ML Crash Course: Classification metrics
- Fast.ai: Practical metrics guide

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║           MODEL EVALUATION METRICS QUICK REFERENCE             ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  REGRESSION METRICS                                            ║
║  ──────────────────                                            ║
║  MSE  = (1/m) Σ(ŷ - y)²        Units: squared, penalizes large║
║  RMSE = √MSE                   Units: same as y, interpretable ║
║  MAE  = (1/m) Σ|ŷ - y|         Units: same as y, robust       ║
║  R²   = 1 - SS_res/SS_tot      Range: (-∞,1], variance explained║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  CLASSIFICATION METRICS                                        ║
╠════════════════════════════════════════════════════════════════╣
║  Accuracy  = (TP+TN) / Total          Overall correctness     ║
║  Precision = TP / (TP+FP)             Of pred +, how many correct║
║  Recall    = TP / (TP+FN)             Of actual +, how many found║
║  F1        = 2·P·R / (P+R)            Harmonic mean of P and R║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  CONFUSION MATRIX                                              ║
╠════════════════════════════════════════════════════════════════╣
║              Predicted                                         ║
║              Neg    Pos                                        ║
║  Actual Neg │ TN  │ FP │                                       ║
║         Pos │ FN  │ TP │                                       ║
║                                                                ║
║  TN: True Negative   FP: False Positive (Type I error)         ║
║  FN: False Negative (Type II error)  TP: True Positive         ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  AUC-ROC                                                       ║
╠════════════════════════════════════════════════════════════════╣
║  Area under ROC curve (TPR vs FPR)                             ║
║                                                                ║
║  1.0     : Perfect                                             ║
║  0.9-1.0 : Excellent                                           ║
║  0.8-0.9 : Very good                                           ║
║  0.7-0.8 : Good                                                ║
║  0.5     : Random (worthless)                                  ║
║  < 0.5   : Worse than random                                   ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  WHEN TO USE                                                   ║
╠════════════════════════════════════════════════════════════════╣
║  Regression:                                                   ║
║    • Default: RMSE + R²                                        ║
║    • Outliers present: MAE                                     ║
║                                                                ║
║  Classification:                                               ║
║    • Balanced classes: Accuracy                                ║
║    • Imbalanced: F1, Precision, Recall                         ║
║    • Need probabilities: AUC-ROC                               ║
║    • Avoid false positives: Precision                          ║
║    • Avoid false negatives: Recall                             ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  COMMON PITFALLS                                               ║
╠════════════════════════════════════════════════════════════════╣
║  ✗ Accuracy on imbalanced data                                 ║
║  ✗ Evaluating on training set                                  ║
║  ✗ Comparing R² across datasets                                ║
║  ✗ Optimizing wrong metric for domain                          ║
║  ✗ Not using probabilities                                     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Congratulations!** You now have a comprehensive understanding of model evaluation metrics!

---

*End of Model_Evaluation_Metrics.md*