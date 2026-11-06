# Multiclass Linear Classification

## Table of Contents
1. [Introduction to Multiclass Classification](#introduction-to-multiclass-classification)
2. [One-vs-Rest (OvR)](#one-vs-rest-ovr)
3. [One-vs-One (OvO)](#one-vs-one-ovo)
4. [Softmax Regression](#softmax-regression)
5. [Multinomial Logistic Regression](#multinomial-logistic-regression)
6. [Comparison of Methods](#comparison-of-methods)
7. [Practical Implementation](#practical-implementation)
8. [Advanced Topics](#advanced-topics)
9. [Summary and Next Steps](#summary-and-next-steps)

---

## Introduction to Multiclass Classification

### What is Multiclass Classification?

**Multiclass classification** extends binary classification to problems with **more than two classes**.

```
Binary:      y ∈ {0, 1}           (2 classes)
Multiclass:  y ∈ {0, 1, 2, ..., K-1}  (K classes)
```

**Examples**:
- **Digit recognition**: 0-9 (10 classes)
- **Animal classification**: Cat, Dog, Bird, Fish (4 classes)
- **Language detection**: English, Spanish, French, German (4 classes)
- **Medical diagnosis**: Healthy, Flu, Cold, COVID (4 classes)

### Approaches

There are two main strategies:

**1. Decomposition Methods**: Break into multiple binary problems
- **One-vs-Rest (OvR)**: K binary classifiers
- **One-vs-One (OvO)**: K(K-1)/2 binary classifiers

**2. Direct Methods**: Single model that handles all classes
- **Softmax Regression**: Direct probability over all classes
- **Multinomial Logistic Regression**: Same as softmax regression

### Notation

Throughout this chapter:
- **K** = number of classes
- **y** ∈ {0, 1, ..., K-1} = true class label
- **ŷ** = predicted class label
- **x** = feature vector
- **wₖ** = weight vector for class k

---

## One-vs-Rest (OvR)

### Also Known As

- **One-vs-All (OvA)**
- **One-Against-All**

### The Idea

Train **K binary classifiers**, one for each class:

```
Classifier 1: Class 0 vs {Class 1, 2, ..., K-1}
Classifier 2: Class 1 vs {Class 0, 2, ..., K-1}
...
Classifier K: Class K-1 vs {Class 0, 1, ..., K-2}
```

**Each classifier**: "Is this example class k, or not class k?"

### Algorithm

**Training Phase**:
```
For k = 0 to K-1:
    Create binary labels:
        yₖ⁽ⁱ⁾ = 1  if y⁽ⁱ⁾ = k
        yₖ⁽ⁱ⁾ = 0  if y⁽ⁱ⁾ ≠ k
    
    Train binary classifier k:
        wₖ = train_binary_classifier(X, yₖ)
```

**Prediction Phase**:
```
For each example x:
    For k = 0 to K-1:
        Compute score: sₖ = wₖᵀx
    
    Predict: ŷ = argmax_k sₖ
```

**In other words**: Choose the class whose classifier is most confident.

### Example: 3-Class Problem

Suppose we have 3 classes: **Cat, Dog, Bird**

**Training**:
```
Classifier 0: Cat vs {Dog, Bird}
    Training data:
        Cat images   → label 1
        Dog images   → label 0
        Bird images  → label 0

Classifier 1: Dog vs {Cat, Bird}
    Training data:
        Cat images   → label 0
        Dog images   → label 1
        Bird images  → label 0

Classifier 2: Bird vs {Cat, Dog}
    Training data:
        Cat images   → label 0
        Dog images   → label 0
        Bird images  → label 1
```

**Prediction**:
```
New image x:
    Classifier 0 (Cat): score = 0.8
    Classifier 1 (Dog): score = 0.3
    Classifier 2 (Bird): score = 0.1
    
    Prediction: Cat (highest score)
```

### Python Implementation

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class OneVsRestClassifier:
    """One-vs-Rest multiclass classifier"""
    
    def __init__(self, base_classifier=None):
        """
        Args:
            base_classifier: Binary classifier (e.g., LogisticRegression)
        """
        if base_classifier is None:
            base_classifier = LogisticRegression(max_iter=1000)
        self.base_classifier = base_classifier
        self.classifiers = []
        self.classes = None
    
    def fit(self, X, y):
        """
        Train K binary classifiers
        
        Args:
            X: Feature matrix (m × n)
            y: Labels (m,) with values in {0, 1, ..., K-1}
        """
        self.classes = np.unique(y)
        K = len(self.classes)
        self.classifiers = []
        
        for k in self.classes:
            # Create binary labels
            y_binary = (y == k).astype(int)
            
            # Train binary classifier
            from copy import deepcopy
            clf = deepcopy(self.base_classifier)
            clf.fit(X, y_binary)
            self.classifiers.append(clf)
            
            print(f"Trained classifier for class {k}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix (m × n)
        
        Returns:
            Probabilities (m × K)
        """
        # Get scores from all classifiers
        scores = np.column_stack([
            clf.decision_function(X) if hasattr(clf, 'decision_function') 
            else clf.predict_proba(X)[:, 1]
            for clf in self.classifiers
        ])
        
        # Normalize to probabilities (softmax)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Feature matrix (m × n)
        
        Returns:
            Predicted labels (m,)
        """
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]
    
    def score(self, X, y):
        """Compute accuracy"""
        return np.mean(self.predict(X) == y)

# Example usage
from sklearn.datasets import make_classification

# Generate 3-class data
X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, 
                           n_classes=3, random_state=42)

# Train OvR classifier
ovr = OneVsRestClassifier()
ovr.fit(X, y)

# Predict
predictions = ovr.predict(X)
accuracy = ovr.score(X, y)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
```

### Visualization

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier as SklearnOvR

# Generate data
X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           n_classes=3, random_state=42)

# Train
ovr = SklearnOvR(LogisticRegression(max_iter=1000))
ovr.fit(X, y)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Create mesh
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                        np.linspace(x2_min, x2_max, 200))

# Plot 1: Decision regions
ax = axes[0, 0]
Z = ovr.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
ax.contourf(xx1, xx2, Z, alpha=0.4, cmap='viridis')
for class_val in [0, 1, 2]:
    ax.scatter(X[y == class_val, 0], X[y == class_val, 1], 
              label=f'Class {class_val}', s=50, edgecolors='k')
ax.set_title('One-vs-Rest Decision Regions', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2-4: Individual binary classifiers
for idx, k in enumerate([0, 1, 2]):
    ax = axes[(idx+1)//2, (idx+1)%2]
    
    # Get decision boundary for classifier k
    clf = ovr.estimators_[k]
    
    # Plot decision function
    Z_binary = clf.decision_function(np.c_[xx1.ravel(), xx2.ravel()])
    Z_binary = Z_binary.reshape(xx1.shape)
    
    contour = ax.contourf(xx1, xx2, Z_binary, levels=20, cmap='RdYlBu', alpha=0.6)
    ax.contour(xx1, xx2, Z_binary, levels=[0], colors='black', linewidths=3)
    
    # Plot data (binary labels)
    y_binary = (y == k).astype(int)
    ax.scatter(X[y_binary == 1, 0], X[y_binary == 1, 1], 
              c='red', marker='s', s=100, edgecolors='k', label=f'Class {k}')
    ax.scatter(X[y_binary == 0, 0], X[y_binary == 0, 1], 
              c='blue', marker='o', s=50, edgecolors='k', alpha=0.5, label='Other')
    
    ax.set_title(f'Classifier {k}: Class {k} vs Rest', fontsize=14)
    ax.legend()
    plt.colorbar(contour, ax=ax)

plt.tight_layout()
plt.show()
```

### Advantages

✅ **Simple**: Easy to understand and implement
✅ **Efficient**: Only K classifiers needed
✅ **Parallelizable**: Can train classifiers independently
✅ **Works with any binary classifier**: Flexible
✅ **Interpretable**: Each classifier has clear meaning

### Disadvantages

❌ **Imbalanced data**: Each classifier sees K-1 negative examples for 1 positive
❌ **Overlapping regions**: Multiple classifiers may predict positive
❌ **Scale issues**: Scores from different classifiers not directly comparable
❌ **Not probabilistically consistent**: Probabilities may not sum to 1

---

## One-vs-One (OvO)

### Also Known As

- **All-vs-All**
- **Pairwise Classification**

### The Idea

Train a binary classifier for **every pair of classes**:

```
Number of classifiers = K(K-1)/2
```

**For 3 classes**: 3(2)/2 = 3 classifiers
**For 4 classes**: 4(3)/2 = 6 classifiers
**For 10 classes**: 10(9)/2 = 45 classifiers

### Algorithm

**Training Phase**:
```
For each pair (i, j) where i < j:
    Select data from classes i and j only
    Create binary labels:
        y_binary = 0 if y = i
        y_binary = 1 if y = j
    
    Train binary classifier:
        w_{ij} = train_binary_classifier(X_subset, y_binary)
```

**Prediction Phase** (Voting):
```
For each example x:
    votes = [0, 0, ..., 0]  (K votes)
    
    For each classifier (i, j):
        prediction = classify_{ij}(x)
        if prediction == i:
            votes[i] += 1
        else:
            votes[j] += 1
    
    Predict: ŷ = argmax_k votes[k]
```

### Example: 3-Class Problem

**Cat (0), Dog (1), Bird (2)**

**Training** (3 classifiers):
```
Classifier (0,1): Cat vs Dog
    Training data: Only Cat and Dog images
    
Classifier (0,2): Cat vs Bird
    Training data: Only Cat and Bird images
    
Classifier (1,2): Dog vs Bird
    Training data: Only Dog and Bird images
```

**Prediction**:
```
New image x:
    Classifier (0,1) predicts: Cat (vote for 0)
    Classifier (0,2) predicts: Cat (vote for 0)
    Classifier (1,2) predicts: Bird (vote for 2)
    
    Votes: [2, 0, 1]
    Prediction: Cat (class 0 has most votes)
```

### Python Implementation

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from itertools import combinations

class OneVsOneClassifier:
    """One-vs-One multiclass classifier"""
    
    def __init__(self, base_classifier=None):
        if base_classifier is None:
            base_classifier = LogisticRegression(max_iter=1000)
        self.base_classifier = base_classifier
        self.classifiers = {}
        self.classes = None
    
    def fit(self, X, y):
        """
        Train K(K-1)/2 binary classifiers
        
        Args:
            X: Feature matrix (m × n)
            y: Labels (m,)
        """
        self.classes = np.unique(y)
        K = len(self.classes)
        self.classifiers = {}
        
        # Train classifier for each pair
        for i, j in combinations(range(K), 2):
            class_i, class_j = self.classes[i], self.classes[j]
            
            # Select data from classes i and j only
            mask = (y == class_i) | (y == class_j)
            X_pair = X[mask]
            y_pair = y[mask]
            
            # Create binary labels
            y_binary = (y_pair == class_j).astype(int)
            
            # Train binary classifier
            from copy import deepcopy
            clf = deepcopy(self.base_classifier)
            clf.fit(X_pair, y_binary)
            self.classifiers[(class_i, class_j)] = clf
            
            print(f"Trained classifier for ({class_i}, {class_j})")
        
        return self
    
    def predict(self, X):
        """
        Predict using voting
        
        Args:
            X: Feature matrix (m × n)
        
        Returns:
            Predicted labels (m,)
        """
        m = X.shape[0]
        K = len(self.classes)
        votes = np.zeros((m, K))
        
        # Collect votes from all classifiers
        for idx, (i, j) in enumerate(combinations(range(K), 2)):
            class_i, class_j = self.classes[i], self.classes[j]
            clf = self.classifiers[(class_i, class_j)]
            
            # Predict (0 = class_i, 1 = class_j)
            predictions = clf.predict(X)
            
            # Add votes
            votes[predictions == 0, i] += 1
            votes[predictions == 1, j] += 1
        
        # Return class with most votes
        return self.classes[np.argmax(votes, axis=1)]
    
    def score(self, X, y):
        """Compute accuracy"""
        return np.mean(self.predict(X) == y)

# Example usage
X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           n_classes=3, random_state=42)

# Train OvO classifier
ovo = OneVsOneClassifier()
ovo.fit(X, y)

# Predict
predictions = ovo.predict(X)
accuracy = ovo.score(X, y)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
```

### Visualization

```python
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsOneClassifier as SklearnOvO

# Train
ovo = SklearnOvO(LogisticRegression(max_iter=1000))
ovo.fit(X, y)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Create mesh
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                        np.linspace(x2_min, x2_max, 200))

# Plot 1: Decision regions
ax = axes[0, 0]
Z = ovo.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
ax.contourf(xx1, xx2, Z, alpha=0.4, cmap='viridis')
for class_val in [0, 1, 2]:
    ax.scatter(X[y == class_val, 0], X[y == class_val, 1],
              label=f'Class {class_val}', s=50, edgecolors='k')
ax.set_title('One-vs-One Decision Regions', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2-4: Individual pairwise classifiers
pairs = [(0, 1), (0, 2), (1, 2)]
for idx, (i, j) in enumerate(pairs):
    ax = axes[(idx+1)//2, (idx+1)%2]
    
    # Get classifier for this pair
    clf = ovo.estimators_[idx]
    
    # Plot decision function
    Z_pair = clf.decision_function(np.c_[xx1.ravel(), xx2.ravel()])
    Z_pair = Z_pair.reshape(xx1.shape)
    
    contour = ax.contourf(xx1, xx2, Z_pair, levels=20, cmap='RdYlBu', alpha=0.6)
    ax.contour(xx1, xx2, Z_pair, levels=[0], colors='black', linewidths=3)
    
    # Plot only classes i and j
    ax.scatter(X[y == i, 0], X[y == i, 1],
              c='red', marker='s', s=100, edgecolors='k', label=f'Class {i}')
    ax.scatter(X[y == j, 0], X[y == j, 1],
              c='blue', marker='o', s=100, edgecolors='k', label=f'Class {j}')
    ax.scatter(X[(y != i) & (y != j), 0], X[(y != i) & (y != j), 1],
              c='gray', marker='x', s=50, alpha=0.3, label='Other')
    
    ax.set_title(f'Classifier: Class {i} vs Class {j}', fontsize=14)
    ax.legend()
    plt.colorbar(contour, ax=ax)

plt.tight_layout()
plt.show()
```

### Advantages

✅ **Balanced data**: Each classifier sees equal examples from both classes
✅ **Simpler problems**: Each classifier only distinguishes 2 classes
✅ **Robust**: Less affected by outliers in other classes
✅ **Often more accurate**: Especially when classes are well-separated

### Disadvantages

❌ **Many classifiers**: K(K-1)/2 grows quadratically
❌ **Slow prediction**: Must evaluate all classifiers
❌ **Memory intensive**: Must store many models
❌ **Ambiguous regions**: Voting can tie
❌ **Not scalable**: Impractical for large K

---

## Softmax Regression

### Introduction

**Softmax regression** (also called **multinomial logistic regression**) is a **direct extension** of binary logistic regression to multiple classes.

**Key idea**: Compute a probability distribution over all K classes simultaneously.

### The Softmax Function

Given K class logits z₁, z₂, ..., zₖ, the **softmax function** converts them to probabilities:

```
P(y=k|x) = exp(zₖ) / Σⱼ₌₁ᴷ exp(zⱼ)
```

Where zₖ = wₖᵀx is the logit for class k.

**Properties**:
1. All probabilities positive: P(y=k|x) > 0
2. Sum to one: Σₖ P(y=k|x) = 1
3. Differentiable everywhere
4. Reduces to sigmoid for K=2

### Mathematical Formulation

**Model**:
```
z = Wx          (K logits)
ŷ = softmax(z)  (K probabilities)
```

Where:
- W is K × (n+1) weight matrix (one row per class)
- x is (n+1) × 1 feature vector (with bias)
- z is K × 1 vector of logits
- ŷ is K × 1 vector of probabilities

**Explicitly**:
```
For each class k:
    zₖ = wₖ₀ + wₖ₁x₁ + wₖ₂x₂ + ... + wₖₙxₙ
    
    P(y=k|x) = exp(zₖ) / Σⱼ exp(zⱼ)
```

### Why "Softmax"?

It's a **soft** (smooth) version of the **max** function:

```
max(z₁, z₂, ..., zₖ)  → Returns one value (hard)

softmax(z₁, z₂, ..., zₖ) → Returns probability distribution (soft)
```

**Example**:
```
z = [2.0, 1.0, 0.1]

Hard max: argmax(z) = 0  (just the index)

Softmax: [0.659, 0.242, 0.099]  (probabilities)
```

The largest logit gets highest probability, but others get non-zero probabilities too.

### Softmax Properties

**Property 1**: **Invariant to constant shift**
```
softmax(z) = softmax(z + c)  for any constant c
```

**Proof**:
```
softmax(z + c)ₖ = exp(zₖ + c) / Σⱼ exp(zⱼ + c)
                = exp(c)·exp(zₖ) / [exp(c)·Σⱼ exp(zⱼ)]
                = exp(zₖ) / Σⱼ exp(zⱼ)
                = softmax(z)ₖ
```

**Practical use**: Subtract max for numerical stability!

**Property 2**: **Temperature scaling**
```
softmax(z/T)  where T is temperature
```

- T → 0: Approaches hard max (peaked distribution)
- T → ∞: Approaches uniform distribution (flat)
- T = 1: Standard softmax

### Python Implementation

```python
import numpy as np

def softmax(z):
    """
    Compute softmax function
    
    Args:
        z: Logits (... × K)
    
    Returns:
        Probabilities (... × K)
    """
    # Numerical stability: subtract max
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

# Example
z = np.array([[2.0, 1.0, 0.1],
              [1.0, 2.0, 3.0],
              [0.0, 0.0, 0.0]])

probs = softmax(z)
print("Logits:")
print(z)
print("\nProbabilities:")
print(probs)
print("\nSum of probabilities:")
print(np.sum(probs, axis=1))
```

**Output**:
```
Logits:
[[2.  1.  0.1]
 [1.  2.  3. ]
 [0.  0.  0. ]]

Probabilities:
[[0.659  0.242  0.099]
 [0.090  0.245  0.665]
 [0.333  0.333  0.333]]

Sum of probabilities:
[1. 1. 1.]
```

### Categorical Cross-Entropy Loss

For multiclass classification, we use **categorical cross-entropy**:

**For single example**:
```
L = -Σₖ₌₁ᴷ yₖ log(ŷₖ)
```

Where:
- yₖ is 1 if true class is k, 0 otherwise (one-hot encoding)
- ŷₖ = P(y=k|x) from softmax

**Simplification**: Since only one yₖ = 1:
```
L = -log(ŷₜ)
```

Where t is the true class.

**For all examples**:
```
J(W) = -(1/m) Σᵢ₌₁ᵐ Σₖ₌₁ᴷ yₖ⁽ⁱ⁾ log(ŷₖ⁽ⁱ⁾)
```

### Gradient Derivation

**Beautiful result**: The gradient has a simple form!

```
∂L/∂zₖ = ŷₖ - yₖ
```

**For weight matrix**:
```
∂J/∂W = (1/m) (ŷ - y)Xᵀ
```

Where:
- ŷ is (m × K) matrix of predictions
- y is (m × K) matrix of one-hot labels
- X is (m × (n+1)) feature matrix

**Derivation sketch** (for one example):
```
Given:
    ŷₖ = exp(zₖ) / Σⱼ exp(zⱼ)
    L = -Σₖ yₖ log(ŷₖ)

Step 1: ∂L/∂ŷₖ = -yₖ/ŷₖ

Step 2: ∂ŷₖ/∂zⱼ = ŷₖ(δₖⱼ - ŷⱼ)
        where δₖⱼ = 1 if k=j, 0 otherwise

Step 3: ∂L/∂zₖ = Σⱼ (∂L/∂ŷⱼ)(∂ŷⱼ/∂zₖ)
                = Σⱼ (-yⱼ/ŷⱼ)·ŷⱼ(δⱼₖ - ŷₖ)
                = -Σⱼ yⱼ(δⱼₖ - ŷₖ)
                = -(yₖ - ŷₖΣⱼ yⱼ)
                = -(yₖ - ŷₖ)    [since Σⱼ yⱼ = 1]
                = ŷₖ - yₖ
```

**Same form as binary logistic regression!**

---

## Multinomial Logistic Regression

### Definition

**Multinomial logistic regression** is another name for **softmax regression**.

**Why "multinomial"?**

The model assumes data comes from a **multinomial distribution**:

```
P(y|x) ~ Multinomial(1, [p₁, p₂, ..., pₖ])
```

Where pₖ = P(y=k|x) are the softmax probabilities.

### Model Specification

**Complete model**:
```
For each class k = 1, 2, ..., K:
    zₖ = wₖᵀx
    
Probabilities:
    P(y=k|x; W) = exp(zₖ) / Σⱼ exp(zⱼ)
```

### Maximum Likelihood Estimation

**Likelihood for one example**:
```
P(y|x; W) = ∏ₖ₌₁ᴷ [P(y=k|x)]^yₖ
```

**For all examples**:
```
L(W) = ∏ᵢ₌₁ᵐ ∏ₖ₌₁ᴷ [P(y=k|x⁽ⁱ⁾)]^yₖ⁽ⁱ⁾
```

**Log-likelihood**:
```
log L(W) = Σᵢ₌₁ᵐ Σₖ₌₁ᴷ yₖ⁽ⁱ⁾ log P(y=k|x⁽ⁱ⁾)
```

**Negative log-likelihood** (our loss):
```
J(W) = -log L(W) = Categorical Cross-Entropy
```

**Key insight**: Minimizing cross-entropy = Maximizing likelihood!

### Complete Implementation

```python
import numpy as np

class SoftmaxRegression:
    """Softmax Regression / Multinomial Logistic Regression"""
    
    def __init__(self, learning_rate=0.01, num_iterations=1000, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.reg_lambda = reg_lambda
        self.W = None
        self.classes = None
        self.cost_history = []
    
    def _softmax(self, z):
        """Softmax function with numerical stability"""
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y, K):
        """Convert labels to one-hot encoding"""
        m = len(y)
        y_one_hot = np.zeros((m, K))
        y_one_hot[np.arange(m), y] = 1
        return y_one_hot
    
    def _compute_cost(self, X, y_one_hot, y_pred):
        """Compute categorical cross-entropy cost"""
        m = X.shape[0]
        
        # Cross-entropy loss
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_one_hot * np.log(y_pred)) / m
        
        # Add L2 regularization
        reg_loss = (self.reg_lambda / (2 * m)) * np.sum(self.W[1:, :] ** 2)
        
        return loss + reg_loss
    
    def fit(self, X, y):
        """
        Train softmax regression
        
        Args:
            X: Feature matrix (m × n)
            y: Labels (m,) with values in {0, 1, ..., K-1}
        """
        # Add bias column
        X_b = np.c_[np.ones(X.shape[0]), X]
        m, n = X_b.shape
        
        # Initialize
        self.classes = np.unique(y)
        K = len(self.classes)
        self.W = np.zeros((n, K))
        
        # Convert labels to one-hot
        y_one_hot = self._one_hot_encode(y, K)
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Forward pass
            z = X_b @ self.W  # (m × K)
            y_pred = self._softmax(z)  # (m × K)
            
            # Compute cost
            cost = self._compute_cost(X_b, y_one_hot, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradient
            gradient = (1/m) * X_b.T @ (y_pred - y_one_hot)  # (n × K)
            
            # Add regularization gradient
            reg_gradient = (self.reg_lambda / m) * self.W
            reg_gradient[0, :] = 0  # Don't regularize bias
            gradient += reg_gradient
            
            # Update weights
            self.W -= self.learning_rate * gradient
            
            # Print progress
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X_b = np.c_[np.ones(X.shape[0]), X]
        z = X_b @ self.W
        return self._softmax(z)
    
    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]
    
    def score(self, X, y):
        """Compute accuracy"""
        return np.mean(self.predict(X) == y)

# Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           n_classes=4, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = SoftmaxRegression(learning_rate=0.1, num_iterations=1000, reg_lambda=0.01)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\nTraining accuracy: {train_acc * 100:.2f}%")
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Plot cost
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(model.cost_history, linewidth=2)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.title('Softmax Regression Training', fontsize=16)
plt.grid(True, alpha=0.3)
plt.show()
```

### Visualization

```python
import matplotlib.pyplot as plt

# Generate 4-class data
X, y = make_classification(n_samples=400, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           n_classes=4, random_state=42)

# Train
model = SoftmaxRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X, y)

# Create mesh
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                        np.linspace(x2_min, x2_max, 200))

# Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Decision regions
ax = axes[0, 0]
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
ax.contourf(xx1, xx2, Z, alpha=0.4, cmap='viridis', levels=3)
for class_val in range(4):
    ax.scatter(X[y == class_val, 0], X[y == class_val, 1],
              label=f'Class {class_val}', s=50, edgecolors='k')
ax.set_title('Decision Regions', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2-5: Probability maps for each class
for k in range(4):
    ax = axes[(k+1)//3, (k+1)%3]
    
    # Get probabilities for class k
    Z_proba = model.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])[:, k]
    Z_proba = Z_proba.reshape(xx1.shape)
    
    contour = ax.contourf(xx1, xx2, Z_proba, levels=20, cmap='RdYlGn', alpha=0.8)
    ax.scatter(X[y == k, 0], X[y == k, 1],
              c='red', marker='s', s=100, edgecolors='k', label=f'Class {k}')
    ax.scatter(X[y != k, 0], X[y != k, 1],
              c='blue', marker='o', s=30, edgecolors='k', alpha=0.3, label='Other')
    
    ax.set_title(f'P(y={k}|x)', fontsize=14)
    ax.legend()
    plt.colorbar(contour, ax=ax, label='Probability')

# Plot 6: Training cost
ax = axes[1, 2]
ax.plot(model.cost_history, linewidth=2)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Cost', fontsize=12)
ax.set_title('Training Cost', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Comparison of Methods

### Summary Table

| Method | # Classifiers | Training Time | Prediction Time | Memory | Probabilities | Notes |
|--------|---------------|---------------|-----------------|--------|---------------|-------|
| **One-vs-Rest** | K | O(K·mn) | O(K·n) | O(K·n) | Pseudo | Simple, imbalanced data |
| **One-vs-One** | K(K-1)/2 | O(K²·mn) | O(K²·n) | O(K²·n) | Voting | Balanced, many classifiers |
| **Softmax** | 1 | O(K·mn) | O(K·n) | O(K·n) | True | Direct, principled |

### When to Use Each

**One-vs-Rest (OvR)**:
- ✅ Simple implementation needed
- ✅ Using existing binary classifier
- ✅ Small to medium K (K < 10)
- ✅ Can train in parallel
- ❌ Avoid if severe class imbalance

**One-vs-One (OvO)**:
- ✅ Classes well-separated
- ✅ Can afford many classifiers
- ✅ Want robustness to outliers
- ❌ Avoid for large K (K > 20)
- ❌ Avoid if memory constrained

**Softmax Regression**:
- ✅ Want true probabilities
- ✅ Large K
- ✅ Need calibrated confidence
- ✅ Principled probabilistic framework
- ✅ **Generally recommended**

### Accuracy Comparison

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    'One-vs-Rest': OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    'One-vs-One': OneVsOneClassifier(LogisticRegression(max_iter=1000)),
    'Softmax (Multinomial)': LogisticRegression(multi_class='multinomial', max_iter=1000)
}

# Compare
results = {}
for name, clf in classifiers.items():
    # Train
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Predict
    start = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - start
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'train_time': train_time,
        'pred_time': pred_time
    }
    
    print(f"\n{name}")
    print("=" * 50)
    print(f"Training time: {train_time:.4f}s")
    print(f"Prediction time: {pred_time:.4f}s")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Summary table
import pandas as pd
df = pd.DataFrame(results).T
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(df)
```

---

## Practical Implementation

### Using Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression

# One-vs-Rest (default for most sklearn classifiers)
ovr_clf = LogisticRegression(multi_class='ovr', max_iter=1000)

# One-vs-One
# (Use OneVsOneClassifier wrapper)
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(LogisticRegression(max_iter=1000))

# Softmax / Multinomial
softmax_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Train
ovr_clf.fit(X_train, y_train)
ovo_clf.fit(X_train, y_train)
softmax_clf.fit(X_train, y_train)

# Predict
y_pred_ovr = ovr_clf.predict(X_test)
y_pred_ovo = ovo_clf.predict(X_test)
y_pred_softmax = softmax_clf.predict(X_test)

# Probabilities
proba_ovr = ovr_clf.predict_proba(X_test)
proba_ovo = ovo_clf.predict_proba(X_test)
proba_softmax = softmax_clf.predict_proba(X_test)
```

### Best Practices

**1. Feature Scaling**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**2. Regularization**:
```python
# L2 regularization (default)
clf = LogisticRegression(C=1.0)  # C = 1/λ

# L1 regularization (feature selection)
clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
```

**3. Class Imbalance**:
```python
# Adjust class weights
clf = LogisticRegression(class_weight='balanced')

# Or manually
class_weights = {0: 1, 1: 2, 2: 1}  # Class 1 weighted more
clf = LogisticRegression(class_weight=class_weights)
```

**4. Cross-Validation**:
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Advanced Topics

### Hierarchical Softmax

For very large K (e.g., K = 100,000 words), regular softmax is too slow.

**Hierarchical softmax** organizes classes in a tree:
- Reduces complexity from O(K) to O(log K)
- Used in word2vec, language models

### Label Smoothing

Regularization technique for softmax:

Instead of hard labels [0, 0, 1, 0], use soft labels [ε, ε, 1-3ε, ε]

```python
def label_smoothing(y_one_hot, epsilon=0.1):
    """Apply label smoothing"""
    K = y_one_hot.shape[1]
    return y_one_hot * (1 - epsilon) + epsilon / K
```

**Benefits**:
- Prevents overconfidence
- Better calibration
- Improved generalization

### Temperature Scaling

Adjust confidence of predictions:

```python
def softmax_with_temperature(z, temperature=1.0):
    """Softmax with temperature scaling"""
    z_scaled = z / temperature
    return softmax(z_scaled)
```

- T < 1: More confident (peaked)
- T > 1: Less confident (smooth)
- Used for model calibration

---

## Summary and Next Steps

### What We Learned

1. **One-vs-Rest (OvR)**
   - K binary classifiers (class k vs rest)
   - Simple, efficient
   - Works with any binary classifier

2. **One-vs-One (OvO)**
   - K(K-1)/2 pairwise classifiers
   - Balanced data per classifier
   - Voting for prediction

3. **Softmax Regression**
   - Direct multiclass approach
   - Outputs probability distribution
   - Elegant gradient: ∂L/∂z = ŷ - y

4. **Multinomial Logistic Regression**
   - Same as softmax regression
   - Maximum likelihood framework
   - Principled probabilistic model

### Key Formulas

**Softmax**:
```
P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
```

**Categorical Cross-Entropy**:
```
L = -Σₖ yₖ log(ŷₖ) = -log(ŷₜ)
```

**Gradient**:
```
∂J/∂W = (1/m) (ŷ - y)Xᵀ
```

### Key Insights

✅ **Softmax** is the standard for multiclass classification  
✅ **OvR/OvO** useful when extending binary classifiers  
✅ **Softmax gradient** has beautiful simple form  
✅ **Numerical stability** matters (subtract max)  
✅ **Regularization** prevents overfitting  

### Prerequisites Checklist

Before moving on:
- [ ] Understand OvR and OvO strategies
- [ ] Know softmax function and properties
- [ ] Can derive categorical cross-entropy gradient
- [ ] Implemented softmax regression from scratch
- [ ] Understand when to use each method
- [ ] Know about numerical stability tricks

### Next Steps

1. **Regularization_L1_L2.md**
   - Ridge (L2) for multiclass
   - Lasso (L1) for feature selection
   - Elastic Net
   - When and why to regularize

2. **Feature_Engineering.md**
   - Polynomial features
   - Interaction terms
   - Feature selection
   - Dimensionality reduction

3. **Model_Evaluation.md**
   - Multi-class metrics
   - Confusion matrices
   - Per-class precision/recall
   - Micro vs macro averaging

### Practice Exercises

1. Implement OvR from scratch
2. Implement OvO from scratch
3. Derive softmax gradient step-by-step
4. Compare all three methods on real dataset
5. Visualize decision boundaries for 4+ classes
6. Implement temperature scaling
7. Apply label smoothing

### Additional Resources

**Papers**:
- Bridle (1990): "Probabilistic Interpretation of Feedforward Networks"
- Bishop (1995): "Neural Networks for Pattern Recognition"

**Books**:
- "Pattern Recognition and Machine Learning" (Bishop) - Chapter 4
- "Elements of Statistical Learning" (Hastie) - Chapter 4
- "Deep Learning" (Goodfellow) - Chapter 6

**Online**:
- CS231n: Multi-class SVM and Softmax
- Stanford CS229: Multiclass classification notes
- Colah's blog: Softmax and temperature

---

## Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║        MULTICLASS CLASSIFICATION QUICK REFERENCE               ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  ONE-VS-REST (OvR)                                             ║
║  ─────────────────                                             ║
║  • K binary classifiers                                        ║
║  • Class k vs all others                                       ║
║  • Predict: argmax_k wₖᵀx                                      ║
║                                                                ║
║  Pros: Simple, parallelizable                                  ║
║  Cons: Imbalanced data                                         ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  ONE-VS-ONE (OvO)                                              ║
║  ─────────────────                                             ║
║  • K(K-1)/2 pairwise classifiers                               ║
║  • Each pair (i,j) trained separately                          ║
║  • Predict: Majority voting                                    ║
║                                                                ║
║  Pros: Balanced data, robust                                   ║
║  Cons: Many classifiers, slow                                  ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  SOFTMAX REGRESSION                                            ║
╠════════════════════════════════════════════════════════════════╣
║  Model:                                                        ║
║    zₖ = wₖᵀx                                                   ║
║    P(y=k|x) = exp(zₖ) / Σⱼ exp(zⱼ)                             ║
║                                                                ║
║  Loss (Categorical Cross-Entropy):                             ║
║    L = -Σₖ yₖ log(ŷₖ) = -log(ŷₜ)                               ║
║                                                                ║
║  Gradient:                                                     ║
║    ∂L/∂zₖ = ŷₖ - yₖ                                            ║
║    ∂J/∂W = (1/m)(ŷ - y)Xᵀ                                      ║
║                                                                ║
║  Properties:                                                   ║
║    • Σₖ ŷₖ = 1                                                 ║
║    • ŷₖ ∈ (0,1)                                                ║
║    • Reduces to sigmoid for K=2                                ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  NUMERICAL STABILITY                                           ║
╠════════════════════════════════════════════════════════════════╣
║  Subtract max before exponential:                              ║
║    z_stable = z - max(z)                                       ║
║    softmax = exp(z_stable) / sum(exp(z_stable))                ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  COMPARISON                                                    ║
╠════════════════════════════════════════════════════════════════╣
║  Method      Classifiers  Time    Memory   Probabilities       ║
║  ─────────   ───────────  ─────   ──────   ─────────────       ║
║  OvR         K            Fast    Low      Pseudo              ║
║  OvO         K(K-1)/2     Slow    High     Voting              ║
║  Softmax     1            Fast    Low      True                ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  RECOMMENDATION                                                ║
╠════════════════════════════════════════════════════════════════╣
║  Default: Use Softmax Regression                               ║
║    • Most principled                                           ║
║    • True probabilities                                        ║
║    • Efficient                                                 ║
║                                                                ║
║  Use OvR when:                                                 ║
║    • Need to use existing binary classifier                    ║
║    • Want parallel training                                    ║
║                                                                ║
║  Use OvO when:                                                 ║
║    • Classes very well separated                               ║
║    • Small K (< 10)                                            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**Ready to move on?** Next: Learn about **regularization** to improve generalization!

---

*End of Multiclass_Linear_Classification.md*