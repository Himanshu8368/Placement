# Creative Projects: Master Linear Predictors

## Complete Guide with 3 Detailed Projects + 5 Practice Projects

This document contains:
- **Part A**: 3 Detailed Projects with Complete Solutions (From Scratch + Libraries)
- **Part B**: 5 Additional Practice Projects with Dataset Links

---

# PART A: THREE DETAILED PROJECTS WITH COMPLETE SOLUTIONS

---

## Project 1: Reddit Comment Karma Predictor 🎯

### Overview
Predict Reddit comment karma (upvotes) based on writing style, sentiment, timing, and subreddit. Train separate models per subreddit to discover how YOU need to adapt communication for different audiences.

### Dataset Links
- **Pushshift Reddit Dataset**: https://files.pushshift.io/reddit/comments/
- **Kaggle Reddit Dataset**: https://www.kaggle.com/datasets/cosmos98/reddit-dataset
- **Alternative (Sample)**: Generate synthetic data (code provided)

### Why This Project is Unique
✨ Personal - It's about YOUR writing style  
✨ Multi-model - One model per subreddit, compare coefficients  
✨ Interpretable - "In r/science, formal language gets +50% upvotes"  
✨ Actionable - Concrete writing advice  

### Learning Objectives
- ✅ Logistic Regression (viral prediction)
- ✅ Ridge Regression (exact karma prediction)
- ✅ Feature engineering from text
- ✅ Regularization (prevent overfitting)
- ✅ Coefficient interpretation

---

### Implementation: Complete Solution

#### Step 1: Data Collection & Feature Engineering

```python
"""
Reddit Comment Karma Predictor
Complete implementation from scratch and with scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# STEP 1: DATA COLLECTION (Synthetic Example)
# ============================================================

print("="*70)
print("REDDIT COMMENT KARMA PREDICTOR")
print("="*70)

# Generate synthetic Reddit data (replace with real Pushshift data)
np.random.seed(42)
n_comments = 5000

data = {
    'comment_id': [f'comment_{i}' for i in range(n_comments)],
    'subreddit': np.random.choice(['science', 'funny', 'AskReddit', 'technology', 'worldnews'], n_comments),
    'score': np.random.exponential(scale=10, size=n_comments).astype(int),
    'created_utc': pd.date_range('2024-01-01', periods=n_comments, freq='5min'),
}

df = pd.DataFrame(data)

# Extract time features
df['hour'] = df['created_utc'].dt.hour
df['day_of_week'] = df['created_utc'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

print(f"\nDataset: {len(df)} comments")
print(f"Subreddits: {df['subreddit'].unique()}")
print(f"Karma stats:\n{df['score'].describe()}")

# ============================================================
# STEP 2: TEXT FEATURE ENGINEERING
# ============================================================

# Synthetic text features (in real project, extract from actual text)
df['length'] = np.random.randint(10, 500, len(df))
df['word_count'] = df['length'] // 5
df['avg_word_length'] = np.random.uniform(3, 8, len(df))
df['sentiment_polarity'] = np.random.uniform(-1, 1, len(df))
df['sentiment_subjectivity'] = np.random.uniform(0, 1, len(df))
df['exclamation_count'] = np.random.poisson(0.5, len(df))
df['question_count'] = np.random.poisson(0.3, len(df))
df['uppercase_ratio'] = np.random.uniform(0, 0.3, len(df))
df['has_link'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
df['has_quote'] = np.random.choice([0, 1], len(df), p=[0.85, 0.15])

# One-hot encode subreddit
subreddit_dummies = pd.get_dummies(df['subreddit'], prefix='sub')
df = pd.concat([df, subreddit_dummies], axis=1)

# Create target variables
df['is_viral'] = (df['score'] > 50).astype(int)  # Binary: viral or not

print(f"\nViral rate: {df['is_viral'].mean() * 100:.1f}%")
print(f"Total features: {len(df.columns)}")
```

#### Step 2: FROM SCRATCH Implementation

```python
# ============================================================
# IMPLEMENTATION FROM SCRATCH (NumPy)
# ============================================================

class LogisticRegressionScratch:
    """Logistic Regression from scratch using NumPy"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """Train the model"""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            predictions = self._sigmoid(z)
            
            # Cost (cross-entropy + L2)
            epsilon = 1e-15
            predictions_clip = np.clip(predictions, epsilon, 1 - epsilon)
            ce_loss = -(1/m) * np.sum(
                y * np.log(predictions_clip) + (1-y) * np.log(1 - predictions_clip)
            )
            l2_penalty = (self.lambda_/(2*m)) * np.sum(self.weights ** 2)
            cost = ce_loss + l2_penalty
            self.cost_history.append(cost)
            
            # Gradients
            dw = (1/m) * (X.T @ (predictions - y)) + (self.lambda_/m) * self.weights
            db = (1/m) * np.sum(predictions - y)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 200 == 0:
                print(f"Iteration {i}/{self.n_iterations}, Cost: {cost:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict classes"""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score(self, X, y):
        """Accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class RidgeRegressionScratch:
    """Ridge Regression from scratch"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Train using closed-form solution"""
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        
        # Ridge: w = (X^T X + αI)^-1 X^T y
        I = np.eye(n + 1)
        I[0, 0] = 0  # Don't regularize bias
        
        w = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        
        self.bias = w[0]
        self.weights = w[1:]
        return self
    
    def predict(self, X):
        """Predictions"""
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# ============================================================
# TRAIN FROM SCRATCH MODELS
# ============================================================

print("\n" + "="*70)
print("FROM SCRATCH IMPLEMENTATION")
print("="*70)

# Prepare features
feature_cols = ['length', 'word_count', 'sentiment_polarity', 'sentiment_subjectivity',
                'exclamation_count', 'question_count', 'uppercase_ratio', 'hour',
                'is_weekend', 'has_link', 'has_quote'] + list(subreddit_dummies.columns)

X = df[feature_cols].values
y_viral = df['is_viral'].values
y_karma = df['score'].values

# Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_viral_train, y_viral_test = y_viral[:split_idx], y_viral[split_idx:]
y_karma_train, y_karma_test = y_karma[:split_idx], y_karma[split_idx:]

# Standardize
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-8
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Train Logistic Regression (Viral Prediction)
print("\n--- Logistic Regression (Viral Prediction) ---")
log_model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000, lambda_=0.1)
log_model.fit(X_train_scaled, y_viral_train)

train_acc = log_model.score(X_train_scaled, y_viral_train)
test_acc = log_model.score(X_test_scaled, y_viral_test)

print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Train Ridge Regression (Karma Prediction)
print("\n--- Ridge Regression (Karma Prediction) ---")
ridge_model = RidgeRegressionScratch(alpha=10.0)
ridge_model.fit(X_train_scaled, y_karma_train)

train_r2 = ridge_model.score(X_train_scaled, y_karma_train)
test_r2 = ridge_model.score(X_test_scaled, y_karma_test)

y_pred_test = ridge_model.predict(X_test_scaled)
test_rmse = np.sqrt(np.mean((y_karma_test - y_pred_test) ** 2))

print(f"\nTraining R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.2f} karma points")

# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'log_coef': log_model.weights,
    'ridge_coef': ridge_model.weights,
    'abs_log': np.abs(log_model.weights)
}).sort_values('abs_log', ascending=False)

print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES (From Scratch)")
print("="*70)
print(importance_df.head(10).to_string(index=False))
```

#### Step 3: WITH Libraries Implementation

```python
# ============================================================
# IMPLEMENTATION WITH SCIKIT-LEARN
# ============================================================

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error, r2_score

print("\n" + "="*70)
print("SCIKIT-LEARN IMPLEMENTATION")
print("="*70)

# Split properly
X = df[feature_cols]
y_viral = df['is_viral']
y_karma = df['score']

X_train, X_test, y_viral_train, y_viral_test = train_test_split(
    X, y_viral, test_size=0.2, stratify=y_viral, random_state=42
)

_, _, y_karma_train, y_karma_test = train_test_split(
    X, y_karma, test_size=0.2, random_state=42
)

# ============================================================
# LOGISTIC REGRESSION WITH HYPERPARAMETER TUNING
# ============================================================

print("\n--- Logistic Regression (Viral Prediction) ---")

log_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=1000, random_state=42))
])

param_grid_log = {
    'logistic__C': [0.01, 0.1, 1.0, 10.0],
    'logistic__penalty': ['l1', 'l2'],
    'logistic__solver': ['liblinear']
}

grid_log = GridSearchCV(log_pipeline, param_grid_log, cv=5, scoring='accuracy', n_jobs=-1)
grid_log.fit(X_train, y_viral_train)

print(f"Best params: {grid_log.best_params_}")
print(f"Best CV score: {grid_log.best_score_ * 100:.2f}%")

y_pred_log = grid_log.predict(X_test)
print("\nTest Set Performance:")
print(classification_report(y_viral_test, y_pred_log, target_names=['Not Viral', 'Viral']))

# ============================================================
# RIDGE REGRESSION WITH HYPERPARAMETER TUNING
# ============================================================

print("\n--- Ridge Regression (Karma Prediction) ---")

ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

param_grid_ridge = {
    'ridge__alpha': np.logspace(-2, 3, 20)
}

grid_ridge = GridSearchCV(ridge_pipeline, param_grid_ridge, cv=5, scoring='r2', n_jobs=-1)
grid_ridge.fit(X_train, y_karma_train)

print(f"Best alpha: {grid_ridge.best_params_['ridge__alpha']:.4f}")
print(f"Best CV R²: {grid_ridge.best_score_:.4f}")

y_pred_ridge = grid_ridge.predict(X_test)
test_r2 = r2_score(y_karma_test, y_pred_ridge)
test_rmse = np.sqrt(mean_squared_error(y_karma_test, y_pred_ridge))

print(f"Test R²: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.2f}")

# ============================================================
# LASSO FOR FEATURE SELECTION
# ============================================================

print("\n--- Lasso (Feature Selection) ---")

lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])

param_grid_lasso = {
    'lasso__alpha': np.logspace(-3, 1, 20)
}

grid_lasso = GridSearchCV(lasso_pipeline, param_grid_lasso, cv=5, scoring='r2', n_jobs=-1)
grid_lasso.fit(X_train, y_karma_train)

lasso_model = grid_lasso.best_estimator_.named_steps['lasso']
selected = np.abs(lasso_model.coef_) > 1e-5

print(f"Selected {np.sum(selected)} out of {len(feature_cols)} features")

selected_features = pd.DataFrame({
    'feature': np.array(feature_cols)[selected],
    'coefficient': lasso_model.coef_[selected]
}).sort_values('coefficient', key=abs, ascending=False)

print("\nMost Important Features (Lasso):")
print(selected_features.head(10))
```

#### Step 4: Subreddit-Specific Analysis

```python
# ============================================================
# SUBREDDIT-SPECIFIC MODELS
# ============================================================

print("\n" + "="*70)
print("SUBREDDIT-SPECIFIC ANALYSIS")
print("="*70)

subreddit_models = {}
subreddit_results = {}

for subreddit in df['subreddit'].unique():
    print(f"\n{'='*70}")
    print(f"Training model for r/{subreddit}")
    print(f"{'='*70}")
    
    # Filter data
    sub_df = df[df['subreddit'] == subreddit]
    
    if len(sub_df) < 100:
        print(f"Skipping - insufficient data ({len(sub_df)} samples)")
        continue
    
    # Features (exclude subreddit dummies)
    sub_feature_cols = [col for col in feature_cols if not col.startswith('sub_')]
    
    X_sub = sub_df[sub_feature_cols]
    y_sub = sub_df['is_viral']
    
    # Split
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=42
    )
    
    # Train
    pipeline_sub = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    
    pipeline_sub.fit(X_train_sub, y_train_sub)
    test_acc_sub = pipeline_sub.score(X_test_sub, y_test_sub)
    
    # Store
    subreddit_models[subreddit] = pipeline_sub
    subreddit_results[subreddit] = {
        'accuracy': test_acc_sub,
        'coefficients': pipeline_sub.named_steps['logistic'].coef_[0],
        'feature_names': sub_feature_cols
    }
    
    print(f"Test Accuracy: {test_acc_sub * 100:.2f}%")

# ============================================================
# CROSS-SUBREDDIT COMPARISON
# ============================================================

print("\n" + "="*70)
print("WRITING STYLE COMPARISON ACROSS SUBREDDITS")
print("="*70)

comparison_features = ['sentiment_polarity', 'length', 'question_count', 'has_link']

for feature in comparison_features:
    print(f"\n{feature.upper().replace('_', ' ')}:")
    print("-"*50)
    
    for subreddit in subreddit_results:
        if feature in subreddit_results[subreddit]['feature_names']:
            feat_idx = subreddit_results[subreddit]['feature_names'].index(feature)
            coef = subreddit_results[subreddit]['coefficients'][feat_idx]
            print(f"  r/{subreddit:<15}: {coef:+.4f}")
```

#### Step 5: Personalized Insights & Visualization

```python
# ============================================================
# PERSONALIZED WRITING ADVICE
# ============================================================

def generate_writing_advice(model, feature_cols, scaler):
    """Generate actionable writing advice"""
    
    coefs = model.named_steps['logistic'].coef_[0]
    
    print("\n" + "="*70)
    print("📊 YOUR PERSONALIZED REDDIT WRITING GUIDE")
    print("="*70)
    
    # Positive features
    positive_features = [(feature_cols[i], coefs[i]) for i in range(len(coefs)) if coefs[i] > 0.1]
    positive_features.sort(key=lambda x: x[1], reverse=True)
    
    if positive_features:
        print("\n✅ DO MORE OF THIS:")
        print("-"*70)
        for feature, coef in positive_features[:5]:
            print(f"\n• {feature.replace('_', ' ').title()}")
            print(f"  Impact: +{coef:.3f} on viral probability")
            
            if 'sentiment' in feature:
                print("  💡 Use positive, enthusiastic language")
            elif 'length' in feature:
                print("  💡 Write detailed, comprehensive responses")
            elif 'question' in feature:
                print("  💡 Ask engaging questions")
            elif 'link' in feature:
                print("  💡 Include relevant sources")
    
    # Negative features
    negative_features = [(feature_cols[i], coefs[i]) for i in range(len(coefs)) if coefs[i] < -0.1]
    negative_features.sort(key=lambda x: x[1])
    
    if negative_features:
        print("\n\n❌ AVOID THIS:")
        print("-"*70)
        for feature, coef in negative_features[:5]:
            print(f"\n• {feature.replace('_', ' ').title()}")
            print(f"  Impact: {coef:.3f} on viral probability")
            
            if 'uppercase' in feature:
                print("  💡 Don't use ALL CAPS excessively")
            elif 'exclamation' in feature:
                print("  💡 Use exclamation marks sparingly")

# Generate advice
generate_writing_advice(grid_log.best_estimator_, feature_cols, StandardScaler())

# ============================================================
# VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Feature importance
ax = axes[0, 0]
top_features = importance_df.head(10)
ax.barh(range(len(top_features)), top_features['log_coef'])
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels([f[:20] for f in top_features['feature']])
ax.set_xlabel('Coefficient')
ax.set_title('Top 10 Features (Logistic Regression)')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

# Plot 2: Training convergence
ax = axes[0, 1]
ax.plot(log_model.cost_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Training Convergence (From Scratch)')
ax.grid(True, alpha=0.3)

# Plot 3: Predictions vs Actual (Ridge)
ax = axes[1, 0]
ax.scatter(y_karma_test, y_pred_ridge, alpha=0.5)
ax.plot([y_karma_test.min(), y_karma_test.max()],
       [y_karma_test.min(), y_karma_test.max()],
       'r--', linewidth=2)
ax.set_xlabel('Actual Karma')
ax.set_ylabel('Predicted Karma')
ax.set_title('Ridge Regression: Predictions vs Actual')
ax.grid(True, alpha=0.3)

# Plot 4: Subreddit comparison
ax = axes[1, 1]
if len(subreddit_results) > 0:
    subreddits = list(subreddit_results.keys())
    accuracies = [subreddit_results[s]['accuracy'] * 100 for s in subreddits]
    ax.bar(range(len(subreddits)), accuracies)
    ax.set_xticks(range(len(subreddits)))
    ax.set_xticklabels(subreddits, rotation=45)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy by Subreddit')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reddit_analysis_complete.png', dpi=150, bbox_inches='tight')
print("\n\n📊 Complete analysis saved as 'reddit_analysis_complete.png'")

print("\n" + "="*70)
print("PROJECT 1 COMPLETE!")
print("="*70)
```

---

## Project 2: Movie Plot Success Predictor 🎬

### Overview
Predict movie success (IMDb rating, box office) from script features and narrative patterns. Discover what plot structures make movies successful.

### Dataset Links
- **Cornell Movie-Dialogs Corpus**: http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
- **Kaggle - The Movies Dataset**: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- **IMDb Datasets**: https://datasets.imdbws.com/

### Why This Project is Unique
✨ Storytelling science - What narrative patterns work?  
✨ Text feature engineering - Sentiment arcs, complexity  
✨ Polynomial features - Non-linear narrative patterns  
✨ Interpretable - "Rising tension in Act 2 = +0.8 rating"  

### Learning Objectives
- ✅ Polynomial regression (non-linear narrative arcs)
- ✅ Interaction features (genre × complexity)
- ✅ Ridge vs Lasso comparison
- ✅ Multiclass classification (genre prediction)
- ✅ Feature scaling

---

### Implementation: Complete Solution

```python
"""
Movie Plot Success Predictor
Predict ratings from narrative features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

print("="*70)
print("MOVIE PLOT SUCCESS PREDICTOR")
print("="*70)

# ============================================================
# STEP 1: GENERATE SYNTHETIC MOVIE DATA
# ============================================================

np.random.seed(42)
n_movies = 1000

# Generate movie data
movies_df = pd.DataFrame({
    'movie_id': [f'movie_{i}' for i in range(n_movies)],
    'title': [f'Movie {i}' for i in range(n_movies)],
    'genre': np.random.choice(['Action', 'Drama', 'Comedy', 'Thriller', 'Romance'], n_movies),
    'year': np.random.randint(1990, 2024, n_movies),
    'rating': np.random.uniform(3.0, 9.0, n_movies),
    'box_office_millions': np.random.exponential(scale=50, size=n_movies),
    'runtime_minutes': np.random.randint(80, 180, n_movies),
})

print(f"\nDataset: {len(movies_df)} movies")
print(f"Genres: {movies_df['genre'].unique()}")
print(f"\nRating statistics:")
print(movies_df['rating'].describe())

# ============================================================
# STEP 2: NARRATIVE FEATURE ENGINEERING
# ============================================================

print("\n" + "="*70)
print("NARRATIVE FEATURE ENGINEERING")
print("="*70)

# Script-level features (synthetic - in real project, extract from actual scripts)
movies_df['total_dialogues'] = np.random.randint(100, 1000, len(movies_df))
movies_df['avg_dialogue_length'] = np.random.uniform(5, 15, len(movies_df))
movies_df['vocabulary_richness'] = np.random.uniform(0.3, 0.8, len(movies_df))

# Narrative arc features
movies_df['arc_slope'] = np.random.uniform(-0.5, 0.5, len(movies_df))  # Rising/falling
movies_df['arc_curvature'] = np.random.uniform(-0.2, 0.2, len(movies_df))  # U-shape vs inverted-U
movies_df['sentiment_variance'] = np.random.uniform(0.1, 0.5, len(movies_df))

# Emotional features
movies_df['emotion_joy'] = np.random.poisson(10, len(movies_df))
movies_df['emotion_sadness'] = np.random.poisson(8, len(movies_df))
movies_df['emotion_anger'] = np.random.poisson(5, len(movies_df))

# Plot complexity
movies_df['num_characters'] = np.random.randint(5, 30, len(movies_df))
movies_df['plot_twists_indicator'] = np.random.poisson(3, len(movies_df))

# Genre encoding
genre_dummies = pd.get_dummies(movies_df['genre'], prefix='genre')
movies_df = pd.concat([movies_df, genre_dummies], axis=1)

print(f"Total features: {len(movies_df.columns)}")

# ============================================================
# POLYNOMIAL REGRESSION FROM SCRATCH
# ============================================================

class PolynomialRegressionScratch:
    """Polynomial Regression from scratch"""
    
    def __init__(self, degree=2, lambda_=0.1):
        self.degree = degree
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
    
    def _create_polynomial_features(self, X):
        """Create polynomial features"""
        n_samples, n_features = X.shape
        poly_features = [X]
        
        # Add polynomial terms
        for d in range(2, self.degree + 1):
            for i in range(n_features):
                poly_features.append((X[:, i] ** d).reshape(-1, 1))
        
        # Add interaction terms
        if self.degree >= 2:
            for i in range(n_features):
                for j in range(i+1, n_features):
                    poly_features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        return np.hstack(poly_features)
    
    def fit(self, X, y):
        """Train using Ridge"""
        X_poly = self._create_polynomial_features(X)
        X_poly_b = np.c_[np.ones(X_poly.shape[0]), X_poly]
        
        n_features = X_poly_b.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0
        
        w = np.linalg.inv(X_poly_b.T @ X_poly_b + self.lambda_ * I) @ X_poly_b.T @ y
        
        self.bias = w[0]
        self.weights = w[1:]
        return self
    
    def predict(self, X):
        """Predictions"""
        X_poly = self._create_polynomial_features(X)
        return X_poly @ self.weights + self.bias
    
    def score(self, X, y):
        """R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# ============================================================
# TRAIN MODELS
# ============================================================

print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70)

# Prepare data
feature_cols = ['runtime_minutes', 'total_dialogues', 'vocabulary_richness',
                'arc_slope', 'arc_curvature', 'sentiment_variance',
                'num_characters', 'plot_twists_indicator'] + list(genre_dummies.columns)

X = movies_df[feature_cols].values
y = movies_df['rating'].values

# Split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Standardize
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-8
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Compare different polynomial degrees
print("\n--- FROM SCRATCH: Polynomial Regression ---")

degrees = [1, 2, 3]
results_scratch = {}

for degree in degrees:
    print(f"\nDegree {degree}:")
    poly_model = PolynomialRegressionScratch(degree=degree, lambda_=1.0)
    poly_model.fit(X_train_scaled, y_train)
    
    train_r2 = poly_model.score(X_train_scaled, y_train)
    test_r2 = poly_model.score(X_test_scaled, y_test)
    
    y_pred = poly_model.predict(X_test_scaled)
    test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    results_scratch[degree] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse
    }
    
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")

# ============================================================
# SCIKIT-LEARN IMPLEMENTATION
# ============================================================

print("\n" + "="*70)
print("SCIKIT-LEARN IMPLEMENTATION")
print("="*70)

# Prepare data properly
X = movies_df[feature_cols]
y = movies_df['rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ridge with Polynomial Features
print("\n--- Polynomial Ridge Regression ---")

poly_ridge_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

param_grid_poly = {
    'poly__degree': [1, 2, 3],
    'ridge__alpha': np.logspace(-2, 2, 20)
}

grid_poly = GridSearchCV(
    poly_ridge_pipeline, param_grid_poly, cv=5,
    scoring='r2', n_jobs=-1, verbose=0
)

grid_poly.fit(X_train, y_train)

print(f"Best parameters: {grid_poly.best_params_}")
print(f"Best CV R²: {grid_poly.best_score_:.4f}")

y_pred_poly = grid_poly.predict(X_test)
test_r2_poly = r2_score(y_test, y_pred_poly)
test_rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))

print(f"Test R²: {test_r2_poly:.4f}")
print(f"Test RMSE: {test_rmse_poly:.4f}")

# Lasso for Feature Selection
print("\n--- Lasso (Feature Selection) ---")

lasso_poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])

param_grid_lasso = {
    'lasso__alpha': np.logspace(-3, 0, 20)
}

grid_lasso = GridSearchCV(
    lasso_poly_pipeline, param_grid_lasso, cv=5,
    scoring='r2', n_jobs=-1, verbose=0
)

grid_lasso.fit(X_train, y_train)

lasso_model = grid_lasso.best_estimator_.named_steps['lasso']
poly_features = grid_lasso.best_estimator_.named_steps['poly']

poly_feature_names = poly_features.get_feature_names_out(feature_cols)
selected = np.abs(lasso_model.coef_) > 1e-5

print(f"Selected {np.sum(selected)} out of {len(poly_feature_names)} features")

important_features = pd.DataFrame({
    'feature': poly_feature_names[selected],
    'coefficient': lasso_model.coef_[selected],
    'abs_coefficient': np.abs(lasso_model.coef_[selected])
}).sort_values('abs_coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(important_features.head(10))

# ============================================================
# NARRATIVE PATTERN INSIGHTS
# ============================================================

print("\n" + "="*70)
print("📽️  WHAT MAKES A SUCCESSFUL MOVIE PLOT")
print("="*70)

# Get coefficients from best model
best_model = grid_poly.best_estimator_
ridge = best_model.named_steps['ridge']
poly = best_model.named_steps['poly']

poly_names = poly.get_feature_names_out(feature_cols)
coefs = ridge.coef_

insights_df = pd.DataFrame({
    'feature': poly_names,
    'coefficient': coefs,
    'abs_coefficient': np.abs(coefs)
}).sort_values('abs_coefficient', ascending=False)

print("\nTOP NARRATIVE SUCCESS FACTORS:")
print("-"*70)

for idx, row in insights_df.head(15).iterrows():
    feature = row['feature']
    coef = row['coefficient']
    
    if abs(coef) > 0.1:
        effect = "increases" if coef > 0 else "decreases"
        print(f"\n{feature}:")
        print(f"  Impact: {coef:+.4f} on rating")
        
        if 'arc_slope' in feature and '^' not in feature:
            if coef > 0:
                print("  💡 Rising emotional trajectory works (hopeful endings)")
            else:
                print("  💡 Falling trajectory preferred (tragedies)")
        
        elif 'arc_curvature' in feature:
            if coef > 0:
                print("  💡 U-shaped arc successful (descent then redemption)")
            else:
                print("  💡 Inverted-U preferred (rise then fall)")
        
        elif 'vocabulary_richness' in feature and '^2' in feature:
            print("  💡 Moderate complexity optimal (neither too simple nor complex)")
        
        elif 'num_characters' in feature:
            if coef > 0:
                print("  💡 Ensemble casts perform better")
            else:
                print("  💡 Character-driven stories preferred")

# ============================================================
# VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Model comparison
ax = axes[0, 0]
degrees = list(results_scratch.keys())
train_r2s = [results_scratch[d]['train_r2'] for d in degrees]
test_r2s = [results_scratch[d]['test_r2'] for d in degrees]

ax.plot(degrees, train_r2s, 'o-', label='Training R²', linewidth=2, markersize=8)
ax.plot(degrees, test_r2s, 's-', label='Test R²', linewidth=2, markersize=8)
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('R² Score')
ax.set_title('Model Performance vs Polynomial Degree')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(degrees)

# Plot 2: Predictions vs Actual
ax = axes[0, 1]
ax.scatter(y_test, y_pred_poly, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
       'r--', linewidth=2)
ax.set_xlabel('Actual Rating')
ax.set_ylabel('Predicted Rating')
ax.set_title('Predictions vs Actual (Best Model)')
ax.grid(True, alpha=0.3)

# Plot 3: Feature importance
ax = axes[0, 2]
top_features = insights_df.head(10)
ax.barh(range(len(top_features)), top_features['coefficient'])
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels([f[:25] + '...' if len(f) > 25 else f for f in top_features['feature']], fontsize=8)
ax.set_xlabel('Coefficient')
ax.set_title('Top 10 Features')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

# Plot 4: Arc patterns
ax = axes[1, 0]
rising = movies_df[movies_df['arc_slope'] > 0]
falling = movies_df[movies_df['arc_slope'] < 0]
ax.boxplot([rising['rating'], falling['rating']], 
           labels=['Rising Arc', 'Falling Arc'])
ax.set_ylabel('Rating')
ax.set_title('Emotional Arc Direction vs Rating')
ax.grid(True, alpha=0.3, axis='y')

# Plot 5: Genre comparison
ax = axes[1, 1]
genre_ratings = movies_df.groupby('genre')['rating'].mean().sort_values()
ax.barh(range(len(genre_ratings)), genre_ratings.values)
ax.set_yticks(range(len(genre_ratings)))
ax.set_yticklabels(genre_ratings.index)
ax.set_xlabel('Average Rating')
ax.set_title('Genre Performance')
ax.grid(True, alpha=0.3, axis='x')

# Plot 6: Complexity sweet spot
ax = axes[1, 2]
ax.scatter(movies_df['vocabulary_richness'], movies_df['rating'], alpha=0.3)
x_complex = movies_df['vocabulary_richness'].values
y_rating = movies_df['rating'].values
z = np.polyfit(x_complex, y_rating, 2)
p = np.poly1d(z)
x_line = np.linspace(x_complex.min(), x_complex.max(), 100)
ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Quadratic fit')
ax.set_xlabel('Vocabulary Richness')
ax.set_ylabel('Rating')
ax.set_title('Complexity Sweet Spot')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('movie_analysis_complete.png', dpi=150, bbox_inches='tight')
print("\n\n📊 Complete analysis saved as 'movie_analysis_complete.png'")

print("\n" + "="*70)
print("PROJECT 2 COMPLETE!")
print("="*70)
```

---

## Project 3: Chess Opening Success Rate Predictor ♟️

### Overview
Predict chess game outcomes based on opening, position evaluation, and time management. Build a personalized model for YOUR playstyle.

### Dataset Links
- **Lichess Open Database**: https://database.lichess.org/
- **Kaggle Chess Dataset**: https://www.kaggle.com/datasets/datasnaek/chess
- **Chess.com API**: https://www.chess.com/news/view/published-data-api

### Why This Project is Unique
✨ Personal - About YOUR playstyle  
✨ Time-series features - Position evaluation over moves  
✨ Multi-target - Win/loss/draw + exact outcome  
✨ Actionable - Concrete improvement recommendations  

### Learning Objectives
- ✅ Multinomial logistic regression (3 outcomes)
- ✅ Feature interactions (opening × rating_diff)
- ✅ Time-series features
- ✅ Personalized recommendations

---

### Implementation: Complete Solution

```python
"""
Chess Opening Success Rate Predictor
Predict game outcomes and generate improvement plan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

print("="*70)
print("CHESS OPENING SUCCESS RATE PREDICTOR")
print("="*70)

# ============================================================
# STEP 1: GENERATE SYNTHETIC CHESS DATA
# ============================================================

np.random.seed(42)
n_games = 2000

chess_df = pd.DataFrame({
    'game_id': [f'game_{i}' for i in range(n_games)],
    'white_rating': np.random.randint(1000, 2500, n_games),
    'black_rating': np.random.randint(1000, 2500, n_games),
    'opening': np.random.choice([
        'Sicilian Defense', 'Italian Game', 'French Defense',
        'Queens Gambit', 'Kings Indian', 'English Opening',
        'Ruy Lopez', 'Caro-Kann', 'Scandinavian', 'Pirc Defense'
    ], n_games),
    'time_control': np.random.choice(['blitz', 'rapid', 'classical'], n_games, p=[0.5, 0.3, 0.2]),
    'result': np.random.choice(['white_win', 'black_win', 'draw'], n_games, p=[0.45, 0.45, 0.1]),
    'num_moves': np.random.randint(20, 100, n_games),
    'player_color': np.random.choice(['white', 'black'], n_games),
})

# Calculate features
chess_df['rating_diff'] = np.where(
    chess_df['player_color'] == 'white',
    chess_df['white_rating'] - chess_df['black_rating'],
    chess_df['black_rating'] - chess_df['white_rating']
)

chess_df['player_won'] = np.where(
    (chess_df['player_color'] == 'white') & (chess_df['result'] == 'white_win') |
    (chess_df['player_color'] == 'black') & (chess_df['result'] == 'black_win'),
    1, 0
)

print(f"\nDataset: {len(chess_df)} games")
print(f"Win rate: {chess_df['player_won'].mean() * 100:.1f}%")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================

print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

# Position evaluation features (synthetic)
chess_df['opening_eval'] = np.random.normal(0, 0.5, len(chess_df))
chess_df['middlegame_eval'] = np.random.normal(0, 1.5, len(chess_df))
chess_df['endgame_eval'] = np.random.normal(0, 2.0, len(chess_df))

# Evaluation trajectory
chess_df['eval_slope'] = (chess_df['endgame_eval'] - chess_df['opening_eval']) / chess_df['num_moves']
chess_df['max_advantage'] = chess_df[['opening_eval', 'middlegame_eval', 'endgame_eval']].max(axis=1)

# Game phase features
chess_df['reached_middlegame'] = (chess_df['num_moves'] > 15).astype(int)
chess_df['reached_endgame'] = (chess_df['num_moves'] > 40).astype(int)

# Time features
chess_df['avg_time_per_move'] = np.where(
    chess_df['time_control'] == 'blitz', 3,
    np.where(chess_df['time_control'] == 'rapid', 10, 60)
)

# One-hot encoding
opening_dummies = pd.get_dummies(chess_df['opening'], prefix='opening')
time_control_dummies = pd.get_dummies(chess_df['time_control'], prefix='tc')

chess_df = pd.concat([chess_df, opening_dummies, time_control_dummies], axis=1)

# Create outcome (0=loss, 1=draw, 2=win)
def map_outcome(row):
    if row['player_won'] == 1:
        return 2  # win
    elif row['result'] == 'draw':
        return 1  # draw
    else:
        return 0  # loss

chess_df['outcome'] = chess_df.apply(map_outcome, axis=1)

print(f"Total features: {len(chess_df.columns)}")

# ============================================================
# MULTINOMIAL LOGISTIC REGRESSION FROM SCRATCH
# ============================================================

class MultinomialLogisticRegression:
    """Multinomial Logistic Regression from scratch"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.weights = None
        self.cost_history = []
    
    def _softmax(self, z):
        """Softmax function"""
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y, num_classes):
        """One-hot encoding"""
        m = len(y)
        y_encoded = np.zeros((m, num_classes))
        y_encoded[np.arange(m), y] = 1
        return y_encoded
    
    def fit(self, X, y):
        """Train the model"""
        m, n = X.shape
        num_classes = len(np.unique(y))
        
        self.weights = np.zeros((n, num_classes))
        y_encoded = self._one_hot_encode(y, num_classes)
        
        for i in range(self.n_iterations):
            z = X @ self.weights
            predictions = self._softmax(z)
            
            # Cost
            epsilon = 1e-15
            predictions_clip = np.clip(predictions, epsilon, 1 - epsilon)
            ce_loss = -(1/m) * np.sum(y_encoded * np.log(predictions_clip))
            l2_penalty = (self.lambda_/(2*m)) * np.sum(self.weights ** 2)
            cost = ce_loss + l2_penalty
            
            self.cost_history.append(cost)
            
            # Gradient
            gradient = (1/m) * X.T @ (predictions - y_encoded)
            gradient += (self.lambda_/m) * self.weights
            
            # Update
            self.weights -= self.learning_rate * gradient
            
            if i % 200 == 0:
                print(f"Iteration {i}/{self.n_iterations}, Cost: {cost:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = X @ self.weights
        return self._softmax(z)
    
    def predict(self, X):
        """Predict classes"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def score(self, X, y):
        """Accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# ============================================================
# TRAIN MODELS
# ============================================================

print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70)

# Prepare data
feature_cols = ['rating_diff', 'num_moves', 'opening_eval', 'middlegame_eval',
                'endgame_eval', 'eval_slope', 'max_advantage',
                'reached_middlegame', 'reached_endgame'] + \
               list(opening_dummies.columns) + list(time_control_dummies.columns)

X = chess_df[feature_cols].values
y = chess_df['outcome'].values

# Split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Standardize
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-8
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Train from scratch
print("\n--- FROM SCRATCH: Multinomial Logistic Regression ---")

multi_model = MultinomialLogisticRegression(
    learning_rate=0.1,
    n_iterations=1000,
    lambda_=0.1
)

multi_model.fit(X_train_scaled, y_train)

train_acc = multi_model.score(X_train_scaled, y_train)
test_acc = multi_model.score(X_test_scaled, y_test)

print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Confusion matrix
y_pred_scratch = multi_model.predict(X_test_scaled)
cm_scratch = confusion_matrix(y_test, y_pred_scratch)

print("\nConfusion Matrix (From Scratch):")
print("         Loss  Draw  Win")
print(f"Loss:    {cm_scratch[0,0]:4d}  {cm_scratch[0,1]:4d}  {cm_scratch[0,2]:4d}")
print(f"Draw:    {cm_scratch[1,0]:4d}  {cm_scratch[1,1]:4d}  {cm_scratch[1,2]:4d}")
print(f"Win:     {cm_scratch[2,0]:4d}  {cm_scratch[2,1]:4d}  {cm_scratch[2,2]:4d}")

# ============================================================
# SCIKIT-LEARN IMPLEMENTATION
# ============================================================

print("\n" + "="*70)
print("SCIKIT-LEARN IMPLEMENTATION")
print("="*70)

# Prepare data properly
X = chess_df[feature_cols]
y = chess_df['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Basic multinomial logistic regression
print("\n--- Multinomial Logistic Regression ---")

basic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(multi_class='multinomial', max_iter=1000))
])

param_grid_basic = {
    'logistic__C': [0.01, 0.1, 1.0, 10.0],
    'logistic__solver': ['lbfgs', 'saga']
}

grid_basic = GridSearchCV(
    basic_pipeline, param_grid_basic, cv=5,
    scoring='accuracy', n_jobs=-1, verbose=0
)

grid_basic.fit(X_train, y_train)

print(f"Best parameters: {grid_basic.best_params_}")
print(f"Best CV accuracy: {grid_basic.best_score_ * 100:.2f}%")

y_pred = grid_basic.predict(X_test)

print("\nTest Set Performance:")
print(classification_report(y_test, y_pred, target_names=['Loss', 'Draw', 'Win']))

# ============================================================
# OPENING-SPECIFIC ANALYSIS
# ============================================================

print("\n" + "="*70)
print("OPENING-SPECIFIC ANALYSIS")
print("="*70)

opening_performance = {}

for opening in chess_df['opening'].unique():
    opening_df = chess_df[chess_df['opening'] == opening]
    
    if len(opening_df) < 50:
        continue
    
    win_rate = (opening_df['outcome'] == 2).mean()
    loss_rate = (opening_df['outcome'] == 0).mean()
    draw_rate = (opening_df['outcome'] == 1).mean()
    
    opening_performance[opening] = {
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate,
        'games': len(opening_df)
    }

opening_perf_df = pd.DataFrame(opening_performance).T
opening_perf_df = opening_perf_df.sort_values('win_rate', ascending=False)

print("\nYOUR BEST OPENINGS:")
print(opening_perf_df.head(5))

print("\nYOUR WORST OPENINGS:")
print(opening_perf_df.tail(5))

# ============================================================
# PERSONALIZED CHESS IMPROVEMENT PLAN
# ============================================================

print("\n" + "="*70)
print("♟️  YOUR PERSONALIZED CHESS IMPROVEMENT PLAN")
print("="*70)

# Get coefficients
model = grid_basic.best_estimator_.named_steps['logistic']
coef_loss = model.coef_[0]
coef_draw = model.coef_[1]
coef_win = model.coef_[2]

coef_df = pd.DataFrame({
    'feature': feature_cols,
    'loss_coef': coef_loss,
    'draw_coef': coef_draw,
    'win_coef': coef_win,
    'win_impact': coef_win - coef_loss
}).sort_values('win_impact', ascending=False)

print("\n🏆 WHAT INCREASES YOUR WIN PROBABILITY:")
print("-"*70)

for idx, row in coef_df.head(10).iterrows():
    feature = row['feature']
    impact = row['win_impact']
    
    if impact > 0.05:
        print(f"\n✓ {feature.replace('_', ' ').title()}:")
        print(f"  Impact: +{impact:.3f}")
        
        if 'rating_diff' in feature:
            print("  💡 You perform best against lower-rated opponents")
        elif 'eval_slope' in feature and impact > 0:
            print("  💡 You convert advantages well (good endgame)")
        elif 'opening_' in feature:
            opening_name = feature.replace('opening_', '').replace('_', ' ')
            print(f"  💡 Strong with {opening_name} - play it more!")
        elif 'tc_blitz' in feature:
            print("  💡 You excel in blitz time control")
        elif 'reached_endgame' in feature:
            print("  💡 Endgame is your strength")

print("\n\n❌ WHAT HURTS YOUR WIN PROBABILITY:")
print("-"*70)

for idx, row in coef_df.tail(10).iterrows():
    feature = row['feature']
    impact = row['win_impact']
    
    if impact < -0.05:
        print(f"\n✗ {feature.replace('_', ' ').title()}:")
        print(f"  Impact: {impact:.3f}")
        
        if 'opening_' in feature:
            opening_name = feature.replace('opening_', '').replace('_', ' ')
            print(f"  💡 Weak with {opening_name} - study or avoid")
        elif 'tc_classical' in feature:
            print("  💡 Struggle in long games - work on stamina")

# Time control analysis
print("\n\n⏱️  TIME CONTROL ANALYSIS:")
print("-"*70)

for tc in ['blitz', 'rapid', 'classical']:
    tc_games = chess_df[chess_df['time_control'] == tc]
    win_rate = (tc_games['outcome'] == 2).mean()
    
    print(f"\n{tc.upper()}:")
    print(f"  Win rate: {win_rate * 100:.1f}%")
    print(f"  Games: {len(tc_games)}")

print("\n\n📚 RECOMMENDED TRAINING FOCUS:")
print("-"*70)

# Analyze weakest phase
phases = ['opening_eval', 'middlegame_eval', 'endgame_eval']
phase_avg = {p: chess_df[p].mean() for p in phases}
weakest_phase = min(phase_avg, key=phase_avg.get)

if 'opening' in weakest_phase:
    print("\n1. Opening Preparation (High Priority)")
    print("   • Study opening principles and common traps")
    print("   • Practice your best openings against computer")
elif 'middlegame' in weakest_phase:
    print("\n1. Tactical Training (High Priority)")
    print("   • Solve 20-30 tactics puzzles daily")
    print("   • Study combination themes")
else:
    print("\n1. Endgame Technique (High Priority)")
    print("   • Study basic endgames")
    print("   • Practice conversion techniques")

print("\n2. Weekly Practice Schedule:")
print("   Monday:    Tactics (30min) + Review losses (15min)")
print("   Tuesday:   Opening prep (45min)")
print("   Wednesday: Endgame study (30min)")
print("   Thursday:  Play 3-5 games in best time control")
print("   Friday:    Study weakest opening (45min)")
print("   Saturday:  Tournament or serious games (2hr)")
print("   Sunday:    Review week's games (1hr)")

# ============================================================
# VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Confusion matrix
ax = axes[0, 0]
sns.heatmap(cm_scratch, annot=True, fmt='d', cmap='Blues', ax=ax,
           xticklabels=['Loss', 'Draw', 'Win'],
           yticklabels=['Loss', 'Draw', 'Win'])
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title('Confusion Matrix')

# Plot 2: Training convergence
ax = axes[0, 1]
ax.plot(multi_model.cost_history)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Training Convergence')
ax.grid(True, alpha=0.3)

# Plot 3: Opening performance
ax = axes[0, 2]
top_openings = opening_perf_df.head(5)
x = range(len(top_openings))
ax.barh(x, top_openings['win_rate'] * 100, alpha=0.7, color='green', label='Win')
ax.barh(x, top_openings['draw_rate'] * 100, left=top_openings['win_rate'] * 100,
       alpha=0.7, color='gray', label='Draw')
ax.set_yticks(x)
ax.set_yticklabels([name[:15] for name in top_openings.index], fontsize=9)
ax.set_xlabel('Percentage')
ax.set_title('Your Best Openings')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Plot 4: Feature importance
ax = axes[1, 0]
top_features = coef_df.head(10)
ax.barh(range(len(top_features)), top_features['win_impact'])
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels([f[:20] for f in top_features['feature']], fontsize=9)
ax.set_xlabel('Win Impact')
ax.set_title('Top 10 Features')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

# Plot 5: Time control comparison
ax = axes[1, 1]
tc_win_rates = chess_df.groupby('time_control')['player_won'].mean() * 100
ax.bar(range(len(tc_win_rates)), tc_win_rates.values)
ax.set_xticks(range(len(tc_win_rates)))
ax.set_xticklabels(tc_win_rates.index)
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate by Time Control')
ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Position evaluation trajectory
ax = axes[1, 2]
won_games = chess_df[chess_df['player_won'] == 1]
lost_games = chess_df[chess_df['player_won'] == 0]

won_traj = [won_games[p].mean() for p in phases]
lost_traj = [lost_games[p].mean() for p in phases]

ax.plot(['Opening', 'Middlegame', 'Endgame'], won_traj,
       'o-', linewidth=2, markersize=8, label='Won Games', color='green')
ax.plot(['Opening', 'Middlegame', 'Endgame'], lost_traj,
       's-', linewidth=2, markersize=8, label='Lost Games', color='red')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_ylabel('Position Evaluation')
ax.set_title('Game Phase Patterns')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chess_analysis_complete.png', dpi=150, bbox_inches='tight')
print("\n\n📊 Complete analysis saved as 'chess_analysis_complete.png'")

print("\n" + "="*70)
print("PROJECT 3 COMPLETE!")
print("="*70)
print("\nYou've completed all 3 detailed projects!")
print("✅ Reddit Comment Karma Predictor")
print("✅ Movie Plot Success Predictor")
print("✅ Chess Opening Success Predictor")
```

---

# PART B: FIVE ADDITIONAL PRACTICE PROJECTS

Below are 5 more creative projects with real dataset links for further practice.

---

## Project 4: Spotify Hit Song Predictor 🎵

### Overview
Predict Billboard hit success from audio features. Discover what makes music successful across genres and decades.

### Dataset Links
- **Spotify Million Song Dataset**: https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs
- **Spotify Hit Predictor**: https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset
- **Billboard Hot 100**: https://www.kaggle.com/datasets/dhruvildave/billboard-the-hot-100-songs

### Key Features
- Audio features: danceability, energy, valence, tempo
- Genre-specific models
- Decade analysis (70s vs 2020s patterns)
- Feature interactions: danceability × energy

### Learning Objectives
- Binary classification (hit vs non-hit)
- Polynomial features for non-linear patterns
- Lasso for feature selection
- Temporal analysis

### Quick Start Code
```python
# Load Spotify data
spotify_df = pd.read_csv('spotify_songs.csv')
billboard_df = pd.read_csv('billboard_hot100.csv')

# Merge to create hit label
spotify_df['is_hit'] = spotify_df['track_id'].isin(billboard_df['track_id']).astype(int)

# Train logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## Project 5: Flight Delay Predictor ✈️

### Overview
Predict flight delays and optimize travel strategy. Choose the best flights based on historical patterns.

### Dataset Links
- **US Flight Delays**: https://www.kaggle.com/datasets/usdot/flight-delays
- **Bureau of Transportation**: https://www.transtats.bts.gov/DL_SelectFields.aspx
- **Airport Weather**: https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events

### Key Features
- Time-based: hour, day, season
- Airport congestion
- Airline performance
- Weather conditions

### Learning Objectives
- Multi-target: delay minutes (regression) + delayed? (classification)
- Feature interactions: airline × airport × time
- Time-series patterns
- Optimization (find best flight time)

### Quick Start Code
```python
# Feature engineering
flights['is_delayed'] = (flights['arrival_delay'] > 15).astype(int)
flights['hour'] = flights['scheduled_departure'].dt.hour
flights['is_rush_hour'] = flights['hour'].isin([7,8,9,17,18,19]).astype(int)

# Train Ridge regression
from sklearn.linear_model import Ridge
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)
```

---

## Project 6: Restaurant Review Rating Predictor 🍽️

### Overview
Predict restaurant ratings from reviews and attributes. Generate recommendations for restaurant owners.

### Dataset Links
- **Yelp Open Dataset**: https://www.yelp.com/dataset
- **Google Places Reviews**: https://www.kaggle.com/datasets/atharvap329/google-reviews-dataset
- **Zomato Dataset**: https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data

### Key Features
- Text features: sentiment, word count, readability
- Restaurant attributes: price, cuisine, location
- Review patterns

### Learning Objectives
- Text + structured data
- Sentiment analysis
- Ridge vs Lasso comparison
- Menu optimization insights

### Quick Start Code
```python
from textblob import TextBlob

# Extract sentiment
def extract_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

yelp_df['sentiment'] = yelp_df['review_text'].apply(extract_sentiment)

# Train model
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

---

## Project 7: GitHub Repository Success Predictor 💻

### Overview
Predict GitHub stars and trending status. Learn what makes open-source projects successful.

### Dataset Links
- **GitHub Archive**: https://www.gharchive.org/
- **BigQuery GitHub**: https://console.cloud.google.com/marketplace/product/github/github-repos
- **Kaggle GitHub Dataset**: https://www.kaggle.com/datasets/nikhil25803/github-dataset

### Key Features
- README quality (length, sections, examples)
- Code metrics (languages, complexity)
- Social signals (forks, issues, PRs)
- Repository age and activity

### Learning Objectives
- Text features from README
- Polynomial regression (star growth)
- Classification (trending prediction)
- Developer insights

### Quick Start Code
```python
# README features
repos_df['has_installation'] = repos_df['readme'].str.contains('install', case=False, na=False).astype(int)
repos_df['num_sections'] = repos_df['readme'].str.count('#')

# Predict stars (log-transformed)
y = np.log1p(repos_df['stars'])
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)
```

---

## Project 8: Apartment Rent Predictor 🏠

### Overview
Predict rental prices with neighborhood analysis. Find bargains and identify gentrifying areas.

### Dataset Links
- **USA Housing Listings**: https://www.kaggle.com/datasets/austinreese/usa-housing-listings
- **Real Estate Data**: https://www.kaggle.com/datasets/rtatman/real-estate-data
- **US Census Data**: https://data.census.gov/

### Key Features
- Spatial features: distance to center, amenities
- Neighborhood clustering
- Transit accessibility
- Time-series trends

### Learning Objectives
- Polynomial regression (price per sqft²)
- Spatial feature engineering
- Bargain detection (outliers)
- Trend analysis (gentrification)

### Quick Start Code
```python
from geopy.distance import geodesic

# Distance to city center
city_center = (40.7589, -73.9851)  # NYC
apts_df['distance_to_center'] = apts_df.apply(
    lambda row: geodesic((row['lat'], row['lon']), city_center).miles,
    axis=1
)

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train Ridge
from sklearn.linear_model import Ridge
model = Ridge(alpha=100.0)
model.fit(X_train_poly, y_train)
```

---

## Summary: Complete Learning Path

### Projects Overview

| # | Project | Dataset Size | Difficulty | Time | Key Skills |
|---|---------|--------------|------------|------|------------|
| 1 | **Reddit Karma** | 5K comments | ⭐⭐⭐ | 2 weeks | Text features, multi-model |
| 2 | **Movie Success** | 1K movies | ⭐⭐⭐ | 2 weeks | Polynomial, narrative patterns |
| 3 | **Chess Openings** | 2K games | ⭐⭐⭐⭐ | 3 weeks | Multinomial, personalization |
| 4 | **Spotify Hits** | 1.2M songs | ⭐⭐⭐ | 2 weeks | Audio features, temporal |
| 5 | **Flight Delays** | 5M flights | ⭐⭐⭐⭐ | 3 weeks | Multi-target, optimization |
| 6 | **Restaurant Reviews** | 7M reviews | ⭐⭐⭐ | 2 weeks | NLP, sentiment |
| 7 | **GitHub Stars** | 100K+ repos | ⭐⭐ | 1 week | Text analysis, developer insights |
| 8 | **Apartment Rent** | 50K listings | ⭐⭐⭐⭐ | 3 weeks | Spatial, trend analysis |

### Skills Mastery Checklist

After completing all projects:

✅ **Implementation**
- [x] From-scratch (NumPy)
- [x] Production (Scikit-learn)
- [x] Hyperparameter tuning
- [x] Pipeline creation

✅ **Models**
- [x] Linear/Ridge/Lasso regression
- [x] Binary/multinomial logistic
- [x] Polynomial regression
- [x] Feature interactions

✅ **Feature Engineering**
- [x] Text features (NLP)
- [x] Time-series features
- [x] Spatial features
- [x] Audio features

✅ **Evaluation**
- [x] Metrics (R², RMSE, accuracy, F1)
- [x] Cross-validation
- [x] Learning curves
- [x] Bias-variance diagnosis

✅ **Interpretation**
- [x] Coefficient analysis
- [x] Feature importance
- [x] Actionable insights
- [x] Personalized recommendations

### Final Tips

1. **Start with Projects 1-3** (detailed solutions provided)
2. **Pick ONE project 4-8** based on your interests
3. **Document everything** - create blog posts
4. **Make it personal** - use YOUR data when possible
5. **Deploy** - create web apps with Streamlit/Flask
6. **Share** - GitHub, LinkedIn, portfolio

---

## Conclusion

You now have:
- **3 Complete Detailed Projects** with full implementations (from scratch + libraries)
- **5 Additional Practice Projects** with real dataset links and quick start code
- **Comprehensive skill coverage** across all linear predictor concepts
- **Portfolio-ready projects** that stand out

**Start coding and become a Linear Predictors expert!** 🚀

---

*End of Creative_Projects_Complete.md*