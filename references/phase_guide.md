# CRISP-DM Phase Guide
## Canonical Code Patterns and Deliverables

This file contains the detailed templates for each of the five pipeline phases. When generating a pipeline, pull the relevant patterns from here and adapt them to the user's dataset and problem type.

---

## Phase 1: Business Understanding

### Purpose (CRISP-DM)
Define the business problem clearly before touching data. Establish success criteria so evaluation has a target to measure against. Assess feasibility across three dimensions: practical impact, data availability, and analytical feasibility.

### Deliverables (per textbook)
- Project scope and objectives
- Feasibility framing (practical impact + data availability + analytical feasibility)
- Success metric (what model performance threshold justifies using this model?)
- Cost/benefit framing (what errors are most expensive?)

### Template (Notebook — markdown cell)

```markdown
# Phase 1: Business Understanding
**CRISP-DM Purpose:** Define the business problem, objectives, and success criteria before any data work begins.

## Problem Statement
- **Business Question:** [What decision or process does this model support?]
- **Target Variable:** `[column_name]` — [description of what it represents]
- **Problem Type:** [Classification / Regression]
- **Positive Class (if classification):** [which class label matters most, and why]

## Feasibility Assessment
| Criterion | Assessment |
|-----------|-----------|
| Practical Impact | [Why does solving this matter? What decisions change?] |
| Data Availability | [What data do we have? Is it sufficient?] |
| Analytical Feasibility | [Is there reason to believe the signal exists?] |

## Success Criteria
- **Primary metric:** [accuracy / F1 / AUC / RMSE / R² — chosen based on the cost of errors]
- **Baseline to beat:** [majority-class rate / mean-prediction RMSE / domain benchmark]
- **Minimum acceptable performance:** [the threshold below which the model isn't worth deploying]

## Error Cost Analysis
- **False Positive cost:** [What happens when we predict positive but it's actually negative?]
- **False Negative cost:** [What happens when we miss a true positive?]
- **Implication for metric choice:** [Which metric does this cost structure prioritize?]
```

---

## Phase 2: Data Understanding

### Purpose (CRISP-DM)
Become familiar with the data's structure, variables, scale, and limitations. Identify the target variable. Surface data quality issues for later remediation. Build intuition about relationships before modeling.

### Deliverables (per textbook)
- Data description report (shape, dtypes, sample rows)
- Univariate statistics table
- Missing value report
- Target variable distribution
- Data quality report (issues identified, not yet fixed)
- Relationship exploration (correlations, pairplots)

### Template

```python
# ── Phase 2: Data Understanding ──────────────────────────────────────────────
# CRISP-DM: Learn what the data contains and what it can support.
# Deliverables: data description report, univariate stats, missing value report,
#               target distribution, relationship exploration, data quality notes.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 2.1 Load and describe ────────────────────────────────────────────────────
df = pd.read_csv("YOUR_DATA.csv")  # adjust path / format as needed
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nData types:\n{df.dtypes.value_counts()}")
df.head()
```

```python
# ── 2.2 Univariate statistics ────────────────────────────────────────────────
# Generates the 'data description report' — distributions, ranges, types.
desc = df.describe(include='all').T
desc['missing'] = df.isnull().sum()
desc['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
desc['nunique'] = df.nunique()
print(desc[['count', 'missing', 'missing_pct', 'nunique', 'mean', 'std', 'min', '50%', 'max']])
```

```python
# ── 2.3 Missing value report ─────────────────────────────────────────────────
# Data quality report: identify issues here; fixes happen in Phase 3.
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing):
    print("Columns with missing values:")
    print((missing / len(df) * 100).round(2).to_string())
else:
    print("No missing values found.")
```

```python
# ── 2.4 Target variable distribution ────────────────────────────────────────
TARGET = "YOUR_TARGET_COLUMN"

# For classification:
print(df[TARGET].value_counts())
print(f"\nBaseline accuracy (majority class): {df[TARGET].value_counts(normalize=True).max():.1%}")
df[TARGET].value_counts().plot(kind='bar', title=f'Target Distribution: {TARGET}')
plt.tight_layout()
plt.show()

# For regression (swap in):
# print(df[TARGET].describe())
# df[TARGET].hist(bins=40)
# plt.title(f'Target Distribution: {TARGET}')
# plt.show()
```

```python
# ── 2.5 Relationship exploration ─────────────────────────────────────────────
# Bivariate: how do numeric features correlate with each other and the target?
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
corr = df[numeric_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Top correlations with target
if TARGET in numeric_cols:
    print(f"\nTop correlations with {TARGET}:")
    print(corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(10))
```

---

## Phase 3: Data Preparation

### Purpose (CRISP-DM)
Transform raw data into a modeling-ready dataset. This is where issues identified in Phase 2 get fixed. All transformations must be encapsulated in a sklearn `Pipeline` to prevent data leakage during cross-validation.

### Deliverables (per textbook)
- Inclusion/exclusion report (which columns were dropped and why)
- Data cleaning report (imputation strategy, outlier handling)
- Feature engineering notes (derived attributes created)
- Merged data report (if joining multiple sources)
- Final `X_train`, `X_test`, `y_train`, `y_test` splits
- `sklearn.Pipeline` with `ColumnTransformer` for all preprocessing

### Template

```python
# ── Phase 3: Data Preparation ─────────────────────────────────────────────────
# CRISP-DM: Clean, transform, and engineer features into a modeling-ready dataset.
# All transforms go inside a sklearn Pipeline — never fit on the full dataset.
# Deliverables: cleaned DataFrame, preprocessing pipeline, train/test splits.

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# ── 3.1 Structural wrangling ─────────────────────────────────────────────────
# Drop columns that are structurally useless (IDs, constants, near-empty).
# Document each drop with a reason.

def drop_useless_columns(df, missing_thresh=0.95, unique_thresh=0.95, verbose=True):
    """Drop columns that cannot contribute meaningful signal to a model."""
    to_drop = []
    for col in df.columns:
        pct_missing = df[col].isna().mean()
        pct_unique = df[col].nunique() / len(df)
        if pct_missing >= missing_thresh:
            to_drop.append((col, f"{pct_missing:.0%} missing"))
        elif pct_unique >= unique_thresh and df[col].dtype in ['object', 'int64']:
            to_drop.append((col, f"{pct_unique:.0%} unique (likely an ID)"))
        elif df[col].nunique() == 1:
            to_drop.append((col, "constant column"))
    if verbose and to_drop:
        print("Dropping columns:")
        for col, reason in to_drop:
            print(f"  {col}: {reason}")
    return df.drop(columns=[c for c, _ in to_drop])

df_clean = drop_useless_columns(df.copy())

# ── 3.2 Manually exclude leakage / irrelevant columns ────────────────────────
# List any columns that shouldn't be features (IDs, future-leaking data, etc.)
EXCLUDE_COLS = []  # e.g. ["id", "date", "post_event_column"]
df_clean = df_clean.drop(columns=[c for c in EXCLUDE_COLS if c in df_clean.columns])
```

```python
# ── 3.3 Feature engineering ──────────────────────────────────────────────────
# Create derived features here, before splitting. Document what was created and why.
# Example: df_clean['ratio'] = df_clean['col_a'] / df_clean['col_b']
# (Add domain-specific feature engineering here)
pass
```

```python
# ── 3.4 Train/test split ─────────────────────────────────────────────────────
# Freeze the test set ONCE. It will not be touched again until final evaluation.
# Use stratify=y for classification to preserve class balance.

TARGET = "YOUR_TARGET_COLUMN"
SEED = 42

X = df_clean.drop(columns=[TARGET])
y = df_clean[TARGET]

# For classification, encode string labels to integers if needed
# from sklearn.preprocessing import LabelEncoder
# y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y  # remove for regression
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

```python
# ── 3.5 Preprocessing pipeline (leakage-safe) ────────────────────────────────
# All imputation, scaling, and encoding goes inside a Pipeline/ColumnTransformer.
# Fitting these on the full dataset before CV would leak test-set statistics.

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),  # median is robust to outliers
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    remainder="drop"
)

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
```

---

## Phase 4: Modeling

### Purpose (CRISP-DM)
Train candidate models on the prepared data. Use cross-validation on the training set to compare models fairly. Tune hyperparameters without touching the test set. Document parameter settings and preliminary performance.

### Deliverables (per textbook)
- Selected candidate techniques with documented assumptions
- Test design specification (CV strategy and rationale)
- Trained model artifacts with parameter settings
- CV performance comparison table
- Preliminary performance summary

### Template — Classification

```python
# ── Phase 4: Modeling (Classification) ───────────────────────────────────────
# CRISP-DM: Select, train, and compare candidate models using cross-validation.
# The test set is NEVER used here — only for final evaluation in Phase 5.
# Deliverables: CV score table, best-tuned pipeline, parameter documentation.

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = "roc_auc"  # change to "f1", "accuracy", etc. based on business need

candidates = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED))
    ]),
    "Decision Tree": Pipeline([
        ("prep", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=8, random_state=SEED))
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingClassifier(n_estimators=200, random_state=SEED))
    ]),
}

print(f"Cross-validation ({CV.n_splits}-fold StratifiedKFold), scoring: {SCORING}\n")
cv_results = {}
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:30s}  mean={scores.mean():.4f}  std={scores.std():.4f}")

best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\nBest candidate: {best_name} (mean CV {SCORING}: {cv_results[best_name].mean():.4f})")
```

```python
# ── 4.2 Hyperparameter tuning (best candidate) ───────────────────────────────
# Tune the best-performing candidate using GridSearchCV or Optuna.
# Still on X_train only — test set remains frozen.

from sklearn.model_selection import GridSearchCV

# Example: tuning Random Forest (adapt param_grid to your best candidate)
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 8, 16],
    "model__min_samples_leaf": [1, 5, 10],
}

best_pipe = candidates[best_name]
search = GridSearchCV(best_pipe, param_grid, cv=CV, scoring=SCORING, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

print(f"Best CV {SCORING}: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")
final_model = search.best_estimator_
```

### Template — Regression (swap in for Phase 4)

```python
# ── Phase 4: Modeling (Regression) ───────────────────────────────────────────
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = "neg_root_mean_squared_error"  # or "r2"

candidates = {
    "Ridge Regression": Pipeline([
        ("prep", preprocessor),
        ("model", Ridge(alpha=1.0))
    ]),
    "Decision Tree": Pipeline([
        ("prep", preprocessor),
        ("model", DecisionTreeRegressor(max_depth=8, random_state=SEED))
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(n_estimators=200, random_state=SEED))
    ]),
}

print(f"Cross-validation ({CV.n_splits}-fold KFold), scoring: {SCORING}\n")
cv_results = {}
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:30s}  mean={-scores.mean():.4f}  std={scores.std():.4f}")

best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\nBest candidate: {best_name}")
```

---

## Phase 5: Evaluation

### Purpose (CRISP-DM)
Determine whether the model meets technical AND business objectives. This is the final phase before a deployment decision. Evaluate against the baseline established in Phase 1. Make a go/no-go recommendation.

### Deliverables (per textbook)
- Evaluation results summary (technical performance vs. business objectives)
- Confusion matrix with interpretation (classification) or residual plots (regression)
- Comparison against no-skill baseline
- Feature importance
- Deployment readiness review
- Final decision and action plan (deploy / refine / cancel)

### Template — Classification

```python
# ── Phase 5: Evaluation (Classification) ─────────────────────────────────────
# CRISP-DM: Assess whether the model meets business objectives.
# This is the ONE time we use the test set.
# Deliverables: confusion matrix, classification report, baseline comparison,
#               feature importance, and a go/no-go recommendation.

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score,
    roc_auc_score, roc_curve
)

# ── 5.1 Final evaluation on frozen test set ───────────────────────────────────
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, 'predict_proba') else None

# ── 5.2 Baseline comparison ───────────────────────────────────────────────────
baseline_acc = y_test.value_counts(normalize=True).max()
model_acc = accuracy_score(y_test, y_pred)
print(f"Baseline accuracy (majority class): {baseline_acc:.4f}")
print(f"Model accuracy:                     {model_acc:.4f}")
print(f"Improvement over baseline:          {model_acc - baseline_acc:+.4f}\n")
```

```python
# ── 5.3 Confusion matrix ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.show()
```

```python
# ── 5.4 Classification report ────────────────────────────────────────────────
print("Classification Report:")
print(classification_report(y_test, y_pred))

if y_prob is not None:
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
```

```python
# ── 5.5 Feature importance ────────────────────────────────────────────────────
# Extract feature importances from tree-based models.
# For linear models, use coefficient magnitudes instead.
try:
    model_step = final_model.named_steps["model"]
    feature_names = (
        final_model.named_steps["prep"]
        .get_feature_names_out()
    )
    importances = model_step.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    feat_imp.head(20).plot(kind='barh', figsize=(8, 6))
    plt.title("Top 20 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
except AttributeError:
    pass  # not all models have feature_importances_
```

```python
# ── 5.6 Deployment readiness review and go/no-go decision ────────────────────
print("=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"Success metric target: [paste your Phase 1 threshold here]")
print(f"Final model {SCORING}: {model_acc:.4f}")
print(f"Baseline {SCORING}: {baseline_acc:.4f}")
print(f"Beats baseline: {'YES' if model_acc > baseline_acc else 'NO'}")
print()
print("Deployment readiness questions:")
print("  [ ] Does model performance meet the Phase 1 success threshold?")
print("  [ ] Is required data available at inference time?")
print("  [ ] Are prediction latency requirements met?")
print("  [ ] Are there fairness, compliance, or risk concerns?")
print()
print("Decision: [DEPLOY / REFINE / CANCEL] — [reason]")
```

### Template — Regression (swap in for Phase 5)

```python
# ── Phase 5: Evaluation (Regression) ─────────────────────────────────────────
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Baseline: predict the training mean for every test observation
baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
baseline_rmse = mean_squared_error(y_test, baseline_pred, squared=False)
model_rmse = mean_squared_error(y_test, y_pred, squared=False)
model_mae = mean_absolute_error(y_test, y_pred)
model_r2 = r2_score(y_test, y_pred)

print(f"Baseline RMSE (mean prediction): {baseline_rmse:.4f}")
print(f"Model RMSE:                      {model_rmse:.4f}")
print(f"Model MAE:                       {model_mae:.4f}")
print(f"Model R²:                        {model_r2:.4f}")

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals vs. Predicted")
plt.tight_layout()
plt.show()
```

---

## Adapting for Time-Series / Walk-Forward Data

When the data has a temporal ordering (e.g., sports seasons, financial data), replace:
- `train_test_split` with a date-based or index-based cutoff split
- `StratifiedKFold`/`KFold` with `TimeSeriesSplit`
- Note in Phase 1 that leakage from future data is a key risk

```python
from sklearn.model_selection import TimeSeriesSplit
CV = TimeSeriesSplit(n_splits=5)
# Sort X_train, y_train by date before passing to cross_val_score
```

---

## Notebook vs. Script: Phase Header Templates

### Notebook markdown cell (use for .ipynb)
```markdown
---
## Phase N: [Phase Name]
**CRISP-DM Purpose:** [one sentence from the textbook definition]
**Key Deliverables:** [bullet list of what this phase produces]
---
```

### Script banner comment (use for .py)
```python
# ============================================================
# PHASE N: [PHASE NAME]
# CRISP-DM Purpose: [one sentence]
# Key Deliverables: [comma-separated list]
# ============================================================
```
