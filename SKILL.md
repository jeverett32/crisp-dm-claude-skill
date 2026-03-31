---
name: crisp-dm-pipeline
description: Builds a complete, documented machine learning pipeline following the CRISP-DM framework. Use this skill whenever the user wants to build an ML pipeline, train a predictive model, do supervised learning, or work through business understanding, data understanding, data cleaning, modeling, or evaluation steps — even if they don't say "CRISP-DM" explicitly. Handles both classification and regression. Outputs either a Jupyter notebook (.ipynb) with markdown-documented phases or a fully runnable Python project that includes a CRISP-DM analysis notebook/script plus an optional operational pipeline (ETL → train → infer → scheduled runs) with saved artifacts and metadata.
---

# CRISP-DM Machine Learning Pipeline

This skill generates a complete, end-to-end ML pipeline grounded in the CRISP-DM framework (problem framing → data understanding → preparation → modeling → evaluation), and can optionally “operationalize” that pipeline into a repeatable training + inference workflow (separate code paths, saved artifacts, metadata, and scheduling readiness).

**Default output** is a CRISP-DM analysis artifact (notebook or script). **Optional output** (when the user wants “deployment”, “production”, “scheduled retraining”, “write predictions back”, or “app integration”) adds a lightweight operational pipeline layout with ETL/train/infer jobs and versioned artifacts.

---

## Step 0: Initialize a Lab Tree (workspace for all outputs)

Before writing analysis/code artifacts, create a **lab workspace folder** in the repo so the project stays organized and repeatable.

### Lab tree rules
- Put **all generated code, notebooks, and reports under the lab tree** (do not scatter files at repo root).
- Do not modify existing application code outside the lab tree unless the user explicitly asks.
- Use a short, filesystem-safe project slug derived from the user’s goal, e.g. `churn_prediction`, `late_delivery`, `house_prices`.

### Standard lab tree layout (create if missing)

```
lab/
  <project_slug>/
    README.md
    requirements.txt              # or pyproject.toml if project uses it
    data/
      raw/                        # immutable inputs (or links/notes)
      interim/                    # intermediate transforms
      processed/                  # modeling-ready tables
    notebooks/
      01_crispdm_pipeline.ipynb   # primary CRISP-DM artifact (if notebook chosen)
    src/
      __init__.py
      config.py                   # paths, constants, random seeds
      data_io.py                  # loading utilities
      features.py                 # feature engineering (shared by train + infer)
      metrics.py                  # metric helpers/baselines
      modeling.py                 # model definitions / tuning helpers
      evaluation.py               # evaluation plots/reports
    jobs/                         # operationalization (optional)
      etl_build_warehouse.py
      train_model.py
      run_inference.py
      utils_db.py                 # if DB source/sink is used
    artifacts/
      models/                     # saved model pipelines (.sav)
      runs/                       # per-run metadata/metrics
    reports/
      figures/
      tables/
    logs/
```

If the user does not want operationalization, you can omit creating `jobs/` initially, but keep the lab tree structure so the work remains organized.

---

## Step 1: Intake Interview (ask once, up front)

Before writing any code, ask a compact set of questions **one time** at the beginning. Extract what you can from the conversation context — only ask what’s genuinely missing.
**Do not generate the pipeline until you have enough information to run it end-to-end** (at minimum: data access + target definition + output choice; and if operationalizing: prediction sink).

### Ask these questions (minimum viable set)
1. **Goal & decision**: What decision will this model support, and who will use it?
2. **Data source & access**:
   - **Files**: path(s) (CSV/Excel/Parquet), or
   - **Database**: engine (SQLite/Postgres/etc), file/connection info, and table(s), or
   - **Multiple sources** that need joining/denormalizing?
3. **Target definition**:
   - Target column name (or how to construct the label)
   - If classification: what is the **positive class** (the “important” class)?
4. **Success criteria**:
   - Which metric matters (and what errors are costly)?
   - Minimum acceptable threshold or “beat baseline by X” (if unknown, propose one)
5. **Constraints**:
   - Any leakage columns to exclude, IDs to drop, time-order constraints, fairness/compliance constraints
6. **Output choice**:
   - **CRISP‑DM artifact**: notebook (`.ipynb`) vs script (`.py`)
   - **Operationalization** (repeatable train + infer workflow): **Yes/No**
     - If Yes: where should predictions go? (file, database table, app integration point)

### Additional “full pipeline” clarifiers (ask only if needed)
- **Grain / unit of analysis**: what does each row represent (customer/order/session/etc.)?
- **Join/denormalization**: do you need to combine multiple tables into one modeling table?
- **Split strategy constraints**:
  - Time-ordered data → time-based split + `TimeSeriesSplit`
  - Multiple rows per entity (e.g., customer) → group-aware split to avoid leakage
- **Inference context** (if operationalizing): what is a “new record” and how do we identify it?

### Decision rules (don’t re-ask later)
- If **problem type** is unclear, infer from target dtype/values and confirm in a single sentence.
- If the user doesn’t know a metric, choose a default based on error costs:
  - **Classification**: prefer F1 / recall / ROC AUC depending on “misses vs false alarms”
  - **Regression**: MAE or RMSE depending on sensitivity to large errors

After receiving answers, give a single brief confirmation summary (1–3 bullets) and proceed immediately to generation.

---

## Step 2: Generate the Pipeline

Generate the full pipeline as either a `.ipynb` or `.py` file depending on the user’s choice. See the format rules below.

Structure the pipeline in **five clearly labeled sections**, one per CRISP-DM phase. Each section must include:
- A **header comment or markdown cell** naming the phase and stating its purpose per CRISP-DM
- A **deliverables note** listing what this phase is supposed to produce (from the textbook)
- The **working code** for that phase

This `SKILL.md` is **self-contained**. Do not rely on any other repository reference files to know what to generate. Use the templates and checklists below as the canonical patterns.

### Step 2B (optional): Operationalize for reliable reuse (deployment-ready workflow)
If (and only if) the user opted into **Operationalization = Yes**, additionally produce a small, runnable project layout that turns CRISP‑DM into repeatable jobs:
**Important**: still generate the CRISP‑DM artifact (notebook/script). The operational job layout is an add‑on, not a replacement.

- **Separation of concerns**: shared data/feature logic is imported by both training and inference
- **Repeatability**: same inputs → same outputs; deterministic seeds where appropriate
- **Traceability**: save model + metadata + metrics with timestamps/versioning
- **Safe failure**: clear logging and explicit exceptions (no silent fallbacks)

Minimum deliverables for the operational add‑on:
- `data/` (or configured path): operational source (e.g., SQLite `shop.db`) and/or warehouse output
- `artifacts/`: saved model file (`.sav` via `joblib`), `model_metadata.json`, `metrics.json`
- `jobs/`:
  - `etl_build_warehouse.py` (if denormalization/joins are needed; otherwise a “load & validate” job)
  - `train_model.py` (reads warehouse, trains pipeline, evaluates, saves artifacts)
  - `run_inference.py` (loads latest model artifact, scores new records, writes predictions to sink)
  - `config.py` (paths, constants)
  - `utils_db.py` (db helpers if using SQLite)
- A short `README.md` section describing how to run the jobs manually and how to schedule them (Windows Task Scheduler / cron), consistent with reliability principles (repeatability, traceability, separation of concerns, safe failure).

---

## Output Format Rules

### Jupyter Notebook (`.ipynb`)
- Use markdown cells as section headers for each CRISP-DM phase
- Every major code block should be preceded by a markdown cell explaining what it does and why
- Include a markdown cell at the top with the project title, problem type, target variable, and success metric
- At the end of each phase, include a markdown cell summarizing key findings/decisions before moving to the next phase

### Python Script (`.py`)
- Use `# ============================================================` banner comments as section headers
- Use `# ---` comments to label sub-steps
- Include a docstring at the top of the file with project context
- Use `print()` statements at key steps so running the script produces a readable log
- Group code logically with blank lines between logical blocks

### Executive Summary (`.md`)
- Create an executive summary as a standalone markdown deliverable saved under the lab tree:
  - `lab/<project_slug>/reports/executive_summary.md`
- This is a **Phase 5 deliverable** and must be produced even when the primary artifact is a `.py` script.
- Keep it concise (aim ~300–800 words) and decision-oriented. Use this structure:
  - **Problem & decision** (1–3 sentences)
  - **Data & scope** (data source(s), rows/cols, time period if relevant, exclusions/leakage safeguards)
  - **Success criteria** (metric, baseline, threshold)
  - **Model result** (final test metric(s), baseline comparison, confidence/risks)
  - **Key drivers** (top features or key patterns, with caveats)
  - **Recommendation** (**GO / NO-GO / NEEDS WORK**) and next actions (max 3–5 bullets)
  - **Risks & monitoring** (what could break; what to monitor post-release)
---

## Critical Rules (The "Why" Behind Them)

**Never fit preprocessing on the full dataset before splitting.** Always put imputation, scaling, and encoding inside a `sklearn.Pipeline` with a `ColumnTransformer`. Fitting a scaler on the full dataset leaks validation statistics into training, producing optimistically biased performance estimates — a mistake that has burned real projects.

**Freeze the test set once, touch it once.** All cross-validation and hyperparameter tuning happens on `X_train` only. The test set is evaluated exactly once, at the very end, to report final performance. Using the test set for model selection is a form of overfitting.

**Always report a baseline.** For classification, compute the no-skill baseline (majority class rate). For regression, compute baseline RMSE using the mean prediction. A model that doesn't beat its baseline provides no real value — the textbook calls this out explicitly.

**Use StratifiedKFold for classification, KFold for regression.** For time-ordered data, use `TimeSeriesSplit` instead. Stratification preserves class proportions across folds, which matters especially for imbalanced datasets.

**Document decisions, not just code.** Each phase should explain *why* a choice was made (e.g., "Used median imputation because this feature is right-skewed — mean would be pulled by outliers"). This is the difference between a notebook that teaches and one that just runs.

---

## Phase Summary (Quick Reference)

| Phase | CRISP-DM Purpose | Key Deliverables |
|-------|-----------------|-----------------|
| 1. Business Understanding | Define the problem, feasibility, and success criteria | Problem statement, success metric, feasibility framing |
| 2. Data Understanding | Learn what the data contains and what it can support | Univariate stats, missing value report, target distribution, relationship plots |
| 3. Data Preparation | Transform raw data into modeling-ready form | Cleaned DataFrame, sklearn Pipeline with preprocessing, feature engineering notes |
| 4. Modeling | Train and compare candidate models using CV | CV score table, tuned model artifacts, preliminary performance |
| 5. Evaluation | Assess whether the model meets business objectives | Final metrics vs. baseline, confusion matrix (classification) or residuals (regression), go/no-go recommendation |

---

## Canonical Phase Templates (self-contained)

Use these templates as the canonical structure. Adapt variable names, file paths, and model choices to the user’s context.

### Phase 1: Business Understanding
**CRISP-DM Purpose:** define the business problem, objectives, and success criteria before touching data.

**Deliverables:**
- Problem statement and scope
- Feasibility framing: practical impact, data availability, analytical feasibility
- Success metric and baseline to beat
- Error cost analysis and metric justification

**Notebook markdown header template:**

```markdown
## Phase 1: Business Understanding
**CRISP-DM Purpose:** Define the business problem, objectives, and success criteria before any data work begins.

### Problem Statement
- **Business Question:** ...
- **Target Variable:** `...`
- **Problem Type:** Classification / Regression
- **Positive Class (if classification):** ...

### Feasibility Assessment
| Criterion | Assessment |
|---|---|
| Practical Impact | ... |
| Data Availability | ... |
| Analytical Feasibility | ... |

### Success Criteria
- **Primary metric:** ...
- **Baseline to beat:** ...
- **Minimum acceptable performance:** ...

### Error Cost Analysis
- **False Positive cost:** ...
- **False Negative cost:** ...
- **Implication for metric choice:** ...
```

### Phase 2: Data Understanding
**CRISP-DM Purpose:** become familiar with the data’s structure, variables, quality, and relationships.

**Deliverables:**
- Data description report (shape, dtypes, sample rows)
- Univariate statistics table
- Missing value report
- Target distribution
- Data quality issues list (identified, not fixed yet)
- Relationship exploration (correlations/plots)

**Canonical code pattern (script or notebook cells):**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load
df = pd.read_csv("PATH_TO_DATA.csv")  # or database query result
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
display(df.head()) if "display" in globals() else print(df.head())

# Univariate stats + missingness
desc = df.describe(include="all").T
desc["missing"] = df.isnull().sum()
desc["missing_pct"] = (df.isnull().mean() * 100).round(2)
desc["nunique"] = df.nunique()
print(desc[["count", "missing", "missing_pct", "nunique"]].head(30))

# Target distribution + baseline
TARGET = "YOUR_TARGET"
if df[TARGET].dtype == "O" or df[TARGET].nunique() <= 20:
    vc = df[TARGET].value_counts()
    print(vc)
    print(f"Baseline accuracy (majority class): {vc.max() / vc.sum():.1%}")
else:
    print(df[TARGET].describe())
```

### Phase 3: Data Preparation
**CRISP-DM Purpose:** transform raw data into modeling-ready form; fixes happen here.

**Deliverables:**
- Inclusion/exclusion report (dropped columns + reasons)
- Cleaning decisions (imputation/outliers)
- Feature engineering notes
- Train/test split (frozen test set)
- Leakage-safe preprocessing using `Pipeline` + `ColumnTransformer`

**Canonical rules:**
- Freeze test set once; do all tuning/CV on train only.
- No preprocessing fit on full dataset outside of a pipeline.

**Canonical code pattern:**

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

SEED = 42
TARGET = "YOUR_TARGET"
EXCLUDE_COLS = []  # leakage IDs, timestamps, etc.

df_clean = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
X = df_clean.drop(columns=[TARGET])
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y if y.nunique() <= 20 else None
)

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    [("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)],
    remainder="drop",
)
```

### Phase 4: Modeling
**CRISP-DM Purpose:** train/compare candidate models using CV on training set only; tune without touching test.

**Deliverables:**
- Candidate techniques + assumptions
- CV design specification (and rationale)
- CV comparison table
- Selected model + tuned parameters

**Canonical patterns:**
- Classification: `StratifiedKFold`, choose scoring aligned to Phase 1 costs.
- Regression: `KFold`, use `neg_root_mean_squared_error` or `r2` as appropriate.

```python
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

# Pick based on problem type
is_classification = (y_train.nunique() <= 20)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) if is_classification else KFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = "roc_auc" if is_classification else "neg_root_mean_squared_error"

# Example candidates (swap as needed)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

candidates = {}
if is_classification:
    candidates["LogReg"] = Pipeline([("prep", preprocessor), ("model", LogisticRegression(max_iter=2000, random_state=SEED))])
    candidates["RF"] = Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))])
else:
    candidates["Ridge"] = Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0))])
    candidates["RF"] = Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1))])

for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
    print(name, float(scores.mean()), float(scores.std()))

# Optional tuning on best candidate
# search = GridSearchCV(best_pipe, param_grid, cv=CV, scoring=SCORING, n_jobs=-1)
# search.fit(X_train, y_train)
# final_model = search.best_estimator_
```

### Phase 5: Evaluation
**CRISP-DM Purpose:** evaluate against business objectives; compare to baseline; make go/no-go recommendation.

**Deliverables:**
- Final metrics vs baseline and success threshold
- Confusion matrix + interpretation (classification) OR residual analysis (regression)
- Feature importance / interpretability summary (as applicable)
- Operational readiness review + decision
- Executive summary (`reports/executive_summary.md`)

**Canonical code pattern (classification):**

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

baseline = y_test.value_counts(normalize=True).max()
acc = accuracy_score(y_test, y_pred)
print(f"Baseline accuracy: {baseline:.4f}")
print(f"Model accuracy:    {acc:.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

**Canonical code pattern (regression):**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

baseline_pred = np.full(shape=len(y_test), fill_value=float(y_train.mean()))
baseline_rmse = mean_squared_error(y_test, baseline_pred, squared=False)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Baseline RMSE: {baseline_rmse:.4f} | Model RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
```

---

## “Done-ness” Gates (do not stop early)

Before considering the task complete, ensure the output includes:
- **Phase 1**: explicit success metric + baseline + minimum acceptable threshold (even if proposed)
- **Phase 2**: missingness report + target distribution + at least one relationship exploration
- **Phase 3**: frozen test set + leakage-safe preprocessing in a pipeline
- **Phase 4**: CV comparison across at least 2 candidate models (or justified single-model choice)
- **Phase 5**: final test evaluation + baseline comparison + go/no-go recommendation

If operationalizing:
- **Artifacts saved**: model pipeline file + metrics + metadata
- **Training/inference separation**: distinct code paths with shared feature logic
- **Prediction sink implemented**: file/db table/app integration location
  
---

---

## Operationalization Checklist (apply when operationalizing)

When generating the operational job layout, ensure:
- **Shared transformations**: feature engineering and preprocessing live in shared code; do not duplicate logic between training and inference.
- **Artifacts are complete**: saved object includes preprocessing + model (prefer saving the full `sklearn.Pipeline`).
- **Metadata exists**: training timestamp, data source snapshot description, feature list, label definition, code/config version (at minimum).
- **Metrics are logged**: include baseline comparison and chosen success metric.
- **Inference writes outputs**: predictions stored where the “app” can read them (often a DB table keyed by entity id).
