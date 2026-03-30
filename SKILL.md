---
name: crisp-dm-pipeline
description: Builds a complete, documented machine learning pipeline following the CRISP-DM framework (as taught in IS 455 at BYU). Use this skill whenever the user wants to build an ML pipeline, train a predictive model, do supervised learning, or work through business understanding, data understanding, data cleaning, modeling, or evaluation steps — even if they don't say "CRISP-DM" explicitly. Handles both classification and regression. Outputs either a Jupyter notebook (.ipynb) with markdown-documented phases or a fully commented Python script (.py).
---

# CRISP-DM Machine Learning Pipeline

This skill generates a complete, end-to-end ML pipeline grounded in the CRISP-DM framework from the IS 455 textbook (*Machine Learning in Python* by Mark Keith, BYU). The pipeline covers five phases: Business Understanding, Data Understanding, Data Preparation, Modeling, and Evaluation.

---

## Step 1: Interview the User

Before writing any code, collect the following. Extract what you can from the conversation context — only ask for what's genuinely missing.

**Required:**
1. **Data source**: What file(s) or data do they have? (CSV, Excel, multiple tables to join?)
2. **Target variable**: What column are they trying to predict?
3. **Problem type**: Classification (categorical outcome) or regression (numeric outcome)? If unclear, look at the target variable — binary/multi-class → classification; continuous → regression.
4. **Success metric**: What does "good" look like? (e.g., "beat a 74% seed baseline", "RMSE under X", "maximize recall for the minority class") If they don't know, suggest a sensible default and explain it.
5. **Output format**: Jupyter notebook (`.ipynb`) or Python script (`.py`)?

**Optional but helpful:**
- Are there columns that should be excluded (IDs, timestamps, leakage columns)?
- Is there a temporal ordering that requires time-aware splitting (walk-forward / TimeSeriesSplit)?
- Is class imbalance a concern?
- Are there domain-specific features worth engineering?

Once you have the answers, briefly confirm your understanding before generating: "Here's what I'm building: [1-sentence summary]. Ready to generate?"

---

## Step 2: Generate the Pipeline

Generate the full pipeline as either a `.ipynb` or `.py` file depending on the user's choice. See the format rules below.

Structure the pipeline in **five clearly labeled sections**, one per CRISP-DM phase. Each section must include:
- A **header comment or markdown cell** naming the phase and stating its purpose per CRISP-DM
- A **deliverables note** listing what this phase is supposed to produce (from the textbook)
- The **working code** for that phase

Read `references/phase_guide.md` for the detailed template of each phase — it contains the canonical code patterns, deliverables, and decision rules for all five phases.

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

For full phase templates with canonical code, read `references/phase_guide.md`.
