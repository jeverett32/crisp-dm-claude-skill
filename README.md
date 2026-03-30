# CRISP-DM Claude Skill (CRISP-DM ML Pipeline)

This skill generates an end-to-end supervised machine learning pipeline based on the CRISP-DM framework (as taught in IS 455). It outputs either:

- a fully documented Python script (`.py`), or
- a Jupyter notebook (`.ipynb`) with markdown + code cells

It covers five phases:

1. Business Understanding  
2. Data Understanding  
3. Data Preparation  
4. Modeling  
5. Evaluation

## How the skill works

### Step 1: Interview the user (collect only what’s missing)
Before any code is generated, the skill gathers the minimum information needed to build the pipeline safely and correctly:

- Required
  - **Data source**: file(s) and format (CSV/Excel/etc.)
  - **Target variable**: the column to predict
  - **Problem type**: classification vs regression (inferred from the target if unclear)
  - **Success metric**: what “good” means (beat a baseline, target metric threshold, etc.)
  - **Output format**: `.ipynb` or `.py`
- Optional but helpful
  - Columns to exclude (IDs, timestamps, leakage columns)
  - Time ordering (if you need walk-forward splitting / `TimeSeriesSplit`)
  - Class imbalance concerns
  - Domain feature engineering ideas

After collecting answers, it confirms a short summary (“Here’s what I’m building… Ready to generate?”) and then proceeds.

### Step 2: Generate the full pipeline (five labeled CRISP-DM sections)
The skill generates the complete pipeline with **five clearly labeled sections**, one per CRISP-DM phase. For the canonical code patterns and deliverables, it follows:

- `references/phase_guide.md`

Each phase includes documentation of what it’s supposed to produce (not just code).

## Output formats

### Jupyter Notebook (`.ipynb`)
- Uses **markdown cells** for phase headers and explanations
- Places a project summary at the top (problem type, target, success metric)
- Summarizes key decisions before moving between phases

### Python Script (`.py`)
- Uses banner comments to mark each phase
- Adds `print()` logging at key steps so the script run shows readable progress
- Organizes code into clean sub-steps with labeled comments

## Key safety rules (important “why”)

The skill is designed to avoid common leakage / overfitting mistakes:

- **Never fit preprocessing on the full dataset before splitting.**  
  Imputation, scaling, and encoding must live inside an sklearn `Pipeline` + `ColumnTransformer`.
- **Freeze the test set once.**  
  Cross-validation and hyperparameter tuning happen only on `X_train`. The test set is evaluated once at the end.
- **Always compute a baseline.**  
  Classification baseline = majority-class rate. Regression baseline = mean-prediction RMSE (etc.).
- **Choose CV properly:**
  - Classification: `StratifiedKFold`
  - Regression: `KFold`
  - Time-aware: `TimeSeriesSplit`
- **Document decisions, not just code.**  
  Each phase explains the rationale for key choices.

## What the user should provide (copy/paste checklist)

Use this checklist when starting a Claude session with the skill:

- Data source:
- Target column:
- Classification or regression:
- Success metric + baseline-to-beat:
- Output format: `.py` or `.ipynb`
- Columns to exclude (optional):
- Time-aware splitting needed? (optional):
- Any domain feature engineering ideas? (optional):

## Example Claude prompts (session-ending prompts)

Below are two prompts designed to “kill off” the session: they give Claude everything needed so it doesn’t ask follow-up questions, and it ends after producing the requested code.

### Example 1: Classification pipeline prompt (Python)

Use when your target is categorical (binary or multi-class).

```text
You are going to generate a complete, leakage-safe ML pipeline using the @crisp-dm-pipeline skill.

GOAL: Produce ONE Python script and then stop.

DATA:
- File: "data/customer_churn.csv" (CSV)
- Target column: "Churn"  (values: "Yes"/"No")

PROBLEM TYPE:
- Classification

SUCCESS METRIC:
- Primary metric: F1 score for the positive class ("Yes")
- Baseline to beat: majority-class (no-skill) F1 using the same metric definition

OUTPUT:
- Generate a fully commented Python script (.py).

EXCLUSIONS / LEAKAGE:
- Exclude columns: ["customer_id", "signup_date"] (use only for filtering if needed; do not include as features)

ASSUMPTIONS:
- If categorical encoding is needed, use an sklearn-compatible approach inside the pipeline.
- Do not ask me any questions. If something is unclear, choose a sensible default and proceed.

END CONDITION:
- Output only the final .py code (no extra explanations outside the code).
- End your response with the line: DONE
```

### Example 2: Regression pipeline prompt (Python)

Use when your target is numeric (continuous).

```text
You are going to generate a complete, leakage-safe ML pipeline using the @crisp-dm-pipeline skill.

GOAL: Produce ONE Python script and then stop.

DATA:
- File: "data/house_prices.csv" (CSV)
- Target column: "SalePrice" (numeric)

PROBLEM TYPE:
- Regression

SUCCESS METRIC:
- Primary metric: RMSE (lower is better)
- Baseline to beat: predict the training mean (baseline RMSE on the test set)

OUTPUT:
- Generate a fully commented Python script (.py).

EXCLUSIONS / LEAKAGE:
- Exclude columns: ["house_id"].
- If there are time-like columns, do not include them as raw features unless they are explicitly safe.

ASSUMPTIONS:
- Use safe preprocessing inside an sklearn Pipeline/ColumnTransformer.
- Do not ask me any questions. If something is unclear, choose a sensible default and proceed.

END CONDITION:
- Output only the final .py code (no extra explanations outside the code).
- End your response with the line: DONE
```

## Notes

- If you need time-aware splitting, include the date/time column(s) in your prompt and state that you want walk-forward / `TimeSeriesSplit`.
- If you want the notebook output instead of `.py`, swap `Generate a Python script (.py)` → `Generate a Jupyter notebook (.ipynb)`.
