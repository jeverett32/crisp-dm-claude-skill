# CRISP-DM Pipeline Skill

This repo is intentionally minimal: it contains a single Cursor/Claude skill (`SKILL.md`) and this `README.md`.

The skill generates a complete, end-to-end supervised machine learning pipeline based on the CRISP-DM framework. It can output:

- a fully documented Python script (`.py`), or
- a Jupyter notebook (`.ipynb`) with markdown + code cells
- and (optionally) a lightweight **operationalization** scaffold (repeatable ETL/train/infer jobs + saved artifacts) inside a structured lab folder.

It covers five phases:

1. Business Understanding  
2. Data Understanding  
3. Data Preparation  
4. Modeling  
5. Evaluation

Phase 5 includes an **Executive Summary** deliverable (standalone markdown).

## How the skill works

### Step 0: Create a lab workspace folder (“lab tree”)
Before generating artifacts, the skill creates (or reuses) a structured project folder:

- `lab/<project_slug>/...`

All generated notebooks, scripts, reports, artifacts, and logs live under that lab tree so work stays organized and reproducible.

### Step 1: Interview the user (ask once, up front)
Before any code is generated, the skill gathers the minimum information needed to build the pipeline end-to-end:

- Required
  - **Goal & decision**: what decision the model supports and who uses it
  - **Data access**: file path(s) or database connection info + tables
  - **Target definition**: target column (or label construction rule); positive class (if classification)
  - **Success criteria**: metric + baseline + minimum acceptable threshold (or the skill proposes defaults)
  - **Output choice**: `.ipynb` or `.py`
  - **Operationalization**: Yes/No (if Yes: where predictions should be written)
- Only if needed
  - Unit of analysis (grain), joins/denormalization requirements
  - Split constraints (time-aware, group leakage)
  - Leakage exclusions / inference-time availability constraints

After collecting answers, it confirms a short summary and proceeds immediately.

### Step 2: Generate the full pipeline (five labeled CRISP-DM sections)
The skill generates the complete pipeline with **five clearly labeled sections**, one per CRISP-DM phase. Each phase includes documentation of what it’s supposed to produce (not just code).

### Step 2B (optional): Operationalization scaffold
If you opt into operationalization, the skill also generates a small “jobs” layout under the lab tree with:

- ETL/build modeling table (if needed)
- Train (fit + evaluate + save artifacts)
- Inference (load latest model + score new records + write predictions)
- Saved outputs (model file + metrics + metadata)

## Output formats

### Jupyter Notebook (`.ipynb`)
- Uses **markdown cells** for phase headers and explanations
- Places a project summary at the top (problem type, target, success metric)
- Summarizes key decisions before moving between phases

### Python Script (`.py`)
- Uses banner comments to mark each phase
- Adds `print()` logging at key steps so the script run shows readable progress
- Organizes code into clean sub-steps with labeled comments

### Executive Summary (`.md`)
- A standalone markdown file written as a Phase 5 deliverable:
  - `lab/<project_slug>/reports/executive_summary.md`
- Contains problem framing, data/scope, metric + baseline, final test results, recommendation (GO/NO-GO/NEEDS WORK), and risks/monitoring.

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

- Goal & decision:
- Data access (file path(s) or DB + table(s)):
- Target definition (and positive class if classification):
- Success metric + baseline + minimum acceptable threshold:
- Output choice: `.py` or `.ipynb`
- Operationalization: Yes/No (if Yes: prediction sink)
- Columns to exclude / leakage constraints (optional):
- Time-aware or group-aware splitting constraints (optional):

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
