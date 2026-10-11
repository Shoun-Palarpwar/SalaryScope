# SalaryScope: Employee Salary Prediction System

## Website

Run `python3 website.py` and open http://127.0.0.1:5050 after installing
`requirements.txt`. The local Flask server connects the frontend to the saved
Linear Regression model; the website does not retrain or replace artifacts.

The website includes an editorial home page, salary explorer with exact feature
contributions, interactive role-filtered data visualization, model explanations,
and a **Behind SalaryScope** navigation tab with the full academic study. Its
eight-part presentation supports arrow-key navigation. The report's print button
opens the browser's print dialog, where it can be saved as PDF. CSV downloads
provide the source dataset and fold metrics. Layouts adapt to narrow screens.

The Ridge diagnostic charts describe cross-validation and are labeled separately
from the deployed Linear Regression model. Google Fonts are optional; system
font fallbacks work offline. This server binds locally and is intended for an
academic demonstration, not public hosting.

SalaryScope is an academic, end-to-end machine-learning project that predicts
annual employee salary from years of experience and job role. It demonstrates
data validation, leakage-free preprocessing, regression modeling, evaluation,
artifact persistence, and reusable inference.

```text
Experience + Role -> Preprocessing + interactions -> Linear regression -> Salary
```

## Project structure

```text
salary_project/
├── data/salaries.csv
├── models/
│   ├── salary_model.joblib
│   ├── preprocess_artifacts.joblib
│   └── metrics.json
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── tests/test_pipeline.py
├── main.py
└── requirements.txt
```

## Dataset

The included synthetic dataset has the following fields:

| Column | Description |
|---|---|
| `Experience` | Years of professional experience |
| `Role` | `Junior`, `Mid`, `Senior`, or `Lead` |
| `Salary` | Annual salary in USD—the prediction target |

If the CSV does not exist, the project generates a reproducible dataset. Each
role has its own base salary, experience range, and salary-growth rate. Gaussian
noise and missing values are included to imitate imperfect real data.

The loader validates replacement datasets before training. Rows without a
salary are excluded because a missing target is not a known training label.
Missing experience is median-imputed using the training data only.

## Modeling methodology

1. Validate the raw dataset.
2. Remove rows whose target salary is missing.
3. Create a role-stratified 80/20 train/test split.
4. Fit the experience imputer and role encoder on the training split only.
5. One-hot encode roles, using `Junior` as the reference category.
6. Add `Experience × Role` interactions so each role can have a distinct slope.
7. Train a linear regression model.
8. Compare it with a mean-prediction baseline.
9. Report held-out, per-role, and five-fold cross-validation metrics.
10. Save the model, preprocessing artifacts, and evaluation results.

This order prevents test-set information from leaking into preprocessing.

## Setup and usage

Requires Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

Optional data-generation arguments:

```bash
python3 main.py --rows 2000
python3 main.py --regenerate-data
```

Make a single prediction after training:

```bash
python3 src/predict.py --experience 6 --role Senior
```

Library usage:

```python
from src.predict import predict_batch, predict_salary

salary = predict_salary(experience=6, role="Senior")

results = predict_batch([
    {"Experience": 1, "Role": "Junior"},
    {"Experience": 10, "Role": "Lead"},
])
```

Prediction inputs are validated for required fields, supported roles, finite
numeric experience, and each role's observed training range. This prevents
silent extrapolation into unsupported scenarios.

## Current results

On the included 1,000-row dataset, the current held-out test result is:

| Metric | Model | Mean baseline |
|---|---:|---:|
| R² | 0.9845 | -0.0000 |
| MAE | $4,299.78 | $41,781.58 |
| RMSE | $6,417.27 | $51,503.71 |

Exact metrics, including per-role and cross-validation results, are stored in
`models/metrics.json` after training. Because the data is synthetic, these
scores demonstrate pipeline behavior rather than real-world salary accuracy.

## Tests

The test suite uses Python's standard library and requires no separate test
runner:

```bash
python3 -m unittest discover -s tests -v
```

It covers deterministic data generation, schema validation, removal of missing
targets, train-fitted imputation, interaction features, invalid prediction
inputs, and single/batch inference.

## Analysis and model comparison

Generate the EDA figures, prediction diagnostics, and leakage-free five-fold
comparison of Linear Regression, Ridge Regression, Random Forest, and Gradient
Boosting:

```bash
python3 src/analyze.py
```

Outputs are written to `reports/`:

- `analysis_report.md` — ready-to-use academic results and interpretation
- `analysis_summary.json` — machine-readable summary
- `model_comparison_folds.csv` — metrics for every model and fold
- `figures/distributions.png`
- `figures/salary_by_role.png`
- `figures/salary_vs_experience.png`
- `figures/model_comparison.png`
- `figures/actual_vs_predicted.png`
- `figures/residuals.png`

## Future academic work

- Add real or publicly available salary data with suitable ethical safeguards.
- Explore additional features such as location, education, skills, and company size.
- Report prediction intervals and study model uncertainty.
- Evaluate errors across relevant groups before drawing compensation conclusions.

SalaryScope is intended for learning and experimentation, not for making real
employment or compensation decisions.
