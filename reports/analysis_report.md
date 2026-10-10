# SalaryScope Analysis Report

## Dataset

- Total rows: 1000
- Labeled rows used for evaluation: 950
- Missing experience values: 50
- Missing salary values excluded: 50

## Five-fold model comparison

All preprocessing was fitted separately inside each training fold.

| Model | R², mean ± SD | MAE, mean ± SD | RMSE, mean ± SD |
|---|---:|---:|---:|
| Ridge Regression | 0.9842 ± 0.0059 | $4,106.02 ± $109.19 | $6,431.35 ± $1,185.99 |
| Random Forest | 0.9881 ± 0.0031 | $4,111.73 ± $106.87 | $5,624.16 ± $673.99 |
| Linear Regression | 0.9842 ± 0.0058 | $4,125.16 ± $112.52 | $6,435.89 ± $1,167.12 |
| Gradient Boosting | 0.9825 ± 0.0035 | $4,616.99 ± $304.06 | $6,826.59 ± $679.73 |

Best model by cross-validated MAE: **Ridge Regression**.

## Best-model MAE by role

| Role | MAE |
|---|---:|
| Junior | $3,538.53 |
| Mid | $3,355.89 |
| Senior | $4,709.15 |
| Lead | $7,140.52 |

## Interpretation

The role–experience scatterplot shows why interaction terms are useful: salary
growth differs by role. The actual-versus-predicted chart shows overall fit,
while the residual chart should be inspected for changing error spread or
systematic patterns. Role-level MAE highlights categories that remain harder
to predict, particularly when they have fewer observations.

These results describe a synthetic academic dataset and must not be treated as
evidence of real-world salary accuracy or used for compensation decisions.
