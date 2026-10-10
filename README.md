# SalaryScope

SalaryScope is an end-to-end machine learning project that predicts employee
salaries from **years of experience** and **job role** using a `scikit-learn`
Linear Regression model.

```
Experience + Role  ─────►  Linear Regression  ─────►  Predicted Salary
```

## Project Structure

```
SalaryScope/
├── data/
│   └── salaries.csv          # Dataset (auto-generated on first run if missing)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Loads/generates the dataset
│   ├── preprocess.py         # Missing-value imputation + one-hot encoding
│   ├── train.py               # Train/test split, model training, evaluation
│   └── predict.py            # Load trained model & predict on new inputs
├── models/                   # Created after training: model + artifacts + metrics
│   ├── salary_model.joblib
│   ├── preprocess_artifacts.joblib
│   └── metrics.json
├── main.py                   # Single entry point — runs the full pipeline
├── requirements.txt
└── README.md
```

## Dataset

`data/salaries.csv` contains three columns:

| Column       | Type    | Description                                  |
|--------------|---------|-----------------------------------------------|
| `Experience` | float   | Years of professional experience              |
| `Role`       | string  | One of: `Junior`, `Mid`, `Senior`, `Lead`      |
| `Salary`     | float   | Annual salary in USD (target variable)        |

If the file doesn't already exist, `main.py` / `src/data_loader.py` will
generate a **realistic 1,000-row synthetic dataset** on first run, with each
role having a distinct base salary and growth-per-year-of-experience, plus
Gaussian noise. ~5% of `Experience` values and ~5% of `Salary` values are
intentionally set to missing (`NaN`) to exercise the preprocessing step, just
like real-world HR data.

You can drop in your own real dataset at `data/salaries.csv` (same three
columns) and the pipeline will use it instead of generating synthetic data.

## Setup

**Requirements:** Python 3.9+

```bash
# 1. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline

Run everything — data loading, preprocessing, training, evaluation, and a
few demo predictions — with a single command from the project root:

```bash
python main.py
```

Optional flags:

```bash
python main.py --rows 2000            # generate a larger synthetic dataset (only if data/salaries.csv doesn't exist)
python main.py --regenerate-data      # force-regenerate data/salaries.csv, overwriting any existing file
```

### What happens when you run it

1. **Load data** — reads `data/salaries.csv`, generating it first if it doesn't exist.
2. **Preprocess** — imputes missing `Experience`/`Salary` values with the column median, and one-hot encodes `Role`.
3. **Split** — 80% train / 20% test, with a fixed random seed for reproducibility.
4. **Train** — fits a `LinearRegression` model on the training set.
5. **Evaluate** — computes **R² Score**, **MAE**, and **RMSE** on the held-out test set and prints the learned coefficients.
6. **Persist** — saves the trained model, preprocessing artifacts, and metrics to `models/`.
7. **Demo predictions** — runs a few example predictions across all four roles.

### Example output

```
[train] === Test Set Performance ===
    R^2 Score : 0.9408
    MAE       : $6,690.50
    RMSE      : $11,595.13

[train] === Learned Coefficients ===
    Experience     : 3,167.03
    Role_Junior    : -48,676.78
    Role_Mid       : -27,347.19
    Role_Senior    : 10,081.84
    Role_Lead      : 65,942.12
    Intercept      : 92,906.32
```

## Making Individual Predictions

Once the model has been trained (i.e. after running `main.py` at least once),
you can predict a salary for a single employee without retraining:

```bash
python src/predict.py --experience 6 --role Senior
```

```
Predicted salary for a Senior with 6.0 years of experience:
    $121,990.31
```

Or use it as a library from your own Python code:

```python
from src.predict import predict_salary

salary = predict_salary(experience=6, role="Senior")
print(salary)   # 121990.31
```

For batch predictions on multiple records at once, use `predict_batch`:

```python
from src.predict import predict_batch

records = [
    {"Experience": 1, "Role": "Junior"},
    {"Experience": 10, "Role": "Lead"},
]
df = predict_batch(records)
print(df)
```

## Re-training on New Data

Replace `data/salaries.csv` with your own data (same `Experience`, `Role`,
`Salary` columns) and re-run:

```bash
python main.py
```

This retrains the model on the new data and overwrites the saved model,
preprocessing artifacts, and metrics in `models/`.

## Notes on Design Choices

- **Median imputation** is used for both `Experience` and `Salary` because
  it's robust to outliers compared to mean imputation, and works well for
  the roughly-linear salary distributions in this dataset.
- **One-hot encoding** (rather than label/ordinal encoding) is used for
  `Role` because job levels aren't necessarily evenly spaced in salary
  impact — one-hot encoding lets the model learn a separate coefficient
  for each role without assuming equal spacing.
- Preprocessing is **fit once on the training data path and reused**
  (via `models/preprocess_artifacts.joblib`) so that predictions on new,
  unseen data are transformed identically to how the model was trained —
  a common production requirement to avoid train/serve skew.
- A fixed `random_state=42` is used throughout for reproducible results.

## Extending This Project

- Swap `LinearRegression` for `RandomForestRegressor` or `GradientBoostingRegressor` in `src/train.py` for non-linear relationships.
- Add more features (e.g. `Location`, `Education`, `CompanySize`) to `data_loader.py`, `preprocess.py`, and `predict.py`.
- Wrap `predict.py` in a REST API (e.g. FastAPI/Flask) for real-time inference.
- Add unit tests (`pytest`) covering `preprocess.py` and `predict.py`.
