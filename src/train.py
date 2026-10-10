"""
train.py
========
Trains a Linear Regression model to predict Salary from Experience
and Role, evaluates it on a held-out test set, and persists both the
trained model and the fitted preprocessing artifacts to disk so
`predict.py` can reuse them later without retraining.
"""

import sys
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Ensure the project root is on sys.path so `src.*` always resolves to the
# same module path (e.g. "src.preprocess"), whether this file is run as
# `python main.py`, `python -m src.train`, or directly as `python src/train.py`.
# This matters for joblib/pickle, which stores the module path of any custom
# classes (like PreprocessArtifacts) and needs it to match at load time.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocess import preprocess_data
from src.data_loader import load_dataset

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "salary_model.joblib"
ARTIFACTS_PATH = MODEL_DIR / "preprocess_artifacts.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute R^2, MAE, and RMSE on the test set."""
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {"r2_score": r2, "mae": mae, "rmse": rmse}


def train_model(csv_path: str = "data/salaries.csv", n_rows: int = 1000) -> dict:
    """
    End-to-end training routine:
        1. Load raw data
        2. Preprocess (impute + encode)
        3. Split 80/20 train/test
        4. Fit LinearRegression
        5. Evaluate on the test set
        6. Persist model + preprocessing artifacts + metrics to models/

    Returns the metrics dict.
    """
    raw_df = load_dataset(csv_path=csv_path, n_rows=n_rows)
    X, y, artifacts = preprocess_data(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[train] Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[train] Model trained.")

    metrics = evaluate_model(model, X_test, y_test)

    print("\n[train] === Test Set Performance ===")
    print(f"    R^2 Score : {metrics['r2_score']:.4f}")
    print(f"    MAE       : ${metrics['mae']:,.2f}")
    print(f"    RMSE      : ${metrics['rmse']:,.2f}")

    # Show learned coefficients for interpretability.
    print("\n[train] === Learned Coefficients ===")
    for name, coef in zip(X.columns, model.coef_):
        print(f"    {name:15s}: {coef:,.2f}")
    print(f"    {'Intercept':15s}: {model.intercept_:,.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(artifacts, ARTIFACTS_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[train] Model saved to '{MODEL_PATH}'")
    print(f"[train] Preprocessing artifacts saved to '{ARTIFACTS_PATH}'")
    print(f"[train] Metrics saved to '{METRICS_PATH}'")

    return metrics


if __name__ == "__main__":
    train_model()
