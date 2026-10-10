"""Train, evaluate, and persist the employee salary regression model."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset
from src.preprocess import build_preprocessor, transform_features

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "salary_model.joblib"
ARTIFACTS_PATH = MODEL_DIR / "preprocess_artifacts.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2_score": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def cross_validate_model(labeled_df) -> dict[str, dict[str, float]]:
    """Run role-stratified CV with preprocessing refitted inside every fold."""
    n_splits = min(5, int(labeled_df["Role"].value_counts().min()))
    if n_splits < 2:
        raise ValueError("Each represented role needs at least two labeled rows.")

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    fold_metrics = []
    for train_indices, test_indices in splitter.split(labeled_df, labeled_df["Role"]):
        fold_train = labeled_df.iloc[train_indices]
        fold_test = labeled_df.iloc[test_indices]
        fold_artifacts = build_preprocessor(fold_train)
        fold_X_train = transform_features(fold_train, fold_artifacts)
        fold_X_test = transform_features(fold_test, fold_artifacts)
        fold_model = LinearRegression().fit(fold_X_train, fold_train["Salary"])
        fold_metrics.append(
            _regression_metrics(
                fold_test["Salary"], fold_model.predict(fold_X_test)
            )
        )

    summary = {"folds": n_splits}
    for metric_name in ("r2_score", "mae", "rmse"):
        values = np.array([fold[metric_name] for fold in fold_metrics])
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return summary


def train_model(csv_path: str | Path = PROJECT_ROOT / "data" / "salaries.csv",
                n_rows: int = 1000) -> dict:
    """Train with leakage-free preprocessing and evaluate on held-out data."""
    raw_df = load_dataset(csv_path=str(csv_path), n_rows=n_rows)
    labeled_df = raw_df.dropna(subset=["Salary"]).copy()
    dropped = len(raw_df) - len(labeled_df)
    if dropped:
        print(f"[train] Dropped {dropped} rows with missing target salaries.")

    role_counts = labeled_df["Role"].value_counts()
    stratify = labeled_df["Role"] if role_counts.min() >= 2 else None
    train_df, test_df = train_test_split(
        labeled_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    artifacts = build_preprocessor(train_df)
    X_train = transform_features(train_df, artifacts)
    X_test = transform_features(test_df, artifacts)
    y_train = train_df["Salary"]
    y_test = test_df["Salary"]

    print(f"[train] Train set: {len(train_df)} rows | Test set: {len(test_df)} rows")
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = _regression_metrics(y_test, predictions)

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    metrics["baseline"] = _regression_metrics(y_test, baseline.predict(X_test))
    metrics["cross_validation"] = cross_validate_model(labeled_df)
    metrics["by_role"] = {}
    for role in sorted(test_df["Role"].unique()):
        mask = test_df["Role"].eq(role).to_numpy()
        metrics["by_role"][role] = {
            "count": int(mask.sum()),
            "mae": float(mean_absolute_error(y_test.iloc[mask], predictions[mask])),
        }

    print("\n[train] === Test Set Performance ===")
    print(f"    R^2 Score   : {metrics['r2_score']:.4f}")
    print(f"    MAE         : ${metrics['mae']:,.2f}")
    print(f"    RMSE        : ${metrics['rmse']:,.2f}")
    print(f"    Baseline MAE: ${metrics['baseline']['mae']:,.2f}")
    cv = metrics["cross_validation"]
    print(
        f"    {cv['folds']}-fold CV MAE: ${cv['mae']['mean']:,.2f} "
        f"(+/- ${cv['mae']['std']:,.2f})"
    )

    print("\n[train] === Learned Coefficients ===")
    for name, coefficient in zip(X_train.columns, model.coef_):
        print(f"    {name:30s}: {coefficient:,.2f}")
    print(f"    {'Intercept':30s}: {model.intercept_:,.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(artifacts, ARTIFACTS_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"\n[train] Saved model artifacts under '{MODEL_DIR}'.")
    return metrics


if __name__ == "__main__":
    train_model()
