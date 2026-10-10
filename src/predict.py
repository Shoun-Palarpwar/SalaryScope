"""
predict.py
==========
Loads the trained model + preprocessing artifacts from disk and uses
them to predict salaries for new, unseen employee records.

Can be used as a library (`predict_salary(...)`) or run directly from
the command line for a quick manual prediction:

    python src/predict.py --experience 5 --role Senior
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

# Ensure the project root is on sys.path so `src.*` always resolves to the
# same module path, whether this file is run as `python main.py`,
# `python -m src.predict`, or directly as `python src/predict.py`.
# This matters for joblib/pickle, which needs the module path of any
# custom classes (like PreprocessArtifacts) to match between save and load.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocess import transform_features

MODEL_DIR = _PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "salary_model.joblib"
ARTIFACTS_PATH = MODEL_DIR / "preprocess_artifacts.joblib"

VALID_ROLES = ["Junior", "Mid", "Senior", "Lead"]


def _validate_prediction_records(df: pd.DataFrame, artifacts=None) -> pd.DataFrame:
    """Return validated prediction inputs with consistent column types."""
    if df.empty:
        raise ValueError("At least one prediction record is required.")
    missing = {"Experience", "Role"}.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction records are missing fields: {sorted(missing)}")

    validated = df.copy()
    try:
        validated["Experience"] = pd.to_numeric(validated["Experience"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Experience must be numeric.") from exc

    invalid_roles = sorted(set(validated["Role"].dropna()) - set(VALID_ROLES))
    if validated["Role"].isna().any() or invalid_roles:
        raise ValueError(f"Role must be one of {VALID_ROLES}; invalid values: {invalid_roles}")
    if validated["Experience"].isna().any():
        raise ValueError("Experience cannot be missing.")
    if (~validated["Experience"].map(lambda value: float("-inf") < value < float("inf"))).any():
        raise ValueError("Experience must be finite.")
    if (validated["Experience"] < 0).any():
        raise ValueError("Experience cannot be negative.")

    if artifacts is not None:
        for index, row in validated.iterrows():
            low, high = artifacts.role_experience_ranges[row["Role"]]
            if not low <= row["Experience"] <= high:
                raise ValueError(
                    f"Experience for {row['Role']} must be within the training "
                    f"range {low:.1f}-{high:.1f} years; got {row['Experience']}."
                )
    return validated


def load_trained_artifacts():
    """Load the persisted model and preprocessing artifacts from disk."""
    if not MODEL_PATH.exists() or not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(
            "Trained model or preprocessing artifacts not found. "
            "Run `python main.py` or `python src/train.py` first to train the model."
        )
    model = joblib.load(MODEL_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)
    return model, artifacts


def predict_salary(experience: float, role: str, model=None, artifacts=None) -> float:
    """
    Predict the salary for a single employee given their years of
    experience and role.

    Parameters
    ----------
    experience : float
        Years of professional experience.
    role : str
        One of: Junior, Mid, Senior, Lead.
    model, artifacts : optional
        Pre-loaded model/artifacts. If omitted, they are loaded from disk.

    Returns
    -------
    float - predicted annual salary in USD.
    """
    if model is None or artifacts is None:
        model, artifacts = load_trained_artifacts()

    df = _validate_prediction_records(
        pd.DataFrame([{"Experience": experience, "Role": role}]), artifacts
    )
    X = transform_features(df, artifacts)
    prediction = model.predict(X)[0]

    return float(prediction)


def predict_batch(records: list, model=None, artifacts=None) -> pd.DataFrame:
    """
    Predict salaries for a batch of employee records.

    Parameters
    ----------
    records : list of dict, each with keys 'Experience' and 'Role'.

    Returns
    -------
    pd.DataFrame with the original columns plus a 'PredictedSalary' column.
    """
    if model is None or artifacts is None:
        model, artifacts = load_trained_artifacts()

    df = _validate_prediction_records(pd.DataFrame(records), artifacts)
    X = transform_features(df, artifacts)
    df["PredictedSalary"] = model.predict(X)
    return df


def _parse_args():
    parser = argparse.ArgumentParser(description="Predict an employee's salary.")
    parser.add_argument("--experience", type=float, required=True,
                         help="Years of professional experience, e.g. 5")
    parser.add_argument("--role", type=str, required=True, choices=VALID_ROLES,
                         help="Job role/level")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    salary = predict_salary(args.experience, args.role)
    print(f"\nPredicted salary for a {args.role} with {args.experience} years of experience:")
    print(f"    ${salary:,.2f}\n")
