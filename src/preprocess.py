"""
preprocess.py
=============
Cleans the raw dataset and converts it into a model-ready feature
matrix (X) and target vector (y).

Steps:
    1. Impute missing `Experience` values with the column median.
    2. Impute missing `Salary` values (target) with the column median
       — done BEFORE the train/test split is created by the caller,
       since rows with a missing target still carry a usable
       Experience/Role signal we don't want to discard entirely from
       the dataset. (Median imputation on the target is a simplifying
       choice for this project; in a stricter production setting you
       may instead choose to drop rows with missing target values.)
    3. One-hot encode the categorical `Role` column.

The module exposes both a convenience `preprocess_data()` function for
one-shot use, and a reusable `build_preprocessor()` that returns a
fitted encoder + imputer bundle so `predict.py` can transform brand
new, unseen records the exact same way they were seen during training.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLUMNS = ["Experience", "Role"]
TARGET_COLUMN = "Salary"
ROLE_CATEGORIES = ["Junior", "Mid", "Senior", "Lead"]


@dataclass
class PreprocessArtifacts:
    """Bundle of fitted transformers + the resulting encoded column names."""
    experience_imputer: SimpleImputer
    salary_imputer: SimpleImputer
    role_encoder: OneHotEncoder
    encoded_columns: list


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Report on missing values before any imputation happens."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("[preprocess] Missing values found:")
        for col, count in missing.items():
            print(f"    - {col}: {count} missing ({count / len(df):.1%})")
    else:
        print("[preprocess] No missing values found.")
    return df


def build_preprocessor(df: pd.DataFrame) -> PreprocessArtifacts:
    """
    Fit imputers and encoder on the given (training) dataframe and
    return them bundled up for reuse on new data.
    """
    experience_imputer = SimpleImputer(strategy="median")
    experience_imputer.fit(df[["Experience"]])

    salary_imputer = SimpleImputer(strategy="median")
    salary_imputer.fit(df[["Salary"]])

    role_encoder = OneHotEncoder(
        categories=[ROLE_CATEGORIES],
        handle_unknown="ignore",
        sparse_output=False,
    )
    role_encoder.fit(df[["Role"]])
    encoded_columns = list(role_encoder.get_feature_names_out(["Role"]))

    return PreprocessArtifacts(
        experience_imputer=experience_imputer,
        salary_imputer=salary_imputer,
        role_encoder=role_encoder,
        encoded_columns=encoded_columns,
    )


def transform_features(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> pd.DataFrame:
    """
    Apply already-fitted imputer + encoder to a dataframe and return
    a clean numeric feature matrix (Experience + one-hot Role columns).
    Does NOT touch the Salary/target column.
    """
    df = df.copy()

    df["Experience"] = artifacts.experience_imputer.transform(df[["Experience"]])

    encoded = artifacts.role_encoder.transform(df[["Role"]])
    encoded_df = pd.DataFrame(encoded, columns=artifacts.encoded_columns, index=df.index)

    X = pd.concat([df[["Experience"]], encoded_df], axis=1)
    return X


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, PreprocessArtifacts]:
    """
    End-to-end preprocessing on the full raw dataset.

    Returns
    -------
    X : pd.DataFrame          - numeric feature matrix (Experience + one-hot Role)
    y : pd.Series             - target Salary values (missing values imputed)
    artifacts : PreprocessArtifacts - fitted transformers, for reuse in predict.py
    """
    df = clean_missing_values(df)

    # Fit transformers on the full dataset (features + target imputers).
    artifacts = build_preprocessor(df)

    X = transform_features(df, artifacts)

    y = df[[TARGET_COLUMN]].copy()
    y[TARGET_COLUMN] = artifacts.salary_imputer.transform(y[[TARGET_COLUMN]])
    y = y[TARGET_COLUMN]

    print(f"[preprocess] Final feature matrix shape: {X.shape}")
    print(f"[preprocess] Feature columns: {list(X.columns)}")

    return X, y, artifacts


if __name__ == "__main__":
    # Quick standalone smoke test using the data_loader module.
    from data_loader import load_dataset

    raw_df = load_dataset()
    X, y, artifacts = preprocess_data(raw_df)
    print(X.head())
    print(y.head())
