"""Preprocessing utilities for salary model training and inference."""

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLUMNS = ["Experience", "Role"]
TARGET_COLUMN = "Salary"
ROLE_CATEGORIES = ["Junior", "Mid", "Senior", "Lead"]


@dataclass
class PreprocessArtifacts:
    """Fitted feature transformers persisted alongside the model."""

    experience_imputer: SimpleImputer
    role_encoder: OneHotEncoder
    encoded_columns: list[str]
    role_experience_ranges: dict[str, tuple[float, float]]


def build_preprocessor(training_df: pd.DataFrame) -> PreprocessArtifacts:
    """Fit feature preprocessing on training data only."""
    experience_imputer = SimpleImputer(strategy="median")
    experience_imputer.fit(training_df[["Experience"]])

    role_encoder = OneHotEncoder(
        categories=[ROLE_CATEGORIES],
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )
    role_encoder.fit(training_df[["Role"]])

    role_ranges = {}
    for role, group in training_df.groupby("Role"):
        observed = group["Experience"].dropna()
        role_ranges[role] = (float(observed.min()), float(observed.max()))

    return PreprocessArtifacts(
        experience_imputer=experience_imputer,
        role_encoder=role_encoder,
        encoded_columns=list(role_encoder.get_feature_names_out(["Role"])),
        role_experience_ranges=role_ranges,
    )


def transform_features(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> pd.DataFrame:
    """Transform records using already-fitted preprocessing artifacts."""
    experience = artifacts.experience_imputer.transform(df[["Experience"]]).ravel()
    encoded = artifacts.role_encoder.transform(df[["Role"]])

    result = pd.DataFrame({"Experience": experience}, index=df.index)
    encoded_df = pd.DataFrame(
        encoded,
        columns=artifacts.encoded_columns,
        index=df.index,
    )
    result = pd.concat([result, encoded_df], axis=1)

    # The generator uses a different experience-based growth rate per role.
    for column in artifacts.encoded_columns:
        result[f"Experience_x_{column}"] = result["Experience"] * result[column]

    return result


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, PreprocessArtifacts]:
    """Convenience helper for exploration; training uses train-only fitting."""
    labeled = df.dropna(subset=[TARGET_COLUMN]).copy()
    artifacts = build_preprocessor(labeled)
    X = transform_features(labeled, artifacts)
    y = labeled[TARGET_COLUMN].astype(float)
    return X, y, artifacts
