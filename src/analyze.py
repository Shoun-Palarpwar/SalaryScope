"""Generate academic EDA, diagnostics, and cross-validated model comparisons."""

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep Matplotlib's cache out of the source tree and support headless execution.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "salaryscope-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset
from src.preprocess import ROLE_CATEGORIES, build_preprocessor, transform_features

DATA_PATH = PROJECT_ROOT / "data" / "salaries.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RANDOM_STATE = 42
COLORS = {
    "Junior": "#4C78A8",
    "Mid": "#59A14F",
    "Senior": "#F28E2B",
    "Lead": "#E15759",
}


def get_candidate_models() -> dict:
    """Return the models included in the academic comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }


def _save_figure(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_eda_figures(df: pd.DataFrame) -> None:
    """Create distribution and feature-relationship figures."""
    labeled = df.dropna(subset=["Salary"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(labeled["Salary"], bins=30, color="#4C78A8", edgecolor="white")
    axes[0].set(title="Salary Distribution", xlabel="Annual salary (USD)", ylabel="Employees")
    axes[0].ticklabel_format(axis="x", style="plain")
    axes[1].hist(df["Experience"].dropna(), bins=25, color="#59A14F", edgecolor="white")
    axes[1].set(title="Experience Distribution", xlabel="Years of experience", ylabel="Employees")
    _save_figure(fig, "distributions.png")

    role_data = [labeled.loc[labeled["Role"].eq(role), "Salary"] for role in ROLE_CATEGORIES]
    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(role_data, tick_labels=ROLE_CATEGORIES, patch_artist=True)
    for patch, role in zip(box["boxes"], ROLE_CATEGORIES):
        patch.set_facecolor(COLORS[role])
        patch.set_alpha(0.75)
    ax.set(title="Salary by Role", xlabel="Role", ylabel="Annual salary (USD)")
    ax.ticklabel_format(axis="y", style="plain")
    _save_figure(fig, "salary_by_role.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for role in ROLE_CATEGORIES:
        group = labeled[labeled["Role"].eq(role)]
        ax.scatter(
            group["Experience"], group["Salary"], s=22, alpha=0.55,
            label=role, color=COLORS[role],
        )
    ax.set(
        title="Salary and Experience by Role",
        xlabel="Years of experience",
        ylabel="Annual salary (USD)",
    )
    ax.legend(title="Role")
    ax.ticklabel_format(axis="y", style="plain")
    _save_figure(fig, "salary_vs_experience.png")


def compare_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate candidates with leakage-free, role-stratified cross-validation."""
    labeled = df.dropna(subset=["Salary"]).reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = get_candidate_models()
    fold_rows = []
    out_of_fold = {name: np.full(len(labeled), np.nan) for name in models}

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(labeled, labeled["Role"]), start=1
    ):
        train_df = labeled.iloc[train_indices]
        test_df = labeled.iloc[test_indices]
        artifacts = build_preprocessor(train_df)
        X_train = transform_features(train_df, artifacts)
        X_test = transform_features(test_df, artifacts)
        y_train = train_df["Salary"]
        y_test = test_df["Salary"]

        for name, estimator in models.items():
            fitted = clone(estimator).fit(X_train, y_train)
            predictions = fitted.predict(X_test)
            out_of_fold[name][test_indices] = predictions
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "r2": r2_score(y_test, predictions),
                    "mae": mean_absolute_error(y_test, predictions),
                    "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                }
            )

    return pd.DataFrame(fold_rows), out_of_fold


def create_evaluation_figures(
    labeled: pd.DataFrame,
    fold_results: pd.DataFrame,
    out_of_fold: dict[str, np.ndarray],
) -> str:
    """Create model comparison and best-model diagnostic figures."""
    summary = fold_results.groupby("model")["mae"].agg(["mean", "std"]).sort_values("mean")
    best_model = summary.index[0]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(
        summary.index,
        summary["mean"],
        xerr=summary["std"],
        color=["#4C78A8", "#59A14F", "#F28E2B", "#E15759"],
        alpha=0.85,
        capsize=4,
    )
    ax.invert_yaxis()
    ax.set(title="Five-Fold Model Comparison", xlabel="Mean absolute error (USD)")
    _save_figure(fig, "model_comparison.png")

    actual = labeled["Salary"].to_numpy()
    predicted = out_of_fold[best_model]
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(actual, predicted, alpha=0.5, s=24, color="#4C78A8")
    ax.plot([low, high], [low, high], linestyle="--", color="#E15759", label="Perfect prediction")
    ax.set(
        title=f"Actual vs Predicted — {best_model}",
        xlabel="Actual salary (USD)",
        ylabel="Predicted salary (USD)",
    )
    ax.legend()
    ax.ticklabel_format(axis="both", style="plain")
    _save_figure(fig, "actual_vs_predicted.png")

    residuals = actual - predicted
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted, residuals, alpha=0.5, s=24, color="#59A14F")
    ax.axhline(0, linestyle="--", color="#E15759")
    ax.set(
        title=f"Residuals — {best_model}",
        xlabel="Predicted salary (USD)",
        ylabel="Residual: actual − predicted (USD)",
    )
    ax.ticklabel_format(axis="both", style="plain")
    _save_figure(fig, "residuals.png")
    return best_model


def _build_summary(
    df: pd.DataFrame,
    fold_results: pd.DataFrame,
    out_of_fold: dict[str, np.ndarray],
    best_model: str,
) -> dict:
    labeled = df.dropna(subset=["Salary"]).reset_index(drop=True)
    comparison = {}
    for name, group in fold_results.groupby("model"):
        comparison[name] = {
            metric: {
                "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=1)),
            }
            for metric in ("r2", "mae", "rmse")
        }

    role_mae = {}
    best_predictions = out_of_fold[best_model]
    for role in ROLE_CATEGORIES:
        mask = labeled["Role"].eq(role).to_numpy()
        role_mae[role] = float(
            mean_absolute_error(labeled.loc[mask, "Salary"], best_predictions[mask])
        )

    return {
        "dataset": {
            "rows": int(len(df)),
            "labeled_rows": int(len(labeled)),
            "missing_experience": int(df["Experience"].isna().sum()),
            "missing_salary": int(df["Salary"].isna().sum()),
        },
        "best_model_by_cv_mae": best_model,
        "model_comparison": comparison,
        "best_model_mae_by_role": role_mae,
    }


def _write_markdown_report(summary: dict) -> None:
    models = sorted(
        summary["model_comparison"].items(), key=lambda item: item[1]["mae"]["mean"]
    )
    rows = "\n".join(
        f"| {name} | {values['r2']['mean']:.4f} ± {values['r2']['std']:.4f} "
        f"| ${values['mae']['mean']:,.2f} ± ${values['mae']['std']:,.2f} "
        f"| ${values['rmse']['mean']:,.2f} ± ${values['rmse']['std']:,.2f} |"
        for name, values in models
    )
    role_rows = "\n".join(
        f"| {role} | ${mae:,.2f} |"
        for role, mae in summary["best_model_mae_by_role"].items()
    )
    text = f"""# SalaryScope Analysis Report

## Dataset

- Total rows: {summary['dataset']['rows']}
- Labeled rows used for evaluation: {summary['dataset']['labeled_rows']}
- Missing experience values: {summary['dataset']['missing_experience']}
- Missing salary values excluded: {summary['dataset']['missing_salary']}

## Five-fold model comparison

All preprocessing was fitted separately inside each training fold.

| Model | R², mean ± SD | MAE, mean ± SD | RMSE, mean ± SD |
|---|---:|---:|---:|
{rows}

Best model by cross-validated MAE: **{summary['best_model_by_cv_mae']}**.

## Best-model MAE by role

| Role | MAE |
|---|---:|
{role_rows}

## Interpretation

The role–experience scatterplot shows why interaction terms are useful: salary
growth differs by role. The actual-versus-predicted chart shows overall fit,
while the residual chart should be inspected for changing error spread or
systematic patterns. Role-level MAE highlights categories that remain harder
to predict, particularly when they have fewer observations.

These results describe a synthetic academic dataset and must not be treated as
evidence of real-world salary accuracy or used for compensation decisions.
"""
    (REPORT_DIR / "analysis_report.md").write_text(text, encoding="utf-8")


def run_analysis(csv_path: str | Path = DATA_PATH) -> dict:
    """Run the complete analysis and save tabular and visual outputs."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path=str(csv_path))
    labeled = df.dropna(subset=["Salary"]).reset_index(drop=True)

    create_eda_figures(df)
    fold_results, out_of_fold = compare_models(df)
    best_model = create_evaluation_figures(labeled, fold_results, out_of_fold)
    summary = _build_summary(df, fold_results, out_of_fold, best_model)

    fold_results.to_csv(REPORT_DIR / "model_comparison_folds.csv", index=False)
    with (REPORT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _write_markdown_report(summary)

    print(f"[analysis] Best model by CV MAE: {best_model}")
    print(f"[analysis] Reports and figures saved under '{REPORT_DIR}'.")
    return summary


if __name__ == "__main__":
    run_analysis()
