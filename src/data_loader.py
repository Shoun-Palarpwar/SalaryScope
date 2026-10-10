"""
data_loader.py
==============
Responsible for producing the raw dataset used by the pipeline.

If `data/salaries.csv` already exists, it is loaded as-is (so you can
swap in your own real-world data). Otherwise, a realistic synthetic
dataset is generated and saved to that path.

The synthetic dataset models:
    - Experience  : years of professional experience (0-25)
    - Role        : job level (Junior, Mid, Senior, Lead)
    - Salary      : annual salary in USD, driven by Experience + Role
                    with realistic noise, plus a handful of missing
                    values sprinkled into both Experience and Salary
                    to exercise the preprocessing step.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Base salary bands and per-year growth rate for each role.
# These make the synthetic data behave like a real salary curve:
# higher roles start higher AND grow faster per year of experience.
ROLE_PARAMS = {
    "Junior": {"base": 45000, "per_year": 2200, "exp_range": (0, 4)},
    "Mid":    {"base": 65000, "per_year": 3200, "exp_range": (2, 9)},
    "Senior": {"base": 95000, "per_year": 4200, "exp_range": (5, 15)},
    "Lead":   {"base": 130000, "per_year": 5000, "exp_range": (8, 25)},
}

RANDOM_SEED = 42


def generate_synthetic_dataset(n_rows: int = 1000, missing_frac: float = 0.05,
                                seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a realistic synthetic Experience/Role/Salary dataset.

    Parameters
    ----------
    n_rows : int
        Number of employee records to generate.
    missing_frac : float
        Fraction of rows to randomly null-out in Experience and Salary
        independently, to simulate real-world data quality issues.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns: Experience, Role, Salary
    """
    rng = np.random.default_rng(seed)
    roles = list(ROLE_PARAMS.keys())

    records = []
    for _ in range(n_rows):
        role = rng.choice(roles, p=[0.35, 0.30, 0.25, 0.10])
        low, high = ROLE_PARAMS[role]["exp_range"]
        experience = round(float(rng.uniform(low, high)), 1)

        base = ROLE_PARAMS[role]["base"]
        per_year = ROLE_PARAMS[role]["per_year"]
        noise = rng.normal(0, 4000)
        salary = round(base + per_year * experience + noise, 2)
        salary = max(salary, 25000)  # floor so noise never produces nonsense

        records.append({"Experience": experience, "Role": role, "Salary": salary})

    df = pd.DataFrame(records)

    # Inject missing values to simulate real-world messiness.
    n_missing_exp = int(n_rows * missing_frac)
    n_missing_sal = int(n_rows * missing_frac)

    exp_idx = rng.choice(df.index, size=n_missing_exp, replace=False)
    sal_idx = rng.choice(df.index, size=n_missing_sal, replace=False)

    df.loc[exp_idx, "Experience"] = np.nan
    df.loc[sal_idx, "Salary"] = np.nan

    return df


def load_dataset(csv_path: str = "data/salaries.csv", n_rows: int = 1000,
                  force_regenerate: bool = False) -> pd.DataFrame:
    """
    Load the dataset from `csv_path` if it exists; otherwise generate
    a synthetic dataset, save it to `csv_path`, and return it.

    Parameters
    ----------
    csv_path : str
        Location of the dataset CSV (relative to project root by default).
    n_rows : int
        Number of rows to generate if the file does not exist.
    force_regenerate : bool
        If True, always regenerate the synthetic dataset and overwrite
        the existing CSV.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(csv_path)

    if path.exists() and not force_regenerate:
        df = pd.read_csv(path)
        print(f"[data_loader] Loaded existing dataset from '{path}' ({len(df)} rows).")
        return df

    print(f"[data_loader] Generating synthetic dataset ({n_rows} rows)...")
    df = generate_synthetic_dataset(n_rows=n_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data_loader] Saved dataset to '{path}'.")

    return df


if __name__ == "__main__":
    # Allow running this module standalone to (re)generate the dataset.
    df = load_dataset(force_regenerate=True)
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
