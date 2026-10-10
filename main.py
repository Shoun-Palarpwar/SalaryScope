"""
main.py
=======
Single entry point for the SalaryScope salary prediction pipeline.

Running this script will:
    1. Load (or generate) the dataset at data/salaries.csv
    2. Preprocess it (impute missing values, one-hot encode Role)
    3. Split into train (80%) / test (20%) sets
    4. Train a Linear Regression model
    5. Evaluate it (R^2, MAE, RMSE) and print a report
    6. Save the trained model + preprocessing artifacts to models/
    7. Run a few example predictions to demonstrate predict.py

Usage
-----
    python main.py
    python main.py --rows 2000          # generate a larger synthetic dataset
    python main.py --regenerate-data    # force-regenerate data/salaries.csv
"""

import argparse

from src.train import train_model
from src.predict import predict_salary, load_trained_artifacts
from src.data_loader import load_dataset

DATA_PATH = "data/salaries.csv"


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_pipeline(n_rows: int = 1000, regenerate_data: bool = False):
    print_header("SALARYSCOPE SALARY PREDICTION PIPELINE")

    if regenerate_data:
        load_dataset(csv_path=DATA_PATH, n_rows=n_rows, force_regenerate=True)

    print_header("STEP 1-5: TRAIN & EVALUATE MODEL")
    metrics = train_model(csv_path=DATA_PATH, n_rows=n_rows)

    print_header("STEP 6: DEMO PREDICTIONS")
    model, artifacts = load_trained_artifacts()

    demo_cases = [
        (1.0, "Junior"),
        (4.0, "Mid"),
        (8.0, "Senior"),
        (15.0, "Lead"),
    ]
    for experience, role in demo_cases:
        salary = predict_salary(experience, role, model=model, artifacts=artifacts)
        print(f"    {role:7s} | {experience:5.1f} yrs experience -> Predicted salary: ${salary:,.2f}")

    print_header("PIPELINE COMPLETE")
    print(f"Final Test R^2 Score : {metrics['r2_score']:.4f}")
    print(f"Final Test MAE       : ${metrics['mae']:,.2f}")
    print(f"Final Test RMSE      : ${metrics['rmse']:,.2f}")
    print("\nModel and artifacts saved under ./models/")
    print("Run `python src/predict.py --experience 6 --role Senior` for a one-off prediction.\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the end-to-end salary prediction pipeline.")
    parser.add_argument("--rows", type=int, default=1000,
                         help="Number of rows to generate if the dataset doesn't exist yet (default: 1000)")
    parser.add_argument("--regenerate-data", action="store_true",
                         help="Force regeneration of data/salaries.csv even if it already exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(n_rows=args.rows, regenerate_data=args.regenerate_data)
