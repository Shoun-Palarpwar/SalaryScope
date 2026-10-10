import math
import unittest

import pandas as pd

from src.data_loader import generate_synthetic_dataset, validate_dataset
from src.analyze import get_candidate_models
from src.predict import _validate_prediction_records, predict_batch, predict_salary
from src.preprocess import build_preprocessor, preprocess_data, transform_features


class PipelineTests(unittest.TestCase):
    def test_analysis_compares_expected_models(self):
        self.assertEqual(
            set(get_candidate_models()),
            {"Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"},
        )

    def test_synthetic_dataset_is_reproducible_and_valid(self):
        first = generate_synthetic_dataset(n_rows=100, seed=7)
        second = generate_synthetic_dataset(n_rows=100, seed=7)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 100)
        self.assertTrue(
            set(first["Role"]).issubset({"Junior", "Mid", "Senior", "Lead"})
        )
        validate_dataset(first)

    def test_missing_targets_are_excluded_from_preprocessing(self):
        df = generate_synthetic_dataset(n_rows=100, missing_frac=0.1)
        X, y, _ = preprocess_data(df)
        self.assertEqual(len(X), 90)
        self.assertEqual(len(y), 90)
        self.assertFalse(y.isna().any())

    def test_preprocessor_uses_training_median_and_creates_interactions(self):
        training = pd.DataFrame(
            {
                "Experience": [1.0, 3.0, None],
                "Role": ["Junior", "Mid", "Senior"],
                "Salary": [40_000, 60_000, 90_000],
            }
        )
        artifacts = build_preprocessor(training)
        transformed = transform_features(
            pd.DataFrame({"Experience": [None], "Role": ["Mid"]}), artifacts
        )
        self.assertEqual(transformed.loc[0, "Experience"], 2.0)
        self.assertEqual(transformed.loc[0, "Role_Mid"], 1.0)
        self.assertEqual(transformed.loc[0, "Experience_x_Role_Mid"], 2.0)
        self.assertNotIn("Role_Junior", transformed.columns)

    def test_prediction_validation_rejects_bad_records(self):
        cases = [
            ({"Experience": -1, "Role": "Junior"}, "negative"),
            ({"Experience": math.inf, "Role": "Junior"}, "finite"),
            ({"Experience": 3, "Role": "Manager"}, "Role must be"),
            ({"Experience": 3}, "missing fields"),
        ]
        for record, message in cases:
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_prediction_records(pd.DataFrame([record]))

    def test_prediction_validation_rejects_empty_batch(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            _validate_prediction_records(pd.DataFrame())

    def test_single_and_batch_predictions_are_finite(self):
        single = predict_salary(6, "Senior")
        batch = predict_batch(
            [
                {"Experience": 1, "Role": "Junior"},
                {"Experience": 10, "Role": "Lead"},
            ]
        )
        self.assertTrue(math.isfinite(single))
        self.assertTrue(batch["PredictedSalary"].map(math.isfinite).all())

    def test_prediction_rejects_role_specific_extrapolation(self):
        with self.assertRaisesRegex(ValueError, "training range"):
            predict_salary(100, "Junior")


if __name__ == "__main__":
    unittest.main()
