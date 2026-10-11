"""Website contract tests, including the real saved prediction model."""
import unittest
from website import app
from src.predict import predict_salary


class WebsiteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_prediction_matches_saved_model(self):
        result = self.client.post('/api/predict', json={'role': 'Senior', 'experience': 6})
        self.assertEqual(result.status_code, 200)
        self.assertAlmostEqual(result.json['salary'], predict_salary(6, 'Senior'))

    def test_invalid_requests_return_client_errors(self):
        for body in [[], ['invalid'], {'role': []}, {'role': 'Manager', 'experience': 6},
                     {'role': 'Junior', 'experience': -1}, {'role': 'Lead', 'experience': 100}]:
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/api/predict', json=body).status_code, 400)

    def test_project_data_and_download_allowlist(self):
        result = self.client.get('/api/project')
        self.assertEqual(len(result.json['records']), 1000)
        self.assertEqual(self.client.get('/downloads/model_comparison_folds.csv').status_code, 200)
        self.assertEqual(self.client.get('/downloads/website.py').status_code, 404)


if __name__ == '__main__':
    unittest.main()
