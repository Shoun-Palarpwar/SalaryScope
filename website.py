"""Local SalaryScope website. Run: python3 website.py."""
import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from src.predict import load_trained_artifacts, predict_salary

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / 'web'), static_url_path='/assets')
model, artifacts = load_trained_artifacts()


@app.get('/')
def index():
    return send_from_directory(ROOT / 'web', 'index.html')


@app.get('/api/project')
def project():
    df = pd.read_csv(ROOT / 'data/salaries.csv')
    return jsonify(summary=json.loads((ROOT / 'reports/analysis_summary.json').read_text()),
                   metrics=json.loads((ROOT / 'models/metrics.json').read_text()),
                   ranges=artifacts.role_experience_ranges,
                   records=json.loads(df.to_json(orient='records')),
                   coefficients=dict(zip(model.feature_names_in_, model.coef_)),
                   intercept=float(model.intercept_))


@app.post('/api/predict')
def predict():
    try:
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict) or not isinstance(body.get('role'), str):
            raise ValueError('Provide a role and numeric experience.')
        experience = float(body.get('experience', 'nan'))
        role = body.get('role')
        salary = predict_salary(experience, role, model=model, artifacts=artifacts)
        return jsonify(salary=salary, model='Linear Regression with role interactions')
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc)), 400


@app.get('/figures/<path:name>')
def figures(name):
    return send_from_directory(ROOT / 'reports/figures', name)


@app.get('/downloads/<name>')
def download(name):
    allowed = {'salaries.csv': ROOT / 'data', 'analysis_report.md': ROOT / 'reports',
               'model_comparison_folds.csv': ROOT / 'reports'}
    if name not in allowed:
        return jsonify(error='File not found'), 404
    return send_from_directory(allowed[name], name, as_attachment=True)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=False)
