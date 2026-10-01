# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

# оценка качества модели
def evaluate_model():
    # прочитайте файл с гиперпараметрами params.yaml
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)    

    # загрузите результат прошлого шага: fitted_model.pkl
    with open('models/fitted_model.pkl', 'rb') as fd:
        model = joblib.load(fd)

    data = pd.read_csv('data/initial_data.csv')
    target_col = params['target_col']    

    # реализуйте основную логику шага с использованием прочтённых гиперпараметров
    cv_strategy = StratifiedKFold(n_splits=params['n_splits'])
    cv_res = cross_validate(
        model,
        data,
        data[target_col],
        cv=cv_strategy,
        n_jobs=-1,
        scoring=['f1', 'roc_auc'],
    )

    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)

    # сохраните результата кросс-валидации в cv_res.json
    os.makedirs('cv_results', exist_ok=True)
    with open('cv_results/cv_res.json', 'w') as f:
        json.dump(cv_res, f)

if __name__ == '__main__':
    evaluate_model()
