# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import yaml
import os
import joblib

# обучение модели
def fit_model():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    data = pd.read_csv('data/initial_data.csv', index_col=params['index_col'])
    target_col = params['target_col']

    cat_features = data.select_dtypes(include='object')
    cat_cols = cat_features.columns.tolist()

    num_features = data.select_dtypes(['float'])
    num_cols = num_features.columns.tolist()

    preprocessor = ColumnTransformer(
        [
            ('cat', OneHotEncoder(drop=params['one_hot_drop']), cat_cols),
            ('num', StandardScaler(), num_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(
        C=params['logreg_C'],
        penalty=params['logreg_penalty'],
    )

    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model),
        ]
    )

    pipeline.fit(data, data[target_col])

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/fitted_model.pkl')

if __name__ == '__main__':
    fit_model()
