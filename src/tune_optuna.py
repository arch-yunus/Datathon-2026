import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMRegressor
import joblib
import optuna
import os


def build_pipeline(X, params):
    numeric_candidates = [c for c in X.columns if X[c].dtype.kind in 'biufc' and not c.startswith('emb_')]
    emb_features = [c for c in X.columns if c.startswith('emb_')]
    numeric_features = numeric_candidates + emb_features

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_candidates = ['department','university_tier','target_role','hobby','preferred_social_media_platform']
    categorical_features = [c for c in categorical_candidates if c in X.columns]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=params.get('learning_rate', 0.05),
        num_leaves=int(params.get('num_leaves', 31)),
        max_depth=int(params.get('max_depth', -1)),
        min_child_samples=int(params.get('min_child_samples', 20)),
        subsample=params.get('subsample', 1.0),
        colsample_bytree=params.get('colsample_bytree', 1.0),
        reg_alpha=params.get('reg_alpha', 0.0),
        reg_lambda=params.get('reg_lambda', 0.0),
        random_state=42
    )

    model = Pipeline(steps=[('preproc', preprocessor), ('lgbm', lgbm)])
    return model


def objective(trial, X, y, cv=3):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'max_depth': trial.suggest_int('max_depth', -1, 25),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
    }

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    scores = []

    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = build_pipeline(X, params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        oof[val_idx] = pred
        scores.append(mean_squared_error(y_val, pred))

    return np.mean(scores)


def run_tuning(train_path, model_out='model_optuna.pkl', n_trials=30, cv=3):
    train = pd.read_csv(train_path)
    if 'career_success_score' not in train.columns:
        raise ValueError('train.csv must contain career_success_score')

    y = train['career_success_score'].values
    X = train.drop(columns=['career_success_score', 'student_id'], errors='ignore')

    # use embeddings if present; otherwise fallback
    study = optuna.create_study(direction='minimize')
    func = lambda trial: objective(trial, X, y, cv=cv)
    study.optimize(func, n_trials=n_trials)

    print('Best params:', study.best_params)
    best_params = study.best_params

    # train on full data with best params
    model = build_pipeline(X, best_params)
    model.fit(X, y)
    joblib.dump(model, model_out)
    print('Saved tuned model to', model_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--out', default='model_optuna.pkl')
    args = parser.parse_args()
    run_tuning(args.train, model_out=args.out, n_trials=args.trials, cv=args.cv)
