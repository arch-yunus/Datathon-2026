import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import json
import os

def load_data(train_path):
    df = pd.read_csv(train_path)
    if 'career_success_score' not in df.columns:
        raise ValueError('train file must contain career_success_score')
    
    y = df['career_success_score'].values
    X = df.drop(columns=['career_success_score', 'student_id', 'mentor_feedback_text'], errors='ignore')
    
    # Simple imputation for numeric features only for Optuna speed
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna('missing')
        # Ordinal encode for LightGBM/XGBoost fast tuning
        X[c] = pd.Categorical(X[c]).codes
        
    return X, y

def tune_lightgbm(X, y, n_trials=200):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X):
            model = LGBMRegressor(**params)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            preds = model.predict(X.iloc[val_idx])
            scores.append(mean_squared_error(y[val_idx], preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_xgboost(X, y, n_trials=200):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X):
            model = XGBRegressor(**params)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            preds = model.predict(X.iloc[val_idx])
            scores.append(mean_squared_error(y[val_idx], preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_catboost(X, y, n_trials=200):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'depth': trial.suggest_int('depth', 3, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 20.0, log=True),
            'random_state': 42,
            'verbose': 0,
            'allow_writing_files': False
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X):
            model = CatBoostRegressor(**params)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            preds = model.predict(X.iloc[val_idx])
            scores.append(mean_squared_error(y[val_idx], preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_fe.csv')
    parser.add_argument('--out', default='best_params.json')
    parser.add_argument('--trials', type=int, default=200) # Default to 200 trials for Phase 3
    parser.add_argument('--models', nargs='+', default=['lightgbm', 'xgboost', 'catboost'])
    args = parser.parse_args()

    print("Loading data...")
    try:
        X, y = load_data(args.train)
    except FileNotFoundError:
        print(f"File {args.train} not found. Please run feature engineering first.")
        exit(1)

    # Load existing parameters if best_params.json exists
    best_params = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, 'r') as f:
                best_params = json.load(f)
            print(f"Loaded existing parameters from {args.out}")
        except Exception as e:
            print(f"Could not load {args.out}: {e}")

    if 'lightgbm' in args.models:
        print(f"Tuning LightGBM for {args.trials} trials...")
        best_params['lightgbm'] = tune_lightgbm(X, y, n_trials=args.trials)
    
    if 'xgboost' in args.models:
        print(f"Tuning XGBoost for {args.trials} trials...")
        best_params['xgboost'] = tune_xgboost(X, y, n_trials=args.trials)
    
    if 'catboost' in args.models:
        print(f"Tuning CatBoost for {args.trials} trials...")
        best_params['catboost'] = tune_catboost(X, y, n_trials=args.trials)
        
    with open(args.out, 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print(f"Finished tuning. Parameters saved to {args.out}")
