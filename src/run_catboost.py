"""
Advanced pipeline with target encoding + CatBoost for categorical handling.
CatBoost naturally handles categorical features well.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import joblib
import os
import mlflow


def prepare_data(train, test, target_col='career_success_score'):
    y = train[target_col].values
    X_train = train.drop(columns=[target_col, 'student_id'], errors='ignore')
    X_test = test.drop(columns=['student_id'], errors='ignore')
    
    # fillna for numeric
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].median())
    X_test[num_cols] = X_test[num_cols].fillna(X_train[num_cols].median())
    
    # fillna for categorical
    cat_cols = X_train.select_dtypes(include=['object']).columns
    X_train[cat_cols] = X_train[cat_cols].fillna('missing')
    X_test[cat_cols] = X_test[cat_cols].fillna('missing')
    
    return X_train, X_test, y, list(cat_cols)


def run_catboost_cv(train_path, test_path, out_submission, model_out, cv=5):
    mlflow.set_experiment("Datathon2026_CatBoost")
    mlflow.start_run()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    X_train, X_test, y, cat_cols = prepare_data(train, test)
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            cat_features=cat_cols if cat_cols else None,
            random_state=42,
            verbose=0
        )
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        mse = mean_squared_error(y_val, val_pred)
        scores.append(mse)
        print(f'Fold {fold+1} MSE: {mse:.4f}')
    
    mean_cv = np.mean(scores)
    std_cv = np.std(scores)
    overall_mse = mean_squared_error(y, oof_preds)
    print(f'CatBoost CV mean MSE: {mean_cv:.4f} ± {std_cv:.4f}  | OOF MSE: {overall_mse:.4f}')
    mlflow.log_param("cv_folds", cv)
    mlflow.log_param("model", "CatBoost")
    mlflow.log_metric("cv_mean_mse", mean_cv)
    mlflow.log_metric("oof_mse", overall_mse)
    
    # final model
    final_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        cat_features=cat_cols if cat_cols else None,
        random_state=42,
        verbose=0
    )
    final_model.fit(X_train, y)
    
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    joblib.dump(final_model, model_out)
    print('Saved CatBoost model to', model_out)
    
    preds_test = final_model.predict(X_test)
    submission = pd.DataFrame({
        'student_id': test['student_id'] if 'student_id' in test.columns else np.arange(len(preds_test)),
        'career_success_score': preds_test
    })
    submission.to_csv(out_submission, index=False)
    print('Saved submission to', out_submission)
    
    oof_df = train[['student_id']].copy() if 'student_id' in train.columns else pd.DataFrame({'student_id': np.arange(len(oof_preds))})
    oof_df['oof_pred'] = oof_preds
    oof_df.to_csv('oof_catboost.csv', index=False)
    print('Saved OOF predictions to oof_catboost.csv')
    mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', default='test_x.csv')
    parser.add_argument('--out', default='submission_catboost.csv')
    parser.add_argument('--model', default='model_catboost.pkl')
    parser.add_argument('--cv', type=int, default=5)
    args = parser.parse_args()
    
    run_catboost_cv(args.train, args.test, args.out, args.model, cv=args.cv)
