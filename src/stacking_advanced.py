import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import joblib
import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subs', nargs='*', default=['submission.csv','submission_catboost.csv','submission_xgb.csv','submission_nn.csv','submission_embeddings.csv'])
    parser.add_argument('--oofs', nargs='*', default=['oof_predictions.csv','oof_catboost.csv','oof_predictions_xgb.csv','oof_predictions_nn.csv','oof_embeddings.csv'])
    parser.add_argument('--out', default='final_submission_stacked.csv')
    args = parser.parse_args()

    # Load OOFs to train the meta-models
    available_subs = []
    available_oofs = []

    for i, s in enumerate(args.subs):
        try:
            df = pd.read_csv(s)
            available_subs.append((s, df))
            
            oof_file = args.oofs[i] if i < len(args.oofs) else None
            if oof_file:
                try:
                    oof_df = pd.read_csv(oof_file)
                    available_oofs.append((oof_file, oof_df))
                except Exception:
                    pass
        except Exception:
            pass

    if len(available_oofs) < 2:
        raise SystemExit('Not enough OOF files found to train a meta-model. Need at least 2.')

    # Load true target from train file
    train_df = pd.read_csv('train.csv')
    if 'career_success_score' not in train_df.columns:
        raise SystemExit('career_success_score target column not found in train.csv')
        
    true_target = train_df.set_index('student_id')['career_success_score']
    
    # Build meta-training set
    X_meta_train = []
    for oof_name, oof_df in available_oofs:
        aligned_pred = oof_df.set_index('student_id').reindex(true_target.index)['oof_pred']
        X_meta_train.append(aligned_pred.values)
        
    X_meta_train = np.column_stack(X_meta_train)
    y_meta_train = true_target.values
    
    print(f"Meta-Training data shape: {X_meta_train.shape}")
    
    # Define meta-models
    ridge = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0), cv=5)
    lasso = LassoCV(cv=5, random_state=42)
    lgb_meta = LGBMRegressor(max_depth=3, n_estimators=50, learning_rate=0.05, random_state=42, n_jobs=-1)
    
    # Out-of-fold predictions for the LightGBM meta-model to avoid overfitting at meta level
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_oof_preds = np.zeros(len(y_meta_train))
    for tr_idx, val_idx in kf.split(X_meta_train):
        lgb_meta.fit(X_meta_train[tr_idx], y_meta_train[tr_idx])
        lgb_oof_preds[val_idx] = lgb_meta.predict(X_meta_train[val_idx])
        
    # Fit full meta-models
    ridge.fit(X_meta_train, y_meta_train)
    lasso.fit(X_meta_train, y_meta_train)
    lgb_meta.fit(X_meta_train, y_meta_train)
    
    # Evaluate OOF MSE of each meta-model
    ridge_oof = ridge.predict(X_meta_train)
    lasso_oof = lasso.predict(X_meta_train)
    
    ridge_mse = mean_squared_error(y_meta_train, ridge_oof)
    lasso_mse = mean_squared_error(y_meta_train, lasso_oof)
    lgb_mse = mean_squared_error(y_meta_train, lgb_oof_preds)
    
    print(f"RidgeCV Meta-Model OOF MSE: {ridge_mse:.4f}")
    print(f"LassoCV Meta-Model OOF MSE: {lasso_mse:.4f}")
    print(f"LightGBM Meta-Model OOF MSE: {lgb_mse:.4f}")
    
    # Blend predictions (weighted average: 0.4 Ridge, 0.4 Lasso, 0.2 LGBM)
    blend_oof = 0.4 * ridge_oof + 0.4 * lasso_oof + 0.2 * lgb_oof_preds
    blend_mse = mean_squared_error(y_meta_train, blend_oof)
    print(f"Blended Meta-Ensemble OOF MSE: {blend_mse:.4f}")
    
    # Save meta-models
    joblib.dump(ridge, 'meta_model_ridge.pkl')
    joblib.dump(lasso, 'meta_model_lasso.pkl')
    joblib.dump(lgb_meta, 'meta_model_lgb.pkl')
    
    # Build meta-test set from submissions
    base_sub = available_subs[0][1][['student_id']].copy()
    X_meta_test = []
    for sname, sdf in available_subs:
        aligned_pred = sdf.set_index('student_id').reindex(base_sub['student_id']).reset_index()['career_success_score']
        X_meta_test.append(aligned_pred.values)
        
    X_meta_test = np.column_stack(X_meta_test)
    
    # Predict final
    ridge_test_preds = ridge.predict(X_meta_test)
    lasso_test_preds = lasso.predict(X_meta_test)
    lgb_test_preds = lgb_meta.predict(X_meta_test)
    
    final_preds = 0.4 * ridge_test_preds + 0.4 * lasso_test_preds + 0.2 * lgb_test_preds
    
    out = base_sub.copy()
    out['career_success_score'] = final_preds
    out.to_csv(args.out, index=False)
    print('Saved stacked ensemble to', args.out)
