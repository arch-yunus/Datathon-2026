import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import joblib

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subs', nargs='*', default=['submission.csv','submission_catboost.csv','submission_xgb.csv','submission_nn.csv','submission_embeddings.csv'])
    parser.add_argument('--oofs', nargs='*', default=['oof_predictions.csv','oof_catboost.csv','oof_predictions_xgb.csv','oof_predictions_nn.csv','oof_embeddings.csv'])
    parser.add_argument('--out', default='final_submission_stacked.csv')
    args = parser.parse_args()

    # Load OOFs to train the meta-model
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
    
    print("Training RidgeCV Meta-Model...")
    meta_model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0), cv=5)
    meta_model.fit(X_meta_train, y_meta_train)
    
    oof_meta_preds = meta_model.predict(X_meta_train)
    meta_mse = mean_squared_error(y_meta_train, oof_meta_preds)
    print(f"Meta-Model OOF MSE: {meta_mse:.4f}")
    print("Meta-Model Coefficients:", meta_model.coef_)
    print("Meta-Model Intercept:", meta_model.intercept_)
    
    # Save meta-model
    joblib.dump(meta_model, 'meta_model.pkl')
    
    # Build meta-test set from submissions
    base_sub = available_subs[0][1][['student_id']].copy()
    X_meta_test = []
    for sname, sdf in available_subs:
        aligned_pred = sdf.set_index('student_id').reindex(base_sub['student_id']).reset_index()['career_success_score']
        X_meta_test.append(aligned_pred.values)
        
    X_meta_test = np.column_stack(X_meta_test)
    
    # Predict final
    final_preds = meta_model.predict(X_meta_test)
    
    out = base_sub.copy()
    out['career_success_score'] = final_preds
    out.to_csv(args.out, index=False)
    print('Saved stacked ensemble to', args.out)
