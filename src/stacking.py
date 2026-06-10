"""
Stacking ensemble: trains multiple base models and uses a meta-learner for final predictions.
Improves performance by capturing diverse patterns.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import os


def create_stacked_predictions(sub_files, oof_files, test_predictions=True):
    """
    Load base model submissions and OOFs, then train meta-learner.
    
    sub_files: list of submission CSV paths
    oof_files: list of OOF predictions CSV paths
    """
    
    # Load OOFs and target
    oofs = []
    target = None
    for oof_file in oof_files:
        try:
            df = pd.read_csv(oof_file)
            if 'career_success_score' in df.columns:
                target = df['career_success_score'].values
            if 'oof_pred' in df.columns:
                oofs.append(df['oof_pred'].values)
        except Exception as e:
            print(f'Warning: could not load {oof_file}: {e}')
    
    if len(oofs) < 2:
        print('Not enough OOF files for stacking. Returning simple ensemble.')
        return None
    
    # Stack OOF predictions (n_samples, n_models)
    X_meta = np.column_stack(oofs)
    
    if target is None:
        print('Could not find target values')
        return None
    
    # Train meta-learner
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta, target)
    
    # Load test submissions and stack
    test_preds = []
    student_ids = None
    for sub_file in sub_files:
        try:
            df = pd.read_csv(sub_file)
            if student_ids is None and 'student_id' in df.columns:
                student_ids = df['student_id'].values
            if 'career_success_score' in df.columns:
                test_preds.append(df['career_success_score'].values)
        except Exception as e:
            print(f'Warning: could not load {sub_file}: {e}')
    
    if len(test_preds) < 2:
        print('Not enough test submissions for stacking')
        return None
    
    X_test_meta = np.column_stack(test_preds)
    stacked_preds = meta_model.predict(X_test_meta)
    
    # Save stacking model and results
    joblib.dump(meta_model, 'stacking_meta_model.pkl')
    
    if student_ids is not None:
        result = pd.DataFrame({
            'student_id': student_ids,
            'career_success_score': stacked_preds
        })
    else:
        result = pd.DataFrame({
            'career_success_score': stacked_preds
        })
    
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subs', nargs='*', default=['submission.csv', 'submission_embeddings.csv', 'submission_fe.csv', 'submission_catboost.csv'])
    parser.add_argument('--oofs', nargs='*', default=['oof_predictions.csv', 'oof_embeddings.csv', 'oof_embeddings.csv', 'oof_catboost.csv'])
    parser.add_argument('--out', default='final_stacking.csv')
    args = parser.parse_args()
    
    result = create_stacked_predictions(args.subs, args.oofs)
    if result is not None:
        result.to_csv(args.out, index=False)
        print(f'Stacking ensemble saved to {args.out}')
    else:
        print('Stacking failed; falling back to simple ensemble')
