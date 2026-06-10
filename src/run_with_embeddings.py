"""
Run CV pipeline with sentence-transformers embeddings for `mentor_feedback_text`.
Produces `submission_embeddings.csv`, `model_embeddings.pkl` and `oof_embeddings.csv`.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import joblib
import os
from src.embed_features import add_embeddings_to_df


def build_pipeline(X):
    # numeric features are auto-detected
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

    model = Pipeline(steps=[
        ('preproc', preprocessor),
        ('lgbm', LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42))
    ])
    return model


def run(train_path, test_path, out_submission, model_out, cv=5, model_name='all-MiniLM-L6-v2'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # add embeddings
    print('Computing embeddings for train...')
    train_emb = add_embeddings_to_df(train, text_col='mentor_feedback_text', model_name=model_name)
    print('Computing embeddings for test...')
    test_emb = add_embeddings_to_df(test, text_col='mentor_feedback_text', model_name=model_name)

    y = train_emb['career_success_score'].values
    X = train_emb.drop(columns=['career_success_score', 'student_id'], errors='ignore')
    X_test = test_emb.drop(columns=['student_id'], errors='ignore')

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = build_pipeline(pd.concat([X_tr, X_val, X_test], axis=0))
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        mse = mean_squared_error(y_val, val_pred)
        scores.append(mse)
        print(f'Fold {fold+1} MSE: {mse:.4f}')

    mean_cv = np.mean(scores)
    std_cv = np.std(scores)
    overall_mse = mean_squared_error(y, oof_preds)
    print(f'Embedding CV mean MSE: {mean_cv:.4f} ± {std_cv:.4f}  | OOF MSE: {overall_mse:.4f}')

    # final model
    final_model = build_pipeline(pd.concat([X, X_test], axis=0))
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    joblib.dump(final_model, model_out)
    print('Saved embedding model to', model_out)

    preds_test = final_model.predict(X_test)
    submission = pd.DataFrame({
        'student_id': test['student_id'] if 'student_id' in test.columns else np.arange(len(preds_test)),
        'career_success_score': preds_test
    })
    submission.to_csv(out_submission, index=False)
    print('Saved submission to', out_submission)

    oof_df = train[['student_id']].copy() if 'student_id' in train.columns else pd.DataFrame({'student_id': np.arange(len(oof_preds))})
    oof_df['oof_pred'] = oof_preds
    oof_df.to_csv('oof_embeddings.csv', index=False)
    print('Saved OOF predictions to oof_embeddings.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', default='test_x.csv')
    parser.add_argument('--out', default='submission_embeddings.csv')
    parser.add_argument('--model', default='model_embeddings.pkl')
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--model-name', default='all-MiniLM-L6-v2')
    args = parser.parse_args()
    run(args.train, args.test, args.out, args.model, cv=args.cv, model_name=args.model_name)
