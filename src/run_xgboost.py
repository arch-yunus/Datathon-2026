"""
End-to-end training + CV + submission script with XGBoost and MLflow.
Usage:
  python src/run_xgboost.py --train train.csv --test test_x.csv --out submission_xgb.csv --model model_xgb.pkl --cv 5
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
import os
import mlflow

def build_pipeline_for_columns(X, text_col='mentor_feedback_text'):
    numeric_candidates = [
        'age','cgpa','english_exam_score','attendance_rate','failed_courses_count',
        'coding_score','problem_solving_score','data_structures_score','sql_score',
        'machine_learning_score','backend_score','frontend_score','cloud_score','devops_score',
        'project_quality_score','real_client_project_count','internship_count','internship_duration_months',
        'freelance_project_count','hackathon_count','hackathon_awards','portfolio_score','github_repo_count',
        'github_avg_stars','open_source_contribution_count','linkedin_profile_score','cv_quality_score',
        'technical_interview_score','hr_interview_score','communication_score','teamwork_score',
        'leadership_score','presentation_score','certification_count','bootcamp_count','applications_sent',
        'interviews_attended'
    ]
    numeric_features = [c for c in numeric_candidates if c in X.columns]

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

    text_features = [text_col] if text_col in X.columns else []
    text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)))])

    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    if text_features:
        transformers.append(('txt', text_transformer, text_col))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    params = {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6, 'random_state': 42, 'n_jobs': -1}
    import os, json
    if os.path.exists('best_params.json'):
        with open('best_params.json', 'r') as f:
            bp = json.load(f)
            if 'xgboost' in bp:
                params.update(bp['xgboost'])

    model = Pipeline(steps=[
        ('preproc', preprocessor),
        ('xgb', XGBRegressor(**params))
    ])
    return model

def run_cv_and_submit(train_path, test_path, out_submission, model_out, cv=5, random_state=42):
    mlflow.set_experiment("Datathon2026_XGBoost")
    with mlflow.start_run():
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        if 'career_success_score' not in train.columns:
            raise ValueError('train file must contain career_success_score')

        y = train['career_success_score'].values
        X = train.drop(columns=['career_success_score', 'student_id'], errors='ignore')
        X_test = test.drop(columns=['student_id'], errors='ignore')

        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        oof_preds = np.zeros(len(X))
        scores = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model = build_pipeline_for_columns(pd.concat([X_tr, X_val, X_test], axis=0))
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_val)
            oof_preds[val_idx] = val_pred
            mse = mean_squared_error(y_val, val_pred)
            scores.append(mse)
            print(f'Fold {fold+1} MSE: {mse:.4f}')

        mean_cv = np.mean(scores)
        std_cv = np.std(scores)
        overall_mse = mean_squared_error(y, oof_preds)
        print(f'CV mean MSE: {mean_cv:.4f} ± {std_cv:.4f}  | OOF MSE: {overall_mse:.4f}')
        
        mlflow.log_param("cv_folds", cv)
        mlflow.log_param("model", "XGBoost")
        mlflow.log_metric("cv_mean_mse", mean_cv)
        mlflow.log_metric("oof_mse", overall_mse)

        # Fit on full training data
        final_model = build_pipeline_for_columns(pd.concat([X, X_test], axis=0))
        final_model.fit(X, y)

        os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
        joblib.dump(final_model, model_out)
        print('Saved model to', model_out)

        preds_test = final_model.predict(X_test)
        submission = pd.DataFrame({
            'student_id': test['student_id'] if 'student_id' in test.columns else np.arange(len(preds_test)),
            'career_success_score': preds_test
        })
        submission.to_csv(out_submission, index=False)
        print('Saved submission to', out_submission)

        oof_df = train[['student_id']].copy() if 'student_id' in train.columns else pd.DataFrame({'student_id': np.arange(len(oof_preds))})
        oof_df['oof_pred'] = oof_preds
        oof_df.to_csv('oof_predictions_xgb.csv', index=False)
        print('Saved OOF predictions to oof_predictions_xgb.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', default='test_x.csv')
    parser.add_argument('--out', default='submission_xgb.csv')
    parser.add_argument('--model', default='model_xgb.pkl')
    parser.add_argument('--cv', type=int, default=5)
    args = parser.parse_args()

    run_cv_and_submit(args.train, args.test, args.out, args.model, cv=args.cv)
