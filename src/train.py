import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import joblib


def build_pipeline(text_col='mentor_feedback_text'):
    numeric_features = [
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
    # Keep only numeric features present in data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['department','university_tier','target_role','hobby','preferred_social_media_platform']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('txt', text_transformer, text_col)
    ], remainder='drop')

    model = Pipeline(steps=[
        ('preproc', preprocessor),
        ('lgbm', LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42))
    ])
    return model


def main(args):
    df = pd.read_csv(args.data)
    target = 'career_success_score'
    if target not in df.columns:
        raise ValueError('train.csv must contain career_success_score')
    X = df.drop(columns=[target, 'student_id'])
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    model = build_pipeline()
    model.fit(X_train, y_train)

    # save model
    joblib.dump(model, args.out)
    print('Model trained and saved to', args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='model.pkl')
    args = parser.parse_args()
    main(args)
