import pandas as pd
import numpy as np


def engineer(df):
    df = df.copy()

    # Add missing value indicators
    for col in df.columns:
        if df[col].isnull().any():
            df[col + '_isnull'] = df[col].isnull().astype(int)

    # Years since graduation relative to application_year if present, else use 2026
    if 'graduation_year' in df.columns and 'application_year' in df.columns:
        df['years_since_grad'] = df['application_year'] - df['graduation_year']
    elif 'graduation_year' in df.columns:
        df['years_since_grad'] = 2026 - df['graduation_year']
    else:
        df['years_since_grad'] = 0

    # Age bucket
    if 'age' in df.columns:
        df['age_bucket'] = pd.cut(df['age'], bins=[0,20,25,30,40,100], labels=['<20','20-25','25-30','30-40','40+'])

    # CGPA interactions and transforms
    if 'cgpa' in df.columns:
        df['cgpa_sq'] = df['cgpa'] ** 2
        df['cgpa_log'] = np.log1p(df['cgpa'].clip(lower=0))

    # Project/internship aggregations
    for c in ['real_client_project_count','internship_count','freelance_project_count','hackathon_count']:
        if c in df.columns:
            df[c+'_log'] = np.log1p(df[c].fillna(0))

    # Github activity transforms
    if 'github_repo_count' in df.columns:
        df['github_repos_per_year'] = df['github_repo_count'] / df.get('years_since_grad', 1).replace(0,1)
        df['github_repo_log'] = np.log1p(df['github_repo_count'].fillna(0))
    if 'github_avg_stars' in df.columns:
        df['github_star_log'] = np.log1p(df['github_avg_stars'].fillna(0))

    # Advanced Interaction features
    if 'coding_score' in df.columns and 'project_quality_score' in df.columns:
        df['coding_x_project'] = df['coding_score'].fillna(0) * df['project_quality_score'].fillna(0)
    if 'technical_interview_score' in df.columns and 'communication_score' in df.columns:
        df['tech_x_comm'] = df['technical_interview_score'].fillna(0) * df['communication_score'].fillna(0)
    
    # New: total interview score
    interview_cols = ['technical_interview_score', 'hr_interview_score']
    if all(c in df.columns for c in interview_cols):
        df['total_interview_score'] = df[interview_cols].sum(axis=1)

    # New: Group Aggregations
    if 'department' in df.columns and 'cgpa' in df.columns:
        dept_cgpa = df.groupby('department')['cgpa'].transform('mean')
        df['cgpa_dept_ratio'] = df['cgpa'] / (dept_cgpa + 1e-5)

    if 'university_tier' in df.columns and 'coding_score' in df.columns:
        tier_coding = df.groupby('university_tier')['coding_score'].transform('mean')
        df['coding_tier_ratio'] = df['coding_score'] / (tier_coding + 1e-5)

    # Fill NA for numeric columns with 0 for safety
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train.csv')
    parser.add_argument('--test', default='test_x.csv')
    parser.add_argument('--out-train', default='train_fe.csv')
    parser.add_argument('--out-test', default='test_fe.csv')
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    train_fe = engineer(train)
    test_fe = engineer(test)

    train_fe.to_csv(args.out_train, index=False)
    test_fe.to_csv(args.out_test, index=False)
    print('Saved', args.out_train, 'and', args.out_test)


if __name__ == '__main__':
    main()
