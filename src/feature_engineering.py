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


def add_nlp_embeddings(df_train, df_test, text_col='mentor_feedback_text', n_components=32):
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
    except ImportError:
        print("sentence_transformers or sklearn not installed. Skipping NLP embeddings.")
        return df_train, df_test

    if text_col not in df_train.columns:
        return df_train, df_test

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    train_texts = df_train[text_col].fillna("").tolist()
    test_texts = df_test[text_col].fillna("").tolist()
    
    print("Encoding texts...")
    train_emb = model.encode(train_texts, show_progress_bar=False)
    test_emb = model.encode(test_texts, show_progress_bar=False)
    
    print(f"Applying PCA down to {n_components} dimensions...")
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_emb)
    test_pca = pca.transform(test_emb)
    
    for i in range(n_components):
        df_train[f'emb_pca_{i}'] = train_pca[:, i]
        df_test[f'emb_pca_{i}'] = test_pca[:, i]
        
    return df_train, df_test

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

    # Use texts from original dataset but append to feature engineered dataset
    train_fe['mentor_feedback_text'] = train['mentor_feedback_text'] if 'mentor_feedback_text' in train.columns else ""
    test_fe['mentor_feedback_text'] = test['mentor_feedback_text'] if 'mentor_feedback_text' in test.columns else ""
    
    train_fe, test_fe = add_nlp_embeddings(train_fe, test_fe, text_col='mentor_feedback_text', n_components=32)

    # Drop the raw text column from fe dataset to save space
    train_fe = train_fe.drop(columns=['mentor_feedback_text'], errors='ignore')
    test_fe = test_fe.drop(columns=['mentor_feedback_text'], errors='ignore')

    train_fe.to_csv(args.out_train, index=False)
    test_fe.to_csv(args.out_test, index=False)
    print('Saved', args.out_train, 'and', args.out_test)

if __name__ == '__main__':
    main()
