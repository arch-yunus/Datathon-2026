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

    # New Phase 3 Interactions
    if 'cgpa' in df.columns and 'portfolio_score' in df.columns:
        df['cgpa_x_portfolio'] = df['cgpa'].fillna(0) * df['portfolio_score'].fillna(0)
    if 'coding_score' in df.columns and 'technical_interview_score' in df.columns:
        df['coding_x_tech_interview'] = df['coding_score'].fillna(0) * df['technical_interview_score'].fillna(0)
    if 'failed_courses_count' in df.columns and 'cgpa' in df.columns:
        df['failed_course_ratio'] = df['failed_courses_count'].fillna(0) / (df['cgpa'].clip(lower=0.1))
    if 'open_source_contribution_count' in df.columns:
        df['open_source_log'] = np.log1p(df['open_source_contribution_count'].fillna(0))

    # New: Group Aggregations
    if 'department' in df.columns and 'cgpa' in df.columns:
        dept_cgpa = df.groupby('department')['cgpa'].transform('mean')
        df['cgpa_dept_ratio'] = df['cgpa'] / (dept_cgpa + 1e-5)
    
    if 'department' in df.columns and 'attendance_rate' in df.columns:
        dept_att = df.groupby('department')['attendance_rate'].transform('mean')
        df['attendance_dept_ratio'] = df['attendance_rate'] / (dept_att + 1e-5)

    if 'university_tier' in df.columns and 'coding_score' in df.columns:
        tier_coding = df.groupby('university_tier')['coding_score'].transform('mean')
        df['coding_tier_ratio'] = df['coding_score'] / (tier_coding + 1e-5)

    # Fill NA for numeric columns with 0 for safety
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df


def add_oof_target_encoding(df_train, df_test, cat_cols, target_col='career_success_score', n_splits=5, smoothing=10, noise_level=0.01):
    """Adds out-of-fold target encoding to categorical columns to avoid leakage."""
    df_train = df_train.copy()
    df_test = df_test.copy()
    
    if target_col not in df_train.columns:
        return df_train, df_test

    y = df_train[target_col].values
    global_mean = y.mean()
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for col in cat_cols:
        if col not in df_train.columns:
            continue
            
        # OOF target encoding on train set
        oof_encoded = np.zeros(len(df_train))
        for tr_idx, val_idx in kf.split(df_train):
            tr_df = df_train.iloc[tr_idx]
            val_df = df_train.iloc[val_idx]
            
            # stats on tr_df
            stats = tr_df.groupby(col)[target_col].agg(['count', 'mean'])
            counts = stats['count']
            means = stats['mean']
            
            # smoothed encoding
            smooth_val = (counts * means + global_mean * smoothing) / (counts + smoothing)
            
            # map to val_df
            mapped = val_df[col].map(smooth_val).fillna(global_mean).values
            if noise_level > 0:
                mapped += np.random.normal(0, noise_level * np.std(y), size=len(mapped))
                
            oof_encoded[val_idx] = mapped
            
        df_train[f'{col}_te'] = oof_encoded
        
        # test set encoding (using full train stats)
        full_stats = df_train.groupby(col)[target_col].agg(['count', 'mean'])
        full_counts = full_stats['count']
        full_means = full_stats['mean']
        full_smooth_val = (full_counts * full_means + global_mean * smoothing) / (full_counts + smoothing)
        
        df_test[f'{col}_te'] = df_test[col].map(full_smooth_val).fillna(global_mean)
        
    return df_train, df_test


def add_kmeans_features(df_train, df_test, n_clusters=5):
    """Performs K-Means clustering and appends distances to cluster centers as features."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    df_train = df_train.copy()
    df_test = df_test.copy()
    
    cols_to_cluster = [
        'age', 'cgpa', 'coding_score', 'problem_solving_score',
        'technical_interview_score', 'attendance_rate', 'internship_count',
        'portfolio_score', 'github_repo_count', 'open_source_contribution_count'
    ]
    cols_to_cluster = [c for c in cols_to_cluster if c in df_train.columns]
    
    if not cols_to_cluster:
        return df_train, df_test
        
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_clust = scaler.fit_transform(imp.fit_transform(df_train[cols_to_cluster]))
    X_test_clust = scaler.transform(imp.transform(df_test[cols_to_cluster]))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_dist = kmeans.fit_transform(X_train_clust)
    test_dist = kmeans.transform(X_test_clust)
    
    for i in range(n_clusters):
        df_train[f'kmeans_dist_c{i}'] = train_dist[:, i]
        df_test[f'kmeans_dist_c{i}'] = test_dist[:, i]
        
    df_train['kmeans_cluster'] = kmeans.labels_
    df_test['kmeans_cluster'] = kmeans.predict(X_test_clust)
    
    return df_train, df_test


def add_nlp_embeddings(df_train, df_test, text_col='mentor_feedback_text', n_components=64):
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
    except ImportError:
        print("sentence_transformers or sklearn not installed. Skipping NLP embeddings.")
        return df_train, df_test

    if text_col not in df_train.columns:
        return df_train, df_test

    print("Loading Multilingual SentenceTransformer model...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
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

    # Custom OOF Target Encoding for high cardinality categorical features
    cat_cols = ['department', 'university_tier', 'target_role', 'hobby', 'preferred_social_media_platform']
    cat_cols = [c for c in cat_cols if c in train.columns]
    
    train_fe, test_fe = add_oof_target_encoding(train_fe, test_fe, cat_cols, target_col='career_success_score')

    # KMeans Archetype Features
    train_fe, test_fe = add_kmeans_features(train_fe, test_fe, n_clusters=5)

    # Use texts from original dataset but append to feature engineered dataset
    train_fe['mentor_feedback_text'] = train['mentor_feedback_text'] if 'mentor_feedback_text' in train.columns else ""
    test_fe['mentor_feedback_text'] = test['mentor_feedback_text'] if 'mentor_feedback_text' in test.columns else ""
    
    train_fe, test_fe = add_nlp_embeddings(train_fe, test_fe, text_col='mentor_feedback_text', n_components=64)

    # Drop the raw text column from fe dataset to save space
    train_fe = train_fe.drop(columns=['mentor_feedback_text'], errors='ignore')
    test_fe = test_fe.drop(columns=['mentor_feedback_text'], errors='ignore')

    train_fe.to_csv(args.out_train, index=False)
    test_fe.to_csv(args.out_test, index=False)
    print('Saved', args.out_train, 'and', args.out_test)


if __name__ == '__main__':
    main()
