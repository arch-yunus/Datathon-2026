import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')


def apply_target_encoding(X_train, X_test, y_train, cat_cols, smoothing=1.0):
    """Apply target encoding to categorical columns"""
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    for col in cat_cols:
        if col not in X_train.columns:
            continue
        
        te = TargetEncoder(smoothing=smoothing)
        X_train_enc[col] = te.fit_transform(X_train[col], y_train)
        X_test_enc[col] = te.transform(X_test[col])
    
    return X_train_enc, X_test_enc


def apply_frequency_encoding(X, cols):
    """Frequency encode categorical columns"""
    X = X.copy()
    for col in cols:
        if col in X.columns:
            freq = X[col].value_counts(normalize=True)
            X[col] = X[col].map(freq)
    return X


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train.csv')
    parser.add_argument('--test', default='test_x.csv')
    parser.add_argument('--out-train', default='train_te.csv')
    parser.add_argument('--out-test', default='test_te.csv')
    args = parser.parse_args()
    
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    
    cat_cols = ['department','university_tier','target_role','hobby','preferred_social_media_platform']
    cat_cols = [c for c in cat_cols if c in train.columns]
    
    y_train = train['career_success_score'] if 'career_success_score' in train.columns else None
    
    if y_train is not None:
        train_te, test_te = apply_target_encoding(train, test, y_train, cat_cols, smoothing=1.0)
    else:
        train_te, test_te = train.copy(), test.copy()
    
    train_te.to_csv(args.out_train, index=False)
    test_te.to_csv(args.out_test, index=False)
    print(f'Target encoded datasets saved to {args.out_train}, {args.out_test}')
