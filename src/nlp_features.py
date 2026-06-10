"""
Advanced NLP feature extraction for mentor_feedback_text.
Combines TF-IDF, character n-grams, and basic statistics.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


def extract_nlp_features(texts, max_features=2000, ngram_range=(1,2)):
    """Extract TF-IDF and statistical features from text"""
    texts = texts.fillna("").astype(str)
    
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2, max_df=0.8)
    tfidf_mat = tfidf.fit_transform(texts).toarray()
    tfidf_df = pd.DataFrame(tfidf_mat, columns=[f'tfidf_{i}' for i in range(tfidf_mat.shape[1])])
    
    # Statistical features
    stats = pd.DataFrame({
        'text_length': texts.str.len(),
        'word_count': texts.str.split().str.len(),
        'avg_word_length': texts.str.split().apply(lambda x: np.mean([len(w) for w in x]) if len(x)>0 else 0),
        'unique_words': texts.str.split().apply(lambda x: len(set(x))),
        'sentence_count': texts.str.count(r'[\.\!\?]').clip(lower=1)
    })
    
    return pd.concat([tfidf_df, stats], axis=1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train.csv')
    parser.add_argument('--test', default='test_x.csv')
    parser.add_argument('--out-train', default='train_nlp.csv')
    parser.add_argument('--out-test', default='test_nlp.csv')
    args = parser.parse_args()
    
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    
    print('Extracting NLP features for train...')
    train_nlp = extract_nlp_features(train['mentor_feedback_text'], max_features=2000)
    print('Extracting NLP features for test...')
    test_nlp = extract_nlp_features(test['mentor_feedback_text'], max_features=2000)
    
    # combine with original data
    train_out = pd.concat([train.reset_index(drop=True), train_nlp.reset_index(drop=True)], axis=1)
    test_out = pd.concat([test.reset_index(drop=True), test_nlp.reset_index(drop=True)], axis=1)
    
    train_out.to_csv(args.out_train, index=False)
    test_out.to_csv(args.out_test, index=False)
    print(f'Saved NLP-enhanced datasets to {args.out_train}, {args.out_test}')
