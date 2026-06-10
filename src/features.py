import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer


class TextSelector(TransformerMixin, BaseEstimator):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key].fillna("")


def build_text_vectorizer(series, max_features=5000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    vec.fit(series.fillna(""))
    return vec
