import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd


def compute_sentence_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=64):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings)


def add_embeddings_to_df(df, text_col='mentor_feedback_text', model_name='all-MiniLM-L6-v2'):
    texts = df[text_col].fillna("").tolist() if text_col in df.columns else [""] * len(df)
    emb = compute_sentence_embeddings(texts, model_name=model_name)
    emb_df = pd.DataFrame(emb, index=df.index)
    # rename columns to avoid clash
    emb_df.columns = [f'emb_{i}' for i in range(emb_df.shape[1])]
    return pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
