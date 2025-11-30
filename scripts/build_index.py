import os, json, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle

BASE = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE, "data", "careers.csv")
models_dir = os.path.join(BASE, "models")
os.makedirs(models_dir, exist_ok=True)

df = pd.read_csv(data_path)

def row_to_blob(row):
    parts = [
        str(row.get("career_name","")),
        str(row.get("core_skills","")).replace(";", " "),
        str(row.get("interests","")).replace(";", " "),
        str(row.get("description",""))
    ]
    return " ".join(parts)

corpus = df.apply(row_to_blob, axis=1).tolist()

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')
X = vectorizer.fit_transform(corpus)

# Save artifacts
with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)
sparse.save_npz(os.path.join(models_dir, "tfidf_matrix.npz"), X)

print("Built TF-IDF index:")
print(" - careers:", X.shape[0])
print(" - features:", X.shape[1])
print("Saved to models/tfidf_vectorizer.pkl and models/tfidf_matrix.npz")