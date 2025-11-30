import os, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle
import re

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data", "careers.csv")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

# --- lightweight cleaner (mirrors app normalizer idea) ---
STOPWORDS = set("""
a an and are as at be but by for from has have if in into is it its of on or that the to was were will with your you
""".split())

def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+/#&\-\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def row_to_blob(row):
    # FIELD WEIGHTING:
    # - skills: x4 (most important)
    # - interests: x2
    # - career_name: x2
    # - description: x1
    name = clean(row.get("career_name",""))
    skills = clean(str(row.get("core_skills","")).replace(";", " "))
    interests = clean(str(row.get("interests","")).replace(";", " "))
    desc = clean(row.get("description",""))

    blob = " ".join([
        (name + " ") * 2,
        (skills + " ") * 4,
        (interests + " ") * 2,
        (desc + " ") * 1
    ])
    return blob.strip()

print("Loading:", DATA)
df = pd.read_csv(DATA)

corpus = df.apply(row_to_blob, axis=1).tolist()

# Stronger vectorizer settings for relevance
vec = TfidfVectorizer(
    ngram_range=(1,2),      # unigrams + bigrams
    min_df=2,               # drop ultra-rare noise
    max_df=0.85,            # drop overly common junk
    sublinear_tf=True,      # dampen term frequency
    stop_words=None         # we cleaned ourselves
)

X = vec.fit_transform(corpus)

with open(os.path.join(MODELS, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vec, f)

sparse.save_npz(os.path.join(MODELS, "tfidf_matrix.npz"), X)

print("Built weighted TF-IDF index:")
print(" - careers:", X.shape[0])
print(" - features:", X.shape[1])
print("Saved to models/tfidf_vectorizer.pkl and models/tfidf_matrix.npz")
