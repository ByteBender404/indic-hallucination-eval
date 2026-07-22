import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_vectorizer = None
_corpus = []

def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=10000,
            sublinear_tf=True
        )
        # fit on a small seed corpus so it's ready
        seed = [
            "यह एक परीक्षण वाक्य है",
            "this is a test sentence",
            "இது ஒரு சோதனை வாக்கியம்",
            "hello world example text",
            "भारत एक महान देश है"
        ]
        _vectorizer.fit(seed)
    return _vectorizer

def embed(text: str):
    vectorizer = get_vectorizer()
    try:
        vec = vectorizer.transform([text]).toarray()[0]
    except Exception:
        vectorizer.fit([text])
        vec = vectorizer.transform([text]).toarray()[0]
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm