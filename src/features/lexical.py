"""BM25 lexical similarity. Replaced TF-IDF: candidate profiles vary widely
in length (80-800 words) and TF-IDF cosine similarity over-rewards verbose
profiles. BM25's length normalisation (b=0.75) and term-frequency saturation
(k1=1.5) fix both issues for JD-vs-profile retrieval.
"""
import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list:
    return (text or "").lower().split()


def fit_bm25(profile_texts: list, jd_text: str):
    """Fit BM25Okapi on corpus. Returns (retriever, jd_tokens)."""
    retriever = BM25Okapi([_tokenize(t) for t in profile_texts])
    return retriever, _tokenize(jd_text)


def score_all(retriever: BM25Okapi, jd_tokens: list) -> np.ndarray:
    """Score all corpus documents against the JD. Returns a min-max
    normalised float array of shape (n_docs,) in original corpus order."""
    raw = retriever.get_scores(jd_tokens)
    lo, hi = raw.min(), raw.max()
    if hi > lo:
        return (raw - lo) / (hi - lo)
    return np.full(len(raw), 0.5)
