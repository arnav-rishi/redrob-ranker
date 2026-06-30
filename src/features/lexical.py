"""Classical TF-IDF lexical similarity. Not a neural model -- one signal
among many, never the decider.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def fit_tfidf(profile_texts: list, jd_text: str, max_features=20000, ngram_range=(1, 2)):
    """Fit on candidate profile texts + the JD reference text as one extra
    doc, so the JD's own vocabulary is guaranteed to be in the fitted
    vocabulary. Returns (vectorizer, candidate_matrix, jd_vector)."""
    vec = TfidfVectorizer(max_features=max_features, ngram_range=tuple(ngram_range),
                          stop_words="english", sublinear_tf=True)
    matrix = vec.fit_transform(list(profile_texts) + [jd_text])
    jd_vec = matrix[-1]
    candidate_matrix = matrix[:-1]
    return vec, candidate_matrix, jd_vec


def lexical_sim(candidate_matrix, jd_vec, idx=None):
    """Cosine similarity (== linear_kernel on L2-normalized TF-IDF) between
    the JD vector and one or more candidate rows."""
    rows = candidate_matrix if idx is None else candidate_matrix[idx]
    return linear_kernel(jd_vec, rows).ravel()
