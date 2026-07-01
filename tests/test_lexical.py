import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.lexical import fit_bm25, score_all  # noqa: E402

JD_TEXT_PATH = REPO_ROOT / "config" / "jd_reference_text.txt"


def test_fit_bm25_and_score_shapes():
    jd_text = JD_TEXT_PATH.read_text(encoding="utf-8")
    profile_texts = [
        "senior ai engineer building retrieval embeddings ranking systems",
        "hr manager handling recruitment payroll employee relations",
        "marketing manager running social media campaigns",
    ]
    retriever, jd_tokens = fit_bm25(profile_texts, jd_text)
    scores = score_all(retriever, jd_tokens)
    assert len(scores) == 3


def test_ai_relevant_text_scores_higher_than_unrelated():
    jd_text = JD_TEXT_PATH.read_text(encoding="utf-8")
    # Three docs so BM25 IDF is meaningful (with 2 docs and zero overlap, both
    # raw scores can be 0 and min-max normalisation collapses them to 0.5 each)
    profile_texts = [
        "nlp engineer retrieval augmented generation embeddings vector search ranking recommendation python pytorch transformers",
        "hr manager recruitment payroll employee relations interviews onboarding benefits compensation",
        "project manager agile scrum stakeholder communication delivery risk management",
    ]
    retriever, jd_tokens = fit_bm25(profile_texts, jd_text)
    scores = score_all(retriever, jd_tokens)
    assert scores[0] > scores[1]


def test_scores_normalised_in_range():
    jd_text = JD_TEXT_PATH.read_text(encoding="utf-8")
    profile_texts = [
        "ai engineer retrieval ranking embeddings vector search",
        "graphic designer photoshop illustrator",
        "data scientist machine learning python pandas",
    ]
    retriever, jd_tokens = fit_bm25(profile_texts, jd_text)
    scores = score_all(retriever, jd_tokens)
    assert all(0.0 <= s <= 1.0 for s in scores)
