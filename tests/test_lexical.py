import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.lexical import fit_tfidf, lexical_sim  # noqa: E402

JD_TEXT_PATH = REPO_ROOT / "config" / "jd_reference_text.txt"


def test_fit_tfidf_and_sim_shapes():
    jd_text = JD_TEXT_PATH.read_text(encoding="utf-8")
    profile_texts = [
        "senior ai engineer building retrieval embeddings ranking systems",
        "hr manager handling recruitment payroll employee relations",
        "marketing manager running social media campaigns",
    ]
    vec, matrix, jd_vec = fit_tfidf(profile_texts, jd_text, max_features=500, ngram_range=(1, 2))
    assert matrix.shape[0] == 3
    sims = lexical_sim(matrix, jd_vec)
    assert len(sims) == 3


def test_ai_relevant_text_scores_higher_than_unrelated():
    jd_text = JD_TEXT_PATH.read_text(encoding="utf-8")
    profile_texts = [
        "senior ai engineer embeddings retrieval vector search ranking ndcg evaluation python",
        "hr manager recruitment payroll employee relations interviews onboarding",
    ]
    vec, matrix, jd_vec = fit_tfidf(profile_texts, jd_text, max_features=500, ngram_range=(1, 2))
    sims = lexical_sim(matrix, jd_vec)
    assert sims[0] > sims[1]


def test_lexical_sim_single_row_index():
    jd_text = JD_TEXT_PATH.read_text(encoding="utf-8")
    profile_texts = ["ai engineer retrieval ranking", "graphic designer photoshop illustrator"]
    vec, matrix, jd_vec = fit_tfidf(profile_texts, jd_text, max_features=500, ngram_range=(1, 2))
    sim_row0 = lexical_sim(matrix, jd_vec, idx=[0])
    assert len(sim_row0) == 1
