import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.extract import extract_features  # noqa: E402
from src.scoring.scorer import (  # noqa: E402
    availability_mult, edu_score, experience_fit, full_score,
    location_score, skill_trust, trajectory_score,
)

CANDIDATES_PATH = REPO_ROOT.parent / "candidates.jsonl"
SAMPLE_PATH = REPO_ROOT.parent / "sample_candidates.json"


def load_jd():
    return yaml.safe_load((REPO_ROOT / "config" / "jd_profile.yaml").read_text(encoding="utf-8"))


def load_ontology():
    return yaml.safe_load((REPO_ROOT / "config" / "skill_ontology.yaml").read_text(encoding="utf-8"))


def load_weights():
    return yaml.safe_load((REPO_ROOT / "config" / "weights.yaml").read_text(encoding="utf-8"))


def _base_candidate(**overrides):
    base = {
        "candidate_id": "CAND_TEST001",
        "profile": {
            "current_title": "ML Engineer", "years_of_experience": 7, "location": "Pune",
            "country": "India", "current_industry": "AI/ML", "current_company": "X",
            "current_company_size": "51-200", "headline": "h", "summary": "s",
        },
        "career_history": [{
            "company": "X", "title": "ML Engineer", "start_date": "2018-01-01", "end_date": None,
            "duration_months": 72, "is_current": True, "industry": "AI/ML",
            "company_size": "51-200", "description": "",
        }],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "open_to_work_flag": True, "recruiter_response_rate": 0.5,
            "last_active_date": "2025-01-01", "interview_completion_rate": 0.5,
            "profile_completeness_score": 50, "saved_by_recruiters_30d": 1,
            "notice_period_days": 30, "github_activity_score": -1,
            "willing_to_relocate": False, "preferred_work_mode": "hybrid",
            "skill_assessment_scores": {}, "endorsements_received": 0,
        },
    }
    base.update(overrides)
    return base


def test_skill_trust_in_range():
    jd, ont, w = load_jd(), load_ontology(), load_weights()
    feat = extract_features(_base_candidate(skills=[
        {"name": "RAG", "proficiency": "expert", "endorsements": 10, "duration_months": 30},
        {"name": "Pinecone", "proficiency": "advanced", "endorsements": 5, "duration_months": 20},
    ]))
    score, evidence = skill_trust(feat, jd, ont, w)
    assert 0.0 <= score <= 1.0
    assert "retrieval_embeddings" in evidence
    assert "vector_search" in evidence


def test_skill_trust_zero_for_no_skills():
    jd, ont, w = load_jd(), load_ontology(), load_weights()
    feat = extract_features(_base_candidate(skills=[]))
    score, evidence = skill_trust(feat, jd, ont, w)
    assert score == 0.0
    assert evidence == []


def test_experience_fit_ideal_band():
    jd = load_jd()
    assert experience_fit(7.0, jd) == 1.0
    assert experience_fit(5.5, jd) == 0.85
    assert experience_fit(1.0, jd) == 0.15


def test_trajectory_score_consulting_only_penalized():
    jd, ont, w = load_jd(), load_ontology(), load_weights()
    feat = extract_features(_base_candidate(career_history=[{
        "company": "TCS", "title": "ML Engineer", "start_date": "2018-01-01", "end_date": None,
        "duration_months": 72, "is_current": True, "industry": "IT", "company_size": "10001+",
        "description": "",
    }]))
    score = trajectory_score(feat, jd, ont, w)
    assert score < 0.85 - 0.5  # full-career-consulting penalty applied


def test_trajectory_score_cv_speech_only_penalized():
    jd, ont, w = load_jd(), load_ontology(), load_weights()
    skills = [
        {"name": "Computer Vision", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
        {"name": "Object Detection", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
    ]
    feat = extract_features(_base_candidate(skills=skills))
    score_with_penalty = trajectory_score(feat, jd, ont, w)

    skills_with_nlp = skills + [{"name": "NLP", "proficiency": "advanced", "endorsements": 5, "duration_months": 24}]
    feat_rescued = extract_features(_base_candidate(skills=skills_with_nlp))
    score_rescued = trajectory_score(feat_rescued, jd, ont, w)
    assert score_with_penalty < score_rescued


def test_trajectory_score_langchain_only_recent_penalized():
    jd, ont, w = load_jd(), load_ontology(), load_weights()
    feat = extract_features(_base_candidate(skills=[
        {"name": "LangChain", "proficiency": "intermediate", "endorsements": 2, "duration_months": 6},
    ]))
    score = trajectory_score(feat, jd, ont, w)
    feat_normal = extract_features(_base_candidate(skills=[]))
    score_normal = trajectory_score(feat_normal, jd, ont, w)
    assert score < score_normal


def test_edu_score_range():
    jd = load_jd()
    feat = extract_features(_base_candidate(education=[
        {"institution": "IIT", "degree": "B.Tech", "field_of_study": "Computer Science",
         "start_year": 2010, "end_year": 2014, "tier": "tier_1"},
    ]))
    assert edu_score(feat) == 1.0  # tier_1 (1.0*0.6) + is_cs (1.0*0.4)


def test_location_score_best_hub():
    jd = load_jd()
    feat = extract_features(_base_candidate())
    assert location_score(feat, jd) == 1.0  # Pune is a "best" hub


def test_availability_mult_in_range():
    w = load_weights()
    feat = extract_features(_base_candidate())
    m = availability_mult(feat, w, ref_date=None)
    assert 0.3 <= m <= 1.1


def test_full_score_keyword_stuffer_scores_lower_than_genuine_fit():
    jd, ont, w = load_jd(), load_ontology(), load_weights()
    from src.scoring.role_fit import role_fit
    from src.scoring.integrity import integrity_mult

    ai_skills = [{"name": s, "proficiency": "expert", "endorsements": 15, "duration_months": 30}
                 for s in ["RAG", "Pinecone", "FAISS", "Embeddings", "Vector Search"]]

    stuffer_candidate = _base_candidate(skills=ai_skills)
    stuffer_candidate["profile"]["current_title"] = "HR Manager"
    stuffer_candidate["career_history"][0]["title"] = "HR Manager"  # no rescue path
    stuffer = extract_features(stuffer_candidate)

    genuine_candidate = _base_candidate(skills=ai_skills)
    genuine_candidate["profile"]["current_title"] = "ML Engineer"
    genuine = extract_features(genuine_candidate)

    def score_of(feat):
        rf = role_fit(feat, jd, ont)
        im = integrity_mult(feat)
        lex = 0.05  # fixed placeholder, not under test here
        return full_score(feat, jd, ont, w, rf, im, lex)

    assert score_of(stuffer) < score_of(genuine)


def test_sample_candidates_no_exceptions():
    jd, ont, w = load_jd(), load_ontology(), load_weights()
    samples = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    for c in samples:
        feat = extract_features(c)
        skill_trust(feat, jd, ont, w)
        experience_fit(feat["yoe"], jd)
        trajectory_score(feat, jd, ont, w)
        edu_score(feat)
        location_score(feat, jd)
        availability_mult(feat, w)
