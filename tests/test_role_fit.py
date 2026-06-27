import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.extract import extract_features  # noqa: E402
from src.scoring.role_fit import role_fit  # noqa: E402

CANDIDATES_PATH = REPO_ROOT.parent / "candidates.jsonl"
SAMPLE_PATH = REPO_ROOT.parent / "sample_candidates.json"


def load_jd():
    return yaml.safe_load((REPO_ROOT / "config" / "jd_profile.yaml").read_text(encoding="utf-8"))


def load_ontology():
    return yaml.safe_load((REPO_ROOT / "config" / "skill_ontology.yaml").read_text(encoding="utf-8"))


def _synthetic_candidate(title, career_titles=None, skills=None):
    career_titles = career_titles or [title]
    career = [
        {"company": f"Co{i}", "title": t, "start_date": "2018-01-01", "end_date": None,
         "duration_months": 24, "is_current": (i == 0), "industry": "IT",
         "company_size": "51-200", "description": ""}
        for i, t in enumerate(career_titles)
    ]
    return {
        "candidate_id": "CAND_TEST001",
        "profile": {
            "current_title": title, "years_of_experience": 6, "location": "Pune",
            "country": "India", "current_industry": "IT Services", "current_company": "X",
            "current_company_size": "51-200", "headline": "h", "summary": "s",
        },
        "career_history": career,
        "education": [],
        "skills": skills or [],
        "redrob_signals": {
            "open_to_work_flag": True, "recruiter_response_rate": 0.5,
            "last_active_date": "2025-01-01", "interview_completion_rate": 0.5,
            "profile_completeness_score": 50, "saved_by_recruiters_30d": 1,
            "notice_period_days": 30, "github_activity_score": -1,
            "willing_to_relocate": False, "preferred_work_mode": "hybrid",
            "skill_assessment_scores": {}, "endorsements_received": 0,
        },
    }


def test_trap_titles_score_low():
    jd, ont = load_jd(), load_ontology()
    for title in ("HR Manager", "Marketing Manager", "Graphic Designer"):
        feat = extract_features(_synthetic_candidate(title))
        assert role_fit(feat, jd, ont) <= 0.2, f"{title} should score <=0.2"


def test_strong_title_scores_max():
    jd, ont = load_jd(), load_ontology()
    feat = extract_features(_synthetic_candidate("ML Engineer"))
    assert role_fit(feat, jd, ont) == 1.0


def test_reject_title_rescued_by_strong_career_history():
    jd, ont = load_jd(), load_ontology()
    feat = extract_features(_synthetic_candidate(
        "HR Manager", career_titles=["HR Manager", "ML Engineer"]))
    assert role_fit(feat, jd, ont) == 0.55


def test_all_real_titles_run_without_exception():
    jd, ont = load_jd(), load_ontology()
    seen_titles = set()
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            t = c["profile"]["current_title"]
            if t in seen_titles:
                continue
            seen_titles.add(t)
            feat = extract_features(c)
            score = role_fit(feat, jd, ont)
            assert 0.15 <= score <= 1.0, f"{t} produced out-of-range score {score}"
    assert len(seen_titles) == 47


def test_sample_candidates_no_exceptions():
    jd, ont = load_jd(), load_ontology()
    samples = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    for c in samples:
        feat = extract_features(c)
        score = role_fit(feat, jd, ont)
        assert 0.15 <= score <= 1.0
