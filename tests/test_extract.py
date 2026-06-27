import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.extract import extract_features  # noqa: E402

SAMPLE_PATH = REPO_ROOT.parent / "sample_candidates.json"

EXPECTED_KEYS = {
    "candidate_id", "current_title", "current_industry", "current_company",
    "current_company_size", "yoe", "skills", "n_skills", "career", "n_jobs",
    "total_career_months", "avg_tenure_months", "edu_best_tier", "edu_is_cs",
    "location", "country", "open_to_work", "recruiter_response_rate",
    "last_active_date", "interview_completion_rate", "profile_completeness",
    "saved_by_recruiters_30d", "notice_period_days", "github_activity_score",
    "willing_to_relocate", "preferred_work_mode", "skill_assessment_scores",
    "endorsements_received", "profile_text",
}


def load_samples():
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def test_extract_features_all_samples_no_exceptions():
    samples = load_samples()
    assert len(samples) == 50
    for c in samples:
        feat = extract_features(c)
        assert set(feat.keys()) == EXPECTED_KEYS


def test_extract_features_consistency():
    for c in load_samples():
        feat = extract_features(c)
        assert feat["n_skills"] == len(c.get("skills", []))
        assert feat["n_jobs"] == len(c.get("career_history", []))
        assert feat["yoe"] >= 0
        assert 0.0 <= feat["edu_best_tier"] <= 1.0
        assert feat["edu_is_cs"] in (0.0, 1.0)


def test_flatten_text_contains_headline():
    for c in load_samples():
        feat = extract_features(c)
        headline = c["profile"].get("headline", "")
        if headline:
            assert headline.lower() in feat["profile_text"]
