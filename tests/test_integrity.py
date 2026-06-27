import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.extract import extract_features  # noqa: E402
from src.scoring.integrity import integrity_mult  # noqa: E402

CANDIDATES_PATH = REPO_ROOT.parent / "candidates.jsonl"

# Locked-in regression fixtures: specific real candidate_ids from each
# honeypot cluster identified during STEP4_SPEC.md's recalibration (S1.2,
# S1.1). If these ever flip, the integrity gate has regressed.
EXPECTED_EXPERT_ZERO_CLUSTER = ["CAND_0003582", "CAND_0016000", "CAND_0033817"]
EXPECTED_MISMATCH_CLUSTER = ["CAND_0001610", "CAND_0003430", "CAND_0005291"]
EXPECTED_NORMAL = ["CAND_0000001", "CAND_0000002", "CAND_0000003"]
EXPECTED_TOTAL_FLAGGED_FULL_POOL = 69


def _base_candidate(**overrides):
    base = {
        "candidate_id": "CAND_TEST001",
        "profile": {
            "current_title": "ML Engineer", "years_of_experience": 6, "location": "Pune",
            "country": "India", "current_industry": "IT Services", "current_company": "X",
            "current_company_size": "51-200", "headline": "h", "summary": "s",
        },
        "career_history": [{
            "company": "X", "title": "ML Engineer", "start_date": "2018-01-01", "end_date": None,
            "duration_months": 72, "is_current": True, "industry": "IT",
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


def test_synthetic_expert_zero_duration_honeypot():
    c = _base_candidate(skills=[
        {"name": s, "proficiency": "expert", "endorsements": 5, "duration_months": 0}
        for s in ["A", "B", "C", "D", "E"]
    ])
    feat = extract_features(c)
    assert integrity_mult(feat) == 0.0


def test_synthetic_yoe_mismatch_honeypot():
    c = _base_candidate(career_history=[{
        "company": "X", "title": "ML Engineer", "start_date": "2010-01-01", "end_date": None,
        "duration_months": 200, "is_current": True, "industry": "IT",
        "company_size": "51-200", "description": "",
    }])
    c["profile"]["years_of_experience"] = 7  # 7*12=84 vs 200 -> mismatch=116mo
    feat = extract_features(c)
    assert integrity_mult(feat) == 0.0


def test_synthetic_normal_candidate_not_flagged():
    c = _base_candidate(skills=[
        {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 60},
    ])
    feat = extract_features(c)
    assert integrity_mult(feat) == 1.0


def test_real_expert_zero_cluster_flagged():
    found = _load_candidates_by_id(EXPECTED_EXPERT_ZERO_CLUSTER)
    for cid in EXPECTED_EXPERT_ZERO_CLUSTER:
        feat = extract_features(found[cid])
        assert integrity_mult(feat) == 0.0, f"{cid} should be flagged"


def test_real_mismatch_cluster_flagged():
    found = _load_candidates_by_id(EXPECTED_MISMATCH_CLUSTER)
    for cid in EXPECTED_MISMATCH_CLUSTER:
        feat = extract_features(found[cid])
        assert integrity_mult(feat) == 0.0, f"{cid} should be flagged"


def test_real_normal_candidates_not_flagged():
    found = _load_candidates_by_id(EXPECTED_NORMAL)
    for cid in EXPECTED_NORMAL:
        feat = extract_features(found[cid])
        assert integrity_mult(feat) == 1.0, f"{cid} should NOT be flagged"


def test_full_pool_flagged_count_matches_calibration():
    flagged = 0
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            feat = extract_features(c)
            if integrity_mult(feat) == 0.0:
                flagged += 1
    assert flagged == EXPECTED_TOTAL_FLAGGED_FULL_POOL, (
        f"expected {EXPECTED_TOTAL_FLAGGED_FULL_POOL} flagged, got {flagged} -- "
        "integrity gate calibration has drifted, see docs/STEP4_SPEC.md S1"
    )


def _load_candidates_by_id(ids):
    wanted = set(ids)
    found = {}
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["candidate_id"] in wanted:
                found[c["candidate_id"]] = c
                wanted.discard(c["candidate_id"])
                if not wanted:
                    break
    return found
