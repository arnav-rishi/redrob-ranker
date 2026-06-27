"""Mechanical JSON -> flat feature dict transform. No JD logic, no scoring.

See docs/STEP2_SPEC.md for rationale. Helper bodies (_best_tier, _is_cs_field,
_flatten_text) were left unspecified in the original design doc; defined here.
"""

_TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25, "unknown": 0.0}

_CS_FIELD_PATTERNS = [
    "computer science", "software engineering", "information technology",
    "data science", "artificial intelligence", "machine learning",
    "computer engineering", "electronics", "information systems",
]


def _best_tier(edu):
    if not edu:
        return 0.0
    return max(_TIER_SCORE.get(e.get("tier", "unknown"), 0.0) for e in edu)


def _is_cs_field(edu):
    for e in edu:
        field = (e.get("field_of_study") or "").lower()
        if any(p in field for p in _CS_FIELD_PATTERNS):
            return 1.0
    return 0.0


def _flatten_text(p, career, skills, edu):
    parts = [p.get("headline", ""), p.get("summary", "")]
    for j in career:
        parts.append(j.get("title", ""))
        parts.append(j.get("description", ""))
    for s in skills:
        parts.append(s.get("name", ""))
    for e in edu:
        parts.append(e.get("degree", ""))
        parts.append(e.get("field_of_study", ""))
    return " | ".join(p for p in parts if p).lower()


def extract_features(c: dict) -> dict:
    p = c["profile"]
    sig = c["redrob_signals"]
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    edu = c.get("education", [])

    total_career_months = sum(j.get("duration_months", 0) for j in career)

    return {
        "candidate_id": c["candidate_id"],
        "current_title": p.get("current_title", ""),
        "current_industry": p.get("current_industry", ""),
        "current_company": p.get("current_company", ""),
        "current_company_size": p.get("current_company_size", ""),
        "yoe": float(p.get("years_of_experience", 0)),
        "skills": skills,
        "n_skills": len(skills),
        "career": career,
        "n_jobs": len(career),
        "total_career_months": total_career_months,
        "avg_tenure_months": (total_career_months / len(career)) if career else 0,
        "edu_best_tier": _best_tier(edu),
        "edu_is_cs": _is_cs_field(edu),
        "location": p.get("location", ""),
        "country": p.get("country", ""),
        "open_to_work": bool(sig.get("open_to_work_flag", False)),
        "recruiter_response_rate": float(sig.get("recruiter_response_rate", 0)),
        "last_active_date": sig.get("last_active_date"),
        "interview_completion_rate": float(sig.get("interview_completion_rate", 0)),
        "profile_completeness": float(sig.get("profile_completeness_score", 0)),
        "saved_by_recruiters_30d": int(sig.get("saved_by_recruiters_30d", 0)),
        "notice_period_days": int(sig.get("notice_period_days", 180)),
        "github_activity_score": float(sig.get("github_activity_score", -1)),
        "willing_to_relocate": bool(sig.get("willing_to_relocate", False)),
        "preferred_work_mode": sig.get("preferred_work_mode", ""),
        "skill_assessment_scores": sig.get("skill_assessment_scores", {}),
        "endorsements_received": int(sig.get("endorsements_received", 0)),
        "profile_text": _flatten_text(p, career, skills, edu),
    }
