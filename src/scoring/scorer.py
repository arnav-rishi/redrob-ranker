"""Scorer: six additive fit components + multipliers/gates. Every penalty
flag was audited for real detectability before being enabled, and the
availability threshold was recalibrated against the real activity
distribution rather than guessed. role_fit and integrity_mult live in their
own files; this module covers the additive components + final wiring.
"""
from datetime import date, datetime


def in_group(skill_name, group, ontology):
    name = (skill_name or "").lower()
    return any(term.lower() == name or term.lower() in name for term in ontology.get(group, []))


def group_in_text(group, text, ontology):
    text = (text or "").lower()
    return any(term.lower() in text for term in ontology.get(group, []))


def days_since(date_str, ref_date=None):
    if not date_str:
        return 0
    ref_date = ref_date or date.today()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return 0
    return (ref_date - d).days


# --- 3A. Skill-trust (anti keyword-stuffing) --------------------------------
_PROFICIENCY_SCORE = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}


def skill_trust(feat, jd, ontology, weights):
    assess = feat.get("skill_assessment_scores", {}) or {}
    dur_cap = weights["skill_trust"]["duration_cap_months"]
    endo_cap = weights["skill_trust"]["endorsement_cap"]
    text_fallback = weights["skill_trust"]["text_corroboration_fallback"]

    score, evidence = 0.0, []
    for group in jd["must_have"]:
        best = 0.0
        for sk in feat.get("skills", []):
            if not in_group(sk.get("name"), group, ontology):
                continue
            prof = _PROFICIENCY_SCORE[sk["proficiency"]]
            dur = min(sk.get("duration_months", 0) / dur_cap, 1.0)
            endo = min(sk.get("endorsements", 0) / endo_cap, 1.0)
            asmt = assess.get(sk.get("name"))
            trust = 0.5 * prof * dur + 0.2 * endo + (0.3 * asmt / 100 if asmt is not None else 0.15 * prof * dur)
            best = max(best, trust)
        if best == 0 and group_in_text(group, feat.get("profile_text", ""), ontology):
            best = text_fallback
        score += best
        if best > 0.4:
            evidence.append(group)
    return score / len(jd["must_have"]), evidence


# --- 3B. Experience fit -----------------------------------------------------
def experience_fit(yoe, jd):
    e = jd["experience"]
    if e["ideal_min"] <= yoe <= e["ideal_max"]:
        return 1.0
    if e["acceptable_min"] <= yoe <= e["acceptable_max"]:
        return 0.85
    if yoe < e["hard_floor"]:
        return 0.15
    return max(0.3, 1.0 - 0.08 * min(abs(yoe - 7), 8))


# --- 3C. Trajectory score (extended with 2 newly-grounded penalties) -------
def _is_cv_speech_only(feat, ontology):
    skill_names = {s["name"].lower() for s in feat.get("skills", [])}
    off_hits = skill_names & {t.lower() for t in ontology["off_target"]}
    nlp_ir_terms = {t.lower() for g in ("nlp", "retrieval_embeddings", "vector_search", "ranking_recsys")
                    for t in ontology[g]}
    nlp_hits = skill_names & nlp_ir_terms
    return len(off_hits) >= 2 and len(nlp_hits) == 0


def _is_langchain_only_recent(feat):
    deep_ml = {"python", "pytorch", "tensorflow", "nlp", "machine learning", "deep learning", "scikit-learn"}
    lc = next((s for s in feat.get("skills", []) if s["name"] == "LangChain"), None)
    if lc is None or lc.get("duration_months", 0) >= 12:
        return False
    return not any(s["name"].lower() in deep_ml and s.get("duration_months", 0) >= 24
                   for s in feat.get("skills", []))


def trajectory_score(feat, jd, ontology, weights):
    p = weights["trajectory_penalties"]
    firms = jd["consulting_firms"]
    companies = [(j.get("company") or "").lower() for j in feat.get("career", [])]
    consulting_flags = [any(f in co for f in firms) for co in companies]
    s = 0.85

    penalize = jd["penalize"]
    if penalize.get("consulting_only") and consulting_flags and all(consulting_flags):
        s -= p["consulting_full_career"]
    elif penalize.get("consulting_only") and consulting_flags and consulting_flags[0]:
        s -= p["consulting_current_only"]

    if penalize.get("title_chasing") and feat["avg_tenure_months"] < 18 and feat["n_jobs"] >= 3:
        s -= p["title_chasing"]

    if penalize.get("cv_speech_robotics_only") and _is_cv_speech_only(feat, ontology):
        s -= p["cv_speech_robotics_only"]

    if penalize.get("langchain_only_recent") and _is_langchain_only_recent(feat):
        s -= p["langchain_only_recent"]

    prod_months = sum(j["duration_months"] for j, c in zip(feat.get("career", []), consulting_flags) if not c)
    s += p["product_company_bonus_cap"] * min(prod_months / 48.0, 1.0)
    return max(0.0, min(s, 1.0))


def trajectory_diagnostics(feat, jd):
    """Two values reasoning.py needs (Step 6) that trajectory_score computes
    internally but doesn't return. Deliberately a small separate function
    that recomputes these cheaply, rather than changing trajectory_score's
    return signature and risking Step 5's already-passing tests."""
    firms = jd["consulting_firms"]
    companies = [(j.get("company") or "").lower() for j in feat.get("career", [])]
    consulting_flags = [any(f in co for f in firms) for co in companies]
    prod_months = sum(j["duration_months"] for j, c in zip(feat.get("career", []), consulting_flags) if not c)
    return {
        "consulting_only": bool(consulting_flags) and all(consulting_flags),
        "product_months": prod_months,
    }


# --- 3D/3E. Education + Location --------------------------------------------
def edu_score(feat):
    return 0.6 * feat["edu_best_tier"] + 0.4 * feat["edu_is_cs"]


def location_score(feat, jd):
    loc = ((feat.get("location") or "") + " " + (feat.get("country") or "")).lower()
    if any(b in loc for b in jd["location"]["best"]):
        return 1.0
    if any(g in loc for g in jd["location"]["good"]):
        return 0.85
    if jd["location"]["in_country"] in loc:
        return 0.7
    return 0.55 if feat.get("willing_to_relocate") else 0.35


# --- 3F. Availability multiplier (NOT additive) -----------------------------
def availability_mult(feat, weights, ref_date=None):
    a = weights["availability"]
    m = 1.0
    m *= 1.05 if feat.get("open_to_work") else 0.85
    m *= 0.5 + 0.5 * feat["recruiter_response_rate"]
    m *= 0.7 + 0.3 * feat["interview_completion_rate"]
    m *= 0.85 + 0.15 * min(feat["profile_completeness"] / 100, 1)
    m *= 0.92 + 0.08 * min(feat["saved_by_recruiters_30d"] / 10, 1)
    github = feat.get("github_activity_score", -1)
    if github >= 0:
        m *= 1.0 + 0.06 * min(github, 1.0)
    if days_since(feat.get("last_active_date"), ref_date) > a["inactive_days_threshold"]:
        m *= a["inactive_penalty_mult"]
    if feat["notice_period_days"] <= a["notice_period_ideal_days"]:
        m *= 1.03
    elif feat["notice_period_days"] > a["notice_period_long_days"]:
        m *= 0.92
    return min(m, 1.1)


# --- Final score wiring ------------------------------------------------------
def full_score(feat, jd, ontology, weights, role_fit_gate, integrity_mult_value, lexical_sim_value):
    w = weights["weights"]
    fit = (w["skill"] * skill_trust(feat, jd, ontology, weights)[0]
         + w["traj"] * trajectory_score(feat, jd, ontology, weights)
         + w["exp"] * experience_fit(feat["yoe"], jd)
         + w["lexical"] * lexical_sim_value
         + w["loc"] * location_score(feat, jd)
         + w["edu"] * edu_score(feat))
    return fit * role_fit_gate * availability_mult(feat, weights) * integrity_mult_value
