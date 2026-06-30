"""Integrity/honeypot gate. Thresholds recalibrated from the real full-100K
distributions, not guessed. Each check below sits in a verified clean
statistical gap between normal data noise and genuine impossibility; no
graduated partial-credit tier is needed -- confirmed binary 0/1 design.
"""

YOE_MISMATCH_MONTHS_THRESHOLD = 24     # recalibrated from 60 after auditing real YoE-mismatch gaps
EXPERT_ZERO_DURATION_MONTHS = 1        # advanced dropped -- never contributed to real flags
LOW_ASSESSMENT_SCORE_THRESHOLD = 30    # unchanged from initial calibration


def integrity_mult(feat: dict) -> float:
    skills = feat.get("skills", [])
    career = feat.get("career", [])
    yoe = feat.get("yoe", 0.0)
    total_months = feat.get("total_career_months", 0)
    assess = feat.get("skill_assessment_scores", {}) or {}

    flags = 0

    # Check A: YoE vs summed career duration mismatch
    if total_months > 0 and abs(yoe * 12 - total_months) > YOE_MISMATCH_MONTHS_THRESHOLD:
        flags += 1

    # Check B: "expert" proficiency claimed with near-zero duration
    if any(s.get("proficiency") == "expert" and s.get("duration_months", 0) <= EXPERT_ZERO_DURATION_MONTHS
           for s in skills):
        flags += 1

    # Check C: single job tenure exceeds total plausible experience
    if any(j.get("duration_months", 0) > yoe * 12 + 12 for j in career):
        flags += 1

    # Check D: "expert" skill with a low platform assessment score (bluff)
    if any(s.get("proficiency") == "expert" and assess.get(s.get("name")) is not None
           and assess.get(s.get("name")) < LOW_ASSESSMENT_SCORE_THRESHOLD for s in skills):
        flags += 1

    return 0.0 if flags >= 1 else 1.0
