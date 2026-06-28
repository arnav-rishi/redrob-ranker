"""Module 9: reasoning generator. Fact-grounded, varied, honest about
concerns. See docs/STEP6_SPEC.md S2.

Anti-hallucination rule: every value referenced here comes from `feat` or an
already-computed score component -- no invented employers/skills, no
external knowledge. This is exactly what Stage-4 manual review checks
(submission_spec.docx S3: "No hallucination... every claim corresponds to
something actually in the candidate's profile").
"""
from src.scoring.scorer import days_since

INACTIVE_DAYS_THRESHOLD = 180  # matches Step 5's recalibrated availability threshold


def reasoning(feat: dict, evidence: list, trajectory_diag: dict, location_score_value: float) -> str:
    pos, concerns = [], []
    pos.append(f"{feat['current_title']} with {feat['yoe']:.1f} yrs")
    if evidence:
        pos.append("strong on " + ", ".join(g.replace("_", " ") for g in evidence))
    prod_years = trajectory_diag["product_months"] // 12
    if prod_years >= 2:
        pos.append(f"{prod_years}+ yrs at product companies")

    if trajectory_diag["consulting_only"]:
        concerns.append("entire career at IT-services firms")
    if feat["recruiter_response_rate"] < 0.2:
        concerns.append(f"low recruiter response ({feat['recruiter_response_rate']:.2f})")
    if feat["notice_period_days"] > 90:
        concerns.append(f"{feat['notice_period_days']}-day notice")
    if location_score_value < 0.6:
        concerns.append("outside preferred India hubs")
    if days_since(feat["last_active_date"]) > INACTIVE_DAYS_THRESHOLD:
        concerns.append("inactive 6+ months")

    s = "; ".join(pos) + "."
    if concerns:
        s += " Concern: " + ", ".join(concerns[:2]) + "."
    return s[:300]
