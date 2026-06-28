"""Module 10: output writer. Implements Fix A (tie-break rounding), agreed
back at the very start of this project. See docs/STEP6_SPEC.md S3.
"""
import csv

SCORE_DP = 4


def write_submission(ranked: list, path):
    """ranked: list of dicts with at least candidate_id, final_score,
    reasoning. Caller is responsible for ranked having exactly 100 entries
    -- this just rounds, sorts, and writes."""
    for r in ranked:
        r["score_out"] = round(r["final_score"], SCORE_DP)
    # Fix A: sort/tie-break on the SAME rounded value the validator checks,
    # not full precision -- otherwise two candidates can print identical
    # 4-decimal scores while having sorted in a different (full-precision)
    # order, which validate_submission.py's "equal score -> candidate_id
    # ascending" check would then fail.
    ranked.sort(key=lambda r: (-r["score_out"], r["candidate_id"]))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, r in enumerate(ranked[:100], 1):
            w.writerow([r["candidate_id"], i, f"{r['score_out']:.4f}", r["reasoning"]])
