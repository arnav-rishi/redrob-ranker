"""Phase B (<=5 min, <=16GB RAM, CPU only, no network): the actual ranking
step. Loads precomputed artifacts (Step 2/5), runs the two-stage funnel
(Module 8), scores the shortlist, generates reasoning, writes the CSV.
See docs/STEP6_SPEC.md.

Fails fast if artifacts/precompute hasn't been run yet (S4 of the spec) --
this script is the one that gets timed for the 5-minute budget, so it must
never silently trigger the unbounded Phase A precompute step itself.

Usage (from repo root, after running precompute.py once):
    python rank.py --candidates ../candidates.jsonl --out submission.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402

from src.features.lexical import lexical_sim  # noqa: E402
from src.output.reasoning import reasoning  # noqa: E402
from src.output.writer import write_submission  # noqa: E402
from src.scoring.integrity import integrity_mult  # noqa: E402
from src.scoring.role_fit import classify_via_substring, role_fit  # noqa: E402
from src.scoring.scorer import (  # noqa: E402
    edu_score, experience_fit, full_score, group_in_text, location_score,
    skill_trust, trajectory_diagnostics, trajectory_score,
)

DEFAULT_SHORTLIST_SIZE = 2000
_BUCKET_NUMERIC = {"strong": 1.0, "adjacent": 0.5, "reject": 0.0}


def _require_artifacts(artifacts_dir: Path):
    required = ["features.parquet", "profile_corpus.txt", "tfidf_index.npz", "tfidf_jd_vector.npz"]
    missing = [f for f in required if not (artifacts_dir / f).exists()]
    if missing:
        print(f"ERROR: missing precomputed artifacts in {artifacts_dir}: {missing}", file=sys.stderr)
        print("Run precompute.py first (Phase A, unbounded time):", file=sys.stderr)
        print("    python -m src.pipeline.precompute --candidates <path> --out artifacts", file=sys.stderr)
        sys.exit(1)


def _load_configs(config_dir: Path):
    jd = yaml.safe_load((config_dir / "jd_profile.yaml").read_text(encoding="utf-8"))
    ontology = yaml.safe_load((config_dir / "skill_ontology.yaml").read_text(encoding="utf-8"))
    weights = yaml.safe_load((config_dir / "weights.yaml").read_text(encoding="utf-8"))
    return jd, ontology, weights


def _row_to_feat(row: pd.Series, profile_text: str) -> dict:
    """Reverse of precompute.py's serialization: rebuild the extract_features
    dict shape the scorer functions expect, reattaching profile_text (which
    isn't a parquet column -- it lives in profile_corpus.txt, aligned by row
    index) and deserializing the JSON-string columns back into lists/dicts."""
    return {
        "candidate_id": row["candidate_id"],
        "current_title": row["current_title"],
        "current_industry": row["current_industry"],
        "current_company": row["current_company"],
        "current_company_size": row["current_company_size"],
        "yoe": row["yoe"],
        "skills": json.loads(row["skills_json"]),
        "n_skills": row["n_skills"],
        "career": json.loads(row["career_json"]),
        "n_jobs": row["n_jobs"],
        "total_career_months": row["total_career_months"],
        "avg_tenure_months": row["avg_tenure_months"],
        "edu_best_tier": row["edu_best_tier"],
        "edu_is_cs": row["edu_is_cs"],
        "location": row["location"],
        "country": row["country"],
        "open_to_work": row["open_to_work"],
        "recruiter_response_rate": row["recruiter_response_rate"],
        "last_active_date": row["last_active_date"],
        "interview_completion_rate": row["interview_completion_rate"],
        "profile_completeness": row["profile_completeness"],
        "saved_by_recruiters_30d": row["saved_by_recruiters_30d"],
        "notice_period_days": row["notice_period_days"],
        "github_activity_score": row["github_activity_score"],
        "willing_to_relocate": row["willing_to_relocate"],
        "preferred_work_mode": row["preferred_work_mode"],
        "skill_assessment_scores": json.loads(row["skill_assessment_scores_json"]),
        "endorsements_received": row["endorsements_received"],
        "profile_text": profile_text,
    }


def _role_bucket_numeric(title: str, jd: dict) -> float:
    bucket = jd["exact_title_map"].get((title or "").strip().lower())
    if bucket is None:
        bucket = classify_via_substring((title or "").strip().lower(), jd["role_titles"])
    return _BUCKET_NUMERIC.get(bucket, 0.3)  # unmapped/unknown title: small nonzero credit


def _skill_group_hit_count(text: str, must_have_groups: list, ontology: dict) -> int:
    return sum(1 for g in must_have_groups if group_in_text(g, text, ontology))


def stage0_prescore(df: pd.DataFrame, corpus_lines: list, jd: dict, ontology: dict) -> pd.Series:
    role_hit = df["current_title"].apply(lambda t: _role_bucket_numeric(t, jd))
    must_have = jd["must_have"]
    skill_hits = pd.Series(
        [_skill_group_hit_count(text, must_have, ontology) for text in corpus_lines],
        index=df.index,
    )
    # Reuse the real (already tested) experience_fit graduated curve instead of
    # a binary in/out-of-band check -- a binary check was found to drop 96/1047
    # strong-titled candidates from the shortlist by a razor-thin margin
    # (mean 0.014 below cutoff) purely because they were 1-2 years under the
    # acceptable band, even though the JD explicitly says "we'll seriously
    # consider candidates outside the band if other signals are strong."
    exp_band = df["yoe"].apply(lambda yoe: experience_fit(yoe, jd))
    availability_cheap = df["open_to_work"].astype(float) * df["recruiter_response_rate"]

    return (0.45 * role_hit
            + 0.25 * (skill_hits / len(must_have))
            + 0.15 * exp_band
            + 0.15 * availability_cheap)


def stage1_full_score(shortlist_df, corpus_lines, candidate_matrix, jd_vec, jd, ontology, weights):
    sim_values = lexical_sim(candidate_matrix, jd_vec, idx=shortlist_df.index.to_numpy())
    results = []
    for pos, (idx, row) in enumerate(shortlist_df.iterrows()):
        feat = _row_to_feat(row, corpus_lines[idx])
        rf = role_fit(feat, jd, ontology)
        im = integrity_mult(feat)
        lex = float(sim_values[pos])
        score = full_score(feat, jd, ontology, weights, rf, im, lex)
        evidence = skill_trust(feat, jd, ontology, weights)[1]
        traj_diag = trajectory_diagnostics(feat, jd)
        loc_score = location_score(feat, jd)
        results.append({
            "candidate_id": feat["candidate_id"],
            "final_score": score,
            "reasoning": reasoning(feat, evidence, traj_diag, loc_score),
        })
    return results


def run(candidates_path, out_path, artifacts_dir, config_dir, shortlist_size=DEFAULT_SHORTLIST_SIZE):
    start = time.time()
    artifacts_dir, config_dir = Path(artifacts_dir), Path(config_dir)
    _require_artifacts(artifacts_dir)
    jd, ontology, weights = _load_configs(config_dir)

    df = pd.read_parquet(artifacts_dir / "features.parquet")
    corpus_lines = (artifacts_dir / "profile_corpus.txt").read_text(encoding="utf-8").split("\n")
    candidate_matrix = sp.load_npz(artifacts_dir / "tfidf_index.npz")
    jd_vec = sp.load_npz(artifacts_dir / "tfidf_jd_vector.npz")

    pre = stage0_prescore(df, corpus_lines, jd, ontology)
    shortlist_df = df.loc[pre.nlargest(shortlist_size).index]
    print(f"Stage 0 done in {time.time() - start:.1f}s -- shortlisted {len(shortlist_df)} of {len(df)}")

    stage1_start = time.time()
    scored = stage1_full_score(shortlist_df, corpus_lines, candidate_matrix, jd_vec, jd, ontology, weights)
    print(f"Stage 1 done in {time.time() - stage1_start:.1f}s")

    top100 = sorted(scored, key=lambda r: -r["final_score"])[:100]
    write_submission(top100, out_path)

    elapsed = time.time() - start
    print(f"Wrote {out_path} -- total runtime {elapsed:.1f}s (budget: 300s)")
    if elapsed > 300:
        print("WARNING: exceeded the 5-minute budget!", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Phase B: produce the ranked submission CSV.")
    parser.add_argument("--candidates", default="../candidates.jsonl",
                        help="Path to candidates.jsonl (kept for CLI compatibility with the "
                             "documented reproduce command; ranking itself reads precomputed "
                             "artifacts, not this file directly).")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path, relative to repo root.")
    parser.add_argument("--artifacts", default="artifacts",
                        help="Precomputed artifacts directory, relative to repo root.")
    parser.add_argument("--config-dir", default="config",
                        help="Config directory, relative to repo root.")
    parser.add_argument("--shortlist-size", type=int, default=DEFAULT_SHORTLIST_SIZE)
    args = parser.parse_args()

    out_path = (REPO_ROOT / args.out).resolve()
    artifacts_dir = (REPO_ROOT / args.artifacts).resolve()
    config_dir = (REPO_ROOT / args.config_dir).resolve()
    run(args.candidates, out_path, artifacts_dir, config_dir, args.shortlist_size)


if __name__ == "__main__":
    main()
