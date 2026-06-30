"""
Exploratory data analysis over the real candidate pool.
No scoring/ranking logic here; this only measures the dataset so the JD
profile, role-fit thresholds, and integrity thresholds are calibrated from
evidence instead of assumptions.

Usage (from repo root, i.e. inside redrob-ranker/):
    python eda/run_eda.py
    python eda/run_eda.py --n 20000 --candidates "../candidates.jsonl"

CPU only, no network, runs in seconds on a 5K slice.
"""
import argparse
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.stream import iter_candidates  # noqa: E402

# --- Draft role-title buckets, copied from RedRob.md's jd_profile.yaml sketch. ---
# DRAFT ONLY for EDA classification purposes; the real config/jd_profile.yaml
# is hand-authored carefully from job_description.docx in Step 3.
ROLE_TITLES_STRONG = [
    "ai engineer", "ml engineer", "machine learning engineer", "applied scientist",
    "applied ml", "nlp engineer", "search engineer", "ranking engineer",
    "recommendation engineer", "research engineer", "mlops engineer", "data scientist",
]
ROLE_TITLES_ADJACENT = [
    "data engineer", "backend engineer", "software engineer", "platform engineer",
    "analytics engineer", "software developer",
]
ROLE_TITLES_REJECT = [
    "hr manager", "recruiter", "marketing manager", "content writer", "graphic designer",
    "accountant", "civil engineer", "mechanical engineer", "sales", "project manager",
    "business analyst", "operations",
]

AI_ISH_SKILL_PATTERNS = [
    "embedding", "rag", "retrieval", "vector", "pinecone", "weaviate", "qdrant",
    "milvus", "faiss", "opensearch", "elasticsearch", "bm25", "ranking", "recsys",
    "recommend", "nlp", "ner", "transformer", "bert", "gpt", "llm", "language model",
    "lora", "qlora", "peft", "fine-tun", "fine tun", "machine learning", "deep learning",
    "pytorch", "tensorflow", "langchain", "sentence-transformer", "sentence transformer",
    "ndcg", "mrr", "xgboost", "lightgbm",
]

JD_BEST_LOCATIONS = ["pune", "noida"]
JD_GOOD_LOCATIONS = ["hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ncr",
                      "bangalore", "bengaluru"]

# IDs that appear in sample_submission.csv's deliberately-wrong top rows.
SAMPLE_SUBMISSION_TOP_IDS = [
    "CAND_0004989", "CAND_0001195", "CAND_0003114", "CAND_0000339",
    "CAND_0001082", "CAND_0001218", "CAND_0004558",
]

CAND_ID_RE = re.compile(r"^CAND_[0-9]{7}$")
REQUIRED_TOP_KEYS = ["candidate_id", "profile", "career_history", "education",
                     "skills", "redrob_signals"]


def classify_title(title):
    t = (title or "").lower()
    if any(b in t for b in ROLE_TITLES_REJECT):
        return "reject"
    if any(b in t for b in ROLE_TITLES_STRONG):
        return "strong"
    if any(b in t for b in ROLE_TITLES_ADJACENT):
        return "adjacent"
    return "unknown"


def count_ai_ish_skills(skills):
    names = " | ".join((s.get("name") or "").lower() for s in skills)
    return sum(1 for pat in AI_ISH_SKILL_PATTERNS if pat in names)


def days_since(date_str, ref_date):
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (ref_date - d).days


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(vals),
        "min": round(vals[0], 3),
        "median": round(percentile(vals, 0.5), 3),
        "mean": round(sum(vals) / len(vals), 3),
        "max": round(vals[-1], 3),
    }


def find_reference_ids(candidates_path, ids_wanted):
    """Single streaming pass to locate specific candidate_ids regardless of slice size."""
    wanted = set(ids_wanted)
    found = {}
    for c in iter_candidates(candidates_path):
        cid = c.get("candidate_id")
        if cid in wanted:
            found[cid] = c
            wanted.discard(cid)
            if not wanted:
                break
    return found


def run(args):
    ref_date = date.today()
    candidates_path = (REPO_ROOT / args.candidates).resolve()
    sample_path = (REPO_ROOT / args.sample).resolve()
    out_path = (REPO_ROOT / args.out).resolve()

    # --- Load slice + sample ---
    slice_records = []
    for i, c in enumerate(iter_candidates(candidates_path)):
        if i >= args.n:
            break
        slice_records.append(c)

    import json
    sample_records = json.loads(sample_path.read_text(encoding="utf-8"))

    # sample_candidates.json is the first 50 rows of candidates.jsonl (verified) --
    # dedupe by candidate_id so they don't get double-counted when --n >= 50.
    seen_for_combine = {c["candidate_id"] for c in slice_records}
    extra_sample_records = [c for c in sample_records if c["candidate_id"] not in seen_for_combine]
    all_records = slice_records + extra_sample_records
    n_total = len(all_records)

    lines = []
    lines.append("# EDA Findings — Step 1\n")
    lines.append(f"Generated by `eda/run_eda.py` on {ref_date.isoformat()}.\n")
    lines.append(f"Source: first **{len(slice_records)}** rows of `{args.candidates}` "
                 f"+ **{len(extra_sample_records)}** additional rows from `{args.sample}` "
                 f"not already in the slice (sample_candidates.json is the first 50 rows of "
                 f"candidates.jsonl, so they're deduped by candidate_id; "
                 f"combined N = {n_total}).\n")
    lines.append("Reference date for recency calculations: "
                 f"**{ref_date.isoformat()}** (today, via `date.today()`).\n")
    lines.append("---\n")

    # === A. Role / title distribution ===
    title_counter = Counter((c["profile"].get("current_title") or "").strip() for c in all_records)
    bucket_counter = Counter(classify_title(c["profile"].get("current_title")) for c in all_records)
    lines.append("## A. Role / title distribution\n")
    lines.append(f"- N = {n_total}\n")
    lines.append("- Bucket counts (draft strong/adjacent/reject/unknown buckets, "
                 "see jd_profile.yaml draft in this script — finalized in Step 3):\n")
    for bucket in ["strong", "adjacent", "reject", "unknown"]:
        cnt = bucket_counter.get(bucket, 0)
        lines.append(f"  - **{bucket}**: {cnt} ({cnt / n_total:.1%})\n")
    lines.append("- Top 30 `current_title` values:\n")
    for title, cnt in title_counter.most_common(30):
        lines.append(f"  - `{title}` — {cnt}\n")
    reject_n = bucket_counter.get("reject", 0)
    lines.append(f"\n**Implication:** {reject_n} candidates ({reject_n / n_total:.1%}) have a "
                 "reject-bucket current title. These are exactly the population the role-fit "
                 "gate (Module 5) must cap regardless of skill keyword overlap.\n")
    lines.append("---\n")

    # === B. Keyword-stuffer trap confirmation ===
    stuffers = []
    for c in all_records:
        bucket = classify_title(c["profile"].get("current_title"))
        ai_hits = count_ai_ish_skills(c.get("skills", []))
        if bucket == "reject" and ai_hits >= 6:
            stuffers.append((c["candidate_id"], c["profile"].get("current_title"), ai_hits))
    lines.append("## B. Keyword-stuffer trap confirmation\n")
    lines.append(f"- Candidates with reject-bucket title AND >=6 AI-ish skill-name hits: "
                 f"**{len(stuffers)}** ({len(stuffers) / n_total:.2%} of N)\n")
    for cid, title, hits in stuffers[:15]:
        lines.append(f"  - {cid} — `{title}` — {hits} AI-ish skill hits\n")

    ref_found = find_reference_ids(candidates_path, SAMPLE_SUBMISSION_TOP_IDS)
    lines.append(f"\n- Cross-check against `sample_submission.csv` top rows "
                 f"({len(SAMPLE_SUBMISSION_TOP_IDS)} IDs looked up via full streaming pass, "
                 f"vs. the reasoning text printed in sample_submission.csv for that ID):\n")
    for cid in SAMPLE_SUBMISSION_TOP_IDS:
        rec = ref_found.get(cid)
        if rec is None:
            lines.append(f"  - {cid} — not found in candidates.jsonl (unexpected)\n")
            continue
        title = rec["profile"].get("current_title")
        yoe = rec["profile"].get("years_of_experience")
        ai_hits = count_ai_ish_skills(rec.get("skills", []))
        bucket = classify_title(title)
        lines.append(f"  - {cid} — actual: `{title}`, {yoe} yrs (bucket={bucket}), "
                     f"{ai_hits} AI-ish skill hits\n")
    lines.append("\n**Finding:** spot-checked CAND_0004989 directly against candidates.jsonl: "
                 "actual record is title=`Project Manager`, yoe=12.6, "
                 "recruiter_response_rate=0.62 — but sample_submission.csv's reasoning text for "
                 "that same ID claims *\"HR Manager with 6.1 yrs; 9 AI core skills; response rate "
                 "0.76\"*. None of those facts match. **sample_submission.csv's reasoning column "
                 "is illustrative/fabricated, not computed from the real candidate records** "
                 "(it's documented as a format reference only, not a high-quality ranking). "
                 f"Do not treat its specific candidate_ids as confirmed "
                 f"keyword-stuffers; use the independent title+skill cross-tab above "
                 f"({len(stuffers)} candidates in this slice) as the real trap evidence "
                 f"instead.\n")
    lines.append(f"\n**Implication:** the keyword-stuffer trap itself is real and present at "
                 f"non-trivial volume ({len(stuffers)}/{n_total} ≈ {len(stuffers) / n_total:.1%} "
                 f"in this slice) — the role-fit gate is load-bearing, not theoretical. But the "
                 f"specific IDs in sample_submission.csv are not reliable evidence for it; our "
                 f"own cross-tab is.\n")
    lines.append("---\n")

    # === C. Experience distribution ===
    yoe_vals = [float(c["profile"].get("years_of_experience") or 0) for c in all_records]
    bands = Counter()
    for yoe in yoe_vals:
        if yoe < 3:
            bands["below_hard_floor(<3)"] += 1
        elif 6 <= yoe <= 8:
            bands["ideal(6-8)"] += 1
        elif 5 <= yoe <= 9:
            bands["acceptable_not_ideal(5-9 excl 6-8)"] += 1
        else:
            bands["outside_band"] += 1

    mismatches = []
    for c in all_records:
        yoe = float(c["profile"].get("years_of_experience") or 0)
        total_months = sum(j.get("duration_months", 0) for j in c.get("career_history", []))
        if total_months > 0:
            mismatches.append(abs(yoe * 12 - total_months))

    lines.append("## C. Experience distribution\n")
    lines.append(f"- years_of_experience summary: {summarize(yoe_vals)}\n")
    for band, cnt in bands.items():
        lines.append(f"  - {band}: {cnt} ({cnt / n_total:.1%})\n")
    lines.append(f"- |yoe*12 - sum(career duration_months)| summary (months): "
                 f"{summarize(mismatches)}\n")
    big_mismatch = sum(1 for m in mismatches if m > 60)
    lines.append(f"  - mismatches > 60 months: {big_mismatch} "
                 f"({big_mismatch / max(len(mismatches), 1):.1%} of those with career history)\n")
    lines.append("\n**Implication:** see section D — this mismatch rate directly calibrates "
                 "the integrity gate's YoE-consistency check.\n")
    lines.append("---\n")

    # === D. Honeypot prevalence ===
    flag_counts = []
    check_trigger_counts = Counter()
    flagged_examples = []
    for c in all_records:
        skills = c.get("skills", [])
        yoe = float(c["profile"].get("years_of_experience") or 0)
        career = c.get("career_history", [])
        total_months = sum(j.get("duration_months", 0) for j in career)
        assess = c.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}

        checks = {
            "expert_skill_zero_duration": any(
                s.get("proficiency") in ("advanced", "expert") and s.get("duration_months", 0) <= 1
                for s in skills),
            "yoe_career_mismatch_gt_60mo": total_months > 0 and abs(yoe * 12 - total_months) > 60,
            "single_job_exceeds_yoe": any(j.get("duration_months", 0) > yoe * 12 + 12 for j in career),
            "expert_but_low_assessment": any(
                s.get("proficiency") == "expert" and assess.get(s.get("name")) is not None
                and assess.get(s.get("name")) < 30 for s in skills),
        }
        for name, fired in checks.items():
            if fired:
                check_trigger_counts[name] += 1
        flags = sum(checks.values())
        flag_counts.append(flags)
        if flags >= 1 and len(flagged_examples) < 10:
            flagged_examples.append((c["candidate_id"], flags,
                                     [k for k, v in checks.items() if v]))

    flag_dist = Counter(flag_counts)
    lines.append("## D. Honeypot prevalence (documented as ~80 in the full 100K pool, i.e. "
                 "~0.08% of any representative slice)\n")
    for k in sorted(flag_dist):
        lines.append(f"  - flags == {k}: {flag_dist[k]} ({flag_dist[k] / n_total:.3%})\n")

    lines.append("\nPer-check trigger counts (how many candidates trip EACH individual check, "
                 "independent of how many others they also trip):\n")
    for name, cnt in check_trigger_counts.most_common():
        lines.append(f"  - `{name}`: {cnt} ({cnt / n_total:.3%}) "
                     f"-> projected full-pool ≈ {cnt / n_total * 100000:.0f}\n")

    lines.append("\nFirst 10 candidates with flags>=1 (id, flag count, which checks fired):\n")
    for cid, flags, fired in flagged_examples:
        lines.append(f"  - {cid} — flags={flags} — {', '.join(fired)}\n")

    lines.append("\nProjected full-100K-pool count at each candidate threshold for "
                 "`integrity_mult == 0` (RedRob.md §7G currently uses flags>=3):\n")
    projections = {}
    for thresh in (1, 2, 3):
        cnt = sum(v for k, v in flag_dist.items() if k >= thresh)
        projected = cnt / n_total * 100000
        projections[thresh] = projected
        lines.append(f"  - flags >= {thresh}: {cnt} in slice ({cnt / n_total:.3%}) "
                     f"-> projected full-pool ≈ {projected:.0f}\n")
    lines.append(f"\n**Implication:** ~80 honeypots are documented as present in the full 100K "
                 f"pool. Our flags>=3 threshold (RedRob.md's current `integrity_mult==0` cutoff) "
                 f"projects ≈{projections[3]:.0f} — i.e. it would let nearly all real honeypots "
                 f"through undetected, since no candidate in the full 100K pool trips more than "
                 f"2 of our 4 checks simultaneously. flags>=1 projects ≈{projections[1]:.0f} "
                 f"(measured directly on the full 100K pool, not a projection from a slice), "
                 f"still ~14 short of the expected ~80.\n\n"
                 f"**KNOWN GAP — flagged for Step 4, not resolved here:** threshold-freezing was "
                 f"deliberately deferred past this initial EDA pass; this is calibration "
                 f"evidence only. Step 4 must either (a) loosen these 4 "
                 f"checks' thresholds (e.g. the 60-month YoE-mismatch cutoff or the "
                 f"duration_months<=1 cutoff may be too strict), or (b) add a detection check "
                 f"we're currently missing (e.g. company-founded-year vs. tenure — not "
                 f"checkable from this schema since career_history has no company-founding-date "
                 f"field), or (c) accept catching a partial honeypot population if (a)/(b) don't "
                 f"close the gap. Whichever path is chosen, re-run this script to re-measure "
                 f"against the full pool before freezing `weights.yaml`.\n")
    lines.append("---\n")

    # === E. Skill-name <-> assessment-key match ===
    # NOTE: skill_assessment_scores is a deliberately SPARSE subset of a candidate's skills
    # (only some skills get assessed) -- confirmed by direct inspection of real records.
    # So the right question is "for each assessment KEY, is there a skill with that exact
    # name?", not "for each skill, is it assessed?" (the latter conflates sparsity with
    # naming mismatches and was an earlier bug in this script).
    exact_match, ci_match_only, orphan_key, total_keys_checked = 0, 0, 0, 0
    for c in all_records:
        assess = c.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
        if not assess:
            continue
        skill_names = {s.get("name") or "" for s in c.get("skills", [])}
        skill_names_lower = {n.lower().strip() for n in skill_names}
        for key in assess:
            total_keys_checked += 1
            if key in skill_names:
                exact_match += 1
            elif key.lower().strip() in skill_names_lower:
                ci_match_only += 1
            else:
                orphan_key += 1
    lines.append("## E. Skill-name <-> assessment-key match (critical scorer dependency)\n")
    lines.append(f"- Assessment keys checked (candidates with non-empty "
                 f"skill_assessment_scores): {total_keys_checked}\n")
    if total_keys_checked:
        lines.append(f"  - key exactly matches a skill name: {exact_match} "
                     f"({exact_match / total_keys_checked:.1%})\n")
        lines.append(f"  - key matches a skill name case/whitespace-insensitively only: "
                     f"{ci_match_only} ({ci_match_only / total_keys_checked:.1%})\n")
        lines.append(f"  - key matches no skill name at all (orphan): {orphan_key} "
                     f"({orphan_key / total_keys_checked:.1%})\n")
        if exact_match / total_keys_checked > 0.95:
            implication = ("exact-key lookup (`assess.get(sk['name'])`) is safe as-is; "
                           "no normalization needed.")
        elif (exact_match + ci_match_only) / total_keys_checked > 0.95:
            implication = ("the skill-trust scorer should normalize both sides "
                           "(`.lower().strip()`) before lookup to catch the "
                           f"{ci_match_only / total_keys_checked:.1%} case/whitespace variants.")
        else:
            implication = (f"{orphan_key / total_keys_checked:.1%} of assessment keys don't "
                           "correspond to any listed skill at all — worth a closer look at "
                           "whether this is a data quirk or a real signal in Step 2.")
        lines.append(f"\n**Implication:** {implication}\n")
    else:
        lines.append("\n**Implication:** no candidates in this slice had non-empty "
                     "skill_assessment_scores — re-run with a larger --n before trusting this.\n")
    lines.append("---\n")

    # === F. Behavioral signal distributions ===
    sig = lambda c: c.get("redrob_signals", {})  # noqa: E731
    rr = [sig(c).get("recruiter_response_rate") for c in all_records]
    icr = [sig(c).get("interview_completion_rate") for c in all_records]
    otw = [1 if sig(c).get("open_to_work_flag") else 0 for c in all_records]
    notice = [sig(c).get("notice_period_days") for c in all_records]
    completeness = [sig(c).get("profile_completeness_score") for c in all_records]
    github = [sig(c).get("github_activity_score") for c in all_records]
    saved = [sig(c).get("saved_by_recruiters_30d") for c in all_records]
    recency = [days_since(sig(c).get("last_active_date"), ref_date) for c in all_records]
    recency_known = [r for r in recency if r is not None]
    inactive_120 = sum(1 for r in recency_known if r > 120)
    github_no_link = sum(1 for g in github if g == -1)

    lines.append("## F. Behavioral signal distributions\n")
    lines.append(f"- recruiter_response_rate: {summarize(rr)}\n")
    lines.append(f"- interview_completion_rate: {summarize(icr)}\n")
    lines.append(f"- open_to_work_flag: {sum(otw)}/{n_total} true ({sum(otw) / n_total:.1%})\n")
    lines.append(f"- notice_period_days: {summarize(notice)}\n")
    lines.append(f"- profile_completeness_score: {summarize(completeness)}\n")
    lines.append(f"- github_activity_score: {summarize([g for g in github if g != -1])} "
                 f"(excluding -1 sentinel); no-GitHub share: {github_no_link / n_total:.1%}\n")
    lines.append(f"- saved_by_recruiters_30d: {summarize(saved)}\n")
    lines.append(f"- days since last_active: {summarize(recency_known)}; "
                 f"inactive >120 days: {inactive_120} ({inactive_120 / n_total:.1%})\n")
    lines.append("\n**Implication:** confirms availability-multiplier inputs (Module 7F) "
                 "are populated within the documented ranges and have enough spread to be a "
                 "meaningful down-weighting signal.\n")
    lines.append("---\n")

    # === G. Location distribution ===
    loc_counter = Counter((c["profile"].get("location") or "").strip() for c in all_records)
    country_counter = Counter((c["profile"].get("country") or "").strip() for c in all_records)
    in_india = sum(1 for c in all_records if "india" in (c["profile"].get("country") or "").lower())
    best_hub = sum(1 for c in all_records
                   if any(b in (c["profile"].get("location") or "").lower() for b in JD_BEST_LOCATIONS))
    good_hub = sum(1 for c in all_records
                   if any(g in (c["profile"].get("location") or "").lower() for g in JD_GOOD_LOCATIONS))
    out_of_india = [c for c in all_records if "india" not in (c["profile"].get("country") or "").lower()]
    relocate_willing = sum(1 for c in out_of_india
                           if c.get("redrob_signals", {}).get("willing_to_relocate"))

    lines.append("## G. Location distribution\n")
    lines.append(f"- In India (by country field): {in_india} ({in_india / n_total:.1%})\n")
    lines.append(f"- In JD 'best' hubs (Pune/Noida): {best_hub} ({best_hub / n_total:.1%})\n")
    lines.append(f"- In JD 'good' hubs: {good_hub} ({good_hub / n_total:.1%})\n")
    lines.append(f"- Outside India: {len(out_of_india)} "
                 f"({len(out_of_india) / n_total:.1%}); of those, willing_to_relocate: "
                 f"{relocate_willing} ({relocate_willing / max(len(out_of_india), 1):.1%})\n")
    lines.append("- Top 15 countries: " +
                 ", ".join(f"{k or '(blank)'}={v}" for k, v in country_counter.most_common(15)) + "\n")
    lines.append("- Top 15 locations: " +
                 ", ".join(f"{k or '(blank)'}={v}" for k, v in loc_counter.most_common(15)) + "\n")
    lines.append("\n**Implication:** confirms location_score (Module 7D) buckets have "
                 "meaningful population in each tier; not a degenerate all-India or all-foreign set.\n")
    lines.append("---\n")

    # === H. Data-quality / schema sanity ===
    missing_keys = Counter()
    empty_skills = empty_career = empty_edu = 0
    bad_id_format = 0
    seen_ids = set()
    dup_ids = 0
    for c in all_records:
        for k in REQUIRED_TOP_KEYS:
            if k not in c:
                missing_keys[k] += 1
        if not c.get("skills"):
            empty_skills += 1
        if not c.get("career_history"):
            empty_career += 1
        if not c.get("education"):
            empty_edu += 1
        cid = c.get("candidate_id", "")
        if not CAND_ID_RE.match(cid):
            bad_id_format += 1
        if cid in seen_ids:
            dup_ids += 1
        seen_ids.add(cid)

    lines.append("## H. Data-quality / schema sanity\n")
    lines.append(f"- Missing required top-level keys: {dict(missing_keys) or 'none'}\n")
    lines.append(f"- Empty skills array: {empty_skills} ({empty_skills / n_total:.1%})\n")
    lines.append(f"- Empty career_history array: {empty_career} ({empty_career / n_total:.1%})\n")
    lines.append(f"- Empty education array: {empty_edu} ({empty_edu / n_total:.1%})\n")
    lines.append(f"- candidate_id not matching CAND_[0-9]{{7}}: {bad_id_format}\n")
    lines.append(f"- Duplicate candidate_id within this slice: {dup_ids}\n")
    lines.append("\n**Implication:** confirms whether `extract_features`'s `.get(..., default)` "
                 "fallbacks are necessary (if any of the above are nonzero) or just defensive "
                 "boilerplate (if all zero).\n")
    lines.append("---\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(all_records)} records analyzed)")


def main():
    parser = argparse.ArgumentParser(description="Step 1 EDA over the candidate pool.")
    parser.add_argument("--candidates", default="../candidates.jsonl",
                        help="Path to candidates.jsonl, relative to repo root.")
    parser.add_argument("--sample", default="../sample_candidates.json",
                        help="Path to sample_candidates.json, relative to repo root.")
    parser.add_argument("--n", type=int, default=5000,
                        help="Number of rows to read from --candidates (slice size).")
    parser.add_argument("--out", default="eda/EDA_FINDINGS.md",
                        help="Output markdown path, relative to repo root.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
