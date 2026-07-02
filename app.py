"""Streamlit demo -- upload candidates.jsonl and get the ranked top-100 CSV.

Mirrors rank.py exactly: streams the file to avoid loading 500MB into memory,
runs Stage-0 pre-score to shortlist ~2000, fits BM25 on the full corpus, then
full-scores only the shortlist. Run locally with: streamlit run app.py
"""
import csv
import io
import json
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402
import yaml  # noqa: E402

try:
    from src.features.extract import extract_features
    from src.features.lexical import fit_bm25, score_all
    from src.output.reasoning import reasoning
    from src.scoring.integrity import integrity_mult
    from src.scoring.role_fit import classify_via_substring, role_fit
    from src.scoring.scorer import (
        experience_fit, full_score, group_in_text,
        location_score, skill_trust, trajectory_diagnostics,
    )
    _IMPORT_ERROR = None
except Exception as _e:
    _IMPORT_ERROR = _e

REPO_ROOT      = _REPO_ROOT
CONFIG_DIR     = REPO_ROOT / "config"
SAMPLE_PATH    = REPO_ROOT / "sample_candidates.json"
SHORTLIST_SIZE = 2000
_BUCKET_NUMERIC = {"strong": 1.0, "adjacent": 0.5, "reject": 0.0}


@st.cache_resource
def load_configs():
    jd       = yaml.safe_load((CONFIG_DIR / "jd_profile.yaml").read_text(encoding="utf-8"))
    ontology = yaml.safe_load((CONFIG_DIR / "skill_ontology.yaml").read_text(encoding="utf-8"))
    weights  = yaml.safe_load((CONFIG_DIR / "weights.yaml").read_text(encoding="utf-8"))
    jd_text  = (CONFIG_DIR / "jd_reference_text.txt").read_text(encoding="utf-8")
    return jd, ontology, weights, jd_text


def _role_bucket_numeric(title, jd):
    bucket = jd["exact_title_map"].get((title or "").strip().lower())
    if bucket is None:
        bucket = classify_via_substring((title or "").strip().lower(), jd["role_titles"])
    return _BUCKET_NUMERIC.get(bucket, 0.3)


def _skill_group_hit_count(text, must_have, ontology):
    return sum(1 for g in must_have if group_in_text(g, text, ontology))


def _days_since(date_str):
    if not date_str:
        return 365
    try:
        return (date.today() - datetime.strptime(date_str, "%Y-%m-%d").date()).days
    except ValueError:
        return 365


def _stage0_prescore(feats, corpus_lines, jd, ontology):
    must_have  = jd["must_have"]
    role_hit   = np.array([_role_bucket_numeric(f["current_title"], jd) for f in feats])
    skill_hits = np.array([_skill_group_hit_count(t, must_have, ontology) for t in corpus_lines])
    exp_band   = np.array([experience_fit(f["yoe"], jd) for f in feats])
    days_inact = np.array([_days_since(f.get("last_active_date")) for f in feats])
    recency    = 1.0 - 0.5 * (days_inact > 180)
    avail      = np.array([float(f["open_to_work"]) * f["recruiter_response_rate"] for f in feats]) * recency
    return 0.45 * role_hit + 0.25 * (skill_hits / max(len(must_have), 1)) + 0.15 * exp_band + 0.15 * avail


def _iter_jsonl(uploaded_file):
    first_chunk = True
    for raw_line in uploaded_file:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if first_chunk and line.startswith("["):
            rest = line + uploaded_file.read().decode("utf-8", errors="replace")
            yield from json.loads(rest)
            return
        first_chunk = False
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def rank_pool(uploaded_file, jd, ontology, weights, jd_text, status):
    feats, corpus_lines = [], []
    status.update(label="Streaming and extracting features...")
    for c in _iter_jsonl(uploaded_file):
        feat = extract_features(c)
        corpus_lines.append(feat.pop("profile_text"))
        feats.append(feat)

    n = len(feats)
    if n == 0:
        return [], 0

    if n > SHORTLIST_SIZE:
        status.update(label=f"Stage-0: shortlisting {SHORTLIST_SIZE} from {n} candidates...")
        pre = _stage0_prescore(feats, corpus_lines, jd, ontology)
        shortlist_idx = np.argsort(pre)[::-1][:SHORTLIST_SIZE].tolist()
    else:
        shortlist_idx = list(range(n))

    status.update(label=f"Fitting BM25 on {n} profiles...")
    retriever, jd_tokens = fit_bm25(corpus_lines, jd_text)
    bm25 = score_all(retriever, jd_tokens)

    status.update(label=f"Full scoring {len(shortlist_idx)} shortlisted candidates...")
    results = []
    for i in shortlist_idx:
        feat = feats[i]
        feat["profile_text"] = corpus_lines[i]
        rf        = role_fit(feat, jd, ontology)
        im        = integrity_mult(feat)
        lex       = float(bm25[i])
        score     = full_score(feat, jd, ontology, weights, rf, im, lex)
        evidence  = skill_trust(feat, jd, ontology, weights)[1]
        traj_diag = trajectory_diagnostics(feat, jd)
        loc_score = location_score(feat, jd)
        results.append({
            "candidate_id": feat["candidate_id"],
            "final_score":  score,
            "reasoning":    reasoning(feat, evidence, traj_diag, loc_score),
        })

    return results, n


def rank_sample(jd, ontology, weights, jd_text, status):
    status.update(label="Loading bundled sample...")
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    feats  = [extract_features(c) for c in sample]
    corpus_lines = [f.pop("profile_text") for f in feats]

    status.update(label="Fitting BM25...")
    retriever, jd_tokens = fit_bm25(corpus_lines, jd_text)
    bm25 = score_all(retriever, jd_tokens)

    status.update(label="Scoring candidates...")
    results = []
    for i, feat in enumerate(feats):
        feat["profile_text"] = corpus_lines[i]
        rf        = role_fit(feat, jd, ontology)
        im        = integrity_mult(feat)
        lex       = float(bm25[i])
        score     = full_score(feat, jd, ontology, weights, rf, im, lex)
        evidence  = skill_trust(feat, jd, ontology, weights)[1]
        traj_diag = trajectory_diagnostics(feat, jd)
        loc_score = location_score(feat, jd)
        results.append({
            "candidate_id": feat["candidate_id"],
            "final_score":  score,
            "reasoning":    reasoning(feat, evidence, traj_diag, loc_score),
        })
    return results, len(sample)


def _build_csv(ranked):
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for i, r in enumerate(ranked, 1):
        w.writerow([r["candidate_id"], i, f"{r['score_out']:.4f}", r["reasoning"]])
    return buf.getvalue().encode("utf-8")


def _render_results(ranked, n_total, source):
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates processed", f"{n_total:,}")
    c2.metric("Top results shown", len(ranked))
    c3.metric("Source", source)

    st.dataframe(
        [{"rank": i + 1,
          "candidate_id": r["candidate_id"],
          "score": r["score_out"],
          "reasoning": r["reasoning"]}
         for i, r in enumerate(ranked)],
        use_container_width=True,
        height=420,
    )
    st.download_button(
        "⬇ Download top-100 CSV",
        _build_csv(ranked),
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main():
    st.set_page_config(page_title="redrob-ranker", page_icon="🏆", layout="wide")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🏆 redrob-ranker")
        st.caption("Redrob × India Runs hackathon — Senior AI Engineer ranker")
        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            "1. Upload `candidates.jsonl` (or use the bundled 50-candidate sample)\n"
            "2. Click **Run ranking**\n"
            "3. Download the top-100 CSV"
        )
        st.divider()
        st.info(
            "**Hosted on Hugging Face Spaces (CPU Basic)**\n\n"
            "Uploading large files (e.g. the full 500 MB `candidates.jsonl`) "
            "will be slow — upload speed is limited by your internet connection, "
            "not the server. For a quick demo, use the bundled sample below.",
            icon="ℹ️",
        )
        st.divider()
        st.markdown("**Pipeline**")
        st.markdown(
            "- Stage-0: role + skill + exp + recency pre-score → top 2000\n"
            "- BM25 lexical similarity on full corpus\n"
            "- Full scorer on shortlist → top 100"
        )

    # ── Import / config guard ─────────────────────────────────────────────────
    if _IMPORT_ERROR is not None:
        st.error("Import error on startup.")
        st.code("".join(traceback.format_exception(
            type(_IMPORT_ERROR), _IMPORT_ERROR, _IMPORT_ERROR.__traceback__
        )))
        st.stop()

    try:
        jd, ontology, weights, jd_text = load_configs()
    except Exception:
        st.error("Failed to load config files.")
        st.code(traceback.format_exc())
        st.stop()

    # ── Session state ─────────────────────────────────────────────────────────
    # Results are stored in session state so they persist when the download
    # button or any other widget triggers a Streamlit re-run.
    if "ranked"  not in st.session_state:
        st.session_state.ranked  = None
        st.session_state.n_total = 0
        st.session_state.source  = ""

    # ── Main UI ───────────────────────────────────────────────────────────────
    st.header("Candidate Ranking")

    uploaded = st.file_uploader(
        "Upload `candidates.jsonl` or a `.json` sample",
        type=["jsonl", "json"],
        help="JSONL recommended for large files — streamed line-by-line so the full file is never held in memory at once.",
    )

    if uploaded is not None:
        st.warning(
            f"**{uploaded.name}** selected. Large files take time to upload "
            "over your internet connection before ranking can begin.",
            icon="⏳",
        )
        source_label = uploaded.name
    else:
        st.info(f"No file uploaded — will use the bundled sample (`{SAMPLE_PATH.name}`, 50 candidates).", icon="📦")
        source_label = SAMPLE_PATH.name

    col_btn, col_clear = st.columns([1, 5])
    run_clicked   = col_btn.button("▶ Run ranking", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("✕ Clear results", use_container_width=False)

    if clear_clicked:
        st.session_state.ranked  = None
        st.session_state.n_total = 0
        st.session_state.source  = ""

    if run_clicked:
        try:
            with st.status("Running pipeline...", expanded=True) as status:
                if uploaded is not None:
                    results, n_total = rank_pool(uploaded, jd, ontology, weights, jd_text, status)
                else:
                    results, n_total = rank_sample(jd, ontology, weights, jd_text, status)

                top100 = sorted(results, key=lambda r: -r["final_score"])[:100]
                for r in top100:
                    r["score_out"] = round(r["final_score"], 4)
                top100.sort(key=lambda r: (-r["score_out"], r["candidate_id"]))
                status.update(label="Done!", state="complete")

            st.session_state.ranked  = top100
            st.session_state.n_total = n_total
            st.session_state.source  = source_label

        except Exception:
            st.error("Ranking failed.")
            st.code(traceback.format_exc())

    if st.session_state.ranked is not None:
        st.divider()
        _render_results(
            st.session_state.ranked,
            st.session_state.n_total,
            st.session_state.source,
        )


if __name__ == "__main__":
    main()
