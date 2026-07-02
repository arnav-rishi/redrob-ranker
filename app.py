"""Hosted demo app -- NOT the real ranking entrypoint.

Accepts a small candidate sample (<=100 rows) and runs it through the same
scoring modules `rank.py` uses, minus the Stage-0 funnel (pointless at this
scale) and the parquet/artifact round-trip (nothing to precompute for a
one-off upload). The real, full-100K-pool ranking step is `rank.py` -- see
README.md.

Run locally with: streamlit run app.py
"""
import csv
import io
import json
import sys
import traceback
from pathlib import Path

# Pin repo root first in sys.path so `import src` always resolves to
# src/ inside this repo and not to Streamlit Cloud's /mount/src/ mount point.
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
    from src.scoring.role_fit import role_fit
    from src.scoring.scorer import (
        full_score, location_score, skill_trust, trajectory_diagnostics,
    )
    _IMPORT_ERROR = None
except Exception as _e:
    _IMPORT_ERROR = _e

REPO_ROOT = _REPO_ROOT
CONFIG_DIR = REPO_ROOT / "config"
SAMPLE_PATH = REPO_ROOT / "sample_candidates.json"
MAX_ROWS = 100


@st.cache_resource
def load_configs():
    jd = yaml.safe_load((CONFIG_DIR / "jd_profile.yaml").read_text(encoding="utf-8"))
    ontology = yaml.safe_load((CONFIG_DIR / "skill_ontology.yaml").read_text(encoding="utf-8"))
    weights = yaml.safe_load((CONFIG_DIR / "weights.yaml").read_text(encoding="utf-8"))
    jd_text = (CONFIG_DIR / "jd_reference_text.txt").read_text(encoding="utf-8")
    return jd, ontology, weights, jd_text


def _parse_candidates(raw_text: str) -> list:
    """Accept either a JSON array (like sample_candidates.json) or
    newline-delimited JSON (like candidates.jsonl)."""
    stripped = raw_text.strip()
    if stripped.startswith("["):
        return json.loads(stripped)
    return [json.loads(line) for line in stripped.splitlines() if line.strip()]


def score_candidates(candidates: list, jd: dict, ontology: dict, weights: dict, jd_text: str) -> list:
    feats = [extract_features(c) for c in candidates]
    corpus_lines = [f["profile_text"] for f in feats]
    retriever, jd_tokens = fit_bm25(corpus_lines, jd_text)
    sims = score_all(retriever, jd_tokens)

    results = []
    for i, feat in enumerate(feats):
        rf = role_fit(feat, jd, ontology)
        im = integrity_mult(feat)
        lex = float(sims[i])
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


def main():
    st.set_page_config(page_title="redrob-ranker sandbox", layout="wide")
    st.title("redrob-ranker -- sandbox demo")

    if _IMPORT_ERROR is not None:
        st.error("Import error on startup -- see details below.")
        st.code(traceback.format_exc())
        st.stop()

    try:
        jd, ontology, weights, jd_text = load_configs()
    except Exception as e:
        st.error(f"Failed to load config files: {e}")
        st.code(traceback.format_exc())
        st.stop()

    uploaded = st.file_uploader(
        "Upload a small candidates file (.jsonl or .json, ≤100 rows)",
        type=["jsonl", "json"],
    )

    if uploaded is not None:
        candidates = _parse_candidates(uploaded.read().decode("utf-8"))
        source_label = uploaded.name
    else:
        st.info(f"No file uploaded -- using the bundled sample ({SAMPLE_PATH.name}, 50 candidates).")
        candidates = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        source_label = SAMPLE_PATH.name

    if len(candidates) > MAX_ROWS:
        st.warning(f"{source_label} has {len(candidates)} rows -- truncating to the first {MAX_ROWS}.")
        candidates = candidates[:MAX_ROWS]

    st.write(f"Loaded **{len(candidates)}** candidates from `{source_label}`.")

    if st.button("Run ranking", type="primary"):
        try:
            with st.spinner(f"Scoring {len(candidates)} candidates..."):
                results = score_candidates(candidates, jd, ontology, weights, jd_text)
                top_n = min(100, len(results))
                ranked = sorted(results, key=lambda r: -r["final_score"])[:top_n]
                for r in ranked:
                    r["score_out"] = round(r["final_score"], 4)
                ranked.sort(key=lambda r: (-r["score_out"], r["candidate_id"]))
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["candidate_id", "rank", "score", "reasoning"])
                for i, r in enumerate(ranked, 1):
                    w.writerow([r["candidate_id"], i, f"{r['score_out']:.4f}", r["reasoning"]])
                csv_bytes = buf.getvalue().encode("utf-8")

            st.success(f"Ranked {len(results)} candidates.")
            st.dataframe(
                [{"rank": i + 1, **r} for i, r in enumerate(ranked)],
                use_container_width=True,
            )
            st.download_button(
                "Download ranked CSV",
                csv_bytes,
                file_name="demo_submission.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Scoring failed: {e}")
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
