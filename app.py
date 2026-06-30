"""Hosted demo app -- NOT the real ranking entrypoint.

Accepts a small candidate sample (<=100 rows) and runs it through the same
scoring modules `rank.py` uses, minus the Stage-0 funnel (pointless at this
scale) and the parquet/artifact round-trip (nothing to precompute for a
one-off upload). The real, full-100K-pool ranking step is `rank.py` -- see
README.md.

Run locally with: streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st
import yaml

from src.features.extract import extract_features
from src.features.lexical import fit_tfidf, lexical_sim
from src.output.reasoning import reasoning
from src.output.writer import write_submission
from src.scoring.integrity import integrity_mult
from src.scoring.role_fit import role_fit
from src.scoring.scorer import (
    full_score, location_score, skill_trust, trajectory_diagnostics,
)

REPO_ROOT = Path(__file__).resolve().parent
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
    _, candidate_matrix, jd_vec = fit_tfidf(
        corpus_lines, jd_text,
        max_features=weights["lexical"]["max_features"],
        ngram_range=tuple(weights["lexical"]["ngram_range"]),
    )
    sims = lexical_sim(candidate_matrix, jd_vec)

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

    jd, ontology, weights, jd_text = load_configs()

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
        with st.spinner(f"Scoring {len(candidates)} candidates..."):
            results = score_candidates(candidates, jd, ontology, weights, jd_text)
            top_n = min(100, len(results))
            ranked = sorted(results, key=lambda r: -r["final_score"])[:top_n]
            out_path = REPO_ROOT / "demo_output.csv"
            write_submission(ranked, out_path)

        st.success(f"Ranked {len(results)} candidates.")
        st.dataframe(
            [{"rank": i + 1, **r} for i, r in enumerate(ranked)],
            use_container_width=True,
        )
        st.download_button(
            "Download ranked CSV",
            out_path.read_bytes(),
            file_name="demo_submission.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
