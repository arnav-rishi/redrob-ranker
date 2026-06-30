"""Phase A (offline, unbounded): stream candidates.jsonl -> extract_features ->
artifacts/features.parquet + artifacts/profile_corpus.txt.

No 5-minute budget applies here -- that's Phase B (src/pipeline/rank.py).
This just needs to finish in reasonable time and stay under the ~4GB RAM
target.

Usage (from repo root):
    python -m src.pipeline.precompute
    python -m src.pipeline.precompute --candidates ../candidates.jsonl --out artifacts
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

from src.features.extract import extract_features  # noqa: E402
from src.features.lexical import fit_tfidf  # noqa: E402
from src.ingestion.stream import iter_candidates  # noqa: E402


def run(candidates_path, out_dir, jd_text_path=None, weights_path=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    rows = []
    corpus_lines = []
    n = 0
    for c in iter_candidates(candidates_path):
        feat = extract_features(c)
        corpus_lines.append(feat.pop("profile_text"))
        # Nested skills/career/assessment dicts don't fit as flat parquet columns
        # -- serialize to JSON strings instead of maintaining a second lookup
        # artifact. Per-candidate payload is small (~1-2KB). Without this,
        # pyarrow auto-converts dict columns into a
        # struct with the UNION of every key seen across all 100K rows, padding
        # each row with null for keys it doesn't have -- functionally harmless
        # for .get() lookups but massively bloats the file (verified: skipping
        # this for skill_assessment_scores alone made every row carry ~54 keys
        # instead of the ~4 it actually has).
        feat["skills_json"] = json.dumps(feat.pop("skills"))
        feat["career_json"] = json.dumps(feat.pop("career"))
        feat["skill_assessment_scores_json"] = json.dumps(feat.pop("skill_assessment_scores"))
        rows.append(feat)
        n += 1

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "features.parquet", index=False)
    (out_dir / "profile_corpus.txt").write_text("\n".join(corpus_lines), encoding="utf-8")

    elapsed = time.time() - start
    print(f"Processed {n} candidates in {elapsed:.1f}s")
    print(f"Wrote {out_dir / 'features.parquet'} ({len(df.columns)} columns)")
    print(f"Wrote {out_dir / 'profile_corpus.txt'} ({len(corpus_lines)} lines)")

    # --- Module 6: fit TF-IDF on the full corpus + JD reference text -------
    tfidf_start = time.time()
    weights = yaml.safe_load(Path(weights_path).read_text(encoding="utf-8"))
    jd_text = Path(jd_text_path).read_text(encoding="utf-8")
    lex_cfg = weights["lexical"]
    _, candidate_matrix, jd_vec = fit_tfidf(
        corpus_lines, jd_text,
        max_features=lex_cfg["max_features"], ngram_range=lex_cfg["ngram_range"],
    )
    sp.save_npz(out_dir / "tfidf_index.npz", candidate_matrix)
    sp.save_npz(out_dir / "tfidf_jd_vector.npz", jd_vec)
    print(f"Fit TF-IDF in {time.time() - tfidf_start:.1f}s "
          f"-> {candidate_matrix.shape[0]} x {candidate_matrix.shape[1]} matrix")
    print(f"Wrote {out_dir / 'tfidf_index.npz'}")
    print(f"Wrote {out_dir / 'tfidf_jd_vector.npz'}")

    try:
        import psutil
        mem = psutil.Process().memory_info()
        # On Windows, peak_wset is the true peak working-set size for this
        # process's lifetime; plain rss only reflects memory at the time of
        # this call, which understates the peak during the streaming loop.
        peak_bytes = getattr(mem, "peak_wset", mem.rss)
        peak_gb = peak_bytes / (1024 ** 3)
        print(f"Peak RAM (process lifetime): {peak_gb:.2f} GB (target < 4 GB)")
    except ImportError:
        print("psutil not installed -- skipping RAM report "
              "(pip install psutil to enable)")


def main():
    parser = argparse.ArgumentParser(description="Phase A precompute: build features.parquet.")
    parser.add_argument("--candidates", default="../candidates.jsonl",
                        help="Path to candidates.jsonl, relative to repo root.")
    parser.add_argument("--out", default="artifacts",
                        help="Output directory for artifacts, relative to repo root.")
    parser.add_argument("--jd-text", default="config/jd_reference_text.txt",
                        help="Path to JD reference text, relative to repo root.")
    parser.add_argument("--weights", default="config/weights.yaml",
                        help="Path to weights.yaml, relative to repo root.")
    args = parser.parse_args()

    candidates_path = (REPO_ROOT / args.candidates).resolve()
    out_dir = (REPO_ROOT / args.out).resolve()
    jd_text_path = (REPO_ROOT / args.jd_text).resolve()
    weights_path = (REPO_ROOT / args.weights).resolve()
    run(candidates_path, out_dir, jd_text_path, weights_path)


if __name__ == "__main__":
    main()
