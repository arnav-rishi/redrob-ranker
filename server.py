"""FastAPI backend for the redrob-ranker demo.

Two capabilities:
  GET  /api/results  -> the precomputed real top-100 from the full 100K pool
                        (data/top100.json, ships in the repo, loads instantly).
  POST /api/rank     -> live re-rank of an uploaded candidates file
                        (.jsonl / .json), streamed and scored in-memory using
                        the exact same src/ pipeline as rank.py.

No precomputed 234MB artifacts are needed on the server: the precomputed view
reads a 30KB JSON, and live upload processes whatever the user sends.

Run locally:  uvicorn server:app --reload --port 7860
"""
import io
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.features.extract import extract_features  # noqa: E402
from src.features.lexical import fit_bm25, score_all  # noqa: E402
from src.output.reasoning import reasoning  # noqa: E402
from src.scoring.integrity import integrity_mult  # noqa: E402
from src.scoring.role_fit import classify_via_substring, role_fit  # noqa: E402
from src.scoring.scorer import (  # noqa: E402
    experience_fit, full_score, group_in_text,
    location_score, skill_trust, trajectory_diagnostics,
)

CONFIG_DIR      = REPO_ROOT / "config"
STATIC_DIR      = REPO_ROOT / "static"
TOP100_PATH     = REPO_ROOT / "data" / "top100.json"
SHORTLIST_SIZE  = 2000
MAX_UPLOAD_MB   = 75          # RAM guard for the hosted demo; the sandbox spec only needs <=100 candidates
_BUCKET_NUMERIC = {"strong": 1.0, "adjacent": 0.5, "reject": 0.0}


def _load_configs():
    jd       = yaml.safe_load((CONFIG_DIR / "jd_profile.yaml").read_text(encoding="utf-8"))
    ontology = yaml.safe_load((CONFIG_DIR / "skill_ontology.yaml").read_text(encoding="utf-8"))
    weights  = yaml.safe_load((CONFIG_DIR / "weights.yaml").read_text(encoding="utf-8"))
    jd_text  = (CONFIG_DIR / "jd_reference_text.txt").read_text(encoding="utf-8")
    return jd, ontology, weights, jd_text


JD, ONTOLOGY, WEIGHTS, JD_TEXT = _load_configs()

app = FastAPI(title="redrob-ranker", docs_url=None, redoc_url=None)


# ── Ranking helpers (mirror src/pipeline/rank.py, in-memory) ──────────────────

def _role_bucket_numeric(title):
    bucket = JD["exact_title_map"].get((title or "").strip().lower())
    if bucket is None:
        bucket = classify_via_substring((title or "").strip().lower(), JD["role_titles"])
    return _BUCKET_NUMERIC.get(bucket, 0.3)


def _skill_group_hit_count(text, must_have):
    return sum(1 for g in must_have if group_in_text(g, text, ONTOLOGY))


def _days_since(date_str):
    if not date_str:
        return 365
    try:
        return (date.today() - datetime.strptime(date_str, "%Y-%m-%d").date()).days
    except ValueError:
        return 365


def _stage0_prescore(feats, corpus_lines):
    must_have  = JD["must_have"]
    role_hit   = np.array([_role_bucket_numeric(f["current_title"]) for f in feats])
    skill_hits = np.array([_skill_group_hit_count(t, must_have) for t in corpus_lines])
    exp_band   = np.array([experience_fit(f["yoe"], JD) for f in feats])
    days_inact = np.array([_days_since(f.get("last_active_date")) for f in feats])
    recency    = 1.0 - 0.5 * (days_inact > 180)
    avail      = np.array([float(f["open_to_work"]) * f["recruiter_response_rate"] for f in feats]) * recency
    return 0.45 * role_hit + 0.25 * (skill_hits / max(len(must_have), 1)) + 0.15 * exp_band + 0.15 * avail


def _iter_records(raw_bytes):
    """Yield candidate dicts from raw uploaded bytes, accepting either JSONL
    (one object per line) or a single JSON array."""
    text = raw_bytes.decode("utf-8", errors="replace").lstrip()
    if text.startswith("["):
        yield from json.loads(text)
        return
    for line in io.StringIO(text):
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _rank(raw_bytes):
    feats, corpus_lines = [], []
    for c in _iter_records(raw_bytes):
        feat = extract_features(c)
        corpus_lines.append(feat.pop("profile_text"))
        feats.append(feat)

    n = len(feats)
    if n == 0:
        return [], 0

    if n > SHORTLIST_SIZE:
        pre = _stage0_prescore(feats, corpus_lines)
        shortlist_idx = np.argsort(pre)[::-1][:SHORTLIST_SIZE].tolist()
    else:
        shortlist_idx = list(range(n))

    retriever, jd_tokens = fit_bm25(corpus_lines, JD_TEXT)
    bm25 = score_all(retriever, jd_tokens)

    results = []
    for i in shortlist_idx:
        feat = feats[i]
        feat["profile_text"] = corpus_lines[i]
        rf        = role_fit(feat, JD, ONTOLOGY)
        im        = integrity_mult(feat)
        lex       = float(bm25[i])
        score     = full_score(feat, JD, ONTOLOGY, WEIGHTS, rf, im, lex)
        evidence  = skill_trust(feat, JD, ONTOLOGY, WEIGHTS)[1]
        traj_diag = trajectory_diagnostics(feat, JD)
        loc_score = location_score(feat, JD)
        results.append({
            "candidate_id": feat["candidate_id"],
            "final_score":  score,
            "reasoning":    reasoning(feat, evidence, traj_diag, loc_score),
        })

    top100 = sorted(results, key=lambda r: -r["final_score"])[:100]
    for r in top100:
        r["score"] = round(r.pop("final_score"), 4)
    top100.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    for i, r in enumerate(top100, 1):
        r["rank"] = i
    return top100, n


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/results")
def results():
    if not TOP100_PATH.exists():
        raise HTTPException(500, "Precomputed results not found (data/top100.json).")
    return JSONResponse(json.loads(TOP100_PATH.read_text(encoding="utf-8")))


@app.post("/api/rank")
async def rank(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not (name.endswith(".jsonl") or name.endswith(".json")):
        raise HTTPException(400, "Please upload a .jsonl or .json file.")

    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            413,
            f"File is {size_mb:.0f} MB — larger than this demo accepts. Try a smaller "
            "sample, or see the full 100,000-candidate ranking in the Top 100 tab.",
        )

    try:
        ranked, n_total = _rank(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Could not parse/rank this file: {e}")

    if n_total == 0:
        raise HTTPException(422, "No valid candidate records found in the file.")

    return {
        "source": f"Live upload: {file.filename}",
        "count": len(ranked),
        "candidates_processed": n_total,
        "results": ranked,
    }


# ── Static frontend (mounted last so /api/* wins) ─────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
