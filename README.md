---
title: redrob-ranker
emoji: 🏆
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# redrob-ranker

A deterministic, rule-based candidate ranker for the Redrob × India Runs
"Intelligent Candidate Discovery & Ranking" hackathon. Given `candidates.jsonl`
(100K profiles) and a fixed Senior AI Engineer job description, it produces a
top-100 ranking with a per-candidate reasoning string. No LLM calls, no
embeddings, no GPU — everything at ranking time is hand-authored rules over
precomputed features, a curated skill ontology, and a BM25 lexical index.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.11 on Windows 11 (10 physical / 16 logical cores, 16.8GB
RAM), CPU only.

## Reproduce

Two steps: an unbounded, offline pre-computation pass followed by the
bounded ranking step:

```bash
# Phase A -- offline pre-computation (unbounded time budget).
# Extracts features, builds the BM25 index. Re-run only when candidates.jsonl
# or config/ changes. ~50-60s for the full 100K pool, ~2GB peak RAM.
python -m src.pipeline.precompute --candidates ./candidates.jsonl --out artifacts

# Phase B -- the actual ranking step (must finish in <=5 min, <=16GB RAM,
# CPU only, no network -- this is what gets timed/reproduced at Stage 3).
# ~25-30s end-to-end on the full 100K pool.
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`rank.py` fails fast with an explicit error if `artifacts/` is missing or
stale, rather than silently re-running precompute inline and burning into the
5-minute ranking budget.

## Web demo (FastAPI)

A minimal, framework-free web UI (`server.py` + `static/`) exposes the same
pipeline as an app — no Streamlit, no build step:

```bash
uvicorn server:app --reload --port 7860
# open http://localhost:7860
```

- **Top 100 (Full 100K)** — instantly shows the real precomputed ranking
  (`data/top100.json`, generated from `submission.csv`). No artifacts needed
  on the server.
- **Rank Your Own File** — upload a `.jsonl`/`.json` sample; the server runs
  the exact `src/` pipeline live (Stage-0 → BM25 → full scorer) and returns a
  downloadable top-100. (The hosted demo guards against oversized uploads; the
  sandbox spec only requires ≤100 candidates.)

### Deploy

The app is a standard persistent Python web server, so it runs on any host
that keeps a process alive (Render, Railway, Fly, HF Spaces) — **not** Vercel,
which is serverless-only.

- **Render:** push to GitHub → New + → Blueprint → pick this repo
  (`render.yaml` is read automatically).
- **HF Spaces / any Docker host:** the `Dockerfile` runs
  `uvicorn server:app` on `$PORT`.

Free tiers sleep after ~15 min idle. Set the `APP_URL` repo secret to enable
the `keep-alive` GitHub Action, or add an UptimeRobot HTTP monitor on
`<url>/api/results`, so judges never hit a cold start.

## Architecture

1. **Feature extraction** (`src/features/extract.py`, run inside
   `src/pipeline/precompute.py`) — flattens each candidate's profile, career
   history, education, and skills into a flat feature row + a free-text
   profile blob.
2. **Lexical index** (`src/features/lexical.py`) — BM25Okapi (rank-bm25,
   k1=1.5 b=0.75) over all 100K profile texts against a hand-authored JD
   reference paragraph (`config/jd_reference_text.txt`); fit once during
   precompute, reused at ranking time.
3. **Two-stage funnel** (`src/pipeline/rank.py`) — a cheap vectorized
   Stage-0 pre-score ranks all 100K candidates and shortlists ~2,000; only
   the shortlist goes through the full Stage-1 scorer. Keeps the ranking
   step well under the 5-minute budget.
4. **Role-fit gate** (`src/scoring/role_fit.py`) — multiplicative gate over
   exact + substring title matching (`config/jd_profile.yaml`); defeats
   keyword-stuffing by capping the ceiling for off-target titles regardless
   of skill density.
5. **Integrity gate** (`src/scoring/integrity.py`) — multiplicative
   honeypot detector (YoE/company-age mismatches, zero-duration "expert"
   skills, implausibly low assessment scores); forces impossible profiles to
   zero.
6. **Scorer** (`src/scoring/scorer.py`) — six additive fit components
   (skill-trust, trajectory, experience, lexical similarity, location,
   education) combined with the role-fit gate, an availability multiplier
   (behavioral signals: recruiter response, notice period, recent activity),
   and the integrity gate.
7. **Reasoning + writer** (`src/output/reasoning.py`,
   `src/output/writer.py`) — generates a fact-grounded reasoning string per
   candidate (no invented claims) and writes the final CSV with
   tie-break-safe score rounding.

Design rationale and the data audits behind every threshold live in
`docs/` (local working notes, not part of the evaluated deliverable).

## Tests

```bash
pytest
```

34 tests covering feature extraction, JD/ontology config coverage, the
role-fit gate, the integrity gate (including a regression lock on the exact
honeypot count caught on the full pool), the scorer, and the lexical index.

## Repo layout

```
config/             jd_profile.yaml, skill_ontology.yaml, weights.yaml, jd_reference_text.txt
src/
  ingestion/         format-agnostic candidates.jsonl[.gz] streaming
  features/          feature extraction, BM25 lexical index
  scoring/           role-fit gate, integrity gate, scorer
  output/            reasoning generation, CSV writer
  pipeline/          precompute.py (Phase A), rank.py (Phase B)
tests/               pytest suite
eda/                 exploratory data analysis script + findings
rank.py              thin CLI wrapper -> src/pipeline/rank.py
```
