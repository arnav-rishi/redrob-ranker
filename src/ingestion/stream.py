"""Streaming JSONL ingestion. Never materializes the full candidate list in memory.

Fix C (see STEP1_SPEC.md §5): accepts both plain .jsonl and gzipped .jsonl.gz,
since the released bundle ships uncompressed but the README documents .gz.
"""
import gzip
import io
import json


def open_candidates(path):
    path = str(path)
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path):
    with open_candidates(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
