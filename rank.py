"""Thin CLI wrapper matching the documented reproduce command:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
Real logic lives in src/pipeline/rank.py.
"""
from src.pipeline.rank import main

if __name__ == "__main__":
    main()
