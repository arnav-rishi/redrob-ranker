import json
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CANDIDATES_PATH = REPO_ROOT.parent / "candidates.jsonl"


def load_jd():
    return yaml.safe_load((REPO_ROOT / "config" / "jd_profile.yaml").read_text(encoding="utf-8"))


def load_ontology():
    return yaml.safe_load((REPO_ROOT / "config" / "skill_ontology.yaml").read_text(encoding="utf-8"))


def real_title_counts():
    titles = Counter()
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            titles[c["profile"]["current_title"]] += 1
    return titles


def test_jd_profile_loads():
    jd = load_jd()
    assert "exact_title_map" in jd
    assert "role_titles" in jd
    assert "must_have" in jd and "nice_to_have" in jd


def test_skill_ontology_loads():
    ont = load_ontology()
    assert "retrieval_embeddings" in ont
    assert "off_target" in ont


def test_every_must_have_and_nice_to_have_group_is_defined():
    jd = load_jd()
    ont = load_ontology()
    for group in jd["must_have"] + jd["nice_to_have"]:
        assert group in ont, f"jd_profile.yaml references undefined group: {group}"


def test_exact_title_map_covers_every_real_title():
    jd = load_jd()
    titles = real_title_counts()
    missing = [t for t in titles if t.lower() not in jd["exact_title_map"]]
    assert missing == [], f"titles not covered by exact_title_map: {missing}"


def test_exact_title_map_values_are_valid_buckets():
    jd = load_jd()
    valid = {"strong", "adjacent", "reject"}
    bad = {t: v for t, v in jd["exact_title_map"].items() if v not in valid}
    assert bad == {}, f"invalid bucket values in exact_title_map: {bad}"
