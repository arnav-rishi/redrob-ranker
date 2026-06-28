"""Module 5: role-fit gate. The primary defense against keyword stuffers --
a 'Marketing Manager' with 9 AI skills is capped here regardless of how high
skill/lexical scores are. See docs/STEP4_SPEC.md S2.

Classification order: exact_title_map (all 47 real titles, Step 3 audit) ->
substring fallback (role_titles, for titles unseen in this 100K) -> unknown
(lean on career evidence).
"""


def role_fit(feat: dict, jd: dict, ontology: dict) -> float:
    title = (feat.get("current_title") or "").strip().lower()
    bucket = jd["exact_title_map"].get(title) or classify_via_substring(title, jd["role_titles"])
    titles_all = " ".join((j.get("title") or "").lower() for j in feat.get("career", []))

    if bucket == "reject":
        if _match_any(titles_all, jd["role_titles"]["strong"]):
            return 0.55   # rescued: current title is a trap, career shows real ML roles
        return 0.15
    if bucket == "strong":
        return 1.00
    if bucket == "adjacent":
        return 0.80 if _built_ml_systems(feat.get("profile_text", ""), ontology) else 0.55
    # bucket is None: title unseen in exact_title_map AND no substring match
    # (only possible on titles outside this 100K, e.g. the 200K production pool)
    return 0.70 if _built_ml_systems(feat.get("profile_text", ""), ontology) else 0.35


def classify_via_substring(title, role_titles):
    for bucket in ("reject", "strong", "adjacent"):
        if _match_any(title, role_titles[bucket]):
            return bucket
    return None


def _match_any(text, patterns):
    return any(p in text for p in patterns)


_ML_CORROBORATION_GROUPS = ("retrieval_embeddings", "vector_search", "ranking_recsys", "nlp")


def _ml_corroboration_terms(ontology: dict) -> list:
    """Flattened term list reused from Step 3's audited skill_ontology.yaml --
    single source of truth, no second hardcoded phrase list to drift out of
    sync (see docs/STEP4_SPEC.md S2 decision)."""
    terms = []
    for group in _ML_CORROBORATION_GROUPS:
        terms.extend(ontology.get(group, []))
    return terms


def _built_ml_systems(profile_text: str, ontology: dict) -> bool:
    terms = _ml_corroboration_terms(ontology)
    text = (profile_text or "").lower()
    return any(t in text for t in terms)
