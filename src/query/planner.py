# src/query/planner.py
"""
Step 10c — Rule-based query planner.
Detects intent, extracts entities, separates explicit from
inferred constraints, sets candidate budgets.

Output is a typed PlannerOutput dict — the contract between
the planner and the retriever.
"""

import re
from typing import TypedDict


# ── Typed contract ────────────────────────────────────────────────────────────

class ExplicitFilters(TypedDict):
    year_min:   int | None
    year_max:   int | None
    journal:    str | None
    instrument: list[str]
    authors:    list[str]


class SoftBoosts(TypedDict):
    sections:   list[str]
    methods:    list[str]
    topics:     list[str]


class Budgets(TypedDict):
    paper_k:        int
    chunk_lex_k:    int
    chunk_dense_k:  int
    rerank_k:       int
    evidence_k:     int
    min_papers:     int


class Entities(TypedDict):
    sources:     list[str]
    methods:     list[str]
    instruments: list[str]
    authors:     list[str]
    years:       dict
    quantities:  list[str]


class PlannerOutput(TypedDict):
    intent:           str
    entities:         Entities
    explicit_filters: ExplicitFilters
    soft_boosts:      SoftBoosts
    budgets:          Budgets
    synthesis_mode:   str
    raw_query:        str


# ── Vocabulary dictionaries ───────────────────────────────────────────────────

KNOWN_CLUSTERS = {
    "a119", "abell 119", "a2255", "abell 2255", "coma", "coma cluster",
    "a400", "abell 400", "a2634", "abell 2634", "a514", "hydra a",
    "3c 449", "perseus", "virgo", "fornax", "centaurus",
    "macs j0717", "a2319", "a2256", "a2744", "el gordo",
    "a2142", "a2029", "a1795", "a2199", "a523", "a2345",
}

KNOWN_METHODS = {
    "grf":              "GRF",
    "gaussian random field": "GRF",
    "bxc":              "BxC",
    "biot-savart":      "BxC",
    "biot savart":      "BxC",
    "rm synthesis":     "RM_synthesis",
    "rm-synthesis":     "RM_synthesis",
    "faraday synthesis": "RM_synthesis",
    "mhd":              "MHD",
    "magnetohydrodynamic": "MHD",
    "analytical":       "analytical",
    "single-scale":     "analytical",
    "burn law":         "Burn_law",
    "depolarization":   "depolarization",
    "faraday rotation": "Faraday_rotation",
    "rotation measure": "RM",
    "power spectrum":   "power_spectrum",
    "structure function": "structure_function",
}

KNOWN_INSTRUMENTS = {
    "vla":      "VLA",
    "lofar":    "LOFAR",
    "meerkat":  "MeerKAT",
    "askap":    "ASKAP",
    "ska":      "SKA",
    "wsrt":     "WSRT",
    "atca":     "ATCA",
    "gmrt":     "GMRT",
    "chandra":  "Chandra",
    "xmm":      "XMM-Newton",
    "xmm-newton": "XMM-Newton",
    "rosat":    "ROSAT",
}

KNOWN_QUANTITIES = {
    "sigma_rm", "σ_rm", "sigma_rm", "rm dispersion",
    "b_0", "b0", "central field", "magnetic field strength",
    "spectral index", "power spectrum slope",
    "lambda_min", "λ_min", "minimum scale",
    "lambda_max", "λ_max", "maximum scale",
    "depolarization", "rotation measure", "faraday depth",
}

KNOWN_JOURNALS = {
    "a&a":   "Astronomy and Astrophysics",
    "aa":    "Astronomy and Astrophysics",
    "apj":   "The Astrophysical Journal",
    "mnras": "Monthly Notices of the Royal Astronomical Society",
    "ara&a": "Annual Review of Astronomy and Astrophysics",
    "astra": "Astronomy and Astrophysics",
}

# ── Intent detection ──────────────────────────────────────────────────────────

FACT_PATTERNS = [
    r"\bwhat (is|are|was|were)\b",
    r"\bhow (much|many|strong|large|small)\b",
    r"\bwhich value\b",
    r"\bfind the\b",
    r"\bwhat (did|does).+find\b",
    r"\bwhat.+(measure|report|estimate|calculate|derive)\b",
    r"\bwhat.+(result|value|number|quantity)\b",
]

COMPARISON_PATTERNS = [
    r"\bcompar(e|ing|ison)\b",
    r"\bdifferen(t|ce)\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bbetter than\b",
    r"\bworse than\b",
    r"\bwhich (model|method|approach|technique) (is|performs|works)\b",
    r"\bhow do.+(differ|compare)\b",
    r"\bgrf.+bxc\b",
    r"\banalytical.+simulation\b",
]

SYNTHESIS_PATTERNS = [
    r"\bwhat does the literature\b",
    r"\boverview\b",
    r"\breview\b",
    r"\bsummar(y|ize|ise)\b",
    r"\bacross (papers|studies|observations|simulations)\b",
    r"\bbroad(ly)?\b",
    r"\bin general\b",
    r"\bstate of the (art|field)\b",
    r"\bwhat is known\b",
    r"\bconsensus\b",
]

DISCOVERY_PATTERNS = [
    r"\brelated to\b",
    r"\bsimilar to\b",
    r"\bwhat papers\b",
    r"\bfind papers\b",
    r"\bwho (worked|studied|investigated|observed)\b",
    r"\bpapers (about|on|studying|using)\b",
    r"\blist (of )?(papers|studies|work)\b",
    r"\bwhich authors\b",
    r"\bpapers\s+(after|before|since|from|between)\b",
]


def detect_intent(query: str) -> str:
    """
    Detect query intent from pattern matching.
    Returns: fact | comparison | synthesis | discovery
    Fallback: fact
    """
    q = query.lower()

    scores = {
        "synthesis":  sum(1 for p in SYNTHESIS_PATTERNS  if re.search(p, q)),
        "comparison": sum(1 for p in COMPARISON_PATTERNS if re.search(p, q)),
        "discovery":  sum(1 for p in DISCOVERY_PATTERNS  if re.search(p, q)),
        "fact":       sum(1 for p in FACT_PATTERNS        if re.search(p, q)),
    }

    # Return highest scoring intent
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "fact"

    # Bias correction: journal + method/quantity → fact not discovery
    has_journal   = any(j in q for j in KNOWN_JOURNALS)
    has_quantity  = any(qty in q for qty in KNOWN_QUANTITIES)
    has_method    = any(m in q for m in KNOWN_METHODS)

    if best == "discovery" and has_journal and (has_quantity or has_method):
        best = "fact"

    return best


# ── Entity extraction ─────────────────────────────────────────────────────────

def extract_entities(query: str) -> Entities:
    """Extract named entities from query."""
    q = query.lower()

    sources = [
        c.upper() if len(c) <= 5 else c.title()
        for c in KNOWN_CLUSTERS
        if c in q
    ]

    methods = list({
        canon for term, canon in KNOWN_METHODS.items()
        if term in q
    })

    instruments = list({
        canon for term, canon in KNOWN_INSTRUMENTS.items()
        if term in q
    })

    quantities = [
        qty for qty in KNOWN_QUANTITIES
        if qty in q
    ]

    # Year extraction
    years = {}
    year_matches = re.findall(r"\b(?:19|20)\d{2}\b", query)
    if year_matches:
        year_ints = [int(y) for y in year_matches]
        years = {"min": min(year_ints), "max": max(year_ints)}

    # Author extraction (Last, F. or Lastname YEAR pattern)
    author_matches = re.findall(
        r"\b([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s+(?:19|20)\d{2}\b", query
    )
    authors = list(set(author_matches))

    return Entities(
        sources=sources,
        methods=methods,
        instruments=instruments,
        authors=authors,
        years=years,
        quantities=quantities,
    )


# ── Explicit vs inferred constraints ─────────────────────────────────────────

def extract_explicit_filters(query: str, entities: Entities) -> ExplicitFilters:
    """
    Extract EXPLICIT constraints — only when user literally states them.
    Never infer hard filters from domain context.
    """
    q = query.lower()

    # Year constraints — only if explicitly stated with constraint language
    year_min = None
    year_max = None

    after_match  = re.search(r"\bafter\s+((?:19|20)\d{2})\b", q)
    before_match = re.search(r"\bbefore\s+((?:19|20)\d{2})\b", q)
    since_match  = re.search(r"\bsince\s+((?:19|20)\d{2})\b", q)

    if after_match:
        year_min = int(after_match.group(1)) + 1
    if since_match:
        year_min = int(since_match.group(1))
    if before_match:
        year_max = int(before_match.group(1)) - 1

    # Journal — only if explicitly named
    journal = None
    for abbrev, full in KNOWN_JOURNALS.items():
        if abbrev in q:
            journal = full
            break

    # Instrument — only if query explicitly constrains it
    # Pattern: "MeerKAT papers", "observed with VLA", "using LOFAR"
    explicit_instruments = []
    for term, canon in KNOWN_INSTRUMENTS.items():
        if re.search(
            rf"\b{re.escape(term)}\s+(paper|observation|data|survey|image)\b"
            rf"|\b(using|with|from|observed with)\s+{re.escape(term)}\b",
            q
        ):
            explicit_instruments.append(canon)

    # Authors — only if query explicitly asks about specific author
    explicit_authors = []
    if entities["authors"]:
        explicit_authors = entities["authors"]

    return ExplicitFilters(
        year_min=year_min,
        year_max=year_max,
        journal=journal,
        instrument=explicit_instruments,
        authors=explicit_authors,
    )


def build_soft_boosts(
    intent:   str,
    entities: Entities,
) -> SoftBoosts:
    """
    Build soft boosts from intent and inferred context.
    These influence scoring but never hard-filter results.
    """
    # Section boosts per intent
    section_map = {
        "fact":       ["results", "abstract", "conclusion"],
        "comparison": ["results", "discussion", "conclusion", "abstract"],
        "synthesis":  ["abstract", "results", "discussion", "conclusion"],
        "discovery":  ["abstract", "introduction"],
    }

    # Method boosts from entities
    methods = entities.get("methods", [])

    # Topic boosts from instruments (inferred, not hard filter)
    topics = []
    if entities.get("instruments"):
        topics.append("radio_observations")
    if "MHD" in entities.get("methods", []):
        topics.append("simulation")
    if any(q in entities.get("quantities", [])
           for q in ["sigma_rm", "σ_rm", "rotation measure"]):
        topics.append("faraday_rotation")

    return SoftBoosts(
        sections=section_map.get(intent, ["abstract", "results"]),
        methods=methods,
        topics=topics,
    )


# ── Candidate budgets ─────────────────────────────────────────────────────────

BUDGET_MAP = {
    "fact": Budgets(
        paper_k=40,
        chunk_lex_k=120,
        chunk_dense_k=120,
        rerank_k=40,
        evidence_k=8,
        min_papers=2,
    ),
    "comparison": Budgets(
        paper_k=80,
        chunk_lex_k=150,
        chunk_dense_k=150,
        rerank_k=50,
        evidence_k=12,
        min_papers=3,
    ),
    "synthesis": Budgets(
        paper_k=150,
        chunk_lex_k=200,
        chunk_dense_k=200,
        rerank_k=60,
        evidence_k=15,
        min_papers=4,
    ),
    "discovery": Budgets(
        paper_k=100,
        chunk_lex_k=100,
        chunk_dense_k=100,
        rerank_k=40,
        evidence_k=10,
        min_papers=1,
    ),
}

SYNTHESIS_MODE_MAP = {
    "fact":       "non_thinking",
    "comparison": "thinking",
    "synthesis":  "thinking",
    "discovery":  "non_thinking",
}


# ── Main planner function ─────────────────────────────────────────────────────

def plan_query(query: str) -> PlannerOutput:
    """
    Main entry point. Takes raw query string.
    Returns fully typed PlannerOutput dict.
    """
    intent   = detect_intent(query)
    entities = extract_entities(query)

    explicit_filters = extract_explicit_filters(query, entities)
    soft_boosts      = build_soft_boosts(intent, entities)
    budgets          = BUDGET_MAP[intent]
    synthesis_mode   = SYNTHESIS_MODE_MAP[intent]

    return PlannerOutput(
        intent=intent,
        entities=entities,
        explicit_filters=explicit_filters,
        soft_boosts=soft_boosts,
        budgets=budgets,
        synthesis_mode=synthesis_mode,
        raw_query=query,
    )