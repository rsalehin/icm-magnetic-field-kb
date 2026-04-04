# src/query/bm25_index.py
"""
Step 10a — BM25 lexical index over all chunk texts.
Canonical store: JSONL corpus in data/bm25/
Cache artifact: pickle in data/bm25/ (regeneratable)

Indexes a weighted text field:
  section_title + chunk_text + keywords
to improve recall on exact scientific terms.
"""

import re
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

BM25_DIR        = Path("data/bm25")
CORPUS_PATH     = BM25_DIR / "bm25_corpus.jsonl"
INDEX_CACHE     = BM25_DIR / "bm25_index.pkl"
ABSTRACT_PATH   = BM25_DIR / "abstracts.jsonl"


# ── Tokeniser ─────────────────────────────────────────────────────────────────

# Scientific tokeniser — preserves Greek letters, subscripts,
# method names, telescope acronyms
_SPLIT_RE = re.compile(r"[^a-zA-Z0-9µσλΛαβγδεζηθικνξπρτυφχψω_\-]+")

def tokenise(text: str) -> list[str]:
    """
    Lowercase and split on non-alphanumeric/scientific chars.
    Preserves: sigma_RM, B_0, beta-model, LOFAR, MeerKAT.
    Filters tokens shorter than 2 chars.
    """
    text   = text.lower()
    tokens = _SPLIT_RE.split(text)
    return [t for t in tokens if len(t) >= 2]


# ── Weighted text field ───────────────────────────────────────────────────────

def build_weighted_text(chunk: dict) -> str:
    """
    Combine section_title + text + keywords into one weighted field.
    Section title repeated 2x for boost.
    """
    section  = chunk.get("section_title", "") or ""
    text     = chunk.get("text", "") or ""
    keywords = " ".join(chunk.get("sparse_tokens", {}).get("keywords", []))
    return f"{section} {section} {text} {keywords}".strip()


# ── Build corpus from DuckDB ──────────────────────────────────────────────────

def build_corpus_from_db(conn) -> None:
    """
    Pull all non-noise chunks from DuckDB and write to JSONL corpus.
    Also extracts abstracts as special high-weight documents.
    Canonical store — must be rebuilt when corpus changes.
    """
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    print("Building BM25 corpus from DuckDB...")

    # Pull all non-noise chunks
    rows = conn.execute("""
        SELECT
            c.chunk_id,
            c.node_id,
            c.section_title,
            c.text,
            c.token_count,
            c.has_equation,
            c.has_numerical_result,
            c.page_num
        FROM chunks c
        WHERE c.is_noise = FALSE
        ORDER BY c.node_id, c.chunk_index
    """).fetchall()

    cols = ["chunk_id", "node_id", "section_title", "text",
            "token_count", "has_equation", "has_numerical_result", "page_num"]

    chunk_count = 0
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="Writing chunk corpus"):
            record = dict(zip(cols, row))
            record["weighted_text"] = build_weighted_text(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            chunk_count += 1

    print(f"  Wrote {chunk_count} chunks to {CORPUS_PATH}")

    # Pull abstracts as special documents
    abstracts = conn.execute("""
        SELECT node_id, abstract, title, year, journal
        FROM papers
        WHERE abstract IS NOT NULL
        ORDER BY year
    """).fetchall()

    abs_cols = ["node_id", "abstract", "title", "year", "journal"]
    abs_count = 0
    with open(ABSTRACT_PATH, "w", encoding="utf-8") as f:
        for row in abstracts:
            record = dict(zip(abs_cols, row))
            # Abstract gets title repeated 3x for strong boost
            record["weighted_text"] = (
                f"{record['title']} {record['title']} {record['title']} "
                f"{record['abstract']}"
            )
            record["chunk_id"]      = f"{record['node_id']}__abstract"
            record["section_title"] = "abstract"
            record["is_abstract"]   = True
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            abs_count += 1

    print(f"  Wrote {abs_count} abstracts to {ABSTRACT_PATH}")


# ── Build BM25 index from corpus ──────────────────────────────────────────────

def build_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Load corpus from JSONL and build BM25Okapi index.
    Returns (index, corpus_records).
    Saves pickle cache.
    """
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found at {CORPUS_PATH}. "
            "Run build_corpus_from_db() first."
        )

    print("Loading corpus and building BM25 index...")

    records = []
    # Load chunks
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Load abstracts
    if ABSTRACT_PATH.exists():
        with open(ABSTRACT_PATH, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

    tokenised = [tokenise(r["weighted_text"]) for r in records]
    index     = BM25Okapi(tokenised)

    # Save cache
    with open(INDEX_CACHE, "wb") as f:
        pickle.dump((index, records), f)

    print(f"  BM25 index built: {len(records)} documents")
    return index, records


def load_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Load BM25 index from pickle cache.
    Rebuilds from JSONL if cache missing.
    """
    if INDEX_CACHE.exists():
        print("Loading BM25 index from cache...")
        with open(INDEX_CACHE, "rb") as f:
            index, records = pickle.load(f)
        print(f"  Loaded {len(records)} documents")
        return index, records
    else:
        print("Cache not found — building from corpus...")
        return build_bm25_index()


# ── Search ────────────────────────────────────────────────────────────────────

def bm25_search(
    index:    BM25Okapi,
    records:  list[dict],
    query:    str,
    top_k:    int = 50,
    node_ids: list[str] | None = None,
) -> list[dict]:
    """
    Search BM25 index for query.
    If node_ids provided, restricts search to chunks from those papers.
    Returns list of {chunk_id, node_id, score, section_title, text} dicts.
    """
    query_tokens = tokenise(query)
    if not query_tokens:
        return []

    scores = index.get_scores(query_tokens)

    # Build results
    results = []
    for i, (score, record) in enumerate(zip(scores, records)):
        if score <= 0:
            continue
        if node_ids and record["node_id"] not in node_ids:
            continue
        results.append({
            "chunk_id":     record["chunk_id"],
            "node_id":      record["node_id"],
            "section_title": record.get("section_title", ""),
            "text":         record.get("text", record.get("abstract", "")),
            "score":        float(score),
            "is_abstract":  record.get("is_abstract", False),
            "bm25_rank":    0,  # filled after sorting
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Assign ranks
    for i, r in enumerate(results):
        r["bm25_rank"] = i + 1

    return results[:top_k]