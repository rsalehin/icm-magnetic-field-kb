# scripts/test_bm25.py
"""
Test Step 10a — BM25 index build and search.
Tests exact term retrieval, section filtering, abstract boosting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db import get_connection
from src.query.bm25_index import (
    build_corpus_from_db, build_bm25_index,
    load_bm25_index, bm25_search,
    CORPUS_PATH, INDEX_CACHE, ABSTRACT_PATH,
)

# ── Clean slate ───────────────────────────────────────────────────────────────
for f in [CORPUS_PATH, INDEX_CACHE, ABSTRACT_PATH]:
    if f.exists():
        f.unlink()

# ── Build corpus and index ────────────────────────────────────────────────────
print("\n[1] Building corpus from DuckDB...")
conn = get_connection()
build_corpus_from_db(conn)
conn.close()

print("\n[2] Building BM25 index...")
index, records = build_bm25_index()
print(f"  Total documents: {len(records)}")

# ── Test reload from cache ────────────────────────────────────────────────────
print("\n[3] Testing cache reload...")
index2, records2 = load_bm25_index()
print(f"  Reloaded: {len(records2)} documents  ✓")

# ── Test exact term queries ───────────────────────────────────────────────────
print("\n[4] Testing exact scientific term retrieval...")
test_queries = [
    "Burn law depolarization sigma_RM",
    "Gaussian random field power spectrum GRF",
    "Faraday rotation measure intracluster magnetic field",
    "LOFAR observations galaxy cluster",
    "BxC Biot-Savart convolution magnetic field",
]

for query in test_queries:
    results = bm25_search(index, records, query, top_k=3)
    print(f"\n  Query: '{query}'")
    print(f"  Hits : {len(results)}")
    for r in results[:3]:
        abs_flag = "[ABSTRACT]" if r["is_abstract"] else ""
        print(f"    [{r['bm25_rank']}] score={r['score']:.2f} "
              f"{abs_flag} {r['node_id'][-20:]} | "
              f"{r['section_title'][:30]} | "
              f"{r['text'][:80]}...")

# ── Test node_id filtering ────────────────────────────────────────────────────
print("\n[5] Testing node_id filter (restrict to Murgia 2004)...")
results = bm25_search(
    index, records,
    "rotation measure power spectrum",
    top_k=5,
    node_ids=["arxiv:astro-ph/0406225"],
)
print(f"  Results restricted to Murgia 2004: {len(results)}")
for r in results[:3]:
    print(f"    score={r['score']:.2f} | "
          f"{r['section_title'][:30]} | "
          f"{r['text'][:80]}...")

# ── Stats ─────────────────────────────────────────────────────────────────────
print("\n[6] Corpus stats...")
abstracts = sum(1 for r in records if r.get("is_abstract"))
chunks    = len(records) - abstracts
print(f"  Chunk documents : {chunks}")
print(f"  Abstract docs   : {abstracts}")
print(f"  Total           : {len(records)}")
print(f"  Corpus file     : {CORPUS_PATH}")
print(f"  Cache file      : {INDEX_CACHE}")

print("\nBM25 test complete.")