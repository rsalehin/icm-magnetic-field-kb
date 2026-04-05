# scripts/test_retriever.py
"""
Test Step 10d — Hybrid retriever on 3 query types.
Verifies paper candidate filtering, BM25+BGE fusion, RRF scoring.
"""
import sys
import faiss
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db         import get_connection
from src.query.planner      import plan_query
from src.query.bm25_index   import load_bm25_index
from src.query.bge_embedder import (
    get_bge_model, get_or_create_bge_index, load_bge_id_map
)
from src.query.retriever    import retrieve
from src.storage.faiss_index import load_id_map, INDEX_PATH
from src.pipeline            import get_model

# ── Load all backends ─────────────────────────────────────────────────────────
print("Loading backends...")
conn             = get_connection()
specter_model    = get_model()
specter_index    = faiss.read_index(str(INDEX_PATH))
specter_id_map   = load_id_map()
bm25_index, bm25_records = load_bm25_index()
bge_model        = get_bge_model()
bge_index        = get_or_create_bge_index()
bge_id_map       = load_bge_id_map()

print(f"  SPECTER2 vectors : {specter_index.ntotal}")
print(f"  BGE-M3 vectors   : {bge_index.ntotal}")
print(f"  BM25 documents   : {len(bm25_records)}")

# ── Test queries ──────────────────────────────────────────────────────────────
test_queries = [
    "What spectral index n did Murgia 2004 find for A119?",
    "Compare GRF and BxC magnetic field models in galaxy clusters",
    "LOFAR observations of galaxy cluster radio halos after 2018",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    plan   = plan_query(query)
    chunks = retrieve(
        plan, conn,
        specter_index, specter_id_map, specter_model,
        bm25_index, bm25_records,
        bge_index, bge_id_map, bge_model,
    )

    print(f"\nTop 5 chunks (of {len(chunks)} retrieved):")
    seen_papers = set()
    for c in chunks[:5]:
        seen_papers.add(c["node_id"])
        print(f"  score={c['retrieval_score']:.4f} "
              f"rrf={c['rrf_score']:.4f} "
              f"sec_prior={c['section_prior']:.2f} "
              f"| {c['node_id'][-20:]} "
              f"| {c['section_title'][:25]} "
              f"| {c['text'][:70]}...")

    print(f"\nDistinct papers in top 5: {len(seen_papers)}")
    print(f"Total chunks retrieved  : {len(chunks)}")

conn.close()
print("\nRetriever test complete.")