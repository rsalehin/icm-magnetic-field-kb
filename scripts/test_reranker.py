# scripts/test_reranker.py
"""
Test Step 10e — BGE reranker on top chunks from retriever.
Verifies cross-encoder rescoring and diversity control.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# everything else below this line
import faiss
from src.storage.db          import get_connection
from src.query.planner       import plan_query
from src.query.bm25_index    import load_bm25_index
from src.query.bge_embedder  import (
    get_bge_model, get_or_create_bge_index, load_bge_id_map
)
from src.query.retriever     import retrieve
from src.query.reranker      import get_reranker, rerank
from src.storage.faiss_index import load_id_map, INDEX_PATH
from src.pipeline            import get_model

# ── Load all backends ─────────────────────────────────────────────────────────
print("Loading backends...")
conn               = get_connection()
specter_model      = get_model()
specter_index      = faiss.read_index(str(INDEX_PATH))
specter_id_map     = load_id_map()
bm25_index, bm25_records = load_bm25_index()
bge_model          = get_bge_model()
bge_index          = get_or_create_bge_index()
bge_id_map         = load_bge_id_map()
reranker           = get_reranker()

# ── Test query ────────────────────────────────────────────────────────────────
query = "What spectral index n did Murgia 2004 find for A119?"
print(f"\nQuery: {query}")

plan   = plan_query(query)
chunks = retrieve(
    plan, conn,
    specter_index, specter_id_map, specter_model,
    bm25_index, bm25_records,
    bge_index, bge_id_map, bge_model,
)

print(f"\nBefore reranking — top 5 by retrieval score:")
for c in chunks[:5]:
    print(f"  ret={c['retrieval_score']:.4f} | "
          f"{c['node_id'][-20:]} | "
          f"{c['section_title'][:25]} | "
          f"{c['text'][:70]}...")

# ── Rerank ────────────────────────────────────────────────────────────────────
evidence = rerank(
    query,
    chunks,
    intent     = plan["intent"],
    evidence_k = plan["budgets"]["evidence_k"],
    min_papers = plan["budgets"]["min_papers"],
    reranker   = reranker,
)

print(f"\nAfter reranking — top {len(evidence)} evidence chunks:")
for i, c in enumerate(evidence):
    print(f"\n  [{i+1}] rerank={c['rerank_score']:.4f}  "
          f"pre={c['pre_rerank_score']:.4f}")
    print(f"       paper   : {c['node_id']}")
    print(f"       section : {c['section_title']}")
    print(f"       text    : {c['text'][:120]}...")

# ── Diversity check ───────────────────────────────────────────────────────────
papers_in_evidence = {c["node_id"] for c in evidence}
print(f"\nDistinct papers in evidence : {len(papers_in_evidence)}")
print(f"Min papers required         : {plan['budgets']['min_papers']}")
print(f"Evidence chunks             : {len(evidence)}")
print(f"Evidence k budget           : {plan['budgets']['evidence_k']}")

# ── Before vs after comparison ────────────────────────────────────────────────
print(f"\nRanking changes (retrieval rank → rerank score):")
for c in evidence:
    orig_rank = next(
        (i+1 for i, x in enumerate(chunks) if x["chunk_id"] == c["chunk_id"]),
        "?"
    )
    print(f"  retrieval rank {orig_rank:>3} → "
          f"rerank={c['rerank_score']:.4f} | "
          f"{c['text'][:60]}...")

conn.close()
print("\nReranker test complete.")