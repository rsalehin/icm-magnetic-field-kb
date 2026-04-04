# scripts/test_assembler.py
"""
Test Step 10f — Evidence assembler.
Tests deduplication, conflict detection, abstention signal,
and evidence grouping on real retriever + reranker output.
"""
import sys
import json
import faiss
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db          import get_connection
from src.query.planner       import plan_query
from src.query.bm25_index    import load_bm25_index
from src.query.bge_embedder  import (
    get_bge_model, get_or_create_bge_index, load_bge_id_map,
    BGE_MATRIX_PATH,
)
from src.query.retriever     import retrieve
from src.query.reranker      import get_reranker, rerank
from src.query.assembler     import assemble_evidence
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

# Load BGE matrix for deduplication
bge_matrix = None
if BGE_MATRIX_PATH.exists():
    print("Loading BGE matrix for deduplication...")
    bge_matrix = np.load(str(BGE_MATRIX_PATH))
    print(f"  Matrix shape: {bge_matrix.shape}")

# ── Test queries ──────────────────────────────────────────────────────────────
test_queries = [
    "What spectral index n did Murgia 2004 find for A119?",
    "Compare GRF and BxC magnetic field models in galaxy clusters",
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
    evidence_chunks = rerank(
        query, chunks,
        intent     = plan["intent"],
        evidence_k = plan["budgets"]["evidence_k"],
        min_papers = plan["budgets"]["min_papers"],
        reranker   = reranker,
    )
    pack = assemble_evidence(
        query, plan, evidence_chunks,
        conn, bge_id_map, bge_matrix,
    )

    # ── Print evidence pack summary ───────────────────────────────────────────
    stats = pack["support_stats"]
    print(f"\nEvidence pack:")
    print(f"  intent           : {pack['intent']}")
    print(f"  abstain          : {pack['abstain']}")
    if pack["abstain_reason"]:
        print(f"  abstain_reason   : {pack['abstain_reason']}")
    print(f"\nSupport stats:")
    print(f"  distinct chunks  : {stats['distinct_chunks']}")
    print(f"  distinct papers  : {stats['distinct_papers']}")
    print(f"  max rerank score : {stats['max_rerank_score']}")
    print(f"  deduplicated     : {stats['chunks_deduplicated']}")
    print(f"  has_disagreement : {stats['has_disagreement']}")
    print(f"  has_uncertainty  : {stats['has_uncertainty']}")
    if stats["conflict_notes"]:
        print(f"  conflict notes   : {stats['conflict_notes']}")

    print(f"\nEvidence by paper:")
    for group in pack["paper_groups"]:
        print(f"\n  [{group['year']}] {group['title'][:55]}")
        print(f"         {group['node_id']}")
        for c in group["chunks"]:
            print(f"    rerank={c['rerank_score']:.4f} "
                  f"| {c['section_title'][:25]} "
                  f"| {c['text'][:70]}...")

conn.close()
print("\nAssembler test complete.")