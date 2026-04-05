# scripts/test_generator.py
"""
Test Step 10h — Full pipeline including Qwen3 generator.
Tests fact, comparison, and abstention queries end to end.
"""
import sys
import faiss
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db          import get_connection
from src.query.planner       import plan_query
from src.query.bm25_index    import load_bm25_index
from src.query.bge_embedder  import (
    get_bge_model, get_or_create_bge_index,
    load_bge_id_map, BGE_MATRIX_PATH,
)
from src.query.retriever     import retrieve
from src.query.reranker      import get_reranker, rerank
from src.query.assembler     import assemble_evidence
from src.query.generator     import generate_answer, print_result
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

bge_matrix = None
if BGE_MATRIX_PATH.exists():
    bge_matrix = np.load(str(BGE_MATRIX_PATH))

print("All backends loaded.\n")


def run_query(query: str) -> None:
    """Run full pipeline on a single query and print result."""
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
    result = generate_answer(
        query,
        pack,
        synthesis_mode = plan["synthesis_mode"],
    )
    print_result(query, result)


# ── Test queries ──────────────────────────────────────────────────────────────

# Test 1 — Fact (non-thinking mode)
run_query("What spectral index n did Murgia 2004 find for A119?")

# Test 2 — Fact with specific numerical value
run_query("What is the typical size of radio mini-halos in galaxy clusters?")

# Test 3 — Comparison (thinking mode)
run_query("How does the GRF magnetic field model differ from MHD cosmological simulations?")

# Test 4 — Out of scope (should abstain or give weak answer)
run_query("What is the optical luminosity of the Coma cluster brightest galaxy?")

conn.close()
print("\nGenerator test complete.")