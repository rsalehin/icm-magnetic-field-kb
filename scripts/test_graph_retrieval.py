# scripts/test_graph_retrieval.py
"""
Test graph-expanded retrieval vs baseline.
"""
import sys
import faiss
import networkx as nx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db          import get_connection
from src.query.planner       import plan_query
from src.query.bm25_index    import load_bm25_index
from src.query.bge_embedder  import (
    get_bge_model, get_or_create_bge_index, load_bge_id_map
)
from src.query.retriever     import retrieve
from src.storage.faiss_index import load_id_map, INDEX_PATH
from src.storage.graph       import get_or_create_graph, GRAPH_PATH
from src.pipeline            import get_model

print("Loading backends...")
conn             = get_connection()
specter_model    = get_model()
specter_index    = faiss.read_index(str(INDEX_PATH))
specter_id_map   = load_id_map()
bm25_index, bm25_records = load_bm25_index()
bge_model        = get_bge_model()
bge_index        = get_or_create_bge_index()
bge_id_map       = load_bge_id_map()
G                = get_or_create_graph()

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

query = "What foundational methods were used to model intracluster magnetic fields?"
plan  = plan_query(query)

print(f"Query: {query}")
print(f"Intent: {plan['intent']}\n")

print("--- Without graph expansion ---")
chunks_no_graph = retrieve(
    plan, conn,
    specter_index, specter_id_map, specter_model,
    bm25_index, bm25_records,
    bge_index, bge_id_map, bge_model,
    G=None,
)
papers_no_graph = {c["node_id"] for c in chunks_no_graph}
print(f"Unique papers: {len(papers_no_graph)}")

print("\n--- With graph expansion ---")
chunks_with_graph = retrieve(
    plan, conn,
    specter_index, specter_id_map, specter_model,
    bm25_index, bm25_records,
    bge_index, bge_id_map, bge_model,
    G=G,
)
papers_with_graph = {c["node_id"] for c in chunks_with_graph}
print(f"Unique papers: {len(papers_with_graph)}")

new_papers = papers_with_graph - papers_no_graph
print(f"New papers added by graph expansion: {len(new_papers)}")
for p in list(new_papers)[:5]:
    print(f"  {p}")

conn.close()
print("\nGraph retrieval test complete.")