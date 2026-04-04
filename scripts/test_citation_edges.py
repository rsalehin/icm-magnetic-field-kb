# scripts/test_citation_edges.py
"""
Test Step 11 — Citation edges on 3 seed papers.
Verifies ADS fetching, edge creation, graph structure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db        import get_connection
from src.storage.graph     import get_or_create_graph, save_graph, GRAPH_PATH
from src.storage.citation_graph import (
    add_citation_edges_from_ads,
    add_co_citation_edges,
)

# Fresh test graph
if GRAPH_PATH.exists():
    GRAPH_PATH.unlink()

conn = get_connection()
G    = get_or_create_graph()

print(f"Testing on 3 papers...")
G, stats = add_citation_edges_from_ads(G, conn, limit=3, sleep=0.5)

print(f"\nStats:")
for k, v in stats.items():
    print(f"  {k}: {v}")

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Check edges from Murgia 2004
murgia_id = "arxiv:astro-ph/0406225"
if murgia_id in G:
    out_edges = list(G.successors(murgia_id))
    print(f"\nMurgia 2004 cites {len(out_edges)} papers")
    print(f"First 5 references:")
    for ref in out_edges[:5]:
        data = G.nodes[ref]
        print(f"  → {ref}")
        print(f"     {data.get('title','?')[:50]}")
        print(f"     role: {G.edges[murgia_id, ref].get('citation_role','?')}")

# Test co-citation
G = add_co_citation_edges(G, min_shared=2)
co_cited = sum(
    1 for _,_,d in G.edges(data=True)
    if d.get("citation_role") == "co_cited"
)
print(f"\nCo-citation edges: {co_cited}")

save_graph(G)
conn.close()
print("\nCitation edge test complete.")