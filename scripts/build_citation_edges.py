# scripts/build_citation_edges.py
"""
Build full citation graph from ADS references.
Processes all 251 papers, adds CITES edges + CO_CITED edges.
Estimated time: ~15-20 minutes (ADS rate limiting).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db        import get_connection
from src.storage.graph     import get_or_create_graph, save_graph
from src.storage.citation_graph import (
    add_citation_edges_from_ads,
    add_co_citation_edges,
)

conn = get_connection()
G    = get_or_create_graph()

print(f"Graph before: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Build CITES edges from ADS
G, stats = add_citation_edges_from_ads(G, conn, sleep=0.3)

print(f"\nCITES edges built:")
for k, v in stats.items():
    print(f"  {k:<22}: {v}")

# Build CO_CITED edges
G = add_co_citation_edges(G, min_shared=3)

# Save
save_graph(G)
conn.close()

print(f"\nFinal graph:")
print(f"  Nodes : {G.number_of_nodes()}")
print(f"  Edges : {G.number_of_edges()}")
print(f"  CITES edges     : {sum(1 for _,_,d in G.edges(data=True) if d.get('citation_role') != 'co_cited')}")
print(f"  CO_CITED edges  : {sum(1 for _,_,d in G.edges(data=True) if d.get('citation_role') == 'co_cited')}")