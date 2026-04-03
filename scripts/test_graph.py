# scripts/test_graph.py
"""
Test Step 8 — NetworkX graph init, add nodes, edges, persist, reload.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.pdf_extractor import extract_paper
from src.enrichment.ads_enricher  import enrich_paper
from src.storage.graph            import (
    get_or_create_graph, save_graph, add_paper_node,
    add_citation_edges, mark_seed, get_graph_stats,
    get_neighbors, GRAPH_PATH,
)

# ── Clean slate ───────────────────────────────────────────────────────────────
if GRAPH_PATH.exists():
    GRAPH_PATH.unlink()

# ── Create graph ──────────────────────────────────────────────────────────────
print("\n[1] Creating graph...")
G = get_or_create_graph()

# ── Add two papers as nodes ───────────────────────────────────────────────────
print("\n[2] Adding paper nodes...")
papers = [
    "papers/2004_Murgia_astro-ph_0406225.pdf",
    "papers/2009_Bonafede_0905.3552.pdf",
]
processed = []
for pdf_path in papers:
    p = Path(pdf_path)
    paper = extract_paper(p)
    paper = enrich_paper(paper)
    G     = add_paper_node(G, paper)
    processed.append(paper)
    print(f"  Added node: {paper['node_id']}")

# ── Mark Murgia as seed ───────────────────────────────────────────────────────
G = mark_seed(G, "arxiv:astro-ph/0406225")

# ── Add synthetic citation edges ──────────────────────────────────────────────
print("\n[3] Adding citation edges...")
refs = [
    {
        "target_node_id":   "arxiv:0905.3552",
        "citation_role":    "methodological",
        "context_chunk_id": "arxiv:astro-ph/0406225__c0042",
    },
    {
        "target_node_id":   "arxiv:astro-ph/0302426",  # Ensslin stub
        "citation_role":    "foundational",
        "context_chunk_id": "arxiv:astro-ph/0406225__c0010",
    },
]
G = add_citation_edges(G, "arxiv:astro-ph/0406225", refs)

# ── Stats ─────────────────────────────────────────────────────────────────────
print("\n[4] Graph stats...")
stats = get_graph_stats(G)
for k, v in stats.items():
    print(f"  {k:<20}: {v}")

# ── Neighbours ────────────────────────────────────────────────────────────────
print("\n[5] Neighbours of Murgia 2004...")
neighbours = get_neighbors(G, "arxiv:astro-ph/0406225")
print(f"  Cites (out)     : {len(neighbours['out'])} papers")
for n in neighbours["out"]:
    print(f"    → {n['node_id']}  [{n['citation_role']}]")
print(f"  Cited by (in)   : {len(neighbours['in'])} papers")

# ── Persist and reload ────────────────────────────────────────────────────────
print("\n[6] Saving graph to disk...")
save_graph(G)

print("\n[7] Reloading graph from disk...")
G2    = get_or_create_graph()
stats2 = get_graph_stats(G2)
print(f"  nodes : {stats2['nodes']}  (expected 3)")
print(f"  edges : {stats2['edges']}  (expected 2)")

# ── Cleanup ───────────────────────────────────────────────────────────────────
GRAPH_PATH.unlink()
print("\nAll graph tests passed."
      if stats2["nodes"] == 3 and stats2["edges"] == 2
      else "ERROR: node/edge count mismatch.")