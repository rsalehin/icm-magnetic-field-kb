# src/storage/graph.py
"""
Step 8 — NetworkX citation graph initialisation and management.
Nodes = papers (node_id as key).
Edges = citation relationships with typed attributes.
Graph persists to disk as GraphML.
"""

import json
import networkx as nx
from pathlib import Path

GRAPH_DIR  = Path("data/graph")
GRAPH_PATH = GRAPH_DIR / "citation_graph.graphml"


def get_or_create_graph() -> nx.DiGraph:
    """
    Load existing graph from disk or create a new directed graph.
    DiGraph: directed — edges go from citing → cited paper.
    """
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    if GRAPH_PATH.exists():
        G = nx.read_graphml(str(GRAPH_PATH))
        print(f"Loaded existing graph: "
              f"{G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")
    else:
        G = nx.DiGraph()
        print("Created new directed citation graph.")

    return G


def save_graph(G: nx.DiGraph) -> None:
    """Persist graph to disk as GraphML."""
    nx.write_graphml(G, str(GRAPH_PATH))


def add_paper_node(G: nx.DiGraph, paper: dict) -> nx.DiGraph:
    """
    Add a paper as a node with Layer 1 attributes.
    Idempotent — updates attributes if node already exists.
    """
    node_id = paper["node_id"]
    l1      = paper["layer_1_bibliographic"]
    m       = paper["meta"]

    G.add_node(node_id, **{
        "arxiv_id":       l1.get("arxiv_id", ""),
        "bibcode":        l1.get("bibcode", ""),
        "title":          l1.get("title", ""),
        "year":           int(l1.get("year", 0)) if l1.get("year") else 0,
        "journal":        l1.get("journal", ""),
        "citation_count": int(l1.get("citation_count") or 0),
        "total_chunks":   int(paper["layer_2_chunks"].get("total_chunks", 0)),
        "stage_a":        m.get("stage_a", ""),
        "stage_b":        m.get("stage_b", ""),
        "stage_c":        m.get("stage_c", ""),
        "stage_d":        m.get("stage_d", ""),
        "is_seed":        False,
    })

    return G


def add_citation_edges(
    G:          nx.DiGraph,
    source_id:  str,
    references: list[dict],
) -> nx.DiGraph:
    """
    Add directed edges from source_id → each referenced paper.
    references: list of {target_node_id, citation_role, context_chunk_id}

    Edge attributes:
      citation_role — foundational | methodological | data_source |
                      comparison | incidental
      context_chunk_id — which chunk contains the citation
    """
    for ref in references:
        target_id = ref.get("target_node_id")
        if not target_id:
            continue

        # Add target as stub node if not yet in graph
        if target_id not in G:
            G.add_node(target_id, **{
                "arxiv_id": "", "bibcode": "", "title": "",
                "year": 0, "journal": "", "citation_count": 0,
                "total_chunks": 0, "stage_a": "pending",
                "stage_b": "pending", "stage_c": "pending",
                "stage_d": "pending", "is_seed": False,
            })

        G.add_edge(source_id, target_id, **{
            "citation_role":    ref.get("citation_role", "incidental"),
            "context_chunk_id": ref.get("context_chunk_id", ""),
        })

    return G


def mark_seed(G: nx.DiGraph, node_id: str) -> nx.DiGraph:
    """Mark a paper as a seed paper in the graph."""
    if node_id in G:
        G.nodes[node_id]["is_seed"] = True
    return G


def get_graph_stats(G: nx.DiGraph) -> dict:
    """Return summary statistics for the current graph."""
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0}

    years = [
        d.get("year", 0)
        for _, d in G.nodes(data=True)
        if d.get("year", 0) > 0
    ]

    return {
        "nodes":          G.number_of_nodes(),
        "edges":          G.number_of_edges(),
        "is_directed":    G.is_directed(),
        "year_range":     f"{min(years)}–{max(years)}" if years else "unknown",
        "seed_papers":    sum(
            1 for _, d in G.nodes(data=True) if d.get("is_seed")
        ),
        "graph_path":     str(GRAPH_PATH),
    }


def get_neighbors(
    G:       nx.DiGraph,
    node_id: str,
    direction: str = "both",
) -> dict:
    """
    Get neighboring nodes for a given paper.
    direction: 'out'  = papers this paper cites
               'in'   = papers that cite this paper
               'both' = both directions
    """
    if node_id not in G:
        return {"out": [], "in": []}

    result = {}
    if direction in ("out", "both"):
        result["out"] = [
            {"node_id": n, **G.edges[node_id, n]}
            for n in G.successors(node_id)
        ]
    if direction in ("in", "both"):
        result["in"] = [
            {"node_id": n, **G.edges[n, node_id]}
            for n in G.predecessors(node_id)
        ]

    return result