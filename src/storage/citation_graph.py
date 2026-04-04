# src/storage/citation_graph.py
"""
Step 11 — Citation edge construction from NASA ADS.
Queries ADS references endpoint per paper bibcode.
Adds typed directed edges to NetworkX graph.
Saves updated graph to disk.
"""

import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import networkx as nx

from src.storage.graph import (
    get_or_create_graph, save_graph,
    add_paper_node,
)
from src.storage.db import get_connection

load_dotenv()
ADS_TOKEN = os.getenv("ADS_TOKEN")
ADS_BASE  = "https://api.adsabs.harvard.edu/v1"
HEADERS   = {"Authorization": f"Bearer {ADS_TOKEN}"}

# Edge type weights
EDGE_WEIGHTS = {
    "foundational":    1.0,
    "methodological":  0.9,
    "data_source":     0.8,
    "comparison":      0.7,
    "incidental":      0.4,
}

# Fields to retrieve for referenced papers
REF_FL = "bibcode,title,author,year,pub,identifier"


# ── ADS reference fetcher ─────────────────────────────────────────────────────

def fetch_references(bibcode: str, sleep: float = 0.3) -> list[dict]:
    """
    Fetch all papers referenced by bibcode from ADS.
    Returns list of paper dicts.
    """
    params = {
        "q":    f"references(bibcode:{bibcode})",
        "fl":   REF_FL,
        "rows": 500,
    }
    try:
        resp = requests.get(
            f"{ADS_BASE}/search/query",
            headers=HEADERS,
            params=params,
            timeout=15,
        )
        time.sleep(sleep)
        if resp.status_code == 200:
            return resp.json().get("response", {}).get("docs", [])
        else:
            print(f"    ADS error {resp.status_code} for {bibcode}")
            return []
    except requests.RequestException as e:
        print(f"    Network error: {e}")
        return []


def extract_arxiv_id(paper: dict) -> str | None:
    """Extract arXiv ID from ADS identifier field."""
    for ident in paper.get("identifier", []):
        ident_lower = ident.lower()
        if ident_lower.startswith("arxiv:"):
            return ident[6:]
        if ident.startswith("astro-ph/") or ident.startswith("hep-"):
            return ident
    return None


# ── Edge type inference ───────────────────────────────────────────────────────

# Known foundational papers in ICM magnetic field literature
FOUNDATIONAL_BIBCODES = {
    "2002ARA&A..40..319C",  # Carilli & Taylor review
    "2004A&A...424..429M",  # Murgia 2004
    "2001ApJ...547L.111C",  # Clarke 2001
    "1991MNRAS.250..726T",  # Tribble 1991
    "2003A&A...401..835E",  # Ensslin & Vogt 2003
    "2003A&A...412..373V",  # Vogt & Ensslin 2003
    "2001A&A...378..777D",  # Dolag 2001
    "2002A&A...387..383D",  # Dolag 2002
}

def infer_citation_role(ref_bibcode: str) -> str:
    """
    Infer citation role from bibcode.
    Uses known foundational papers list as heuristic.
    """
    if ref_bibcode in FOUNDATIONAL_BIBCODES:
        return "foundational"
    return "incidental"


# ── Graph edge builder ────────────────────────────────────────────────────────

def add_citation_edges_from_ads(
    G:       nx.DiGraph,
    conn,
    limit:   int = None,
    sleep:   float = 0.3,
) -> tuple[nx.DiGraph, dict]:
    """
    For each paper in DuckDB with a bibcode, fetch its references
    from ADS and add directed edges to the graph.

    Returns (updated_G, stats_dict).
    """
    # Get all papers with bibcodes
    rows = conn.execute("""
        SELECT node_id, bibcode, title, year
        FROM papers
        WHERE bibcode IS NOT NULL
        ORDER BY year
    """).fetchall()

    if limit:
        rows = rows[:limit]

    total_papers = len(rows)
    total_edges  = 0
    total_new_nodes = 0
    failed       = 0

    print(f"Building citation edges for {total_papers} papers...")

    for i, (node_id, bibcode, title, year) in enumerate(rows):
        print(f"  [{i+1}/{total_papers}] {bibcode} ({year})")

        if not bibcode:
            continue

        refs = fetch_references(bibcode, sleep=sleep)

        if not refs:
            failed += 1
            continue

        paper_edges = 0
        for ref in refs:
            ref_bibcode = ref.get("bibcode")
            if not ref_bibcode:
                continue

            # Build target node_id from arXiv ID if available
            arxiv_id = extract_arxiv_id(ref)
            if arxiv_id:
                target_node_id = f"arxiv:{arxiv_id}"
            else:
                target_node_id = f"bibcode:{ref_bibcode}"

            # Add stub node if not in graph
            if target_node_id not in G:
                G.add_node(target_node_id, **{
                    "arxiv_id":       arxiv_id or "",
                    "bibcode":        ref_bibcode,
                    "title":          ref.get("title", [None])[0] or "",
                    "year":           int(ref.get("year", 0)) if ref.get("year") else 0,
                    "journal":        ref.get("pub", ""),
                    "citation_count": 0,
                    "total_chunks":   0,
                    "stage_a":        "pending",
                    "stage_b":        "pending",
                    "stage_c":        "pending",
                    "stage_d":        "pending",
                    "is_seed":        False,
                    "is_stub":        True,
                })
                total_new_nodes += 1

            # Infer role
            role   = infer_citation_role(ref_bibcode)
            weight = EDGE_WEIGHTS.get(role, 0.4)

            # Add directed edge: source → target (source cites target)
            G.add_edge(node_id, target_node_id, **{
                "citation_role":    role,
                "weight":           weight,
                "context_chunk_id": "",
            })
            paper_edges += 1

        total_edges += paper_edges
        print(f"    → {paper_edges} references added")

    stats = {
        "papers_processed": total_papers,
        "total_edges":      total_edges,
        "new_stub_nodes":   total_new_nodes,
        "failed":           failed,
        "total_nodes":      G.number_of_nodes(),
    }

    return G, stats


# ── Co-citation edges ─────────────────────────────────────────────────────────

def add_co_citation_edges(
    G:          nx.DiGraph,
    min_shared: int = 3,
) -> nx.DiGraph:
    """
    Add undirected CO_CITED edges between papers that share
    at least min_shared common references.
    Strong signal: co-cited papers address the same sub-problem.
    """
    # Get all corpus nodes (non-stub)
    corpus_nodes = [
        n for n, d in G.nodes(data=True)
        if not d.get("is_stub", False)
    ]

    print(f"\nBuilding co-citation edges for {len(corpus_nodes)} corpus papers...")

    added = 0
    for i, node_a in enumerate(corpus_nodes):
        refs_a = set(G.successors(node_a))
        if not refs_a:
            continue
        for node_b in corpus_nodes[i+1:]:
            refs_b   = set(G.successors(node_b))
            shared   = refs_a & refs_b
            if len(shared) >= min_shared:
                G.add_edge(node_a, node_b, **{
                    "citation_role": "co_cited",
                    "weight":        min(1.0, len(shared) / 10),
                    "shared_count":  len(shared),
                })
                added += 1

    print(f"  Added {added} co-citation edges (min_shared={min_shared})")
    return G