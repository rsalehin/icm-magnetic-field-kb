# src/enrichment/ads_enricher.py
"""
Stage B — ADS metadata enrichment.
Takes a node_id (arxiv_id), queries NASA ADS API,
fills in Layer 1 bibliographic fields.
No PDF needed — purely network + JSON.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

ADS_TOKEN = os.getenv("ADS_TOKEN")
ADS_BASE  = "https://api.adsabs.harvard.edu/v1"
HEADERS   = {"Authorization": f"Bearer {ADS_TOKEN}"}

# Fields to retrieve from ADS
FL = ",".join([
    "bibcode", "title", "author", "year",
    "pub", "volume", "page", "abstract",
    "keyword", "citation_count", "doi",
    "identifier",
])


def _query_ads(query: str, rows: int = 1) -> list[dict]:
    """Run a search query against ADS. Returns list of paper dicts."""
    params = {"q": query, "fl": FL, "rows": rows}
    try:
        resp = requests.get(
            f"{ADS_BASE}/search/query",
            headers=HEADERS,
            params=params,
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("response", {}).get("docs", [])
        else:
            print(f"  ADS error {resp.status_code}: {resp.text[:100]}")
            return []
    except requests.RequestException as e:
        print(f"  Network error: {e}")
        return []


def enrich_from_arxiv_id(arxiv_id: str) -> dict | None:
    """
    Query ADS by arxiv_id.
    Returns filled Layer 1 dict or None if not found.
    """
    # ADS accepts both formats: astro-ph/0406225 and 0905.3552
    docs = _query_ads(f"arxiv:{arxiv_id}", rows=1)

    if not docs:
        # Fallback: try identifier field search
        docs = _query_ads(f"identifier:{arxiv_id}", rows=1)

    if not docs:
        return None

    doc = docs[0]

    # Extract DOI from identifier list
    doi = None
    for ident in doc.get("identifier", []):
        if ident.startswith("10."):
            doi = ident
            break

    return {
        "arxiv_id":       arxiv_id,
        "bibcode":        doc.get("bibcode"),
        "doi":            doi,
        "title":          doc.get("title", [None])[0],
        "authors":        doc.get("author", []),
        "year":           str(doc.get("year", "")),
        "journal":        doc.get("pub"),
        "volume":         doc.get("volume"),
        "pages":          doc.get("page", [None])[0] if doc.get("page") else None,
        "abstract":       doc.get("abstract"),
        "keywords":       doc.get("keyword", []),
        "citation_count": doc.get("citation_count"),
    }


def enrich_paper(paper_dict: dict, sleep: float = 0.3) -> dict:
    """
    Takes a Stage A output dict, queries ADS, merges Layer 1.
    Updates meta.stage_b status.
    Returns the enriched dict.
    """
    arxiv_id = paper_dict["layer_1_bibliographic"]["arxiv_id"]
    print(f"  Querying ADS for: {arxiv_id} ...", end=" ", flush=True)

    ads_data = enrich_from_arxiv_id(arxiv_id)
    time.sleep(sleep)  # polite rate limiting

    if ads_data:
        # Merge ADS data into Layer 1, preserving pdf_path and total_pages
        existing = paper_dict["layer_1_bibliographic"]
        paper_dict["layer_1_bibliographic"] = {
            **existing,    # keeps pdf_path, total_pages
            **ads_data,    # overwrites with ADS data
        }
        paper_dict["meta"]["stage_b"] = "done"
        print(f"✓  {ads_data.get('title', '')[:50]}")
    else:
        paper_dict["meta"]["stage_b"] = "failed"
        print("✗  not found on ADS")

    return paper_dict