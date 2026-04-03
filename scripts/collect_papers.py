"""
collect_papers.py
─────────────────────────────────────────────────────────────────────────────
NASA ADS citation-graph crawler + arXiv PDF bulk downloader
For: Tautenburg ICM magnetic field research

Usage:
    python collect_papers.py

Requirements:
    pip install requests tqdm

Set your ADS token in ADS_TOKEN below (or as env var ADS_TOKEN).
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import json
import time
import logging
import requests
from pathlib import Path
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────

ADS_TOKEN    = os.environ.get("ADS_TOKEN", "YOUR_ADS_TOKEN_HERE")
OUTPUT_DIR   = Path("papers")          # where PDFs go
META_FILE    = Path("papers_meta.json")  # collected metadata
BIBTEX_FILE  = Path("papers.bib")

# Citation-graph depth: 1 = only direct refs/cites of seeds
#                        2 = also refs/cites of those (gets large fast)
GRAPH_DEPTH  = 1

# Pull both directions?
FETCH_REFERENCES = True   # papers your seeds cite  (foundational)
FETCH_CITATIONS  = True   # papers that cite your seeds (follow-up)

# Max citations to retrieve per paper (ADS cap: 2000)
MAX_RESULTS = 500

# Seconds to wait between ADS API calls (be polite)
API_SLEEP = 0.5

# ─── SEED PAPERS ─────────────────────────────────────────────────────────────
# Provide arXiv IDs *or* ADS bibcodes — the script handles both.
# Format: "arxiv:XXXXXXX" or ADS bibcode like "2004A&A...424..429M"

SEED_PAPERS = [
    "arxiv:astro-ph/0406225",   # Murgia+2004 — FARADAY, GRF model, A119
    # ── Add your other known seeds below ──
    # "arxiv:astro-ph/0302062", # Vogt & Enßlin 2003
    # "arxiv:astro-ph/0204439", # Enßlin & Vogt 2003
    # "arxiv:astro-ph/0108362", # Dolag+2001 RM-Sx correlation
    # "arxiv:astro-ph/0112525", # Dolag+2002 MHD cosmo sims
    # "arxiv:astro-ph/0107145", # Clarke+2001 statistical study
    # "arxiv:astro-ph/0205286", # Carilli & Taylor 2002 review ARA&A
]

# ─── ADS API HELPERS ─────────────────────────────────────────────────────────

ADS_BASE = "https://api.adsabs.harvard.edu/v1"
HEADERS  = {"Authorization": f"Bearer {ADS_TOKEN}"}

FL = "bibcode,title,author,year,abstract,identifier,arxiv_class,pub"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def ads_search(query: str, rows: int = MAX_RESULTS) -> list[dict]:
    """Run an ADS search query, return list of paper dicts."""
    params = {"q": query, "fl": FL, "rows": rows, "sort": "citation_count desc"}
    resp = requests.get(f"{ADS_BASE}/search/query", headers=HEADERS, params=params)
    if resp.status_code != 200:
        log.warning(f"ADS query failed ({resp.status_code}): {query[:80]}")
        return []
    docs = resp.json().get("response", {}).get("docs", [])
    time.sleep(API_SLEEP)
    return docs


def resolve_seed_to_bibcode(seed: str) -> str | None:
    """Convert 'arxiv:XXX' or raw arXiv ID to ADS bibcode."""
    # Clean up
    seed = seed.strip()
    if seed.startswith("arxiv:"):
        arxiv_id = seed[6:]
    elif re.match(r"^\d{4}\.\d{4,5}", seed) or "/" in seed:
        arxiv_id = seed
    else:
        # Assume it's already a bibcode
        return seed

    docs = ads_search(f"arxiv:{arxiv_id}", rows=1)
    if docs:
        return docs[0]["bibcode"]
    log.warning(f"Could not resolve seed: {seed}")
    return None


def get_references(bibcode: str) -> list[dict]:
    """Papers that bibcode cites (references)."""
    return ads_search(f"references(bibcode:{bibcode})")


def get_citations(bibcode: str) -> list[dict]:
    """Papers that cite bibcode."""
    return ads_search(f"citations(bibcode:{bibcode})")


# ─── ARXIV ID EXTRACTION ─────────────────────────────────────────────────────

def extract_arxiv_id(paper: dict) -> str | None:
    """Pull arXiv ID from ADS identifier field."""
    for ident in paper.get("identifier", []):
        ident_lower = ident.lower()
        if ident_lower.startswith("arxiv:"):
            return ident[6:]
        # Sometimes stored as raw IDs like "1234.5678" or "astro-ph/0406225"
        if re.match(r"^\d{4}\.\d{4,5}$", ident):
            return ident
        if re.match(r"^[a-z\-]+/\d{7}$", ident):
            return ident
    return None


# ─── PDF DOWNLOAD ─────────────────────────────────────────────────────────────

def download_pdf(arxiv_id: str, dest_path: Path) -> bool:
    """Download a PDF from arXiv. Returns True on success."""
    if dest_path.exists():
        return True  # already downloaded

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        resp = requests.get(url, timeout=30, stream=True)
        if resp.status_code == 200 and "pdf" in resp.headers.get("Content-Type", ""):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            time.sleep(0.8)  # polite delay
            return True
        else:
            log.warning(f"PDF not available for {arxiv_id} (status {resp.status_code})")
    except Exception as e:
        log.warning(f"Download error for {arxiv_id}: {e}")
    return False


# ─── BIBTEX EXPORT ───────────────────────────────────────────────────────────

def fetch_bibtex(bibcodes: list[str]) -> str:
    """Fetch BibTeX for a list of bibcodes from ADS."""
    if not bibcodes:
        return ""
    # ADS export endpoint accepts up to ~100 at a time
    chunk_size = 100
    all_bibtex = []
    for i in range(0, len(bibcodes), chunk_size):
        chunk = bibcodes[i:i + chunk_size]
        payload = {"bibcode": chunk}
        resp = requests.post(f"{ADS_BASE}/export/bibtex",
                             headers=HEADERS, json=payload)
        if resp.status_code == 200:
            all_bibtex.append(resp.json().get("export", ""))
        time.sleep(API_SLEEP)
    return "\n\n".join(all_bibtex)


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Step 1: Resolve seeds to ADS bibcodes ──
    log.info("=== Step 1: Resolving seed papers ===")
    seed_bibcodes = []
    for seed in SEED_PAPERS:
        bib = resolve_seed_to_bibcode(seed)
        if bib:
            log.info(f"  ✓ {seed}  →  {bib}")
            seed_bibcodes.append(bib)
        else:
            log.warning(f"  ✗ Could not resolve: {seed}")

    # ── Step 2: Crawl citation graph ──
    log.info(f"\n=== Step 2: Crawling citation graph (depth={GRAPH_DEPTH}) ===")
    all_papers: dict[str, dict] = {}   # bibcode → paper dict

    # Seed the first layer
    queue = list(seed_bibcodes)
    visited = set()

    for depth in range(GRAPH_DEPTH):
        next_layer = []
        for bibcode in tqdm(queue, desc=f"Depth {depth+1}"):
            if bibcode in visited:
                continue
            visited.add(bibcode)

            # Fetch references (papers this paper cites)
            if FETCH_REFERENCES:
                refs = get_references(bibcode)
                for p in refs:
                    bc = p["bibcode"]
                    if bc not in all_papers:
                        all_papers[bc] = p
                        next_layer.append(bc)
                log.info(f"  {bibcode}: {len(refs)} references")

            # Fetch citations (papers that cite this paper)
            if FETCH_CITATIONS:
                cites = get_citations(bibcode)
                for p in cites:
                    bc = p["bibcode"]
                    if bc not in all_papers:
                        all_papers[bc] = p
                        next_layer.append(bc)
                log.info(f"  {bibcode}: {len(cites)} citing papers")

        # Add seed papers themselves
        for bc in seed_bibcodes:
            if bc not in all_papers:
                docs = ads_search(f"bibcode:{bc}", rows=1)
                if docs:
                    all_papers[bc] = docs[0]

        queue = list(set(next_layer) - visited)
        log.info(f"  → {len(all_papers)} unique papers after depth {depth+1}")

    log.info(f"\nTotal collected: {len(all_papers)} papers")

    # ── Step 3: Save metadata ──
    log.info("\n=== Step 3: Saving metadata ===")
    with open(META_FILE, "w") as f:
        json.dump(all_papers, f, indent=2)
    log.info(f"  Saved to {META_FILE}")

    # ── Step 4: Download PDFs ──
    log.info("\n=== Step 4: Downloading PDFs ===")
    papers_with_arxiv = []
    papers_no_arxiv   = []

    for bibcode, paper in all_papers.items():
        arxiv_id = extract_arxiv_id(paper)
        if arxiv_id:
            papers_with_arxiv.append((bibcode, arxiv_id, paper))
        else:
            papers_no_arxiv.append((bibcode, paper))

    log.info(f"  {len(papers_with_arxiv)} papers have arXiv IDs → will download")
    log.info(f"  {len(papers_no_arxiv)} papers have no arXiv ID → skipped")

    success, fail = 0, 0
    for bibcode, arxiv_id, paper in tqdm(papers_with_arxiv, desc="Downloading PDFs"):
        year  = paper.get("year", "unknown")
        # First author last name
        authors = paper.get("author", ["Unknown"])
        first_author = authors[0].split(",")[0].replace(" ", "") if authors else "Unknown"
        # Sanitize filename
        title_short = re.sub(r'[^\w\s-]', '', paper.get("title", [""])[0])[:40].strip()
        safe_id = arxiv_id.replace("/", "_")
        filename = f"{year}_{first_author}_{safe_id}.pdf"
        dest = OUTPUT_DIR / filename

        ok = download_pdf(arxiv_id, dest)
        if ok:
            success += 1
        else:
            fail += 1

    log.info(f"  Downloaded: {success}  |  Failed/unavailable: {fail}")

    # ── Step 5: Export BibTeX ──
    log.info("\n=== Step 5: Exporting BibTeX ===")
    all_bibcodes = list(all_papers.keys())
    bibtex = fetch_bibtex(all_bibcodes)
    with open(BIBTEX_FILE, "w") as f:
        f.write(bibtex)
    log.info(f"  Saved {len(all_bibcodes)} entries to {BIBTEX_FILE}")

    # ── Step 6: Print summary ──
    log.info("\n=== Summary ===")
    log.info(f"  Total unique papers found : {len(all_papers)}")
    log.info(f"  PDFs downloaded           : {success}")
    log.info(f"  Metadata file             : {META_FILE}")
    log.info(f"  BibTeX file               : {BIBTEX_FILE}")
    log.info(f"  PDF folder                : {OUTPUT_DIR}/")

    # Papers without arXiv (print so you can find them manually)
    if papers_no_arxiv:
        log.info(f"\n  Papers with no arXiv ID ({len(papers_no_arxiv)}) — get manually:")
        for bibcode, paper in papers_no_arxiv[:20]:
            title = paper.get("title", ["?"])[0][:60]
            year  = paper.get("year", "?")
            log.info(f"    [{year}] {bibcode}  {title}")


if __name__ == "__main__":
    main()
