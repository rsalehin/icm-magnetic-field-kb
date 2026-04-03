# scripts/test_ads_enricher.py
"""
Test Stage B ADS enrichment on 3 papers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.pdf_extractor import extract_paper
from src.enrichment.ads_enricher  import enrich_paper

test_papers = [
    "papers/2004_Murgia_astro-ph_0406225.pdf",
    "papers/2009_Bonafede_0905.3552.pdf",
    "papers/2019_Loi_1902.05953.pdf",
]

for pdf_path in test_papers:
    p = Path(pdf_path)
    if not p.exists():
        print(f"SKIP: {p.name}")
        continue

    print(f"\n{'='*55}")
    print(f"File: {p.name}")

    # Stage A
    paper = extract_paper(p)

    # Stage B
    paper = enrich_paper(paper)

    l1 = paper["layer_1_bibliographic"]
    print(f"  bibcode        : {l1['bibcode']}")
    print(f"  title          : {l1['title']}")
    print(f"  authors        : {l1['authors'][:3]}")
    print(f"  journal        : {l1['journal']}")
    print(f"  year           : {l1['year']}")
    print(f"  doi            : {l1['doi']}")
    print(f"  citation_count : {l1['citation_count']}")
    print(f"  abstract[:100] : {str(l1['abstract'])[:100]}")
    print(f"  stage_b        : {paper['meta']['stage_b']}")