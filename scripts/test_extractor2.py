# scripts/test_extractor2.py
"""
Test extractor on a second paper — different format, different era.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.pdf_extractor import extract_paper

papers = [
    "papers/2009_Bonafede_0905.3552.pdf",
    "papers/2010_Bonafede_1002.0594.pdf",
    "papers/2019_Loi_1902.05953.pdf",
]

for pdf_path in papers:
    p = Path(pdf_path)
    if not p.exists():
        print(f"SKIP (not found): {p.name}")
        continue

    result = extract_paper(p)
    l1     = result["layer_1_bibliographic"]
    l2     = result["layer_2_chunks"]

    print(f"\n{'='*55}")
    print(f"File     : {p.name}")
    print(f"node_id  : {result['node_id']}")
    print(f"arxiv_id : {l1['arxiv_id']}")
    print(f"year     : {l1['year']}")
    print(f"pages    : {l1['total_pages']}")
    print(f"chunks   : {l2['total_chunks']}")
    print(f"sections : {l2['sections_detected'][:5]}")

    # First clean chunk (skip any boilerplate)
    clean = [c for c in l2["chunks"] if c["token_count"] > 50]
    if clean:
        c = clean[0]
        print(f"\nFirst clean chunk ({c['token_count']} tokens):")
        print(f"  section : {c['section_title']}")
        print(f"  text    : {c['text'][:150]}")

    # Chunk size distribution
    sizes = [c["token_count"] for c in l2["chunks"]]
    print(f"\nToken count — min:{min(sizes)}  "
          f"max:{max(sizes)}  "
          f"avg:{sum(sizes)//len(sizes)}")
    small = sum(1 for s in sizes if s < 30)
    print(f"Chunks under 30 tokens (noise): {small}")