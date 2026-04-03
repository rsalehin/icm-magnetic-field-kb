# scripts/diagnose_pdf.py
"""
Diagnose paragraph splitting on a single page.
Shows exactly what clean_page and split_into_paragraphs produce.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from src.extraction.text_cleaner import clean_page, join_soft_wrapped_lines, split_into_paragraphs

pdf  = Path("papers/2004_Murgia_astro-ph_0406225.pdf")
doc  = fitz.open(pdf)

for page_num in [1, 2, 3]:   # pages 2, 3, 4 (0-indexed)
    page     = doc[page_num]
    raw      = page.get_text()
    cleaned  = clean_page(raw)
    joined   = join_soft_wrapped_lines(cleaned)
    paras    = split_into_paragraphs(cleaned)

    print(f"\n{'='*60}")
    print(f"PAGE {page_num+1}")
    print(f"{'='*60}")
    print(f"RAW chars    : {len(raw)}")
    print(f"CLEANED chars: {len(cleaned)}")
    print(f"PARAGRAPHS   : {len(paras)}")
    print(f"\n--- CLEANED (first 600 chars) ---")
    print(repr(cleaned[:600]))
    print(f"\n--- PARAGRAPHS ---")
    for i, p in enumerate(paras):
        print(f"  [{i}] ({len(p)} chars) {p[:120]}")

doc.close()