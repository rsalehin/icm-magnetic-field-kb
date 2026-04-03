# scripts/test_extractor.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.extraction.pdf_extractor import extract_paper

# Test on Murgia 2004 — our seed paper
pdf = Path("papers/2004_Murgia_astro-ph_0406225.pdf")
print(f"Extracting: {pdf.name} ...")

result = extract_paper(pdf)

print(f"\nnode_id        : {result['node_id']}")
print(f"arxiv_id       : {result['layer_1_bibliographic']['arxiv_id']}")
print(f"year           : {result['layer_1_bibliographic']['year']}")
print(f"total_pages    : {result['layer_1_bibliographic']['total_pages']}")
print(f"total_chunks   : {result['layer_2_chunks']['total_chunks']}")
print(f"sections found : {result['layer_2_chunks']['sections_detected'][:8]}")

print(f"\n--- First chunk ---")
c0 = result['layer_2_chunks']['chunks'][0]
print(f"chunk_id    : {c0['chunk_id']}")
print(f"section     : {c0['section_title']}")
print(f"tokens      : {c0['token_count']}")
print(f"has_eq      : {c0['content_flags']['has_equation']}")
print(f"text[:200]  : {c0['text'][:200]}")

print(f"\n--- Chunk with equation flag ---")
eq_chunks = [c for c in result['layer_2_chunks']['chunks']
             if c['content_flags']['has_equation']]
print(f"Chunks with equations: {len(eq_chunks)}")
if eq_chunks:
    print(f"Example: {eq_chunks[0]['text'][:200]}")

print(f"\n--- Chunk with numerical result ---")
num_chunks = [c for c in result['layer_2_chunks']['chunks']
              if c['content_flags']['has_numerical_result']]
print(f"Chunks with quantities: {len(num_chunks)}")
if num_chunks:
    print(f"Example: {num_chunks[0]['text'][:200]}")