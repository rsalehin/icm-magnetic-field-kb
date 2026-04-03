# scripts/check_pdf_quality.py
import fitz  # pymupdf
import os
from pathlib import Path

pdf_dir = Path("papers")
pdfs = sorted(pdf_dir.glob("*.pdf"))

print(f"Total PDFs: {len(pdfs)}\n")
print(f"{'File':<55} {'Pages':>6} {'Chars/pg':>9} {'Type':<12}")
print("-" * 85)

text_based, image_based, mixed = 0, 0, 0

for pdf_path in pdfs[:20]:  # sample first 20
    doc = fitz.open(pdf_path)
    total_chars = sum(len(page.get_text()) for page in doc)
    pages = len(doc)
    chars_per_page = total_chars / pages if pages else 0

    if chars_per_page > 500:
        kind = "text-based"
        text_based += 1
    elif chars_per_page > 50:
        kind = "mixed"
        mixed += 1
    else:
        kind = "IMAGE/SCAN"
        image_based += 1

    print(f"{pdf_path.name:<55} {pages:>6} {chars_per_page:>9.0f} {kind:<12}")
    doc.close()

print(f"\nSample summary (first 20):")
print(f"  Text-based : {text_based}")
print(f"  Mixed      : {mixed}")
print(f"  Image/scan : {image_based}")