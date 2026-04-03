# scripts/test_batch_small.py
"""
Batch test on 5 papers — verify no crashes, check chunk stats.
"""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.pdf_extractor import extract_paper

# Pick 5 papers spanning different years and formats
test_papers = [
    "papers/1999_Feretti_astro-ph_9810305.pdf",
    "papers/2004_Murgia_astro-ph_0406225.pdf",
    "papers/2012_Vacca_1201.4119.pdf",
    "papers/2019_Loi_1902.05953.pdf",
    "papers/2024_Vacca_2410.08697.pdf",
]

results  = []
failures = []

for pdf_path in test_papers:
    p = Path(pdf_path)
    if not p.exists():
        print(f"SKIP: {p.name}")
        continue
    try:
        result = extract_paper(p)
        l1 = result["layer_1_bibliographic"]
        l2 = result["layer_2_chunks"]
        sizes = [c["token_count"] for c in l2["chunks"]]
        noise = sum(1 for s in sizes if s < 30)
        results.append({
            "file":    p.name,
            "node_id": result["node_id"],
            "pages":   l1["total_pages"],
            "chunks":  l2["total_chunks"],
            "avg_tok": sum(sizes) // len(sizes) if sizes else 0,
            "noise":   noise,
            "status":  "ok",
        })
    except Exception as e:
        failures.append({"file": p.name, "error": str(e)})
        traceback.print_exc()

print(f"\n{'File':<45} {'Pages':>5} {'Chunks':>7} "
      f"{'Avg tok':>8} {'Noise':>6} {'Status'}")
print("-" * 80)
for r in results:
    print(f"{r['file']:<45} {r['pages']:>5} {r['chunks']:>7} "
          f"{r['avg_tok']:>8} {r['noise']:>6}   {r['status']}")

if failures:
    print(f"\nFAILURES ({len(failures)}):")
    for f in failures:
        print(f"  {f['file']}: {f['error']}")
else:
    print(f"\nAll {len(results)} papers processed successfully.")