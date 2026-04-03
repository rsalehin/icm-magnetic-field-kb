# scripts/test_duckdb_idempotent.py
"""
Test idempotency — insert same paper twice, confirm second is skipped.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.pdf_extractor import extract_paper
from src.enrichment.ads_enricher  import enrich_paper
from src.storage.db               import get_connection, init_schema, \
                                          insert_paper

TEST_DB = Path("data/test_idempotent.duckdb")
if TEST_DB.exists():
    TEST_DB.unlink()

conn = get_connection(TEST_DB)
init_schema(conn)

pdf   = Path("papers/2004_Murgia_astro-ph_0406225.pdf")
paper = extract_paper(pdf)
paper = enrich_paper(paper)

print("\n--- First insert ---")
insert_paper(conn, paper)

print("\n--- Second insert (should skip) ---")
insert_paper(conn, paper)

count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
print(f"\npapers table : {count} row  (expected 1)")
print(f"chunks table : {chunks} rows (expected 236)")

conn.close()
TEST_DB.unlink()
print("Idempotency confirmed." if count == 1 else "ERROR: duplicate rows found.")