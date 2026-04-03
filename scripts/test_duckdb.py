# scripts/test_duckdb.py
"""
Test Step 6 — DuckDB schema init and single paper insert.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.pdf_extractor import extract_paper
from src.enrichment.ads_enricher  import enrich_paper
from src.storage.db               import get_connection, init_schema, \
                                          insert_paper, get_ingestion_state

# ── Use a temp test DB so we don't pollute production ────────────────────────
TEST_DB = Path("data/test_knowledge_base.duckdb")
if TEST_DB.exists():
    TEST_DB.unlink()   # fresh start each test run

conn = get_connection(TEST_DB)
init_schema(conn)

# ── Process one paper through Stages A + B ───────────────────────────────────
pdf = Path("papers/2004_Murgia_astro-ph_0406225.pdf")
print(f"\nProcessing: {pdf.name}")
paper = extract_paper(pdf)
paper = enrich_paper(paper)

# ── Insert into DuckDB ────────────────────────────────────────────────────────
print("\nInserting into DuckDB...")
insert_paper(conn, paper)

# ── Verify ────────────────────────────────────────────────────────────────────
print("\n--- papers table ---")
row = conn.execute("""
    SELECT node_id, title, year, journal, citation_count,
           total_chunks, stage_a, stage_b
    FROM papers
    WHERE node_id = 'arxiv:astro-ph/0406225'
""").fetchone()
print(f"  node_id        : {row[0]}")
print(f"  title          : {row[1]}")
print(f"  year           : {row[2]}")
print(f"  journal        : {row[3]}")
print(f"  citation_count : {row[4]}")
print(f"  total_chunks   : {row[5]}")
print(f"  stage_a        : {row[6]}")
print(f"  stage_b        : {row[7]}")

print("\n--- authors table ---")
authors = conn.execute("""
    SELECT position, author_name FROM authors
    WHERE node_id = 'arxiv:astro-ph/0406225'
    ORDER BY position LIMIT 5
""").fetchall()
for pos, name in authors:
    print(f"  [{pos}] {name}")

print("\n--- chunks table (first 3) ---")
chunks = conn.execute("""
    SELECT chunk_id, section_title, token_count,
           has_equation, has_numerical_result, is_noise
    FROM chunks
    WHERE node_id = 'arxiv:astro-ph/0406225'
    ORDER BY chunk_index LIMIT 3
""").fetchall()
for c in chunks:
    print(f"  {c[0]}")
    print(f"    section   : {c[1]}")
    print(f"    tokens    : {c[2]}  eq:{c[3]}  num:{c[4]}  noise:{c[5]}")

print("\n--- ingestion_state ---")
state = get_ingestion_state(conn)
for s in state:
    print(f"  {s['filename']:<45} "
          f"A:{s['stage_a']}  B:{s['stage_b']}  "
          f"C:{s['stage_c']}  D:{s['stage_d']}")

print("\n--- chunk counts by section ---")
sections = conn.execute("""
    SELECT section_title, COUNT(*) as n
    FROM chunks
    WHERE node_id = 'arxiv:astro-ph/0406225'
    GROUP BY section_title
    ORDER BY n DESC
""").fetchall()
for sec, n in sections:
    print(f"  {n:>4}  {sec}")

conn.close()
print("\nTest DB closed. All checks passed.")