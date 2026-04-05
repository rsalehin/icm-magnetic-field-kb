# scripts/test_structured_extraction.py
"""
Test Step 12 — structured extraction on 3 papers.
Verifies Qwen3 extraction quality and DuckDB storage.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db import get_connection
from src.extraction.structured_extractor import (
    extract_paper_metadata,
    store_extraction,
    assemble_extraction_text,
    is_extracted,
)

conn = get_connection()

# Ensure table exists
conn.execute("""
    CREATE TABLE IF NOT EXISTS paper_extractions (
        node_id           VARCHAR PRIMARY KEY,
        methods           JSON,
        key_quantities    JSON,
        scientific_claims JSON,
        physical_domain   JSON,
        instruments       JSON,
        clusters          JSON,
        extraction_model  VARCHAR,
        extracted_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    )
""")

# Test on 3 key papers
test_papers = [
    "arxiv:astro-ph/0406225",  # Murgia 2004 — GRF model
    "arxiv:1002.0594",          # Bonafede 2010 — Coma cluster
    "arxiv:2201.12207",         # Recent LOFAR paper
]

for node_id in test_papers:
    print(f"\n{'='*60}")
    print(f"Extracting: {node_id}")

    # Get paper metadata
    row = conn.execute(
        "SELECT title, year FROM papers WHERE node_id = ?",
        [node_id]
    ).fetchone()

    if not row:
        print(f"  Paper not found in DB")
        continue

    title, year = row
    print(f"  Title: {title[:55]}")
    print(f"  Year : {year}")

    # Get chunks
    chunk_rows = conn.execute("""
        SELECT section_title, text FROM chunks
        WHERE node_id = ? AND is_noise = FALSE AND token_count > 30
        ORDER BY chunk_index
    """, [node_id]).fetchall()

    chunks = [{"section_title": r[0], "text": r[1]} for r in chunk_rows]
    print(f"  Chunks: {len(chunks)}")

    # Show assembled text sample
    text = assemble_extraction_text(chunks)
    print(f"  Assembled text ({len(text)} chars): {text[:100]}...")

    # Extract
    data = extract_paper_metadata(node_id, title, year, chunks)

    if data is None:
        print(f"  ✗ Extraction failed")
        continue

    # Store
    store_extraction(conn, node_id, data, "qwen3:14b")

    # Print results
    print(f"\n  methods          : {data.get('methods', [])}")
    print(f"  physical_domain  : {data.get('physical_domain', [])}")
    print(f"  instruments      : {data.get('instruments', [])}")
    print(f"  clusters         : {data.get('clusters', [])}")
    print(f"\n  key_quantities ({len(data.get('key_quantities',[]))}):")
    for q in data.get("key_quantities", [])[:5]:
        print(f"    {q.get('name','?')} = {q.get('value','?')} "
              f"{q.get('unit','?')} | {q.get('cluster','?')} "
              f"| {q.get('context','?')[:40]}")
    print(f"\n  scientific_claims:")
    for c in data.get("scientific_claims", []):
        print(f"    - {c[:80]}")

# Verify storage
print(f"\n{'='*60}")
print("Verifying storage...")
count = conn.execute(
    "SELECT COUNT(*) FROM paper_extractions"
).fetchone()[0]
print(f"  Extractions in DB: {count}")

# Test SQL query on extracted data
print("\nTest SQL query: find papers mentioning GRF method...")
rows = conn.execute("""
    SELECT node_id, methods FROM paper_extractions
    WHERE methods::VARCHAR ILIKE '%GRF%'
       OR methods::VARCHAR ILIKE '%Gaussian%'
""").fetchall()
print(f"  Found {len(rows)} papers:")
for r in rows[:5]:
    print(f"    {r[0]}: {r[1][:60]}")

conn.close()
print("\nStructured extraction test complete.")