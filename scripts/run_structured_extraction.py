# scripts/run_structured_extraction.py
"""
Run full structured extraction over all 251 papers.
Estimated time: ~45-60 minutes.
Resumable — skips already-extracted papers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db import get_connection
from src.extraction.structured_extractor import run_extraction_batch

# Ensure table exists
conn = get_connection()
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
conn.commit()

stats = run_extraction_batch(conn, sleep=0.2, resume=True)
conn.close()