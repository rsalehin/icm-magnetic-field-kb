# scripts/test_pipeline_single.py
"""
Test Step 9 — Full pipeline on a single paper.
Stages A + B + C (GPU embedding) + D (all three backends).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import get_model, ingest_paper
from src.storage.db          import get_connection, init_schema
from src.storage.faiss_index import (
    get_or_create_index, load_id_map,
    save_index, save_id_map, INDEX_PATH, ID_MAP_PATH,
)
from src.storage.graph import (
    get_or_create_graph, save_graph, GRAPH_PATH,
)

# ── Clean test environment ────────────────────────────────────────────────────
TEST_DB = Path("data/test_pipeline.duckdb")
TEST_INDEX  = INDEX_PATH.parent / "test_chunks.index"
TEST_IDMAP  = INDEX_PATH.parent / "test_chunks_id_map.json"

for f in [TEST_DB, TEST_INDEX, TEST_IDMAP, GRAPH_PATH]:
    if f.exists():
        f.unlink()

# ── Initialise backends ───────────────────────────────────────────────────────
print("\n[1] Initialising backends...")
conn   = get_connection(TEST_DB)
init_schema(conn)
index  = get_or_create_index()
id_map = load_id_map()
G      = get_or_create_graph()

# ── Load model ────────────────────────────────────────────────────────────────
print("\n[2] Loading SPECTER2...")
model = get_model()

# ── Run pipeline on Murgia 2004 ───────────────────────────────────────────────
print("\n[3] Running full pipeline...")
pdf = Path("papers/2004_Murgia_astro-ph_0406225.pdf")
paper, id_map, G = ingest_paper(
    pdf, conn, index, id_map, G, model
)

# ── Save all backends ─────────────────────────────────────────────────────────
save_index(index)
save_id_map(id_map)
save_graph(G)

# ── Verify DuckDB ─────────────────────────────────────────────────────────────
print("\n[4] Verifying DuckDB...")
row = conn.execute("""
    SELECT node_id, title, year, total_chunks,
           stage_a, stage_b, stage_c, stage_d
    FROM papers WHERE node_id = 'arxiv:astro-ph/0406225'
""").fetchone()
print(f"  title        : {row[1][:50]}")
print(f"  year         : {row[2]}")
print(f"  total_chunks : {row[3]}")
print(f"  stages       : A:{row[4]} B:{row[5]} C:{row[6]} D:{row[7]}")

embedded_count = conn.execute("""
    SELECT COUNT(*) FROM chunks
    WHERE node_id = 'arxiv:astro-ph/0406225'
    AND faiss_index_id IS NOT NULL
""").fetchone()[0]
print(f"  chunks with FAISS id : {embedded_count}")

# ── Verify FAISS ──────────────────────────────────────────────────────────────
print("\n[5] Verifying FAISS...")
print(f"  vectors in index : {index.ntotal}")
print(f"  entries in id_map: {len(id_map)}")

# ── Verify NetworkX ───────────────────────────────────────────────────────────
print("\n[6] Verifying NetworkX...")
print(f"  nodes : {G.number_of_nodes()}")
node_data = G.nodes["arxiv:astro-ph/0406225"]
print(f"  title : {node_data.get('title', '')[:50]}")
print(f"  year  : {node_data.get('year')}")

# ── Idempotency check ─────────────────────────────────────────────────────────
print("\n[7] Idempotency check (re-run same paper)...")
paper2, id_map, G = ingest_paper(
    pdf, conn, index, id_map, G, model
)
print(f"  vectors after re-run: {index.ntotal}  (should be unchanged)")

# ── Cleanup ───────────────────────────────────────────────────────────────────
conn.close()
for f in [TEST_DB, GRAPH_PATH]:
    if f.exists(): f.unlink()
TEST_INDEX.unlink() if TEST_INDEX.exists() else None
TEST_IDMAP.unlink() if TEST_IDMAP.exists() else None

print("\nPipeline test complete.")