# scripts/debug_retriever.py
import sys, faiss, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db          import get_connection
from src.storage.faiss_index import load_id_map, INDEX_PATH
from src.pipeline            import get_model

conn           = get_connection()
specter_model  = get_model()
specter_index  = faiss.read_index(str(INDEX_PATH))
specter_id_map = load_id_map()

query = "intracluster magnetic field models"

# Step 1 — embed query
query_vec = specter_model.encode(
    [query], convert_to_numpy=True, show_progress_bar=False
).astype(np.float32)
norm = np.linalg.norm(query_vec)
if norm > 0:
    query_vec = query_vec / norm

# Step 2 — FAISS search
scores, rows = specter_index.search(query_vec, 20)
print(f"FAISS rows returned: {rows[0][:5]}")
print(f"FAISS scores: {scores[0][:5]}")

# Step 3 — map rows to node_ids
row_to_chunk = {v: k for k, v in specter_id_map.items()}
candidate_ids = set()
for row in rows[0]:
    if row == -1:
        continue
    chunk_id = row_to_chunk.get(int(row))
    if chunk_id:
        node_id = chunk_id.rsplit("__c", 1)[0]
        candidate_ids.add(node_id)

print(f"\ncandidate_ids ({len(candidate_ids)}):")
for cid in list(candidate_ids)[:5]:
    print(f"  {cid}")

# Step 4 — DuckDB lookup
print(f"\nChecking DuckDB for these node_ids...")
sample = list(candidate_ids)[:3]
for nid in sample:
    row = conn.execute(
        "SELECT node_id, year FROM papers WHERE node_id = ?", [nid]
    ).fetchone()
    print(f"  {nid} → {row}")

# Step 5 — IN clause test
print(f"\nIN clause test with {len(candidate_ids)} ids...")
placeholders = ",".join(["?" for _ in candidate_ids])
sql = f"SELECT node_id FROM papers WHERE node_id IN ({placeholders}) LIMIT 5"
try:
    result = conn.execute(sql, list(candidate_ids)).fetchall()
    print(f"  Returned {len(result)} rows")
    for r in result:
        print(f"    {r[0]}")
except Exception as e:
    print(f"  SQL error: {e}")

conn.close()