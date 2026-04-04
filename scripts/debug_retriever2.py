# scripts/debug_retriever2.py
"""
Trace the exact execution path inside retrieve_paper_candidates.
"""
import sys
import faiss
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db          import get_connection
from src.storage.faiss_index import load_id_map, INDEX_PATH
from src.pipeline            import get_model
from src.query.planner       import plan_query

conn           = get_connection()
specter_model  = get_model()
specter_index  = faiss.read_index(str(INDEX_PATH))
specter_id_map = load_id_map()

query    = "What foundational methods were used to model intracluster magnetic fields?"
plan     = plan_query(query)
paper_k  = plan["budgets"]["paper_k"]
ef       = plan["explicit_filters"]

print(f"paper_k       : {paper_k}")
print(f"explicit_filters: {ef}")

# Step 1 — embed
query_vec = specter_model.encode(
    [query], convert_to_numpy=True, show_progress_bar=False
).astype(np.float32)
norm = np.linalg.norm(query_vec)
if norm > 0:
    query_vec = query_vec / norm

# Step 2 — FAISS
n_retrieve = min(paper_k * 20, specter_index.ntotal)
scores, rows = specter_index.search(query_vec, n_retrieve)
print(f"\nFAISS n_retrieve  : {n_retrieve}")
print(f"FAISS rows shape  : {rows.shape}")
print(f"Non-(-1) rows     : {sum(1 for r in rows[0] if r != -1)}")

# Step 3 — map to node_ids
row_to_chunk  = {v: k for k, v in specter_id_map.items()}
candidate_ids = set()
for row in rows[0]:
    if row == -1:
        continue
    chunk_id = row_to_chunk.get(int(row))
    if chunk_id:
        node_id = chunk_id.rsplit("__c", 1)[0]
        candidate_ids.add(node_id)
    if len(candidate_ids) >= paper_k * 3:
        break

print(f"\ncandidate_ids count: {len(candidate_ids)}")
print(f"Sample: {list(candidate_ids)[:3]}")

# Step 4 — build SQL exactly as retriever does
placeholders = ",".join(["?" for _ in candidate_ids])
conditions   = [f"node_id IN ({placeholders})"]
params       = list(candidate_ids)

print(f"\nSQL conditions: {conditions}")
print(f"params count  : {len(params)}")
print(f"year_min      : {ef.get('year_min')}")
print(f"year_max      : {ef.get('year_max')}")
print(f"journal       : {ef.get('journal')}")

if ef.get("year_min"):
    conditions.append("year >= ?")
    params.append(ef["year_min"])
if ef.get("year_max"):
    conditions.append("year <= ?")
    params.append(ef["year_max"])
if ef.get("journal"):
    conditions.append("journal ILIKE ?")
    params.append(f"%{ef['journal']}%")

where = " AND ".join(conditions)
sql   = f"""
    SELECT node_id FROM papers
    WHERE {where}
    ORDER BY citation_count DESC NULLS LAST
    LIMIT {paper_k}
"""
print(f"\nFull SQL:\n{sql}")
print(f"Total params: {len(params)}")

try:
    result = conn.execute(sql, params).fetchall()
    print(f"\nSQL returned: {len(result)} rows")
    for r in result[:5]:
        print(f"  {r[0]}")
except Exception as e:
    print(f"\nSQL ERROR: {e}")

conn.close()