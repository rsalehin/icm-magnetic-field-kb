# scripts/test_faiss.py
"""
Test Step 7 — FAISS index init, add, search, idempotency.
Uses synthetic random vectors — no GPU or model loading needed.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.faiss_index import (
    get_or_create_index, load_id_map, save_id_map,
    save_index, add_embeddings, search, get_index_stats,
    INDEX_PATH, ID_MAP_PATH,
)

# ── Clean slate ───────────────────────────────────────────────────────────────
if INDEX_PATH.exists():  INDEX_PATH.unlink()
if ID_MAP_PATH.exists(): ID_MAP_PATH.unlink()

# ── Create index ──────────────────────────────────────────────────────────────
print("\n[1] Creating index...")
index  = get_or_create_index()
id_map = load_id_map()

# ── Add synthetic embeddings ──────────────────────────────────────────────────
print("\n[2] Adding 10 synthetic chunk embeddings...")
np.random.seed(42)
chunk_ids = [f"arxiv:astro-ph/0406225__c{i:04d}" for i in range(10)]
vectors   = np.random.randn(10, 768).astype(np.float32)

index, id_map = add_embeddings(index, id_map, chunk_ids, vectors)
save_index(index)
save_id_map(id_map)

# ── Idempotency check ─────────────────────────────────────────────────────────
print("\n[3] Adding same chunks again (should skip)...")
index, id_map = add_embeddings(index, id_map, chunk_ids, vectors)

# ── Add 5 more ────────────────────────────────────────────────────────────────
print("\n[4] Adding 5 new chunks...")
new_ids  = [f"arxiv:astro-ph/0406225__c{i:04d}" for i in range(10, 15)]
new_vecs = np.random.randn(5, 768).astype(np.float32)
index, id_map = add_embeddings(index, id_map, new_ids, new_vecs)
save_index(index)
save_id_map(id_map)

# ── Search ────────────────────────────────────────────────────────────────────
print("\n[5] Searching with query vector (top 3)...")
query = vectors[0]   # should return chunk c0000 as top hit
results = search(index, id_map, query, top_k=3)
for r in results:
    print(f"  {r['chunk_id']}  score={r['score']:.4f}  row={r['faiss_row']}")

print(f"\n  Top result is c0000: "
      f"{'✓' if 'c0000' in results[0]['chunk_id'] else '✗'}")

# ── Reload from disk and verify ───────────────────────────────────────────────
print("\n[6] Reload index from disk...")
index2  = get_or_create_index()
id_map2 = load_id_map()
stats   = get_index_stats(index2, id_map2)
print(f"  total_vectors : {stats['total_vectors']}  (expected 15)")
print(f"  total_mapped  : {stats['total_mapped']}   (expected 15)")
print(f"  dimension     : {stats['dimension']}")
print(f"  index_type    : {stats['index_type']}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
INDEX_PATH.unlink()
ID_MAP_PATH.unlink()
print("\nAll FAISS tests passed." if stats["total_vectors"] == 15
      else "ERROR: vector count mismatch.")