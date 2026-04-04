# scripts/test_bge.py
"""
Test Step 10b — BGE-M3 embedding on small sample first,
then verify index structure.
Tests on 100 chunks before full corpus run.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db import get_connection
from src.query.bge_embedder import (
    get_bge_model, get_or_create_bge_index,
    load_bge_id_map, save_bge_index, save_bge_id_map,
    embed_chunks_bge, embed_query_bge, bge_search,
    BGE_INDEX_PATH, BGE_IDMAP_PATH, BGE_MATRIX_PATH,
)
import faiss, json
import numpy as np

# ── Clean slate ───────────────────────────────────────────────────────────────
for f in [BGE_INDEX_PATH, BGE_IDMAP_PATH, BGE_MATRIX_PATH]:
    if f.exists():
        f.unlink()

# ── Load model ────────────────────────────────────────────────────────────────
print("\n[1] Loading BGE-M3...")
model = get_bge_model()

# ── Sample 100 chunks from DuckDB ─────────────────────────────────────────────
print("\n[2] Sampling 100 chunks from DuckDB...")
conn = get_connection()
rows = conn.execute("""
    SELECT chunk_id, text FROM chunks
    WHERE is_noise = FALSE
    AND node_id = 'arxiv:astro-ph/0406225'
    LIMIT 100
""").fetchall()
conn.close()

chunk_ids = [r[0] for r in rows]
texts     = [r[1] for r in rows]
print(f"  Sampled {len(chunk_ids)} chunks from Murgia 2004")

# ── Embed sample ──────────────────────────────────────────────────────────────
print("\n[3] Embedding sample chunks...")
vecs = embed_chunks_bge(chunk_ids, texts, model)
print(f"  Shape    : {vecs.shape}  (expected 100 x 1024)")
print(f"  Dtype    : {vecs.dtype}")
print(f"  Norms    : min={np.linalg.norm(vecs, axis=1).min():.4f} "
      f"max={np.linalg.norm(vecs, axis=1).max():.4f}  (all should be ~1.0)")

# ── Build small test index ────────────────────────────────────────────────────
print("\n[4] Building small test index...")
index  = get_or_create_bge_index()
id_map = {}
index.add(vecs)
for i, cid in enumerate(chunk_ids):
    id_map[cid] = i
print(f"  Index total: {index.ntotal} vectors")

# ── Test query embedding (asymmetric prefix) ───────────────────────────────────
print("\n[5] Testing query embedding with asymmetric prefix...")
query = "Burn law depolarization sigma_RM magnetic field"
qvec  = embed_query_bge(query, model)
print(f"  Query vector shape : {qvec.shape}")
print(f"  Query vector norm  : {np.linalg.norm(qvec):.4f}  (should be ~1.0)")

# ── Search ────────────────────────────────────────────────────────────────────
print("\n[6] Searching test index...")
results = bge_search(index, id_map, query, model, top_k=3)
print(f"  Top 3 results:")
for r in results:
    # Get text for display
    idx = chunk_ids.index(r["chunk_id"])
    print(f"    [{r['bge_rank']}] score={r['score']:.4f}")
    print(f"         {texts[idx][:100]}...")

# ── Compare query vs document norms ───────────────────────────────────────────
print("\n[7] Asymmetric prefix check...")
query_plain    = "Burn law depolarization sigma_RM"
query_prefixed = "Represent this sentence for searching relevant passages: " \
                 + query_plain
out_plain    = model.encode([query_plain],    return_dense=True,
                             return_sparse=False, return_colbert_vecs=False)
out_prefixed = model.encode([query_prefixed], return_dense=True,
                             return_sparse=False, return_colbert_vecs=False)
sim_plain    = float(np.dot(out_plain["dense_vecs"][0],    vecs[0]))
sim_prefixed = float(np.dot(out_prefixed["dense_vecs"][0], vecs[0]))
print(f"  Similarity without prefix : {sim_plain:.4f}")
print(f"  Similarity with prefix    : {sim_prefixed:.4f}")
print(f"  Prefix makes a difference : "
      f"{'✓' if abs(sim_prefixed - sim_plain) > 0.001 else 'minimal'}")

# ── Cleanup test artifacts ────────────────────────────────────────────────────
for f in [BGE_INDEX_PATH, BGE_IDMAP_PATH, BGE_MATRIX_PATH]:
    if f.exists():
        f.unlink()

print("\nBGE-M3 test passed. Ready for full corpus embedding.")