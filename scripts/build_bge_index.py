# scripts/build_bge_index.py
"""
Build full BGE-M3 index over all 58,077 non-noise chunks.
Resumable — skips already-embedded chunks.
Estimated time: ~20-25 minutes on RTX 5070 Ti.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db      import get_connection
from src.query.bge_embedder import (
    get_bge_model, build_bge_index_from_db,
    get_or_create_bge_index, load_bge_id_map,
    BGE_INDEX_PATH, BGE_IDMAP_PATH, BGE_MATRIX_PATH,
)

print("Starting full BGE-M3 corpus embedding...")
conn  = get_connection()
model = get_bge_model()
build_bge_index_from_db(conn)
conn.close()

# Final verification
import faiss, json
index  = faiss.read_index(str(BGE_INDEX_PATH))
id_map = json.load(open(BGE_IDMAP_PATH))
print(f"\nFinal verification:")
print(f"  FAISS vectors : {index.ntotal}")
print(f"  ID map entries: {len(id_map)}")
print(f"  Match         : {'✓' if index.ntotal == len(id_map) else '✗'}")