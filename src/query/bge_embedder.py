# src/query/bge_embedder.py
"""
Step 10b — BGE-M3 chunk embeddings.
Second dense index alongside SPECTER2.
SPECTER2 = paper-level similarity (768-dim)
BGE-M3   = chunk-level evidence retrieval (1024-dim)

Asymmetric retrieval:
  Query side  : instruction prefix applied
  Document side: no prefix
"""

import json
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
# src/pipeline.py  — add after existing imports
import warnings
import logging
warnings.filterwarnings("ignore", message=".*position_ids.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"]                      = os.getenv("HF_TOKEN", "")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BGE_DIR        = Path("data/faiss")
BGE_INDEX_PATH = BGE_DIR / "bge_chunks.index"
BGE_IDMAP_PATH = BGE_DIR / "bge_chunks_id_map.json"
BGE_MATRIX_PATH = BGE_DIR / "bge_chunks_matrix.npy"

BGE_DIM        = 1024
BGE_BATCH_SIZE = 64
BGE_MODEL_ID   = "BAAI/bge-m3"

# Asymmetric query prefix — applied to queries only, not documents
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_bge_model = None


def get_bge_model() -> BGEM3FlagModel:
    """Load BGE-M3 onto GPU once, reuse across calls."""
    global _bge_model
    if _bge_model is None:
        print("Loading BGE-M3 on GPU...")
        _bge_model = BGEM3FlagModel(
            BGE_MODEL_ID,
            use_fp16=True,
            device="cuda",
        )
        print("BGE-M3 ready.")
    return _bge_model


def get_or_create_bge_index() -> faiss.IndexFlatIP:
    """Load existing BGE FAISS index or create new one."""
    BGE_DIR.mkdir(parents=True, exist_ok=True)
    if BGE_INDEX_PATH.exists():
        index = faiss.read_index(str(BGE_INDEX_PATH))
        print(f"Loaded existing BGE index: {index.ntotal} vectors")
    else:
        index = faiss.IndexFlatIP(BGE_DIM)
        print(f"Created new BGE index (dim={BGE_DIM})")
    return index


def load_bge_id_map() -> dict:
    """Load chunk_id → bge_faiss_row mapping."""
    if BGE_IDMAP_PATH.exists():
        with open(BGE_IDMAP_PATH) as f:
            return json.load(f)
    return {}


def save_bge_index(index: faiss.IndexFlatIP) -> None:
    faiss.write_index(index, str(BGE_INDEX_PATH))


def save_bge_id_map(id_map: dict) -> None:
    with open(BGE_IDMAP_PATH, "w") as f:
        json.dump(id_map, f)


def embed_chunks_bge(
    chunk_ids: list[str],
    texts:     list[str],
    model:     BGEM3FlagModel,
) -> np.ndarray:
    """
    Embed chunk texts with BGE-M3 (document side — no prefix).
    Returns normalised float32 array of shape (N, 1024).
    """
    all_vecs = []
    for i in range(0, len(texts), BGE_BATCH_SIZE):
        batch = texts[i:i + BGE_BATCH_SIZE]
        out   = model.encode(
            batch,
            batch_size=BGE_BATCH_SIZE,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = out["dense_vecs"].astype(np.float32)
        # Normalise to unit length for cosine via inner product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        all_vecs.append(vecs / norms)

    return np.vstack(all_vecs)


def embed_query_bge(
    query: str,
    model: BGEM3FlagModel,
) -> np.ndarray:
    """
    Embed a query with BGE-M3 (query side — with instruction prefix).
    Returns normalised float32 array of shape (1, 1024).
    """
    prefixed = QUERY_PREFIX + query
    out = model.encode(
        [prefixed],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vec = out["dense_vecs"].astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def build_bge_index_from_db(conn) -> None:
    """
    Pull all non-noise chunks from DuckDB, embed with BGE-M3,
    store in FAISS index + id_map + raw matrix.
    Skips chunks already in id_map (resumable).
    """
    model  = get_bge_model()
    index  = get_or_create_bge_index()
    id_map = load_bge_id_map()

    # Pull chunks not yet embedded
    rows = conn.execute("""
        SELECT chunk_id, text
        FROM chunks
        WHERE is_noise = FALSE
        ORDER BY node_id, chunk_index
    """).fetchall()

    # Filter already embedded
    to_embed = [(cid, txt) for cid, txt in rows if cid not in id_map]
    print(f"  Chunks to embed: {len(to_embed)} "
          f"({len(rows) - len(to_embed)} already done)")

    if not to_embed:
        print("  All chunks already embedded.")
        return

    chunk_ids = [r[0] for r in to_embed]
    texts     = [r[1] for r in to_embed]

    # Embed in batches with progress bar
    print(f"  Embedding {len(texts)} chunks with BGE-M3...")
    all_vecs = []
    batch_size = BGE_BATCH_SIZE

    for i in tqdm(range(0, len(texts), batch_size), desc="BGE-M3 embedding"):
        batch_ids  = chunk_ids[i:i + batch_size]
        batch_txts = texts[i:i + batch_size]
        vecs = embed_chunks_bge(batch_ids, batch_txts, model)
        all_vecs.append(vecs)

    all_vecs = np.vstack(all_vecs)

    # Add to FAISS
    start_row = index.ntotal
    index.add(all_vecs)

    # Update id_map
    for i, cid in enumerate(chunk_ids):
        id_map[cid] = start_row + i

    # Save everything
    save_bge_index(index)
    save_bge_id_map(id_map)

    # Save raw matrix (append if exists)
    if BGE_MATRIX_PATH.exists():
        existing = np.load(str(BGE_MATRIX_PATH))
        combined = np.vstack([existing, all_vecs])
        np.save(str(BGE_MATRIX_PATH), combined)
    else:
        np.save(str(BGE_MATRIX_PATH), all_vecs)

    print(f"  BGE index total: {index.ntotal} vectors")
    print(f"  Saved: {BGE_INDEX_PATH}")
    print(f"  Saved: {BGE_IDMAP_PATH}")
    print(f"  Saved: {BGE_MATRIX_PATH}")


def bge_search(
    index:    faiss.IndexFlatIP,
    id_map:   dict,
    query:    str,
    model:    BGEM3FlagModel,
    top_k:    int = 50,
    node_ids: list[str] | None = None,
) -> list[dict]:
    """
    Search BGE FAISS index for query.
    If node_ids provided, filters to chunks from those papers.
    Returns list of {chunk_id, node_id, score, bge_rank} dicts.
    """
    query_vec = embed_query_bge(query, model)
    scores, rows = index.search(query_vec, min(top_k * 3, index.ntotal))

    row_to_chunk = {v: k for k, v in id_map.items()}

    results = []
    for score, row in zip(scores[0], rows[0]):
        if row == -1:
            continue
        chunk_id = row_to_chunk.get(int(row))
        if not chunk_id:
            continue
        # Extract node_id from chunk_id
        node_id = chunk_id.rsplit("__c", 1)[0]
        if node_ids and node_id not in node_ids:
            continue
        results.append({
            "chunk_id":  chunk_id,
            "node_id":   node_id,
            "score":     float(score),
            "bge_rank":  0,
        })
        if len(results) >= top_k:
            break

    for i, r in enumerate(results):
        r["bge_rank"] = i + 1

    return results