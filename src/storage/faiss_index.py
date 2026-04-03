# src/storage/faiss_index.py
"""
Step 7 — FAISS index initialisation and management.
Handles dense vector storage for SPECTER2 embeddings.
CPU FAISS — sufficient for 251 papers (~20k chunks).
GPU used upstream for embedding generation (Stage C).
"""

import faiss
import numpy as np
import json
from pathlib import Path

FAISS_DIR       = Path("data/faiss")
INDEX_PATH      = FAISS_DIR / "chunks.index"
ID_MAP_PATH     = FAISS_DIR / "chunks_id_map.json"
EMBEDDING_DIM   = 768   # SPECTER2 output dimension


def get_or_create_index() -> faiss.IndexFlatIP:
    """
    Load existing FAISS index from disk, or create a new one.
    Uses IndexFlatIP (inner product = cosine similarity on
    normalised vectors). Exact search — no approximation.
    Appropriate for < 100k vectors.
    """
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        print(f"Loaded existing FAISS index: {index.ntotal} vectors")
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        print(f"Created new FAISS index (dim={EMBEDDING_DIM})")

    return index


def load_id_map() -> dict:
    """
    Load chunk_id → faiss_row_index mapping from disk.
    Returns empty dict if not found.
    """
    if ID_MAP_PATH.exists():
        with open(ID_MAP_PATH, "r") as f:
            return json.load(f)
    return {}


def save_id_map(id_map: dict) -> None:
    """Persist chunk_id → faiss_row_index mapping to disk."""
    with open(ID_MAP_PATH, "w") as f:
        json.dump(id_map, f, indent=2)


def save_index(index: faiss.IndexFlatIP) -> None:
    """Write FAISS index to disk."""
    faiss.write_index(index, str(INDEX_PATH))


def add_embeddings(
    index:    faiss.IndexFlatIP,
    id_map:   dict,
    chunk_ids: list[str],
    vectors:   np.ndarray,
) -> tuple[faiss.IndexFlatIP, dict]:
    """
    Add a batch of embeddings to the FAISS index.
    Skips chunk_ids already in id_map (idempotent).
    Normalises vectors to unit length before adding
    (required for cosine similarity with IndexFlatIP).

    Returns updated index and id_map.
    """
    new_ids     = [cid for cid in chunk_ids if cid not in id_map]
    new_indices = [chunk_ids.index(cid) for cid in new_ids]

    if not new_ids:
        print(f"  All {len(chunk_ids)} chunks already indexed, skipping.")
        return index, id_map

    new_vectors = vectors[new_indices].astype(np.float32)

    # Normalise to unit length → cosine similarity via inner product
    norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)   # avoid div by zero
    new_vectors = new_vectors / norms

    start_row = index.ntotal
    index.add(new_vectors)

    # Update id_map
    for i, chunk_id in enumerate(new_ids):
        id_map[chunk_id] = start_row + i

    print(f"  Added {len(new_ids)} vectors. "
          f"Index total: {index.ntotal}")

    return index, id_map


def search(
    index:   faiss.IndexFlatIP,
    id_map:  dict,
    query_vector: np.ndarray,
    top_k:   int = 10,
) -> list[dict]:
    """
    Search FAISS index for nearest neighbours.
    Returns list of {chunk_id, score, faiss_row} dicts,
    sorted by descending similarity score.
    """
    # Normalise query
    query = query_vector.astype(np.float32).reshape(1, -1)
    norm  = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    scores, rows = index.search(query, top_k)

    # Reverse id_map: faiss_row → chunk_id
    row_to_chunk = {v: k for k, v in id_map.items()}

    results = []
    for score, row in zip(scores[0], rows[0]):
        if row == -1:   # FAISS returns -1 for empty slots
            continue
        chunk_id = row_to_chunk.get(int(row), f"unknown_row_{row}")
        results.append({
            "chunk_id": chunk_id,
            "score":    float(score),
            "faiss_row": int(row),
        })

    return results


def get_index_stats(
    index:  faiss.IndexFlatIP,
    id_map: dict,
) -> dict:
    """Return summary statistics for the current index."""
    return {
        "total_vectors": index.ntotal,
        "total_mapped":  len(id_map),
        "dimension":     index.d,
        "index_type":    type(index).__name__,
        "index_path":    str(INDEX_PATH),
        "id_map_path":   str(ID_MAP_PATH),
    }