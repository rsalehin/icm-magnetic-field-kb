# src/pipeline.py
"""
Step 9 — Full ingestion pipeline.
Runs Stages A, B, C, D on a single paper end to end.
Writes to all three backends: DuckDB, FAISS, NetworkX.

Stage A — PDF extraction      (pdf_extractor.py)
Stage B — ADS enrichment      (ads_enricher.py)
Stage C — SPECTER2 embedding  (this file)
Stage D — Storage write       (db.py, faiss_index.py, graph.py)
"""

import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer

from src.extraction.pdf_extractor import extract_paper
from src.enrichment.ads_enricher  import enrich_paper
from src.storage.db               import (
    get_connection, init_schema, insert_paper, DB_PATH
)
from src.storage.faiss_index      import (
    get_or_create_index, load_id_map, save_id_map,
    save_index, add_embeddings,
)
from src.storage.graph            import (
    get_or_create_graph, save_graph, add_paper_node,
)

# ── Embedding model (loaded once per pipeline run) ────────────────────────────

_model = None

def get_model() -> SentenceTransformer:
    """Load SPECTER2 onto GPU once, reuse across papers."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SPECTER2 on {device}...")
        _model = SentenceTransformer("allenai/specter2_base", device=device)
        print("Model ready.")
    return _model


# ── Stage C — Embedding ───────────────────────────────────────────────────────

def embed_chunks(paper: dict, model: SentenceTransformer) -> dict:
    """
    Stage C — Generate SPECTER2 embeddings for all non-noise chunks.
    Attaches embedding vectors directly to each chunk dict.
    Skips noise chunks (token_count < 30).
    """
    chunks     = paper["layer_2_chunks"]["chunks"]
    to_embed   = [c for c in chunks if c["token_count"] >= 30]
    texts      = [c["text"] for c in to_embed]

    if not texts:
        print("  No chunks to embed.")
        paper["meta"]["stage_c"] = "done"
        return paper

    print(f"  Embedding {len(texts)} chunks "
          f"({len(chunks) - len(texts)} noise skipped)...")

    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # Attach vectors back to chunks
    vec_iter = iter(vectors)
    for chunk in chunks:
        if chunk["token_count"] >= 30:
            chunk["embedding"]["vector"] = next(vec_iter).tolist()
        else:
            chunk["embedding"]["vector"] = None

    paper["meta"]["stage_c"]        = "done"
    paper["meta"]["embedding_model"] = "allenai/specter2_base"
    return paper


# ── Stage D — Storage write ───────────────────────────────────────────────────

def write_to_storage(
    paper:  dict,
    conn,
    index,
    id_map: dict,
    G,
) -> tuple[dict, dict]:
    """
    Stage D — Write paper to all three backends.
    Returns updated (id_map, G).
    """
    chunks = paper["layer_2_chunks"]["chunks"]

    # ── DuckDB ───────────────────────────────────────────────────────────────
    insert_paper(conn, paper)

    # ── FAISS ────────────────────────────────────────────────────────────────
    embeddable = [
        c for c in chunks
        if c["embedding"]["vector"] is not None
    ]
    if embeddable:
        chunk_ids = [c["chunk_id"]          for c in embeddable]
        vectors   = np.array(
            [c["embedding"]["vector"] for c in embeddable],
            dtype=np.float32
        )
        index, id_map = add_embeddings(index, id_map, chunk_ids, vectors)

        # Write FAISS row indices back to DuckDB
        for chunk in embeddable:
            cid     = chunk["chunk_id"]
            row_idx = id_map.get(cid)
            if row_idx is not None:
                conn.execute(
                    "UPDATE chunks SET faiss_index_id = ? "
                    "WHERE chunk_id = ?",
                    [row_idx, cid]
                )
        conn.commit()

    # ── NetworkX ──────────────────────────────────────────────────────────────
    G = add_paper_node(G, paper)

    paper["meta"]["stage_d"]       = "done"
    paper["meta"]["ingestion_date"] = datetime.now().isoformat()

    # Update stage_d in DuckDB now that all writes are confirmed
    conn.execute(
        "UPDATE papers SET stage_d = 'done' WHERE node_id = ?",
        [paper["node_id"]]
    )
    conn.commit()

    return paper, id_map, G


# ── Main pipeline function ────────────────────────────────────────────────────

def ingest_paper(
    pdf_path: Path,
    conn,
    index,
    id_map:  dict,
    G,
    model:   SentenceTransformer,
    skip_ads: bool = False,
) -> tuple[dict, dict]:
    """
    Run full pipeline on a single PDF.
    Returns (paper_dict, updated_id_map, updated_G).
    Skips if paper already in DuckDB.
    """
    pdf_path = Path(pdf_path)
    t0       = time.time()

    print(f"\n{'─'*55}")
    print(f"Ingesting: {pdf_path.name}")

    # ── Stage A ───────────────────────────────────────────────────────────────
    paper = extract_paper(pdf_path)
    node_id = paper["node_id"]

    # Check if already ingested
    existing = conn.execute(
        "SELECT stage_d FROM papers WHERE node_id = ?",
        [node_id]
    ).fetchone()
    if existing and existing[0] == "done":
        print(f"  Already fully ingested, skipping.")
        return paper, id_map, G

    print(f"  Stage A: {paper['layer_2_chunks']['total_chunks']} chunks")

    # ── Stage B ───────────────────────────────────────────────────────────────
    if not skip_ads:
        paper = enrich_paper(paper)
    else:
        paper["meta"]["stage_b"] = "skipped"

    # ── Stage C ───────────────────────────────────────────────────────────────
    paper = embed_chunks(paper, model)

    # ── Stage D ───────────────────────────────────────────────────────────────
    paper, id_map, G = write_to_storage(paper, conn, index, id_map, G)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  "
          f"[A:done B:{paper['meta']['stage_b']} "
          f"C:{paper['meta']['stage_c']} "
          f"D:{paper['meta']['stage_d']}]")

    return paper, id_map, G