# src/query/retriever.py
"""
Step 10d — Hybrid retriever.
Combines SPECTER2 paper-level FAISS + DuckDB hard filters +
BM25 lexical + BGE-M3 dense chunk retrieval.
Fuses rankings with Reciprocal Rank Fusion (RRF).
Applies section priors and metadata boosts.
Graph expansion disabled until Step 11 populates edges.
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

from src.query.planner      import PlannerOutput
from src.query.bm25_index   import bm25_search, load_bm25_index, BM25Okapi
from src.query.bge_embedder import bge_search, embed_query_bge


# ── RRF constants ─────────────────────────────────────────────────────────────
RRF_K   = 60     # standard RRF constant
ALPHA   = 0.15   # section prior weight
BETA    = 0.10   # metadata boost weight


# ── Section prior lookup ──────────────────────────────────────────────────────
SECTION_PRIORS = {
    "fact": {
        "abstract":     0.8,
        "results":      1.0,
        "conclusion":   0.7,
        "conclusions":  0.7,
        "summary":      0.7,
        "methods":      0.6,
        "introduction": 0.3,
        "references":   0.0,
        "preamble":     0.1,
    },
    "comparison": {
        "abstract":     0.7,
        "results":      0.9,
        "discussion":   0.9,
        "conclusion":   0.8,
        "conclusions":  0.8,
        "methods":      0.7,
        "introduction": 0.4,
        "references":   0.0,
        "preamble":     0.1,
    },
    "synthesis": {
        "abstract":     0.8,
        "results":      0.8,
        "discussion":   0.9,
        "conclusion":   0.9,
        "conclusions":  0.9,
        "introduction": 0.5,
        "methods":      0.6,
        "references":   0.0,
        "preamble":     0.1,
    },
    "discovery": {
        "abstract":     0.9,
        "introduction": 0.8,
        "results":      0.6,
        "methods":      0.5,
        "discussion":   0.5,
        "conclusion":   0.5,
        "references":   0.0,
        "preamble":     0.1,
    },
}


def get_section_prior(section_title: str, intent: str) -> float:
    """Look up section prior score for given intent."""
    priors = SECTION_PRIORS.get(intent, {})
    sec    = (section_title or "").lower().strip()
    # Exact match first
    if sec in priors:
        return priors[sec]
    # Partial match
    for key, val in priors.items():
        if key in sec:
            return val
    return 0.4  # default for unknown sections


# ── Stage 1 — Paper candidate retrieval ──────────────────────────────────────

def retrieve_paper_candidates(
    query:          str,
    plan:           PlannerOutput,
    conn,
    specter_index,
    specter_id_map: dict,
    specter_model:  SentenceTransformer,
    G               = None,
) -> list[str]:
    """
    Stage 1: Get candidate paper node_ids via SPECTER2 + DuckDB filters
    + optional graph expansion.
    """
    paper_k = plan["budgets"]["paper_k"]
    ef      = plan["explicit_filters"]

    # ── A. SPECTER2 dense retrieval ───────────────────────────────────────────
    query_vec = specter_model.encode(
        [query], convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    n_retrieve   = min(paper_k * 20, specter_index.ntotal)
    scores, rows = specter_index.search(query_vec, n_retrieve)

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

    # ── B. DuckDB explicit hard filters ───────────────────────────────────────
    if not candidate_ids:
        print("  No FAISS candidates found.")
        return []

    placeholders = ",".join(["?" for _ in candidate_ids])
    conditions   = [f"node_id IN ({placeholders})"]
    params       = list(candidate_ids)

    if ef.get("year_min"):
        conditions.append("year >= ?")
        params.append(ef["year_min"])
    if ef.get("year_max"):
        conditions.append("year <= ?")
        params.append(ef["year_max"])
    if ef.get("journal"):
        conditions.append("journal ILIKE ?")
        params.append(f"%{ef['journal']}%")

    where     = " AND ".join(conditions)
    query_sql = f"""
        SELECT node_id FROM papers
        WHERE {where}
        ORDER BY citation_count DESC NULLS LAST
        LIMIT {paper_k}
    """

    rows_db      = conn.execute(query_sql, params).fetchall()
    filtered_ids = [r[0] for r in rows_db]

    # ── C. Graph expansion (1-hop typed) ─────────────────────────────────────
    if G is not None and G.number_of_edges() > 0 and filtered_ids:
        expanded   = set(filtered_ids)
        intent     = plan["intent"]
        all_corpus = {
            r[0] for r in conn.execute("SELECT node_id FROM papers").fetchall()
        }

        for node_id in list(filtered_ids):
            if node_id not in G:
                continue

            # Citation neighbours — strong edges only
            cites_added = 0
            for neighbour in G.successors(node_id):
                if cites_added >= 15:
                    break
                if neighbour not in all_corpus:
                    continue
                role = G.edges[node_id, neighbour].get("citation_role", "incidental")
                if role in ("foundational", "methodological", "data_source"):
                    expanded.add(neighbour)
                    cites_added += 1

            # Co-citation — synthesis/discovery only
            if intent in ("synthesis", "discovery"):
                cocite_added = 0
                for neighbour in G.successors(node_id):
                    if cocite_added >= 10:
                        break
                    if neighbour not in all_corpus:
                        continue
                    if G.edges[node_id, neighbour].get("citation_role") == "co_cited":
                        expanded.add(neighbour)
                        cocite_added += 1

        if len(expanded) > len(filtered_ids):
            print(f"  Graph expansion: {len(filtered_ids)} → {len(expanded)} candidates")
        filtered_ids = list(expanded)[:paper_k]

    print(f"  Paper candidates: {len(filtered_ids)} "
          f"(SPECTER2={len(candidate_ids)} → after filters={len(filtered_ids)})")

    return filtered_ids

# ── Stage 2 — Chunk retrieval + RRF fusion ────────────────────────────────────

def retrieve_chunks(
    query:      str,
    plan:       PlannerOutput,
    node_ids:   list[str],
    bm25_index: BM25Okapi,
    bm25_records: list[dict],
    bge_index,
    bge_id_map: dict,
    bge_model:  BGEM3FlagModel,
) -> list[dict]:
    """
    Stage 2: Retrieve chunks from paper pool using BM25 + BGE-M3.
    Fuse with RRF. Apply section priors and metadata boosts.

    Returns list of chunk dicts with final retrieval_score.
    """
    lex_k   = plan["budgets"]["chunk_lex_k"]
    dense_k = plan["budgets"]["chunk_dense_k"]
    intent  = plan["intent"]

    # ── BM25 lexical retrieval ────────────────────────────────────────────────
    bm25_results = bm25_search(
        bm25_index, bm25_records, query,
        top_k=lex_k, node_ids=node_ids
    )
    print(f"  BM25 hits: {len(bm25_results)}")

    # ── BGE-M3 dense retrieval ────────────────────────────────────────────────
    bge_results = bge_search(
        bge_index, bge_id_map, query,
        bge_model, top_k=dense_k, node_ids=node_ids
    )
    print(f"  BGE-M3 hits: {len(bge_results)}")

    # ── RRF fusion ────────────────────────────────────────────────────────────
    # Build rank dicts: chunk_id → rank
    bm25_ranks = {r["chunk_id"]: r["bm25_rank"] for r in bm25_results}
    bge_ranks  = {r["chunk_id"]: r["bge_rank"]  for r in bge_results}

    # Union of all chunk_ids
    all_chunk_ids = set(bm25_ranks) | set(bge_ranks)

    # Build lookup for chunk metadata
    chunk_meta = {}
    for r in bm25_results:
        chunk_meta[r["chunk_id"]] = r
    for r in bge_results:
        if r["chunk_id"] not in chunk_meta:
            chunk_meta[r["chunk_id"]] = r

    # Compute RRF score for each chunk
    fused = []
    for chunk_id in all_chunk_ids:
        rrf_score = 0.0
        if chunk_id in bm25_ranks:
            rrf_score += 1.0 / (RRF_K + bm25_ranks[chunk_id])
        if chunk_id in bge_ranks:
            rrf_score += 1.0 / (RRF_K + bge_ranks[chunk_id])

        meta    = chunk_meta.get(chunk_id, {})
        node_id = chunk_id.rsplit("__c", 1)[0]
        section = meta.get("section_title", "")

        # Section prior
        section_prior = get_section_prior(section, intent)

        # Metadata boost
        meta_boost = 0.0
        if meta.get("has_numerical_result"):
            meta_boost += 0.1
        if meta.get("has_equation") and intent in ("fact", "comparison"):
            meta_boost += 0.05

        # Final score
        final_score = rrf_score + ALPHA * section_prior + BETA * meta_boost

        fused.append({
            "chunk_id":        chunk_id,
            "node_id":         node_id,
            "section_title":   section,
            "text":            meta.get("text", meta.get("abstract", "")),
            "retrieval_score": final_score,
            "rrf_score":       rrf_score,
            "section_prior":   section_prior,
            "meta_boost":      meta_boost,
            "bm25_rank":       bm25_ranks.get(chunk_id),
            "bge_rank":        bge_ranks.get(chunk_id),
            "is_abstract":     meta.get("is_abstract", False),
        })

    # Sort by final score
    fused.sort(key=lambda x: x["retrieval_score"], reverse=True)

    print(f"  Fused chunks: {len(fused)} "
          f"(BM25={len(bm25_ranks)} BGE={len(bge_ranks)} "
          f"union={len(all_chunk_ids)})")

    return fused


# ── Main retriever function ───────────────────────────────────────────────────

def retrieve(
    plan:           PlannerOutput,
    conn,
    specter_index,
    specter_id_map: dict,
    specter_model:  SentenceTransformer,
    bm25_index:     BM25Okapi,
    bm25_records:   list[dict],
    bge_index,
    bge_id_map:     dict,
    bge_model:      BGEM3FlagModel,
    G             = None,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline.
    Returns top rerank_k chunks sorted by retrieval_score.
    """
    query    = plan["raw_query"]
    rerank_k = plan["budgets"]["rerank_k"]

    print(f"\nRetriever: intent={plan['intent']} "
          f"paper_k={plan['budgets']['paper_k']}")

    # Stage 1 — paper candidates
    node_ids = retrieve_paper_candidates(
        query, plan, conn,
        specter_index, specter_id_map, specter_model,
        G=G,
    )

    if not node_ids:
        print("  No paper candidates found.")
        return []

    # Stage 2 — chunk retrieval + fusion
    chunks = retrieve_chunks(
        query, plan, node_ids,
        bm25_index, bm25_records,
        bge_index, bge_id_map, bge_model,
    )

    # Return top rerank_k
    return chunks[:rerank_k]