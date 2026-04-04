# src/query/reranker.py
"""
Step 10e — BGE cross-encoder reranker.
Uses sentence_transformers CrossEncoder for stability
with transformers 5.x.

Model: BAAI/bge-reranker-v2-m3
"""

import os
import torch
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
_reranker = None


def get_reranker() -> CrossEncoder:
    """Load BGE reranker onto GPU once, reuse across calls."""
    global _reranker
    if _reranker is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BGE reranker on {device}...")
        _reranker = CrossEncoder(
            RERANKER_MODEL_ID,
            max_length=512,
            device=device,
        )
        print("Reranker ready.")
    return _reranker


def rerank_chunks(
    query:    str,
    chunks:   list[dict],
    reranker: CrossEncoder,
) -> list[dict]:
    """
    Rerank chunks using BGE cross-encoder.
    CrossEncoder sees query + chunk text together.
    Stores rerank_score and pre_rerank_score on each chunk.
    Returns chunks sorted by rerank_score descending.
    """
    if not chunks:
        return []

    pairs  = [[query, c["text"]] for c in chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)

    for chunk, score in zip(chunks, scores):
        chunk["pre_rerank_score"] = chunk.get("retrieval_score", 0.0)
        chunk["rerank_score"]     = float(score)

    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks


def apply_diversity(
    chunks:      list[dict],
    intent:      str,
    evidence_k:  int,
    min_papers:  int,
) -> list[dict]:
    """
    Apply diversity constraints after reranking.
    Enforces max chunks per paper and minimum distinct papers.
    """
    max_per_paper = {
        "fact":       3,
        "comparison": 2,
        "synthesis":  2,
        "discovery":  3,
    }.get(intent, 3)

    paper_counts: dict[str, int] = {}
    selected = []
    deferred = []

    for chunk in chunks:
        node_id = chunk["node_id"]
        count   = paper_counts.get(node_id, 0)
        if count < max_per_paper:
            selected.append(chunk)
            paper_counts[node_id] = count + 1
        else:
            deferred.append(chunk)

    distinct_papers = len(paper_counts)
    if distinct_papers < min_papers and deferred:
        for chunk in deferred:
            if len(selected) >= evidence_k:
                break
            node_id = chunk["node_id"]
            if node_id not in paper_counts:
                selected.append(chunk)
                paper_counts[node_id] = 1
                distinct_papers += 1
            if distinct_papers >= min_papers:
                break

    return selected[:evidence_k]


def rerank(
    query:      str,
    chunks:     list[dict],
    intent:     str,
    evidence_k: int,
    min_papers: int,
    reranker:   CrossEncoder,
) -> list[dict]:
    """
    Full reranking pipeline:
    1. Cross-encoder rescoring
    2. Diversity control
    3. Trim to evidence_k
    """
    if not chunks:
        return []

    print(f"  Reranking {len(chunks)} chunks...")
    reranked = rerank_chunks(query, chunks, reranker)
    diverse  = apply_diversity(reranked, intent, evidence_k, min_papers)

    print(f"  After rerank+diversity: {len(diverse)} chunks "
          f"from {len({c['node_id'] for c in diverse})} papers")

    return diverse