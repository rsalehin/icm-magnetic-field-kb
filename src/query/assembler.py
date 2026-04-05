# src/query/assembler.py
"""
Step 10f — Evidence assembler.
Takes reranked chunks, produces a structured evidence pack
ready for the generator.

Steps:
1. Deduplicate near-identical chunks (cosine similarity)
2. Group by paper + section
3. Detect conflicts via rule-based lexical cues
4. Compute abstention signal (numeric rules)
5. Return typed evidence pack
"""

import re
import numpy as np
from pathlib import Path


# ── Contradiction cue patterns ────────────────────────────────────────────────

POSITIVE_CUES = [
    r"\bconfirm(s|ed|ing)?\b",
    r"\bconsistent with\b",
    r"\bin agreement\b",
    r"\bsupport(s|ed|ing)?\b",
    r"\bvalidat(e|es|ed|ing)\b",
]

NEGATIVE_CUES = [
    r"\binconsistent with\b",
    r"\bcontradicts?\b",
    r"\bno evidence\b",
    r"\bfails? to detect\b",
    r"\bnot significant\b",
    r"\bin contrast\b",
    r"\bhowever\b",
    r"\bdisputes?\b",
    r"\bsuggests? otherwise\b",
    r"\bopposite\b",
    r"\bunlike\b",
]

UNCERTAINTY_CUES = [
    r"\btentative\b",
    r"\bunclear\b",
    r"\bdebated?\b",
    r"\bcontroversial\b",
    r"\buncertain\b",
    r"\bremains? unclear\b",
    r"\bpoorly constrained\b",
]


def detect_cues(text: str) -> dict:
    """Count positive, negative, uncertainty cues in text."""
    t = text.lower()
    return {
        "positive":    sum(1 for p in POSITIVE_CUES    if re.search(p, t)),
        "negative":    sum(1 for p in NEGATIVE_CUES    if re.search(p, t)),
        "uncertainty": sum(1 for p in UNCERTAINTY_CUES if re.search(p, t)),
    }


# ── Cosine similarity for deduplication ──────────────────────────────────────

def cosine_sim(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    n1 = np.linalg.norm(a)
    n2 = np.linalg.norm(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(a, b) / (n1 * n2))


def deduplicate_chunks(
    chunks:    list[dict],
    bge_id_map: dict,
    bge_matrix: np.ndarray | None,
    threshold: float = 0.92,
) -> list[dict]:
    """
    Remove near-duplicate chunks using cosine similarity.
    If BGE matrix available, uses stored embeddings.
    Falls back to text overlap if not available.
    Keeps chunk with higher rerank_score when duplicates found.
    """
    if len(chunks) <= 1:
        return chunks

    kept    = []
    dropped = set()

    for i, chunk_i in enumerate(chunks):
        if i in dropped:
            continue
        for j, chunk_j in enumerate(chunks[i+1:], start=i+1):
            if j in dropped:
                continue

            # Try embedding-based similarity first
            sim = 0.0
            cid_i = chunk_i["chunk_id"]
            cid_j = chunk_j["chunk_id"]

            if bge_matrix is not None and cid_i in bge_id_map \
                    and cid_j in bge_id_map:
                row_i = bge_id_map[cid_i]
                row_j = bge_id_map[cid_j]
                v1    = bge_matrix[row_i]
                v2    = bge_matrix[row_j]
                sim   = float(np.dot(v1, v2))  # already normalised
            else:
                # Fallback: word overlap ratio
                words_i = set(chunk_i["text"].lower().split())
                words_j = set(chunk_j["text"].lower().split())
                if words_i and words_j:
                    sim = len(words_i & words_j) / len(words_i | words_j)

            if sim >= threshold:
                # Keep higher rerank score
                if chunk_j.get("rerank_score", 0) > \
                        chunk_i.get("rerank_score", 0):
                    dropped.add(i)
                else:
                    dropped.add(j)

        if i not in dropped:
            kept.append(chunk_i)

    return kept


# ── Conflict detection ────────────────────────────────────────────────────────

def detect_conflicts(chunks: list[dict]) -> dict:
    """
    Rule-based conflict detection across chunks.
    Returns conflict summary dict.
    """
    all_cues = [detect_cues(c["text"]) for c in chunks]

    total_negative    = sum(c["negative"]    for c in all_cues)
    total_uncertainty = sum(c["uncertainty"] for c in all_cues)
    total_positive    = sum(c["positive"]    for c in all_cues)

    # Conflict: negative cues from multiple papers
    papers_with_negative = set()
    for chunk, cues in zip(chunks, all_cues):
        if cues["negative"] > 0:
            papers_with_negative.add(chunk["node_id"])

    has_disagreement = len(papers_with_negative) >= 2
    has_uncertainty  = total_uncertainty >= 2

    # Collect conflict evidence
    conflict_notes = []
    if has_disagreement:
        conflict_notes.append(
            f"Negative cues found in {len(papers_with_negative)} papers"
        )
    if has_uncertainty:
        conflict_notes.append(
            f"Uncertainty cues found ({total_uncertainty} instances)"
        )

    return {
        "has_disagreement":   has_disagreement,
        "has_uncertainty":    has_uncertainty,
        "papers_with_negative": list(papers_with_negative),
        "total_negative_cues": total_negative,
        "total_uncertainty_cues": total_uncertainty,
        "total_positive_cues":  total_positive,
        "conflict_notes":     conflict_notes,
    }
    

def expand_with_neighbours(
    evidence: list[dict],
    conn,
    n_neighbours: int = 1,
) -> list[dict]:
    """
    For each evidence chunk, fetch n_neighbours before and after
    from the same paper. Adds context without changing ranking.
    """
    expanded = list(evidence)
    existing_chunk_ids = {e["chunk_id"] for e in evidence}

    for e in evidence:
        chunk_id = e.get("chunk_id", "")
        if not chunk_id:
            continue

        # Parse chunk index from chunk_id: arxiv:xxx__c0042 → 42
        try:
            chunk_idx = int(chunk_id.rsplit("__c", 1)[1])
        except (IndexError, ValueError):
            continue

        node_id = e.get("paper_id", "")

        # Fetch neighbours
        rows = conn.execute("""
            SELECT chunk_id, chunk_index, section_title, text
            FROM chunks
            WHERE node_id = ?
              AND chunk_index BETWEEN ? AND ?
              AND is_noise = FALSE
            ORDER BY chunk_index
        """, [
            node_id,
            chunk_idx - n_neighbours,
            chunk_idx + n_neighbours,
        ]).fetchall()

        for row in rows:
            cid = row[0]
            if cid in existing_chunk_ids or cid == chunk_id:
                continue
            existing_chunk_ids.add(cid)
            expanded.append({
                **e,  # inherit paper metadata
                "chunk_id":     cid,
                "section":      row[2],
                "text":         row[3],
                "rerank_score": e["rerank_score"] * 0.8,  # slightly lower score
                "is_neighbour": True,
            })

    return expanded


def compute_abstention(
    chunks:    list[dict],
    conflicts: dict,
    min_papers: int,
    threshold:  float = 0.30,
) -> tuple[bool, str | None]:
    """
    Compute abstention signal based on numeric rules.
    Returns (should_abstain, reason).
    """
    if not chunks:
        return True, "No evidence chunks retrieved"

    distinct_papers = len({c["node_id"] for c in chunks})
    max_score       = max((c.get("rerank_score", 0) for c in chunks),
                          default=0)

    if len(chunks) < 3:
        return True, f"Insufficient evidence: only {len(chunks)} chunks"

    if distinct_papers < min_papers:
        return True, (f"Insufficient diversity: {distinct_papers} papers "
                      f"(minimum {min_papers})")

    if max_score < threshold:
        return True, (f"Low confidence: max rerank score {max_score:.3f} "
                      f"below threshold {threshold}")

    if conflicts["has_disagreement"]:
        return False, "disagreement_present"

    return False, None


# ── Evidence grouping ─────────────────────────────────────────────────────────

def group_by_paper(chunks: list[dict], conn) -> list[dict]:
    """
    Group chunks by paper, attach paper metadata.
    Returns list of paper groups with their chunks.
    """
    # Get paper metadata for all node_ids
    # Normalise abstract chunk node_ids
    node_ids = list({c["node_id"].replace("__abstract", "") for c in chunks})
    if not node_ids:
        return []

    placeholders = ",".join(["?" for _ in node_ids])
    rows = conn.execute(f"""
        SELECT node_id, title, year, journal, bibcode
        FROM papers
        WHERE node_id IN ({placeholders})
    """, node_ids).fetchall()

    paper_meta = {
        r[0]: {"title": r[1], "year": r[2],
               "journal": r[3], "bibcode": r[4]}
        for r in rows
    }

    # Group chunks by paper
    groups: dict[str, dict] = {}
    for chunk in chunks:
        nid = chunk["node_id"].replace("__abstract", "")
        if nid not in groups:
            meta = paper_meta.get(nid, {})
            groups[nid] = {
                "node_id": nid,
                "title":   meta.get("title", "Unknown"),
                "year":    meta.get("year"),
                "journal": meta.get("journal", ""),
                "bibcode": meta.get("bibcode", ""),
                "chunks":  [],
            }
        groups[nid]["chunks"].append(chunk)

    # Sort chunks within each paper by rerank_score
    for g in groups.values():
        g["chunks"].sort(key=lambda x: x.get("rerank_score", 0),
                         reverse=True)

    # Sort paper groups by best chunk rerank_score
    sorted_groups = sorted(
        groups.values(),
        key=lambda g: g["chunks"][0].get("rerank_score", 0),
        reverse=True
    )

    return sorted_groups


# ── Main assembler ────────────────────────────────────────────────────────────

def assemble_evidence(
    query:      str,
    plan:       dict,
    chunks:     list[dict],
    conn,
    bge_id_map: dict,
    bge_matrix: np.ndarray | None = None,
) -> dict:
    """
    Full evidence assembly pipeline.
    Returns typed evidence pack ready for generator.
    """
    intent     = plan["intent"]
    min_papers = plan["budgets"]["min_papers"]

    # Step 1 — deduplicate
    deduped = deduplicate_chunks(chunks, bge_id_map, bge_matrix)
    n_deduped = len(chunks) - len(deduped)
    if n_deduped > 0:
        print(f"  Deduplicated: removed {n_deduped} near-duplicate chunks")

    # Step 2 — conflict detection
    conflicts = detect_conflicts(deduped)

    # Step 3 — abstention signal
    should_abstain, abstain_reason = compute_abstention(
        deduped, conflicts, min_papers
    )

    # Step 4 — group by paper
    groups = group_by_paper(deduped, conn)

    # Step 5 — build flat evidence list for generator
    evidence_list = []
    for group in groups:
        for chunk in group["chunks"]:
            evidence_list.append({
                "chunk_id":     chunk["chunk_id"],
                "paper_id":     group["node_id"],
                "bibcode":      group["bibcode"],
                "title":        group["title"],
                "year":         group["year"],
                "journal":      group["journal"],
                "section":      chunk.get("section_title", ""),
                "text":         chunk["text"],
                "rerank_score": chunk.get("rerank_score", 0.0),
                "retrieval_score": chunk.get("retrieval_score", 0.0),
            })

    # ── ADD HERE ──────────────────────────────────────────────────────────────
    # Step 5b — expand with neighbours for deep queries
    if intent in ("comparison", "synthesis"):
        evidence_list = expand_with_neighbours(evidence_list, conn, n_neighbours=1)
        evidence_list.sort(key=lambda x: x["rerank_score"], reverse=True)
    

    # Support stats
    distinct_papers = len({e["paper_id"] for e in evidence_list})
    max_score       = max((e["rerank_score"] for e in evidence_list),
                          default=0.0)

    support_stats = {
        "distinct_chunks":        len(deduped),
        "distinct_papers":        distinct_papers,
        "max_rerank_score":       round(max_score, 4),
        "chunks_deduplicated":    n_deduped,
        "has_disagreement":       conflicts["has_disagreement"],
        "has_uncertainty":        conflicts["has_uncertainty"],
        "total_negative_cues":    conflicts["total_negative_cues"],
        "total_uncertainty_cues": conflicts["total_uncertainty_cues"],
        "single_source":          distinct_papers == 1,
        "conflict_notes":         conflicts["conflict_notes"],
    }

    return {
        "query":         query,
        "intent":        intent,
        "evidence":      evidence_list,
        "support_stats": support_stats,
        "abstain":       should_abstain,
        "abstain_reason": abstain_reason
                          if should_abstain else
                          (abstain_reason if abstain_reason else None),
        "paper_groups":  groups,
    }