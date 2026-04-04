# scripts/generate_eval_set.py
"""
Automated evaluation set generation.
Samples chunks from DuckDB stratified by era, section, content type.
Uses Qwen3 locally to generate Q&A pairs from chunk text.
Outputs eval_set.json with verified ground truth.
"""

import sys
import json
import time
import re
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
from src.storage.db import get_connection

EVAL_SET_PATH  = Path("data/eval_set.json")
OUTPUT_PATH    = Path("data/eval_set_generated.json")
QWEN_MODEL     = "qwen3:14b"
RANDOM_SEED    = 42
random.seed(RANDOM_SEED)

# ── Sampling strategy ─────────────────────────────────────────────────────────

ERA_BINS = [
    ("1995-2004", 1995, 2004, 5),
    ("2005-2009", 2005, 2009, 5),
    ("2010-2014", 2010, 2014, 5),
    ("2015-2019", 2015, 2019, 5),
    ("2020-2026", 2020, 2026, 5),
]

SECTION_PREFS = ["results", "methods", "abstract", "conclusion", "discussion"]


# ── Chunk sampling ────────────────────────────────────────────────────────────

def sample_chunks(conn, n: int, year_min: int, year_max: int) -> list[dict]:
    """
    Sample n diverse chunks from papers in year range.
    Prefers chunks with numerical results or equations.
    """
    rows = conn.execute("""
        SELECT
            c.chunk_id,
            c.node_id,
            c.section_title,
            c.text,
            c.token_count,
            c.has_numerical_result,
            c.has_equation,
            p.year,
            p.title,
            p.arxiv_id
        FROM chunks c
        JOIN papers p ON c.node_id = p.node_id
        WHERE c.is_noise = FALSE
          AND c.token_count BETWEEN 60 AND 300
          AND p.year BETWEEN ? AND ?
          AND (c.has_numerical_result = TRUE OR c.has_equation = TRUE)
        ORDER BY RANDOM()
        LIMIT ?
    """, [year_min, year_max, n * 5]).fetchall()

    cols = ["chunk_id", "node_id", "section_title", "text",
            "token_count", "has_numerical_result", "has_equation",
            "year", "title", "arxiv_id"]

    chunks = [dict(zip(cols, r)) for r in rows]

    # Prefer diverse papers — max 2 chunks per paper
    seen_papers: dict[str, int] = {}
    selected = []
    for chunk in chunks:
        nid   = chunk["node_id"]
        count = seen_papers.get(nid, 0)
        if count < 2:
            selected.append(chunk)
            seen_papers[nid] = count + 1
        if len(selected) >= n:
            break

    return selected[:n]


# ── Qwen3 Q&A generation ──────────────────────────────────────────────────────

GENERATION_PROMPT = """You are evaluating a scientific retrieval system for astrophysics papers about intracluster magnetic fields.

Given the following chunk of text from a paper, generate ONE precise factual question whose answer is EXPLICITLY stated in the text. The question must be answerable from this text alone without any external knowledge.

Rules:
- The question must be specific and factual (not vague like "What did the authors find?")
- The answer must be a specific value, method, result, or definition present in the text
- Do not ask about figure numbers, table numbers, or references to other papers
- The question should be useful for a researcher studying intracluster magnetic fields

Paper: {title} ({year})
Section: {section}
Text:
{text}

Respond with ONLY a JSON object, no markdown, no explanation:
{{"question": "...", "answer": "...", "key_terms": ["term1", "term2", "term3"]}}"""

ABSTENTION_PROMPT = """You are designing evaluation questions for a scientific retrieval system about intracluster magnetic fields in galaxy clusters.

The system's corpus covers: Faraday rotation, RM synthesis, magnetic field models (GRF, BxC, MHD), depolarization, radio halos, galaxy cluster observations.

Generate ONE question that SOUNDS like it could be in this corpus but is actually NOT answerable from this literature. The question should be plausibly related to radio astronomy or galaxy clusters but about a topic genuinely absent from ICM magnetic field literature.

Examples of good abstention questions:
- Questions about specific clusters not typically studied with RM
- Questions about unrelated physics topics
- Questions requiring engineering/instrumental knowledge not in papers

Respond with ONLY a JSON object:
{{"question": "...", "reason_absent": "..."}}"""


def generate_qa_pair(chunk: dict) -> dict | None:
    """
    Use Qwen3 to generate a Q&A pair from a chunk.
    Returns None if generation fails or quality is poor.
    """
    prompt = GENERATION_PROMPT.format(
        title   = chunk["title"][:80],
        year    = chunk["year"],
        section = chunk["section_title"],
        text    = chunk["text"][:800],
    )

    try:
        response = ollama.chat(
            model    = QWEN_MODEL,
            messages = [{"role": "user", "content": prompt}],
            think    = False,
            options  = {"temperature": 0.3, "num_predict": 500},
        )
        raw = response["message"]["content"].strip()
        #print(f"      raw: {raw[:100]}")

        # Strip any markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()

        # Parse JSON
        data = json.loads(raw)

        # Quality checks
        question = data.get("question", "").strip()
        answer   = data.get("answer", "").strip()
        terms    = data.get("key_terms", [])

        if not question or not answer:
            return None
        if len(question) < 15 or len(answer) < 3:
            return None
        if "figure" in question.lower() or "table" in question.lower():
            return None
        if "?" not in question:
            return None

        # Verify answer appears in chunk text (loose check)
        answer_words = set(answer.lower().split())
        chunk_words  = set(chunk["text"].lower().split())
        overlap      = len(answer_words & chunk_words) / max(len(answer_words), 1)
        if overlap < 0.3:
            return None

        return {
            "question":   question,
            "answer":     answer,
            "key_terms":  terms[:5],
            "source_chunk_id": chunk["chunk_id"],
        }

    except (json.JSONDecodeError, KeyError, Exception):
        return None


def generate_abstention_question() -> dict | None:
    """Generate a plausible-but-absent question using Qwen3."""
    try:
        response = ollama.chat(
            model    = QWEN_MODEL,
            messages = [{"role": "user", "content": ABSTENTION_PROMPT}],
            think    = False,
            options  = {"temperature": 0.7, "num_predict": 300},
        )
        raw  = response["message"]["content"].strip()
        raw  = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        return data
    except Exception:
        return None


# ── Build eval set ────────────────────────────────────────────────────────────

def build_eval_set(conn) -> list[dict]:
    """
    Build full evaluation set:
    - 25 factual Q&A pairs (stratified by era)
    - 5 abstention cases
    """
    eval_set = []
    qid      = 0

    # ── Era-stratified factual questions ──────────────────────────────────────
    print("Generating factual questions by era...")

    for era_label, year_min, year_max, n_questions in ERA_BINS:
        print(f"\n  Era: {era_label}")
        chunks  = sample_chunks(conn, n_questions * 3, year_min, year_max)
        success = 0

        for chunk in chunks:
            if success >= n_questions:
                break

            print(f"    [{chunk['arxiv_id']}] {chunk['title'][:40]}...")
            qa = generate_qa_pair(chunk)

            if qa is None:
                print(f"    ✗ Generation failed or low quality")
                continue

            qid += 1
            item = {
                "id":                f"Q{qid:03d}",
                "era":               era_label,
                "intent":            "fact",
                "question":          qa["question"],
                "answer":            qa["answer"],
                "key_terms":         qa["key_terms"],
                "expected_papers":   [chunk["node_id"]],
                "expected_answer_contains": qa["key_terms"],
                "source_chunk_id":   qa["source_chunk_id"],
                "source_paper":      chunk["title"][:60],
                "source_year":       chunk["year"],
                "source_arxiv":      chunk["arxiv_id"],
                "section":           chunk["section_title"],
                "should_abstain":    False,
                "notes":             f"Auto-generated from {era_label}",
            }
            eval_set.append(item)
            success += 1
            print(f"    ✓ Q: {qa['question'][:60]}")
            print(f"      A: {qa['answer'][:60]}")
            time.sleep(0.3)  # polite pause

    # ── Abstention questions ──────────────────────────────────────────────────
    print(f"\nGenerating abstention questions...")
    abs_success = 0

    while abs_success < 5:
        data = generate_abstention_question()
        if data is None:
            continue

        question = data.get("question", "")
        reason   = data.get("reason_absent", "")

        if not question or len(question) < 15:
            continue

        qid += 1
        item = {
            "id":                    f"A{abs_success+1:03d}",
            "era":                   "N/A",
            "intent":                "fact",
            "question":              question,
            "answer":                None,
            "key_terms":             [],
            "expected_papers":       [],
            "expected_answer_contains": [],
            "source_chunk_id":       None,
            "source_paper":          None,
            "source_year":           None,
            "source_arxiv":          None,
            "section":               None,
            "should_abstain":        True,
            "notes":                 f"Abstention: {reason}",
        }
        eval_set.append(item)
        abs_success += 1
        print(f"  ✓ Abstention Q{abs_success}: {question[:60]}")
        time.sleep(0.3)

    return eval_set


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conn = get_connection()

    print(f"Building evaluation set...")
    print(f"Model: {QWEN_MODEL}")
    print(f"Target: 25 factual + 5 abstention = 30 questions\n")

    eval_set = build_eval_set(conn)
    conn.close()

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"Generated {len(eval_set)} questions")
    print(f"  Factual    : {sum(1 for q in eval_set if not q['should_abstain'])}")
    print(f"  Abstention : {sum(1 for q in eval_set if q['should_abstain'])}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"\nReview the output before using as ground truth:")
    print(f"  python -c \"import json; "
          f"[print(q['id'], q['question'][:60]) "
          f"for q in json.load(open('{OUTPUT_PATH}'))]\"")