# scripts/regenerate_eval_questions.py
"""
Regenerate off-topic eval questions with ICM-targeted prompts.
Replaces questions identified as off-topic in review.
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

GENERATED_PATH = Path("data/eval_set_generated.json")
random.seed(99)

# IDs to replace
REPLACE_IDS = {
    "Q003", "Q007", "Q009", "Q010", "Q012",
    "Q013", "Q016", "Q019", "Q021", "Q022",
    "Q024", "Q025", "A002", "A005"
}

# ── Targeted ICM prompt ───────────────────────────────────────────────────────

ICM_PROMPT = """You are evaluating a scientific retrieval system for astrophysics research on intracluster magnetic fields.

Given this chunk from a paper about intracluster magnetic fields, Faraday rotation, radio observations of galaxy clusters, or related topics, generate ONE precise factual question whose answer is EXPLICITLY stated in the text.

The question MUST be about one of these topics:
- Magnetic field strength (B₀, sigma_B, central field)
- Rotation measure (RM, sigma_RM, mean RM)
- Depolarization (Burn law, beam depolarization)
- Power spectrum (spectral index n, Λmin, Λmax)
- Radio observations (frequency, resolution, flux)
- Cluster properties (core radius, beta-model, electron density)
- Simulation parameters (grid size, cell size, model)

Do NOT ask about: cosmological parameters, dark matter, neutrinos, chameleon fields, GRBs, cluster mass estimates, jet spectral indices unrelated to ICM.

Paper: {title} ({year})
Section: {section}
Text:
{text}

Respond with ONLY a JSON object:
{{"question": "...", "answer": "...", "key_terms": ["term1", "term2", "term3"]}}"""

ICM_ABSTENTION_PROMPT = """Generate ONE question that sounds related to galaxy cluster magnetic fields but is NOT answerable from literature about Faraday rotation, RM synthesis, GRF/BxC models, or radio observations of clusters.

The question should NOT be about: dark matter, neutrinos, chameleon fields, or X-ray emission mechanisms.

Instead ask about something like:
- Optical properties of cluster galaxies
- Gravitational wave signatures from clusters
- Chemical enrichment of the ICM
- Stellar populations in cluster galaxies
- Gamma-ray emission mechanisms not covered in radio magnetic field papers

Respond with ONLY a JSON object:
{{"question": "...", "reason_absent": "..."}}"""


def sample_icm_chunks(conn, n: int, year_min: int, year_max: int) -> list[dict]:
    """Sample chunks from papers likely about ICM magnetic fields."""
    rows = conn.execute("""
        SELECT
            c.chunk_id, c.node_id, c.section_title, c.text,
            c.token_count, c.has_numerical_result, c.has_equation,
            p.year, p.title, p.arxiv_id
        FROM chunks c
        JOIN papers p ON c.node_id = p.node_id
        WHERE c.is_noise = FALSE
          AND c.token_count BETWEEN 80 AND 350
          AND p.year BETWEEN ? AND ?
          AND c.has_numerical_result = TRUE
          AND (
              c.text ILIKE '%rotation measure%'
              OR c.text ILIKE '%magnetic field%'
              OR c.text ILIKE '%sigma_RM%'
              OR c.text ILIKE '%Faraday%'
              OR c.text ILIKE '%depolarization%'
              OR c.text ILIKE '%power spectrum%'
              OR c.text ILIKE '% µG%'
              OR c.text ILIKE '%rad m%'
          )
        ORDER BY RANDOM()
        LIMIT ?
    """, [year_min, year_max, n * 6]).fetchall()

    cols = ["chunk_id", "node_id", "section_title", "text",
            "token_count", "has_numerical_result", "has_equation",
            "year", "title", "arxiv_id"]
    chunks = [dict(zip(cols, r)) for r in rows]

    # Max 1 chunk per paper for diversity
    seen: set[str] = set()
    selected = []
    for c in chunks:
        if c["node_id"] not in seen:
            selected.append(c)
            seen.add(c["node_id"])
        if len(selected) >= n:
            break

    return selected[:n]


def generate_icm_qa(chunk: dict) -> dict | None:
    """Generate ICM-targeted Q&A pair."""
    prompt = ICM_PROMPT.format(
        title   = chunk["title"][:80],
        year    = chunk["year"],
        section = chunk["section_title"],
        text    = chunk["text"][:800],
    )
    try:
        response = ollama.chat(
            model    = "qwen3:14b",
            messages = [{"role": "user", "content": prompt}],
            think    = False,
            options  = {"temperature": 0.3, "num_predict": 500},
        )
        raw  = response["message"]["content"].strip()
        raw  = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)

        question = data.get("question", "").strip()
        answer   = data.get("answer", "").strip()
        terms    = data.get("key_terms", [])

        if not question or not answer or len(question) < 15:
            return None
        if "?" not in question:
            return None

        # Verify answer in chunk
        answer_words = set(answer.lower().split())
        chunk_words  = set(chunk["text"].lower().split())
        overlap = len(answer_words & chunk_words) / max(len(answer_words), 1)
        if overlap < 0.25:
            return None

        return {"question": question, "answer": answer, "key_terms": terms[:5]}

    except Exception:
        return None


def generate_icm_abstention() -> dict | None:
    """Generate ICM-specific abstention question."""
    try:
        response = ollama.chat(
            model    = "qwen3:14b",
            messages = [{"role": "user", "content": ICM_ABSTENTION_PROMPT}],
            think    = False,
            options  = {"temperature": 0.7, "num_predict": 300},
        )
        raw  = response["message"]["content"].strip()
        raw  = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        return data
    except Exception:
        return None


# ── Era mapping for replacements ──────────────────────────────────────────────

ERA_YEAR_MAP = {
    "Q003": (1995, 2004), "Q007": (2005, 2009),
    "Q009": (2005, 2009), "Q010": (2005, 2009),
    "Q012": (2010, 2014), "Q013": (2010, 2014),
    "Q016": (2015, 2019), "Q019": (2015, 2019),
    "Q021": (2020, 2026), "Q022": (2020, 2026),
    "Q024": (2020, 2026), "Q025": (2020, 2026),
}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(GENERATED_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    conn = get_connection()
    new_eval_set = []

    for item in eval_set:
        qid = item["id"]

        if qid not in REPLACE_IDS:
            new_eval_set.append(item)
            continue

        # Abstention replacement
        if qid in {"A002", "A005"}:
            print(f"\nRegenerating abstention {qid}...")
            for _ in range(5):
                data = generate_icm_abstention()
                if data and len(data.get("question", "")) > 15:
                    item = dict(item)
                    item["question"] = data["question"]
                    item["notes"]    = f"Abstention: {data.get('reason_absent', '')}"
                    print(f"  ✓ {data['question'][:60]}")
                    break
                time.sleep(0.3)
            new_eval_set.append(item)
            continue

        # Factual replacement
        year_min, year_max = ERA_YEAR_MAP.get(qid, (2000, 2026))
        era_label = item.get("era", f"{year_min}-{year_max}")
        print(f"\nRegenerating {qid} ({era_label})...")

        chunks  = sample_icm_chunks(conn, 6, year_min, year_max)
        success = False

        for chunk in chunks:
            qa = generate_icm_qa(chunk)
            if qa is None:
                continue

            new_item = dict(item)
            new_item.update({
                "question":               qa["question"],
                "answer":                 qa["answer"],
                "key_terms":              qa["key_terms"],
                "expected_papers":        [chunk["node_id"]],
                "expected_answer_contains": qa["key_terms"],
                "source_chunk_id":        chunk["chunk_id"],
                "source_paper":           chunk["title"][:60],
                "source_year":            chunk["year"],
                "source_arxiv":           chunk["arxiv_id"],
                "section":                chunk["section_title"],
                "notes":                  f"Regenerated ICM-targeted from {era_label}",
            })
            new_eval_set.append(new_item)
            print(f"  ✓ Q: {qa['question'][:60]}")
            print(f"    A: {qa['answer'][:60]}")
            success = True
            time.sleep(0.3)
            break

        if not success:
            print(f"  ✗ Failed to regenerate {qid} — keeping original")
            new_eval_set.append(item)

    conn.close()

    # Save
    out_path = Path("data/eval_set_final.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_eval_set, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"Final eval set: {len(new_eval_set)} questions")
    factual   = sum(1 for q in new_eval_set if not q["should_abstain"])
    abstain   = sum(1 for q in new_eval_set if q["should_abstain"])
    print(f"  Factual    : {factual}")
    print(f"  Abstention : {abstain}")
    print(f"Saved to: {out_path}")