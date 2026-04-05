# src/extraction/structured_extractor.py
"""
Step 12 — Layer 3 structured extraction.
Uses Qwen3:14b to extract structured metadata from each paper.

Extracts per paper:
  methods          — modelling approaches used
  key_quantities   — numerical results with values and units
  scientific_claims — main findings as atomic statements
  physical_domain  — ICM sub-topics addressed
  instruments      — telescopes/surveys used
  clusters         — galaxy clusters studied

Stored as JSON in paper_extractions DuckDB table.
Enables SQL queries: "find papers with B0 > 5 µG"
"""

import json
import re
import time
from pathlib import Path
import ollama

QWEN_MODEL = "qwen3:14b"

# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are extracting structured metadata from an astrophysics paper about intracluster magnetic fields, Faraday rotation, and radio observations of galaxy clusters.

Extract the following from the paper text below. Be precise and conservative — only extract what is explicitly stated.

Paper title: {title}
Year: {year}

Text (first 2000 characters of key sections):
{text}

Extract and return ONLY a JSON object with these fields:

{{
  "methods": [
    "list of modelling/analysis methods used",
    "e.g. GRF simulation, BxC model, RM synthesis, MHD simulation, Bayesian analysis"
  ],
  "key_quantities": [
    {{
      "name": "quantity name e.g. B0, sigma_RM, n, Lambda_min",
      "value": "numeric value as string e.g. 5.0",
      "unit": "unit e.g. µG, rad/m2, kpc",
      "cluster": "cluster name if specific e.g. A119 or null",
      "context": "brief context e.g. central magnetic field strength"
    }}
  ],
  "scientific_claims": [
    "list of main findings as short atomic statements",
    "e.g. The magnetic field follows a power law with n=2",
    "Maximum 5 claims"
  ],
  "physical_domain": [
    "list of ICM sub-topics e.g. Faraday rotation, depolarization, radio halos, magnetic field modelling, turbulence"
  ],
  "instruments": [
    "list of telescopes or surveys used e.g. VLA, LOFAR, MeerKAT, Chandra"
  ],
  "clusters": [
    "list of galaxy cluster names studied e.g. Abell 119, Coma, A2255"
  ]
}}

Return ONLY the JSON object. No explanation, no markdown fences."""


# ── Text assembler for extraction ─────────────────────────────────────────────

def assemble_extraction_text(chunks: list[dict], max_chars: int = 2000) -> str:
    """
    Assemble key chunks for extraction.
    Prioritises abstract, results, conclusion sections.
    """
    priority_sections = [
    "abstract", "conclusion", "conclusions", "summary",
    "results", "method", "methods", "discussion",
]

    # Sort chunks by section priority
    def section_priority(chunk):
        sec = (chunk.get("section_title") or "").lower()
        for i, ps in enumerate(priority_sections):
            if ps in sec:
                return i
        return len(priority_sections)

    sorted_chunks = sorted(chunks, key=section_priority)

    # Assemble text up to max_chars
    parts = []
    total = 0
    for chunk in sorted_chunks:
        text = chunk.get("text", "").strip()
        # Skip table-like chunks
        if text.startswith("Column") or text.count("|") > 5:
            continue
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(text[:remaining])
            break
        parts.append(text)
        total += len(text)

    return "\n\n".join(parts)


# ── Qwen3 extraction ──────────────────────────────────────────────────────────

def extract_paper_metadata(
    node_id: str,
    title:   str,
    year:    int,
    chunks:  list[dict],
) -> dict | None:
    """
    Extract structured metadata from a paper using Qwen3.
    Returns parsed dict or None on failure.
    """
    text   = assemble_extraction_text(chunks)
    prompt = EXTRACTION_PROMPT.format(
        title=title[:100],
        year=year or "unknown",
        text=text,
    )

    try:
        response = ollama.chat(
            model    = QWEN_MODEL,
            messages = [{"role": "user", "content": prompt}],
            think    = False,
            options  = {"temperature": 0.1, "num_predict": 800},
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        data = json.loads(raw)

        # Validate required keys present
        required = [
            "methods", "key_quantities", "scientific_claims",
            "physical_domain", "instruments", "clusters"
        ]
        for key in required:
            if key not in data:
                data[key] = []

        return data

    except (json.JSONDecodeError, KeyError, Exception) as e:
        return None


# ── DuckDB storage ────────────────────────────────────────────────────────────

def store_extraction(conn, node_id: str, data: dict, model: str) -> None:
    """Store extraction result in paper_extractions table."""
    conn.execute("""
        INSERT OR REPLACE INTO paper_extractions
        (node_id, methods, key_quantities, scientific_claims,
         physical_domain, instruments, clusters, extraction_model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        node_id,
        json.dumps(data.get("methods", [])),
        json.dumps(data.get("key_quantities", [])),
        json.dumps(data.get("scientific_claims", [])),
        json.dumps(data.get("physical_domain", [])),
        json.dumps(data.get("instruments", [])),
        json.dumps(data.get("clusters", [])),
        model,
    ])


def is_extracted(conn, node_id: str) -> bool:
    """Check if paper already extracted."""
    row = conn.execute(
        "SELECT 1 FROM paper_extractions WHERE node_id = ?",
        [node_id]
    ).fetchone()
    return row is not None


# ── Batch extraction runner ───────────────────────────────────────────────────

def run_extraction_batch(
    conn,
    limit:    int   = None,
    sleep:    float = 0.2,
    resume:   bool  = True,
) -> dict:
    """
    Run structured extraction over all papers in DuckDB.
    Resumable — skips already-extracted papers.

    Returns stats dict.
    """
    # Get all papers
    rows = conn.execute("""
        SELECT node_id, title, year
        FROM papers
        ORDER BY year ASC
    """).fetchall()

    if limit:
        rows = rows[:limit]

    total    = len(rows)
    success  = 0
    skipped  = 0
    failed   = 0

    print(f"Running structured extraction on {total} papers...")
    print(f"Model: {QWEN_MODEL}\n")

    for i, (node_id, title, year) in enumerate(rows):

        # Skip if already done
        if resume and is_extracted(conn, node_id):
            skipped += 1
            continue

        print(f"[{i+1}/{total}] {node_id} ({year})")
        print(f"  {title[:60]}")

        # Fetch chunks for this paper
        chunk_rows = conn.execute("""
            SELECT section_title, text
            FROM chunks
            WHERE node_id = ?
              AND is_noise = FALSE
              AND token_count > 30
            ORDER BY chunk_index
        """, [node_id]).fetchall()

        chunks = [
            {"section_title": r[0], "text": r[1]}
            for r in chunk_rows
        ]

        if not chunks:
            print(f"  ✗ No chunks found")
            failed += 1
            continue

        # Extract
        data = extract_paper_metadata(node_id, title or "", year or 0, chunks)

        if data is None:
            print(f"  ✗ Extraction failed")
            failed += 1
            time.sleep(sleep)
            continue

        # Store
        store_extraction(conn, node_id, data, QWEN_MODEL)

        # Print summary
        n_quantities = len(data.get("key_quantities", []))
        n_claims     = len(data.get("scientific_claims", []))
        methods      = data.get("methods", [])[:3]
        print(f"  ✓ methods={methods} "
              f"quantities={n_quantities} claims={n_claims}")

        success += 1
        time.sleep(sleep)

    stats = {
        "total":   total,
        "success": success,
        "skipped": skipped,
        "failed":  failed,
    }

    print(f"\nExtraction complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats