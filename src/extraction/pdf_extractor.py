# src/extraction/pdf_extractor.py
"""
Stage A — PDF → structured chunk object.
Purely local, no network calls.
Input : path to a PDF file
Output: dict matching Layer 1 (partial) + Layer 2 of paper_node_schema.json
"""

import re
import fitz
from pathlib import Path
from src.extraction.text_cleaner import clean_page, split_into_paragraphs


# ── Filename parser ───────────────────────────────────────────────────────────

def parse_filename(pdf_path: Path) -> dict:
    """
    Derive arxiv_id from filename.
    2004_Murgia_astro-ph_0406225.pdf → astro-ph/0406225
    2009_Bonafede_0905.3552.pdf      → 0905.3552
    """
    stem  = pdf_path.stem
    parts = stem.split("_")
    year         = parts[0] if parts         else "unknown"
    first_author = parts[1] if len(parts) > 1 else "unknown"
    remainder    = "_".join(parts[2:])

    old_style = re.match(r"^([a-z\-]+)_(\d{7})$", remainder)
    new_style = re.match(r"^(\d{4}\.\d{4,5})$",   remainder)

    if old_style:
        arxiv_id = f"{old_style.group(1)}/{old_style.group(2)}"
    elif new_style:
        arxiv_id = new_style.group(1)
    else:
        arxiv_id = remainder

    return {
        "year":         year,
        "first_author": first_author,
        "arxiv_id":     arxiv_id,
        "node_id":      f"arxiv:{arxiv_id}",
        "pdf_path":     str(pdf_path),
    }


# ── Section detection ─────────────────────────────────────────────────────────

# Replace SECTION_PATTERNS and detect_section in pdf_extractor.py

SECTION_PATTERNS = [
    # Numbered: "1. Introduction" / "3.1 A magnetic field model"
    # Must start with digit, have a space, then capital letter word
    re.compile(r"^\d+\.?\d*\.?\d*\s+[A-Z][a-zA-Z\s]{3,50}$"),
    # Unnumbered named sections — exact matches only
    re.compile(
        r"^(Abstract|ABSTRACT|Introduction|INTRODUCTION"
        r"|Conclusions?|CONCLUSIONS?|Summary|SUMMARY"
        r"|References|REFERENCES"
        r"|Acknowledgements?|ACKNOWLEDGEMENTS?)$"
    ),
]

def detect_section(line: str, prev_section: str) -> str:
    line = line.strip()
    # Hard rules: reject if too long, too short, or contains math/numbers
    if not line:
        return prev_section
    if len(line) > 60:
        return prev_section
    if len(line) < 4:
        return prev_section
    # Reject lines containing equation-like content
    if re.search(r"[=±·×∝∼≃≈∫∑∂µλΛ]|\d+\.\d+", line):
        return prev_section
    for pattern in SECTION_PATTERNS:
        if pattern.match(line):
            return line
    return prev_section


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_paper(pdf_path: Path) -> dict:
    pdf_path = Path(pdf_path)
    meta     = parse_filename(pdf_path)
    doc      = fitz.open(pdf_path)
    total_pages = len(doc)

    chunks        = []
    chunk_index   = 0
    current_section = "preamble"
    section_index   = 0
    section_texts   = {"preamble": ""}

    for page_num, page in enumerate(doc):
        raw_text = page.get_text()
        if not raw_text.strip():
            continue

        clean_text = clean_page(raw_text)

        # Detect section changes line by line
        for line in clean_text.split("\n"):
            new_sec = detect_section(line, current_section)
            if new_sec != current_section:
                current_section = new_sec
                section_index  += 1
                section_texts.setdefault(current_section, "")

        # Paragraph-level chunks
        paragraphs = split_into_paragraphs(clean_text)

        for para in paragraphs:
            section_texts[current_section] = (
                section_texts.get(current_section, "") + " " + para
            ).strip()

            has_equation = bool(re.search(
                r"[∝∼≃≈∫∑∂σμλΛ]|\\[a-zA-Z]+|\bEq\.\s*\d", para))
            has_table    = bool(re.search(
                r"\bTab(?:le)?\.?\s*\d", para, re.IGNORECASE))
            has_fig_ref  = bool(re.search(
                r"\bFig(?:ure)?\.?\s*\d", para, re.IGNORECASE))
            fig_refs     = re.findall(
                r"Fig(?:ure)?\.?\s*\d+", para, re.IGNORECASE)
            eq_labels    = re.findall(
                r"Eq(?:uation)?\.?\s*\d+", para, re.IGNORECASE)
            has_quantity = bool(re.search(
                r"\d+\.?\d*\s*(?:µG|uG|kpc|Mpc|rad\s*m|MHz|GHz|cm|keV)", para))

            chunk = {
                "chunk_id":      f"{meta['node_id']}__c{chunk_index:04d}",
                "chunk_index":   chunk_index,
                "section_title": current_section,
                "section_index": section_index,
                "subsection":    None,
                "text":          para,
                "token_count":   len(para.split()),
                "page_num":      page_num + 1,
                "context_window": {
                    "section_summary": "",
                    "prev_chunk_id": (
                        f"{meta['node_id']}__c{chunk_index-1:04d}"
                        if chunk_index > 0 else None
                    ),
                    "next_chunk_id": f"{meta['node_id']}__c{chunk_index+1:04d}",
                },
                "content_flags": {
                    "has_equation":         has_equation,
                    "has_table":            has_table,
                    "has_figure_ref":       has_fig_ref,
                    "figure_refs":          fig_refs,
                    "equation_labels":      eq_labels,
                    "has_numerical_result": has_quantity,
                },
                "embedding": {
                    "model":    "allenai/specter2_base",
                    "vector":   None,
                    "index_id": None,
                },
                "sparse_tokens": {"keywords": []},
            }
            chunks.append(chunk)
            chunk_index += 1

    doc.close()

    # Post-pass: fix last next_chunk_id + attach section summaries
    for i, chunk in enumerate(chunks):
        if i == len(chunks) - 1:
            chunk["context_window"]["next_chunk_id"] = None
        sec = chunk["section_title"]
        chunk["context_window"]["section_summary"] = (
            section_texts.get(sec, "")[:300].strip()
        )

    return {
        "node_id":        meta["node_id"],
        "schema_version": "1.0",
        "layer_1_bibliographic": {
            "arxiv_id":       meta["arxiv_id"],
            "bibcode":        None,
            "doi":            None,
            "title":          None,
            "authors":        [],
            "year":           meta["year"],
            "journal":        None,
            "volume":         None,
            "pages":          None,
            "abstract":       None,
            "keywords":       [],
            "citation_count": None,
            "pdf_path":       meta["pdf_path"],
            "total_pages":    total_pages,
        },
        "layer_2_chunks": {
            "chunks":            chunks,
            "total_chunks":      len(chunks),
            "sections_detected": list(section_texts.keys()),
        },
        "layer_3_structured": None,
        "layer_4_graph":      None,
        "meta": {
            "schema_version":    "1.0",
            "stage_a":           "done",
            "stage_b":           "pending",
            "stage_c":           "pending",
            "stage_d":           "pending",
            "ingestion_date":    None,
            "extraction_model":  None,
            "embedding_model":   None,
            "manually_verified": False,
            "notes":             None,
        }
    }