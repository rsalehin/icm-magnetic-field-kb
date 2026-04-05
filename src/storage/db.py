# src/storage/db.py
"""
DuckDB schema initialisation and connection management.
Handles Layer 1 (bibliographic) and Layer 2 (chunks) storage.
Layer 3 structured extraction stored as JSON columns for flexibility.
"""

import duckdb
import json
from pathlib import Path

DB_PATH = Path("data/knowledge_base.duckdb")


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection, creating the file if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create all tables if they don't exist.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """

    # ── papers table — Layer 1 bibliographic ─────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            node_id         VARCHAR PRIMARY KEY,
            arxiv_id        VARCHAR,
            bibcode         VARCHAR,
            doi             VARCHAR,
            title           VARCHAR,
            year            INTEGER,
            journal         VARCHAR,
            volume          VARCHAR,
            pages           VARCHAR,
            abstract        VARCHAR,
            keywords        VARCHAR,   -- JSON array stored as string
            citation_count  INTEGER,
            pdf_path        VARCHAR,
            total_pages     INTEGER,
            total_chunks    INTEGER,
            sections        VARCHAR,   -- JSON array stored as string
            stage_a         VARCHAR DEFAULT 'done',
            stage_b         VARCHAR DEFAULT 'pending',
            stage_c         VARCHAR DEFAULT 'pending',
            stage_d         VARCHAR DEFAULT 'pending',
            ingestion_date  TIMESTAMP DEFAULT current_timestamp,
            manually_verified BOOLEAN DEFAULT FALSE,
            notes           VARCHAR
        )
    """)
    
    # ── paper_extractions table — Layer 3 structured extraction ──────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_extractions (
            node_id           VARCHAR PRIMARY KEY,
            methods           JSON,
            key_quantities    JSON,
            scientific_claims JSON,
            physical_domain   JSON,
            instruments       JSON,
            clusters          JSON,
            extraction_model  VARCHAR,
            extracted_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        )
    """)

    conn.commit()
    print("Schema initialised successfully.")

    # ── authors table — normalised author list ────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS authors (
            node_id     VARCHAR,
            position    INTEGER,
            author_name VARCHAR,
            PRIMARY KEY (node_id, position)
        )
    """)

    # ── chunks table — Layer 2 paragraph chunks ───────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id        VARCHAR PRIMARY KEY,
            node_id         VARCHAR,
            chunk_index     INTEGER,
            section_title   VARCHAR,
            section_index   INTEGER,
            text            VARCHAR,
            token_count     INTEGER,
            page_num        INTEGER,
            has_equation    BOOLEAN,
            has_table       BOOLEAN,
            has_figure_ref  BOOLEAN,
            has_numerical_result BOOLEAN,
            figure_refs     VARCHAR,   -- JSON array
            equation_labels VARCHAR,   -- JSON array
            prev_chunk_id   VARCHAR,
            next_chunk_id   VARCHAR,
            section_summary VARCHAR,
            faiss_index_id  INTEGER,   -- row index in FAISS flat index
            is_noise        BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (node_id) REFERENCES papers(node_id)
        )
    """)

    # ── ingestion_state table — tracks pipeline progress per paper ────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_state (
            node_id     VARCHAR PRIMARY KEY,
            filename    VARCHAR,
            stage_a     VARCHAR DEFAULT 'pending',
            stage_b     VARCHAR DEFAULT 'pending',
            stage_c     VARCHAR DEFAULT 'pending',
            stage_d     VARCHAR DEFAULT 'pending',
            last_updated TIMESTAMP DEFAULT current_timestamp
        )
    """)

    conn.commit()
    print("Schema initialised successfully.")


def insert_paper(conn: duckdb.DuckDBPyConnection, paper: dict) -> None:
    """
    Insert a fully enriched paper dict (Stages A+B) into DuckDB.
    Skips if node_id already exists (idempotent).
    """
    node_id = paper["node_id"]

    # Check if already exists
    existing = conn.execute(
        "SELECT node_id FROM papers WHERE node_id = ?", [node_id]
    ).fetchone()
    if existing:
        print(f"  Already in DB, skipping: {node_id}")
        return

    l1 = paper["layer_1_bibliographic"]
    l2 = paper["layer_2_chunks"]
    m  = paper["meta"]

    # ── Insert into papers ────────────────────────────────────────────────────
    conn.execute("""
        INSERT INTO papers (
            node_id, arxiv_id, bibcode, doi, title, year, journal,
            volume, pages, abstract, keywords, citation_count,
            pdf_path, total_pages, total_chunks, sections,
            stage_a, stage_b, stage_c, stage_d,
            manually_verified, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        node_id,
        l1.get("arxiv_id"),
        l1.get("bibcode"),
        l1.get("doi"),
        l1.get("title"),
        int(l1.get("year", 0)) if l1.get("year") else None,
        l1.get("journal"),
        l1.get("volume"),
        l1.get("pages"),
        l1.get("abstract"),
        json.dumps(l1.get("keywords", [])),
        l1.get("citation_count"),
        l1.get("pdf_path"),
        l1.get("total_pages"),
        l2.get("total_chunks"),
        json.dumps(l2.get("sections_detected", [])),
        m.get("stage_a", "done"),
        m.get("stage_b", "pending"),
        m.get("stage_c", "pending"),
        m.get("stage_d", "pending"),
        m.get("manually_verified", False),
        m.get("notes"),
    ])

    # ── Insert authors ────────────────────────────────────────────────────────
    for i, author in enumerate(l1.get("authors", [])):
        conn.execute("""
            INSERT INTO authors (node_id, position, author_name)
            VALUES (?, ?, ?)
        """, [node_id, i, author])

    # ── Insert chunks ─────────────────────────────────────────────────────────
    for chunk in l2.get("chunks", []):
        flags = chunk["content_flags"]
        cw    = chunk["context_window"]
        conn.execute("""
            INSERT INTO chunks (
                chunk_id, node_id, chunk_index, section_title,
                section_index, text, token_count, page_num,
                has_equation, has_table, has_figure_ref,
                has_numerical_result, figure_refs, equation_labels,
                prev_chunk_id, next_chunk_id, section_summary, is_noise
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            chunk["chunk_id"],
            node_id,
            chunk["chunk_index"],
            chunk["section_title"],
            chunk["section_index"],
            chunk["text"],
            chunk["token_count"],
            chunk["page_num"],
            flags["has_equation"],
            flags["has_table"],
            flags["has_figure_ref"],
            flags["has_numerical_result"],
            json.dumps(flags["figure_refs"]),
            json.dumps(flags["equation_labels"]),
            cw["prev_chunk_id"],
            cw["next_chunk_id"],
            cw["section_summary"],
            chunk["token_count"] < 30,   # mark noise chunks
        ])

    # ── Update ingestion state ────────────────────────────────────────────────
    conn.execute("""
        INSERT OR REPLACE INTO ingestion_state
            (node_id, filename, stage_a, stage_b, stage_c, stage_d)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [
        node_id,
        l1.get("pdf_path", "").split("\\")[-1],
        m.get("stage_a", "done"),
        m.get("stage_b", "pending"),
        m.get("stage_c", "pending"),
        m.get("stage_d", "pending"),
    ])

    conn.commit()


def get_ingestion_state(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Return current ingestion state for all papers."""
    rows = conn.execute("""
        SELECT node_id, filename, stage_a, stage_b, stage_c, stage_d,
               last_updated
        FROM ingestion_state
        ORDER BY last_updated
    """).fetchall()
    cols = ["node_id", "filename", "stage_a", "stage_b",
            "stage_c", "stage_d", "last_updated"]
    return [dict(zip(cols, row)) for row in rows]