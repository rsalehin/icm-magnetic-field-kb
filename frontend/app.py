# frontend/app.py
"""
FastAPI server for ICM Knowledge Base chatbot UI.
Loads all pipeline backends at startup.
Serves static files + API endpoints.
"""

import sys
import json
import uuid
import asyncio
import warnings
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", message=".*position_ids.*")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("FlagEmbedding").setLevel(logging.ERROR)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from src.storage.db          import get_connection
from src.storage.graph       import get_or_create_graph
from src.storage.faiss_index import load_id_map, INDEX_PATH
from src.query.planner       import plan_query
from src.query.bm25_index    import load_bm25_index
from src.query.bge_embedder  import (
    get_bge_model, get_or_create_bge_index,
    load_bge_id_map, BGE_MATRIX_PATH,
)
from src.query.retriever     import retrieve
from src.query.reranker      import get_reranker, rerank
from src.query.assembler     import assemble_evidence
from src.query.generator     import generate_answer
from src.pipeline            import get_model

from frontend.conversations_db import (
    get_conv_connection, init_conv_schema,
    create_conversation, add_message,
    list_conversations, get_messages,
    delete_conversation,
)

# ── Global state ──────────────────────────────────────────────────────────────

STATE = {}
EXECUTOR = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all backends at startup."""
    print("Loading backends...")

    STATE["conn"]           = get_connection()
    STATE["specter_model"]  = get_model()
    STATE["specter_index"]  = faiss.read_index(str(INDEX_PATH))
    STATE["specter_id_map"] = load_id_map()
    STATE["bm25_index"], STATE["bm25_records"] = load_bm25_index()
    STATE["bge_model"]      = get_bge_model()
    STATE["bge_index"]      = get_or_create_bge_index()
    STATE["bge_id_map"]     = load_bge_id_map()
    STATE["reranker"]       = get_reranker()
    STATE["G"]              = get_or_create_graph()

    # Load BGE matrix for deduplication
    if BGE_MATRIX_PATH.exists():
        STATE["bge_matrix"] = np.load(str(BGE_MATRIX_PATH))
    else:
        STATE["bge_matrix"] = None

    # Conversations DB
    STATE["conv_conn"] = get_conv_connection()
    init_conv_schema(STATE["conv_conn"])

    print("All backends ready.")
    yield

    STATE["conn"].close()
    STATE["conv_conn"].close()


app = FastAPI(lifespan=lifespan, title="ICM Knowledge Base")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:        str
    conversation_id: str | None = None


# ── Pipeline runner ───────────────────────────────────────────────────────────
def _best_snippet(text: str, max_chars: int = 350) -> str:
    """
    Find the most informative sentence in a chunk.
    Prefers sentences with numbers, units, or key ICM terms.
    Falls back to first sentence if no strong candidate found.
    """
    if not text:
        return ""

    # Split into sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return text[:max_chars]

    # Score each sentence
    SIGNAL_TERMS = {
        'µg', 'μg', 'rad', 'kpc', 'mpc', 'ghz', 'mhz',
        'sigma', 'λ', 'b_0', 'b0', 'sigma_rm', '=', '≈', '∼', '~',
        'magnetic', 'rotation measure', 'faraday', 'power spectrum',
        'depolarization', 'spectral', 'cluster', '%',
    }

    def score(s: str) -> float:
        sl   = s.lower()
        hits = sum(1 for t in SIGNAL_TERMS if t in sl)
        # Bonus for numbers
        hits += len(re.findall(r'\d+\.?\d*', s)) * 0.5
        # Prefer medium-length sentences
        length_bonus = 1.0 if 60 < len(s) < 300 else 0.5
        return hits * length_bonus

    scored = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
    best_idx, best_sent = scored[0]

    # Take best sentence + one following sentence for context
    result_parts = [best_sent]
    if best_idx + 1 < len(sentences):
        result_parts.append(sentences[best_idx + 1])
    result = ' '.join(result_parts)

    # Trim to max_chars at word boundary
    if len(result) > max_chars:
        result = result[:max_chars]
        pos = result.rfind(' ')
        if pos > max_chars * 0.7:
            result = result[:pos] + '…'

    return result



def run_pipeline(question: str) -> dict:
    """
    Run full pipeline synchronously.
    Called in thread pool to avoid blocking event loop.
    """
    pipeline_trace = {}

    # Planner
    plan = plan_query(question)
    pipeline_trace["planner"] = {
        "intent":         plan["intent"],
        "synthesis_mode": plan["synthesis_mode"],
        "entities":       plan["entities"],
        "budgets":        plan["budgets"],
    }

    # Retriever
    chunks = retrieve(
        plan,
        STATE["conn"],
        STATE["specter_index"],
        STATE["specter_id_map"],
        STATE["specter_model"],
        STATE["bm25_index"],
        STATE["bm25_records"],
        STATE["bge_index"],
        STATE["bge_id_map"],
        STATE["bge_model"],
        G=STATE["G"],
    )
    pipeline_trace["retriever"] = {
        "paper_k":      plan["budgets"]["paper_k"],
        "chunks_fused": len(chunks),
    }

    # Reranker
    evidence_chunks = rerank(
        question,
        chunks,
        intent     = plan["intent"],
        evidence_k = plan["budgets"]["evidence_k"],
        min_papers = plan["budgets"]["min_papers"],
        reranker   = STATE["reranker"],
    )
    pipeline_trace["reranker"] = {
        "input_chunks":   len(chunks),
        "output_chunks":  len(evidence_chunks),
        "distinct_papers": len({c["node_id"] for c in evidence_chunks}),
        "max_rerank_score": max(
            (c.get("rerank_score", 0) for c in evidence_chunks), default=0
        ),
        "top_chunks": [
            {
                "paper_id":     c["node_id"],
                "section":      c.get("section_title", ""),
                "rerank_score": round(c.get("rerank_score", 0), 4),
                "text_snippet": c.get("text", "")[:120],
            }
            for c in evidence_chunks[:5]
        ],
    }

    # Assembler
    pack = assemble_evidence(
        question,
        plan,
        evidence_chunks,
        STATE["conn"],
        STATE["bge_id_map"],
        STATE["bge_matrix"],
    )
    pipeline_trace["assembler"] = {
        "distinct_chunks":     pack["support_stats"]["distinct_chunks"],
        "distinct_papers":     pack["support_stats"]["distinct_papers"],
        "max_rerank_score":    pack["support_stats"]["max_rerank_score"],
        "has_disagreement":    pack["support_stats"]["has_disagreement"],
        "has_uncertainty":     pack["support_stats"]["has_uncertainty"],
        "abstain":             pack["abstain"],
        "abstain_reason":      pack.get("abstain_reason"),
        "deduplicated":        pack["support_stats"]["chunks_deduplicated"],
    }

    # Generator
    result = generate_answer(
        question,
        pack,
        synthesis_mode=plan["synthesis_mode"],
    )

    # Build evidence list for pipeline trace
    # Get page numbers for evidence chunks
    # Get page numbers for evidence chunks
    chunk_page_map = {}
    evidence_chunks_ids = [c.get("chunk_id","") for c in evidence_chunks]
    if evidence_chunks_ids:
        placeholders = ",".join(["?" for _ in evidence_chunks_ids])
        page_rows = STATE["conn"].execute(
            f"SELECT chunk_id, page_num FROM chunks WHERE chunk_id IN ({placeholders})",
            evidence_chunks_ids
        ).fetchall()
        chunk_page_map = {r[0]: r[1] for r in page_rows}

    pipeline_trace["evidence"] = [
        {
            "rank":         i + 1,
            "paper_id":     e.get("paper_id", ""),
            "title":        e.get("title", "")[:80],
            "year":         e.get("year"),
            "journal":      e.get("journal", ""),
            "section":      e.get("section", ""),
            "rerank_score": round(e.get("rerank_score", 0), 4),
            "text_snippet": _best_snippet(e.get("text", ""), 350),
            "chunk_id":     e.get("chunk_id", ""),
            "page_num":     chunk_page_map.get(e.get("chunk_id",""), 1),
            "chunk_id":     e.get("chunk_id", ""),
            "page_num":     chunk_page_map.get(e.get("chunk_id",""), 1),
        }
        for i, e in enumerate(pack.get("evidence", [])[:12])
    ]

    return {
        "answer":         result["answer"],
        "intent":         result["intent"],
        "abstained":      result["abstained"],
        "citations":      result["citations"],
        "pipeline_trace": pipeline_trace,
    }


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # Run pipeline in thread pool
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        EXECUTOR, run_pipeline, req.question
    )

    conv_conn = STATE["conv_conn"]

    # Get or create conversation
    conv_id = req.conversation_id
    if not conv_id:
        conv_id = create_conversation(conv_conn, req.question)

    # Store user message
    add_message(
        conv_conn,
        conversation_id = conv_id,
        role            = "user",
        content         = req.question,
    )

    # Store assistant message
    add_message(
        conv_conn,
        conversation_id = conv_id,
        role            = "assistant",
        content         = result["answer"],
        intent          = result["intent"],
        abstained       = result["abstained"],
        pipeline_trace  = result["pipeline_trace"],
        citations       = result["citations"],
    )

    return {
        "conversation_id": conv_id,
        "answer":          result["answer"],
        "intent":          result["intent"],
        "abstained":       result["abstained"],
        "pipeline_trace":  result["pipeline_trace"],
    }


@app.get("/api/conversations")
async def get_conversations():
    return list_conversations(STATE["conv_conn"])


@app.get("/api/conversation/{conv_id}")
async def get_conversation(conv_id: str):
    msgs = get_messages(STATE["conv_conn"], conv_id)
    if not msgs:
        raise HTTPException(404, "Conversation not found")
    return msgs


@app.delete("/api/conversation/{conv_id}")
async def del_conversation(conv_id: str):
    delete_conversation(STATE["conv_conn"], conv_id)
    return {"status": "deleted"}


@app.get("/api/graph")
async def get_graph():
    """Return corpus papers as nodes + edges between them."""
    G    = STATE["G"]
    conn = STATE["conn"]

    # Get all corpus papers
    rows = conn.execute("""
        SELECT node_id, title, year, journal, citation_count
        FROM papers
        ORDER BY year
    """).fetchall()

    corpus_ids = {r[0] for r in rows}

    # Build era color mapping
    def era_color(year):
        if not year:
            return "#3a4a5a"
        if year < 2000: return "#2d5a3d"
        if year < 2005: return "#3d6b2d"
        if year < 2010: return "#6b7a2d"
        if year < 2015: return "#7a5c2d"
        if year < 2020: return "#7a3d2d"
        return "#5a2d7a"

    nodes = [
        {
            "id":             r[0],
            "title":          (r[1] or "")[:60],
            "year":           r[2],
            "journal":        r[3] or "",
            "citation_count": r[4] or 0,
            "color":          era_color(r[2]),
            "size":           max(4, min(20, (r[4] or 0) / 40)),
        }
        for r in rows
    ]

    # Get edges between corpus papers only
    edges = []
    for u, v, data in G.edges(data=True):
        if u in corpus_ids and v in corpus_ids:
            edges.append({
                "source": u,
                "target": v,
                "type":   data.get("citation_role", "incidental"),
                "weight": data.get("weight", 0.4),
            })

    return {"nodes": nodes, "edges": edges}


@app.get("/api/stats")
async def get_stats():
    conn = STATE["conn"]

    era_rows = conn.execute("""
        SELECT
            CASE
                WHEN year < 2000 THEN '1995-1999'
                WHEN year < 2005 THEN '2000-2004'
                WHEN year < 2010 THEN '2005-2009'
                WHEN year < 2015 THEN '2010-2014'
                WHEN year < 2020 THEN '2015-2019'
                ELSE '2020-2026'
            END as era,
            COUNT(*) as n
        FROM papers GROUP BY era ORDER BY era
    """).fetchall()

    author_rows = conn.execute("""
        SELECT author_name, COUNT(*) as n
        FROM authors WHERE position = 0
        GROUP BY author_name ORDER BY n DESC LIMIT 10
    """).fetchall()

    journal_rows = conn.execute("""
        SELECT journal, COUNT(*) as n
        FROM papers WHERE journal IS NOT NULL
        GROUP BY journal ORDER BY n DESC LIMIT 8
    """).fetchall()

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE is_noise = FALSE").fetchone()[0]
    G = STATE["G"]

    return {
        "total_papers":   total,
        "total_chunks":   chunks,
        "graph_nodes":    G.number_of_nodes(),
        "graph_edges":    G.number_of_edges(),
        "by_era":         [{"era": r[0], "count": r[1]} for r in era_rows],
        "top_authors":    [{"name": r[0], "count": r[1]} for r in author_rows],
        "by_journal":     [{"journal": r[0], "count": r[1]} for r in journal_rows],
    }

import urllib.parse
from fastapi.responses import FileResponse, HTMLResponse

@app.get("/api/pdf/{node_id:path}")
async def serve_pdf(node_id: str):
    """Serve PDF file for a given node_id."""
    row = STATE["conn"].execute(
        "SELECT pdf_path FROM papers WHERE node_id = ?",
        [node_id]
    ).fetchone()
    if not row or not row[0]:
        raise HTTPException(404, f"No PDF found for {node_id}")

    pdf_path = Path(__file__).parent.parent / row[0]
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF file not found: {pdf_path}")

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/api/chunk-page/{chunk_id:path}")
async def get_chunk_page(chunk_id: str):
    """Return page number for a chunk_id."""
    row = STATE["conn"].execute(
        "SELECT page_num, node_id FROM chunks WHERE chunk_id = ?",
        [chunk_id]
    ).fetchone()
    if not row:
        raise HTTPException(404, "Chunk not found")
    return {"page_num": row[0], "node_id": row[1]}


@app.get("/paper-viewer")
async def paper_viewer():
    """Serve the PDF viewer page."""
    return FileResponse(
        str(Path(__file__).parent / "static" / "paper_viewer.html")
    )
# ── Static files ──────────────────────────────────────────────────────────────

app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)

@app.get("/")
async def root():
    return FileResponse(
        str(Path(__file__).parent / "static" / "index.html")
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "frontend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )