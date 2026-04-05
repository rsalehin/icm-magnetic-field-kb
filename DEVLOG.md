# ICM Magnetic Field Knowledge Base — DEVLOG

**Project:** icm-magnetic-field-kb  

---

## Architecture Overview

```
papers/ (251 PDFs)
    ↓
Stage A — PDF extraction + chunking (pdfplumber, text_cleaner)
Stage B — ADS API enrichment (bibcode, citation count, authors)
Stage C — SPECTER2 GPU embedding (768-dim, 58,077 vectors)
Stage D — DuckDB + FAISS + NetworkX storage

Query Engine:
Query → Planner → Retriever (SPECTER2+BM25+BGE-M3+RRF+Graph)
      → Reranker (BGE-reranker-v2-m3)
      → Assembler (dedup, conflict detection, abstention)
      → Generator (DeepSeek V3.2 API / Qwen3:14b fallback)
      → Grounded answer with citations

Frontend:
FastAPI server → index.html (Chat + Graph + Stats tabs)
Chrome Extension → auto-inject prompts into Claude/ChatGPT/Gemini
```

---

## Step 1 — PDF Ingestion Pipeline (Stage A)

**Files:** `src/extraction/pdf_extractor.py`, `src/extraction/text_cleaner.py`

- PDF → section-aware chunks using pdfplumber
- Content flags: `has_equation`, `has_table`, `has_numerical_result`, `has_figure_ref`
- Context window: `prev_chunk_id`, `next_chunk_id`, `section_summary`
- Noise filtering: chunks < 30 tokens marked `is_noise = TRUE`
- 251 papers → 58,077 non-noise chunks

---

## Step 2 — ADS Enrichment (Stage B)

**Files:** `src/enrichment/ads_enricher.py`

- NASA ADS API: bibcode, DOI, citation count, journal, authors, keywords
- Token in `.env` as `ADS_TOKEN`
- Idempotent: skips already-enriched papers

---

## Step 3 — SPECTER2 Embeddings (Stage C)

**Files:** `src/storage/faiss_index.py`, `src/pipeline.py`

- Model: `allenai/specter2_base` (768-dim)
- GPU encoding, batch size 64
- 58,077 vectors in `data/faiss/chunks.index`
- ID map: `data/faiss/chunks_id_map.json`

---

## Step 4 — DuckDB + NetworkX Storage (Stage D)

**Files:** `src/storage/db.py`, `src/storage/graph.py`

**DuckDB tables:**
- `papers` — bibliographic metadata
- `authors` — normalised author list
- `chunks` — 58,077 chunk records with content flags
- `ingestion_state` — pipeline progress per paper
- `paper_extractions` — Layer 3 structured extraction (JSON)

**NetworkX graph:** `data/graph/citation_graph.graphml`
Populated in Step 11.

---

## Step 5 — Ingestion Orchestrator

**Files:** `src/pipeline.py`, `scripts/ingest_all.py`

- Orchestrates Stages A→B→C→D
- Idempotent — resumable on crash
- 251/251 papers, 16.8 min, 0 failures

---

## Step 10a — BM25 Lexical Index

**Files:** `src/query/bm25_index.py`, `scripts/test_bm25.py`

- Scientific tokeniser preserving Greek letters, subscripts, telescope acronyms
- Corpus: 58,077 chunks + 251 abstracts = 58,328 documents
- Section-weighted: title 3x, section 2x
- Cache: `data/bm25/bm25_index.pkl`
- node_id filter enables paper-pool restriction

---

## Step 10b — BGE-M3 Chunk Embeddings

**Files:** `src/query/bge_embedder.py`, `scripts/build_bge_index.py`

- Model: `BAAI/bge-m3` (1024-dim), fp16, GPU
- Asymmetric retrieval: query prefix applied on query side only
- 58,077 vectors in `data/faiss/bge_chunks.index`
- Raw matrix: `data/faiss/bge_chunks_matrix.npy` (used for dedup)
- Resumable build

**Issue fixed:** FlagEmbedding 1.3.5 incompatible with transformers 5.x.
Fixed by upgrading FlagEmbedding from source.

---

## Step 10c — Query Planner

**Files:** `src/query/planner.py`, `scripts/test_planner.py`

**Typed contract output (PlannerOutput):**
```json
{
  "intent": "comparison",
  "entities": {"methods": ["GRF", "BxC"], "instruments": ["VLA"]},
  "explicit_filters": {"year_min": 2011, "instrument": ["VLA"]},
  "soft_boosts": {"sections": ["results", "discussion"]},
  "budgets": {"paper_k": 80, "evidence_k": 12, "min_papers": 3},
  "synthesis_mode": "thinking"
}
```

- 4 intents: fact / comparison / synthesis / discovery
- Rule-based — microsecond latency, fully auditable
- Explicit vs inferred constraint separation
- 11/11 test accuracy after 3 bug fixes

**Bugs fixed:**
- Year regex captured group not full year → non-capturing group
- "MeerKAT papers after 2020" wrong intent → added discovery pattern
- "MNRAS papers about RM" wrong intent → journal+method bias correction

---

## Step 10d — Hybrid Retriever

**Files:** `src/query/retriever.py`, `scripts/test_retriever.py`

**Scoring formula:**
```
retrieval_score = rrf_score
                + 0.15 × section_prior
                + 0.10 × metadata_boost

rrf_score = Σ 1/(60 + rank_i)
```

**Bugs fixed:**
- DuckDB `ANY(?)` → `IN (?,?,?)` for Windows compatibility
- FAISS search n_retrieve = paper_k×20 to get enough unique papers
- Graph expansion referenced `rows_db` out of scope → fixed to `all_corpus`

---

## Step 10e — BGE Reranker

**Files:** `src/query/reranker.py`, `scripts/test_reranker.py`

- Model: `BAAI/bge-reranker-v2-m3` via `sentence_transformers.CrossEncoder`
- GPU, fp16, diversity control after reranking
- Key result: wrong rank 1 demoted to rank 6, correct chunk promoted to rank 1

**Issue fixed:** FlagReranker incompatible with transformers 5.x →
switched to CrossEncoder.

---

## Step 10f — Evidence Assembler

**Files:** `src/query/assembler.py`, `scripts/test_assembler.py`

- Cosine deduplication via BGE matrix (threshold=0.92)
- Rule-based conflict detection (positive/negative/uncertainty cues)
- Abstention: <3 chunks, <min_papers, max_score<0.30
- Neighbour expansion for comparison/synthesis (±1 chunk)

**Bug fixed:** Abstract chunks `arxiv:....__abstract` causing DuckDB lookup
failure → stripped suffix before lookup.

---

## Step 10g — Evaluation Set

**Files:** `scripts/generate_eval_set.py`, `scripts/evaluate.py`,
`data/eval_set_final.json`

- 30 questions: 25 factual (stratified 1995–2026) + 5 abstention
- Auto-generated by Qwen3 from corpus chunks
- Results: **77% abstention, 60% paper recall, 64% contains, 0.839 avg rerank**

---

## Step 10h — Generator

**Files:** `src/query/generator.py`, `scripts/test_generator.py`

**Routing:**
- `fact`, `discovery` → `deepseek-chat` (V3.2 non-thinking, fast)
- `comparison`, `synthesis` → `deepseek-reasoner` (V3.2 thinking mode)
- Fallback → Qwen3:14b local on any DeepSeek failure

**Features:**
- `_fix_math_notation()` — normalises broken LaTeX (B 0 → $B_0$, etc.)
- 2-retry with exponential backoff on DeepSeek 503
- `extract_with_deepseek()` for structured extraction tasks

---

## Step 11 — Citation Graph

**Files:** `src/storage/citation_graph.py`, `scripts/build_citation_edges.py`

- ADS references endpoint, 0.3s rate limiting
- CITES edges: 19,919 | Stub nodes: 7,924 | CO_CITED edges: 10,604
- Total: 8,232 nodes, 30,523 edges
- Graph expansion in retriever: 1-hop typed, strong edges only

---

## Step 12 — Structured Extraction (Layer 3)

**Files:** `src/extraction/structured_extractor.py`,
`scripts/run_structured_extraction.py`

Per paper: methods, key_quantities, scientific_claims, physical_domain,
instruments, clusters — stored as JSON in `paper_extractions` table.

---

## Frontend — Chat UI

**Files:** `frontend/app.py`, `frontend/static/index.html`,
`frontend/static/script.js`, `frontend/static/style.css`

**Three tabs:**
- **Chat** — conversation history, expandable pipeline trace, math+markdown rendering
- **Graph** — D3 citation network, query focus mode, year-lane corpus view
- **Stats** — corpus overview + per-query evidence analysis

**Features:**
- KaTeX math rendering (inline and display)
- Marked.js markdown (bold, bullets, tables, code)
- Citation tooltips on hover (title, year, journal, section)
- Click [E1] → PDF opens at correct page with PyMuPDF highlight
- Highlight expands ±2 surrounding text blocks for context
- Conversation history in `data/conversations.duckdb`
- Auto-reload on code change
- Elapsed timer during generation
- Collapsible sidebar
- Auto-focus citation graph on query results

**Graph modes:**
- Query mode: force simulation, settles and pins nodes, evidence nodes amber+ranked
- Corpus mode: static year-lane layout, instant render, no simulation

---

## Chrome Extension — LLM Export

**Files:** `chrome-extension/manifest.json`, `chrome-extension/content.js`,
`chrome-extension/background.js`

- Retrieval-only endpoint `/api/retrieve-only` — full pipeline, skip generation
- Prompt stored server-side (60s expiry), fetched by extension via ID
- Auto-injects + submits prompt into Claude ✓, ChatGPT ✓, Gemini ✓
- Fallback: copy-to-clipboard always available
- UI toggle: Built-in (DeepSeek) ↔ Export (Claude/ChatGPT/Gemini)

---

## Key Files Reference

```
src/
  extraction/
    pdf_extractor.py          Stage A
    text_cleaner.py           Stage A
    structured_extractor.py   Step 12
  enrichment/
    ads_enricher.py           Stage B
  storage/
    db.py                     DuckDB schema
    faiss_index.py            SPECTER2 FAISS
    graph.py                  NetworkX I/O
    citation_graph.py         Step 11
  query/
    planner.py                Step 10c
    bm25_index.py             Step 10a
    bge_embedder.py           Step 10b
    retriever.py              Step 10d
    reranker.py               Step 10e
    assembler.py              Step 10f
    generator.py              Step 10h
  pipeline.py                 Ingestion orchestrator

frontend/
  app.py                      FastAPI server
  conversations_db.py         Chat history DuckDB
  static/
    index.html                Main UI
    script.js                 All JS logic
    style.css                 Observatory Terminal theme
    paper_viewer.html         PDF viewer with highlighting

chrome-extension/
  manifest.json
  content.js
  background.js

data/
  knowledge_base.duckdb
  conversations.duckdb
  faiss/chunks.index           SPECTER2 (58,077 × 768)
  faiss/bge_chunks.index       BGE-M3 (58,077 × 1024)
  faiss/bge_chunks_matrix.npy
  bm25/bm25_index.pkl
  graph/citation_graph.graphml  8,232 nodes, 30,523 edges
  eval_set_final.json
  eval_results.json
```

---

## Environment

```
# .env
ADS_TOKEN=...
DEEPSEEK_API_KEY=...
HF_TOKEN=...

# Start
cd Knowledge_Base
.venv\Scripts\Activate.ps1
python -m uvicorn frontend.app:app --host 0.0.0.0 --port 8000 \
  --reload --reload-dir src --reload-dir frontend \
  --reload-include "*.html" --reload-include "*.css" --reload-include "*.js"

# Browser
http://localhost:8000

# Extension
chrome://extensions → Developer mode → Load unpacked → chrome-extension/
```

---

## Corpus Statistics

| Metric | Value |
|--------|-------|
| Papers | 251 |
| Chunks (non-noise) | 58,077 |
| BM25 documents | 58,328 |
| Graph nodes | 8,232 |
| CITES edges | 19,919 |
| CO_CITED edges | 10,604 |
| Era coverage | 1995–2026 |
| Top authors | Govoni (11), Vazza (10), Bonafede (7) |
| Top journals | MNRAS (68), A&A (64), ApJ (44) |

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Abstention accuracy | 77% (23/30) |
| Paper recall | 60% |
| Contains score | 64% |
| Avg rerank score | 0.839 |

---

## Known Limitations

- FAISS runs on CPU only (Blackwell sm_120 incompatible with faiss-gpu)
- Abstention questions fail when too close to corpus content
- Structured extraction quantity values imperfect when text starts with figure captions
- PDF highlighting finds adjacent blocks when chunk text differs from raw PDF
- Chrome extension selectors may break if LLM UIs update their DOM

---

## Future Tasks

### Task 1 — Figure Extraction (Highest Research Value)

Extract figures from PDFs, describe with Claude vision, make searchable.

**Architecture:**
- `src/extraction/figure_extractor.py` — PyMuPDF extracts images + captions
- New DuckDB table: `figures` (figure_id, node_id, figure_num, caption, page_num, image_path)
- Caption embedding → separate FAISS index for figure retrieval
- Claude vision API → rich text description of each figure (axis labels, values, trends)
- FastAPI: `/api/figures/{node_id}`, `/api/figure/{figure_id}`
- UI: figure thumbnails in evidence panel, lightbox on click
- Link figures to chunks via `figure_refs` already stored in chunks table

**Value:** RM maps, magnetic field profiles, depolarization curves become
viewable alongside evidence. Figure descriptions become searchable text.

**Estimated effort:** 2–3 days.

---

### Task 2 — Domain-Agnostic Refactor (Highest Career Value)

Turn this into a reusable open-source tool for any field of knowledge.

**Single config file:**
```yaml
# config/domain.yaml
domain:
  name: "Intracluster Magnetic Fields"
  short_name: "ICM·KB"

entities:
  sources: ["coma cluster", "abell 119", ...]
  methods:
    "grf": "GRF"
    "mhd": "MHD"
  instruments:
    "vla": "VLA"
    "lofar": "LOFAR"

generator:
  system_prompt: |
    You are a scientific assistant for astrophysics research...

enrichment:
  source: "ads"   # ads | pubmed | crossref | arxiv | none
  api_key_env: "ADS_TOKEN"

ui:
  welcome_title: "Intracluster Magnetic Fields"
  suggestions:
    - "What spectral index did Murgia 2004 find for A119?"
```

**What this enables:**
- Computational fluid dynamics papers
- Medical oncology literature
- Legal case law
- Economics journals
- Philosophy papers

**New files needed:**
- `src/config.py` — singleton YAML loader
- `scripts/setup.py` — one-command setup
- `README.md` — "Deploy for your domain in 5 steps"
- CrossRef API integration (enrichment for any DOI)

**Estimated effort:** 2–3 days.

---

### Task 3 — Concept Dependency Graph (Highest Pedagogical Value)

Interactive knowledge prerequisite graph for ICM concepts.

**Phase 1 — Extract from existing corpus:**
- DeepSeek reasoner extracts concept dependency pairs from chunks
- Example edges: σ_RM → Faraday_rotation → electromagnetic_polarization → Maxwell
- ~80–120 concept nodes for ICM domain
- Store as `data/graph/concept_graph.graphml`

**Phase 2 — Textbook ingestion:**
- 5–8 key textbooks: Rybicki & Lightman, Longair, Jackson, Pacholczyk
- Same chunking + extraction pipeline
- Extends graph to foundational physics

**UI — new Concepts tab:**
- D3 force graph of concept nodes
- Node states: green (understood) / yellow (partial) / red (don't understand)
- States persisted in `conversations.duckdb` → new `concept_states` table
- "Explain my gaps" button → topological sort of red nodes → LLM explains each
  in prerequisite order, grounded in corpus chunks

**Critical human step:** After extraction, manually verify top 30–40 dependency
edges before building UI. Domain expertise required — only Abir can validate.

**Estimated effort:** 1 week.

---

### Task 4 — Docker Packaging (Broadest Deployment)

One-command deployment on any machine.

**Config A — Full build (GPU required):**
```yaml
services:
  app:    # FastAPI + full pipeline
  ollama: # Qwen3:14b local fallback
```

**Config B — Query only (no GPU, DeepSeek API):**
```yaml
services:
  app:    # FastAPI + pre-built indices only
          # No Ollama, no GPU needed
```

**What to ship (~3GB without PDFs):**
- `data/knowledge_base.duckdb`
- `data/faiss/*.index` + id maps
- `data/bm25/bm25_index.pkl`
- `data/graph/citation_graph.graphml`

**Desktop launcher:**
```batch
# launch.bat — double-click to start
python launch.py   # starts server + opens browser automatically
```

**Prerequisite:** Complete domain-agnostic refactor (Task 2) first.

**Estimated effort:** 1–2 days after Task 2.

---

*Last updated: 2026-04-05*
