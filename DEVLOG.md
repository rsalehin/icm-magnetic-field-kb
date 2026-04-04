# ICM Magnetic Field — Knowledge Base Build Log

**Project:** Local hybrid knowledge graph over the intracluster medium (ICM)
magnetic field literature, supporting structured, semantic, and graph-hop queries.

**Repository:** `C:\Users\rsalehin\OneDrive\Desktop\Projects\TLS\Papers\Knowledge_Base`

**Author:** Abir (Rafiqus Salehin)
**Started:** 2026-04-03

---

## Architecture Overview

A three-backend local system, fully offline:

```
PDFs
  ↓
PyMuPDF / pdfplumber     → text extraction + chunking
  ↓
SPECTER2 on GPU          → scientific embeddings
  ↓
  ├── FAISS (CPU)        → dense vector search (semantic queries)
  ├── DuckDB             → structured SQL (metadata + extracted quantities)
  └── NetworkX           → citation graph (relational + graph-hop queries)
```

## Motivation & Design Philosophy

**Date added:** 2026-04-04

### Why build this instead of using existing tools?

This section documents the reasoning behind building a custom knowledge
base rather than using off-the-shelf solutions. Written at Step 6,
after the core architecture was established.

### Existing tools considered

| Tool | What it does well | Why it was insufficient |
|---|---|---|
| NotebookLM (Google) | PDF Q&A, cited answers, clean UI | Cloud-only, 50 paper limit, no graph, no SQL, no domain tuning |
| Elicit / Consensus | Literature Q&A | Public corpus only, cannot add own papers, no graph |
| ResearchRabbit | Citation graph visualisation | No text retrieval, no Q&A, visual only |
| Zotero + plugins | Reference management | No semantic search, no graph queries |
| Claude / ChatGPT | Q&A over uploaded PDFs | Context window limits, no persistence, no graph traversal |
| LlamaIndex / LangChain | Framework-level RAG | Shallow graph, no domain chunking, no structured extraction |

### What this system does that no existing tool does

**Graph-hop queries**
No existing tool supports queries like:
- *"Trace the methodological lineage from Tribble 1991 through
  Murgia 2004 to present day"*
- *"What did papers that cited Murgia 2004 conclude about the
  magnetic field power spectrum slope?"*
- *"Which papers use both GRF simulation AND compare against
  the analytical single-scale model?"*

These require traversing the citation graph while simultaneously
retrieving grounded text — a qualitatively different query type
that vector search alone cannot support.

**Structured domain extraction (Layer 3)**
Storing extracted physical quantities as SQL — σ_RM values,
spectral indices n, central field strengths B₀, cluster names,
field models used — enables queries like:
- *"Which papers report B₀ > 5 µG in non-cooling-flow clusters?"*
- *"Show all papers that study A119 using RM structure functions"*

This is a database query, not a fuzzy semantic search. The
precision difference is significant for scientific work.

**Domain-specific embeddings**
SPECTER2 was trained on 146 million scientific citation pairs.
It understands that "σ_RM", "rotation measure dispersion", and
"RM scatter" refer to the same concept. General-purpose models
used by NotebookLM and ChatGPT treat these as loosely related.
For astrophysics text with heavy notation, the retrieval accuracy
difference is substantial.

**Full ownership and privacy**
The entire system runs locally on the research workstation.
No paper content, no query history, no extracted knowledge
leaves the machine. This matters for unpublished results and
pre-submission work.

**Incremental and persistent**
The system grows with the research. Drop a new PDF in the
`papers/` folder, re-run the pipeline — only the new paper
is processed. All prior ingestion is preserved. No existing
cloud tool offers this with full control over the pipeline.

### When existing tools are better

This system is not always the right choice. For simple queries
over a small number of papers, NotebookLM or Claude with an
uploaded PDF is faster and easier. This system becomes strictly
better when:

- Queries span more than 10 papers simultaneously
- Relational queries are needed (which papers, which clusters,
  which methods, which quantities)
- Citation graph traversal is required
- Privacy and offline operation matter
- The corpus will grow incrementally over time
- GNN analysis of the citation network is a future goal

All five conditions apply to this project.

### Research context

This knowledge base is built specifically for research on
turbulent magnetic fields in galaxy clusters at
Karl-Schwarzschild-Observatorium Tautenburg. The corpus covers
30 years of ICM magnetic field literature (1995–2026, 251 papers)
spanning:
- Faraday rotation measure (RM) observations
- GRF and BxC/Biot-Savart magnetic field models
- MHD cosmological simulations
- Depolarization studies
- Radio halo morphology

The structured extraction vocabulary (GRF, BxC, σ_RM, β-model,
Λ_min, Λ_max, spectral index n) is domain-specific and would
be missed entirely by general-purpose tools.

### Long-term vision

Beyond literature search, the graph structure and node features
(Layer 4) are designed to support:
- GNN-based paper recommendation
- Automated detection of methodological lineages
- Community detection within the citation graph
- Identification of contradicting claims across papers
- Semi-automated literature review generation

The schema was designed with these use cases in mind from the
start — not retrofitted later.

### Why this stack
- **FAISS over Qdrant/Weaviate:** corpus is bounded (~200–500 papers, ~20k chunks);
  no Docker daemon needed; CPU FAISS is fast enough at this scale.
- **DuckDB over SQLite:** columnar, faster analytical queries, native Python,
  no server process.
- **NetworkX over Neo4j:** graph fits in RAM (500 nodes); no JVM overhead;
  full Python API for custom traversal logic.
- **SPECTER2 over general embeddings:** trained on 146M scientific citation pairs;
  understands domain vocabulary (σ_RM, beta-model, Faraday rotation) out of the box.

### Paper node schema
Four-layer JSON structure per paper:
- **Layer 1:** Bibliographic metadata (from NASA ADS)
- **Layer 2:** Paragraph-level chunks with section context + embeddings
- **Layer 3:** LLM-extracted structured knowledge (methods, quantities,
  claims, datasets, physical domain vocabulary)
- **Layer 4:** Graph topology features (edges, node features for GNN)

Schema file: `paper_node_schema.json`

---

## Step 1 — Environment Diagnostic
**Date:** 2026-04-03
**Script:** `step1_env_check.py`

### What we did
Ran a diagnostic script to check Python version, GPU, RAM, and which
required packages were already installed before touching anything.

### Output
```
Python : 3.12.10 [MSC v.1943 64 bit (AMD64)]
OS     : Windows-11-10.0.26200-SP0

GPU    : NVIDIA GeForce RTX 5070 Ti  |  VRAM: 17.1 GB
CUDA   : 13.0 (dev PyTorch build 2.12.0.dev20260304+cu130)

RAM Total     : 67.8 GB
RAM Available : 53.1 GB

✓ networkx   3.6.1
✓ requests   2.33.0
✓ tqdm       4.67.1
✗ faiss, duckdb, sentence_transformers, transformers, pdfplumber, pymupdf
```

### Key decisions made
| Decision | Reason |
|---|---|
| Use `faiss-cpu` not `faiss-gpu` | Blackwell (sm_120) not yet supported in faiss-gpu PyPI builds |
| Use PyTorch nightly cu130 | Only build with sm_120 support for RTX 5070 Ti |
| SPECTER2 as embedding model | Scientific paper training corpus; understands physics vocabulary |
| CPU FAISS sufficient | ~500 papers → ~20k chunks; fits easily in 67 GB RAM |

---

## Step 2 — Virtual Environment + Stack Installation
**Date:** 2026-04-03

### What we did
Created an isolated virtual environment to protect the global Python
installation, then installed all required packages inside it.

### Why a venv
Global Python was at `C:\Users\rsalehin\AppData\Local\Programs\Python\Python312`
(no environment active). Installing globally risked overwriting the
dev PyTorch cu130 build. Venv keeps everything isolated — delete the
folder to start over with zero damage.

### Commands run
```powershell
# PowerShell execution policy (one-time fix)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create and activate venv
cd C:\Users\rsalehin\OneDrive\Desktop\Projects\TLS\Papers\Knowledge_Base
python -m venv .venv
.venv\Scripts\Activate.ps1

# Core data layer
pip install duckdb faiss-cpu

# ML stack (--no-deps protects existing PyTorch install)
pip install sentence-transformers --no-deps
pip install transformers tokenizers huggingface-hub safetensors accelerate numpy scipy scikit-learn

# PDF parsers
pip install pymupdf pdfplumber

# PyTorch — cu124 stable first attempt (failed: sm_120 warning)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# ↑ Gave UserWarning: sm_120 not compatible. Uninstalled and switched.

# PyTorch — nightly cu130 (correct build for RTX 5070 Ti)
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

### Issue encountered
`cu124` stable PyTorch installed cleanly but emitted:
```
UserWarning: NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120
is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60
sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
```
**Fix:** Switched to nightly `cu130` build which includes sm_120 support.

### Verified stack (`step2_verify.py` output)
```
✓ duckdb                    1.5.1
✓ faiss                     1.13.2
✓ networkx                  3.6.1
✓ sentence_transformers     5.3.0
✓ transformers              5.5.0
✓ fitz (pymupdf)            1.27.2.2
✓ pdfplumber                0.11.9

torch : 2.12.0.dev20260403+cu130  |  CUDA: True
GPU   : NVIDIA GeForce RTX 5070 Ti
```

---

## Step 3 — Embedding Model Verification
**Date:** 2026-04-03
**Script:** `step3_embedding_check.py`

VRAM: 0.47 GB | Encoding: 6.5 it/s | Shape: (3, 768)
Warnings: symlinks (harmless), position_ids unexpected (harmless)

### What we did
Verified that sentence-transformers correctly dispatches SPECTER2
to the RTX 5070 Ti GPU, not CPU. First run downloads ~500 MB of
model weights to `C:\Users\rsalehin\.cache\huggingface\hub`.

### Output
```
[paste output here]
```

### Result
- [ ] GPU confirmed / CPU fallback (circle one after running)

---

## Step 4 — PDF Text Extraction (Stage A)
**Date:** 2026-04-04
**Branch:** feature/step-04-pdf-extraction
**Files added:**
- `src/extraction/pdf_extractor.py`
- `src/extraction/text_cleaner.py`
- `src/extraction/__init__.py`
- `src/__init__.py`
- `scripts/test_extractor.py`
- `scripts/test_extractor2.py`
- `scripts/test_batch_small.py`
- `scripts/diagnose_pdf.py`

### Goal
Build Stage A of the ingestion pipeline: take a raw PDF file and
produce a structured Python dict matching Layer 1 (partial) + Layer 2
of the paper node schema. Purely local — no network calls.

### What was built
**`pdf_extractor.py`** — main extractor:
- Parses arxiv_id and year from filename (handles both old-style
  `astro-ph_0406225` and new-style `0905.3552` formats)
- Opens PDF with PyMuPDF, iterates page by page
- Calls text_cleaner per page, then splits into paragraph chunks
- Detects section boundaries line by line
- Attaches content flags per chunk (has_equation, has_table,
  has_figure_ref, has_numerical_result)
- Runs a post-pass to fix next_chunk_id and attach section summaries
- Returns a fully structured dict ready for storage

**`text_cleaner.py`** — page-level boilerplate removal + paragraph splitting:
- Strips arXiv stamps, journal headers, running headers, affiliation
  lines, email lines, manuscript lines, standalone numbers
- Joins soft-wrapped lines back into full sentences (arXiv PDFs wrap
  at ~80 chars with no blank line)
- Handles hyphenated word breaks (removes hyphen, joins without space)
- Splits into paragraphs: blank-line boundaries first, then sentence
  boundary heuristic for large blocks (>400 chars)
- min_chars=150 filter removes noise fragments

### Problems encountered and fixes

**Problem 1 — Only 19 chunks for a 19-page paper (one chunk per page)**
Root cause: arXiv PDFs use single `\n` between paragraphs, not `\n\n`.
The original splitter used `re.split(r"\n\s*\n")` which found no splits.
Fix: added `join_soft_wrapped_lines()` to first reconstruct full
sentences from wrapped lines, then applied a sentence-boundary
heuristic splitter for large blocks.
Result: 19 chunks → 236 chunks for Murgia 2004.

**Problem 2 — Running page headers not stripped**
Root cause: Pattern `^\d{1,3}\n[A-Z]...` expected a page number
before the header, but A&A papers format it as `\nAuthor et al.: Title\n`
without a preceding digit on the same match.
Fix: broadened RUNNING_HEADER pattern to `^[A-Z][^\n]{5,80}et al\.`.
Result: Page 3 went from 1 paragraph to 20 paragraphs.

**Problem 3 — Equation fragments detected as section headings**
Symptom: `'1.4 GHz = (0.393 ± 0.002) ·'` appearing in sections list.
Root cause: Section pattern `^\d+\.?\d*` was too loose — matched any
line starting with a number.
Fix: added hard rejection rules to `detect_section()` — lines
containing `=`, `±`, `·`, `×` or decimal numbers rejected immediately.
Also tightened SECTION_PATTERNS to require `[a-zA-Z\s]{3,50}` after
the section number.
Result: False sections eliminated across all tested papers.

**Problem 4 — Affiliation lines leaking into first chunks**
Symptom: `"Poggio dei Pini, I-09012 Capoterra..."` appearing as chunk text.
Root cause: AFFILIATION pattern only matched lines ending in country
names — missed institution names like INAF, NRAO, MPIfR.
Fix: extended AFFILIATION pattern to also match Observatory, Institute,
University, Universit, Dipartimento, INAF, CNRS, MPIfR, NRAO, ESO.
Also added EMAIL_LINE and CORRESPONDENCE_LINE patterns.

**Problem 5 — min_chars too low (80) letting noise through**
Symptom: 15–37 noise chunks under 30 tokens per paper.
Fix: raised min_chars from 80 to 150.
Residual noise (5–20 chunks per paper) is equation fragments that
survive cleaning — these are flagged at query time rather than
discarded, preserving equation content.

### Final test results (5-paper batch)

| Paper | Pages | Chunks | Avg tokens | Noise |
|---|---|---|---|---|
| 1999_Feretti | 30 | 183 | 47 | 10 |
| 2004_Murgia | 19 | 236 | 47 | 15 |
| 2012_Vacca | 17 | 193 | 47 | 11 |
| 2019_Loi | 10 | 116 | 50 | 5 |
| 2024_Vacca | 12 | 177 | 46 | 20 |

All 5 papers: zero failures, consistent ~47 tokens/chunk average.

### Key decisions made

| Decision | Reason |
|---|---|
| Stage A purely local (no ADS calls) | Isolates failure modes — network issues can't stall PDF parsing |
| Paragraph-level chunks (~47 tokens) | Balance between retrieval precision and context; sentence-level too small, page-level too large |
| min_chars=150 not higher | Preserves short but meaningful equation-context chunks |
| Noise chunks kept, not deleted | Equation fragments have retrieval value; filtered at query time |
| Sentence-boundary heuristic only on blocks >400 chars | Avoids over-splitting naturally short paragraphs |
| Section detection rejects lines with math symbols | Equation-rich lines are structurally identical to headings at regex level |

### What Stage A does NOT do
- Does not call ADS API (Stage B)
- Does not generate embeddings (Stage C)
- Does not extract structured knowledge from Layer 3 (Stage D)
- Does not write to DuckDB, FAISS, or NetworkX (Stage D)

### Known limitations
- Section titles that span two lines are truncated at the first line
  e.g. `"2. Faraday rotation in clusters of galaxies and the"` — the
  second line `"FARADAY tool"` is lost. Acceptable for now.
- Residual noise from inline equations rendered as isolated text
  blocks by PyMuPDF — these survive all cleaning passes.
- Papers with two-column layouts may have occasional merge artifacts
  where columns are concatenated mid-sentence.

---

## Step 5 — ADS Metadata Enrichment (Stage B)
**Date:** 2026-04-04
**Branch:** feature/step-05-ads-enrichment
**Files added:**
- `src/enrichment/__init__.py`
- `src/enrichment/ads_enricher.py`
- `scripts/test_ads_enricher.py`

### Goal
Fill in Layer 1 bibliographic fields (title, authors, journal, DOI,
abstract, citation count, bibcode) by querying the NASA ADS API
using the arxiv_id derived from the filename in Stage A.

### What was built
**`ads_enricher.py`**:
- Loads ADS token from `.env` via python-dotenv
- `enrich_from_arxiv_id()` — queries ADS search endpoint,
  handles both old-style (astro-ph/0406225) and new-style
  (0905.3552) arxiv IDs
- Fallback: if primary query fails, retries with identifier field
- Extracts DOI from identifier list (starts with "10.")
- `enrich_paper()` — takes Stage A dict, merges ADS data into
  Layer 1, updates meta.stage_b status, respects rate limit
  via 0.3s sleep between calls

### Design decisions
| Decision | Reason |
|---|---|
| ADS token in `.env` not source code | Never expose credentials in git |
| 0.3s sleep between calls | ADS rate limit ~5000/day unauthenticated, polite crawling |
| Merge strategy: existing fields preserved | pdf_path and total_pages from Stage A must not be overwritten |
| stage_b = "failed" not exception | Pipeline continues on ADS miss; paper still usable with Stage A data |

### Test results
| Paper | Bibcode | Citations | stage_b |
|---|---|---|---|
| Murgia 2004 | 2004A&A...424..429M | 255 | done |
| Bonafede 2009 | 2009A&A...503..707B | 129 | done |
| Loi 2019 | 2019MNRAS.485.5285L | 16 | done |

All 3 resolved correctly. Zero failures.

### Known limitations
- ADS occasionally returns arXiv DOI (10.48550/arXiv.XXX) instead
  of journal DOI — happens when journal has not yet registered DOI
  with ADS. Acceptable for now.
- No retry logic on network timeout — will add in pipeline manager.

---

## Step 6 — DuckDB Schema Initialisation
**Date:** 2026-04-04
**Branch:** feature/step-06-duckdb-schema
**Files added:**
- `src/storage/__init__.py`
- `src/storage/db.py`
- `scripts/test_duckdb.py`
- `scripts/test_duckdb_idempotent.py`

### Goal
Set up the persistent structured storage layer. DuckDB holds
Layer 1 bibliographic metadata, normalised authors, Layer 2
chunks with all content flags, and ingestion state tracking
per paper across all four pipeline stages.

### What was built
**`db.py`** — four functions:
- `get_connection()` — opens/creates DuckDB file, creates
  parent directory if needed
- `init_schema()` — creates four tables if not exist:
  `papers`, `authors`, `chunks`, `ingestion_state`
- `insert_paper()` — takes Stage A+B dict, inserts into all
  four tables atomically, idempotent (skips if node_id exists)
- `get_ingestion_state()` — returns pipeline progress for all
  papers as list of dicts

### Schema design decisions
| Decision | Reason |
|---|---|
| Four separate tables | Normalisation — authors queried independently of chunks |
| JSON arrays as VARCHAR | keywords, sections, figure_refs don't need SQL querying; avoids complexity of array types |
| is_noise flag on chunks | Noise chunks preserved for equation content but excluded from embedding |
| ingestion_state separate table | Single source of truth for pipeline progress — survives paper re-ingestion |
| INSERT OR REPLACE on ingestion_state | State always reflects latest run |
| Idempotent insert | Safe to re-run pipeline on same paper — no duplicates |

### Test results
| Check | Result |
|---|---|
| Schema init | ✓ |
| Paper insert (Murgia 2004) | ✓ 236 chunks, 5 authors |
| Noise flagging | ✓ chunks < 30 tokens correctly marked |
| Second insert skips | ✓ "Already in DB, skipping" |
| Row counts after double insert | ✓ 1 paper, 236 chunks |

### Known limitations
- Layer 3 structured extraction not yet stored — will add
  JSON column to papers table in Step 10
- No index on chunks.node_id yet — will add before full
  corpus ingestion in Step 9
- authors table uses position integer as part of PK —
  sufficient for read queries but not for fuzzy author search

---

## Step 7 — FAISS Index Initialisation
**Date:** 2026-04-04
**Branch:** feature/step-07-faiss-index
**Files added:**
- `src/storage/faiss_index.py`
- `scripts/test_faiss.py`

### Goal
Set up the vector storage layer. FAISS IndexFlatIP holds
SPECTER2 embeddings for all non-noise chunks. Supports
incremental addition of new vectors and exact cosine
similarity search.

### What was built
**`faiss_index.py`** — six functions:
- `get_or_create_index()` — loads existing index from disk
  or creates new IndexFlatIP (dim=768)
- `load_id_map()` — loads chunk_id → faiss_row mapping from
  JSON file
- `save_id_map()` — persists id_map to disk
- `save_index()` — writes FAISS index to disk
- `add_embeddings()` — adds batch of vectors, normalises to
  unit length, skips already-indexed chunks (idempotent)
- `search()` — searches index for top-k nearest neighbours,
  returns list of {chunk_id, score, faiss_row} dicts
- `get_index_stats()` — summary stats for current index

### Design decisions
| Decision | Reason |
|---|---|
| IndexFlatIP not IndexIVFFlat | Exact search — no approximation error; corpus (~20k chunks) well within exact search range |
| Cosine via inner product | Normalise vectors to unit length first, then inner product = cosine similarity |
| Separate id_map JSON | FAISS only stores row indices — need external chunk_id mapping for retrieval |
| CPU FAISS | Blackwell sm_120 unsupported in faiss-gpu; 20k vectors trivially fast on CPU |
| GPU used upstream | Embedding generation (Stage C) uses RTX 5070 Ti; FAISS only does lookup |
| Idempotent add | Skip already-indexed chunks — safe to re-run pipeline |

### Test results
| Check | Result |
|---|---|
| Index creation | ✓ dim=768 IndexFlatIP |
| Add 10 vectors | ✓ |
| Idempotency (skip duplicates) | ✓ |
| Add 5 more incremental | ✓ 15 total |
| Search top hit | ✓ c0000 score=1.0000 exact match |
| Persist and reload | ✓ 15 vectors restored from disk |

### Known limitations
- Vectors not yet populated with real SPECTER2 embeddings —
  tested with synthetic random vectors; real embeddings added
  in Step 9
- No FAISS GPU index — faiss-gpu does not support Blackwell
  sm_120; CPU index sufficient at current scale
- id_map loaded fully into RAM — fine for 20k chunks,
  revisit if corpus grows beyond 500k vectors
---

## Step 8 — NetworkX Citation Graph
**Date:** 2026-04-04
**Branch:** feature/step-08-networkx-graph
**Files added:**
- `src/storage/graph.py`
- `scripts/test_graph.py`

### Goal
Set up the graph storage layer. NetworkX DiGraph holds papers
as nodes and citation relationships as directed typed edges.
Persists to disk as GraphML between sessions.

### What was built
**`graph.py`** — six functions:
- `get_or_create_graph()` — loads GraphML from disk or creates
  new DiGraph
- `save_graph()` — persists to GraphML
- `add_paper_node()` — adds paper with Layer 1 attributes as
  node attributes; idempotent
- `add_citation_edges()` — adds directed edges with citation_role
  and context_chunk_id; creates stub nodes for cited papers not
  yet ingested
- `mark_seed()` — flags seed papers in graph
- `get_neighbors()` — returns in/out neighbors with edge
  attributes; supports direction='in', 'out', 'both'
- `get_graph_stats()` — summary stats including year range
  and seed count

### Design decisions
| Decision | Reason |
|---|---|
| DiGraph not Graph | Citations are directed — A cites B ≠ B cites A |
| GraphML format | Human-readable XML, NetworkX native, survives version changes |
| Stub nodes for uncrawled refs | Graph stays consistent even when target paper not yet ingested |
| citation_role on edges | Enables filtering by relationship type — foundational vs methodological vs incidental |
| context_chunk_id on edges | Traces exactly which chunk contains the citation |
| is_seed flag on nodes | Needed for graph traversal strategies that start from seeds |

### Test results
| Check | Result |
|---|---|
| Graph creation | ✓ |
| 2 full nodes + 1 stub | ✓ |
| Directed edges with roles | ✓ methodological + foundational |
| Neighbour traversal | ✓ |
| Persist and reload | ✓ 3 nodes, 2 edges |

### Known limitations
- Citation edges currently added manually with synthetic data —
  real citation extraction from ADS happens in Step 9
- GraphML loads entire graph into RAM — fine for 251 papers,
  revisit if corpus grows beyond ~5000 nodes
- No community detection or graph metrics computed yet —
  planned for Step 11
---

## Step 9 — Full Single-Paper Ingestion Pipeline
**Date:** 2026-04-04
**Branch:** feature/step-09-ingestion-pipeline
**Files added:**
- `src/pipeline.py`
- `scripts/test_pipeline_single.py`

### Goal
Wire all four stages into a single end-to-end pipeline function
that takes a PDF path and writes to all three backends atomically.
First milestone where the full system runs together.

### What was built
**`pipeline.py`** — four components:
- `get_model()` — loads SPECTER2 onto GPU once, reuses across
  papers via module-level singleton
- `embed_chunks()` — Stage C: generates embeddings for all
  non-noise chunks (token_count >= 30), batch size 64, GPU
- `write_to_storage()` — Stage D: writes to DuckDB, FAISS,
  NetworkX in sequence; updates faiss_index_id in DuckDB
  after FAISS write; marks stage_d = done in DB
- `ingest_paper()` — orchestrates A→B→C→D; checks DuckDB
  first and skips if already fully ingested

### Performance
- SPECTER2 load time : ~15s (cached after first run)
- Per-paper time     : ~3.1s (GPU embedding dominant)
- Full corpus (251)  : ~13 minutes estimated
- Noise skipped      : 15/236 chunks for Murgia 2004

### Bug found and fixed
stage_d was showing 'pending'

---
## Step 9b — Batch Ingestion Runner
**Date:** 2026-04-04
**Branch:** feature/step-09b-batch-ingestion
**Files added:**
- `scripts/ingest_all.py`

### Goal
Process all 251 PDFs through the full pipeline in one run.
Resumable, fault-tolerant, with progress bar and log file.

### What was built
**`ingest_all.py`** — batch runner with:
- `--limit N` flag for test runs on first N papers
- `--skip-ads` flag for offline testing
- tqdm progress bar per paper
- saves all three backends after every paper — no data
  loss on crash or interrupt
- timestamped log file at `data/ingestion_log.txt`
- full summary on completion

### Full corpus results
| Metric | Result |
|---|---|
| Papers processed | 251/251 |
| Papers failed | 0 |
| Total time | 16.8 minutes |
| Avg per paper | ~4s |
| FAISS vectors | 58,077 |
| Graph nodes | 251 |
| DuckDB rows | 251 papers, ~58k chunks |

### Test methodology
- Ran with `--limit 5` first — verified 5/5 clean
- Then ran full corpus — 251/251 clean
- Confirmed resumability: 5 pre-ingested papers
  correctly skipped on full run

### Known limitations
- Graph edges = 0 — citation edge extraction from
  ADS references endpoint planned for Step 11
- ADS called for every paper even on resume — will
  add local cache in future
- No parallel processing — sequential by design to
  respect ADS rate limits; GPU is the bottleneck
  anyway at ~4s/paper
## Step 10a — BM25 Lexical Index
**Date:** 2026-04-04
**Branch:** feature/step-10a-bm25-index
**Files added:**
- `src/query/__init__.py`
- `src/query/bm25_index.py`
- `scripts/test_bm25.py`

### Goal
Build sparse lexical retrieval index over all 58,077 chunks
plus 251 abstracts. Handles exact scientific term retrieval
that dense SPECTER2 embeddings miss (σ_RM, Burn law, LOFAR,
BxC, Briggs weighting).

### What was built
**`bm25_index.py`** — five components:
- `tokenise()` — scientific tokeniser preserving Greek letters,
  subscripts, telescope acronyms, method names
- `build_weighted_text()` — section_title repeated 2x +
  chunk_text + keywords for weighted field
- `build_corpus_from_db()` — pulls non-noise chunks from
  DuckDB, writes JSONL canonical store; also extracts
  abstracts as special documents with title repeated 3x
- `build_bm25_index()` — builds BM25Okapi from JSONL,
  saves pickle cache
- `bm25_search()` — searches with optional node_id filter
  to restrict to paper pool from Stage 1 retrieval

### Design decisions
| Decision | Reason |
|---|---|
| JSONL as canonical store | Reproducible, version-safe, DuckDB-independent |
| Pickle as cache artifact | Fast load at query time, regeneratable |
| Abstracts as separate boosted docs | Higher signal for discovery queries |
| Title 3x, section 2x repetition | Simple weighted field without schema changes |
| node_id filter in search | Enables chunk-phase restriction to paper pool |
| Scientific tokeniser | Preserves σ_RM, beta-model, LOFAR, MeerKAT exactly |

### Test results
| Query | Top result | Correct? |
|---|---|---|
| Burn law depolarization sigma_RM | arxiv:1002.0811 — depolarization section | ✓ |
| Gaussian random field power spectrum | arxiv:2507.22006 — turbulent magnetic sim | ✓ |
| Faraday rotation intracluster magnetic field | astro-ph/0505144 — RM diagnostics | ✓ |
| LOFAR observations galaxy cluster | arxiv:2201.12207 — LOFAR 144MHz | ✓ |
| BxC Biot-Savart convolution | arxiv:1711.03252 — deep learning paper | ✗ |

### Known limitation
BxC query returned off-topic results — "convolution" and
"field" match deep learning papers. This is expected BM25
behaviour on a broad corpus. Fixed by paper-pool restriction:
SPECTER2 paper filtering in Step 10d removes off-topic papers
before BM25 runs on the remaining pool.

### Corpus stats
- Chunk documents : 58,077
- Abstract docs   : 251
- Total indexed   : 58,328
- Build time      : <1 second
- Cache size      : ~200MB


## Step 10b — BGE-M3 Chunk Embeddings
**Date:** 2026-04-04
**Branch:** feature/step-10b-bge-embeddings
**Files added:**
- `src/query/bge_embedder.py`
- `scripts/test_bge.py`
- `scripts/build_bge_index.py`

### Goal
Build second dense index for chunk-level evidence retrieval.
BGE-M3 is better than SPECTER2 for exact chunk retrieval —
SPECTER2 handles paper-level similarity, BGE-M3 handles
precise evidence grounding.

### What was built
**`bge_embedder.py`** — eight functions:
- `get_bge_model()` — loads BAAI/bge-m3 on GPU, fp16,
  singleton pattern
- `get_or_create_bge_index()` — loads/creates IndexFlatIP
  dim=1024
- `load_bge_id_map()` / `save_bge_id_map()` — chunk_id →
  bge_faiss_row mapping
- `embed_chunks_bge()` — document-side embedding, no prefix,
  normalised to unit length
- `embed_query_bge()` — query-side embedding with asymmetric
  instruction prefix
- `build_bge_index_from_db()` — pulls all non-noise chunks
  from DuckDB, embeds in batches, saves FAISS + id_map +
  raw numpy matrix; resumable
- `bge_search()` — searches with optional node_id filter

### Design decisions
| Decision | Reason |
|---|---|
| Separate from SPECTER2 index | Different roles — paper vs chunk retrieval |
| fp16 precision | Halves VRAM usage, negligible quality loss |
| Asymmetric prefix on query | BGE-M3 explicitly designed for this |
| Raw matrix saved as .npy | Enables offline rescoring, clustering, debugging |
| Resumable embedding | Safe to interrupt and restart |
| node_id filter in search | Restricts to paper pool from Stage 1 |

### Issue encountered
FlagEmbedding 1.3.5 incompatible with transformers 5.5.0 —
`is_torch_fx_available` removed in transformers 5.x.
Fix: upgraded FlagEmbedding from source via pip install
from GitHub.

### Test results (100-chunk sample)
| Check | Result |
|---|---|
| Shape | ✓ (100, 1024) |
| All norms = 1.0 | ✓ |
| Search relevance | ✓ power law magnetic field chunk top hit |
| Asymmetric prefix | ✓ changes similarity score |

### Full corpus results
| Metric | Value |
|---|---|
| Chunks embedded | 58,077 |
| FAISS vectors | 58,077 |
| ID map entries | 58,077 |
| Match | ✓ |
| Index file | data/faiss/bge_chunks.index |
| Raw matrix | data/faiss/bge_chunks_matrix.npy |
## Step 11 — Query Engine
*(Pending)*

---

## Decisions Log
*Running record of non-obvious choices made during the build.*

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-03 | faiss-cpu over faiss-gpu | Blackwell sm_120 unsupported in faiss-gpu |
| 2026-04-03 | PyTorch nightly cu130 | Only build supporting RTX 5070 Ti (sm_120) |
| 2026-04-03 | DuckDB over SQLite | Columnar, faster analytical queries, no server |
| 2026-04-03 | NetworkX over Neo4j | Graph fits in RAM; no JVM; full Python API |
| 2026-04-03 | SPECTER2 over general embeddings | Trained on scientific papers; domain vocabulary |
| 2026-04-03 | Paragraph-level chunks | Balance between precision and context |
| 2026-04-03 | Hierarchical chunks | Section summary attached to each paragraph chunk |

---

## Issues & Fixes Log
*Running record of problems encountered and how they were resolved.*

| Date | Issue | Fix |
|---|---|---|
| 2026-04-03 | PowerShell blocked `.ps1` activation | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| 2026-04-03 | cu124 PyTorch: sm_120 incompatibility warning | Switched to nightly cu130 build |
| 2026-04-03 | Venv got CPU-only PyTorch by default | Explicitly reinstalled with cu130 index URL |

---

## File Structure
```
Knowledge_Base/
├── .venv/                      # virtual environment (not committed to git)
├── papers/                     # downloaded PDFs
├── paper_node_schema.json      # four-layer node schema definition
├── collect_papers.py           # ADS citation graph crawler + PDF downloader
├── DEVLOG.md                   # this file
├── step1_env_check.py
├── step2_verify.py
├── step3_embedding_check.py
└── notebooks/                  # Jupyter notebooks (Steps 4+)
```

---

## Reference: Useful Commands

```powershell
# Activate venv (run from project root every session)
.venv\Scripts\Activate.ps1

# Launch VS Code in project folder
code .

# Check which Python / pip you're using
where python
where pip

# Confirm GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Start Jupyter in VS Code
# Ctrl+Shift+P → "Create New Jupyter Notebook" → select .venv kernel
```
