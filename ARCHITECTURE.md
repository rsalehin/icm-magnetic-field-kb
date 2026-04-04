# ICM Knowledge Base — Full System Architecture
**Version:** 1.0  
**Date:** 2026-04-04  
**Status:** Finalized before Step 10 implementation

---

## 1. System Overview

A local, offline, private hybrid knowledge graph over 251 ICM magnetic
field papers (1995–2026). Supports three query types that no existing
tool handles simultaneously:

1. **Factual lookup** — "What σ_RM did Murgia 2004 find for A119?"
2. **Structured query** — "Papers measuring B₀ > 5 µG after 2010"
3. **Relational/graph** — "What followed Murgia 2004 methodologically?"

---

## 2. Ingestion Pipeline (Complete — Steps 1–9)

### Stage A — PDF Extraction (local, no network)
- PyMuPDF extracts raw text page by page
- `text_cleaner.py` strips boilerplate (arXiv stamps, running headers,
  affiliations, equation labels)
- Soft-wrap joining reconstructs full sentences from column-wrapped lines
- Paragraph splitter: blank-line boundaries first, sentence-boundary
  heuristic for large blocks (>400 chars)
- Section detection: numbered headings + named sections (Abstract,
  Introduction, etc.)
- min_chars=150 filters noise fragments
- Content flags per chunk: has_equation, has_table, has_figure_ref,
  has_numerical_result
- Output: Layer 1 (partial) + Layer 2 structured dict

### Stage B — ADS Enrichment (network)
- arxiv_id derived from filename
- NASA ADS API: bibcode, title, authors, journal, DOI, abstract,
  keywords, citation_count
- 0.3s polite rate limit between calls
- Failure → stage_b = "failed", pipeline continues

### Stage C — SPECTER2 Embedding (GPU)
- Model: allenai/specter2_base (768-dim)
- Device: CUDA (RTX 5070 Ti, 17GB VRAM)
- Batch size: 64
- Skips noise chunks (token_count < 30)
- ~221 embeddings per paper average

### Stage D — Storage Write
- DuckDB: papers, authors, chunks, ingestion_state tables
- FAISS: IndexFlatIP, cosine via normalised inner product
- NetworkX: DiGraph node added, graph saved as GraphML
- faiss_index_id written back to DuckDB chunks table

### Full corpus results
- 251 papers, 16.8 minutes, 0 failures
- 58,077 SPECTER2 vectors in FAISS
- 251 nodes in NetworkX (0 edges — pending Step 11)

---

## 3. Storage Layer (Three Backends)

### 3.1 DuckDB (structured queries)
**Tables:**
- `papers` — Layer 1 bibliographic + stage flags
- `authors` — normalised, position-ordered
- `chunks` — Layer 2 with all content flags + faiss_index_id
- `ingestion_state` — pipeline progress per paper

**Role at query time:**
- Hard filters: year, journal, explicit instrument/source constraints
- Soft boost source: section_title, has_equation, has_numerical_result
- Abstention signal: evidence support count

**Design decisions:**
- JSON arrays stored as VARCHAR (keywords, sections, figure_refs)
- is_noise flag on chunks < 30 tokens — excluded from embedding
- Idempotent insert — safe to re-run pipeline

### 3.2 FAISS (dense vector search)
**Index 1 — SPECTER2 paper+chunk level**
- IndexFlatIP, dim=768, exact search
- Normalised vectors → cosine similarity via inner product
- id_map: chunk_id → faiss_row_index (JSON)
- Used for: paper-level retrieval, initial chunk candidates

**Index 2 — BGE-M3 chunk level (Step 10b)**
- IndexFlatIP, dim=1024
- Separate index file and id_map
- Used for: precise chunk-level dense retrieval
- Asymmetric retrieval: query prefix applied, document side no prefix

**Design decisions:**
- CPU FAISS — Blackwell sm_120 unsupported in faiss-gpu
- Exact search appropriate at 58k vectors
- Raw embedding matrix saved separately for offline rescoring/clustering

### 3.3 NetworkX (citation graph)
**Graph:** DiGraph, persisted as GraphML

**Node attributes:**
- arxiv_id, bibcode, title, year, journal
- citation_count, total_chunks
- stage_a/b/c/d flags
- is_seed flag

**Edge types (Step 11):**
- CITES / CITED_BY — weight: strong
- SHARED_OBJECT — weight: strong
- SHARED_INSTRUMENT + SHARED_METHOD — weight: medium
- SHARED_BROAD_TOPIC — weight: weak (use sparingly)

**Expansion rules (Step 10d):**
- Max 15 citation neighbours
- Max 10 shared-method neighbours
- Max 10 shared-object neighbours
- Hard cap: 1-hop only for answer extraction
- 2-hop only for "related work" discovery mode
- Graph expansion disabled until Step 11 populates edges

---

## 4. Query Engine (Steps 10a–10h + Step 11)

### 4.1 Step 10a — BM25 Index

**What:** Lexical sparse retrieval over all chunk texts

**Implementation:**
- `rank_bm25` library
- Index field: weighted combination of
  `title + section_title + chunk_text + keywords`
- Persistence: DuckDB/JSONL as canonical store,
  pickle as regeneratable cache artifact
- Abstract/title chunks indexed separately with boosted weight

**Why BM25 matters:**
Scientific queries contain exact terms that dense retrieval misses:
`Burn law`, `RM synthesis`, `LOFAR HBA`, `σ_RM`, `BxC`, `Briggs weighting`

**Rebuild trigger:** any new paper added to corpus

---

### 4.2 Step 10b — BGE-M3 Chunk Embeddings

**What:** Second dense embedding model, chunk-level

**Model:** BAAI/bge-m3 (1024-dim)
- Supports dense, sparse, and multi-vector retrieval
- Max 8192 token input
- Better than SPECTER2 for chunk-level evidence retrieval
- SPECTER2 remains for paper-level similarity

**Implementation:**
- Re-embed all 58,077 non-noise chunks
- Asymmetric retrieval: query side uses instruction prefix
  `"Represent this sentence for searching relevant passages: "`
  document side: no prefix
- Store in separate FAISS IndexFlatIP (dim=1024)
- Separate id_map JSON: chunk_id → bge_faiss_row
- Raw embedding matrix saved as numpy .npy file for offline use
- Model name + revision + prefix convention recorded in metadata

**Estimated time:** ~20 minutes on RTX 5070 Ti

---

### 4.3 Step 10c — Query Planner

**What:** Rule-based intent detection and query decomposition

**Input:** raw user query string

**Output (typed contract):**
```json
{
  "intent": "fact | comparison | synthesis | discovery",
  "entities": {
    "sources":     ["A119", "Coma"],
    "methods":     ["GRF", "RM synthesis"],
    "instruments": ["VLA", "MeerKAT"],
    "authors":     ["Murgia", "Bonafede"],
    "years":       {"min": 2010, "max": 2024},
    "quantities":  ["sigma_RM", "B_0"]
  },
  "explicit_filters": {
    "year_min":    2020,
    "year_max":    null,
    "journal":     null,
    "instrument":  ["MeerKAT"]
  },
  "soft_boosts": {
    "sections":    ["results", "abstract"],
    "methods":     ["rm_synthesis"],
    "topics":      ["polarization"]
  },
  "budgets": {
    "paper_k":       40,
    "chunk_lex_k":   120,
    "chunk_dense_k": 120,
    "rerank_k":      40,
    "evidence_k":    10
  },
  "synthesis_mode": "non_thinking | thinking"
}
```

**Intent detection rules:**
- `fact`: query contains specific value, quantity, measurement, "what is",
  "how much", "which value" → budgets: paper_k=40, evidence_k=8
- `comparison`: "compare", "difference between", "versus", "better than"
  → budgets: paper_k=80, evidence_k=12, min 3 distinct papers
- `synthesis`: "what does the literature say", "overview", "review",
  "across papers" → budgets: paper_k=150, evidence_k=15, thinking mode
- `discovery`: "related to", "similar to", "what papers", "who worked on"
  → budgets: paper_k=100, SPECTER2 weighted heavily

**Fallback:** if no intent matches confidently → fact/discovery hybrid,
no overfiltering, minimal hard constraints

**Explicit vs inferred constraint rule:**
- Explicit: user literally states year, instrument, journal → hard filter
- Inferred: query domain implies instrument class → soft boost only
- Never hard-filter on inferred metadata

---

### 4.4 Step 10d — Hybrid Retriever

**What:** Multi-signal retrieval combining all three backends

**Architecture:**

```
Query
  ↓
[Paper Phase]
  A. SPECTER2 FAISS → top paper_k papers (intent-dependent)
  B. DuckDB explicit hard filters applied
  C. NetworkX typed 1-hop expansion (disabled until Step 11)
     - citation edges: up to 15 neighbours
     - shared-method edges: up to 10
     - shared-object edges: up to 10
  → Scored paper pool

[Chunk Phase — only from paper pool]
  A. BM25 lexical search → top chunk_lex_k chunks
  B. BGE-M3 dense search → top chunk_dense_k chunks
     (asymmetric prefix applied to query)
  C. Fusion via Reciprocal Rank Fusion
  D. Section priors + metadata boosts applied

[Scoring formula]
  retrieval_score =
    rrf_score
    + α * section_prior
    + β * metadata_boost

  where:
    rrf_score = Σ 1/(k + rank_i), k=60 (standard)
    α = 0.15 (section prior weight, tunable)
    β = 0.10 (metadata boost weight, tunable)

  section_prior values per intent:
    fact:       {abstract: 0.8, results: 1.0, conclusion: 0.7,
                 methods: 0.6, introduction: 0.3}
    comparison: {abstract: 0.7, results: 0.9, discussion: 0.9,
                 conclusion: 0.8, methods: 0.7}
    synthesis:  {abstract: 0.8, results: 0.8, discussion: 0.9,
                 conclusion: 0.9, introduction: 0.5}
    discovery:  all sections equal weight (0.5)

  metadata_boost sources:
    has_numerical_result = True  → +0.1
    has_equation = True          → +0.05 (for fact/method queries)
    year in preferred range      → +0.05

→ Top rerank_k chunks sent to reranker
```

---

### 4.5 Step 10e — BGE Reranker

**What:** Cross-encoder precision reranking

**Model:** BAAI/bge-reranker-v2-m3
- Cross-encoder: sees query + chunk together
- Much more precise than bi-encoder similarity
- Too slow for full corpus, correct for top-N candidates

**Implementation:**
- Input: top rerank_k chunks from Step 10d (default 40)
- Output: reranked scores for all 40 chunks
- Store per chunk: rerank_score, pre_rerank_score, paper_id, section

**Diversity enforcement (after reranking, not before):**
- Sort by rerank score descending
- Apply caps:
  - Max 3 chunks per paper (fact/comparison)
  - Max 2 chunks per paper (synthesis — forces diversity)
  - Minimum 2 distinct papers for comparison queries
  - Minimum 3 distinct papers for synthesis queries
- Final output: top evidence_k chunks after diversity

---

### 4.6 Step 10f — Evidence Assembler

**What:** Structure retrieved chunks into an evidence pack before generation

**Deduplication (cosine similarity):**
- Compute pairwise cosine between top chunks using stored embeddings
- Chunks with cosine > 0.92 → keep highest rerank score, discard rest
- cosine for deduplication ONLY — not for conflict detection

**Grouping:**
- Group by paper_id
- Within paper: group by section_title
- Order: highest rerank score first within each group

**Conflict detection (rule-based lexical, not cosine):**
Scan chunk texts for contradiction cue phrases:
```
positive cues:  "confirms", "consistent with", "in agreement"
negative cues:  "inconsistent with", "contradicts", "no evidence",
                "fails to detect", "not significant", "in contrast",
                "however", "disputes", "suggests otherwise"
uncertainty:    "tentative", "unclear", "debated", "controversial"
```
If negative/uncertainty cues found across different papers on same
topic → flag disagreement in evidence pack

**Abstention signal (numeric rules):**
Abstain or flag weak support if ANY of:
- Fewer than 3 distinct supporting chunks
- Fewer than 2 distinct papers
- Maximum rerank score < 0.30
- All supporting chunks from single paper (single-source flag)
- Contradictory evidence present (disagreement flag)

**Evidence pack output:**
```json
{
  "query": "...",
  "intent": "fact",
  "evidence": [
    {
      "chunk_id": "arxiv:...__c0042",
      "paper_id": "arxiv:astro-ph/0406225",
      "title": "Magnetic fields and Faraday rotation...",
      "year": 2004,
      "journal": "A&A",
      "section": "4. Simulated Rotation Measures",
      "text": "...",
      "rerank_score": 0.87,
      "retrieval_score": 0.73
    }
  ],
  "support_stats": {
    "distinct_chunks": 8,
    "distinct_papers": 4,
    "max_rerank_score": 0.87,
    "has_disagreement": false,
    "single_source": false
  },
  "abstain": false,
  "abstain_reason": null
}
```

---

### 4.7 Step 10g — Evaluation Set

**What:** 50 benchmark questions before wiring the generator

**Question types (balanced):**
- 15 × fact lookups (specific values, authors, years)
- 15 × method comparisons (GRF vs BxC, analytical vs simulation)
- 10 × cross-paper synthesis (what does literature say about X)
- 5  × discovery (find papers about X)
- 5  × abstention cases (questions with no answer in corpus)

**Per question record:**
```json
{
  "question": "What spectral index n did Murgia 2004 find for A119?",
  "intent": "fact",
  "expected_papers": ["arxiv:astro-ph/0406225"],
  "expected_sections": ["6. Application to the data"],
  "expected_answer_contains": ["n = 2", "spectral index"],
  "should_abstain": false
}
```

**Evaluation metrics:**
- Paper recall@K: were expected papers in top-K retrieved?
- Chunk precision: were relevant chunks in evidence pack?
- Abstention accuracy: did system abstain when it should?
- Answer accuracy: manual review for factual correctness

**Purpose:** identify which stage (retrieval / reranking / assembly)
is failing before the generator is added

---

### 4.8 Step 10h — Qwen3 Generator

**What:** Local LLM synthesis from evidence pack

**Model:** qwen3:14b via Ollama
- ~9GB VRAM — fits alongside SPECTER2 + BGE-M3 simultaneously
- Supports thinking / non-thinking mode routing
- Strong scientific reasoning and instruction following

**Routing:**
- `non_thinking`: fact, direct evidence-grounded answers
- `thinking`: comparison, synthesis, cross-paper reasoning

**Evidence budget per intent:**
- fact: max 8 chunks, from at least 2 papers
- comparison: max 12 chunks, from at least 3 papers
- synthesis: max 15 chunks, from at least 4 papers
- discovery: max 10 chunks, no minimum paper constraint

**Prompt structure:**
```
[SYSTEM]
You are a scientific assistant for astrophysics research.
Answer ONLY from the provided evidence.
Cite every claim with [paper_id, section].
If evidence is insufficient, say so explicitly.
If evidence contradicts itself, report the disagreement.
Do not synthesise beyond what the evidence supports.

[EVIDENCE]
{structured evidence pack}

[QUESTION]
{user query}
```

**Output schema (typed):**
```json
{
  "answer": "...",
  "claims": [
    {
      "text": "Murgia 2004 found spectral index n=2 for A119.",
      "citations": ["arxiv:astro-ph/0406225, §6"],
      "confidence": "strong"
    }
  ],
  "disagreements": [],
  "insufficiencies": [],
  "abstain": false
}
```

**Abstention behaviour:**
- If `evidence_pack.abstain = true` → generator told to abstain
- Generator may additionally abstain if evidence doesn't answer question
- Both cases produce explicit abstention reason in output

---

### 4.9 Step 11 — Citation Edge Construction

**What:** Populate NetworkX graph with real typed citation edges

**Source:** NASA ADS references endpoint per paper bibcode

**Edge types constructed:**
```
CITES(A → B)         from ADS references list
CITED_BY(B → A)      inverse, derived
```

**Additional edges (derived from Layer 3 extraction — Step 12):**
```
SHARED_OBJECT(A, B)      both study same cluster/source
SHARED_METHOD(A, B)      both use same method (GRF, BxC, etc.)
SHARED_INSTRUMENT(A, B)  both use same telescope/survey
```

**Edge attributes:**
- citation_role: foundational | methodological | data_source |
  comparison | incidental
- context_chunk_id: which chunk contains the citation
- weight: float (strong=1.0, medium=0.6, weak=0.3)

**Expansion upgrade:**
After Step 11, Step 10d graph expansion activates:
- 1-hop citation: up to 15 neighbours
- 1-hop shared-method: up to 10 neighbours
- 1-hop shared-object: up to 10 neighbours

---

## 5. Layer 3 Structured Extraction (Step 12)

**What:** LLM-based extraction of domain knowledge per paper

**Extracted fields:**
- `methods`: name, aliases, category, inputs, outputs
- `datasets_and_objects`: cluster name, type, redshift, instrument
- `key_quantities`: symbol, value, unit, object, condition, chunk_id
- `scientific_claims`: claim text, type, confidence, evidence
- `physical_domain`: controlled vocabulary (GRF, BxC, σ_RM, β-model)
- `compared_against`: which prior work is explicitly compared
- `limitations`: explicitly stated caveats
- `open_questions`: questions raised but not answered

**Model:** Qwen3:14b (already loaded for query engine)
**Persistence:** DuckDB Layer 3 JSON column on papers table
**Enables:** SHARED_METHOD and SHARED_OBJECT edges in Step 11

---

## 6. Design Principles

### Separation of concerns
- Stage A: pure local, no network
- Stage B: network only, no PDF
- Stage C: GPU only, no storage
- Stage D: storage only, no computation

### Incrementality
- New paper → run all 4 stages → backends updated
- Existing paper → skip at DuckDB check → no reprocessing
- BM25 index rebuilt when corpus changes
- BGE-M3 index appended incrementally

### Failure isolation
- Stage B failure (ADS) → paper still ingested with partial Layer 1
- Stage C failure → paper in DuckDB/graph but not FAISS
- Stage D failure → paper processed but not stored → re-run

### Query-time efficiency
All static work precomputed:
- SPECTER2 embeddings
- BGE-M3 embeddings
- BM25 index
- Graph adjacency lists
- DuckDB metadata projections

Query-time only:
- Query embedding (SPECTER2 + BGE-M3)
- BM25 query
- FAISS search
- DuckDB filter
- RRF fusion
- BGE reranking
- Evidence assembly
- Qwen3 synthesis

### Abstention over hallucination
The system is designed to say "insufficient evidence" rather than
fabricate. Numeric abstention rules are enforced at the evidence
assembler before the generator is invoked. The generator is
additionally instructed to abstain explicitly.

### Graph expansion is typed and capped
Generic 1-hop expansion is noise injection. All expansion is:
- Typed by edge category
- Capped per edge type
- Limited to 1-hop for answer extraction
- 2-hop only for discovery mode

---

## 7. File Structure

```
Knowledge_Base/
├── papers/                          # PDF corpus (gitignored)
├── data/
│   ├── knowledge_base.duckdb        # primary structured store
│   ├── faiss/
│   │   ├── chunks.index             # SPECTER2 FAISS index
│   │   ├── chunks_id_map.json       # chunk_id → faiss_row
│   │   ├── bge_chunks.index         # BGE-M3 FAISS index (Step 10b)
│   │   ├── bge_chunks_id_map.json   # chunk_id → bge_row
│   │   └── bge_chunks_matrix.npy    # raw BGE-M3 embeddings
│   ├── bm25/
│   │   ├── bm25_corpus.jsonl        # canonical tokenized corpus
│   │   └── bm25_index.pkl           # cache artifact (regeneratable)
│   ├── graph/
│   │   └── citation_graph.graphml   # NetworkX graph
│   └── ingestion_log.txt
├── src/
│   ├── extraction/
│   │   ├── pdf_extractor.py         # Stage A
│   │   └── text_cleaner.py
│   ├── enrichment/
│   │   └── ads_enricher.py          # Stage B
│   ├── storage/
│   │   ├── db.py                    # DuckDB interface
│   │   ├── faiss_index.py           # FAISS interface
│   │   └── graph.py                 # NetworkX interface
│   ├── query/                       # Step 10
│   │   ├── bm25_index.py            # Step 10a
│   │   ├── bge_embedder.py          # Step 10b
│   │   ├── planner.py               # Step 10c
│   │   ├── retriever.py             # Step 10d
│   │   ├── reranker.py              # Step 10e
│   │   ├── assembler.py             # Step 10f
│   │   └── generator.py             # Step 10h
│   └── pipeline.py                  # orchestrator
├── schemas/
│   └── paper_node_schema.json
├── scripts/
│   ├── ingest_all.py
│   ├── test_*.py
│   └── evaluate.py                  # Step 10g
├── notebooks/
├── DEVLOG.md
├── ARCHITECTURE.md                  # this document
├── CONTRIBUTING.md
├── README.md
└── requirements.txt
```

---

## 8. Known Limitations and Future Work

| Limitation | Impact | Future fix |
|---|---|---|
| Graph edges = 0 until Step 11 | Graph expansion disabled | Step 11 |
| Layer 3 extraction pending | No structured claim queries | Step 12 |
| Abstract not in chunk FAISS | Lower discovery recall | Step 10a fix |
| Section titles truncated at first line | Minor mislabelling | Chunker v2 |
| Two-column layout merge artifacts | Occasional garbled chunks | Chunker v2 |
| Figure/table captions not isolated | Evidence gaps | Chunker v2 |
| BM25 rebuilt on corpus change | ~2 min rebuild | Acceptable |
| No NLI conflict detection | Rule-based only | Future model |
| faiss-gpu incompatible with sm_120 | CPU FAISS only | NVIDIA driver update |
| No evaluation set yet | Can't measure precision | Step 10g |
