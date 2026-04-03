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

*(Fill in after running)*

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

## Step 4 — PDF Text Extraction
*(Pending)*

---

## Step 5 — Chunking Pipeline
*(Pending)*

---

## Step 6 — DuckDB Schema Initialisation
*(Pending)*

---

## Step 7 — FAISS Index Initialisation
*(Pending)*

---

## Step 8 — NetworkX Graph Initialisation
*(Pending)*

---

## Step 9 — Full Ingestion Pipeline (single paper)
*(Pending)*

---

## Step 10 — Layer 3 LLM Extraction
*(Pending)*

---

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
