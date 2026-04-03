# ICM Magnetic Field — Knowledge Base

A local hybrid knowledge graph over the intracluster medium (ICM)
magnetic field literature, supporting structured, semantic, and
citation-graph queries.

## Architecture

```
PDFs
  ↓
PyMuPDF / pdfplumber     → text extraction + chunking
  ↓
SPECTER2 (GPU)           → scientific embeddings
  ↓
  ├── FAISS              → dense vector search
  ├── DuckDB             → structured metadata + Layer 3 extraction
  └── NetworkX           → citation graph traversal
```

## Stack

| Component | Version | Role |
|---|---|---|
| Python | 3.12 | Runtime |
| PyTorch | 2.12.0 + cu130 | GPU backend |
| sentence-transformers | 5.3.0 | SPECTER2 embeddings |
| FAISS | 1.13.2 | Vector search |
| DuckDB | 1.5.1 | Structured queries |
| NetworkX | 3.6.1 | Citation graph |
| PyMuPDF + pdfplumber | 1.27 / 0.11 | PDF parsing |

## Setup

```powershell
git clone https://github.com/rsalehin/icm-magnetic-field-kb.git
cd icm-magnetic-field-kb
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note:** PyTorch must be installed separately with the correct
> CUDA index URL for your GPU. See `DEVLOG.md` Step 2 for details.

## Project Structure

```
icm-magnetic-field-kb/
├── notebooks/              # Jupyter notebooks (one per pipeline step)
├── src/                    # Source modules
│   ├── extraction/         # PDF parsing + chunking
│   ├── embedding/          # SPECTER2 embedding pipeline
│   ├── storage/            # DuckDB + FAISS + NetworkX interfaces
│   └── query/              # Query engine
├── schemas/
│   └── paper_node_schema.json
├── scripts/                # One-off utility scripts
├── tests/                  # Unit tests
├── DEVLOG.md               # Step-by-step build log
├── requirements.txt
└── .gitignore
```

## Build Progress

See `DEVLOG.md` for the full step-by-step build log including
decisions, issues encountered, and fixes.

## Domain

Research context: turbulent magnetic fields in galaxy clusters —
GRF vs BxC/Biot-Savart models, Faraday rotation measure (RM)
synthesis, depolarization, and magnetic power spectra.

Target journals: MNRAS, ApJ, A&A.
