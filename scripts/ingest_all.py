# scripts/ingest_all.py
"""
Step 9b — Batch ingestion runner.
Processes all PDFs in papers/ folder through the full pipeline.
Resumable — skips already-ingested papers.
Saves all backends after every paper.
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline        import get_model, ingest_paper
from src.storage.db      import get_connection, init_schema
from src.storage.faiss_index import (
    get_or_create_index, load_id_map,
    save_index, save_id_map,
)
from src.storage.graph   import (
    get_or_create_graph, save_graph,
)

PAPERS_DIR = Path("papers")
LOG_FILE   = Path("data/ingestion_log.txt")


def log(msg: str) -> None:
    """Write timestamped message to log file and stdout."""
    ts  = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_batch(
    papers_dir: Path = PAPERS_DIR,
    limit:      int  = None,   # set to N to process only first N papers
    skip_ads:   bool = False,
) -> None:
    """
    Run full ingestion pipeline on all PDFs in papers_dir.
    Resumable — already-ingested papers are skipped.

    Args:
        papers_dir : folder containing PDF files
        limit      : optional cap for testing (None = all papers)
        skip_ads   : skip ADS enrichment (faster, offline testing)
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 55)
    log("Batch ingestion started")

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    log(f"Found {len(pdfs)} PDFs to process")

    # ── Initialise backends ───────────────────────────────────────────────────
    conn   = get_connection()
    init_schema(conn)
    index  = get_or_create_index()
    id_map = load_id_map()
    G      = get_or_create_graph()
    model  = get_model()

    # ── Track results ─────────────────────────────────────────────────────────
    success  = 0
    skipped  = 0
    failed   = 0
    failures = []

    t_start = time.time()

    # ── Process each paper ────────────────────────────────────────────────────
    for pdf_path in tqdm(pdfs, desc="Ingesting papers", unit="paper"):
        try:
            paper, id_map, G = ingest_paper(
                pdf_path, conn, index, id_map, G, model,
                skip_ads=skip_ads,
            )

            stage_d = paper["meta"].get("stage_d", "")
            if stage_d == "done":
                # Check if it was a skip (already ingested)
                if "Already fully ingested" in str(paper.get("_skip_reason", "")):
                    skipped += 1
                else:
                    success += 1
            else:
                success += 1

            # Save backends every paper — ensures no data loss on crash
            save_index(index)
            save_id_map(id_map)
            save_graph(G)

        except Exception as e:
            failed += 1
            failures.append((pdf_path.name, str(e)))
            log(f"FAILED: {pdf_path.name} — {e}")
            traceback.print_exc()
            continue

    # ── Final save ────────────────────────────────────────────────────────────
    save_index(index)
    save_id_map(id_map)
    save_graph(G)
    conn.close()

    elapsed = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    log("=" * 55)
    log(f"Batch ingestion complete in {elapsed/60:.1f} min")
    log(f"  Processed : {success}")
    log(f"  Skipped   : {skipped}")
    log(f"  Failed    : {failed}")
    log(f"  FAISS vectors : {index.ntotal}")
    log(f"  Graph nodes   : {G.number_of_nodes()}")
    log(f"  Graph edges   : {G.number_of_edges()}")

    if failures:
        log(f"\nFailed papers:")
        for name, err in failures:
            log(f"  {name}: {err}")

    print(f"\nIngestion log saved to: {LOG_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch ingest all papers into the knowledge base."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N papers (for testing)"
    )
    parser.add_argument(
        "--skip-ads", action="store_true",
        help="Skip ADS enrichment (faster, offline)"
    )
    args = parser.parse_args()

    run_batch(limit=args.limit, skip_ads=args.skip_ads)