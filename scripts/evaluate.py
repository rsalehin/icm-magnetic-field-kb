# scripts/evaluate.py
"""
Step 10g — Evaluation script.
Runs retrieval pipeline on benchmark questions.
Measures paper recall, chunk relevance, abstention accuracy.
Does NOT include generator — tests retrieval quality only.
"""
import sys
import json
import faiss
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db          import get_connection
from src.query.planner       import plan_query
from src.query.bm25_index    import load_bm25_index
from src.query.bge_embedder  import (
    get_bge_model, get_or_create_bge_index,
    load_bge_id_map, BGE_MATRIX_PATH,
)
from src.query.retriever     import retrieve
from src.query.reranker      import get_reranker, rerank
from src.query.assembler     import assemble_evidence
from src.storage.faiss_index import load_id_map, INDEX_PATH
from src.pipeline            import get_model

EVAL_SET_PATH = Path("data/eval_set_final.json")


def run_evaluation(
    limit: int = None,
    intent_filter: str = None,
) -> None:
    """
    Run evaluation on benchmark questions.
    Reports per-question and aggregate metrics.
    """
    # ── Load eval set ─────────────────────────────────────────────────────────
    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    if intent_filter:
        eval_set = [q for q in eval_set if q["intent"] == intent_filter]
    if limit:
        eval_set = eval_set[:limit]

    print(f"Running evaluation on {len(eval_set)} questions...")

    # ── Load backends ─────────────────────────────────────────────────────────
    print("Loading backends...")
    conn              = get_connection()
    specter_model     = get_model()
    specter_index     = faiss.read_index(str(INDEX_PATH))
    specter_id_map    = load_id_map()
    bm25_index, bm25_records = load_bm25_index()
    bge_model         = get_bge_model()
    bge_index         = get_or_create_bge_index()
    bge_id_map        = load_bge_id_map()
    reranker          = get_reranker()

    bge_matrix = None
    if BGE_MATRIX_PATH.exists():
        bge_matrix = np.load(str(BGE_MATRIX_PATH))

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = []

    for item in eval_set:
        qid      = item["id"]
        question = item["question"]
        expected_papers  = set(item["expected_papers"])
        expected_contains = item["expected_answer_contains"]
        should_abstain   = item["should_abstain"]

        print(f"\n[{qid}] {question[:65]}...")

        # Run pipeline
        plan   = plan_query(question)
        chunks = retrieve(
            plan, conn,
            specter_index, specter_id_map, specter_model,
            bm25_index, bm25_records,
            bge_index, bge_id_map, bge_model,
        )
        evidence_chunks = rerank(
            question, chunks,
            intent     = plan["intent"],
            evidence_k = plan["budgets"]["evidence_k"],
            min_papers = plan["budgets"]["min_papers"],
            reranker   = reranker,
        )
        pack = assemble_evidence(
            question, plan, evidence_chunks,
            conn, bge_id_map, bge_matrix,
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        retrieved_papers = {
            e["paper_id"].replace("__abstract", "")
            for e in pack["evidence"]
        }

        # Paper recall — did expected papers appear in evidence?
        if expected_papers:
            paper_recall = len(
                expected_papers & retrieved_papers
            ) / len(expected_papers)
        else:
            paper_recall = None  # no expected papers specified

        # Answer contains — do top chunks contain expected terms?
        top_text = " ".join(
            e["text"] for e in pack["evidence"][:3]
        ).lower()
        if expected_contains:
            terms_found = [
                t for t in expected_contains
                if t.lower() in top_text
            ]
            contains_score = len(terms_found) / len(expected_contains)
        else:
            terms_found    = []
            contains_score = None

        # Abstention accuracy
        abstain_correct = (pack["abstain"] == should_abstain)

        result = {
            "id":             qid,
            "intent":         item["intent"],
            "question":       question,
            "should_abstain": should_abstain,
            "did_abstain":    pack["abstain"],
            "abstain_correct": abstain_correct,
            "paper_recall":   paper_recall,
            "contains_score": contains_score,
            "terms_found":    terms_found,
            "terms_expected": expected_contains,
            "max_rerank":     pack["support_stats"]["max_rerank_score"],
            "distinct_papers": pack["support_stats"]["distinct_papers"],
            "has_disagreement": pack["support_stats"]["has_disagreement"],
            "retrieved_papers": list(retrieved_papers)[:5],
        }
        results.append(result)

        # Print result
        status = "✓" if abstain_correct else "✗"
        recall_str = f"{paper_recall:.0%}" if paper_recall is not None else "N/A"
        contains_str = f"{contains_score:.0%}" if contains_score is not None else "N/A"
        print(f"  intent={plan['intent']:<12} "
              f"abstain={str(pack['abstain']):<6} {status}  "
              f"recall={recall_str:<6} "
              f"contains={contains_str:<6} "
              f"max_rerank={pack['support_stats']['max_rerank_score']:.3f}")

        if not abstain_correct:
            print(f"  ✗ Abstention mismatch — "
                  f"expected={should_abstain} got={pack['abstain']}")
        if expected_papers and paper_recall == 0:
            print(f"  ✗ Expected papers not found: {expected_papers}")
        if expected_contains and contains_score == 0:
            print(f"  ✗ No expected terms in top chunks")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"EVALUATION SUMMARY ({len(results)} questions)")
    print(f"{'='*65}")

    abstain_correct  = sum(1 for r in results if r["abstain_correct"])
    recall_scores    = [r["paper_recall"]   for r in results
                        if r["paper_recall"]   is not None]
    contains_scores  = [r["contains_score"] for r in results
                        if r["contains_score"] is not None]

    print(f"Abstention accuracy  : "
          f"{abstain_correct}/{len(results)} "
          f"({abstain_correct/len(results):.0%})")

    if recall_scores:
        print(f"Paper recall (avg)   : "
              f"{sum(recall_scores)/len(recall_scores):.0%} "
              f"({len(recall_scores)} questions with expected papers)")

    if contains_scores:
        print(f"Contains score (avg) : "
              f"{sum(contains_scores)/len(contains_scores):.0%} "
              f"({len(contains_scores)} questions with expected terms)")

    # By intent
    print(f"\nBy intent:")
    for intent in ["fact", "comparison", "synthesis", "discovery"]:
        subset = [r for r in results if r["intent"] == intent]
        if not subset:
            continue
        correct = sum(1 for r in subset if r["abstain_correct"])
        avg_rerank = sum(r["max_rerank"] for r in subset) / len(subset)
        print(f"  {intent:<12} : {correct}/{len(subset)} abstention  "
              f"avg_rerank={avg_rerank:.3f}")

    # Failures
    failures = [r for r in results if not r["abstain_correct"]]
    if failures:
        print(f"\nAbstention failures:")
        for r in failures:
            print(f"  [{r['id']}] expected={r['should_abstain']} "
                  f"got={r['did_abstain']} — {r['question'][:50]}...")

    # Save results
    results_path = Path("data/eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--intent", type=str, default=None,
                        choices=["fact","comparison","synthesis","discovery"])
    args = parser.parse_args()
    run_evaluation(limit=args.limit, intent_filter=args.intent)