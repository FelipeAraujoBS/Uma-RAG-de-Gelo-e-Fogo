import sys
import os
import json
import time
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from app.services.retrieval import search

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "questions.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "eval", "comparison")
os.makedirs(RESULTS_DIR, exist_ok=True)


def ndcg_at_k(relevance: list[float], k: int) -> float:
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(relevance[:k]))
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(ranked_docs: list[str], gold_ranked_docs: list[str]) -> float:
    gold_top1 = gold_ranked_docs[0] if gold_ranked_docs else None
    for rank, doc in enumerate(ranked_docs, 1):
        if doc == gold_top1:
            return 1.0 / rank
    return 0.0


def compute_metrics(results_ce: dict, results_lw: dict) -> dict:
    docs_ce = results_ce["documents"]
    docs_lw = results_lw["documents"]

    mrr = compute_mrr(docs_lw, docs_ce)

    top10_ce = set(docs_ce[:10])
    top10_lw = set(docs_lw[:10])
    overlap_at_10 = len(top10_ce & top10_lw) / 10.0

    top5_ce = set(docs_ce[:5])
    top5_lw = set(docs_lw[:5])
    overlap_at_5 = len(top5_ce & top5_lw) / 5.0

    relevance = [
        1.0 if d in docs_ce[:3] else 0.5 if d in docs_ce[:10] else 0.0
        for d in docs_lw[:10]
    ]
    ndcg = ndcg_at_k(relevance, 10)

    return {
        "mrr": round(mrr, 4),
        "ndcg@10": round(ndcg, 4),
        "overlap@10": round(overlap_at_10, 4),
        "overlap@5": round(overlap_at_5, 4),
    }


def run_mode(mode: str, questions: list[dict]) -> dict:
    os.environ["RERANKER_MODE"] = mode
    results = []
    total_time = 0.0

    for i, q in enumerate(questions):
        question = q["question"]
        print(f"  [{i+1}/{len(questions)}] {question[:60]}...", end=" ", flush=True)

        t0 = time.time()
        r = search(question, n_results=10)
        elapsed = time.time() - t0
        total_time += elapsed

        print(f"{elapsed:.2f}s | {len(r['documents'])} chunks", flush=True)

        results.append({
            "question": question,
            "documents": r["documents"],
            "metadatas": r["metadatas"],
            "distances": r["distances"],
            "time_seconds": round(elapsed, 3),
        })

    return {
        "mode": mode,
        "results": results,
        "avg_time": round(total_time / len(questions), 3),
        "total_time": round(total_time, 3),
    }


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    print("\n=== MODO: cross_encoder ===")
    data_ce = run_mode("cross_encoder", questions)

    print("\n=== MODO: lightweight ===")
    data_lw = run_mode("lightweight", questions)

    print("\n=== COMPARACAO POR QUERY ===")
    all_metrics = []
    for q_ce, q_lw in zip(data_ce["results"], data_lw["results"]):
        assert q_ce["question"] == q_lw["question"]
        metrics = compute_metrics(q_ce, q_lw)
        all_metrics.append(metrics)
        print(f"  {q_ce['question'][:50]}")
        print(f"    MRR={metrics['mrr']:.4f}  NDCG@10={metrics['ndcg@10']:.4f}  "
              f"Overlap@5={metrics['overlap@5']:.4f}  Overlap@10={metrics['overlap@10']:.4f}")

    avg_mrr = float(np.mean([m["mrr"] for m in all_metrics]))
    avg_ndcg = float(np.mean([m["ndcg@10"] for m in all_metrics]))
    avg_overlap_5 = float(np.mean([m["overlap@5"] for m in all_metrics]))
    avg_overlap_10 = float(np.mean([m["overlap@10"] for m in all_metrics]))

    print("\n" + "=" * 70)
    print("RESUMO DA COMPARACAO")
    print("=" * 70)
    print(f"  queries:                       {len(questions)}")
    print(f"  cross_encoder total_time:      {data_ce['total_time']:.2f}s (avg {data_ce['avg_time']:.2f}s)")
    print(f"  lightweight total_time:        {data_lw['total_time']:.2f}s (avg {data_lw['avg_time']:.2f}s)")
    print(f"  speedup:                       {data_ce['total_time'] / data_lw['total_time']:.1f}x")
    print(f"  MRR (lightweight vs CE):       {avg_mrr:.4f}")
    print(f"  NDCG@10 (lightweight vs CE):   {avg_ndcg:.4f}")
    print(f"  Overlap@5:                     {avg_overlap_5:.4f}")
    print(f"  Overlap@10:                    {avg_overlap_10:.4f}")

    summary = {
        "queries": len(questions),
        "cross_encoder": {
            "total_time_seconds": data_ce["total_time"],
            "avg_time_seconds": data_ce["avg_time"],
        },
        "lightweight": {
            "total_time_seconds": data_lw["total_time"],
            "avg_time_seconds": data_lw["avg_time"],
        },
        "speedup_x": round(data_ce["total_time"] / data_lw["total_time"], 1),
        "avg_mrr": float(avg_mrr),
        "avg_ndcg@10": float(avg_ndcg),
        "avg_overlap@5": float(avg_overlap_5),
        "avg_overlap@10": float(avg_overlap_10),
        "per_query": [
            {**m, "question": q_ce["question"]}
            for m, q_ce in zip(all_metrics, data_ce["results"])
        ],
    }

    out_path = os.path.join(RESULTS_DIR, "comparison_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {out_path}")


if __name__ == "__main__":
    main()
