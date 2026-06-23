import sys
import os
import json
import time
import math
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from app.services.retrieval import search
import app.config

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "questions.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "eval", "benchmark_cascade")
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


def compute_metrics(ranked_docs: list[str], gold_ranked_docs: list[str]) -> dict:
    mrr = compute_mrr(ranked_docs, gold_ranked_docs)
    gold_top10 = set(gold_ranked_docs[:10])
    relevance = [
        1.0 if d in gold_ranked_docs[:3] else 0.5 if d in gold_top10 else 0.0
        for d in ranked_docs[:10]
    ]
    ndcg = ndcg_at_k(relevance, 10)
    overlap_at_10 = len(set(ranked_docs[:10]) & set(gold_ranked_docs[:10])) / 10.0
    return {
        "mrr": round(mrr, 4),
        "ndcg@10": round(ndcg, 4),
        "overlap@10": round(overlap_at_10, 4),
    }


def load_gold() -> dict | None:
    path = os.path.join(RESULTS_DIR, "gold_results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_gold(data: dict):
    path = os.path.join(RESULTS_DIR, "gold_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_run(label: str, data: dict):
    safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
    path = os.path.join(RESULTS_DIR, f"run_{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_mode(label: str, mode: str, questions: list[dict],
             cross_encoder_top_n: int | None = None) -> dict:
    os.environ["RERANKER_MODE"] = mode
    if cross_encoder_top_n is not None:
        os.environ["CROSS_ENCODER_TOP_N"] = str(cross_encoder_top_n)

    results = []
    all_times = []

    for i, q in enumerate(questions):
        question = q["question"]
        print(f"  [{i+1}/{len(questions)}] {question[:60]}...", end=" ", flush=True)
        t0 = time.time()
        r = search(question, n_results=10)
        elapsed = time.time() - t0
        all_times.append(elapsed)
        print(f"{elapsed:.2f}s", flush=True)
        results.append({
            "question": question,
            "documents": r["documents"],
            "metadatas": r["metadatas"],
            "distances": r["distances"],
            "time_seconds": round(elapsed, 3),
        })

    sorted_times = sorted(all_times)
    n = len(sorted_times)
    return {
        "label": label,
        "mode": mode,
        "cross_encoder_top_n": cross_encoder_top_n,
        "results": results,
        "avg_time": round(sum(all_times) / n, 3),
        "total_time": round(sum(all_times), 3),
        "latency_p50": round(sorted_times[int(n * 0.5)], 3),
        "latency_p95": round(sorted_times[int(n * 0.95)], 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["gold", "lightweight", "cascade", "cross-encoder"])
    parser.add_argument("--ce-top-n", type=int, default=None,
                        help="CROSS_ENCODER_TOP_N for cascade mode")
    parser.add_argument("--label", type=str, default=None,
                        help="Custom label for this run")
    args = parser.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    label = args.label or args.mode
    if args.mode == "cascade" and args.ce_top_n is not None:
        label = f"cascade N={args.ce_top_n}"

    print(f"Rodada: {label} ({len(questions)} queries)")
    print("=" * 60)

    if args.mode == "gold":
        data = run_mode(label, "cross-encoder", questions, None)
        save_gold(data)
    else:
        data = run_mode(label, args.mode, questions, args.ce_top_n)
        save_run(label, data)

    print(f"\nConcluido: {label}")
    print(f"  avg_time={data['avg_time']:.3f}s  P50={data['latency_p50']:.3f}s  P95={data['latency_p95']:.3f}s")


if __name__ == "__main__":
    main()
