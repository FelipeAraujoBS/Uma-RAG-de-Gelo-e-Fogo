import sys
import os
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import app.services.retrieval as ret_mod
from app.services.reranker import _normalize, lightweight_rerank, DEFAULT_WEIGHTS

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "questions.json")
WEIGHTS_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "reranker_weights.json")


def collect_training_data(questions: list[dict]) -> list[dict]:
    os.environ["RERANKER_MODE"] = "cross_encoder"
    os.environ["RERANKER_COLLECT_DATA"] = "1"

    samples = []
    for i, q in enumerate(questions):
        question = q["question"]
        print(f"  [{i+1}/{len(questions)}] {question[:60]}...", end=" ", flush=True)

        t0 = time.time()
        _ = ret_mod.search(question, n_results=10)
        elapsed = time.time() - t0

        data = ret_mod._training_data
        if data is None:
            print("SKIP", flush=True)
            continue

        for chunk, ce_score in zip(data["enriched_chunks"], data["ce_scores"]):
            samples.append({
                "question": question,
                "bm25_score": chunk["bm25_score"],
                "dense_cosine": chunk["dense_cosine"],
                "rrf_score": chunk["rrf_score"],
                "ce_score": ce_score,
            })

        print(f"{elapsed:.2f}s | {len(data['enriched_chunks'])} chunks", flush=True)

    return samples


def train(samples: list[dict]) -> tuple[dict, float]:
    bm25 = np.array([s["bm25_score"] for s in samples], dtype=float)
    dense = np.array([s["dense_cosine"] for s in samples], dtype=float)
    rrf = np.array([s["rrf_score"] for s in samples], dtype=float)
    y = np.array([s["ce_score"] for s in samples], dtype=float)

    n_bm25 = _normalize(bm25.tolist())
    n_dense = _normalize(dense.tolist())
    n_rrf = _normalize(rrf.tolist())

    X = np.column_stack([n_bm25, n_dense, n_rrf])

    coeffs, residuals, rank, singular = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    weights = {
        "bm25": round(float(coeffs[0]), 4),
        "dense": round(float(coeffs[1]), 4),
        "rrf": round(float(coeffs[2]), 4),
    }

    return weights, r2


def compute_ordering_metrics(
    question: str,
    enriched_chunks: list[dict],
    ce_scores: list[float],
    lw_weights: dict[str, float],
) -> dict:
    lw_result = lightweight_rerank(enriched_chunks, weights=lw_weights)
    lw_order = [c["doc_id"] for c in lw_result]
    ce_order = [
        c["doc_id"] for c, _ in sorted(
            zip(enriched_chunks, ce_scores),
            key=lambda x: x[1], reverse=True,
        )
    ]

    k = 10
    set_ce = set(ce_order[:k])
    set_lw = set(lw_order[:k])
    overlap = len(set_ce & set_lw) / k

    ce_top1 = ce_order[0]
    mrr = 0.0
    for rank, doc_id in enumerate(lw_order, 1):
        if doc_id == ce_top1:
            mrr = 1.0 / rank
            break

    return {"mrr": round(mrr, 4), "overlap@10": round(overlap, 4)}


def main():
    print("Carregando perguntas de avaliacao...")
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  {len(questions)} perguntas.\n")

    print("Coletando dados de treino (cross-encoder scores + sinais)...")
    samples = collect_training_data(questions)
    print(f"\n  Total de amostras: {len(samples)} ({len(samples)//len(questions)} por query)\n")

    learned_weights, r2 = train(samples)

    print("=" * 60)
    print("RESULTADOS DA REGRESSAO")
    print("=" * 60)
    print(f"  Pesos aprendidos:")
    print(f"    bm25:  {learned_weights['bm25']:.4f}")
    print(f"    dense: {learned_weights['dense']:.4f}")
    print(f"    rrf:   {learned_weights['rrf']:.4f}")
    print(f"  R²:                  {r2:.4f}")
    print(f"  Default weights:     bm25={DEFAULT_WEIGHTS['bm25']}, dense={DEFAULT_WEIGHTS['dense']}, rrf={DEFAULT_WEIGHTS['rrf']}")
    print()

    if r2 > 0.8:
        print("  R² > 0.8 -> combinacao linear explica bem o cross-encoder.")
    elif r2 > 0.5:
        print("  R² entre 0.5 e 0.8 -> correlacao moderada.")
    else:
        print("  R² < 0.5 -> combinacao linear explica pouco do cross-encoder.")
    print()

    print(f"Salvando em {WEIGHTS_OUT_PATH}...")
    with open(WEIGHTS_OUT_PATH, "w") as f:
        json.dump(learned_weights, f, indent=2)
    print("  OK\n")

    print("Comparando ordenacao: learned vs default (re-avaliando todas as queries)...")
    os.environ["RERANKER_MODE"] = "cross_encoder"
    os.environ["RERANKER_COLLECT_DATA"] = "1"

    default_metrics = []
    learned_metrics = []

    for i, q in enumerate(questions):
        question = q["question"]
        ret_mod._training_data = None
        _ = ret_mod.search(question, n_results=10)
        data = ret_mod._training_data
        if data is None:
            continue

        m_default = compute_ordering_metrics(
            question, data["enriched_chunks"], data["ce_scores"], DEFAULT_WEIGHTS
        )
        m_learned = compute_ordering_metrics(
            question, data["enriched_chunks"], data["ce_scores"], learned_weights
        )
        default_metrics.append(m_default)
        learned_metrics.append(m_learned)

    avg_default_mrr = np.mean([m["mrr"] for m in default_metrics])
    avg_learned_mrr = np.mean([m["mrr"] for m in learned_metrics])
    avg_default_overlap = np.mean([m["overlap@10"] for m in default_metrics])
    avg_learned_overlap = np.mean([m["overlap@10"] for m in learned_metrics])

    print()
    print("=" * 60)
    print("COMPARACAO DE ORDENACAO (vs cross-encoder)")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Default':>10} {'Learned':>10} {'Diff':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'MRR':<25} {avg_default_mrr:>10.4f} {avg_learned_mrr:>10.4f} {avg_learned_mrr - avg_default_mrr:>+10.4f}")
    print(f"  {'Overlap@10':<25} {avg_default_overlap:>10.4f} {avg_learned_overlap:>10.4f} {avg_learned_overlap - avg_default_overlap:>+10.4f}")
    print()

    if avg_learned_mrr > avg_default_mrr:
        print(f"  Pesos aprendidos melhoram MRR em {avg_learned_mrr - avg_default_mrr:.4f}")
    else:
        print(f"  Pesos default mantem MRR {avg_default_mrr - avg_learned_mrr:.4f} melhor que learned")

    print("\nConcluido!")


if __name__ == "__main__":
    main()
