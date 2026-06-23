import sys
import os
import json
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "eval", "benchmark_cascade")


def load_results(label: str) -> dict:
    safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
    if label == "gold":
        path = os.path.join(RESULTS_DIR, "gold_results.json")
    else:
        path = os.path.join(RESULTS_DIR, f"run_{safe}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def main():
    gold = load_results("gold")
    gold_docs_by_q = {r["question"]: r["documents"] for r in gold["results"]}
    gold_times = [r["time_seconds"] for r in gold["results"]]
    gold_sorted = sorted(gold_times)
    n_g = len(gold_sorted)
    gold_p50 = gold_sorted[int(n_g * 0.5)]
    gold_p95 = gold_sorted[int(n_g * 0.95)]
    gold_avg = sum(gold_times) / n_g

    configs = [
        ("lightweight", None),
        ("cascade_N", 8),
        ("cascade_N", 12),
        ("cascade_N", 16),
        ("cascade_N", 20),
    ]

    safe_labels = ["lightweight", "cascade_N_8", "cascade_N_12", "cascade_N_16", "cascade_N_20"]

    label_map = {
        "lightweight": "Lightweight",
        "cascade_N_8": "Cascade N=8",
        "cascade_N_12": "Cascade N=12",
        "cascade_N_16": "Cascade N=16",
        "cascade_N_20": "Cascade N=20",
    }

    all_metrics = {}
    all_latency = {}

    for label in safe_labels:
        run = load_results(label)
        metrics_list = []
        for r in run["results"]:
            q = r["question"]
            metrics = compute_metrics(r["documents"], gold_docs_by_q[q])
            metrics_list.append({**metrics, "question": q})
        all_metrics[label] = metrics_list
        all_latency[label] = {
            "avg": run["avg_time"],
            "p50": run["latency_p50"],
            "p95": run["latency_p95"],
        }

    # Per-query MRR table
    print("=" * 130)
    print("MRR POR QUERY")
    print("=" * 130)
    header = f"{'Query':<60} {'Lightweight':<13} {'Casc N=8':<13} {'Casc N=12':<13} {'Casc N=16':<13} {'Casc N=20':<13}"
    print(header)
    print("-" * len(header))
    for q_idx in range(18):
        q_short = gold["results"][q_idx]["question"][:57]
        mrr_vals = []
        for label in safe_labels:
            mrr = all_metrics[label][q_idx]["mrr"]
            mrr_vals.append(f"{mrr:.4f}")
        print(f"{q_short:<60} {mrr_vals[0]:<13} {mrr_vals[1]:<13} {mrr_vals[2]:<13} {mrr_vals[3]:<13} {mrr_vals[4]:<13}")

    print()

    # MRR stats
    print("\nMRR STATS (agregado)")
    print("-" * 40)
    for label in safe_labels:
        mrr_vals = [m["mrr"] for m in all_metrics[label]]
        avg_mrr = sum(mrr_vals) / len(mrr_vals)
        min_mrr = min(mrr_vals)
        max_mrr = max(mrr_vals)
        print(f"  {label_map[label]:<20} avg={avg_mrr:.4f}  min={min_mrr:.4f}  max={max_mrr:.4f}")

    print()

    # Main results table
    print("=" * 150)
    print("TABELA COMPLETA — Todos os modos vs Cross-Encoder (gold)")
    print("=" * 150)
    header = f"{'Modo':<20} {'N':<6} {'MRR':<10} {'Overlap@10':<12} {'NDCG@10':<10} {'Lat P50':<10} {'Lat P95':<10} {'Lat média':<10}"
    print(header)
    print("-" * len(header))

    for label, n in configs:
        safe = f"{label}_{n}" if n is not None else label
        ls = all_metrics[safe]
        lt = all_latency[safe]
        display_n = str(n) if n is not None else "-"
        mrr_vals = [m["mrr"] for m in ls]
        avg_mrr = sum(mrr_vals) / len(mrr_vals)
        avg_ndcg = sum(m["ndcg@10"] for m in ls) / len(ls)
        avg_overlap = sum(m["overlap@10"] for m in ls) / len(ls)
        print(f"{label_map[safe]:<20} {display_n:<6} {avg_mrr:<10.4f} {avg_overlap:<12.4f} {avg_ndcg:<10.4f} {lt['p50']:<10.3f} {lt['p95']:<10.3f} {lt['avg']:<10.3f}")

    print()
    print(f"{'Cross-encoder (gold)':<20} {'40':<6} {'1.0000':<10} {'1.0000':<12} {'1.0000':<10} {gold_p50:<10.3f} {gold_p95:<10.3f} {gold_avg:<10.3f}")

    print()
    print()

    # MRR x Latency curve (Bloco B)
    print("=" * 60)
    print("CURVA MRR x LATÊNCIA (Bloco B — Cascade variando N)")
    print("=" * 60)
    print(f"{'N':<8} {'MRR':<10} {'Ganho/CE':<12} {'Lat P50':<10} {'Lat P95':<10}")
    print("-" * 50)

    baseline_mrr = sum(m["mrr"] for m in all_metrics["lightweight"]) / len(all_metrics["lightweight"])
    ce_mrr = 1.0
    gain_total = ce_mrr - baseline_mrr

    print(f"{'LW (base)':<8} {baseline_mrr:<10.4f} {'-':<12} {'-':<10} {'-':<10}")

    for n in [8, 12, 16, 20]:
        safe = f"cascade_N_{n}"
        lt = all_latency[safe]
        mrr_vals = [m["mrr"] for m in all_metrics[safe]]
        avg_mrr = sum(mrr_vals) / len(mrr_vals)
        gain = avg_mrr - baseline_mrr
        pct_ce = (gain / gain_total * 100) if gain_total > 0 else 0
        print(f"  N={n:<5} {avg_mrr:<10.4f} {gain:<+8.4f} ({pct_ce:.0f}%)   {lt['p50']:<10.3f} {lt['p95']:<10.3f}")

    print(f"{'CE full':<8} {ce_mrr:<10.4f} {gain_total:<+8.4f} (100%)   {gold_p50:<10.3f} {gold_p95:<10.3f}")

    print()
    print(f"Lightweight baseline MRR: {baseline_mrr:.4f}")
    print(f"Cross-encoder full MRR:    {ce_mrr:.4f}")
    print(f"Ganho total CE sobre LW:   {gain_total:.4f}")
    print(f"90% do ganho = baseline + {gain_total * 0.9:.4f} = {baseline_mrr + gain_total * 0.9:.4f}")
    print()

    # Criteria evaluation
    print("=" * 80)
    print("AVALIAÇÃO POR CRITÉRIOS")
    print("=" * 80)

    overlap_lw = sum(m["overlap@10"] for m in all_metrics["lightweight"]) / len(all_metrics["lightweight"])
    print(f"\n1. Overlap@10 do baseline lightweight (guarda-corrosão): {overlap_lw:.4f}")
    for label in safe_labels[1:]:
        overlap = sum(m["overlap@10"] for m in all_metrics[label]) / len(all_metrics[label])
        status = "OK" if overlap >= overlap_lw else "REGRESSÃO (CRÍTICO)"
        print(f"   {label_map[label]:<20} Overlap@10={overlap:.4f} — {status}")

    print(f"\n2. NDCG@10 do baseline lightweight: {sum(m['ndcg@10'] for m in all_metrics['lightweight']) / len(all_metrics['lightweight']):.4f}")
    for label in safe_labels[1:]:
        ndcg = sum(m["ndcg@10"] for m in all_metrics[label]) / len(all_metrics[label])
        print(f"   {label_map[label]:<20} NDCG@10={ndcg:.4f}")

    print(f"\n3. Critério de adoção do cascade como default:")
    for n in [8, 12, 16, 20]:
        safe = f"cascade_N_{n}"
        lt = all_latency[safe]
        mrr_vals = [m["mrr"] for m in all_metrics[safe]]
        avg_mrr = sum(mrr_vals) / len(mrr_vals)
        gain = avg_mrr - baseline_mrr
        pct_ce = (gain / gain_total * 100) if gain_total > 0 else 0
        target_90 = baseline_mrr + gain_total * 0.9
        meets_mrr = avg_mrr >= target_90
        meets_lat = lt["p95"] <= 3.5
        print(f"\n   N={n}: MRR={avg_mrr:.4f} ({pct_ce:.0f}% do ganho CE), Lat P95={lt['p95']:.3f}s")
        meets_mrr_str = "SIM" if meets_mrr else "NAO"
        meets_lat_str = "SIM" if meets_lat else "NAO (Lat P95 >> 3.5s)"
        print(f"   -> MRR >= {target_90:.4f} (90% ganho)? {meets_mrr_str}")
        print(f"   -> Lat P95 <= 3.5s? {meets_lat_str}")


if __name__ == "__main__":
    main()
