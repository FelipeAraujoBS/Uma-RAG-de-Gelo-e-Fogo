import json
import os

import numpy as np
from app.config import RERANKER_WEIGHTS_PATH

DEFAULT_WEIGHTS = {"bm25": 0.2, "dense": 0.6, "rrf": 0.2}


def _load_weights(path: str | None = None) -> dict[str, float]:
    path = path or RERANKER_WEIGHTS_PATH
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return DEFAULT_WEIGHTS.copy()


def _normalize(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    min_v, max_v = arr.min(), arr.max()
    if max_v - min_v > 1e-10:
        return (arr - min_v) / (max_v - min_v)
    return np.full_like(arr, 0.5)


def lightweight_rerank(
    chunks: list[dict],
    weights: dict[str, float] | None = None,
) -> list[dict]:
    if weights is None:
        weights = _load_weights()

    norm_bm25 = _normalize([c["bm25_score"] for c in chunks])
    norm_dense = _normalize([c["dense_cosine"] for c in chunks])
    norm_rrf = _normalize([c["rrf_score"] for c in chunks])

    for scores in (norm_bm25, norm_dense, norm_rrf):
        assert len(scores) == len(chunks), "normalization length mismatch"

    final_scores = (
        weights["bm25"] * norm_bm25
        + weights["dense"] * norm_dense
        + weights["rrf"] * norm_rrf
    )

    ranked = sorted(
        zip(final_scores, chunks),
        key=lambda x: x[0], reverse=True,
    )

    result = []
    for score, chunk in ranked:
        chunk["final_score"] = float(score)
        result.append(chunk)

    return result
