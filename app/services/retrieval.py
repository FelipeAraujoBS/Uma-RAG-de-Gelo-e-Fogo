import sys
import os
import re
from collections import defaultdict

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from app.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL, RERANKER_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(COLLECTION_NAME)


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def _init_bm25():
    all_data = collection.get(include=["documents", "metadatas"])
    if not all_data["ids"]:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' está vazia. "
            "Execute 'python scripts/embed_paragraphs.py --rebuild' primeiro."
        )
    texts = all_data["documents"]
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, texts, ids, metadatas


bm25_index, all_texts, all_ids, all_metadatas = _init_bm25()
id_to_idx = {doc_id: i for i, doc_id in enumerate(all_ids)}


def search(question: str, n_results: int = 20) -> dict:
    query_emb = model.encode(
        f"Represent this sentence for searching relevant passages: {question}"
    ).tolist()

    sem = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results * 3,
        include=["documents", "metadatas", "distances"],
    )

    sem_by_id = {}
    for i, doc_id in enumerate(sem["ids"][0]):
        sem_by_id[doc_id] = {
            "document": sem["documents"][0][i],
            "metadata": sem["metadatas"][0][i],
            "distance": sem["distances"][0][i],
        }

    tokenized_q = _tokenize(question)
    bm25_scores = bm25_index.get_scores(tokenized_q)
    bm25_ranked = np.argsort(bm25_scores)[::-1]

    K = 60
    rrf = defaultdict(float)
    for rank, doc_id in enumerate(sem["ids"][0]):
        rrf[doc_id] += 1.0 / (K + rank)
    for rank, idx in enumerate(bm25_ranked[:n_results * 3]):
        rrf[all_ids[idx]] += 1.0 / (K + rank)

    top_ids = [
        doc_id for doc_id, _ in sorted(
            rrf.items(), key=lambda x: x[1], reverse=True
        )[:n_results * 2]
    ]

    docs, metas, dists = [], [], []
    for doc_id in top_ids:
        if doc_id in sem_by_id:
            docs.append(sem_by_id[doc_id]["document"])
            metas.append(sem_by_id[doc_id]["metadata"])
            dists.append(sem_by_id[doc_id]["distance"])
        else:
            idx = id_to_idx[doc_id]
            docs.append(all_texts[idx])
            metas.append(all_metadatas[idx])
            dists.append(1.0)

    pairs = [(question, d) for d in docs]
    ce_scores = reranker.predict(pairs)

    combined = sorted(
        zip(ce_scores, docs, metas, dists),
        key=lambda x: x[0], reverse=True,
    )

    return {
        "documents": [item[1] for item in combined[:n_results]],
        "metadatas": [item[2] for item in combined[:n_results]],
        "distances": [item[3] for item in combined[:n_results]],
    }
