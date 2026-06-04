import sys
import os
import re
import sqlite3
from collections import defaultdict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from app.config import DB_PATH, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL, RERANKER_MODEL


def _escape_fts5(query: str) -> str:
    return (
        query
        .replace('"', '""')
        .replace('+', '')
        .replace('~', '')
        .replace('(', '')
        .replace(')', '')
        .replace(':', '')
    )


def _is_chroma_healthy() -> bool:
    try:
        global _chroma_client, _collection
        if _chroma_client is None:
            import chromadb
            _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            _collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)
        _collection.count()
        return True
    except Exception:
        return False


def _fts5_search(question: str, n_results: int = 20) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma query_only = on")

    terms = re.findall(r'\w+', question)
    if not terms:
        conn.close()
        return {"documents": [], "metadatas": [], "distances": []}

    if len(terms) == 1:
        fts_query = _escape_fts5(terms[0])
    else:
        escaped = [_escape_fts5(t) for t in terms]
        fts_query = f'NEAR({" ".join(escaped)}, 12)'

    rows = conn.execute("""
        SELECT book_number, book_title, chapter_number, chapter_title,
               pov, paragraph_index, text
        FROM paragraphs
        WHERE paragraphs MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (fts_query, n_results)).fetchall()

    conn.close()

    documents, metadatas = [], []
    for r in rows:
        documents.append(r["text"])
        metadatas.append({
            "book_number": r["book_number"],
            "book_title": r["book_title"],
            "chapter_number": r["chapter_number"],
            "chapter_title": r["chapter_title"],
            "pov": r["pov"],
            "paragraph_index": r["paragraph_index"],
        })

    distances = [1.0 / (i + 1) for i in range(len(documents))]
    return {"documents": documents, "metadatas": metadatas, "distances": distances}


_bm25_cache = None
_sentence_model = None
_reranker = None
_chroma_client = None
_collection = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def _init_models():
    global _sentence_model, _reranker, _chroma_client, _collection
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(EMBEDDING_MODEL)
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)


def _init_bm25():
    global _bm25_cache
    total = _collection.count()
    if total == 0:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' está vazia. "
            "Execute 'python scripts/embed_paragraphs.py --rebuild' primeiro."
        )
    texts, ids, metadatas = [], [], []
    offset = 0
    limit = 10000
    while offset < total:
        batch = _collection.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )
        if not batch["ids"]:
            break
        texts.extend(batch["documents"])
        ids.extend(batch["ids"])
        metadatas.extend(batch["metadatas"])
        offset += limit
    tokenized = [_tokenize(t) for t in texts]
    from rank_bm25 import BM25Okapi
    _bm25_cache = (BM25Okapi(tokenized), texts, ids, metadatas)


def search(question: str, n_results: int = 20) -> dict:
    global _bm25_cache, _sentence_model, _reranker, _chroma_client, _collection

    if not _is_chroma_healthy():
        return _fts5_search(question, n_results)

    if _sentence_model is None:
        _init_models()

    if _bm25_cache is None:
        _init_bm25()
    bm25_index, all_texts, all_ids, all_metadatas = _bm25_cache
    id_to_idx = {doc_id: i for i, doc_id in enumerate(all_ids)}

    query_emb = _sentence_model.encode(
        f"Represent this sentence for searching relevant passages: {question}"
    ).tolist()

    sem = _collection.query(
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
    ce_scores = _reranker.predict(pairs)

    combined = sorted(
        zip(ce_scores, docs, metas, dists),
        key=lambda x: x[0], reverse=True,
    )

    return {
        "documents": [item[1] for item in combined[:n_results]],
        "metadatas": [item[2] for item in combined[:n_results]],
        "distances": [item[3] for item in combined[:n_results]],
    }
