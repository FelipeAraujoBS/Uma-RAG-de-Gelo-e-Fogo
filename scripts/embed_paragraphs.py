import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import sqlite3
import nltk
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import (
    DB_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    BATCH_SIZE
)

nltk.download("punkt_tab", quiet=True)

SENTENCE_WINDOW = 5
SENTENCE_STRIDE = 3

# ── clientes ──────────────────────────────────────────────
print("Carregando modelo de embeddings...")
model = SentenceTransformer(EMBEDDING_MODEL)

chroma = chromadb.PersistentClient(path=CHROMA_PATH)

# ── stage 1: leitura ──────────────────────────────────────
def load_paragraphs() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT book_number, book_title, chapter_number,
               chapter_title, pov, paragraph_index, text
        FROM paragraphs
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# ── stage 2: chunking ─────────────────────────────────────
def chunk_paragraph(text: str) -> list[str]:
    sentences = nltk.sent_tokenize(text, language="portuguese")
    if len(sentences) <= SENTENCE_WINDOW:
        return [text]

    chunks = []
    for i in range(0, len(sentences) - SENTENCE_WINDOW + 1, SENTENCE_STRIDE):
        chunk = " ".join(sentences[i : i + SENTENCE_WINDOW])
        chunks.append(chunk)
    return chunks


def expand_paragraphs(paragraphs: list[dict]) -> list[dict]:
    expanded = []
    for p in paragraphs:
        chunks = chunk_paragraph(p["text"])
        for i, text in enumerate(chunks):
            entry = {**p, "text": text, "chunk_index": i}
            expanded.append(entry)
    return expanded

# ── stage 3: batching ─────────────────────────────────────
def make_batches(chunks: list[dict]) -> list[list[dict]]:
    return [chunks[i : i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

# ── stage 4: embeddings ───────────────────────────────────
def embed_batch(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()

# ── stage 5: checkpoint ───────────────────────────────────
def filter_new(batch: list[dict], collection) -> list[dict]:
    ids = [build_id(c) for c in batch]
    existing = collection.get(ids=ids)["ids"]
    existing_set = set(existing)
    return [c for c in batch if build_id(c) not in existing_set]

# ── helpers ───────────────────────────────────────────────
def build_id(chunk: dict) -> str:
    return (
        f"b{chunk['book_number']}_c{chunk['chapter_number']}"
        f"_p{chunk['paragraph_index']}_g{chunk['chunk_index']}"
    )

def build_metadata(chunk: dict) -> dict:
    return {
        "book_number":     int(chunk["book_number"] or 0),
        "book_title":      str(chunk["book_title"] or ""),
        "chapter_number":  int(chunk["chapter_number"] or 0),
        "chapter_title":   str(chunk["chapter_title"] or ""),
        "pov":             str(chunk["pov"] or ""),
        "paragraph_index": int(chunk["paragraph_index"] or 0),
        "chunk_index":     int(chunk["chunk_index"] or 0),
    }

# ── stage 6: pipeline principal ───────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Remove coleção existente antes de re-embedar")
    args = parser.parse_args()

    collection = chroma.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    if args.rebuild:
        print("Limpando coleção existente...")
        n_total = collection.count()
        if n_total:
            collection.delete(collection.get()["ids"])
            print(f"Removidos {n_total} chunks antigos.\n")
    print("Carregando parágrafos do SQLite...")
    paragraphs = load_paragraphs()
    print(f"{len(paragraphs)} parágrafos carregados.")

    print("Expandindo parágrafos em chunks de sentenças...")
    chunks = expand_paragraphs(paragraphs)
    print(f"{len(chunks)} chunks gerados (window={SENTENCE_WINDOW}, stride={SENTENCE_STRIDE}).")

    batches = make_batches(chunks)
    print(f"{len(batches)} batches de {BATCH_SIZE} chunks.")

    for i, batch in enumerate(batches):
        new = filter_new(batch, collection)

        if not new:
            print(f"Batch {i+1}/{len(batches)} — já existente, pulando.")
            continue

        texts     = [c["text"] for c in new]
        ids       = [build_id(c) for c in new]
        metadatas = [build_metadata(c) for c in new]

        embeddings = embed_batch(texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"Batch {i+1}/{len(batches)} — {len(new)} chunks embedados.")

    print("Concluído.")

if __name__ == "__main__":
    main()
