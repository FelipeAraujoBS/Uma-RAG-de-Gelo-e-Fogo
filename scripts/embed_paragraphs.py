import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import sqlite3
import time
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import (
    DB_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    BATCH_SIZE
)

# ── clientes ──────────────────────────────────────────────
print("Carregando modelo de embeddings...")
model = SentenceTransformer(EMBEDDING_MODEL)

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(COLLECTION_NAME)

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

# ── stage 2: batching ─────────────────────────────────────
def make_batches(paragraphs: list[dict]) -> list[list[dict]]:
    return [
        paragraphs[i : i + BATCH_SIZE]
        for i in range(0, len(paragraphs), BATCH_SIZE)
    ]

# ── stage 3: embeddings ───────────────────────────────────
def embed_batch(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()

# ── stage 4: checkpoint ───────────────────────────────────
def filter_new(batch: list[dict]) -> list[dict]:
    ids = [build_id(p) for p in batch]
    existing = collection.get(ids=ids)["ids"]
    existing_set = set(existing)
    return [p for p in batch if build_id(p) not in existing_set]

# ── helpers ───────────────────────────────────────────────
def build_id(paragraph: dict) -> str:
    return f"b{paragraph['book_number']}_c{paragraph['chapter_number']}_p{paragraph['paragraph_index']}"

def build_metadata(paragraph: dict) -> dict:
    return {
        "book_number":     int(paragraph["book_number"] or 0),
        "book_title":      str(paragraph["book_title"] or ""),
        "chapter_number":  int(paragraph["chapter_number"] or 0),
        "chapter_title":   str(paragraph["chapter_title"] or ""),
        "pov":             str(paragraph["pov"] or ""),
        "paragraph_index": int(paragraph["paragraph_index"] or 0)
    }

# ── stage 5: pipeline principal ───────────────────────────
def main():
    print("Carregando parágrafos do SQLite...")
    paragraphs = load_paragraphs()
    print(f"{len(paragraphs)} parágrafos carregados.")

    batches = make_batches(paragraphs)
    print(f"{len(batches)} batches de {BATCH_SIZE} parágrafos.")

    for i, batch in enumerate(batches):
        new = filter_new(batch)

        if not new:
            print(f"Batch {i+1}/{len(batches)} — já existente, pulando.")
            continue

        texts     = [p["text"] for p in new]
        ids       = [build_id(p) for p in new]
        metadatas = [build_metadata(p) for p in new]

        embeddings = embed_batch(texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"Batch {i+1}/{len(batches)} — {len(new)} parágrafos embedados.")

    print("Concluído.")

if __name__ == "__main__":
    main()