import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

# ── clientes ──────────────────────────────────────────────
model = SentenceTransformer(EMBEDDING_MODEL)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(COLLECTION_NAME)

# ── busca ─────────────────────────────────────────────────
def search(question: str, n_results: int = 5):
    embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        print(f"\n── Resultado {i+1} ──────────────────────────────")
        print(f"Livro:     {meta['book_title']}")
        print(f"Capítulo:  {meta['chapter_title']}")
        print(f"POV:       {meta['pov']}")
        print(f"Distância: {dist:.4f}")
        print(f"Texto:     {doc[:300]}...")

if __name__ == "__main__":
    question = input("Pergunta: ")
    search(question)