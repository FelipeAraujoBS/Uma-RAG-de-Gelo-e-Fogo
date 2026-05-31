from sentence_transformers import SentenceTransformer
import chromadb
from app.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(COLLECTION_NAME)


def search(question: str, n_results: int = 5) -> dict:
    embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n_results)

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }
