import os
import sys
import chromadb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.config import CHROMA_PATH, COLLECTION_NAME

def main():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count > 0:
            print(f"[STARTUP] chroma_store pronto ({count} chunks).", flush=True)
        else:
            print(f"[STARTUP] AVISO: chroma_store vazio.", flush=True)
    except Exception as e:
        print(f"[STARTUP] AVISO: chroma_store nao disponivel: {e}", flush=True)

    print("[STARTUP] Iniciando uvicorn...", flush=True)
    os.execvp(sys.executable, [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", os.getenv("PORT", "7860")])

if __name__ == "__main__":
    main()
