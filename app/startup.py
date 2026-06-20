import os
import sys
import time
import subprocess
import chromadb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.config import CHROMA_PATH, DB_PATH, COLLECTION_NAME

def chroma_has_data() -> tuple[bool, int]:
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        return count > 0, count
    except Exception:
        return False, 0

def ensure_database():
    if os.path.isfile(DB_PATH):
        print(f"[STARTUP] Banco encontrado em {DB_PATH}", flush=True)
        return
    db_url = os.getenv("DB_DOWNLOAD_URL")
    if db_url:
        print(f"[STARTUP] Baixando database de {db_url}...", flush=True)
        import urllib.request
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        urllib.request.urlretrieve(db_url, DB_PATH)
        print(f"[STARTUP] Database baixado com sucesso ({os.path.getsize(DB_PATH) / 1e6:.1f} MB)", flush=True)
        return
    print(f"[STARTUP] ERRO: {DB_PATH} não encontrado. Defina DB_DOWNLOAD_URL ou monte o arquivo.", flush=True)
    sys.exit(1)

def rebuild_chroma():
    print("[STARTUP] chroma_store vazio ou ausente. Reconstruindo a partir do SQLite...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "scripts/embed_paragraphs.py", "--rebuild"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True, text=True
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(f"[STARTUP] ERRO na reconstrução:\n{result.stderr}", flush=True)
        raise RuntimeError("Falha ao reconstruir chroma_store")
    print(f"[STARTUP] Reconstrução concluída em {time.time() - t0:.1f}s", flush=True)

def main():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    has_data, count = chroma_has_data()
    if not has_data:
        ensure_database()
        rebuild_chroma()
    else:
        print(f"[STARTUP] chroma_store já populado ({count} chunks). Pulando rebuild.", flush=True)

    print("[STARTUP] Iniciando uvicorn...", flush=True)
    os.execvp(sys.executable, [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", os.getenv("PORT", "7860")])

if __name__ == "__main__":
    main()
