import os
import sys
import asyncio
import logging
import chromadb
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request

sys.path.append(os.path.dirname(__file__))

from app.config import CHROMA_PATH, COLLECTION_NAME, GROQ_API_KEY, API_KEY
from app.routes.chat import router as chat_router, _merge_results
from app.services.query_expansion import expand_queries
from app.services.retrieval import search
from app.services.generation import generate

logger = logging.getLogger("rag_app")

# Verificação inicial do ChromaDB
os.makedirs(CHROMA_PATH, exist_ok=True)
try:
    _client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = _client.get_collection(COLLECTION_NAME)
    _count = _collection.count()
    if _count > 0:
        print(f"[STARTUP] chroma_store pronto ({_count} chunks indexados).", flush=True)
    else:
        print("[STARTUP] AVISO: chroma_store encontrado mas está vazio.", flush=True)
except Exception as e:
    print(f"[STARTUP] AVISO: chroma_store indisponível ({e}). Sistema usará FTS5 fallback.", flush=True)


async def ask_rag(message: str, history: list) -> str:
    """Função de resposta para a interface Gradio."""
    if not message or not message.strip():
        return "Por favor, faça uma pergunta sobre as Crônicas de Gelo e Fogo."

    try:
        queries = await expand_queries(message)
        tasks = [asyncio.to_thread(search, q) for q in queries]
        all_results = await asyncio.gather(*tasks)
        results = _merge_results(all_results)

        if not results["documents"]:
            return "Não encontrei trechos relevantes nos livros para responder à sua pergunta."

        context = "\n\n".join(results["documents"])
        answer = await generate(message, context)

        # Adicionar fontes citadas de forma legível
        if results.get("metadatas"):
            sources_md = "\n\n---\n**📚 Fontes consultadas:**\n"
            seen = set()
            for meta in results["metadatas"]:
                key = f"{meta.get('book_title')} - {meta.get('chapter_title')} ({meta.get('pov')})"
                if key not in seen:
                    seen.add(key)
                    sources_md += f"- *{meta.get('book_title')}* — **{meta.get('chapter_title')}** (POV: {meta.get('pov')})\n"
            answer += sources_md

        return answer
    except Exception as e:
        logger.error(f"Erro ao processar consulta: {e}", exc_info=True)
        return f"❌ Desculpe, ocorreu um erro ao consultar os pergaminhos: {str(e)}"


# Construção da interface Gradio
with gr.Blocks(title="Uma RAG de Gelo e Fogo") as demo:
    gr.Markdown(
        """
        # 🐺 Uma RAG de Gelo e Fogo
        Consulte os meistres da Cidadela! Faça perguntas em linguagem natural sobre os livros de *As Crônicas de Gelo e Fogo*.
        *Busca Híbrida (Dense + BM25 + Lightweight Reranker) & LLM Llama 3 via Groq.*
        """
    )
    gr.ChatInterface(
        fn=ask_rag,
        cache_examples=False,
        examples=[
            "Quem matou o Rei Louco?",
            "Quais são as palavras da Casa Stark?",
            "O que aconteceu no Casamento Vermelho?",
            "Quem é o príncipe prometido segundo Melisandre?",
        ],
    )

# 1. Configurar CORS na aplicação FastAPI interna do Gradio
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Middleware de autenticação opcional para rotas /api/*
@demo.app.middleware("http")
async def validate_api_key(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if API_KEY and request.url.path.startswith("/api/"):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth.removeprefix("Bearer ") != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"error": "Não autorizado. Header 'Authorization: Bearer <token>' é obrigatório."},
            )
    return await call_next(request)

# 3. Incluir endpoints REST para o Backend Node.js
demo.app.include_router(chat_router, prefix="/api")

@demo.app.get("/health")
async def health():
    chroma_ok = False
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.heartbeat()
        chroma_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "chromadb": chroma_ok,
        "groq_configured": bool(GROQ_API_KEY),
        "auth_enabled": bool(API_KEY),
    }

# 4. Iniciar via demo.launch() — o método nativo esperado pelo Hugging Face Spaces
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
