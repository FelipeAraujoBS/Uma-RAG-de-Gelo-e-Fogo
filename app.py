import os
import sys
import asyncio
import logging
import chromadb
import uvicorn
import gradio as gr

sys.path.append(os.path.dirname(__file__))

from app.config import CHROMA_PATH, COLLECTION_NAME, GROQ_API_KEY
from app.main import app
from app.services.query_expansion import expand_queries
from app.services.retrieval import search
from app.services.generation import generate
from app.routes.chat import _merge_results

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

# Monta a UI do Gradio na raiz da aplicação FastAPI mantendo os endpoints REST (/api/chat, /health)
app = gr.mount_gradio_app(app, demo, path="/")

# No Hugging Face Spaces, a plataforma já sobe o servidor na porta 7860 automaticamente.
# Executamos o uvicorn manualmente apenas em desenvolvimento local.
if __name__ == "__main__" and not os.getenv("SPACE_ID"):
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
