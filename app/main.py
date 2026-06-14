from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.routes.chat import router as chat_router
from app.config import API_KEY
import chromadb
from app.config import CHROMA_PATH, GROQ_API_KEY
from app.limiter import limiter

app = FastAPI(title="Uma RAG de Gelo e Fogo")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def validate_api_key(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth.removeprefix("Bearer ") != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"error": "Não autorizado. Header 'Authorization: Bearer <token>' é obrigatório."},
            )
    return await call_next(request)

app.include_router(chat_router, prefix="/api")

@app.get("/health")
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
