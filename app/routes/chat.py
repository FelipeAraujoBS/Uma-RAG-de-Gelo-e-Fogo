import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.services.retrieval import search
from app.services.generation import generate
from app.limiter import limiter

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    book: str
    chapter: str
    pov: str
    distance: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: ChatRequest, request_obj: Request):
    try:
        results = await asyncio.to_thread(search, request.question)
        context = "\n\n".join(results["documents"])
        sources = [
            Source(
                book=meta["book_title"],
                chapter=meta["chapter_title"],
                pov=meta["pov"],
                distance=dist,
            )
            for meta, dist in zip(results["metadatas"], results["distances"])
        ]
        answer = await generate(request.question, context)
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
