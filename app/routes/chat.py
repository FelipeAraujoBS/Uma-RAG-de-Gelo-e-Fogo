import asyncio
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.services.retrieval import search
from app.services.generation import generate
from app.services.query_expansion import expand_queries
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


def _merge_results(results_list: list[dict], top_n: int = 20) -> dict:
    seen = set()
    merged_docs, merged_metas, merged_dists = [], [], []
    for results in results_list:
        for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
            key = f"{meta['book_title']}|{meta['chapter_title']}|{doc[:80]}"
            if key not in seen:
                seen.add(key)
                merged_docs.append(doc)
                merged_metas.append(meta)
                merged_dists.append(dist)
    return {
        "documents": merged_docs[:top_n],
        "metadatas": merged_metas[:top_n],
        "distances": merged_dists[:top_n],
    }


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(body: ChatRequest, request: Request):
    t0 = time.time()
    try:
        t1 = time.time()
        queries = await expand_queries(body.question)
        print(f"[TIMING] expand_queries = {time.time() - t1:.2f}s | {queries}", flush=True)

        tasks = [asyncio.to_thread(search, q) for q in queries]
        all_results = await asyncio.gather(*tasks)
        results = _merge_results(all_results)
        print(f"[TIMING] retrieval.search() x{len(queries)} = {time.time() - t1:.2f}s | {len(results['documents'])} chunks", flush=True)

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

        t2 = time.time()
        answer = await generate(body.question, context)
        print(f"[TIMING] generate() (Groq) = {time.time() - t2:.2f}s", flush=True)

        print(f"[TIMING] TOTAL = {time.time() - t0:.2f}s", flush=True)
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        print(f"[TIMING] ERRO em {time.time() - t0:.2f}s: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
