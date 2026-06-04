---
title: Uma RAG de Gelo e Fogo
emoji: 🐺
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# Uma RAG de Gelo e Fogo

**RAG (Retrieval-Augmented Generation) microservice** for *A Song of Ice and Fire*. Answers natural language questions about George R. R. Martin's books by combining hybrid retrieval (dense + sparse + reranking) with LLM generation via Groq.

Part of the **A Procura de Gelo e Fogo** ecosystem — a full-text search and Q&A platform across 10 ASOIAF books.

---

## Table of Contents

- [Architecture](#architecture)
- [Hybrid Retrieval Pipeline](#hybrid-retrieval-pipeline)
- [Generation](#generation)
- [Fallback Mechanism](#fallback-mechanism)
- [API](#api)
- [Embedding Pipeline](#embedding-pipeline)
- [Evaluation](#evaluation)
- [Environment Variables](#environment-variables)
- [Deploy](#deploy)
- [Tech Stack](#tech-stack)

---

## Architecture

```
POST /api/chat { question }
        │
        ▼
┌─────────────────────────────────────┐
│      Hybrid Search (retrieval.py)    │
│                                     │
│  ┌──────────┐  ┌─────────────────┐  │
│  │ ChromaDB │  │  BM25 (sparse)  │  │
│  │  dense   │  │  rank-bm25      │  │
│  │ bge-m3   │  │  tokenize +     │  │
│  │ 768-dim  │  │  score          │  │
│  └────┬─────┘  └────────┬────────┘  │
│       │                 │           │
│       └──────┬──────────┘           │
│              ▼                      │
│      RRF Fusion (K=60)              │
│              │                      │
│              ▼                      │
│    Cross-encoder Rerank             │
│   (bge-reranker-v2-m3)             │
│              │                      │
│              ▼                      │
│    Top 20 chunks + metadata         │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       Generation (generation.py)     │
│                                     │
│  Prompt: context + question          │
│  → Groq Llama 3.3 70B               │
│                                     │
│  Returns: { answer, sources[] }      │
└─────────────────────────────────────┘
```

### Intelligent Fallback

If ChromaDB is unavailable, the system falls back to **FTS5 search directly on the SQLite database** (same database used by the backend), ensuring the service works even without the vector store.

---

## Hybrid Retrieval Pipeline

The retrieval combines three techniques for maximum precision and recall:

### 1. Dense Retrieval (Semantic)

| Component | Detail |
|-----------|--------|
| **Model** | `BAAI/bge-m3` (768-dim, multilingual) |
| **Vector DB** | ChromaDB (persistent, cosine similarity) |
| **Query prefix** | `"Represent this sentence for searching relevant passages: {question}"` |
| **Top K** | 60 |

### 2. Sparse Retrieval (Lexical)

| Component | Detail |
|-----------|--------|
| **Algorithm** | BM25 (Okapi) |
| **Tokenization** | Regex `\w+`, lowercase |
| **Top K** | 60 |

### 3. Fusion — RRF (Reciprocal Rank Fusion)

- K = 60 (smoothing parameter)
- Penalizes low-ranked results from either method
- Produces top 40 candidates

### 4. Reranking — Cross-Encoder

- Model: `BAAI/bge-reranker-v2-m3`
- Scores (question, chunk) pairs with cross-attention
- More accurate than embedding cosine similarity
- Final output: top 20 chunks

---

## Generation

### Prompt Strategy

The generation prompt instructs the LLM to:

1. Use retrieved context as the primary source
2. Answer directly first (1-2 sentences max)
3. Only cite context when the question asks for evidence
4. If context lacks the answer, use general knowledge but note it

### Model

- **Provider**: Groq Cloud
- **Model**: `llama-3.3-70b-versatile`
- **SDK**: OpenAI-compatible client

---

## Fallback Mechanism

```
ChromaDB healthy?
  ├─ Yes → Hybrid Search (dense + sparse + RRF + rerank)
  └─ No  → FTS5 fallback on SQLite
              ├─ NEAR operator for multi-term queries
              ├─ Single term for exact matching
              └─ ORDER BY rank (FTS5 relevance)
```

The FTS5 fallback mimics the search engine used by the backend service, ensuring consistency.

---

## API

### `POST /api/chat`

**Request:**
```json
{
  "question": "Who killed the Mad King?"
}
```

**Response:**
```json
{
  "answer": "Jaime Lannister.",
  "sources": [
    {
      "book": "A Tormenta de Espadas",
      "chapter": "Jaime VIII",
      "pov": "Jaime Lannister",
      "distance": 0.321
    }
  ]
}
```

**Error Response:**
```json
{
  "detail": "Error description"
}
```

---

## Embedding Pipeline

Run `scripts/embed_paragraphs.py` to embed the SQLite paragraphs into ChromaDB:

```
SQLite paragraphs
  → NLTK sent_tokenize (Portuguese)
  → Sliding window: 5 sentences, stride 3
  → Batching: 100 chunks per batch
  → bge-m3 embedding
  → ChromaDB persist (with ID-based checkpointing)
```

**Usage:**
```bash
python scripts/embed_paragraphs.py           # Incremental (skips existing IDs)
python scripts/embed_paragraphs.py --rebuild  # Rebuild from scratch
```

---

## Evaluation

The `scripts/run_eval.py` script evaluates the RAG pipeline with 18 questions about ASOIAF using LLM-as-Judge metrics:

| Metric | Description |
|--------|-------------|
| **Context Precision** | Cosine similarity between query and chunk embeddings |
| **Answer Relevancy** | LLM generates derived questions → cosine similarity with original query |
| **Context Recall** | LLM extracts claims from ground truth → verifies each against context |
| **Faithfulness** | LLM extracts claims from answer → verifies each against context |

**Eval models:** `llama-3.1-8b-instant` via Groq

**Questions file:** `eval/questions.json`

**Results:** saved incrementally to `eval/results.json`

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key for LLM inference |
| `DB_PATH` | No | `../backend/database.db` | Path to SQLite database (FTS5 fallback) |
| `CHROMA_PATH` | No | `./chroma_store` | Path to persisted ChromaDB |

### Config (`app/config.py`)

| Setting | Value |
|---------|-------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` |
| `COLLECTION_NAME` | `asoiaf_paragraphs` |
| `BATCH_SIZE` | 100 |

---

## Deploy

### Docker

```bash
docker build -t uma-rag-de-gelo-e-fogo .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key uma-rag-de-gelo-e-fogo
```

### Hugging Face Spaces

The project includes a GitHub Action (`.github/workflows/deploy-hf.yml`) for automatic deployment to Hugging Face Spaces on push to `master`.

### Docker Compose

The project is designed to work with the parent `docker-compose.yml` as part of the three-service ecosystem:

```yaml
services:
  backend:   # A-Procura-de-Gelo-e-Fogo-Backend
  frontend:  # A-Procura-de-Gelo-e-Fogo-Frontend
  rag:       # Uma-RAG-de-Gelo-e-Fogo
```

---

## Tech Stack

| Technology | Role |
|-----------|------|
| **Python 3.12** | Runtime |
| **FastAPI** | HTTP server |
| **ChromaDB** | Vector database |
| **sentence-transformers** | Embeddings + Cross-encoder |
| **rank-bm25** | Sparse retrieval |
| **NLTK** | Sentence tokenization (Portuguese) |
| **Groq API** | LLM inference (Llama 3.3 70B) |
| **Uvicorn** | ASGI server |
| **Docker** | Containerization |

---

## Related Projects

- [A-Procura-de-Gelo-e-Fogo-Backend](https://github.com/FelipeAraujoBS/search) — Fastify + SQLite FTS5 search API
- [A-Procura-de-Gelo-e-Fogo-Frontend](https://github.com/FelipeAraujoBS/search) — Next.js search & chat interface

---

> Designed and developed by [FelipeAraujoBS](https://github.com/FelipeAraujoBS)
