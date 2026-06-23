import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = os.getenv("DB_PATH", "../backend/database.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")
API_KEY = os.getenv("API_KEY")

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GROQ_MODEL = "qwen/qwen3.6-27b"

RERANKER_MODE = os.getenv("RERANKER_MODE", "lightweight")
RERANKER_WEIGHTS_PATH = os.getenv("RERANKER_WEIGHTS_PATH", "reranker_weights.json")

RERANK_WEIGHT_BM25 = os.getenv("RERANK_WEIGHT_BM25")
RERANK_WEIGHT_DENSE = os.getenv("RERANK_WEIGHT_DENSE")
RERANK_WEIGHT_RRF = os.getenv("RERANK_WEIGHT_RRF")

COLLECTION_NAME = "asoiaf_paragraphs"
BATCH_SIZE = 100

TOP_N_CONTEXT = int(os.getenv("TOP_N_CONTEXT", "5"))
CROSS_ENCODER_TOP_N = int(os.getenv("CROSS_ENCODER_TOP_N", "12"))