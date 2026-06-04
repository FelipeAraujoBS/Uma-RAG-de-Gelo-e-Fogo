import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = os.getenv("DB_PATH", "../backend/database.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GROQ_MODEL = "llama-3.3-70b-versatile"

COLLECTION_NAME = "asoiaf_paragraphs"
BATCH_SIZE = 100