import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = os.getenv("DB_PATH", "../backend/database.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GENERATION_MODEL = "gemini-2.0-flash"
DEEPSEEK_MODEL = "deepseek-v4-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"

COLLECTION_NAME = "asoiaf_paragraphs"
BATCH_SIZE = 100