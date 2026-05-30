import os 
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_PATH = os.getenv("DB_PATH", "../A-Procura-de-Gelo-e-Fogo-Backend/database.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.0-flash"

COLLECTION_NAME = "asoiaf_paragraphs"
BATCH_SIZE = 100

