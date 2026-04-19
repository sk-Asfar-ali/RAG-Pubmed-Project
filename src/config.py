import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys and configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Please set it in the .env file.")
if not ENTREZ_EMAIL:
    raise ValueError("ENTREZ_EMAIL is missing. Please set it in the .env file. NCBI requires developers to provide an email.")

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# LLM Configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# PubMed Search Configuration
MAX_RESULTS_TO_FETCH = 3
