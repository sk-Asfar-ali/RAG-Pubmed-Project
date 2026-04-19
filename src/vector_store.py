import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIRECTORY

def get_embeddings_model():
    """Initialize and return the HuggingFace embeddings model."""
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    # Optional model_kwargs = {'device': 'cpu'} could be added here if needed
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vector_store() -> Chroma:
    """Get the Chroma vector store instance. Creates it if it doesn't exist."""
    embeddings = get_embeddings_model()
    
    # Initialize Chroma vector store with a persistence directory
    vector_store = Chroma(
        collection_name="pubmed_research",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    return vector_store

def add_documents_to_store(chunks: List[Document]):
    """Add document chunks to the Chroma vector store."""
    if not chunks:
        print("No chunks to add to the vector store.")
        return
    
    print(f"Adding {len(chunks)} chunks to ChromaDB...")
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print("Successfully stored embeddings in ChromaDB.")

def similarity_search(query: str, k: int = 3) -> List[Document]:
    """Search the vector store for chunks most similar to the query."""
    print(f"Performing similarity search for: '{query}'...")
    vector_store = get_vector_store()
    
    # Check if we have documents in the store
    if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
         print("Vector store does not exist yet. Please query PubMed first.")
         return []
         
    try:
        results = vector_store.similarity_search(query, k=k)
        print(f"Found {len(results)} relevant chunks.")
        return results
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []
