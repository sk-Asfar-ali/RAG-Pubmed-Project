import os
import sys
import warnings
import logging

# Suppress warnings for a clean terminal
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="huggingface_hub")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from src.pubmed_loader import load_from_pubmed
from src.processor import process_documents
from src.vector_store import add_documents_to_store, similarity_search
from src.llm_engine import generate_answer

def process_query(question: str):
    """End-to-End Pipeline: Search PubMed, Ingest, and Answer."""
    print(f"\n--- Searching PubMed for recent articles on: '{question}' ---")
    
    # 1. Extract
    documents = load_from_pubmed(question)
    
    if not documents:
        print("Could not find relevant articles on PubMed. Attempting to answer from existing knowledge base...")
    else:
        # 2. Transform
        chunks = process_documents(documents)
        
        # 3. Load
        add_documents_to_store(chunks)

    print("\n--- Generating Answer ---")
    # 4. Retrieve
    retrieved_docs = similarity_search(question)
    
    if not retrieved_docs:
        print("No context found in the database. Please try a different question.")
        return
        
    # 5. Generate
    answer = generate_answer(question, retrieved_docs)
    
    print("\n" + "="*50)
    print("Medical Chatbot Response:")
    print("="*50)
    print(answer)
    print("="*50 + "\n")

def main_loop():
    """Terminal-based UI loop."""
    print("Welcome to the Medical Research Chatbot!")
    print("Type your medical question. The chatbot will search PubMed, process the articles, and answer it.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            question = input("Enter your medical question: ").strip()
            
            if question.lower() == 'exit':
                print("Exiting. Goodbye!")
                sys.exit(0)
            elif question:
                process_query(question)
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    # Ensure keys are loaded via config
    import src.config
    main_loop()
