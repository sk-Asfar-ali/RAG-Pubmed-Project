# Medical Research Chatbot

A modular Medical Research Chatbot in Python that queries PubMed, processes articles into a vector database, and generates structured answers using a Gemini-powered LLM.

## Project Architecture

This project follows an Extract, Transform, Load (ETL) pattern for data ingestion and a Retrieval-Augmented Generation (RAG) pattern for querying.

### File Responsibilities

- **`config.py`**: Stores environment variables and constants (API keys, chunk sizes, model names).
- **`pubmed_loader.py`**: Handles PubMed API calls via BioPython (Entrez) to fetch research papers.
- **`processor.py`**: Uses LangChain's `RecursiveCharacterTextSplitter` to split text into manageable chunks.
- **`vector_store.py`**: Handles embeddings generation (Sentence Transformers) and ChromaDB initialization/similarity search.
- **`llm_engine.py`**: Manages the connection to Google Gemini via LangChain and constructs the RAG prompt.
- **`main.py`**: The terminal-based entry point for user input and workflow orchestration.

## Setup Instructions

1. **Install Dependencies**
   Ensure you have Python 3.9+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory (already done if following the guide). Make sure it contains:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   PUBMED_API_KEY=your_pubmed_api_key
   ENTREZ_EMAIL=your_email@example.com
   ```

3. **Running the Application**
   Start the interactive terminal application:
   ```bash
   python main.py
   ```

## Workflow Guide

1. **Data Pipeline (ETL)**: From the main menu, select option `1` to search PubMed. Enter a query like "Diabetes". The system will fetch the top articles, chunk their text, create embeddings, and store them in a local `./chroma_db` directory.
2. **RAG Logic (Query)**: Select option `2` and ask a question related to the data you ingested (e.g., "What are the treatments for diabetes?"). The system performs a vector search in ChromaDB, injects the relevant chunks into a hidden prompt, and asks Gemini to formulate a structured answer based *only* on that evidence.

## Important Notes
- The vector database is persisted locally in `./chroma_db`. If you search multiple topics, it will accumulate knowledge.
- If you encounter a `urllib.error.HTTPError: HTTP Error 429: Too Many Requests`, you have hit the PubMed API limit. Using a `PUBMED_API_KEY` increases the rate limit significantly.
