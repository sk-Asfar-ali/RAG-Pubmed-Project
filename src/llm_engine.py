from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_MODEL_NAME, GOOGLE_API_KEY

def get_llm():
    """Initialize and return the Google Gemini LLM."""
    print(f"Initializing LLM: {GEMINI_MODEL_NAME}...")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2 # Lower temperature for more factual responses
    )

def generate_answer(query: str, retrieved_docs: List[Document]) -> str:
    """Generate an answer using the LLM based on retrieved context."""
    if not retrieved_docs:
        return "I could not find any relevant context in the database to answer your question."

    # Format the context from retrieved documents
    context_text = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nTitle: {doc.metadata.get('title', 'No Title')}\nContent: {doc.page_content}" for doc in retrieved_docs])
    
    # Extract unique citations
    citations = set([doc.metadata.get('source') for doc in retrieved_docs if doc.metadata.get('source')])
    citations_text = "\n".join([f"- {url}" for url in citations])

    # Define the RAG prompt template
    prompt_template = """
    You are a Medical Research Chatbot specializing in summarizing scientific literature.
    Use ONLY the following research snippets to answer the user's question. If the snippets don't contain the answer, say "I don't have enough information to answer this based on the retrieved documents."
    Do NOT use outside knowledge.

    Retrieved Research Snippets:
    {context}

    Question:
    {question}

    Output Format:
    **Key Findings:**
    [A concise summary of the findings addressing the user's question]

    **Detailed Explanation:**
    [More detailed points based on the context]

    **Citations:**
    {citations}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "citations"]
    )
    
    # Initialize the LLM
    llm = get_llm()
    
    # Format the prompt
    formatted_prompt = prompt.format(
        context=context_text,
        question=query,
        citations=citations_text
    )
    
    print("Generating answer with Gemini...")
    # Generate the response
    try:
        response = llm.invoke(formatted_prompt)
        return response.content
    except Exception as e:
        return f"Error generating answer from LLM: {e}"
