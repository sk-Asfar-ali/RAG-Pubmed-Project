import urllib.error
from typing import List, Dict
from Bio import Entrez
from langchain_core.documents import Document
from src.config import ENTREZ_EMAIL, MAX_RESULTS_TO_FETCH, PUBMED_API_KEY

# Set up Entrez email and API key
Entrez.email = ENTREZ_EMAIL
if PUBMED_API_KEY:
    Entrez.api_key = PUBMED_API_KEY

def search_pubmed(query: str, max_results: int = MAX_RESULTS_TO_FETCH) -> List[str]:
    """Search PubMed for a given query and return a list of article IDs."""
    try:
        print(f"Searching PubMed for: '{query}'")
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return []

def fetch_pubmed_details(id_list: List[str]) -> List[Document]:
    """Fetch article details (title, abstract) for a given list of PubMed IDs."""
    if not id_list:
        return []

    try:
        print(f"Fetching details for {len(id_list)} articles...")
        handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="xml", retmode="text")
        records = Entrez.read(handle)
        handle.close()
        
        documents = []
        for pubmed_article in records['PubmedArticle']:
            medline_citation = pubmed_article['MedlineCitation']
            article = medline_citation['Article']
            
            pmid = str(medline_citation['PMID'])
            title = article.get('ArticleTitle', 'No Title Available')
            
            # Extract abstract text
            abstract_text = ""
            if 'Abstract' in article and 'AbstractText' in article['Abstract']:
                abstract_parts = article['Abstract']['AbstractText']
                # Sometimes abstract_parts is a string, sometimes a list of strings
                if isinstance(abstract_parts, list):
                    abstract_text = " ".join([str(part) for part in abstract_parts])
                else:
                    abstract_text = str(abstract_parts)
            else:
                abstract_text = "No abstract available."
            
            # Combine title and abstract as the main content
            page_content = f"Title: {title}\n\nAbstract:\n{abstract_text}"
            
            # Store metadata
            metadata = {
                "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "pmid": pmid,
                "title": title
            }
            
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)
            
        return documents
    except urllib.error.HTTPError as e:
        print(f"HTTP Error fetching from PubMed: {e}")
        return []
    except Exception as e:
        print(f"Error fetching article details: {e}")
        return []

def load_from_pubmed(query: str, max_results: int = MAX_RESULTS_TO_FETCH) -> List[Document]:
    """End-to-end pipeline to search and fetch documents from PubMed."""
    article_ids = search_pubmed(query, max_results)
    if not article_ids:
        print("No articles found.")
        return []
    
    documents = fetch_pubmed_details(article_ids)
    print(f"Successfully loaded {len(documents)} articles from PubMed.")
    return documents

if __name__ == "__main__":
    # Test the loader
    docs = load_from_pubmed("Metformin Type 2 Diabetes", max_results=2)
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Source: {doc.metadata['source']}")
        print(f"Title: {doc.metadata['title']}")
        print(f"Content snippet: {doc.page_content[:150]}...")
