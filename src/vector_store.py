from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os


def create_vector_store(chunks, persist_directory="data/chroma_db"):
    """Create a ChromaDB vector store from text chunks."""
    # Use Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create vector store
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store created with {len(chunks)} chunks")
    return vector_store


def load_vector_store(persist_directory="data/chroma_db"):
    """Load an existing vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vector_store


def search_similar(vector_store, query, k=4):
    """Find the most relevant chunks for a query."""
    results = vector_store.similarity_search_with_score(query, k=k)
    
    for doc, score in results:
        print(f"Score: {score:.4f} | {doc.page_content[:100]}...")
    
    return results


if __name__ == "__main__":
    print("Vector store module ready.")