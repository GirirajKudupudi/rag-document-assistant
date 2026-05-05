from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def create_vector_store(chunks, persist_directory="data/chroma_db"):
    """Create a ChromaDB vector store from text chunks."""
    embeddings = OllamaEmbeddings(model="llama3.2:1b")

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"Vector store created with {len(chunks)} chunks")
    return vector_store


def load_vector_store(persist_directory="data/chroma_db"):
    """Load an existing vector store."""
    embeddings = OllamaEmbeddings(model="llama3.2:1b")

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vector_store