from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os


def create_rag_chain(vector_store):
    """Create a simple RAG chain."""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant that answers questions based on the provided document content.

RULES:
1. ONLY answer based on the context provided below. Do not use outside knowledge.
2. If the answer is not in the context, say "I couldn't find this information in the uploaded document."
3. Always cite which part of the document your answer comes from.
4. Keep answers clear, concise, and well-structured.

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

ANSWER:""")
    
    return llm, retriever, prompt


def ask_question(llm, retriever, prompt, question):
    """Ask a question and get answer with sources."""
    
    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    
    # Build context from chunks
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Get answer from LLM
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    # Extract sources
    source_texts = []
    for doc in docs:
        snippet = doc.page_content[:200].strip()
        if snippet not in source_texts:
            source_texts.append(snippet)
    
    return answer, source_texts


if __name__ == "__main__":
    print("RAG chain module ready.")