# Technical Decisions

## Why Ollama + Llama 3.2 Instead of OpenAI or Gemini

I initially built this with Google Gemini's API. It worked well but I ran into free tier rate limits during development and testing. Rather than pay for API access for a portfolio project, I switched to Ollama which runs Llama 3.2 entirely on my local machine.

This turned out to be a better architectural choice. In production, many companies process sensitive documents — contracts, financial reports, medical records — where sending data to a third-party API raises compliance concerns. A local-first architecture eliminates that problem entirely. It also eliminates API costs and rate limiting. The tradeoff is that a 1B parameter model is less capable than GPT-4, but for document Q&A where the answer is in the retrieved context, it performs surprisingly well.

## Why ChromaDB Over Pinecone or FAISS

ChromaDB runs locally without any server setup, which aligns with the local-first architecture. Pinecone is cloud-hosted which introduces the same data privacy concerns I was trying to avoid. FAISS is powerful but requires more manual setup for persistence and metadata handling. ChromaDB gives me the simplicity of a local solution with proper persistence and metadata support out of the box.

## Why LangChain

LangChain is the most widely adopted framework for LLM applications. It appears in the majority of RAG-related job postings. Using it demonstrates familiarity with the standard tooling that production teams use. I used the latest modular architecture (langchain-core, langchain-community, langchain-ollama) rather than the older monolithic package.

## Chunking Strategy

I chose a chunk size of 1000 characters with 200-character overlap after experimenting with different sizes. Smaller chunks (500 chars) led to fragments that lacked enough context for meaningful answers. Larger chunks (2000 chars) reduced retrieval precision because the chunks covered too many topics. The 200-character overlap ensures that if a key sentence falls at a chunk boundary, it appears in both chunks and won't be missed during retrieval.

## Prompt Engineering

The system prompt explicitly instructs the LLM to only answer from the provided context and to say "I couldn't find this" when the answer isn't there. This is critical for reducing hallucinations. Without these constraints, the model would happily answer questions using its training data, which defeats the purpose of RAG.

## What I Learned

Building this project taught me that dataset and API issues are a normal part of development. I went through three Gemini API keys, dealt with package version conflicts between LangChain and its submodules, and had to restructure the code when LangChain's import paths changed in newer versions. These are exactly the kinds of problems you face in production, and working through them is part of the skill set.