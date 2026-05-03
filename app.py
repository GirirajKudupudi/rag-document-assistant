import streamlit as st
import os
import tempfile
from src.document_processor import extract_text_from_pdf, split_text_into_chunks
from src.vector_store import create_vector_store
from src.rag_chain import create_rag_chain, ask_question

st.set_page_config(
    page_title='RAG Document Assistant',
    page_icon='📄',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background: #0a0e17; color: #e0e6ed; }
    [data-testid="stSidebar"] { background: #141b2d; border-right: 2px solid rgba(102,126,234,0.15); }
    
    .main-title {
        font-size: 2.5rem; font-weight: 900;
        background: linear-gradient(135deg, #667eea, #a78bfa, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 5px;
    }
    .sub-title {
        color: #6b7c93; text-align: center; font-size: 1rem; margin-bottom: 25px;
    }
    
    .chat-user {
        background: rgba(102,126,234,0.12);
        border: 1px solid rgba(102,126,234,0.25);
        border-radius: 14px; padding: 15px 20px; margin: 10px 0;
        color: #c4d0ff;
    }
    .chat-ai {
        background: rgba(46,204,113,0.08);
        border: 1px solid rgba(46,204,113,0.2);
        border-radius: 14px; padding: 15px 20px; margin: 10px 0;
        color: #e0e6ed; line-height: 1.7;
    }
    .source-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 12px 15px; margin: 5px 0;
        font-size: 0.82rem; color: #7a8ba5;
        border-left: 3px solid #667eea;
    }
    
    .status-card {
        background: #141b2d;
        border-radius: 12px; padding: 15px 20px;
        border: 1px solid rgba(255,255,255,0.05);
        text-align: center;
    }
    .status-num { font-size: 1.8rem; font-weight: 800; color: #667eea; }
    .status-label { font-size: 0.8rem; color: #7a8ba5; margin-top: 3px; }
    
    .upload-area {
        border: 2px dashed rgba(102,126,234,0.3);
        border-radius: 14px; padding: 30px; text-align: center;
        background: rgba(102,126,234,0.03);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none; border-radius: 12px;
        font-weight: 700; letter-spacing: 0.5px;
    }
    
    .footer {
        text-align: center; color: #4a5568; font-size: 0.82rem;
        padding: 15px; margin-top: 30px;
        border-top: 1px solid rgba(255,255,255,0.04);
    }
    .footer a { color: #667eea; text-decoration: none; }
    .footer b { color: #98aaff; }
</style>
""", unsafe_allow_html=True)

# ---- Initialize session state ----
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'doc_processed' not in st.session_state:
    st.session_state.doc_processed = False
if 'doc_stats' not in st.session_state:
    st.session_state.doc_stats = {}

# ---- Sidebar: Document Upload ----
with st.sidebar:
    st.markdown("### 📁 Upload Document")
    
    api_key = st.text_input("Gemini API Key", type="password", 
                            value=os.getenv("GOOGLE_API_KEY", ""),
                            help="Get your free key at aistudio.google.com")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file and api_key:
        if st.button("📄 Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing document..."):
                # Extract text
                text = extract_text_from_pdf(uploaded_file)
                
                # Split into chunks
                chunks = split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200)
                
                # Create vector store
                vector_store = create_vector_store(chunks)
                
                # Create RAG chain
                st.session_state.chain = create_rag_chain(vector_store)
                st.session_state.doc_processed = True
                st.session_state.chat_history = []
                st.session_state.doc_stats = {
                    'filename': uploaded_file.name,
                    'pages': len(extract_text_from_pdf(uploaded_file).split('\n\n')),
                    'chunks': len(chunks),
                    'characters': len(text)
                }
                
                st.success(f"✅ Document processed! {len(chunks)} chunks created.")
    
    if st.session_state.doc_processed:
        st.markdown("---")
        st.markdown("### 📊 Document Stats")
        stats = st.session_state.doc_stats
        st.markdown(f"**File:** {stats.get('filename', 'N/A')}")
        st.markdown(f"**Characters:** {stats.get('characters', 0):,}")
        st.markdown(f"**Chunks:** {stats.get('chunks', 0)}")
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ---- Main Area ----
st.markdown('<p class="main-title">📄 RAG Document Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload a PDF and ask questions — AI answers using only your document</p>', unsafe_allow_html=True)

if not st.session_state.doc_processed:
    # Welcome state
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="status-card">
            <div class="status-num">📤</div>
            <div style="color: #c4d0ff; font-weight: 600; margin: 8px 0;">Step 1: Upload</div>
            <div class="status-label">Upload any PDF document using the sidebar</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="status-card">
            <div class="status-num">🧠</div>
            <div style="color: #c4d0ff; font-weight: 600; margin: 8px 0;">Step 2: Process</div>
            <div class="status-label">AI splits, embeds, and indexes your document</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="status-card">
            <div class="status-num">💬</div>
            <div style="color: #c4d0ff; font-weight: 600; margin: 8px 0;">Step 3: Ask</div>
            <div class="status-label">Ask any question and get answers with sources</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔧 How RAG Works Under the Hood")
    st.markdown("""
    1. **Document Ingestion** — PDF text is extracted page by page
    2. **Chunking** — Text is split into overlapping 1000-character chunks
    3. **Embedding** — Each chunk is converted to a vector using Google's embedding model
    4. **Storage** — Vectors are stored in ChromaDB for fast similarity search
    5. **Retrieval** — When you ask a question, the most relevant chunks are found
    6. **Generation** — Gemini reads the relevant chunks and generates an accurate answer
    7. **Citation** — The system shows which document sections the answer came from
    """)

else:
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 <b>Assistant:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get('sources'):
                with st.expander("📎 View Sources", expanded=False):
                    for i, src in enumerate(msg['sources'], 1):
                        st.markdown(f'<div class="source-box"><b>Source {i}:</b> {src}...</div>', unsafe_allow_html=True)
    
    # Chat input
    question = st.chat_input("Ask a question about your document...")
    
    if question:
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': question})
        st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {question}</div>', unsafe_allow_html=True)
        
        # Get answer
        with st.spinner("🔍 Searching document and generating answer..."):
            llm, retriever, prompt = st.session_state.chain
            answer, sources = ask_question(llm, retriever, prompt, question)
        
        # Add AI response
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': answer,
            'sources': sources
        })
        
        st.markdown(f'<div class="chat-ai">🤖 <b>Assistant:</b><br>{answer}</div>', unsafe_allow_html=True)
        if sources:
            with st.expander("📎 View Sources", expanded=True):
                for i, src in enumerate(sources, 1):
                    st.markdown(f'<div class="source-box"><b>Source {i}:</b> {src}...</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Built by <b>Giriraj Kudupudi</b> | RAG Pipeline: LangChain + ChromaDB + Gemini | 
    <a href="https://github.com/GirirajKudupudi">GitHub</a>
</div>
""", unsafe_allow_html=True)