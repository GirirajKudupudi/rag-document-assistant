import streamlit as st
import os
from src.document_processor import extract_text_from_pdf, split_text_into_chunks
from src.vector_store import create_vector_store
from src.rag_chain import create_rag_chain, ask_question

st.set_page_config(
    page_title='RAG Document Assistant',
    page_icon='📄',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .stApp { background: #0a0e17; color: #e0e6ed; }
    [data-testid="stSidebar"] { background: #111827; border-right: 1px solid rgba(102,126,234,0.12); }
    [data-testid="stHeader"] { background: transparent; }

    /* Title */
    .hero-title {
        font-size: 2.4rem; font-weight: 900;
        background: linear-gradient(135deg, #667eea, #a78bfa, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin: 0;
    }
    .hero-sub {
        color: #6b7c93; text-align: center; font-size: 0.95rem; margin: 5px 0 20px;
    }

    /* Tech Badges */
    .tech-row { text-align: center; margin: 10px 0 20px; }
    .tech-tag {
        display: inline-block;
        background: rgba(102,126,234,0.1);
        color: #98aaff;
        padding: 4px 14px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(102,126,234,0.2);
    }

    /* Pipeline Steps */
    .pipeline-box {
        background: #141b2d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border-top: 3px solid;
        margin: 5px 0;
    }
    .pipeline-num { font-size: 1.6rem; font-weight: 900; }
    .pipeline-box h4 { color: #e0e6ed; margin: 6px 0 4px; font-size: 0.95rem; }
    .pipeline-box p { color: #7a8ba5; font-size: 0.78rem; line-height: 1.4; margin: 0; }

    /* Sidebar Model Card */
    .model-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(167,139,250,0.1));
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 12px;
        padding: 14px;
        margin: 10px 0;
    }
    .model-card-title { color: #c4d0ff; font-weight: 700; font-size: 0.9rem; margin-bottom: 8px; }
    .model-detail { color: #7a8ba5; font-size: 0.8rem; margin: 4px 0; }
    .model-detail b { color: #a8b8ff; }

    /* Stats Card */
    .stat-pill {
        display: inline-block;
        background: #141b2d;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 8px 16px;
        margin: 3px;
        text-align: center;
    }
    .stat-pill-num { font-size: 1.2rem; font-weight: 800; color: #667eea; }
    .stat-pill-label { font-size: 0.7rem; color: #7a8ba5; }

    /* Chat Messages */
    .msg-user {
        background: rgba(102,126,234,0.08);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 16px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #c4d0ff;
    }
    .msg-user b { color: #a8b8ff; }
    .msg-ai {
        background: rgba(16,185,129,0.06);
        border: 1px solid rgba(16,185,129,0.15);
        border-radius: 16px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #e0e6ed;
        line-height: 1.7;
    }
    .msg-ai b { color: #6ee7b7; }
    .source-card {
        background: rgba(255,255,255,0.02);
        border-left: 3px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.8rem;
        color: #7a8ba5;
        line-height: 1.5;
    }
    .source-label { color: #667eea; font-weight: 700; font-size: 0.75rem; margin-bottom: 4px; }

    /* Welcome Cards */
    .welcome-card {
        background: #141b2d;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .welcome-icon { font-size: 2rem; margin-bottom: 8px; }
    .welcome-card h4 { color: #c4d0ff; margin: 0 0 6px; font-size: 0.95rem; }
    .welcome-card p { color: #7a8ba5; font-size: 0.82rem; line-height: 1.4; margin: 0; }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none; border-radius: 10px;
        font-weight: 700; letter-spacing: 0.3px;
    }

    /* Footer */
    .footer {
        text-align: center; color: #4a5568; font-size: 0.78rem;
        padding: 12px; margin-top: 25px;
        border-top: 1px solid rgba(255,255,255,0.04);
    }
    .footer a { color: #667eea; text-decoration: none; }
    .footer b { color: #98aaff; }

    /* Key Features Banner */
    .key-features {
        background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(167,139,250,0.08));
        border: 1px solid rgba(102,126,234,0.15);
        border-radius: 14px;
        padding: 18px 24px;
        margin: 15px 0;
    }
    .kf-title { color: #c4d0ff; font-weight: 700; font-size: 0.95rem; margin-bottom: 10px; }
    .kf-item { color: #8899aa; font-size: 0.85rem; margin: 4px 0; line-height: 1.5; }
    .kf-item b { color: #a8b8ff; }
</style>
""", unsafe_allow_html=True)

# ---- Session State ----
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'doc_processed' not in st.session_state:
    st.session_state.doc_processed = False
if 'doc_stats' not in st.session_state:
    st.session_state.doc_stats = {}

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 📁 Document Upload")

    # Model Info Card
    st.markdown("""
    <div class="model-card">
        <div class="model-card-title">🧠 Model Configuration</div>
        <div class="model-detail"><b>LLM:</b> Llama 3.2 (1B params)</div>
        <div class="model-detail"><b>Embeddings:</b> Llama 3.2 via Ollama</div>
        <div class="model-detail"><b>Vector DB:</b> ChromaDB (local)</div>
        <div class="model-detail"><b>Framework:</b> LangChain</div>
        <div class="model-detail"><b>Chunk Size:</b> 1,000 chars</div>
        <div class="model-detail"><b>Chunk Overlap:</b> 200 chars</div>
        <div class="model-detail"><b>Top-K Retrieval:</b> 4 chunks</div>
        <div class="model-detail" style="margin-top: 8px; color: #6ee7b7;">✅ Runs locally — no API key needed</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("⚡ Process Document", type="primary", use_container_width=True):
            with st.spinner("Extracting text, chunking, embedding..."):
                text = extract_text_from_pdf(uploaded_file)
                chunks = split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200)
                vector_store = create_vector_store(chunks)
                st.session_state.chain = create_rag_chain(vector_store)
                st.session_state.doc_processed = True
                st.session_state.chat_history = []
                st.session_state.doc_stats = {
                    'filename': uploaded_file.name,
                    'chunks': len(chunks),
                    'characters': len(text)
                }
                st.success(f"✅ Ready! {len(chunks)} chunks indexed.")

    if st.session_state.doc_processed:
        st.markdown("---")
        stats = st.session_state.doc_stats
        st.markdown(f"""
        <div style="text-align: center;">
            <div class="stat-pill">
                <div class="stat-pill-num">{stats.get('chunks', 0)}</div>
                <div class="stat-pill-label">Chunks</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-num">{stats.get('characters', 0):,}</div>
                <div class="stat-pill-label">Characters</div>
            </div>
        </div>
        <div style="text-align: center; color: #7a8ba5; font-size: 0.8rem; margin-top: 8px;">
            📄 {stats.get('filename', '')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ---- Main Area ----
st.markdown('<p class="hero-title">📄 RAG Document Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a PDF and ask questions — AI answers grounded in your document with source citations</p>', unsafe_allow_html=True)

# Tech badges
st.markdown("""
<div class="tech-row">
    <span class="tech-tag">LangChain</span>
    <span class="tech-tag">ChromaDB</span>
    <span class="tech-tag">Llama 3.2</span>
    <span class="tech-tag">Ollama</span>
    <span class="tech-tag">RAG Pipeline</span>
    <span class="tech-tag">Streamlit</span>
</div>
""", unsafe_allow_html=True)

if not st.session_state.doc_processed:
    # Key Features
    st.markdown("""
    <div class="key-features">
        <div class="kf-title">✨ What Makes This Different</div>
        <div class="kf-item">🔒 <b>100% Local</b> — Your documents never leave your machine. No data sent to external APIs.</div>
        <div class="kf-item">📎 <b>Source Citations</b> — Every answer shows exactly which part of the document it came from.</div>
        <div class="kf-item">🧠 <b>RAG Architecture</b> — Retrieval-Augmented Generation ensures answers are grounded in facts, not hallucinations.</div>
        <div class="kf-item">⚡ <b>No API Key</b> — Runs entirely on Ollama + Llama 3.2. Free forever.</div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline
    st.markdown("---")
    st.markdown("##### ⚙️ RAG Pipeline Architecture")

    c1, c2, c3, c4, c5 = st.columns(5)
    steps = [
        ("01", "Ingest", "PDF text extraction page by page", "#667eea"),
        ("02", "Chunk", "Split into 1000-char overlapping segments", "#a78bfa"),
        ("03", "Embed", "Convert chunks to vectors via Llama 3.2", "#4facfe"),
        ("04", "Retrieve", "Find top-4 relevant chunks per query", "#f093fb"),
        ("05", "Generate", "LLM answers using only retrieved context", "#6ee7b7"),
    ]
    for col, (num, title, desc, color) in zip([c1,c2,c3,c4,c5], steps):
        with col:
            st.markdown(f"""
            <div class="pipeline-box" style="border-top-color: {color};">
                <div class="pipeline-num" style="color: {color};">{num}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # Welcome cards
    st.markdown("---")
    st.markdown("##### 📌 Get Started")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">📤</div>
            <h4>Upload</h4>
            <p>Drag any PDF into the sidebar uploader</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">⚡</div>
            <h4>Process</h4>
            <p>Click "Process Document" to index it</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">💬</div>
            <h4>Ask</h4>
            <p>Type any question in natural language</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Chat Interface
    st.markdown("---")

    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f'<div class="msg-user"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-ai"><b>Assistant:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get('sources'):
                with st.expander("📎 View Sources", expanded=False):
                    for i, src in enumerate(msg['sources'], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-label">SOURCE {i}</div>
                            {src}...
                        </div>
                        """, unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.chat_history.append({'role': 'user', 'content': question})

        with st.spinner("🔍 Retrieving relevant chunks and generating answer..."):
            llm, retriever, prompt = st.session_state.chain
            answer, sources = ask_question(llm, retriever, prompt, question)

        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': answer,
            'sources': sources
        })
        st.rerun()

# Footer
st.markdown("""
<div class="footer">
    Built by <b>Giriraj Kudupudi</b> | RAG: LangChain + ChromaDB + Llama 3.2 |
    <a href="https://github.com/GirirajKudupudi">GitHub</a>
</div>
""", unsafe_allow_html=True)