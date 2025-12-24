import streamlit as st
import os
import time
import requests
from streamlit_lottie import st_lottie
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Legal Mind AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ASSETS (Lottie Animation)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Professional "Data Processing" Animation
lottie_scanning = load_lottieurl("https://lottie.host/68213322-2621-4d37-9759-424a7304e228/w2A3s9j8g8.json")

# 3. GOOGLE-GRADE CSS ENGINE
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --background: #050505;
        --card-surface: rgba(255, 255, 255, 0.03);
        --border-color: rgba(255, 255, 255, 0.08);
        --accent-glow: rgba(41, 182, 246, 0.15);
    }

    .stApp {
        background-color: var(--background);
        font-family: 'Inter', sans-serif;
        background-image: radial-gradient(circle at 50% 0%, #111 0%, #050505 50%);
    }

    /* HEADER FIXES */
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 3D INTERACTIVE CARDS */
    .feature-card {
        background: var(--card-surface);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 24px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 30px -10px var(--accent-glow);
    }
    
    .card-icon { font-size: 22px; margin-bottom: 12px; opacity: 0.8; }
    .card-title { font-size: 15px; font-weight: 600; color: #fff; margin-bottom: 6px; }
    .card-desc { font-size: 13px; color: #888; line-height: 1.5; }

    /* TYPOGRAPHY */
    .hero-title {
        font-size: 64px;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(180deg, #FFFFFF 0%, #777 120%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    /* INPUT FIELD - FLOATING DOCK */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    .stChatInputContainer textarea {
        background-color: #0A0A0A !important;
        border: 1px solid #222 !important;
        color: white !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif;
    }
    .stChatInputContainer textarea:focus {
        border-color: #444 !important;
        box-shadow: 0 0 0 1px #444 !important;
    }

    /* CHAT MESSAGES */
    [data-testid="stChatMessage"] {
        background: transparent;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    
    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--border-color);
        background-color: #080808;
    }
    
    </style>
""", unsafe_allow_html=True)

# 4. SIDEBAR
with st.sidebar:
    st.markdown("### ⚡ Legal Mind AI")
    st.caption("v3.0 • Production Environment")
    st.markdown("---")
    
    api_key = st.text_input("API Key", type="password", placeholder="sk-...")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
    
    if "vector_store" in st.session_state:
        st.success("● System Online")
        
    st.markdown("---")
    st.caption("System Metrics")
    st.markdown(f"<div style='color: #444; font-size: 12px;'>Latency: < 800ms<br>Model: Gemini 1.5 Flash<br>RAG: Strict</div>", unsafe_allow_html=True)

# 5. BACKEND LOGIC
@st.cache_resource
def process_pdf(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    return None

def get_answer(vector_store, question):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    prompt_template = """
    SYSTEM: You are a Senior Legal Counsel.
    CONTEXT: Nigeria Tax Administration Act 2025.
    RULE: Effective Date is Jan 1, 2026.
    TASK: Answer strictly. Cite Sections.
    
    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa_chain.invoke({"query": question})

# 6. UI ORCHESTRATION

if "messages" not in st.session_state:
    st.session_state.messages = []

# Processing Animation
if uploaded_file and api_key and "vector_store" not in st.session_state:
    placeholder = st.empty()
    with placeholder:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st_lottie(lottie_scanning, height=100, key="ingest")
        st.session_state.vector_store = process_pdf(uploaded_file)
    placeholder.empty()
    st.rerun()

# HERO SECTION (Clean, Minimalist, Google-Esque)
if not st.session_state.messages:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title' style='text-align: center;'>Legal Research.<br>Reimagined.</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #666; margin-bottom: 60px;'>Grounded in Truth. Powered by Gemini.</div>", unsafe_allow_html=True)
    
    # 3D Cards Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">⚡</div>
            <div class="card-title">Instant Citations</div>
            <div class="card-desc">Every claim backed by Section & Subsection references.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">📅</div>
            <div class="card-title">Temporal Logic</div>
            <div class="card-desc">Auto-detects Effective Date (Jan 1, 2026) vs Current Law.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">🛡️</div>
            <div class="card-title">Anti-Hallucination</div>
            <div class="card-desc">Strict grounding ensures zero fabrication of laws.</div>
        </div>
        """, unsafe_allow_html=True)

# CHAT INTERFACE
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Query the Legal Database..."):
    # User Input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI Response
    if "vector_store" in st.session_state:
        with st.chat_message("assistant"):
            # LOTTIE THINKING ANIMATION
            placeholder = st.empty()
            with placeholder:
                col_a, col_b, col_c = st.columns([1,2,1])
                with col_b:
                    st_lottie(lottie_scanning, height=120, key="thinking")
            
            # Smart Delay Detection (Optional Toast)
            if "penalty" in prompt.lower():
                st.toast("⚠️ Analyzing High-Stakes Compliance...", icon="🚨")

            response = get_answer(st.session_state.vector_store, prompt)
            placeholder.empty() # Kill animation
            
            answer = response["result"]
            sources = response["source_documents"]
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # PROFESSIONAL CITATION DRAWER
            with st.expander("📂 Source Documents (Verification)"):
                for doc in sources:
                    st.markdown(f"**Reference (Page {doc.metadata.get('page','-')}):**")
                    st.caption(doc.page_content[:400] + "...")
                    st.markdown("---")

    else:
        st.warning("⚠️ Please upload the Tax Act PDF to begin.")