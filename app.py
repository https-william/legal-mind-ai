import streamlit as st
import os
import time
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import fitz  # PyMuPDF

# 1. CONFIGURATION
st.set_page_config(page_title="Legal Mind AI", page_icon="⚖️", layout="wide")

# 2. CSS STYLING
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    .stApp { background-color: #050505; font-family: 'Inter', sans-serif; }
    header[data-testid="stHeader"] { background: transparent; }
    .stChatInputContainer textarea { background-color: #121212 !important; border: 1px solid #333 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
    
    .neural-loader { display: flex; justify-content: center; align-items: center; height: 60px; gap: 8px; }
    .bar { width: 6px; height: 20px; background: linear-gradient(180deg, #D4AF37, #AA8C2C); border-radius: 3px; animation: pulse 1s ease-in-out infinite; }
    .bar:nth-child(1) { animation-delay: 0.0s; height: 20px; }
    .bar:nth-child(2) { animation-delay: 0.1s; height: 35px; }
    .bar:nth-child(3) { animation-delay: 0.2s; height: 45px; }
    
    @keyframes pulse { 0% { opacity: 0.6; } 50% { transform: scaleY(1.5); opacity: 1; } 100% { opacity: 0.6; } }
    </style>
""", unsafe_allow_html=True)

# 3. ENGINE LOGIC

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_files(uploaded_files):
    if not uploaded_files: return None
    
    documents = []
    progress = st.progress(0, text="Reading Legal Documents...")
    
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                text = page.get_text()
                if text:
                    documents.append(Document(page_content=text, metadata={"page": i+1, "source": uploaded_file.name}))
    
    if not documents: return None

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    
    progress.progress(0.5, text="Building Neural Index (This takes ~60s)...")
    
    # Embed
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save locally so we can download it
    vector_store.save_local("faiss_index_tax_act")
    
    progress.empty()
    return vector_store

# 4. UI ORCHESTRATION

if "messages" not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=40)
    st.markdown("### Legal Mind AI")
    st.caption("v5.0 • Pre-Loaded Core")
    st.markdown("---")
    
    api_key = st.text_input("🔑 API Credentials", type="password", placeholder="Paste Google Key")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    # MODE 1: PRE-LOADED BRAIN (The Fast Way)
    # Checks if you uploaded the 'faiss_index_tax_act' folder to GitHub
    if os.path.exists("faiss_index_tax_act"):
        if "vector_store" not in st.session_state or st.session_state.vector_store is None:
            try:
                embeddings = get_embeddings()
                st.session_state.vector_store = FAISS.load_local("faiss_index_tax_act", embeddings, allow_dangerous_deserialization=True)
                st.success("⚡ Tax Act 2025 Pre-Loaded")
            except Exception as e:
                st.error(f"Could not load brain: {e}")

    # MODE 2: MANUAL UPLOAD (The Slow Way)
    uploaded_files = st.file_uploader("📂 Upload New Files", type="pdf", accept_multiple_files=True)
    
    if st.button("⚡ Process New Files", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing..."):
                store = process_files(uploaded_files)
                if store:
                    st.session_state.vector_store = store
                    st.success("New Database Online")
                    st.rerun()

    # ADMIN: Download the Brain
    # This button lets YOU download the processed file to put on GitHub
    if os.path.exists("faiss_index_tax_act"):
        shutil.make_archive("legal_brain", 'zip', "faiss_index_tax_act")
        with open("legal_brain.zip", "rb") as fp:
            st.download_button(
                label="💾 Download Processed Brain",
                data=fp,
                file_name="legal_brain.zip",
                mime="application/zip"
            )

# Chat Interface
if not st.session_state.messages:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #e0e0e0;'>Legal Research, Perfected.</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Strict Grounding • Verifiable Citations • Zero Hallucinations</p>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Query the Legal Database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if "vector_store" in st.session_state and st.session_state.vector_store is not None and api_key:
        with st.chat_message("assistant"):
            
            placeholder_anim = st.empty()
            placeholder_anim.markdown("""<div class="neural-loader"><div class="bar"></div><div class="bar"></div><div class="bar"></div></div>""", unsafe_allow_html=True)
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, streaming=True)
            
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
            
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            placeholder_anim.empty()
            
            full_response = ""
            message_placeholder = st.empty()
            
            try:
                stream = llm.stream(PROMPT.format(context=context_text, question=prompt))
                for chunk in stream:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                with st.expander("⚖️  Evidence Locker"):
                    for doc in docs:
                        st.markdown(f"**Reference:**")
                        st.caption(doc.page_content[:300] + "...")
                        st.markdown("---")
            except Exception as e:
                 message_placeholder.error(f"Generation Error: {e}")
    elif not api_key:
        st.warning("Please enter API Key.")
    else:
        st.warning("Database not loaded.")