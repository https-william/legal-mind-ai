import streamlit as st
import os
import time
import concurrent.futures
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import tempfile

# 1. CONFIGURATION
st.set_page_config(page_title="Legal Mind AI", page_icon="⚖️", layout="wide")

# 2. CSS STYLING (Premium Law Theme)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp { background-color: #050505; font-family: 'Inter', sans-serif; }
    header[data-testid="stHeader"] { background: transparent; }
    
    /* INPUT & SIDEBAR */
    .stChatInputContainer textarea { 
        background-color: #121212 !important; 
        border: 1px solid #333 !important; 
        color: #e0e0e0 !important; 
    }
    [data-testid="stSidebar"] { 
        background-color: #0a0a0a; 
        border-right: 1px solid #222; 
    }
    
    /* NEURAL PULSE (Gold/Legal Theme) */
    .neural-loader { display: flex; justify-content: center; align-items: center; height: 60px; gap: 8px; }
    .bar { width: 6px; height: 20px; background: linear-gradient(180deg, #D4AF37, #AA8C2C); border-radius: 3px; animation: pulse 1s ease-in-out infinite; }
    .bar:nth-child(1) { animation-delay: 0.0s; height: 20px; }
    .bar:nth-child(2) { animation-delay: 0.1s; height: 35px; }
    .bar:nth-child(3) { animation-delay: 0.2s; height: 45px; }
    .bar:nth-child(4) { animation-delay: 0.3s; height: 35px; }
    .bar:nth-child(5) { animation-delay: 0.4s; height: 20px; }
    
    @keyframes pulse { 0% { opacity: 0.6; } 50% { transform: scaleY(1.5); opacity: 1; } 100% { opacity: 0.6; } }
    
    /* TOAST STYLING */
    div[data-testid="stToast"] { background-color: #111 !important; border: 1px solid #333 !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# 3. HIGH-SPEED LOGIC (MEMORY ONLY)

def embed_batch(args):
    """Worker function to embed a single batch of text."""
    batch, embeddings = args
    try:
        vector_store = FAISS.from_documents(batch, embeddings)
        return vector_store
    except Exception:
        time.sleep(2)
        try:
            vector_store = FAISS.from_documents(batch, embeddings)
            return vector_store
        except:
            return None

def create_vector_store_concurrent(chunks, embeddings):
    batch_size = 100
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    progress_bar = st.progress(0, text="⚖️  Analyzing Legal Framework...")
    
    args = [(batch, embeddings) for batch in batches]
    main_vector_store = None
    completed = 0
    
    # 4 Parallel Workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(embed_batch, args)
        
        for result in results:
            if result is not None:
                if main_vector_store is None:
                    main_vector_store = result
                else:
                    main_vector_store.merge_from(result)
            
            completed += 1
            progress = min(completed / len(batches), 1.0)
            progress_bar.progress(progress, text=f"⚖️  Indexing Volume {completed}/{len(batches)} (Ram-Jet Engine)...")
            
    progress_bar.empty()
    return main_vector_store

@st.cache_resource
def process_files(uploaded_files):
    if not uploaded_files: return None
    
    documents = []
    
    # --- DIRECT MEMORY READ (Safe Mode) ---
    for uploaded_file in uploaded_files:
        # Use getvalue() instead of read() to prevent pointer exhaustion
        file_bytes = uploaded_file.getvalue()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                text = page.get_text()
                if text:
                    documents.append(Document(
                        page_content=text, 
                        metadata={"page": i+1, "source": uploaded_file.name}
                    ))
    
    if not documents: 
        st.error("❌ Error: Could not extract text from PDF.")
        return None

    # Structure-Aware Splitter (Legal Headers)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\nSection ", "\nArticle ", "\nPART ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    st.toast(f"📜 Extracted {len(chunks)} legal clauses.", icon="🏛️")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        vector_store = create_vector_store_concurrent(chunks, embeddings)
        st.toast("Jurisprudence Online.", icon="⚖️")
        return vector_store
    except Exception as e:
        # SHOW THE ERROR ON SCREEN so we know why it failed
        st.error(f"❌ Connection Error: {e}")
        return None

# 4. UI ORCHESTRATION

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=40)
    st.markdown("### Legal Mind AI")
    st.caption("v3.5 • High-Velocity RAG")
    st.markdown("---")
    
    api_key = st.text_input("🔑 API Credentials", type="password", placeholder="Paste Google Key")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    uploaded_files = st.file_uploader("📂 Upload Case Files", type="pdf", accept_multiple_files=True)
    
    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        st.success(f"● System Ready")
        
    # RESET BUTTON (The Fix for Zombie State)
    if st.button("↻ Reset System", use_container_width=True):
        st.session_state.vector_store = None
        st.rerun()

if "messages" not in st.session_state: st.session_state.messages = []

# Process Uploads - FIXED LOGIC HERE
# If files exist, key exists, AND (store is missing OR store is broken/None) -> Try processing
if uploaded_files and api_key and ("vector_store" not in st.session_state or st.session_state.vector_store is None):
    st.session_state.vector_store = process_files(uploaded_files)

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

    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        with st.chat_message("assistant"):
            
            # ANIMATION
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
            
            # Retrieval
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            placeholder_anim.empty()
            
            # Streaming Generation
            full_response = ""
            message_placeholder = st.empty()
            
            try:
                stream = llm.stream(PROMPT.format(context=context_text, question=prompt))
                for chunk in stream:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                with st.expander("⚖️  Evidence Locker (Citations)"):
                    for doc in docs:
                        st.markdown(f"**Section Reference:**")
                        st.caption(doc.page_content[:300] + "...")
                        st.markdown("---")
            except Exception as e:
                 message_placeholder.error(f"Generation Error: {e}")

    else:
        st.warning("⚠️ Please authenticate and upload a document.")