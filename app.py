import streamlit as st
import os
import time
import requests
from streamlit_lottie import st_lottie
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
# Try importing chains safely
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_community.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# 1. CONFIGURATION
st.set_page_config(page_title="Legal Mind AI", page_icon="⚡", layout="wide")

# 2. ASSETS
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_scanning = load_lottieurl("https://lottie.host/68213322-2621-4d37-9759-424a7304e228/w2A3s9j8g8.json")

# 3. CSS STYLING
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    .stApp { background-color: #000000; font-family: 'Inter', sans-serif; }
    header[data-testid="stHeader"] { background: transparent; }
    .stChatInputContainer textarea { background-color: #111 !important; border: 1px solid #333 !important; color: white !important; }
    [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
    /* Success Message Style */
    .stSuccess { background-color: rgba(0, 255, 0, 0.1) !important; color: #00ff00 !important; }
    </style>
""", unsafe_allow_html=True)

# 4. SMART BATCHING FUNCTION (The Speed Fix)
def create_vector_store_batched(chunks, embeddings, batch_size=50):
    total_chunks = len(chunks)
    vector_store = None
    
    # Progress Bar for Embedding
    embed_bar = st.progress(0, text="Creating Neural Embeddings...")
    
    for i in range(0, total_chunks, batch_size):
        # Slice the batch
        batch = chunks[i : i + batch_size]
        
        # Create (or add to) Vector Store
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)
            
        # Update Progress
        progress = min((i + batch_size) / total_chunks, 1.0)
        embed_bar.progress(progress, text=f"Embedding batch {i//batch_size + 1} ({int(progress*100)}%)...")
        
        # CRITICAL: Wait 1.5 seconds to respect Google's Rate Limit
        time.sleep(1.5)
        
    embed_bar.empty()
    return vector_store

@st.cache_resource
def process_files(uploaded_files):
    progress_text = "Initializing High-Speed Engine..."
    my_bar = st.progress(0, text=progress_text)
    
    documents = []
    total_files = len(uploaded_files)
    
    # 1. READ FILES (Fast)
    for i, uploaded_file in enumerate(uploaded_files):
        my_bar.progress((i / total_files) * 0.3, text=f"Reading File {i+1}/{total_files}...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyMuPDFLoader(tmp_file_path)
        docs = loader.load()
        documents.extend(docs)
        
    # 2. SPLIT TEXT
    my_bar.progress(0.4, text="Splitting Documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # 3. EMBED (The "Smart" Way)
    my_bar.progress(0.5, text=f"Connecting to Google Brain ({len(chunks)} chunks)...")
    
    # Switch back to Google Embeddings (Fast API)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        vector_store = create_vector_store_batched(chunks, embeddings)
        my_bar.progress(1.0, text="Indexing Complete!")
        time.sleep(1)
        my_bar.empty()
        return vector_store
        
    except Exception as e:
        st.error(f"Error connecting to Google: {e}")
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

# 5. UI LAYOUT
with st.sidebar:
    st.markdown("### ⚡ Legal Mind AI")
    api_key = st.text_input("API Key", type="password", placeholder="sk-...")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", 
        type="pdf", 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if "vector_store" in st.session_state:
        st.success(f"● Online")

if "messages" not in st.session_state: st.session_state.messages = []

# Trigger Processing
if uploaded_files and api_key and "vector_store" not in st.session_state:
    st.session_state.vector_store = process_files(uploaded_files)

# Hero Section
if not st.session_state.messages:
    st.markdown("<br><h1 style='text-align: center; color: white;'>Legal Research, Accelerated.</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Upload the Act. Get answers in seconds.</p>", unsafe_allow_html=True)

# Chat Loop
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if "vector_store" in st.session_state:
        with st.chat_message("assistant"):
            # Lottie Animation
            placeholder = st.empty()
            with placeholder:
                col1, col2, col3 = st.columns([1,1,1])
                with col2: st_lottie(lottie_scanning, height=100, key="loading")
            
            response = get_answer(st.session_state.vector_store, prompt)
            placeholder.empty()
            
            st.markdown(response["result"])
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})
            with st.expander("Citations"):
                for doc in response["source_documents"]:
                    st.markdown(f"**Page {doc.metadata.get('page','-')}**")
                    st.caption(doc.page_content[:300])
    else:
        st.warning("Please upload documents first.")