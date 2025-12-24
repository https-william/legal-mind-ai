import streamlit as st
import os
import time
import requests
from streamlit_lottie import st_lottie
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
# Try importing chains safely
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_community.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# 1. CONFIG
st.set_page_config(page_title="Legal Mind AI", page_icon="⚡", layout="wide")

# 2. ASSETS
def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_scanning = load_lottieurl("https://lottie.host/68213322-2621-4d37-9759-424a7304e228/w2A3s9j8g8.json")

# 3. CSS (Kept your Vercel/Linear Style)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    .stApp { background-color: #000000; font-family: 'Inter', sans-serif; }
    header[data-testid="stHeader"] { background: transparent; }
    .stChatInputContainer textarea { background-color: #111 !important; border: 1px solid #333 !important; color: white !important; }
    [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
    </style>
""", unsafe_allow_html=True)

# 4. BACKEND LOGIC (Optimized for Speed)
@st.cache_resource
def process_files(uploaded_files):
    # Progress Bar
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    documents = []
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        my_bar.progress((i / total_files) * 0.5, text=f"Reading File {i+1}/{total_files}...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Use PyMuPDF (Faster than PyPDF)
        loader = PyMuPDFLoader(tmp_file_path)
        docs = loader.load()
        documents.extend(docs)
        
    # Chunking
    my_bar.progress(0.6, text="Splitting Text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Embedding (The slow part)
    my_bar.progress(0.7, text=f"Embedding {len(chunks)} chunks (This uses CPU)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    my_bar.progress(1.0, text="Indexing Complete!")
    time.sleep(1)
    my_bar.empty()
    
    return vector_store

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

# 5. SIDEBAR
with st.sidebar:
    st.markdown("### ⚡ Legal Mind AI")
    api_key = st.text_input("API Key", type="password", placeholder="sk-...")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    # MULTIPLE FILES ENABLED HERE
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", 
        type="pdf", 
        accept_multiple_files=True, # <--- THIS IS THE FIX
        label_visibility="collapsed"
    )
    
    if "vector_store" in st.session_state:
        st.success(f"● Online ({len(uploaded_files)} files)")

# 6. ORCHESTRATION
if "messages" not in st.session_state: st.session_state.messages = []

# Trigger Processing
if uploaded_files and api_key and "vector_store" not in st.session_state:
    st.session_state.vector_store = process_files(uploaded_files)

# Hero Section
if not st.session_state.messages:
    st.markdown("<br><h1 style='text-align: center; color: white;'>Legal Research, Accelerated.</h1>", unsafe_allow_html=True)

# Chat Loop
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if "vector_store" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = get_answer(st.session_state.vector_store, prompt)
                st.markdown(response["result"])
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})
                with st.expander("Citations"):
                    for doc in response["source_documents"]:
                        st.markdown(f"**Page {doc.metadata.get('page','-')}**")
                        st.caption(doc.page_content[:300])
    else:
        st.warning("Please upload documents first.")