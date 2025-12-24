import streamlit as st
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

# --- FIX 1: NEW IMPORT LOCATIONS ---
from langchain_core.prompts import PromptTemplate
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_community.chains import RetrievalQA
    
import tempfile

# 1. CONFIGURATION
st.set_page_config(page_title="Legal Mind AI", page_icon="⚡", layout="wide")

# 2. CSS STYLING (Neural Pulse)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp { background-color: #000000; font-family: 'Inter', sans-serif; }
    header[data-testid="stHeader"] { background: transparent; }
    
    .stChatInputContainer textarea { 
        background-color: #111 !important; 
        border: 1px solid #333 !important; 
        color: white !important; 
    }
    
    [data-testid="stSidebar"] { 
        background-color: #050505; 
        border-right: 1px solid #222; 
    }
    
    /* NEURAL PULSE ANIMATION */
    .neural-loader {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 60px;
        gap: 8px;
    }
    .bar {
        width: 6px;
        height: 20px;
        background: linear-gradient(180deg, #4F46E5, #9333EA);
        border-radius: 3px;
        animation: pulse 1s ease-in-out infinite;
    }
    .bar:nth-child(1) { animation-delay: 0.0s; height: 20px; }
    .bar:nth-child(2) { animation-delay: 0.1s; height: 35px; }
    .bar:nth-child(3) { animation-delay: 0.2s; height: 45px; }
    .bar:nth-child(4) { animation-delay: 0.3s; height: 35px; }
    .bar:nth-child(5) { animation-delay: 0.4s; height: 20px; }
    
    @keyframes pulse {
        0% { transform: scaleY(1); opacity: 0.6; }
        50% { transform: scaleY(1.5); opacity: 1; box-shadow: 0 0 10px rgba(147, 51, 234, 0.5); }
        100% { transform: scaleY(1); opacity: 0.6; }
    }
    </style>
""", unsafe_allow_html=True)

# 3. BACKEND LOGIC 

def create_vector_store_optimistic(chunks, embeddings):
    vector_store = None
    batch_size = 100 
    total_chunks = len(chunks)
    
    progress_bar = st.progress(0, text="Accelerating Neural Engine...")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        success = False
        retries = 0
        while not success and retries < 3:
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                success = True 
            except Exception as e:
                retries += 1
                time.sleep(2 * retries)
        
        progress = min((i + batch_size) / total_chunks, 1.0)
        progress_bar.progress(progress, text=f"Indexing batch {i//batch_size + 1}...")
        
    progress_bar.empty()
    return vector_store

@st.cache_resource
def process_files(uploaded_files):
    if not uploaded_files: return None
    
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyMuPDFLoader(tmp_file_path)
        docs = loader.load()
        documents.extend(docs)
    
    if not documents: return None

    # STRUCTURE-AWARE CHUNKING
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\nSection ", "\nArticle ", "\nPART ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    st.toast(f"Structure-Aware Splitter created {len(chunks)} nodes.", icon="🧠")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        vector_store = create_vector_store_optimistic(chunks, embeddings)
        st.toast("System Online!", icon="✅")
        return vector_store
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# 4. UI ORCHESTRATION

with st.sidebar:
    st.markdown("### ⚡ Legal Mind AI")
    api_key = st.text_input("API Key", type="password", placeholder="sk-...")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    
    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        st.success(f"● Online")

if "messages" not in st.session_state: st.session_state.messages = []

# Process Uploads
if uploaded_files and api_key and "vector_store" not in st.session_state:
    st.session_state.vector_store = process_files(uploaded_files)

# Hero
if not st.session_state.messages:
    st.markdown("<br><h1 style='text-align: center; color: white;'>Legal Research, Accelerated.</h1>", unsafe_allow_html=True)

# Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & Logic
if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        with st.chat_message("assistant"):
            
            # ANIMATION
            placeholder_anim = st.empty()
            placeholder_anim.markdown("""
                <div class="neural-loader">
                    <div class="bar"></div><div class="bar"></div><div class="bar"></div>
                </div>
            """, unsafe_allow_html=True)
            
            # STREAMING LOGIC
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
            
            # Kill Animation
            placeholder_anim.empty()
            
            # Stream the Response
            full_response = ""
            message_placeholder = st.empty()
            
            try:
                stream = llm.stream(PROMPT.format(context=context_text, question=prompt))
                for chunk in stream:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Save and Show Citations
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                with st.expander("Citations (Structure-Aware)"):
                    for doc in docs:
                        st.markdown(f"**Reference:**")
                        st.caption(doc.page_content[:300])
                        st.markdown("---")
            except Exception as e:
                 message_placeholder.error(f"Generation Error: {e}")

    else:
        st.warning("Please upload a document to begin.")