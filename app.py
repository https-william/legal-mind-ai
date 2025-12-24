import streamlit as st
import os
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import fitz  # PyMuPDF
import google.generativeai as genai

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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    
    progress.progress(0.5, text="Building Neural Index...")
    
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_tax_act")
    
    progress.empty()
    return vector_store

# 4. UI ORCHESTRATION

if "messages" not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=40)
    st.markdown("### Legal Mind AI")
    st.caption("v5.3 • Auto-Fallback Core")
    st.markdown("---")
    
    api_key = st.text_input("🔑 API Credentials", type="password", placeholder="Paste Google Key")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    # LOAD BRAIN
    if os.path.exists("faiss_index_tax_act"):
        if "vector_store" not in st.session_state or st.session_state.vector_store is None:
            try:
                embeddings = get_embeddings()
                st.session_state.vector_store = FAISS.load_local("faiss_index_tax_act", embeddings, allow_dangerous_deserialization=True)
                st.success("⚡ Brain Loaded from Disk")
            except:
                pass

    uploaded_files = st.file_uploader("📂 Upload New Files", type="pdf", accept_multiple_files=True)
    
    if st.button("⚡ Process Files", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing..."):
                store = process_files(uploaded_files)
                if store:
                    st.session_state.vector_store = store
                    st.success("Database Online")
                    st.rerun()

    if os.path.exists("faiss_index_tax_act"):
        shutil.make_archive("legal_brain", 'zip', "faiss_index_tax_act")
        with open("legal_brain.zip", "rb") as fp:
            st.download_button("💾 Download Brain", fp, "legal_brain.zip", "application/zip")

# Chat Interface
if not st.session_state.messages:
    st.markdown("<br><br><h1 style='text-align: center; color: #e0e0e0;'>Legal Research, Perfected.</h1>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Query the Legal Database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if "vector_store" in st.session_state and st.session_state.vector_store is not None and api_key:
        with st.chat_message("assistant"):
            
            # VISUAL LOADING STATE
            status_box = st.status("⚖️ Analyzing Legal Precedents...", expanded=True)
            
            # 1. RETRIEVE DOCUMENTS
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            status_box.write("✅ Evidence Retrieved.")
            status_box.update(label="Drafting Response...", state="running", expanded=False)
            
            # 2. GENERATE (With Fallback)
            
            # Helper to try generation
            def try_generate(model_name):
                llm = ChatGoogleGenerativeAI(
                    model=model_name, 
                    temperature=0.0, 
                    streaming=True,
                    safety_settings={
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                    }
                )
                prompt_template = f"""
                SYSTEM: You are a Senior Legal Counsel.
                CONTEXT: {context_text}
                QUESTION: {prompt}
                TASK: Answer strictly based on the context. Cite sections.
                ANSWER:
                """
                return llm.stream(prompt_template)

            full_response = ""
            message_placeholder = st.empty()
            
            try:
                # TRY 1: Flash Model
                stream = try_generate("models/gemini-1.5-flash")
                for chunk in stream:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                
            except Exception as e:
                # TRY 2: Pro Model (Fallback)
                try:
                    status_box.write("⚠️ Switching to Backup Model...")
                    stream = try_generate("gemini-pro")
                    for chunk in stream:
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")
                except Exception as final_e:
                    status_box.update(label="Error", state="error")
                    message_placeholder.error(f"⚠️ AI Generation Error: {str(final_e)}")
                    st.info("Raw Evidence Retrieved:")
                    st.write(context_text)

            if full_response:
                message_placeholder.markdown(full_response)
                status_box.update(label="Complete", state="complete", expanded=False)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                with st.expander("⚖️  View Source Evidence"):
                    for doc in docs:
                        st.markdown(f"**Section Reference:**")
                        st.caption(doc.page_content[:300] + "...")
                        st.markdown("---")

    elif not api_key:
        st.warning("Please enter API Key.")
    else:
        st.warning("Database not loaded.")