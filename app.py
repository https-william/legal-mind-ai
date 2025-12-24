import streamlit as st
import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
import fitz  # PyMuPDF

# 1. CONFIGURATION
st.set_page_config(page_title="Legal Mind AI", page_icon="⚖️", layout="wide")

# 2. GLASSMORPHISM THEME
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400&display=swap');
    .stApp { background-color: #050505; background-image: radial-gradient(circle at 50% 50%, #1a1a1a 0%, #000000 100%); font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: rgba(10, 10, 10, 0.7); backdrop-filter: blur(12px); border-right: 1px solid rgba(255, 255, 255, 0.08); }
    .stChatInputContainer textarea { background-color: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; color: #e0e0e0 !important; border-radius: 12px; }
    h1, h2, h3 { color: #fff; } p { color: #ccc; }
    .gradient-text { background: linear-gradient(90deg, #D4AF37, #F1C40F); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold; }
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

# --- NEW: QUERY REFINER ---
def refine_query(raw_prompt):
    """Uses a small, fast model to fix typos and add legal context."""
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile") # Using 70b because it's smarter at logic
    system = "You are a Legal Search Assistant. Reformulate the user's query into a precise legal search term. Correct typos (e.g., 'Task' -> 'Tax'). If they ask about '2026 reform', mention 'Nigeria Tax Act 2025' since it is effective 2026. Return ONLY the reformulated query."
    try:
        response = llm.invoke([("system", system), ("human", raw_prompt)])
        return response.content.strip()
    except:
        return raw_prompt # Fallback

# 4. UI ORCHESTRATION

if "messages" not in st.session_state: st.session_state.messages = []
if "usage_count" not in st.session_state: st.session_state.usage_count = 0

with st.sidebar:
    st.markdown("## ⚖️ Legal Mind <span style='font-size:0.8em; color:#D4AF37'>AI</span>", unsafe_allow_html=True)
    st.caption("v8.0 • Smart Query Refinement")
    st.markdown("---")
    
    daily_limit = 1000
    remaining = daily_limit - st.session_state.usage_count
    st.markdown(f"**Credits:** `{remaining}/{daily_limit}`")
    st.progress(st.session_state.usage_count / daily_limit)
    
    mode = st.radio("Mode:", ["Strict Compliance (Docs)", "Web Research (Live)"], label_visibility="collapsed")
    st.markdown("---")
    
    api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
    if api_key: os.environ["GROQ_API_KEY"] = api_key
    
    if os.path.exists("faiss_index_tax_act") and mode == "Strict Compliance (Docs)":
        if "vector_store" not in st.session_state or st.session_state.vector_store is None:
            try:
                embeddings = get_embeddings()
                st.session_state.vector_store = FAISS.load_local("faiss_index_tax_act", embeddings, allow_dangerous_deserialization=True)
                st.success("⚡ Strict Brain Loaded")
            except: pass

    if mode == "Strict Compliance (Docs)":
        uploaded_files = st.file_uploader("📂 Upload New Files", type="pdf", accept_multiple_files=True)
        if st.button("⚡ Process Files", type="primary"):
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
    st.markdown("<br><br><h1 style='text-align: center;'>Legal Research, <span class='gradient-text'>Perfected.</span></h1>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Query the Legal Database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if not api_key:
        st.warning("Please enter Groq API Key.")
        st.stop()

    st.session_state.usage_count += 1
    
    with st.chat_message("assistant"):
        
        status_box = st.status("🧠 Refining Legal Query...", expanded=True)
        
        # STEP 1: REFINE QUERY (The "Sharpness" Fix)
        refined_prompt = refine_query(prompt)
        status_box.write(f"**Optimized Search:** `{refined_prompt}`")
        
        context_text = ""
        docs = []
        
        # STEP 2: RETRIEVE
        if mode == "Strict Compliance (Docs)":
            if "vector_store" in st.session_state and st.session_state.vector_store:
                status_box.write("🔍 Deep-Scanning Internal Documents...")
                # Increased 'k' to 6 for broader context
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 6})
                docs = retriever.invoke(refined_prompt)
                context_text = "\n\n".join([d.page_content for d in docs])
                status_box.write("✅ Relevant Clauses Locked.")
            else:
                status_box.error("No documents loaded.")
                st.stop()

        else:
            status_box.write("🌍 Scanning Global Legal Web...")
            search = DuckDuckGoSearchRun()
            try:
                # Search using the REFINED prompt, not the user's typo
                context_text = search.run(f"{refined_prompt} Nigeria law")
                status_box.write("✅ Web Evidence Retrieved.")
            except Exception as e:
                status_box.error(f"Search failed: {e}")
                st.stop()
        
        status_box.update(label="Drafting Legal Opinion...", state="running", expanded=False)
        
        # STEP 3: GENERATE
        llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile")
        
        system_instruction = f"""
        You are a Senior Legal Counsel. 
        Your goal is to connect the user's intent to the provided legal text, even if the phrasing isn't exact.
        
        CONTEXT:
        {context_text}
        
        USER QUESTION: {prompt} (Refined as: {refined_prompt})
        
        TASK: 
        1. Answer strictly based on the context.
        2. If the user asks about "2026 reform" and the text mentions "2025 Act effective 2026", EXPLAIN that connection.
        3. Cite your sources.
        """
        
        messages = [("system", system_instruction), ("human", prompt)]
        
        full_response = ""
        message_placeholder = st.empty()
        
        try:
            stream = llm.stream(messages)
            for chunk in stream:
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            status_box.update(label="Complete", state="complete", expanded=False)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            if mode == "Strict Compliance (Docs)" and docs:
                with st.expander("⚖️  View Source Evidence"):
                    for doc in docs:
                        st.markdown(f"**Page {doc.metadata.get('page')} Reference:**")
                        st.caption(doc.page_content[:300] + "...")
                        st.markdown("---")
                    
        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Generation Error: {e}")