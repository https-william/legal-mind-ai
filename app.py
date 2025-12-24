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

# 2. ULTRA-PREMIUM UI THEME
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400&display=swap');
    
    /* Background */
    .stApp { 
        background-color: #050505; 
        background-image: radial-gradient(circle at 50% 0%, #1a1a1a 0%, #000000 70%);
        font-family: 'Inter', sans-serif; 
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { 
        background-color: rgba(10, 10, 10, 0.6); 
        backdrop-filter: blur(20px); 
        border-right: 1px solid rgba(255, 255, 255, 0.05); 
    }

    /* Input Fields */
    .stTextInput input, .stChatInputContainer textarea { 
        background-color: rgba(255, 255, 255, 0.03) !important; 
        border: 1px solid rgba(255, 255, 255, 0.1) !important; 
        color: #e0e0e0 !important; 
        border-radius: 8px;
    }
    
    /* Status Box */
    .stStatusWidget {
        background-color: rgba(20, 20, 20, 0.8) !important;
        border: 1px solid #333 !important;
    }

    /* Custom Text */
    h1, h2, h3 { color: #fff; letter-spacing: -0.02em; }
    p { color: #888; font-size: 0.95rem; line-height: 1.6; }
    
    .gradient-text {
        background: linear-gradient(135deg, #fff 0%, #888 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .gold-text {
        color: #D4AF37;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        color: #444;
        background: rgba(0,0,0,0.5);
        backdrop-filter: blur(5px);
        pointer-events: none;
    }
    </style>
""", unsafe_allow_html=True)

# 3. ENGINE LOGIC

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_files(uploaded_files):
    if not uploaded_files: return None
    documents = []
    progress = st.progress(0, text="Ingesting Legal Corpus...")
    
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
    
    progress.progress(0.5, text="Vectorizing Knowledge Base...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_tax_act")
    progress.empty()
    return vector_store

def refine_query(raw_prompt):
    """Silent corrector for user typos."""
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    system = "You are a Legal Search Assistant. Reformulate the user's query into a precise legal search term. Correct typos (e.g., 'Task' -> 'Tax'). If they ask about '2026 reform', mention 'Nigeria Tax Act 2025' since it is effective 2026. Return ONLY the reformulated query."
    try:
        response = llm.invoke([("system", system), ("human", raw_prompt)])
        return response.content.strip()
    except:
        return raw_prompt

# 4. UI ORCHESTRATION

if "messages" not in st.session_state: st.session_state.messages = []
if "usage_count" not in st.session_state: st.session_state.usage_count = 0

with st.sidebar:
    st.markdown("## ⚖️ Legal Mind <span class='gold-text'>AI</span>", unsafe_allow_html=True)
    st.caption("v9.0 • Production Build")
    st.markdown("---")
    
    # METRICS
    daily_limit = 1000
    remaining = daily_limit - st.session_state.usage_count
    st.markdown(f"**Compute Credits:** `{remaining}`")
    st.progress(st.session_state.usage_count / daily_limit)
    
    st.markdown("### 🧠 Knowledge Source")
    mode = st.radio("Source:", ["Strict Compliance (Docs)", "Web Research (Live)"], label_visibility="collapsed")
    
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
        uploaded_files = st.file_uploader("📂 Upload Corpus", type="pdf", accept_multiple_files=True)
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

# MAIN CHAT
if not st.session_state.messages:
    st.markdown("<br><br><h1 style='text-align: center;'>Legal Research, <span class='gradient-text'>Perfected.</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Strict Grounding • Verifiable Citations • Zero Hallucinations</p>", unsafe_allow_html=True)

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
        
        status_box = st.status("🧠 Processing Query...", expanded=True)
        
        # 1. REFINE
        refined_prompt = refine_query(prompt)
        status_box.write(f"**Interpreted Intent:** `{refined_prompt}`")
        
        context_text = ""
        docs = []
        search_failed = False
        
        # 2. RETRIEVE
        if mode == "Strict Compliance (Docs)":
            if "vector_store" in st.session_state and st.session_state.vector_store:
                status_box.write("🔍 Scanning Internal Documents...")
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 6})
                docs = retriever.invoke(refined_prompt)
                context_text = "\n\n".join([d.page_content for d in docs])
                status_box.write("✅ Evidence Locked.")
            else:
                status_box.error("No documents loaded.")
                st.stop()

        else:
            status_box.write("🌍 Scanning Global Legal Web...")
            search = DuckDuckGoSearchRun()
            try:
                # Attempt Search
                context_text = search.run(f"{refined_prompt} Nigeria law")
                status_box.write("✅ Web Evidence Retrieved.")
            except Exception:
                # SILENT FAILOVER
                search_failed = True
                status_box.write("⚠️ Web Unavailable. Switching to General Knowledge.")
                context_text = "" # Empty context forces AI to rely on training data
        
        status_box.update(label="Drafting Legal Opinion...", state="running", expanded=False)
        
        # 3. GENERATE
        llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile")
        
        # Dynamic System Prompt based on failure state
        if search_failed:
             system_instruction = f"""
             You are a Senior Legal Counsel.
             User asked: {prompt}
             Refined as: {refined_prompt}
             
             CRITICAL: The external search tool FAILED. 
             You must answer based on your internal knowledge of Nigerian Law.
             
             REQUIRED: You MUST end your response with this exact disclaimer in italics:
             "⚠️ Note: Live web verification was unavailable. This response is based on general legal principles and internal training data."
             """
        else:
             system_instruction = f"""
             You are a Senior Legal Counsel.
             CONTEXT: {context_text}
             USER QUESTION: {prompt} (Refined as: {refined_prompt})
             
             TASK: 
             1. Answer strictly based on the provided context if available.
             2. If the user asks about "2026 reform", connect it to the "2025 Act" if evident in the text.
             3. Cite sources/URLs if available.
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
            
            # Show Citations only if we actually found docs
            if mode == "Strict Compliance (Docs)" and docs:
                with st.expander("⚖️  View Source Evidence"):
                    for doc in docs:
                        st.markdown(f"**Page {doc.metadata.get('page')} Reference:**")
                        st.caption(doc.page_content[:300] + "...")
                        st.markdown("---")
                    
        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Generation Error: {e}")

# Footer
st.markdown("<div class='footer'>Legal Mind AI • Powered by Llama-3.3 & Groq LPU</div>", unsafe_allow_html=True)