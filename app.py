import streamlit as st
import os
import time
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# 1. CONFIGURATION
st.set_page_config(page_title="Legal Mind AI", page_icon="⚖️", layout="wide")

# 2. CSS STYLING
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp { background-color: #050505; font-family: 'Inter', sans-serif; }
    header[data-testid="stHeader"] { background: transparent; }
    
    .stChatInputContainer textarea { 
        background-color: #121212 !important; 
        border: 1px solid #333 !important; 
        color: #e0e0e0 !important; 
    }
    [data-testid="stSidebar"] { 
        background-color: #0a0a0a; 
        border-right: 1px solid #222; 
    }
    
    /* NEURAL PULSE */
    .neural-loader { display: flex; justify-content: center; align-items: center; height: 60px; gap: 8px; }
    .bar { width: 6px; height: 20px; background: linear-gradient(180deg, #D4AF37, #AA8C2C); border-radius: 3px; animation: pulse 1s ease-in-out infinite; }
    .bar:nth-child(1) { animation-delay: 0.0s; height: 20px; }
    .bar:nth-child(2) { animation-delay: 0.1s; height: 35px; }
    .bar:nth-child(3) { animation-delay: 0.2s; height: 45px; }
    .bar:nth-child(4) { animation-delay: 0.3s; height: 35px; }
    .bar:nth-child(5) { animation-delay: 0.4s; height: 20px; }
    
    @keyframes pulse { 0% { opacity: 0.6; } 50% { transform: scaleY(1.5); opacity: 1; } 100% { opacity: 0.6; } }
    </style>
""", unsafe_allow_html=True)

# 3. STABLE ENGINE (Low Memory Usage)

def process_files_safe(uploaded_files):
    if not uploaded_files: return None
    
    main_vector_store = None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    status_text = st.empty()
    progress_bar = st.progress(0, text="Initializing Safe Mode...")
    
    total_docs_processed = 0
    
    try:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)
            
            # PROCESS IN BATCHES OF 20 PAGES (Prevents Memory Crash)
            batch_size = 20
            for start_page in range(0, total_pages, batch_size):
                end_page = min(start_page + batch_size, total_pages)
                
                # 1. Extract Text for this batch only
                batch_docs = []
                for i in range(start_page, end_page):
                    page = doc[i]
                    text = page.get_text()
                    if text:
                        batch_docs.append(Document(
                            page_content=text, 
                            metadata={"page": i+1, "source": uploaded_file.name}
                        ))
                
                if not batch_docs:
                    continue
                    
                # 2. Split
                chunks = text_splitter.split_documents(batch_docs)
                
                # 3. Embed immediately and discard text
                if chunks:
                    if main_vector_store is None:
                        main_vector_store = FAISS.from_documents(chunks, embeddings)
                    else:
                        main_vector_store.add_documents(chunks)
                
                # Update UI
                progress = min(end_page / total_pages, 1.0)
                progress_bar.progress(progress, text=f"Processing Pages {start_page}-{end_page} of {total_pages}...")
                
                # Force Memory Cleanup
                del batch_docs
                del chunks
                
            doc.close()
            
    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
        return None

    progress_bar.empty()
    status_text.empty()
    
    if main_vector_store:
        st.toast("System Online (Safe Mode)", icon="✅")
        return main_vector_store
    else:
        st.error("Could not extract any text.")
        return None

# 4. UI ORCHESTRATION

if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/924/924915.png", width=40)
    st.markdown("### Legal Mind AI")
    st.caption("v3.6 • Stable Core")
    st.markdown("---")
    
    api_key = st.text_input("🔑 API Credentials", type="password", placeholder="Paste Google Key")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    uploaded_files = st.file_uploader("📂 Upload Case Files", type="pdf", accept_multiple_files=True)
    
    # TRIGGER BUTTON
    if st.button("⚡ Analyze Documents", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter an API Key first.")
        elif not uploaded_files:
            st.error("Please upload a PDF first.")
        else:
            with st.spinner("Processing large file (Page by Page)..."):
                store = process_files_safe(uploaded_files)
                if store:
                    st.session_state.vector_store = store
                    st.success("Indexing Complete.")
                    st.rerun()
    
    if st.session_state.vector_store is not None:
        st.success(f"● Online")
        
    if st.button("↻ Reset System", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.rerun()

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

    if st.session_state.vector_store is not None:
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
                with st.expander("⚖️  Evidence Locker (Citations)"):
                    for doc in docs:
                        st.markdown(f"**Section Reference:**")
                        st.caption(doc.page_content[:300] + "...")
                        st.markdown("---")
            except Exception as e:
                 message_placeholder.error(f"Generation Error: {e}")

    else:
        st.warning("Please upload a document and click 'Analyze Documents' to begin.")