from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import tempfile
import time
import re
import streamlit as st

from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings
)

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    Chroma
)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Enterprise PDF AI",
    page_icon="📄",
    layout="wide"
)

# ============================================
# SESSION STATE
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "queries" not in st.session_state:
    st.session_state.queries = 0

if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0

# ============================================
# HEADER
# ============================================

st.title("📄 Enterprise PDF AI Assistant")

st.markdown(
"""
Secure Local RAG Application

Powered by

- Ollama (Llama3)
- ChromaDB
- LangChain
- Streamlit

Designed for Secure Enterprise Document Question Answering.
"""
)

# ============================================
# DASHBOARD
# ============================================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "LLM",
        "Llama3"
    )

with c2:
    st.metric(
        "Embeddings",
        "Nomic"
    )

with c3:
    st.metric(
        "Vector DB",
        "Chroma"
    )

with c4:
    st.metric(
        "Mode",
        "Local"
    )

st.divider()

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.title("⚙ Enterprise Dashboard")

    st.markdown("---")

    st.subheader("Project")

    st.write("Enterprise PDF Question Answering")

    st.write("Architecture")

    st.success("Local RAG")

    st.success("Private AI")

    st.success("Persistent Vector Store")

    st.markdown("---")

    st.subheader("Security")

    st.info("✓ Local Processing")

    st.info("✓ No Cloud APIs")

    st.info("✓ Local Embeddings")

    st.info("✓ Local LLM")

    st.markdown("---")

    st.subheader("Current Session")

    st.metric(
        "Questions Asked",
        st.session_state.queries
    )

    st.metric(
        "Chat Messages",
        len(st.session_state.messages)
    )

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages=[]

        st.rerun()

    if st.button("🧹 Reset Vector Database"):

        if os.path.exists("./chroma_db"):

            shutil.rmtree("./chroma_db")

            st.success("Database Deleted")

    st.markdown("---")

    st.caption(
        "Enterprise AI Demo\n\nVersion 1.0"
    )


# ============================================
# FUNCTIONS
# ============================================

def save_pdf(uploaded):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded.read())

        return tmp.name


# --------------------------------------------

def split_pdf(path):

    loader = PyPDFLoader(path)

    docs = loader.load()
    if len(docs) == 0:
        st.error("No readable text found.")
        st.stop()
    splitter = RecursiveCharacterTextSplitter(

        chunk_size=600,
        chunk_overlap=100

    )

    chunks = splitter.split_documents(docs)

    return docs, chunks


# --------------------------------------------

def create_vector_database(chunks):

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    persist_path = "./chroma_db"

    # SAFE DELETE (Windows compatible)
    if os.path.exists(persist_path):
        try:
            shutil.rmtree(persist_path)
        except PermissionError:
            # fallback cleanup strategy
            time.sleep(1)
            try:
                shutil.rmtree(persist_path)
            except Exception as e:
                st.warning(f"Could not fully reset DB: {e}")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )

    return db


# --------------------------------------------
def sanitize_context(context):

    # remove database credentials / secrets
    patterns = [
        r"password\s*[:=]\s*\S+",
        r"api\s*key\s*[:=]\s*\S+",
        r"secret\s*[:=]\s*\S+",
        r"token\s*[:=]\s*\S+"
    ]

    for pattern in patterns:
        context = re.sub(pattern, "[REDACTED]", context, flags=re.IGNORECASE)

    return context
def retrieve_context(db, query):

    related = db.similarity_search_with_score(
    query,
    k=5
    )

    context = ""

    sources = []

    for doc, score in related:

        page = doc.metadata.get("page","?")
        context += f"\n\n[Page {page+1}] (Similarity: {score:.4f})\n"
        sources.append({
        "page": page + 1,
        "score": score
        })
        context += doc.page_content

    return context, sources


# --------------------------------------------

def generate_answer(context, query):

    llm = ChatOllama(

        model="llama3",

        temperature=0

    )

    prompt = f"""

You are an Enterprise AI Assistant.

Your only knowledge source is the retrieved content from the uploaded PDF.

Strict Rules

1. Never use outside knowledge.

2. Never hallucinate.

3. Never guess.

4. If information is missing say:

"The requested information was not found in the uploaded PDF."

5. Never expose:

- Passwords

- API Keys

- Access Tokens

- Secret Credentials

- Personally Identifiable Information

unless the user is explicitly authorized.

6. Preserve:

• dates

• names

• numbers

• financial values

exactly as written.

7. If the document appears unreadable or contains insufficient context,

clearly state that.

8. Format using concise bullet points.

9. Mention uncertainty whenever appropriate.
11.
Every factual statement must come from the retrieved context.

12.
Do not combine unrelated chunks.

13.
If retrieved context is insufficient, refuse politely.

14.
Never summarize information that was not retrieved.

15.
Mention page numbers whenever possible.

Retrieved Context

{context}

User Question

{query}

"""

    response = llm.invoke(prompt)

    return response.content


# --------------------------------------------

def get_file_size(uploaded):

    return round(

        uploaded.size / 1024,

        2

    )


# --------------------------------------------

def download_button(answer):

    st.download_button(

        "⬇ Download Answer",

        answer,

        file_name="answer.txt"

    )

# =====================================
# MAIN UI (ENTERPRISE UPGRADE)
# =====================================

uploaded = st.file_uploader(
    "📤 Upload your PDF document",
    type="pdf"
)

if uploaded:

    start_time = time.time()

    # =====================================
    # PDF METADATA PANEL
    # =====================================

    col1, col2, col3 = st.columns(3)

    file_size_kb = get_file_size(uploaded)

    with col1:
        st.metric("File Name", uploaded.name)

    with col2:
        st.metric("File Size (KB)", file_size_kb)

    with col3:
        st.metric("Status", "Uploaded")

    # =====================================
    # PROCESSING
    # =====================================

    with st.spinner("📄 Processing PDF & Building Knowledge Base..."):

        path = save_pdf(uploaded)

        docs, chunks = split_pdf(path)

        db = create_vector_database(chunks)

        processing_time = round(time.time() - start_time, 2)

        st.session_state.processing_time = processing_time

        # cleanup temp file
        try:
            os.remove(path)
        except:
            pass

    st.success("✅ PDF indexed successfully")

    st.info(f"📑 Pages Extracted: {len(docs)}")
    st.info(f"🧩 Chunks Created: {len(chunks)}")
    st.info(f"⚡ Processing Time: {processing_time} sec")
    st.info("🗄 Vector DB stored in ./chroma_db")

    st.divider()

    # =====================================
    # CHAT HISTORY DISPLAY
    # =====================================

    if st.session_state.messages:

        st.subheader("💬 Chat History")

        for msg in st.session_state.messages:

            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # =====================================
    # CHAT INPUT
    # =====================================

    query = st.chat_input("Ask a question from your PDF")

    if query:

        st.session_state.queries += 1

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("🧐 Searching document..."):

            context, sources = retrieve_context(db, query)

            context = sanitize_context(context)

            with st.expander("📂 Retrieved Context (Debug View)"):
                st.write(context)

            answer = generate_answer(context, query)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):

            st.markdown("### 🧠 Answer")
            st.write(answer)

            # =====================================
            # SOURCE DISPLAY
            # =====================================

            if sources:
                st.markdown("### 📄 Sources")

            for source in sources:
                st.write(
                f"Page {source['page']}  |  Similarity: {source['score']:.4f}"
                ) 

            # =====================================
            # DOWNLOAD BUTTON
            # =====================================

            download_button(answer)

    # =====================================
    # FOOTER
    # =====================================

    st.divider()

    st.caption(
        "🔒 Enterprise PDF AI | Local RAG System | No Data Leaves Device"
    )

else:

    st.info("😐 Upload a PDF to start querying your documents")