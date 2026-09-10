"""
Enterprise PDF AI Assistant - Main Application.
Orchestrates the Streamlit user interface, document ingestion, vector storage, and RAG workflow.
"""

import os
import time
from dotenv import load_dotenv
import streamlit as st

# Local module imports
from config import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    CHROMA_PERSIST_DIRECTORY
)
from document_processor import (
    save_pdf,
    split_pdf,
    get_file_size
)
from vector_store import (
    create_vector_database,
    reset_vector_database
)
from security import (
    sanitize_context
)
from rag_engine import (
    retrieve_context,
    generate_answer
)
from ui_components import (
    render_header,
    render_dashboard,
    render_sidebar,
    download_button,
    render_footer
)

# Export for backward compatibility
__all__ = [
    "save_pdf",
    "split_pdf",
    "create_vector_database",
    "reset_vector_database",
    "sanitize_context",
    "retrieve_context",
    "generate_answer",
    "get_file_size",
    "download_button"
]

# Load environment variables
load_dotenv()


def main():
    # ============================================
    # PAGE CONFIG
    # ============================================
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT
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
    # HEADER & DASHBOARD
    # ============================================
    render_header()
    render_dashboard()

    # ============================================
    # SIDEBAR
    # ============================================
    render_sidebar()

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
        st.info(f"🗄 Vector DB stored in {CHROMA_PERSIST_DIRECTORY}")

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
        render_footer()

    else:
        st.info("😐 Upload a PDF to start querying your documents")


if __name__ == "__main__":
    main()