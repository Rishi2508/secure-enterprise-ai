"""
UI components module for Enterprise PDF AI Assistant.
Encapsulates Streamlit interface elements including dashboards, sidebars, and widgets.
"""

import streamlit as st
from vector_store import reset_vector_database
from config import CHROMA_PERSIST_DIRECTORY


def render_header():
    """Renders the main application title and architecture badges."""
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


def render_dashboard():
    """Renders top-level operational KPI metrics."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("LLM", "Llama3")

    with c2:
        st.metric("Embeddings", "Nomic")

    with c3:
        st.metric("Vector DB", "Chroma")

    with c4:
        st.metric("Mode", "Local")

    st.divider()


def render_sidebar():
    """Renders the enterprise sidebar with project details, security info, and controls."""
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
            st.session_state.messages = []
            st.rerun()

        if st.button("🧹 Reset Vector Database"):
            if reset_vector_database(CHROMA_PERSIST_DIRECTORY):
                st.success("Database Deleted")

        st.markdown("---")

        st.caption(
            "Enterprise AI Demo\n\nVersion 1.0"
        )


def download_button(answer):
    """Renders a button allowing the user to download the generated answer as a text file."""
    st.download_button(
        "⬇ Download Answer",
        answer,
        file_name="answer.txt"
    )


def render_footer():
    """Renders the privacy and compliance footer."""
    st.divider()
    st.caption(
        "🔒 Enterprise PDF AI | Local RAG System | No Data Leaves Device"
    )
