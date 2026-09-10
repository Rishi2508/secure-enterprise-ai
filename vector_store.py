"""
Vector store module for Enterprise PDF AI Assistant.
Manages embeddings generation, Chroma vector database persistence, and lifecycle.
"""

import os
import shutil
import time
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIRECTORY


def reset_vector_database(persist_path=CHROMA_PERSIST_DIRECTORY):
    """
    Safely deletes the local Chroma vector database directory.
    
    Args:
        persist_path (str): Path to Chroma persistence directory.
        
    Returns:
        bool: True if deleted successfully, False otherwise.
    """
    if os.path.exists(persist_path):
        try:
            shutil.rmtree(persist_path)
            return True
        except PermissionError:
            time.sleep(1)
            try:
                shutil.rmtree(persist_path)
                return True
            except Exception as e:
                st.warning(f"Could not fully reset DB: {e}")
                return False
    return False


def create_vector_database(chunks, persist_path=CHROMA_PERSIST_DIRECTORY):
    """
    Generates embeddings for text chunks and initializes a Chroma vector store.
    
    Args:
        chunks (list): Document chunks to embed.
        persist_path (str): Directory where the vector store is saved.
        
    Returns:
        Chroma: Configured vector store instance.
    """
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    # SAFE DELETE (Windows compatible)
    if os.path.exists(persist_path):
        reset_vector_database(persist_path)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )

    return db
