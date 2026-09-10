"""
Document processing module for Enterprise PDF AI Assistant.
Handles PDF saving, text extraction, document chunking, and file statistics.
"""

import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def save_pdf(uploaded):
    """
    Saves an uploaded PDF file to a temporary file on disk.
    
    Args:
        uploaded: Streamlit UploadedFile object.
        
    Returns:
        str: Absolute path to the temporary PDF file.
    """
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:
        tmp.write(uploaded.read())
        return tmp.name


def split_pdf(path):
    """
    Loads and splits a PDF into manageable text chunks.
    
    Args:
        path (str): Filepath to the PDF document.
        
    Returns:
        tuple: (list of documents, list of chunked documents)
    """
    loader = PyPDFLoader(path)
    docs = loader.load()
    if len(docs) == 0:
        st.error("No readable text found.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)
    return docs, chunks


def get_file_size(uploaded):
    """
    Calculates the size of an uploaded file in kilobytes (KB).
    
    Args:
        uploaded: Streamlit UploadedFile object.
        
    Returns:
        float: File size rounded to 2 decimal places.
    """
    return round(
        uploaded.size / 1024,
        2
    )
