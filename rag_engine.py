"""
RAG (Retrieval-Augmented Generation) engine for Enterprise PDF AI Assistant.
Handles similarity search over vector stores and LLM response generation.
"""

from langchain_ollama import ChatOllama

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    SIMILARITY_TOP_K,
    SYSTEM_PROMPT_TEMPLATE
)


def retrieve_context(db, query, k=SIMILARITY_TOP_K):
    """
    Performs similarity search on the vector store and formats retrieved context with source attribution.
    
    Args:
        db: Chroma vector store instance.
        query (str): User question.
        k (int): Number of top similar chunks to retrieve.
        
    Returns:
        tuple: (formatted_context_str, list of source dicts with page and similarity score)
    """
    related = db.similarity_search_with_score(
        query,
        k=k
    )

    context = ""
    sources = []

    for doc, score in related:
        page = doc.metadata.get("page", "?")
        context += f"\n\n[Page {page+1}] (Similarity: {score:.4f})\n"
        sources.append({
            "page": page + 1,
            "score": score
        })
        context += doc.page_content

    return context, sources


def generate_answer(context, query):
    """
    Generates a grounded, hallucination-free answer using a local LLM via Ollama.
    
    Args:
        context (str): Sanitized context from retrieved document chunks.
        query (str): User question.
        
    Returns:
        str: Assistant response text.
    """
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )

    prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)
    response = llm.invoke(prompt)

    return response.content
