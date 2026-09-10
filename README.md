# Secure Enterprise AI - PDF Assistant

Developed a secure enterprise RAG (Retrieval-Augmented Generation) application that retrieves information from PDF documents, generates grounded responses using local LLMs, and incorporates security controls such as prompt guardrails and sensitive data protection.

---

## 🏛️ Modular Architecture Overview

The project is structured according to clean code principles and single-responsibility modularity:

```
ai_pdf_chat/
│
├── config.py              # Centralized configuration, model parameters, chunking & prompts
├── document_processor.py   # PDF ingestion, parsing, chunking, and file metrics
├── vector_store.py        # Embeddings generation, Chroma vector database lifecycle & persistence
├── security.py            # Guardrails, sensitive data regex redaction (passwords, tokens, keys)
├── rag_engine.py          # Similarity search, context building, and LLM response generation
├── ui_components.py       # Streamlit UI elements, metrics dashboard, sidebar & downloads
├── app.py                 # Main application orchestrator and entry point
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation and architectural guide
```

---

## ⚙️ Technology Stack

- **Frontend & Dashboard:** Streamlit
- **LLM Engine:** Ollama (Llama 3) - 100% Local Inference
- **Embeddings:** Nomic Embed Text (`nomic-embed-text`)
- **Vector Database:** ChromaDB (Local Persistent Storage)
- **Document Processing:** LangChain (`PyPDFLoader`, `RecursiveCharacterTextSplitter`)

---

## 🚀 Running the Application

1. **Activate Virtual Environment:**
   ```bash
   .\venv\Scripts\activate
   ```

2. **Run Application:**
   ```bash
   streamlit run app.py
   ```
