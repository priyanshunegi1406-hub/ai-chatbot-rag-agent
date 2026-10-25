# 🧠 Enterprise Gemini RAG Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot application built with **LangChain (LCEL)**, **Google Gemini LLM**, **Google AI Embeddings**, **ChromaDB** (local vector store with disk persistence), and **Streamlit**.

---

## 🌟 Key Features

- **Google Gemini Integration**: Powered by `gemini-3.8-flash` for high-speed, cost-effective reasoning, and Google AI Embeddings for vector representations.
- **Hybrid Live Web-Surfing AI Agent**: Surf the live internet on the fly via DuckDuckGo and BeautifulSoup to gather real-time web context alongside private document data.
- **Cross-Encoder Re-Ranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to score and re-rank both local document chunks and web results into a unified context pool.
- **Smart Agent Mode Toggle**: Easily switch between **Hybrid Web Agent** (combining local files + live internet) and **Local Files Only** with a single click.
- **Persistent Local ChromaDB**: Embeddings and chunk metadata persist locally on disk (`./chroma_db`) across restarts—no redundant embedding calls.
- **Conversational Memory**: Chat history-aware retrieval chain that contextualizes follow-up questions before document retrieval.
- **Interactive Dual Citation Cards**: Differentiates between local documents (filename, page number, snippet) and live web sources (clickable URL link, page title, excerpt).
- **Dual Ingestion Workflow**:
  - Ingest documents directly from the `./data/` folder via CLI or the UI.
  - Interactive file upload via the Streamlit sidebar (supports `.pdf`, `.txt`, `.md`).
- **Production Containerization**: Dockerfile configured for non-root execution, healthchecks, and direct deployment to Cloud Run, Render, or AWS ECS.

---

## 📂 Project Structure

```text
ai chatbot/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules for keys, db, and caches
├── Dockerfile                # Production Docker multi-stage container
├── .dockerignore             # Excludes unnecessary files from container builds
├── README.md                 # Complete documentation
├── requirements.txt          # Python dependencies
├── tasks.md                  # Implementation plan and tracking
├── config.py                 # Centralized configuration and validation
├── ingest.py                 # Document processing, chunking, and ChromaDB persistence
├── rag_chain.py              # Conversational LCEL pipeline and citation extraction
├── app.py                    # Modern Streamlit UI with chat interface & sidebar
├── data/                     # Source documents directory (PDF, TXT, MD)
└── chroma_db/                # Local persistent Chroma vector database
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Google Gemini API Key ([Get one free at Google AI Studio](https://aistudio.google.com/))

### 2. Environment Setup

Clone or open the workspace and copy the environment template:

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

*(Note: You can also enter or override your API key directly in the Streamlit web sidebar during runtime).*

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Ingest Documents via Command Line

Place your `.pdf`, `.txt`, or `.md` files in the `./data` directory, then run:

```bash
python ingest.py
```

### 5. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

The application will be accessible at: `http://localhost:8501`.

---

## 🐳 Docker Deployment

### Build the Docker Image
```bash
docker build -t gemini-rag-chatbot .
```

### Run Locally with Docker
```bash
docker run -p 8501:8501 -e GEMINI_API_KEY="your_api_key" gemini-rag-chatbot
```

### Deploy to Google Cloud Run
```bash
# Authenticate & submit build to Google Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/gemini-rag-chatbot

# Deploy service
gcloud run deploy gemini-rag-chatbot \
    --image gcr.io/YOUR_PROJECT_ID/gemini-rag-chatbot \
    --platform managed \
    --port 8501 \
    --set-env-vars GEMINI_API_KEY="your_gemini_api_key" \
    --allow-unauthenticated
```

### Deploy to Render
1. Create a **New Web Service** pointing to your repository.
2. Select **Docker** environment.
3. In the Render Dashboard under **Environment Variables**, add:
   - `GEMINI_API_KEY`: `your_gemini_api_key`
   - `PORT`: `8501`

---

## ⚙️ Architecture & Hyperparameters

| Variable | Default Value | Description |
|---|---|---|
| `LLM_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Google embedding model |
| `CHUNK_SIZE` | `1000` | Character count per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RETRIEVAL_K` | `4` | Top K chunks retrieved for context |
| `LLM_TEMPERATURE` | `0.2` | Controls deterministic factual generation |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db` | Disk directory for vector persistence |
