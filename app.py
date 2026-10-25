"""
Streamlit Web Application for the Enterprise Gemini Hybrid RAG & Live Web-Surfing Assistant.
Provides a modern dual-mode chat interface, document uploader, persistence metrics,
agent toggle, and interactive citation cards for both local files and live web sources.
"""

import os
from pathlib import Path
import streamlit as st

from config import config
from ingest import (
    ingest_documents,
    get_vector_store_stats,
    reset_vector_store,
    get_vector_store,
    get_all_indexed_documents,
    delete_documents_by_source
)
from rag_chain import ask_rag

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gemini Hybrid RAG & Web-Surfing Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern enterprise aesthetics
st.markdown("""
<style>
    /* Global Page adjustments */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header & Badge Styling */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-primary {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
    }
    .badge-success {
        background-color: #10b981;
        color: white;
    }
    .badge-warning {
        background-color: #f59e0b;
        color: white;
    }
    .badge-info {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
        color: white;
    }
    .badge-secondary {
        background-color: #64748b;
        color: white;
    }
    
    /* Citation Cards */
    .citation-card {
        background: rgba(240, 244, 248, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .citation-header {
        font-weight: 600;
        font-size: 0.85rem;
        color: #38bdf8;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .citation-snippet {
        font-size: 0.82rem;
        opacity: 0.88;
        line-height: 1.4;
    }
    
    /* Sidebar info box */
    .sidebar-metric {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


def render_citation_card(cite: dict, idx: int):
    """
    Renders an interactive citation card styled distinctly for local documents vs live web URLs.
    """
    source_name = cite.get("source", "Unknown Source")
    page_num = cite.get("page", 1)
    snippet = cite.get("snippet", "")
    is_web = bool(
        cite.get("is_web", False)
        or str(source_name).startswith("http://")
        or str(source_name).startswith("https://")
    )
    title = cite.get("title", source_name)

    if is_web:
        card_html = f"""
        <div class="citation-card" style="border-left: 3px solid #06b6d4;">
            <div class="citation-header">
                <span>🌐 <a href="{source_name}" target="_blank" rel="noopener noreferrer" style="color: #38bdf8; text-decoration: none; font-weight: 600;">{title}</a></span>
                <span class="badge badge-info" style="font-size: 0.72rem;">Live Web #{idx}</span>
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 5px; word-break: break-all;">
                🔗 <a href="{source_name}" target="_blank" rel="noopener noreferrer" style="color: #64748b; text-decoration: underline;">{source_name}</a>
            </div>
            <div class="citation-snippet">"{snippet}"</div>
        </div>
        """
    else:
        card_html = f"""
        <div class="citation-card" style="border-left: 3px solid #8b5cf6;">
            <div class="citation-header">
                <span>📄 {source_name} (Page {page_num})</span>
                <span class="badge badge-primary" style="font-size: 0.72rem;">Document #{idx}</span>
            </div>
            <div class="citation-snippet">"{snippet}"</div>
        </div>
        """
    st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = config.gemini_api_key or ""


# -----------------------------------------------------------------------------
# Sidebar: Configuration, Ingestion & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ System Control")
    st.caption("Google Gemini Hybrid RAG & Live Web Agent")

    # 1. API Key Input / Status
    st.markdown("### 🔑 API Configuration")
    user_api_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Reads from GEMINI_API_KEY / GOOGLE_API_KEY by default. Overriding here persists for this session."
    )
    if user_api_key:
        st.session_state.api_key = user_api_key

    has_valid_key = bool(st.session_state.api_key and st.session_state.api_key != "your_gemini_api_key_here")
    if has_valid_key:
        st.success("API Key detected and active", icon="✅")
    else:
        st.error("No valid API Key detected. Please enter one above or set GEMINI_API_KEY in .env", icon="⚠️")

    st.markdown("---")

    # 2. AI Agent Mode Toggle
    st.markdown("### 🌐 AI Agent Mode")
    agent_mode = st.sidebar.toggle(
        "Enable Hybrid Web Search Agent",
        value=True,
        help="Combines your local documents with live internet searches."
    )
    if agent_mode:
        st.caption("🟢 **Hybrid Mode Active**: Blends local ChromaDB knowledge with live DuckDuckGo web surfing.")
    else:
        st.caption("📁 **Local Only Mode**: Retrieval strictly searches persistent ChromaDB documents.")

    st.markdown("---")

    # 3. Vector Database Stats
    st.markdown("### 📊 Vector Index Metrics")
    stats = get_vector_store_stats(st.session_state.api_key if has_valid_key else None)
    total_chunks = stats.get("total_chunks", 0)

    st.markdown(f"""
    <div class="sidebar-metric">
        <div><strong>Status:</strong> {'🟢 Indexed' if total_chunks > 0 else '🟠 Empty'}</div>
        <div><strong>Total Chunks:</strong> <code>{total_chunks}</code></div>
        <div><strong>Collection:</strong> <code>{stats.get('collection_name', 'rag_documents')}</code></div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Document Ingestion / Upload Section
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Uploaded files will be saved to the /data folder and automatically embedded into ChromaDB."
    )

    if uploaded_files:
        if not has_valid_key:
            st.error("Please provide a valid Gemini API Key before uploading and indexing.")
        else:
            if st.button("📥 Process & Index Uploaded Files", type="primary", use_container_width=True):
                with st.spinner("Processing, chunking, and embedding documents..."):
                    new_paths = []
                    for uploaded_file in uploaded_files:
                        save_path = config.data_dir / uploaded_file.name
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        new_paths.append(save_path)

                    processed_files, new_chunks = ingest_documents(
                        file_paths=new_paths,
                        api_key=st.session_state.api_key
                    )
                    st.success(f"Indexed {processed_files} files into {new_chunks} chunks!")
                    st.rerun()

    # Re-index all files button
    if st.button("🔄 Sync All Documents from /data", use_container_width=True):
        if not has_valid_key:
            st.error("Please provide a valid Gemini API Key first.")
        else:
            with st.spinner("Scanning and indexing all files in /data..."):
                files_cnt, chunks_cnt = ingest_documents(api_key=st.session_state.api_key)
                st.success(f"Synchronized {files_cnt} files ({chunks_cnt} chunks).")
                st.rerun()

    # --- Document Management Dashboard inside Sidebar ---
    st.markdown("---")
    st.subheader("📂 Document Management")

    if has_valid_key:
        try:
            vector_store = get_vector_store(st.session_state.api_key)
            if vector_store:
                indexed_docs = get_all_indexed_documents(vector_store)
                
                if indexed_docs:
                    st.write(f"**Indexed Files ({len(indexed_docs)}):**")
                    for filename, chunk_count in indexed_docs.items():
                        cols = st.columns([3, 1])
                        cols[0].text(f"{filename} ({chunk_count} chunks)")
                        if cols[1].button("🗑️", key=f"del_{filename}", help=f"Delete {filename}"):
                            success = delete_documents_by_source(vector_store, filename)
                            if success:
                                st.success(f"Deleted {filename}!")
                                st.rerun()
                            else:
                                st.error("Failed to delete.")
                else:
                    st.info("No documents currently indexed.")
        except Exception as e:
            st.warning(f"Could not load indexed documents: {e}")
    else:
        st.info("Enter a valid API key to view and manage indexed documents.")

    st.markdown("---")

    # 5. Hyperparameter Controls
    st.markdown("### 🎛️ Retrieval Settings")
    retrieval_k = st.slider(
        "Top K Context Chunks",
        min_value=1,
        max_value=10,
        value=config.retrieval_k,
        help="Number of relevant document chunks retrieved per question."
    )
    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=config.temperature,
        step=0.05,
        help="Lower values make answers more factual and deterministic."
    )

    st.markdown("---")

    # 6. Database & Session Management
    col_clear, col_reset = st.columns(2)
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_reset:
        if st.button("🗑️ Reset DB", use_container_width=True, help="Deletes all vectors in ChromaDB"):
            if reset_vector_store(st.session_state.api_key if has_valid_key else None):
                st.warning("Vector store cleared.")
                st.rerun()


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
# Header
st.markdown(f"""
<div class="main-header">
    <div>
        <h1 style="margin: 0; font-size: 1.85rem;">🧠 Gemini Hybrid RAG & Live Web Agent</h1>
        <p style="margin: 0.2rem 0 0 0; opacity: 0.75; font-size: 0.95rem;">
            Dual-Mode Intelligence: Grounded Local Document RAG + Dynamic Live Web Surfing
        </p>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
        <span class="badge badge-primary">{config.llm_model}</span>
        <span class="badge {'badge-info' if agent_mode else 'badge-secondary'}">
            {'🌐 Web Agent Active' if agent_mode else '📁 Local Only'}
        </span>
        <span class="badge {'badge-success' if total_chunks > 0 else 'badge-warning'}">
            {f'{total_chunks} chunks indexed' if total_chunks > 0 else 'Local DB Empty'}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Guidance if index is empty
if total_chunks == 0 and not agent_mode:
    st.info(
        "👋 **Welcome! Your local vector database is currently empty.**\n\n"
        "To get started:\n"
        "1. Upload PDF, TXT, or MD documents using the **Sidebar Uploader** (or place files in `./data/`).\n"
        "2. Click **Process & Index Uploaded Files**.\n"
        "3. Or toggle **Enable Hybrid Web Search Agent** in the sidebar to surf the live web immediately!",
        icon="ℹ️"
    )
elif total_chunks == 0 and agent_mode:
    st.info(
        "🌐 **Hybrid Web Search Agent is Active!**\n\n"
        "You can ask any question to surf the live internet on the fly. "
        "Upload local documents in the sidebar anytime to query your private files and live web simultaneously.",
        icon="🚀"
    )

# Render Chat History
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    citations = message.get("citations", [])

    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        st.markdown(content)

        # Render structured citation accordion if present
        if citations:
            with st.expander(f"📚 Verified Sources & Citations ({len(citations)})", expanded=False):
                for idx, cite in enumerate(citations, 1):
                    render_citation_card(cite, idx)


# Sample quick prompt buttons if history is empty
if len(st.session_state.messages) == 0:
    st.markdown("**Suggested Quick Prompts:**")
    prompt_cols = st.columns(3)
    if agent_mode and total_chunks > 0:
        sample_prompts = [
            "Summarize the main findings in my uploaded documents.",
            "Compare the concepts in my uploaded notes with current developments on the live web.",
            "Search the live web for the latest updates related to the topics in my files."
        ]
    elif agent_mode:
        sample_prompts = [
            "What are the latest breakthroughs in AI agents and multimodal models this week?",
            "Search the live web for the current status and key features of Python 3.13.",
            "What are the top AI conferences scheduled for this year?"
        ]
    else:
        sample_prompts = [
            "What are the main topics and objectives discussed in the documents?",
            "Provide a structured executive summary with key takeaways.",
            "List all critical recommendations or conclusions mentioned."
        ]

    for col, prompt_text in zip(prompt_cols, sample_prompts):
        if col.button(f"💡 {prompt_text}", use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.rerun()

# Check for pending prompt from suggested buttons
chat_placeholder = (
    "Ask a question (searching local documents & surfing live web)..."
    if agent_mode
    else "Ask a question about your indexed documents..."
)
user_query = st.chat_input(chat_placeholder)
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    user_query = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# Handle User Query
if user_query:
    # 1. API key validation
    if not has_valid_key:
        st.error("Please configure your Gemini API Key in the sidebar or `.env` file to chat.")
        st.stop()

    # 2. Append & render user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)

    # 3. Generate Assistant response
    with st.chat_message("assistant", avatar="🤖"):
        spinner_msg = (
            "Surfing the live web & retrieving local documents with Google Gemini..."
            if agent_mode
            else "Analyzing indexed documents and generating verified answer..."
        )
        with st.spinner(spinner_msg):
            result = ask_rag(
                question=user_query,
                chat_history=st.session_state.messages[:-1],  # prior history
                api_key=st.session_state.api_key,
                k=retrieval_k,
                temperature=temperature,
                enable_web_search=agent_mode
            )

            answer = result.get("answer", "No answer could be retrieved.")
            citations = result.get("citations", [])

            st.markdown(answer)

            if citations:
                with st.expander(f"📚 Verified Sources & Citations ({len(citations)})", expanded=True):
                    for idx, cite in enumerate(citations, 1):
                        render_citation_card(cite, idx)

            # Save to conversation memory
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })
