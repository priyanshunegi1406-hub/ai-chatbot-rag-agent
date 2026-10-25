"""
Conversational RAG Pipeline for the Chatbot.
Integrates Google Gemini LLM with ChromaDB vector store using LangChain LCEL,
handling chat history, contextual question rephrasing, and source citation extraction.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from config import config
from ingest import get_vector_store

import warnings
# Filter DDG rename warning
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

from web_search_agent import search_web_and_scrape

logger = logging.getLogger("RAGChain")

# Initialize a lightweight, high-performance cross-encoder model
# (Downloads automatically on first run)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query: str, docs: List[Document], top_n: int = 4) -> List[Document]:
    """
    Takes a larger pool of retrieved documents (local + web) and re-ranks them 
    using a Cross-Encoder to return the absolute best top_n documents.
    """
    if not docs:
        return []
    
    # Prepare pairs of [query, document_content]
    pairs = [[query, doc.page_content] for doc in docs]
    
    # Get relevance scores from the cross-encoder
    scores = reranker_model.predict(pairs)
    
    # Attach scores to documents and sort descending
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    # Return top_n documents
    best_docs = [doc for doc, score in scored_docs[:top_n]]
    return best_docs


def get_gemini_llm(
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    model_name: Optional[str] = None
) -> ChatGoogleGenerativeAI:
    """
    Initializes the Google Gemini Chat Model.
    """
    valid_key = config.validate_api_key(api_key)
    return ChatGoogleGenerativeAI(
        model=model_name or config.llm_model,
        google_api_key=valid_key,
        temperature=temperature if temperature is not None else config.temperature,
        top_p=config.top_p,
        max_output_tokens=config.max_output_tokens
    )


def build_rag_chain(
    api_key: Optional[str] = None,
    k: Optional[int] = None,
    temperature: Optional[float] = None,
    enable_web_search: bool = False
):
    """
    Constructs the end-to-end Conversational Hybrid Retrieval Chain.
    Supports local ChromaDB search, dynamic web search scraping, cross-encoder re-ranking,
    and grounded multi-source QA synthesis.
    """
    top_n = k or config.retrieval_k
    # Retrieve a larger pool of candidate chunks from ChromaDB
    initial_k = max(20, top_n * 3)

    # 1. Base ChromaDB retriever
    vector_store = get_vector_store(api_key)
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": initial_k}
    )

    # 2. Hybrid Retrieval + Re-ranking step using Cross-Encoder
    def retrieve_and_rerank(query: str) -> List[Document]:
        local_docs: List[Document] = []
        try:
            local_docs = base_retriever.invoke(query)
            logger.info(f"Retrieved {len(local_docs)} candidate chunks from local ChromaDB for query: '{query}'")
        except Exception as exc:
            logger.warning(f"Error querying local ChromaDB (possibly empty vector store): {exc}")
            local_docs = []

        web_docs: List[Document] = []
        if enable_web_search:
            try:
                logger.info(f"Agent mode enabled: Triggering live web search and scrape for: '{query}'")
                web_docs = search_web_and_scrape(query, max_results=3)
                logger.info(f"Retrieved {len(web_docs)} live web document chunks.")
            except Exception as exc:
                logger.error(f"Live web retrieval failed: {exc}", exc_info=True)
                web_docs = []

        # Combine into unified context pool
        combined_docs = local_docs + web_docs
        logger.info(f"Total unified candidate pool: {len(combined_docs)} (Local: {len(local_docs)}, Web: {len(web_docs)})")

        if not combined_docs:
            return []

        # If web search is active, ensure top_n is sufficient to cover both sources if present
        effective_top_n = max(top_n, 4) if enable_web_search else top_n
        reranked_docs = rerank_documents(query=query, docs=combined_docs, top_n=effective_top_n)
        logger.info(f"Re-ranker surfaced top {len(reranked_docs)} most relevant documents.")
        return reranked_docs

    reranking_retriever = RunnableLambda(retrieve_and_rerank)

    # 3. Initialize Gemini LLM
    llm = get_gemini_llm(api_key=api_key, temperature=temperature)

    # 4. Prompt for Contextualizing Question (History-aware)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference "
        "prior context in the conversation history, formulate a standalone question "
        "which can be understood without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, reranking_retriever, contextualize_q_prompt
    )

    # 5. Prompt for Question Answering based on Retrieved Context
    qa_system_prompt = (
        "You are an enterprise AI Research Assistant powered by Google Gemini and a "
        "Hybrid Retrieval-Augmented Generation (RAG) system with live web search capabilities.\n\n"
        "Use the following pieces of retrieved context (from local uploaded documents and/or live web search results) to answer the user's question accurately.\n\n"
        "STRICT GUIDELINES:\n"
        "1. Ground your answers strictly in the provided context.\n"
        "2. If the answer cannot be determined from the context or the context is empty, "
        "honestly state: 'I could not find the answer to that in the provided documents or web search results.' "
        "Do NOT invent or hallucinate information.\n"
        "3. Provide structured, readable answers using clear markdown, headings, and bullet points.\n"
        "4. When citing facts, explicitly reference the source:\n"
        "   - For local documents: Mention the document name and page number (e.g., [Document: filename.pdf, Page 2]).\n"
        "   - For web sources: Mention the website title and/or URL (e.g., [Source: https://...]).\n"
        "5. If information comes from both local files and web sources, synthesize them clearly, distinguishing internal private data from live web data.\n\n"
        "Retrieved Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 6. Combined Retrieval Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def format_citations(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Extracts structured source citations and snippet previews from retrieved documents (local or web).
    """
    citations = []
    seen = set()

    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        chunk_index = doc.metadata.get("chunk_index", i)
        is_web = bool(
            doc.metadata.get("is_web", False)
            or str(source).startswith("http://")
            or str(source).startswith("https://")
        )
        title = doc.metadata.get("title", str(page) if is_web else source)

        key = (source, str(page), chunk_index)
        if key in seen:
            continue
        seen.add(key)

        snippet = doc.page_content.strip()
        if len(snippet) > 350:
            snippet = snippet[:350] + "..."

        citations.append({
            "source": source,
            "page": page,
            "title": title,
            "is_web": is_web,
            "chunk_index": chunk_index,
            "snippet": snippet,
            "full_content": doc.page_content
        })

    return citations


def convert_chat_history(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """
    Converts list of role/content message dicts into LangChain BaseMessage objects.
    """
    langchain_history: List[BaseMessage] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            langchain_history.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_history.append(AIMessage(content=content))
    return langchain_history


def ask_rag(
    question: str,
    chat_history: List[Dict[str, str]],
    api_key: Optional[str] = None,
    k: Optional[int] = None,
    temperature: Optional[float] = None,
    enable_web_search: bool = False
) -> Dict[str, Any]:
    """
    Executes a RAG query (with optional live web search augmentation) and returns the answer along with structured citations.
    """
    if not question or not question.strip():
        return {
            "answer": "Please provide a valid question.",
            "citations": [],
            "web_search_enabled": enable_web_search,
            "error": "Empty query"
        }

    try:
        chain = build_rag_chain(
            api_key=api_key,
            k=k,
            temperature=temperature,
            enable_web_search=enable_web_search
        )
        formatted_history = convert_chat_history(chat_history)

        result = chain.invoke({
            "input": question.strip(),
            "chat_history": formatted_history
        })

        raw_context = result.get("context", [])
        citations = format_citations(raw_context)

        return {
            "answer": result.get("answer", "No answer generated."),
            "citations": citations,
            "web_search_enabled": enable_web_search,
            "error": None
        }
    except Exception as exc:
        logger.error(f"Error executing RAG chain: {exc}", exc_info=True)
        return {
            "answer": f"An error occurred while processing your query: {str(exc)}",
            "citations": [],
            "error": str(exc)
        }
