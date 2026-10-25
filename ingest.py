"""
Document Ingestion Pipeline for the RAG Chatbot.
Loads PDF and TXT documents, chunks them with RecursiveCharacterTextSplitter,
computes embeddings via GoogleGenerativeAIEmbeddings, and persists them into ChromaDB.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RAGIngest")


def get_embedding_function(api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Generative AI Embeddings model.
    """
    valid_key = config.validate_api_key(api_key)
    return GoogleGenerativeAIEmbeddings(
        model=config.embedding_model,
        google_api_key=valid_key
    )


def get_vector_store(api_key: Optional[str] = None) -> Chroma:
    """
    Initializes and returns a persistent ChromaDB vector store instance.
    """
    embedding_fn = get_embedding_function(api_key)
    return Chroma(
        collection_name=config.chroma_collection_name,
        embedding_function=embedding_fn,
        persist_directory=str(config.chroma_persist_dir)
    )


def load_single_document(file_path: Path) -> List[Document]:
    """
    Loads a single PDF or TXT file into LangChain Document objects.
    Preserves and cleans document metadata.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    docs: List[Document] = []

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        elif suffix in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
        else:
            logger.warning(f"Unsupported file format '{suffix}' for file: {file_path.name}")
            return []
    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        raise

    # Normalize metadata
    for i, doc in enumerate(docs):
        doc.metadata["source"] = file_path.name
        doc.metadata["file_path"] = str(file_path)
        if "page" not in doc.metadata:
            doc.metadata["page"] = i + 1

    return docs


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Splits documents into smaller semantic chunks with overlapping boundaries.
    """
    size = chunk_size or config.chunk_size
    overlap = chunk_overlap or config.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    # Add unique chunk ids and clean metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
    return chunks


def ingest_documents(
    file_paths: Optional[List[Path]] = None,
    api_key: Optional[str] = None
) -> Tuple[int, int]:
    """
    Loads, splits, embeds, and stores documents in ChromaDB.
    If file_paths is None, scans config.data_dir for all supported files.
    Returns (num_files_processed, num_chunks_created).
    """
    if file_paths is None:
        supported_exts = {".pdf", ".txt", ".md"}
        file_paths = [
            p for p in config.data_dir.glob("*")
            if p.is_file() and p.suffix.lower() in supported_exts
        ]

    if not file_paths:
        logger.info(f"No documents found to ingest in {config.data_dir}")
        return 0, 0

    all_raw_docs: List[Document] = []
    processed_count = 0

    for path in file_paths:
        try:
            logger.info(f"Processing document: {path.name}")
            docs = load_single_document(path)
            if docs:
                all_raw_docs.extend(docs)
                processed_count += 1
        except Exception as err:
            logger.error(f"Skipping {path.name} due to error: {err}")

    if not all_raw_docs:
        logger.warning("No readable content extracted from files.")
        return processed_count, 0

    # Chunking
    chunks = split_documents(all_raw_docs)
    logger.info(f"Split {len(all_raw_docs)} pages/sections into {len(chunks)} chunks.")

    # Embedding & Storage
    vector_store = get_vector_store(api_key)
    vector_store.add_documents(chunks)
    logger.info(f"Successfully persisted {len(chunks)} chunks to ChromaDB at {config.chroma_persist_dir}")

    return processed_count, len(chunks)


def get_vector_store_stats(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns statistics regarding the Chroma vector store.
    """
    try:
        store = get_vector_store(api_key)
        count = store._collection.count()
        return {
            "total_chunks": count,
            "persist_directory": str(config.chroma_persist_dir),
            "collection_name": config.chroma_collection_name,
            "status": "ready" if count > 0 else "empty"
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "error": str(e),
            "status": "uninitialized"
        }


def reset_vector_store(api_key: Optional[str] = None) -> bool:
    """
    Clears all documents from the vector store collection.
    """
    try:
        store = get_vector_store(api_key)
        store.delete_collection()
        logger.info("Successfully reset Chroma vector store collection.")
        return True
    except Exception as e:
        logger.error(f"Failed to reset vector store: {e}")
        return False


def get_all_indexed_documents(vector_store=None) -> Dict[str, int]:
    """Retrieves unique source filenames and their chunk counts from ChromaDB."""
    try:
        if vector_store is None:
            vector_store = get_vector_store()
        data = vector_store.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        doc_stats = {}
        for meta in metadatas:
            if not meta:
                continue
            source = meta.get("source", "Unknown Source")
            # Extract just the filename from path if needed
            filename = source.split("/")[-1].split("\\")[-1]
            doc_stats[filename] = doc_stats.get(filename, 0) + 1
            
        return doc_stats
    except Exception as e:
        print(f"Error fetching documents: {e}")
        logger.error(f"Error fetching documents: {e}")
        return {}


def delete_documents_by_source(vector_store, filename: str) -> bool:
    """Deletes all chunks belonging to a specific source file from ChromaDB."""
    try:
        data = vector_store.get(include=["metadatas"])
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])
        
        ids_to_delete = []
        for doc_id, meta in zip(ids, metadatas):
            if not meta:
                continue
            source = meta.get("source", "")
            source_filename = source.split("/")[-1].split("\\")[-1]
            if filename == source_filename or filename in source:
                ids_to_delete.append(doc_id)
                
        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
            return True
        return False
    except Exception as e:
        print(f"Error deleting document {filename}: {e}")
        logger.error(f"Error deleting document {filename}: {e}")
        return False


if __name__ == "__main__":
    print(f"--- Starting Document Ingestion Pipeline ---")
    print(f"Data directory: {config.data_dir}")
    print(f"Chroma persistence: {config.chroma_persist_dir}")
    try:
        files, chunks = ingest_documents()
        print(f"Ingestion Complete: Processed {files} files into {chunks} chunks.")
        stats = get_vector_store_stats()
        print(f"ChromaDB Total Chunk Count: {stats.get('total_chunks')}")
    except Exception as e:
        print(f"Ingestion failed: {e}")
