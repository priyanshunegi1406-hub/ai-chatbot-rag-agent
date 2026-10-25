"""
Configuration module for the RAG Chatbot application.
Loads and validates environment variables securely using python-dotenv.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Load local .env file if available
load_dotenv(BASE_DIR / ".env", override=False)


@dataclass(frozen=True)
class AppConfig:
    """Production application settings and hyperparameters."""

    # Gemini API Credentials
    gemini_api_key: Optional[str] = (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

    # Models
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

    # Hyperparameters
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.95"))
    max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))

    # Vector Storage & Ingestion
    chroma_persist_dir: Path = BASE_DIR / os.getenv(
        "CHROMA_PERSIST_DIRECTORY", "chroma_db"
    )
    chroma_collection_name: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "rag_documents"
    )
    data_dir: Path = BASE_DIR / os.getenv("DATA_DIRECTORY", "data")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))

    def validate_api_key(self, custom_key: Optional[str] = None) -> str:
        """
        Validates whether a valid API key is present in environment or provided explicitly.
        Returns the valid key or raises ValueError.
        """
        key = custom_key or self.gemini_api_key
        if not key or key.strip() == "" or key.strip() == "your_gemini_api_key_here":
            raise ValueError(
                "Gemini API Key is missing! Please configure GEMINI_API_KEY in your .env file "
                "or enter it directly in the application settings."
            )
        return key.strip()


# Global Singleton Configuration
config = AppConfig()

# Ensure directories exist
config.data_dir.mkdir(parents=True, exist_ok=True)
config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
