"""
Web Search and Web Scraping Module for Hybrid RAG Agent.
Performs live internet searches via DuckDuckGo and fetches/cleans web page content
to create dynamic LangChain Document objects for real-time context grounding.
"""

import logging
import re
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

logger = logging.getLogger("WebSearchAgent")

# Safe realistic browser headers to avoid basic bot blocks
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Max characters to extract per web page to prevent overwhelming context window
MAX_SCRAPED_CHARS = 4000


def clean_html(html_content: str) -> str:
    """
    Cleans raw HTML using BeautifulSoup by stripping out scripts, styles,
    navigation, headers, footers, and other boilerplate elements.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content and layout tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "svg"]):
        tag.decompose()

    # Extract text with space separation
    text = soup.get_text(separator=" ", strip=True)

    # Normalize excessive whitespace and blank lines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scrape_url(
    url: str,
    title: str = "Web Page",
    fallback_text: str = "",
    timeout: int = 6
) -> Optional[Document]:
    """
    Fetches a web page URL, cleans its HTML content, and packages it into a LangChain Document.
    Falls back to DDG snippet text if the page fetch fails or returns empty.
    """
    cleaned_content = ""
    try:
        response = requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            allow_redirects=True
        )
        if response.status_code == 200:
            cleaned_content = clean_html(response.text)
    except Exception as exc:
        logger.warning(f"Failed to fetch live web page '{url}': {exc}")

    # If scraping yielded too little content (e.g. JS-heavy SPA or bot protection),
    # fall back to search result snippet
    if not cleaned_content or len(cleaned_content) < 100:
        if fallback_text:
            cleaned_content = fallback_text.strip()
        else:
            return None

    # Truncate to maximum characters to keep prompt focused
    if len(cleaned_content) > MAX_SCRAPED_CHARS:
        cleaned_content = cleaned_content[:MAX_SCRAPED_CHARS] + "..."

    # Create LangChain Document with metadata
    # Metadata includes source (live URL) and page (title) as requested
    doc = Document(
        page_content=cleaned_content,
        metadata={
            "source": url,
            "page": title or "Web Result",
            "title": title or "Web Result",
            "is_web": True,
            "url": url,
        }
    )
    return doc


def search_web_and_scrape(query: str, max_results: int = 3) -> List[Document]:
    """
    Executes a web search on DuckDuckGo and scrapes top result URLs into LangChain Document objects.
    
    Args:
        query: The search term or user question.
        max_results: Maximum number of search results to fetch and scrape (default: 3).
        
    Returns:
        List of LangChain Document objects containing scraped web text and citation metadata.
    """
    if not query or not query.strip():
        return []

    if DDGS is None:
        logger.error("duckduckgo-search is not installed. Live web search is disabled.")
        return []

    logger.info(f"Initiating live web search for: '{query}' (max_results={max_results})")
    search_results = []

    # Query DuckDuckGo using DDGS().text
    try:
        with DDGS() as ddgs:
            try:
                results = list(ddgs.text(query, max_results=max_results))
            except TypeError:
                results = list(ddgs.text(keywords=query, max_results=max_results))
            if results:
                search_results = results
                logger.info(f"DDGS returned {len(search_results)} results.")
    except Exception as exc:
        logger.debug(f"Direct DDGS call failed: {exc}, trying backend fallbacks...")

    # Fallback to backend attempts if direct call returned empty or failed
    if not search_results:
        for backend in ("auto", "html", "lite"):
            try:
                with DDGS() as ddgs:
                    try:
                        results = list(ddgs.text(query, backend=backend, max_results=max_results))
                    except TypeError:
                        results = list(ddgs.text(keywords=query, backend=backend, max_results=max_results))
                    if results:
                        search_results = results
                        logger.info(f"DDGS returned {len(search_results)} results using backend '{backend}'.")
                        break
            except Exception as exc:
                logger.debug(f"DDGS search with backend '{backend}' failed: {exc}")
                continue

    if not search_results:
        logger.warning(f"No web search results found for query: '{query}'")
        return []

    documents: List[Document] = []
    for item in search_results:
        url = item.get("href") or item.get("link") or item.get("url")
        title = item.get("title") or "Web Search Result"
        snippet = item.get("body") or item.get("snippet") or ""

        if not url:
            continue

        doc = scrape_url(url=url, title=title, fallback_text=snippet)
        if doc:
            documents.append(doc)

    logger.info(f"Successfully scraped {len(documents)} live web documents.")
    return documents
