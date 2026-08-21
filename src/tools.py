"""Retrieval and web search tool integrations for Kara AWS Assistant.

Provides thread-safe singleton access to the local FAISS vectorstore and
exponential backoff search execution via the Tavily Search API.
"""

import logging
import os
import random
import time
import warnings
from threading import Lock
import transformers

warnings.filterwarnings("ignore", category=DeprecationWarning)
transformers.logging.set_verbosity_error()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)

VECTORSTORE_DIRECTORY = os.path.join("vectorstore", "faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Process-level singleton cache for embedding model and FAISS vector index
_vectorstore: FAISS | None = None
_vectorstore_lock = Lock()


def _get_vectorstore() -> FAISS:
    """Thread-safe lazy initialization and retrieval of the persisted FAISS vectorstore."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    with _vectorstore_lock:
        if _vectorstore is not None:
            return _vectorstore

        if not os.path.exists(VECTORSTORE_DIRECTORY):
            raise FileNotFoundError(
                f"FAISS index not found at '{VECTORSTORE_DIRECTORY}'. Please execute 'uv run ingest.py' first."
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = FAISS.load_local(
            VECTORSTORE_DIRECTORY,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS vectorstore initialized and cached in memory.")
        return _vectorstore


def get_internal_retriever(k: int = 4):
    """Returns a vectorstore retriever configured with search depth k."""
    vectorstore = _get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def search_web(
    query: str, max_results: int = 5, retries: int = 3, base_delay: float = 2.0
) -> tuple[str, list[str]]:
    """Executes real-time web search via Tavily with exponential backoff and jitter.

    Args:
        query: Search query string.
        max_results: Maximum number of search results to retrieve.
        retries: Maximum number of retry attempts upon failure.
        base_delay: Base backoff delay in seconds.

    Returns:
        Tuple of (formatted_context_string, list_of_unique_source_urls).
    """
    if not query or not query.strip():
        return "No query provided for web search.", []

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        logger.error("TAVILY_API_KEY environment variable is not configured.")
        return (
            "Web search is unavailable due to missing API credentials. "
            "Please check system configuration.",
            [],
        )

    last_error = ""

    for attempt in range(retries):
        try:
            tavily = TavilySearch(
                max_results=max_results,
                topic="general",
            )
            raw_response = tavily.invoke({"query": query})

            if not raw_response:
                if attempt < retries - 1:
                    time.sleep(1.0)
                    continue
                return "No relevant web results found.", []

            items = []
            if isinstance(raw_response, dict):
                items = raw_response.get("results", [])
            elif isinstance(raw_response, list):
                items = raw_response

            if not items:
                return "No relevant web results found.", []

            context_parts: list[str] = []
            sources: list[str] = []

            for item in items:
                if not isinstance(item, dict):
                    continue

                title = item.get("title", "Web Result")
                content = item.get("content", "")
                url = item.get("url", "")

                if not url:
                    continue

                sources.append(url)
                # Formats web context with standard citation tags matching the internal retriever
                context_parts.append(f"[Source: {url}]\n{title}: {content}")

            if not context_parts:
                return "No relevant web results found.", []

            unique_sources = list(dict.fromkeys(sources))
            return "\n\n".join(context_parts), unique_sources

        except Exception as e:
            last_error = f"Tavily search exception: {e}"
            logger.warning(
                "Tavily search attempt %d/%d failed: %s",
                attempt + 1,
                retries,
                e,
            )
            if attempt < retries - 1:
                backoff_time = (base_delay * (2**attempt)) + random.uniform(0.2, 0.8)
                time.sleep(backoff_time)
                continue

    logger.error("Web search failed across all %d attempts: %s", retries, last_error)
    return (
        "Web search is temporarily unavailable. Let the user know the search "
        "couldn't be completed right now and they can try again shortly.",
        [],
    )