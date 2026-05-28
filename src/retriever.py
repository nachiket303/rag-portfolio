"""
retriever.py - Vector store search logic.
Loads ChromaDB and exposes a retriever that returns the top-5 relevant chunks.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "portfolio_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_chroma_exists() -> None:
    """Raise a clear error if the vector store has not been built yet."""
    if not Path(CHROMA_DIR).exists():
        raise FileNotFoundError(
            "Vector store not found. Please run python src/ingest.py first."
        )


def _load_store() -> Chroma:
    """Load the persisted ChromaDB vector store."""
    _check_chroma_exists()
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return store


# APi

def get_retriever():
    """
    Return a LangChain retriever object that fetches the top-5 most
    relevant chunks from the ChromaDB vector store.

    Usage:
        retriever = get_retriever()
        docs = retriever.get_relevant_documents("your question here")
    """
    store = _load_store()
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return retriever


def test_retrieval(query: str) -> None:
    """
    Debug helper: prints the top retrieved chunks for a given query,
    including similarity scores and source filenames.

    Usage:
        python -c "from src.retriever import test_retrieval; \
                   test_retrieval('did you work with Databricks')"
    """
    _check_chroma_exists()
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    # SImilarity search is happening here, we are getting the 5 most relevant chunks based on the query
    results = store.similarity_search_with_score(query, k=TOP_K)

    print(f"\nQuery: '{query}'")
    print("-" * 60)

    if not results:
        print("No results found. Is the vector store populated?")
        return

    for idx, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        # ChromaDB returns L2 distance; convert to a 0-1 similarity score
        similarity = round(1 / (1 + score), 4)
        snippet = doc.page_content[:300].replace("\n", " ").strip()
        print(
            f"\nResult {idx} (score: {similarity}) from: {source}\n"
            f"{snippet}{'...' if len(doc.page_content) > 300 else ''}"
        )

    print("\n" + "-" * 60 + "\n")
