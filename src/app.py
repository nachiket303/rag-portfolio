"""
app.py - FastAPI backend for Nachiket's Portfolio RAG chatbot.
Exposes /chat, /health, and /docs-loaded endpoints.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "portfolio_docs"

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nachiket Portfolio RAG API",
    description=(
        "A Retrieval-Augmented Generation API that answers recruiter "
        "questions about Nachiket Shejwal based on his personal documents."
    ),
    version="1.0.0",
)

# CORS – allow all origins so GitHub Pages (or any frontend) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info(
        "\n"
        "╔══════════════════════════════════════════════════╗\n"
        "║   Nachiket Portfolio RAG API running             ║\n"
        "║                                                  ║\n"
        "║  Health check:  http://localhost:8000/health     ║\n"
        "║  Chat endpoint: POST http://localhost:8000/chat  ║\n"
        "║  Docs loaded:   http://localhost:8000/docs-loaded║\n"
        "╚══════════════════════════════════════════════════╝"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Accepts a question, retrieves relevant context from ChromaDB,
    calls the Groq LLM, and returns the answer with source filenames.
    """
    logger.info(f"Incoming question: {request.question!r}")

    try:
        from src.chain import ask_with_sources

        result = ask_with_sources(request.question)
        logger.info(
            f"Question: {request.question!r} | Sources used: {result['sources']}"
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
        )

    except FileNotFoundError as exc:
        logger.error(f"Vector store not found: {exc}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Vector store not found. "
                "Please run `python src/ingest.py` first to build the knowledge base."
            ),
        )
    except EnvironmentError as exc:
        logger.error(f"Environment error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception(f"Unexpected error while answering question: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(exc)}",
        )


@app.get("/health")
async def health():
    """
    Quick status check:
    - Is the Groq API key present?
    - Is the ChromaDB vector store built?
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_status = (
        "connected"
        if groq_key and groq_key != "your_groq_api_key_here"
        else "api key missing"
    )

    chroma_path = Path(CHROMA_DIR)
    has_files = chroma_path.exists() and any(chroma_path.iterdir())
    vectorstore_status = (
        "ready"
        if has_files
        else "not found - run ingest.py"
    )

    overall = "ok" if groq_status == "connected" and has_files else "degraded"

    logger.info(f"Health check: groq={groq_status}, vectorstore={vectorstore_status}")

    return {
        "status": overall,
        "groq": groq_status,
        "vectorstore": vectorstore_status,
    }


@app.get("/docs-loaded")
async def docs_loaded():
    """
    Inspect ChromaDB to return all unique source filenames and total chunk count.
    """
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        return {
            "documents": [],
            "total_chunks": 0,
            "message": "No documents loaded. Run python src/ingest.py first.",
        }

    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
        from langchain_chroma import Chroma

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

        total_chunks = store._collection.count()

        # Pull all metadata to extract unique source filenames
        raw = store._collection.get(include=["metadatas"])
        sources = list(
            dict.fromkeys(
                meta.get("source", "")
                for meta in (raw.get("metadatas") or [])
                if meta.get("source")
            )
        )

        logger.info(f"Docs loaded: {sources} | Total chunks: {total_chunks}")
        return {
            "documents": sources,
            "total_chunks": total_chunks,
        }

    except Exception as exc:
        logger.error(f"Error reading ChromaDB metadata: {exc}")
        return {
            "documents": [],
            "total_chunks": 0,
            "message": f"Error reading vector store: {str(exc)}",
        }
