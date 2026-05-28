"""
ingest.py - Load PDFs, chunk text, embed, and store in ChromaDB.
Run this script to build (or update) the vector knowledge base.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")  # suppress deprecation noise on Windows

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Third-party imports
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Constants
DOCS_DIR = Path("./docs")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "portfolio_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
DIVIDER = "-" * 50


# PDF extraction

def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """
    Extract all text from a PDF using PyMuPDF.
    Returns (combined_text, total_pages).
    Pages with no extractable text are skipped with a warning.
    """
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    pages_text: list[str] = []

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text().strip()
        if not text:
            print(
                f"  Warning: page {page_num + 1} of {pdf_path.name} "
                "appears to be scanned, skipping"
            )
        else:
            pages_text.append(text)

    doc.close()
    combined = "\n\n".join(pages_text)
    return combined, total_pages


def load_pdfs() -> list[Document]:
    """
    Scan ./docs/ recursively for PDF files.
    Returns a list of LangChain Document objects.
    """
    pdf_files = sorted(DOCS_DIR.rglob("*.pdf"))

    if not pdf_files:
        print(
            "No PDF files found in ./docs/ folder.\n"
            "Please add your PDF files and run again."
        )
        sys.exit(0)

    documents: list[Document] = []
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        text, total_pages = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"  Skipped {pdf_path.name}: no extractable text found.")
            continue
        doc = Document(
            page_content=text,
            metadata={
                "source": pdf_path.name,
                "file_type": "pdf",
                "total_pages": total_pages,
            },
        )
        documents.append(doc)

    return documents


# Chunking

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


# Embeddings

def get_embeddings() -> FastEmbedEmbeddings:
    """Load the FastEmbed ONNX embedding model (no PyTorch required)."""
    print(f"Loading embedding model: {EMBED_MODEL}")
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


# ChromaDB helpers

def chroma_exists_and_has_docs() -> int:
    """
    Returns the number of chunks already in the vector store,
    or 0 if the store does not exist / is empty.
    """
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        return 0
    try:
        embeddings = get_embeddings()
        store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        count = store._collection.count()
        return count
    except Exception:
        return 0


def store_chunks(chunks: list[Document], wipe: bool = False) -> Chroma:
    """
    Persist chunks into ChromaDB.
    If wipe=True, delete the existing collection first.
    """
    embeddings = get_embeddings()

    if wipe:
        print("Wiping existing vector store and rebuilding from scratch...")
        import shutil
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    print(f"Storing {len(chunks)} chunks in ChromaDB at {CHROMA_DIR} ...")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    return store


# Main

def main():
    print("\n== Nachiket Portfolio RAG - Ingestion Pipeline ==\n")

    # 1. Discover PDFs
    documents = load_pdfs()
    print(f"\nFound {len(documents)} PDF(s) with extractable text.\n")

    # 2. Check existing vector store
    existing_count = chroma_exists_and_has_docs()
    wipe = False

    if existing_count > 0:
        print(
            f"Vector store already exists with {existing_count} chunks.\n"
            "  1. Add new documents to existing store\n"
            "  2. Wipe and rebuild from scratch"
        )
        while True:
            choice = input("Enter 1 or 2: ").strip()
            if choice == "1":
                wipe = False
                break
            elif choice == "2":
                wipe = True
                break
            else:
                print("Please enter 1 or 2.")

    # 3. Spliting into chunks
    print("\nSplitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} PDF(s).")

    # 4. Embeding and storing in ChromaDB
    store_chunks(chunks, wipe=wipe)

    # 5. Summary 
    print(
        f"\n{DIVIDER}\n"
        "Ingestion complete!\n"
        f"PDFs processed:       {len(documents)}\n"
        f"Total chunks created: {len(chunks)}\n"
        f"Vector store:         {CHROMA_DIR}\n"
        "You can now start the server with:\n"
        "  py -m uvicorn src.app:app --reload\n"
        f"{DIVIDER}\n"
    )


if __name__ == "__main__":
    main()
