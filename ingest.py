"""Offline ingestion pipeline for the AWS Well-Architected Framework knowledge base.

Extracts text from the source PDF, partitions into overlapping semantic chunks,
computes dense vector embeddings, and serializes a local FAISS vector index.
"""

import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = os.path.join("data", "aws_well_architected.pdf")
VECTORSTORE_DIRECTORY = os.path.join("vectorstore", "faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration: preserves technical context across multi-page sections
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300


def load_pdf_documents(file_path: str) -> list[Document]:
    """Extracts non-empty pages from a target PDF into LangChain Document objects."""
    reader = PdfReader(file_path)
    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num},
                )
            )
    return documents


def run_ingestion() -> None:
    """Executes the end-to-end PDF ingestion and FAISS vector index construction."""
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Target PDF document not found at '{PDF_PATH}'.")

    print(f"[Ingest] Loading PDF document from: {PDF_PATH}...")
    pages = load_pdf_documents(PDF_PATH)
    print(f"[Ingest] Successfully extracted {len(pages)} pages.")

    print(f"[Ingest] Splitting pages into semantic chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(pages)
    print(f"[Ingest] Generated {len(chunks)} total text chunks.")

    print(f"[Ingest] Initializing local embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[Ingest] Generating vector embeddings and constructing FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(VECTORSTORE_DIRECTORY), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIRECTORY)
    print(f"[Ingest] Ingestion complete. FAISS index saved to '{VECTORSTORE_DIRECTORY}'.")


if __name__ == "__main__":
    run_ingestion()