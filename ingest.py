import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = os.path.join("data", "aws_well_architected.pdf")
VECTORSTORE_DIRECTORY = os.path.join("vectorstore", "faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_pdf_documents(file_path: str) -> list[Document]:
    """Extracts pages using pypdf and returns standard langchain_core Documents."""
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

def run_ingestion():
    # Check for path
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"Target PDF not found at '{PDF_PATH}'."
        )
    # Loading pdf
    print(f"Extracting PDF document from: {PDF_PATH}...")
    pages = load_pdf_documents(PDF_PATH)
    print(f"Successfully loaded {len(pages)} pages from the pdf.")

    # Splitting pages into semantic chunks
    print("Splitting text into chunks (Chunk_size=1200, overlap=200)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Generated {len(chunks)} total text chunks.")

    # Load local hugging face model
    print(f"Loading local embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Build and persist FAISS index
    print("Generating vector embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(VECTORSTORE_DIRECTORY), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIRECTORY)
    print(f"Success! FAISS index created and saved to '{VECTORSTORE_DIRECTORY}'.")

if __name__ == "__main__":
    run_ingestion()