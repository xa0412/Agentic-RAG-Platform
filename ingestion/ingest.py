from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from app.retriever import add_documents


# ---------------------------------------------------------------------------
# CHUNKING STRATEGY
# chunk_size=800  → each chunk is ~800 characters (~150 words)
# chunk_overlap=100 → 100 characters shared between neighbouring chunks
#                     prevents answers being cut off at chunk boundaries
# ---------------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)


def load_pdf(file_path: str) -> list[Document]:
    """Extract text from a PDF file, one Document per page."""
    reader = PdfReader(file_path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():  # skip blank pages
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": Path(file_path).name,
                    "page": page_num + 1,
                }
            ))

    print(f"[ingest] Loaded {len(documents)} pages from '{Path(file_path).name}'")
    return documents


def load_text(file_path: str) -> list[Document]:
    """Load a plain .txt file as a single Document."""
    text = Path(file_path).read_text(encoding="utf-8")
    return [Document(
        page_content=text,
        metadata={"source": Path(file_path).name, "page": 1}
    )]


def ingest_file(file_path: str) -> int:
    """
    Full pipeline: load → chunk → store.
    Returns the number of chunks created.
    """
    path = Path(file_path)

    # Step 1: Load raw text from file
    if path.suffix.lower() == ".pdf":
        raw_docs = load_pdf(file_path)
    elif path.suffix.lower() == ".txt":
        raw_docs = load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .pdf or .txt")

    # Step 2: Split into chunks
    chunks = splitter.split_documents(raw_docs)
    print(f"[ingest] Split into {len(chunks)} chunks")

    # Step 3: Store chunks in vector store
    add_documents(chunks)

    return len(chunks)
