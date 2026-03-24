from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from app.llm import VertexEmbeddings

# ---------------------------------------------------------------------------
# VECTOR STORE — InMemoryVectorStore with real Vertex AI embeddings
# Now uses semantic meaning to match documents, not text length.
# Note: resets on server restart (fine for demo — production would use
# a persistent store like Firestore or Vertex AI Vector Search)
# ---------------------------------------------------------------------------
embeddings = VertexEmbeddings()
vector_store = InMemoryVectorStore(embedding=embeddings)


def add_documents(documents: list[Document]) -> None:
    """Add chunked documents to the vector store."""
    vector_store.add_documents(documents)
    print(f"[retriever] Added {len(documents)} chunks to vector store.")


def retrieve_documents(query: str, k: int = 3) -> list[Document]:
    """
    Find the k most relevant document chunks for a query.
    k=3 means return the 3 closest matches.
    """
    results = vector_store.similarity_search(query, k=k)
    print(f"[retriever] Found {len(results)} chunks for query: '{query[:50]}...'")
    return results


def get_store_size() -> int:
    """How many chunks are currently stored."""
    return len(vector_store.store)
