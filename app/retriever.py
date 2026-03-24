from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from app.llm import VertexEmbeddings

embeddings = VertexEmbeddings()
vector_store = InMemoryVectorStore(embedding=embeddings)


def add_documents(documents: list[Document]) -> None:
    vector_store.add_documents(documents)
    print(f"[retriever] Added {len(documents)} chunks to vector store.")


def retrieve_documents(query: str, k: int = 3) -> list[Document]:
    results = vector_store.similarity_search(query, k=k)
    print(f"[retriever] Found {len(results)} chunks for query: '{query[:50]}...'")
    return results


def get_store_size() -> int:
    return len(vector_store.store)
