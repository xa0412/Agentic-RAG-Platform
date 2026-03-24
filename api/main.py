from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.agent import agent
from app.retriever import get_store_size
import tempfile
import os

app = FastAPI(
    title="Agentic RAG Platform",
    description="An autonomous research agent powered by LangGraph and Vertex AI",
    version="0.1.0",
)


# --- Request / Response schemas ---
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []


# --- Upload and ingest a document ---
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    from ingestion.ingest import ingest_file

    # Only allow PDF and TXT
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    # Save upload to a temp file, then ingest it
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunk_count = ingest_file(tmp_path)
    finally:
        os.unlink(tmp_path)  # always clean up the temp file

    return {
        "message": f"Successfully ingested '{file.filename}'",
        "chunks_created": chunk_count,
        "total_chunks_in_store": get_store_size(),
    }


# --- Health check (Cloud Run uses this to confirm the container is alive) ---
@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}


# --- Main query endpoint ---
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Run the LangGraph agent
    result = agent.invoke({"question": request.question, "documents": [], "answer": "", "retries": 0})

    # Extract sources from retrieved documents
    sources = list({doc.metadata.get("source", "unknown") for doc in result["documents"]})

    return QueryResponse(answer=result["answer"], sources=sources)
